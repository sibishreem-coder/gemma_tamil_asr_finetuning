import os, gc, torch, wandb, datasets
from huggingface_hub import login
from unsloth import FastModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from gemma_tamil_asr_finetuning.config import (
    HF_TOKEN, WB_TOKEN, MODEL_PATH, RUN_NAME, HUB_MODEL_ID,
    MAX_SEQ_LENGTH, TRAIN_SAMPLES, EVAL_SAMPLES, indic_CONFIGS
)
from gemma_tamil_asr_finetuning.dataset import load_all_entries, ASRDataset
from gemma_tamil_asr_finetuning.callbacks import PrintLossCallback, WandbMetricsCallback

# ── Environment setup ─────────────────────────────────────────────────────────
datasets.config.AUDIO_DECODE_BACKEND = "soundfile"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

login(token=HF_TOKEN)
wandb.login(key=WB_TOKEN)

# ── Data ──────────────────────────────────────────────────────────────────────
train_entries, eval_entries = load_all_entries(indic_CONFIGS, TRAIN_SAMPLES, EVAL_SAMPLES)

# ── Model ─────────────────────────────────────────────────────────────────────
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_PATH,
    use_gradient_checkpointing="unsloth",
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, "gemma-4")

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32,
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
)

# ── W&B + Callbacks ───────────────────────────────────────────────────────────
wandb.init(project="gemma4-asr-ft", name=RUN_NAME)

callbacks = [
    PrintLossCallback(),
    WandbMetricsCallback(
        processor=tokenizer,
        eval_entries=eval_entries,
        sample_size=len(eval_entries)
    )
]

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=ASRDataset(train_entries),
    eval_dataset=ASRDataset(eval_entries),
    processing_class=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    callbacks=callbacks,
    args=SFTConfig(
        output_dir=f"./{RUN_NAME}",
        num_train_epochs=1,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        tf32=True,
        logging_steps=50,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=400,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        length_column_name="length",
        report_to="all",
        push_to_hub=True,
        hub_strategy="checkpoint",
        hub_model_id=HUB_MODEL_ID,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_SEQ_LENGTH,
    ),
)

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
model.save_pretrained(f"./{RUN_NAME}/final_adapter")
tokenizer.save_pretrained(f"./{RUN_NAME}/final_adapter")
wandb.finish()