import random
import torch
import wandb
import jiwer
import soundfile as sf
from transformers import TrainerCallback
from gemma_tamil_asr_finetuning.config import TARGET_SR,INSTRUCTION


class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step     = state.global_step
            loss     = logs.get("loss", "")
            eval_loss = logs.get("eval_loss", "")
            if loss:
                print(f"Step {step} | train_loss: {loss:.4f}")
            if eval_loss:
                print(f"Step {step} | eval_loss: {eval_loss:.4f}")


class WandbMetricsCallback(TrainerCallback):
    def __init__(self, processor, eval_entries, sample_size=10):
        self.processor    = processor
        self.eval_entries = eval_entries
        self.sample_size  = sample_size

    def compute_wer(self, model):
        model.eval()
        preds, refs = [], []
        skipped = 0

        indices = random.sample(
            range(len(self.eval_entries)),
            min(len(self.eval_entries), self.sample_size)
        )

        for idx in indices:
            entry = self.eval_entries[idx]
            audio_array, _ = sf.read(entry["audio"], dtype="float32")

            try:
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": (audio_array, TARGET_SR)},
                        {"type": "text",  "text": INSTRUCTION}
                    ]
                }]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(
                    audio=[audio_array],
                    text=[prompt],
                    sampling_rate=TARGET_SR,
                    return_tensors="pt",
                    padding=True
                ).to("cuda")

                with torch.no_grad():
                    output    = model.generate(
                        **inputs, max_new_tokens=128,
                        use_cache=True, do_sample=False,
                        pad_token_id=self.processor.pad_token_id,
                        eos_token_id=self.processor.eos_token_id,
                    )
                    input_len = inputs["input_ids"].shape[1]
                    pred      = self.processor.decode(
                        output[0][input_len:], skip_special_tokens=True
                    ).strip()

                preds.append(pred)
                refs.append(entry["text"])

            except ValueError:
                skipped += 1
                continue

        model.train()

        if not preds:
            print(f"⚠️  WER: all {skipped} samples skipped.")
            return None

        return jiwer.wer(refs, preds)

    def on_evaluate(self, args, state, control, model, **kwargs):
        wer = self.compute_wer(model)
        if wer is not None:
            wandb.log({"eval/wer": wer}, step=state.global_step)
            print(f"\n[WandB] Step {state.global_step} | Eval WER: {wer:.4f} ({wer*100:.1f}%)")