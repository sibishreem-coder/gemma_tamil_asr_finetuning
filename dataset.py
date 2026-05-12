import gc, random
import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset, Audio
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from gemma_tamil_asr_finetuning.config import TARGET_SR, TRAIN_AUDIO_DIR, EVAL_AUDIO_DIR, INSTRUCTION,SVARAH_CONFIGS, TRAIN_SAMPLES
import os

os.makedirs(TRAIN_AUDIO_DIR, exist_ok=True)
os.makedirs(EVAL_AUDIO_DIR, exist_ok=True)


def process_stream_to_disk(dataset, lang, n_samples, audio_dir, tag, prefix):
    entries = []
    skipped = 0
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=TARGET_SR))
    stream  = iter(dataset)
    pbar    = tqdm(stream, desc=f"Processing {lang} {tag}", total=n_samples)

    for sample in pbar:
        if len(entries) >= n_samples:
            break
        text = (
            sample.get("transcript") or
            sample.get("transcription") or
            sample.get("text") or ""
        ).strip()
        if not text:
            continue

        try:
            if sample["audio_filepath"] is None or sample["audio_filepath"].get("array") is None:
                continue
            array = sample["audio_filepath"]["array"].astype(np.float32)
        except Exception:
            continue

        duration = len(array) / TARGET_SR
        if duration > 30.0 or duration < 1.0:
            skipped += 1
            pbar.set_postfix(saved=len(entries), skipped=skipped)
            continue

        wav_path = os.path.join(audio_dir, f"{prefix}_{tag}_{len(entries):06d}.wav")
        sf.write(wav_path, array, TARGET_SR)
        entries.append({"audio": wav_path, "text": text, "lang": lang.capitalize()})
        pbar.set_postfix(saved=len(entries), skipped=skipped)

    print(f"  ✅ {lang} {tag}: {len(entries)} saved, {skipped} skipped")
    return entries


def load_all_entries(indic_configs, train_samples, eval_samples):
    train_entries, eval_entries = [], []

    for lang_name, indic_code, prefix in indic_configs:
        train_ds = load_dataset("ai4bharat/IndicVoices", indic_code, split="train", streaming=True)
        train_entries.extend(process_stream_to_disk(
            train_ds, lang_name, train_samples[lang_name], TRAIN_AUDIO_DIR, "train", prefix
        ))
        gc.collect(); torch.cuda.empty_cache()

        eval_ds = load_dataset("ai4bharat/IndicVoices", indic_code, split="valid", streaming=True)
        eval_entries.extend(process_stream_to_disk(
            eval_ds, lang_name, eval_samples[lang_name], EVAL_AUDIO_DIR, "eval", prefix
        ))
        gc.collect(); torch.cuda.empty_cache()
    for lang_name, config_code, prefix in SVARAH_CONFIGS:
        try:            
            ds_stream = load_dataset("ai4bharat/Svarah", split="test", streaming=True)

            train_data = process_stream_to_disk(
                ds_stream, lang_name, TRAIN_SAMPLES.get(lang_name, 6656), 
                TRAIN_AUDIO_DIR, "train", prefix
            )
            train_entries.extend(train_data)
        except Exception as e:
            print(f"❌ Error loading Svarah: {e}") 

    random.shuffle(train_entries)
    print(f"Train: {len(train_entries)}, Eval: {len(eval_entries)}")
    return train_entries, eval_entries


class ASRDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        sample = self.entries[idx]
        audio_array, _ = sf.read(sample["audio"], dtype="float32")
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "audio", "audio": (audio_array, TARGET_SR)},
                    {"type": "text",  "text": INSTRUCTION}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": sample["text"]}
                ]}
            ],
            "length": int(len(audio_array) / TARGET_SR * 50) + len(sample["text"]),
        }