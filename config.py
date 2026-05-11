import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN      = os.getenv("HF_TOKEN")
WB_TOKEN      = os.getenv("WB_TOKEN")

MODEL_PATH    = "unsloth/gemma-4-E4B"
RUN_NAME      = "gemm4-asr-E4B-V1"
HUB_MODEL_ID  = "Sibishreekapture/gemma4-asr-indic"
MAX_SEQ_LENGTH = 4096
TARGET_SR     = 16000

TRAIN_SAMPLES = {"tamil": 423000}
EVAL_SAMPLES  = {"tamil": 4426}
indic_CONFIGS = [("tamil", "tamil", "ta")]

DISK_ROOT       = "./asr_data_indic"
TRAIN_AUDIO_DIR = os.path.join(DISK_ROOT, "train_wavs")
EVAL_AUDIO_DIR  = os.path.join(DISK_ROOT, "eval_wavs")

INSTRUCTION = (
    "Transcribe the following Tamil audio accurately. "
    "Output only the transcription text, nothing else."
)