#!/usr/bin/env bash
# Merge the LoRA adapter into the base model, convert to CTranslate2 int8,
# and upload for CPU serving via faster-whisper.
set -euo pipefail

BUCKET="CHANGE_ME"  # e.g. yourname-whisper-finetune

aws s3 sync "s3://${BUCKET}/whisper-lora/adapter/" ./adapter/

python - <<'PY'
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

base = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=torch.float16)
merged = PeftModel.from_pretrained(base, "./adapter").merge_and_unload()
merged.save_pretrained("./merged")
WhisperProcessor.from_pretrained("./adapter").save_pretrained("./merged")
PY

pip install ctranslate2
ct2-transformers-converter --model ./merged --output_dir ./whisper-ct2 \
  --quantization int8 --copy_files tokenizer_config.json preprocessor_config.json

aws s3 sync ./whisper-ct2 "s3://${BUCKET}/whisper-ct2/"
echo "Done: s3://${BUCKET}/whisper-ct2/"
