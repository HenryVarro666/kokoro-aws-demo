import argparse

import torch
from datasets import Audio, load_dataset
from jiwer import cer
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load(model_id, adapter=None):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="openai/whisper-large-v3")
    p.add_argument("--adapter", default=None)
    p.add_argument("--samples", type=int, default=200)
    args = p.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model_id, language="zh", task="transcribe")
    ds = load_dataset("google/fleurs", "cmn_hans_cn",
                      split=f"test[:{args.samples}]", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    model, device = load(args.model_id, args.adapter)

    refs, hyps = [], []
    for ex in ds:
        feats = processor.feature_extractor(
            ex["audio"]["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=model.dtype)
        with torch.no_grad():
            ids = model.generate(input_features=feats, language="zh", task="transcribe")
        hyps.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        refs.append(ex["transcription"])

    label = "fine-tuned" if args.adapter else "baseline"
    print(f"CER = {cer(refs, hyps):.4f}  ({label})")


if __name__ == "__main__":
    main()
