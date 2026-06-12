"""Per-language evaluation: CER for Chinese, WER for English.

Run once per language, with and without --adapter, to build the
before/after table:
  python evaluate_cer.py --dataset_config cmn_hans_cn --language zh
  python evaluate_cer.py --dataset_config en_us       --language en
  python evaluate_cer.py --dataset_config cmn_hans_cn --language zh --adapter ./output/adapter
  python evaluate_cer.py --dataset_config en_us       --language en --adapter ./output/adapter
"""
import argparse

import torch
from datasets import Audio, load_dataset
from jiwer import cer, wer
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
    p.add_argument("--dataset_config", default="cmn_hans_cn")
    p.add_argument("--language", default="zh")
    p.add_argument("--samples", type=int, default=200)
    args = p.parse_args()

    processor = WhisperProcessor.from_pretrained(
        args.model_id, language=args.language, task="transcribe")
    ds = load_dataset("google/fleurs", args.dataset_config,
                      split=f"test[:{args.samples}]", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    model, device = load(args.model_id, args.adapter)

    refs, hyps = [], []
    for ex in ds:
        feats = processor.feature_extractor(
            ex["audio"]["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=model.dtype)
        with torch.no_grad():
            ids = model.generate(input_features=feats,
                                 language=args.language, task="transcribe")
        hyps.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        refs.append(ex["transcription"])

    # Chinese has no word boundaries -> character error rate; English -> WER.
    metric, name = (cer, "CER") if args.language == "zh" else (wer, "WER")
    label = "fine-tuned" if args.adapter else "baseline"
    print(f"{name} = {metric(refs, hyps):.4f}  "
          f"({args.dataset_config}/{args.language}, {label}, n={len(refs)})")


if __name__ == "__main__":
    main()
