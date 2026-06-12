import argparse
import os

import torch
from datasets import Audio, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="openai/whisper-large-v3")
    p.add_argument("--dataset", default="google/fleurs")
    p.add_argument("--dataset_config", default="cmn_hans_cn,en_us",
                   help="Comma-separated FLEURS configs, paired 1:1 with --language.")
    p.add_argument("--language", default="zh,en",
                   help="Comma-separated Whisper language codes, paired with configs.")
    p.add_argument("--train_samples", type=int, default=2000,
                   help="Samples PER language config (2 configs x 2000 = 4000 total).")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--use_8bit", action="store_true")
    # SageMaker injects the artifact dir via this env var; local runs use ./output
    p.add_argument("--output_dir", default=os.environ.get("SM_MODEL_DIR", "./output"))
    return p.parse_args()


class SpeechCollator:
    """Pad input_features into a batch; pad labels with -100 (ignored by the loss)."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = [{"input_ids": f["labels"]} for f in features]
        labels = self.processor.tokenizer.pad(labels, return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.processor.tokenizer.bos_token_id).all():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def main():
    args = parse_args()
    configs = [c.strip() for c in args.dataset_config.split(",")]
    langs = [lang.strip() for lang in args.language.split(",")]
    assert len(configs) == len(langs), "--dataset_config and --language must pair 1:1"

    processor = WhisperProcessor.from_pretrained(args.model_id, task="transcribe")

    # Mixed-language training: each sample's labels must carry its OWN
    # language token (<|zh|>/<|en|>), so tokenize per config with the
    # prefix set accordingly, then concatenate and shuffle.
    parts = []
    for config, lang in zip(configs, langs):
        ds = load_dataset(args.dataset, config,
                          split=f"train[:{args.train_samples}]",
                          trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        processor.tokenizer.set_prefix_tokens(language=lang, task="transcribe")

        def prepare(batch):
            audio = batch["audio"]
            batch["input_features"] = processor.feature_extractor(
                audio["array"], sampling_rate=16000
            ).input_features[0]
            batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
            return batch

        parts.append(ds.map(prepare, remove_columns=ds.column_names, num_proc=2))
        print(f"[data] {config} ({lang}): {len(parts[-1])} samples")

    if len(parts) > 1:
        from datasets import concatenate_datasets
        ds = concatenate_datasets(parts).shuffle(seed=42)
    else:
        ds = parts[0]
    print(f"[data] total training samples: {len(ds)}")

    quant = BitsAndBytesConfig(load_in_8bit=True) if args.use_8bit else None
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant,
        device_map="auto" if args.use_8bit else None,
    )
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    if args.use_8bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()  # expect ~1% trainable params

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        save_strategy="epoch",
        remove_unused_columns=False,  # required with PEFT + custom collator
        label_names=["labels"],
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=train_args,
        train_dataset=ds, data_collator=SpeechCollator(processor),
    )
    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    processor.save_pretrained(os.path.join(args.output_dir, "adapter"))


if __name__ == "__main__":
    main()
