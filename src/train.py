"""
train.py — shared training logic (Task 4)
=========================================
Fine-tunes the model with the Hugging Face Trainer API and logs everything to
Weights & Biases. The Kaggle notebooks call the same logic; this module lets you
reproduce a run locally too.

IMPORTANT (per the assignment): training is intended for Kaggle Notebooks (free
GPU). GitHub Actions must NOT run training — it is used only for CI + inference.

Secrets are read from the environment (set them via Kaggle Secrets or locally):
  WANDB_API_KEY, HF_TOKEN

Example:
    python src/train.py --version v1 --learning-rate 3e-5 --epochs 3 --batch-size 16
    python src/train.py --version v2 --learning-rate 5e-5 --epochs 3 --batch-size 16
"""
import argparse
import os


import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (DataCollatorWithPadding, Trainer, TrainingArguments)

from model_utils import DEFAULT_MODEL, load_model, load_tokenizer
from prepare_data import clean_text

DATASET_NAME = "dair-ai/emotion"
PROJECT = "mlops-assignment3"


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def build_datasets(tokenizer, max_len=128):
    ds = load_dataset(DATASET_NAME)

    def _clean(batch):
        return {"text": [clean_text(t) for t in batch["text"]]}

    ds = ds.map(_clean, batched=True)

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    ds = ds.map(_tok, batched=True)
    keep = ["input_ids", "attention_mask", "label"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    wandb.init(
        project=PROJECT,
        name=f"run-{args.version}",
        config={
            "model": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_len": args.max_len,
            "version": args.version,
            "platform": os.environ.get("PLATFORM", "local"),
        },
    )

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name)
    ds = build_datasets(tokenizer, args.max_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb",
        run_name=f"run-{args.version}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(ds["test"])
    print("Test metrics:", metrics)
    wandb.run.summary.update({f"test_{k}": v for k, v in metrics.items()})

    # Save locally so push_to_hub.py can upload the best checkpoint.
    save_dir = f"best_model_{args.version}"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved best model -> {save_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
