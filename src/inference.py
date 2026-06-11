"""
inference.py — used by the Dockerfile and the GitHub Actions inference workflow
===============================================================================
Loads the fine-tuned model + tokenizer from the Hugging Face Hub and classifies
a single piece of input text. Designed to run with NO manual setup inside a
container or CI runner.

Configuration (all via environment variables):
  HF_MODEL_NAME  - the public HF model repo to load
                   (default below; overridable via Docker ARG/ENV or workflow env)
  INPUT_TEXT     - the text to classify (default provided for smoke tests)
  HF_TOKEN       - optional; only needed for private models

Exit code is 0 on success so it can gate a CI job.
"""
import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = os.environ.get("HF_MODEL_NAME", "your-username/distilbert-emotion")


def main() -> int:
    model_name = os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL)
    input_text = os.environ.get("INPUT_TEXT", "I can't believe how wonderful today turned out!")
    token = os.environ.get("HF_TOKEN")  # optional, for private repos

    print(f"[inference] model      : {model_name}")
    print(f"[inference] input_text : {input_text!r}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
    model.eval()

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))

    id2label = model.config.id2label
    pred_label = id2label.get(pred_id, id2label.get(str(pred_id), str(pred_id)))

    ranked = sorted(
        ((id2label.get(i, id2label.get(str(i), str(i))), float(probs[i])) for i in range(len(probs))),
        key=lambda x: x[1],
        reverse=True,
    )

    result = {
        "input": input_text,
        "prediction": pred_label,
        "confidence": round(float(probs[pred_id]), 4),
        "all_scores": {k: round(v, 4) for k, v in ranked},
    }
    print("[inference] result:")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
