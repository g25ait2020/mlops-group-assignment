"""
prepare_data.py — Task 2: Data Preparation & Normalisation
==========================================================
Reusable script that downloads the `emotion` dataset (dair-ai/emotion),
inspects it, cleans the text, builds the label mapping, and saves a
prepared copy locally plus an id2label.json mapping file.

Modality: TEXT (6-class emotion classification).

Cleaning steps applied (see report for justification):
  - drop rows with missing / empty text
  - strip surrounding whitespace and collapse internal runs of whitespace
  - lowercase (DistilBERT 'uncased' expects lowercase input)
  - remove exact duplicate (text, label) rows

Usage:
    python src/prepare_data.py --out data/processed
    python src/prepare_data.py --out data/processed --max-samples 20000
"""
import argparse
import json
import os
import re
from collections import Counter

from datasets import load_dataset, DatasetDict

DATASET_NAME = "dair-ai/emotion"
# Canonical label order for the emotion dataset.
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

_ws = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise a single text sample."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = _ws.sub(" ", text)
    return text


def inspect(ds: DatasetDict) -> None:
    """Print size, structure, and class distribution for each split."""
    print("=" * 60)
    print("RAW DATA INSPECTION")
    print("=" * 60)
    print("Splits:", list(ds.keys()))
    print("Features:", ds["train"].features)
    for split in ds:
        n = len(ds[split])
        dist = Counter(ds[split]["label"])
        dist_named = {LABELS[k]: v for k, v in sorted(dist.items())}
        print(f"\n[{split}] {n} samples")
        print("  class distribution:", dist_named)


def clean_split(split):
    """Clean one split: normalise text, drop empties, drop duplicates."""
    seen = set()
    keep_text, keep_label = [], []
    dropped_empty = dropped_dupe = 0
    for text, label in zip(split["text"], split["label"]):
        ct = clean_text(text)
        if not ct:
            dropped_empty += 1
            continue
        key = (ct, label)
        if key in seen:
            dropped_dupe += 1
            continue
        seen.add(key)
        keep_text.append(ct)
        keep_label.append(label)
    print(f"  cleaned: kept {len(keep_text)} | dropped empty {dropped_empty} | dropped dupes {dropped_dupe}")
    return split.from_dict({"text": keep_text, "label": keep_label})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed", help="output directory for the prepared dataset")
    ap.add_argument("--mapping", default="id2label.json", help="path to write the id2label mapping")
    ap.add_argument("--max-samples", type=int, default=0, help="optional cap on train samples (0 = all)")
    args = ap.parse_args()

    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    inspect(ds)

    print("\n" + "=" * 60)
    print("CLEANING")
    print("=" * 60)
    cleaned = {}
    for split in ds:
        print(f"[{split}]")
        cleaned[split] = clean_split(ds[split])
    ds = DatasetDict(cleaned)

    if args.max_samples and len(ds["train"]) > args.max_samples:
        ds["train"] = ds["train"].select(range(args.max_samples))
        print(f"\nCapped train split to {args.max_samples} samples.")

    # Build and save the label mapping (Task 2: encode labels + save id2label.json)
    id2label = {str(i): name for i, name in enumerate(LABELS)}
    with open(args.mapping, "w") as f:
        json.dump(id2label, f, indent=2)
    print(f"\nSaved label mapping -> {args.mapping}")
    print("  id2label:", id2label)

    # Save the prepared dataset locally (NOT committed — see .gitignore)
    os.makedirs(args.out, exist_ok=True)
    ds.save_to_disk(args.out)
    print(f"Saved prepared dataset -> {args.out}")
    print("\nDone. Commit only id2label.json; the data/ folder is gitignored.")


if __name__ == "__main__":
    main()
