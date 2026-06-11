"""
model_utils.py — Task 3: Select & Load a Model from Hugging Face
================================================================
Helper functions to load the tokenizer and the sequence-classification
model with the correct number of output labels derived from id2label.json.

Chosen model: distilbert-base-uncased  (see report for the rationale).
"""
import json

from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "distilbert-base-uncased"


def load_id2label(path: str = "id2label.json"):
    """Load the id2label mapping and derive label2id + label count."""
    with open(path) as f:
        raw = json.load(f)
    id2label = {int(k): v for k, v in raw.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id, len(id2label)


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    """Task 3: load the tokenizer for the chosen model."""
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str = DEFAULT_MODEL, mapping_path: str = "id2label.json"):
    """Task 3: load the model with the correct number of output labels."""
    id2label, label2id, num_labels = load_id2label(mapping_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model


if __name__ == "__main__":
    tok = load_tokenizer()
    mdl = load_model()
    print("Tokenizer:", tok.__class__.__name__)
    print("Model:", mdl.__class__.__name__, "| num_labels:", mdl.config.num_labels)
    print("id2label:", mdl.config.id2label)
