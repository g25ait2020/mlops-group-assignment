"""
push_to_hub.py — Task 5: Push the Trained Model to Hugging Face Hub
===================================================================
Uploads the best local checkpoint (weights + tokenizer) to a PUBLIC Hugging
Face repo so the GitHub Actions inference workflow and the Docker image can
load it, then records the model URL in the W&B run summary.

Run this on Kaggle right after training (HF_TOKEN comes from Kaggle Secrets),
or locally with HF_TOKEN exported.

Example:
    python src/push_to_hub.py \
        --local-dir best_model_v1 \
        --repo your-username/distilbert-emotion
"""
import argparse
import os

from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", required=True, help="folder containing the trained model + tokenizer")
    ap.add_argument("--repo", required=True, help="target HF repo, e.g. your-username/distilbert-emotion")
    ap.add_argument("--private", action="store_true", help="create as private (default public)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set. Add it via Kaggle Secrets or export it locally.")
    login(token=token)

    print(f"Loading local model from {args.local_dir} ...")
    model = AutoModelForSequenceClassification.from_pretrained(args.local_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.local_dir)

    print(f"Pushing to https://huggingface.co/{args.repo} (private={args.private}) ...")
    model.push_to_hub(args.repo, private=args.private)
    tokenizer.push_to_hub(args.repo, private=args.private)

    url = f"https://huggingface.co/{args.repo}"
    print("Pushed:", url)

    # Log the model URL into the active/most-recent W&B run summary (Task 5).
    try:
        import wandb
        if wandb.run is None:
            wandb.init(project="mlops-assignment3", name="push-to-hub", resume="allow")
        wandb.run.summary["huggingface_model"] = url
        wandb.finish()
        print("Logged model URL to W&B run summary.")
    except Exception as e:  # pragma: no cover
        print(f"(Skipped W&B logging: {e})")


if __name__ == "__main__":
    main()
