# Dockerfile — Task 6
# Packages the inference code so anyone can run predictions with no setup.
# Uses a slim base image and installs ONLY inference dependencies.

FROM python:3.11-slim

# --- Build argument: which HF model to serve (sensible default) ---
ARG HF_MODEL_NAME=your-username/distilbert-emotion
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# Keep Python lean and quiet; cache HF models in a writable dir.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    INPUT_TEXT="I can't believe how wonderful today turned out!"

WORKDIR /app

# Install only inference dependencies (CPU torch keeps the image small).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy just the source needed for inference.
COPY src/ ./src/
COPY id2label.json .

# Default command runs a single classification using INPUT_TEXT.
ENTRYPOINT ["python", "src/inference.py"]
