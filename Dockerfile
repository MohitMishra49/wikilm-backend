# ─────────────────────────────────────────────────────────────────────────────
# WikiLM — Dockerfile for Hugging Face Spaces (Docker SDK)
#
# HF Spaces Docker requirements:
#   - Must expose port 7860 (HF routes external traffic here)
#   - Must run as non-root user (HF runs containers as UID 1000)
#   - Container must start within ~120 seconds or HF marks it as failed
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Metadata labels (optional but good practice)
LABEL maintainer="your-name"
LABEL description="WikiLM: GPT-2 fine-tuned on WikiText-103"

# ── System dependencies ───────────────────────────────────────────────────────
# git is needed by some transformers internals
# build-essential is needed for some pip packages that compile C extensions
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ──────────────────────────────────────────────────────
# HF Spaces runs as UID 1000. We create a matching user to avoid permission errors.
RUN useradd -m -u 1000 appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker layer caches the pip install step.
# If only main.py changes, Docker reuses the cached pip layer (faster rebuilds).
COPY requirements.txt .

# Install CPU-only PyTorch from the official PyTorch index.
# This is much smaller (~700MB vs ~2.5GB for CUDA).
# If you upgrade to a paid GPU Space, change the index URL to:
#   https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application files ───────────────────────────────────────────────────
COPY main.py .

# ── Copy your fine-tuned model files ─────────────────────────────────────────
# Your model folder (from the notebook's SAVE_DIR = "./gpt2-finetuned-final")
# must be placed in the same directory as this Dockerfile and renamed to "model/".
#
# Expected files inside model/:
#   - pytorch_model.bin  OR  model.safetensors
#   - config.json
#   - vocab.json
#   - merges.txt
#   - tokenizer_config.json
#   - special_tokens_map.json
#
# The COPY below copies that entire folder into the container.
COPY  gpt2-finetuned-final/ ./gpt2-finetuned-final/

# ── Fix permissions ───────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# ── Expose port ───────────────────────────────────────────────────────────────
# HF Spaces Docker REQUIRES port 7860.
# Uvicorn must listen on this exact port.
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
# Docker will mark the container unhealthy if this fails.
# HF Spaces also uses this to determine readiness.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
# --host 0.0.0.0  → listen on all interfaces (required in containers)
# --port 7860     → HF Spaces required port
# --workers 1     → single worker (model is too large for multiple)
# No --reload     → reload is for development only, wastes memory in prod
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
