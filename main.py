"""
WikiLM — FastAPI Inference Server
Fine-tuned GPT-2 on WikiText-103

Endpoints:
  GET  /          → health check (HF Spaces pings this)
  GET  /health    → detailed status
  POST /generate  → text generation
"""

import os
import time
import torch
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

# MODEL_PATH: where to load from.
# On HF Spaces, model files sit next to this file, so "./model" works.
# You can also use "gpt2" to load the base model if you haven't uploaded
# your fine-tuned weights yet (useful for testing the deployment pipeline).
MODEL_PATH = os.getenv("MODEL_PATH", "./gpt2-finetuned-final")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global state — loaded once at startup
_model: Optional[GPT2LMHeadModel] = None
_tokenizer: Optional[GPT2Tokenizer] = None


# ─── Lifespan: load model on startup ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler.
    Model is loaded once when the server starts.
    This avoids reloading on every request (which would be extremely slow).
    """
    global _model, _tokenizer

    log.info(f"Loading model from '{MODEL_PATH}' on device '{DEVICE}'...")

    try:
        _tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        _tokenizer.pad_token = _tokenizer.eos_token  # standard GPT-2 fix

        _model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        _model = _model.to(DEVICE)
        _model.eval()  # disable dropout for inference

        param_count = sum(p.numel() for p in _model.parameters())
        log.info(f"Model ready — {param_count:,} parameters on {DEVICE}")

    except Exception as e:
        log.error(f"Failed to load model: {e}")
        # We don't raise here — the server still starts so HF Spaces
        # doesn't crash the container. /health will report the failure.

    yield  # server runs here

    # Cleanup on shutdown
    log.info("Shutting down — releasing model from memory")
    del _model, _tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="WikiLM API",
    description="GPT-2 fine-tuned on WikiText-103. Generates Wikipedia-style text.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow requests from:
#   - your Vercel frontend URL
#   - localhost (for development)
#   - any HF Spaces URL (*.hf.space)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",        # Vite dev server
        "http://localhost:3000",
        "https://*.vercel.app",         # your Vercel deployment
        "https://*.hf.space",           # HF Spaces (if frontend is also on HF)
        # ADD YOUR EXACT FRONTEND URL HERE after deploying:
        # "https://your-project.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.(vercel\.app|hf\.space)$",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class GenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The text prompt to continue"
    )
    max_new_tokens: int = Field(
        default=150,
        ge=10,
        le=300,
        description="Number of new tokens to generate"
    )
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Sampling temperature. Lower = more focused, higher = more creative"
    )
    top_p: float = Field(
        default=0.92,
        ge=0.1,
        le=1.0,
        description="Nucleus sampling probability threshold"
    )
    repetition_penalty: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Penalty for repeating tokens. 1.0 = no penalty"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "The history of artificial intelligence began with",
                "max_new_tokens": 150,
                "temperature": 0.8,
                "top_p": 0.92,
                "repetition_penalty": 1.2
            }
        }
    }


class GenerationResponse(BaseModel):
    generated_text: str      # full text: prompt + continuation
    prompt: str              # original prompt echoed back
    continuation: str        # only the newly generated part
    tokens_generated: int    # how many tokens were produced
    inference_time_ms: int   # wall-clock time for generation
    device: str              # "cuda" or "cpu" — useful for debugging


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str
    cuda_available: bool
    torch_version: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """
    Root endpoint — HF Spaces checks this to know the app is alive.
    Must return 200 quickly.
    """
    return {"status": "ok", "service": "WikiLM API"}


@app.get("/health", response_model=HealthResponse)
def health():
    """Detailed health check — useful for debugging deployment issues."""
    return HealthResponse(
        status="ok" if _model is not None else "model_not_loaded",
        model_loaded=_model is not None,
        model_path=MODEL_PATH,
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
        torch_version=torch.__version__,
    )


@app.post("/generate", response_model=GenerationResponse)
def generate(req: GenerationRequest):
    """
    Main inference endpoint.

    Takes a text prompt and returns a Wikipedia-style continuation
    generated by the fine-tuned GPT-2 model.
    """
    # Guard: model must be loaded
    if _model is None or _tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please wait a moment and try again."
        )

    log.info(f"Generating for prompt: '{req.prompt[:60]}...' | "
             f"temp={req.temperature} top_p={req.top_p} tokens={req.max_new_tokens}")

    # ── Tokenize ──────────────────────────────────────────────────
    # We truncate the prompt to 128 tokens max.
    # GPT-2's context window is 1024 — we leave room for generation.
    inputs = _tokenizer(
        req.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    prompt_token_count = inputs["input_ids"].shape[1]

    # ── Generate ──────────────────────────────────────────────────
    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                do_sample=True,          # use sampling, not greedy
                pad_token_id=_tokenizer.eos_token_id,
                # Stop generating at EOS token
                eos_token_id=_tokenizer.eos_token_id,
            )
    except RuntimeError as e:
        # Catch CUDA OOM or other GPU errors gracefully
        log.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

    inference_ms = int((time.perf_counter() - t0) * 1000)

    # ── Decode ────────────────────────────────────────────────────
    # output_ids contains the full sequence (prompt + generated tokens)
    # We split them so we can return both separately

    all_ids = output_ids[0]                             # shape: [total_tokens]
    generated_ids = all_ids[prompt_token_count:]        # only new tokens

    continuation = _tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    full_text = req.prompt + " " + continuation

    log.info(f"Generated {len(generated_ids)} tokens in {inference_ms}ms")

    return GenerationResponse(
        generated_text=full_text,
        prompt=req.prompt,
        continuation=continuation,
        tokens_generated=len(generated_ids),
        inference_time_ms=inference_ms,
        device=DEVICE,
    )
