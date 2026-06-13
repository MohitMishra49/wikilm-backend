---
title: WikiLM API
emoji: 📖
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# WikiLM — GPT-2 Fine-tuned on WikiText-103

FastAPI inference server for a GPT-2 model fine-tuned on WikiText-103-raw-v1.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Root health check |
| GET | `/health` | Detailed status |
| POST | `/generate` | Generate text |
| GET | `/docs` | Interactive Swagger UI |

## Example request

```bash
curl -X POST https://YOUR-SPACE-URL.hf.space/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The history of machine learning began with",
    "max_new_tokens": 150,
    "temperature": 0.8,
    "top_p": 0.92
  }'
```

## Example response

```json
{
  "generated_text": "The history of machine learning began with early statistical methods...",
  "prompt": "The history of machine learning began with",
  "continuation": "early statistical methods...",
  "tokens_generated": 142,
  "inference_time_ms": 3200,
  "device": "cpu"
}
```
Model weights are hosted on Hugging Face Spaces.

link for the check the project - https://wikilm-frontend.vercel.app/