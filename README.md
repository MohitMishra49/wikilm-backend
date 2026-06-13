---

title: WikiLM API
emoji: 📖
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
--------------

# WikiLM — GPT-2 Fine-Tuned on WikiText-103

WikiLM is a GPT-2 language model fine-tuned on the WikiText-103 dataset and deployed using FastAPI on Hugging Face Spaces. The project includes a React + TypeScript frontend hosted on Vercel.

## Live Demo

🌐 Frontend: https://wikilm-frontend.vercel.app/

🤗 Hugging Face Space: https://huggingface.co/spaces/MohitMishra4905/wikilm-api

🚀 API Endpoint: https://mohitmishra4905-wikilm-api.hf.space

## Tech Stack

* GPT-2
* Hugging Face Transformers
* FastAPI
* PyTorch
* React
* TypeScript
* Vite
* Hugging Face Spaces
* Vercel

## API Endpoints

| Method | Endpoint    | Description   |
| ------ | ----------- | ------------- |
| GET    | `/`         | Health check  |
| GET    | `/health`   | Model status  |
| POST   | `/generate` | Generate text |
| GET    | `/docs`     | Swagger UI    |

## Example Request

```bash
curl -X POST https://mohitmishra4905-wikilm-api.hf.space/generate \
-H "Content-Type: application/json" \
-d '{
  "prompt": "The history of machine learning began with",
  "max_new_tokens": 150,
  "temperature": 0.8,
  "top_p": 0.92
}'
```

## Example Response

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

## Model Information

* Base Model: GPT-2
* Dataset: WikiText-103
* Framework: PyTorch + Transformers
* Deployment: Hugging Face Spaces (Docker)

## Author

Mohit Mishra

B.Tech Student | AI & Machine Learning Enthusiast

LinkedIn: https://linkedin.com/in/YOUR-LINKEDIN

GitHub: https://github.com/MohitMishra49
