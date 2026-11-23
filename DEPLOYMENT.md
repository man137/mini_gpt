# Deployment Guide: mini-chatgpt

This guide covers deploying `mini-chatgpt` into production using Docker or cloud hosting platforms.

---

## 1. Quick Start with Docker (Recommended)

The project includes a multi-stage `Dockerfile` that packages both the React UI and the FastAPI RAG backend into a single container.

### Step 1: Configure `.env`

Copy `.env.example` to `.env` (if not done already):
```bash
cp .env.example .env
```

To use a cloud API provider (instant, fast, and no GPU required), configure your chosen backend in `.env`:

#### Option A: Groq (Ultra-fast & free tier available)
```env
LLM_BACKEND=openai
OPENAI_API_KEY=gsk_your_groq_api_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant
```

#### Option B: OpenAI
```env
LLM_BACKEND=openai
OPENAI_API_KEY=sk-your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

#### Option C: OpenRouter
```env
LLM_BACKEND=openai
OPENAI_API_KEY=sk-or-your_openrouter_api_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

#### Option D: Fake / Demo Mode (No API key needed)
```env
LLM_BACKEND=fake
```

---

### Step 2: Build & Run with Docker Compose

Run from inside the `mini-chatgpt` directory:

```bash
docker compose up -d --build
```

View the logs:
```bash
docker compose logs -f
```

Open your browser at:
- **Application UI**: [http://localhost:8000](http://localhost:8000)
- **API Health**: [http://localhost:8000/health](http://localhost:8000/health)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

To stop the container:
```bash
docker compose down
```

---

## 2. Deploying to Cloud Platforms

### A. Deploy to Cloud VPS (DigitalOcean, Hetzner, AWS EC2, GCP Compute Engine)
1. Launch any Linux VPS (Ubuntu 22.04 / 24.04 with Docker installed).
2. Clone the repository and navigate into it:
   ```bash
   git clone <repo-url>
   cd mini-chatgpt/mini-chatgpt
   ```
3. Create your `.env` file with your settings and API keys.
4. Launch with Docker Compose:
   ```bash
   docker compose up -d --build
   ```
5. *(Optional)* Add Nginx or Caddy with Let's Encrypt for automatic HTTPS/SSL on your domain.

---

### B. Deploy to Render / Railway / Fly.io
Because the container is self-contained with a multi-stage `Dockerfile`:
1. Connect your GitHub repository to **Render**, **Railway**, or **Fly.io**.
2. Select **Dockerfile** as the build method.
3. In the platform dashboard, set environment variables (e.g., `LLM_BACKEND=openai`, `OPENAI_API_KEY=...`, `OPENAI_MODEL=...`).
4. Set port to `8000`.
5. Deploy! Both the frontend React UI and backend API will be served seamlessly from the single deployment URL.

