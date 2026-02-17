# AI Research Assistant 🤖

An open-source AI research assistant to **search papers**, **read summaries**, and **chat with a selected paper PDF** using retrieval-augmented generation (RAG).

Built with Streamlit + LangChain + Groq + FAISS.

---

## Features

- 🔍 Search Semantic Scholar papers by topic
- 📄 View paper title, authors, abstract/TLDR, and PDF link
- 🤖 Ask questions about one selected paper with contextual Q&A
- 🧠 Vector search powered by sentence-transformer embeddings and FAISS
- 🚀 Deployment-ready setup with Docker, health checks, CI workflow, and environment templates

---

## Quickstart (Local)

### 1) Clone and install

```bash
git clone https://github.com/Saravanan-S-315/AI-RESEARCH-ASSISTANT.git
cd AI-RESEARCH-ASSISTANT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure API key (choose one)

**Option A (recommended for deployments):** set `GROQ_API` in environment/secrets.

```bash
cp .env.example .env
# edit .env and set GROQ_API
```

**Option B (quick local run):** paste Groq API key in the app sidebar field `Groq API Key`.

Optional:

- `GROQ_MODEL` (default: `llama3-70b-8192`)
- `LOG_LEVEL` (default: `INFO`)
- `REQUEST_TIMEOUT_SECONDS` (default: `30`)

### 3) Run app

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Docker Deployment

### Build and run with Docker

```bash
docker build -t ai-research-assistant .
docker run --rm -p 8501:8501 --env-file .env ai-research-assistant
```

### Or use Docker Compose

```bash
docker compose up --build
```

The container includes a healthcheck against:

- `GET /_stcore/health`

---

## Deployment Notes

- Keep `GROQ_API` in secret manager / environment variables, never commit it.
- The app runs in headless mode and binds to `0.0.0.0:8501` in containers.
- CI pipeline (`.github/workflows/ci.yml`) validates installation and source compilation on push/PR.

---

## Project Structure

```text
.
├── app.py
├── retriver.py
├── requirements.txt
├── .env.example
├── .streamlit/config.toml
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---


## Reliable Deployment (Docker + CI)

If your goal is: **"can be deployed and run reliably with Docker + CI"**, use this exact flow.

### 1) One-time local setup

```bash
cp .env.example .env
# edit .env and set GROQ_API
```

### 2) Build and run locally with Docker

```bash
docker build -t ai-research-assistant .
docker run --rm -p 8501:8501 --env-file .env ai-research-assistant
```

Verify health endpoint:

```bash
curl -f http://localhost:8501/_stcore/health
```

Expected: `ok` (exit code 0).

### 3) Run with Docker Compose (recommended)

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f ai-research-assistant
```

### 4) CI checks (GitHub Actions)

On every push/PR, workflow `.github/workflows/ci.yml` runs:

- dependency install from `requirements.txt`
- python source compile check for `app.py` and `retriver.py`

You can mirror this locally:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m compileall app.py retriver.py
```

### 5) Production checklist

- Store `GROQ_API` in deployment secret manager (not in git), or provide it at runtime via the sidebar key field for local/dev use.
- Keep container health checks enabled.
- Pin image tags in deployment environment.
- Deploy via PR merge only (so CI always gates changes).
- Roll back by redeploying the previous image tag.

## Troubleshooting

- **Missing `GROQ_API`**: Provide a key in one of these places: sidebar `Groq API Key`, `.env` (`GROQ_API=...`), or Streamlit secrets.
- **Paper fetch fails**: Usually transient API/network issue; retry search.
- **PDF parsing errors**: Some links may be blocked or malformed; choose another paper.
- **`pip install -r requirements.txt` fails with proxy / 403**:
  - Configure pip for your network/proxy:
    ```bash
    pip config set global.proxy http://<user>:<pass>@<proxy-host>:<port>
    pip config set global.index-url https://pypi.org/simple
    ```
  - Or set temporary env vars before install:
    ```bash
    export HTTPS_PROXY=http://<user>:<pass>@<proxy-host>:<port>
    export HTTP_PROXY=http://<user>:<pass>@<proxy-host>:<port>
    pip install -r requirements.txt
    ```
- **`ModuleNotFoundError: No module named 'langchain.chains'`**:
  - This project no longer depends on `langchain.chains`; it uses `langchain-core` + `langchain-community` APIs.
  - Recreate a clean environment to remove conflicting old packages:
    ```bash
    rm -rf .venv
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```
