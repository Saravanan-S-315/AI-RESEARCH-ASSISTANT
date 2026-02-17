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

### 2) Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
GROQ_API=your_groq_api_key_here
```

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

## Troubleshooting

- **Missing `GROQ_API`**: App stops at startup with a clear error banner.
- **Paper fetch fails**: Usually transient API/network issue; retry search.
- **PDF parsing errors**: Some links may be blocked or malformed; choose another paper.
