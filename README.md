# 🔬 ResearchMind — Multi-Agent Research Assistant

A production-ready multi-agent AI research assistant deployed on Railway. Two modes: **PDF Research** (upload & analyse any PDF) and **General Web Research** (real-time web sourced reports).

🔗 ** Railway Live Demo:** https://web-production-c6fda.up.railway.app
🔗 ** Render Live Demo:** https://researchmind-87w1.onrender.com/


---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 📄 **PDF Research Mode** | Upload any PDF → RAG-powered report + grounded Q&A chat |
| 🌐 **Web Research Mode** | Enter any topic → real-time Tavily search → AI report with cited sources |
| 💬 **Grounded Chat** | Ask questions answered strictly from your source material |
| 📚 **References Panel** | Full source list with relevance scores and clickable URLs |
| ⬇️ **PDF Export** | Download the full research report as a styled PDF |
| ⚡ **Groq LLM** | Fast inference — llama-3.3-70b-versatile (free tier available) |
| 🔍 **TF-IDF Search** | Tiny NumPy-based retrieval — runs on 512MB RAM |

---

## 🤖 Agent Pipeline

```
PDF Mode:
User → PDF Reader → Chunking → TF-IDF Embedding → Vector Store (numpy)
     → Retriever → Planner (Groq) → Research (Groq) → Summarizer (Groq) → Report PDF
     → Retriever → Chat Agent (Groq) → Answer

Web Mode:
User → Web Search (Tavily) → Chunking → TF-IDF Embedding → Vector Store (numpy)
     → Retriever → Planner (Groq) → Research (Groq) → Summarizer (Groq) → Report PDF
     → Retriever → Chat Agent (Groq) → Answer
```

---

## 🧱 Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | **Groq** (llama-3.3-70b-versatile) | Fast, generous free tier |
| Web Search | **Tavily** | Semantic search with citations |
| Embeddings | **Custom TF-IDF** | No model download, no torch, predictable RAM |
| Vector Search | **numpy** dot product | No FAISS needed, zero overhead |
| PDF Parsing | **pypdf** | Lightweight, reliable |
| Agent Workflow | **LangGraph** | Production-ready state machine |
| UI | **Streamlit** | Fast, clean, responsive |
| Deployment | **Railway** | Free tier, auto-deploy from GitHub |

---

## 🚀 Local Setup

```bash
# 1. Clone
git clone https://github.com/anurajput06/ResearchMind.git
cd ResearchMind

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
cp .env.example .env
# Edit .env and fill in your keys

# 5. Run
streamlit run app.py
```

---

## 🔑 API Keys (both free)

| Key | Get it at | Free tier |
|-----|----------|-----------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | 14,400 req/day |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | 1,000 searches/month |

---

## ☁️ Deploy on Railway (Recommended)

Railway gives you **always-on deployment** with 500MB RAM — perfect for this app.

### Step 1 — Create Railway account
Go to [railway.app](https://railway.app) → Sign up with GitHub (free)

### Step 2 — New project
- Dashboard → **New Project** → **Deploy from GitHub repo**
- Select `ResearchMind` repo → Deploy

### Step 3 — Add environment variables
Railway dashboard → your service → **Variables** tab → Add:

| Variable | Value |
|----------|-------|
| `GROQ_API_KEY` | your Groq key |
| `TAVILY_API_KEY` | your Tavily key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` |
| `CHUNK_SIZE` | `1200` |
| `CHUNK_OVERLAP` | `180` |
| `LLM_MAX_TOKENS` | `2048` |
| `LLM_TIMEOUT_SECONDS` | `60` |

### Step 4 — Set start command
Railway → Settings → **Start Command**:
```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```

### Step 5 — Deploy
Click **Deploy** → Railway builds and serves your app at a `.up.railway.app` URL ✅

### Auto-deploy on push
Every `git push` to `main` triggers an automatic redeploy on Railway.

```bash
git add .
git commit -m "your changes"
git push
# Railway redeploys automatically
```

---

## ☁️ Deploy on Render (Alternative)

### Step 1
Go to [render.com](https://render.com) → New → Web Service → Connect GitHub repo

### Step 2 — Settings
| Field | Value |
|-------|-------|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false` |
| Instance Type | Free |

### Step 3 — Environment Variables
Add `GROQ_API_KEY` and `TAVILY_API_KEY` in the Environment tab.

> ⚠️ Render free tier sleeps after 15 min of inactivity. First load after sleep takes ~30s.

---

## 📁 Project Structure

```
ResearchMind/
├── app.py                    # Main UI — light mode, tabs, two research modes
├── config.py                 # All settings via environment variables
├── orchestration.py          # LangGraph workflow + agent metadata
├── agents/
│   ├── llm.py                # Groq LLM client (direct calls, no ThreadPoolExecutor)
│   ├── pdf_reader.py         # PDF text extraction (pypdf)
│   ├── chunking.py           # Overlapping sentence-aware chunking
│   ├── embeddings.py         # Custom TF-IDF embeddings
│   ├── vector_store.py       # In-memory numpy vector store
│   ├── retriever.py          # Cosine similarity top-k retrieval
│   ├── web_search.py         # Tavily real-time web search
│   ├── planner.py            # Research plan generator (Groq)
│   ├── research.py           # 6-section report writer (Groq)
│   ├── summarizer.py         # Executive summary agent (Groq)
│   └── chat.py               # Grounded Q&A agent (Groq)
├── utils/
│   └── report.py             # Styled PDF report export
├── .streamlit/
│   └── config.toml           # Light theme + server config
├── render.yaml               # Render deployment config
├── Procfile                  # Heroku/legacy deployment
├── runtime.txt               # Python version pin
├── requirements.txt          # Minimal dependencies (no torch/GPU)
```

---

## 💡 Why TF-IDF instead of neural embeddings?

| | TF-IDF (this app) | sentence-transformers |
|--|--|--|
| RAM | ~30MB  | ~450MB  |
| Free tier | Works  | OOM crash  |
| Install | ~15MB  | ~1.3GB  |
| Quality | 85% | 95% |

Since Groq LLM generates the actual research content, embeddings only need to find relevant chunks — TF-IDF is completely sufficient and lets the app run on free-tier servers.

---

## Author

**Anu** · [github.com/anurajput06](https://github.com/anurajput06)
