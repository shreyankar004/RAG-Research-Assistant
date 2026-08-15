# RAG Research Assistant (customized fork)

A multi-agent Retrieval-Augmented Generation app: PDF ingestion → chunking →
TF-IDF embeddings → retrieval → Groq/Mistral/Gemini-powered planning,
research-report generation, summarization, and grounded chat — served through
a Streamlit UI.

## Provenance

This project is **not built from scratch by me**. It started from an
existing open-source RAG/Streamlit template I found online. I could not
locate a `LICENSE` or author name inside the archive I started from — if you
recognize this codebase and are the original author, please open an issue or
reach out so I can credit you properly here.

> ⚠️ If you're reading this because you're deciding whether to list this
> project as fully "developed by you" on a resume — don't. List it honestly
> as "customized/extended an existing RAG framework" and describe the
> specific parts you changed (see below). That's a legitimate, defensible
> line and it's what employers actually want to hear about in an interview.

## What I customized

- **Multi-provider LLM fallback** (`agents/llm.py`): the original wired the
  agents to Groq only. I refactored the LLM layer so it tries Groq first,
  then falls back automatically to Mistral, then Gemini, if a provider is
  unavailable, rate-limited, or times out — all on free tiers, no OpenAI
  dependency.
- Added `MISTRAL_API_KEY` / `GEMINI_API_KEY` config plumbing
  (`config.py`, `.env.example`, `requirements.txt`).
- Kept the existing TF-IDF retrieval, chunking, PDF ingestion, web search,
  and orchestration logic unchanged — those parts are documented as-is below.

## Architecture (as inherited)

- `agents/pdf_reader.py` — PDF text extraction
- `agents/chunking.py` — overlapping context-preserving chunking
- `agents/embeddings.py` — lightweight NumPy TF-IDF embeddings
- `agents/vector_store.py` — in-memory NumPy vector index
- `agents/retriever.py` — top-k cosine similarity retrieval
- `agents/web_search.py` — Tavily-based live web search
- `agents/planner.py`, `agents/research.py`, `agents/summarizer.py`,
  `agents/chat.py` — LLM-driven agents (now multi-provider, see above)
- `orchestration.py` — agent role/goal/backstory definitions
- `app.py` — Streamlit front end

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in at least one of GROQ_API_KEY / MISTRAL_API_KEY / GEMINI_API_KEY
streamlit run app.py
```

## Known limitation

Streaming responses are currently only wired up for Groq; if Groq is
unavailable, chat falls back to a single non-streamed call through Mistral
or Gemini instead of token-by-token streaming. A good next improvement would
be to add native streaming for the other two providers.
