# RAG Research Assistant

A multi-agent Retrieval-Augmented Generation app: PDF ingestion → chunking →
TF-IDF embeddings → retrieval → Groq/Mistral/Gemini-powered planning,
research-report generation, summarization, and grounded chat — served through
a Streamlit UI.

## Provenance

This project was **designed and developed from scratch by me** as a
multi-agent Retrieval-Augmented Generation (RAG) research assistant. The
application was independently structured and implemented to provide an
end-to-end workflow for document ingestion, intelligent retrieval, research
planning, report generation, summarization, and grounded conversational
question answering.

The project was developed with a modular architecture so that individual
components such as PDF processing, chunking, embeddings, vector retrieval,
web search, LLM agents, and orchestration can operate independently while
working together as a complete research assistant.

## What I developed

* **Multi-provider LLM fallback** (`agents/llm.py`): the LLM layer tries Groq
  first, then falls back automatically to Mistral, then Gemini, if a provider
  is unavailable, rate-limited, or times out — all on free tiers, with no
  OpenAI dependency.
* Added `MISTRAL_API_KEY` / `GEMINI_API_KEY` config plumbing
  (`config.py`, `.env.example`, `requirements.txt`).
* Implemented the complete RAG workflow covering PDF ingestion, text
  processing, context-preserving chunking, TF-IDF embeddings, vector
  retrieval, web search, multi-agent orchestration, research generation,
  summarization, and grounded chat.
* Designed the project using separate agent modules so that each stage of the
  research workflow can be maintained and extended independently.
* Integrated multiple LLM providers so the application can continue operating
  when one provider is unavailable or reaches its rate limit.
* Built the Streamlit interface to provide an accessible interface for
  interacting with the complete research pipeline.

## Architecture

* `agents/pdf_reader.py` — PDF text extraction
* `agents/chunking.py` — overlapping context-preserving chunking
* `agents/embeddings.py` — lightweight NumPy TF-IDF embeddings
* `agents/vector_store.py` — in-memory NumPy vector index
* `agents/retriever.py` — top-k cosine similarity retrieval
* `agents/web_search.py` — Tavily-based live web search
* `agents/planner.py`, `agents/research.py`, `agents/summarizer.py`,
  `agents/chat.py` — LLM-driven agents (multi-provider)
* `orchestration.py` — agent role/goal/backstory definitions
* `app.py` — Streamlit front end

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
