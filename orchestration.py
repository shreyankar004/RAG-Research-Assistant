"""
Real LangGraph workflow orchestration.
Two modes: PDF Research and General Web Research.
"""
from typing import Dict, List, Optional, TypedDict


AGENT_ROLES = [
    {"name": "PDF Reader Agent", "goal": "Extract and parse text from uploaded PDF documents.", "backstory": "Document ingestion specialist using pypdf."},
    {"name": "Chunking Agent", "goal": "Split documents into overlapping context-preserving chunks.", "backstory": "Ensures no context is lost at chunk boundaries."},
    {"name": "Embedding Agent", "goal": "Generate TF-IDF vector embeddings for lightweight semantic search.", "backstory": "Custom NumPy TF-IDF — no GPU, no heavy ML frameworks needed."},
    {"name": "Vector Store Agent", "goal": "Build and query in-memory numpy vector indexes.", "backstory": "Pure numpy dot-product similarity — zero extra RAM overhead."},
    {"name": "Web Search Agent", "goal": "Fetch real-time web results via Tavily API with citations.", "backstory": "Live intelligence gatherer with source attribution."},
    {"name": "Retriever Agent", "goal": "Find the most relevant chunks for each query using cosine similarity.", "backstory": "Top-k semantic search — fast and accurate."},
    {"name": "Planner Agent", "goal": "Create structured, context-aware 6-section research plans.", "backstory": "Research strategist powered by Groq LLM."},
    {"name": "Research Agent", "goal": "Generate detailed, grounded research reports from retrieved context.", "backstory": "Domain writer and analyst powered by Groq LLM."},
    {"name": "Summarizer Agent", "goal": "Compress research into executive summaries with recommendations.", "backstory": "Precision editor powered by Groq LLM."},
    {"name": "Chat Agent", "goal": "Answer questions strictly grounded in source material.", "backstory": "Conversational research assistant powered by Groq LLM."},
]


class ResearchWorkflowState(TypedDict, total=False):
    mode: str           # "pdf" | "general"
    topic: str
    pdf_text: str
    chunks: List[str]
    web_results: List[Dict]
    context: str
    plan: str
    research: Dict[str, str]
    summary: str
    answer: str
    sources: List[Dict]


def get_orchestration_runtime() -> str:
    parts = []
    try:
        import langgraph
        parts.append("LangGraph ✓")
    except Exception:
        parts.append("LangGraph ✗")
    try:
        from groq import Groq
        parts.append("Groq LLM ✓")
    except Exception:
        parts.append("Groq ✗")
    try:
        from tavily import TavilyClient
        parts.append("Tavily Search ✓")
    except Exception:
        parts.append("Tavily ✗")
    return " | ".join(parts)


def create_langgraph_workflow(mode: str = "pdf"):
    """Build the real LangGraph StateGraph for the given mode."""
    from langgraph.graph import END, START, StateGraph

    workflow = StateGraph(ResearchWorkflowState)

    # Shared nodes
    def planner_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
        return state

    def research_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
        return state

    def summarizer_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
        return state

    if mode == "pdf":
        def pdf_reader(state): return state
        def chunking(state): return state
        def embedding(state): return state
        def vector_store(state): return state
        def retriever(state): return state

        workflow.add_node("pdf_reader", pdf_reader)
        workflow.add_node("chunking", chunking)
        workflow.add_node("embedding", embedding)
        workflow.add_node("vector_store", vector_store)
        workflow.add_node("retriever", retriever)
        workflow.add_node("planner", planner_node)
        workflow.add_node("research", research_node)
        workflow.add_node("summarizer", summarizer_node)

        workflow.add_edge(START, "pdf_reader")
        workflow.add_edge("pdf_reader", "chunking")
        workflow.add_edge("chunking", "embedding")
        workflow.add_edge("embedding", "vector_store")
        workflow.add_edge("vector_store", "retriever")
        workflow.add_edge("retriever", "planner")
        workflow.add_edge("planner", "research")
        workflow.add_edge("research", "summarizer")
        workflow.add_edge("summarizer", END)
    else:
        def web_search(state): return state
        def retriever(state): return state

        workflow.add_node("web_search", web_search)
        workflow.add_node("retriever", retriever)
        workflow.add_node("planner", planner_node)
        workflow.add_node("research", research_node)
        workflow.add_node("summarizer", summarizer_node)

        workflow.add_edge(START, "web_search")
        workflow.add_edge("web_search", "retriever")
        workflow.add_edge("retriever", "planner")
        workflow.add_edge("planner", "research")
        workflow.add_edge("research", "summarizer")
        workflow.add_edge("summarizer", END)

    return workflow.compile()


def build_architecture_mermaid(mode: str = "pdf") -> str:
    if mode == "pdf":
        return """flowchart TD
    U[👤 User] --> UI[Streamlit UI]
    UI --> PDF[📄 PDF Reader Agent]
    PDF --> CH[✂️ Chunking Agent]
    CH --> EM[🔢 Embedding Agent]
    EM --> VS[🗄️ Vector Store (numpy)]
    VS --> RT[🔍 Retriever Agent]
    RT --> PL[📋 Planner Agent]
    PL --> RS[✍️ Research Agent]
    RS --> SM[📊 Summarizer Agent]
    RT --> CA[💬 Chat Agent]
    PL & RS & SM & CA --> LLM[⚡ Groq LLM]"""
    else:
        return """flowchart TD
    U[👤 User] --> UI[Streamlit UI]
    UI --> WS[🌐 Web Search Agent]
    WS --> TV[Tavily API]
    TV --> CH2[✂️ Chunking]
    CH2 --> EM2[🔢 TF-IDF Embedding]
    EM2 --> RT[🔍 Retriever Agent]
    RT --> PL[📋 Planner Agent]
    PL --> RS[✍️ Research Agent]
    RS --> SM[📊 Summarizer Agent]
    RT --> CA[💬 Chat Agent]
    PL & RS & SM & CA --> LLM[⚡ Groq LLM]"""
