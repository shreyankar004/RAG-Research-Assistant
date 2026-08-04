from typing import List
from agents.llm import GroqAgentMixin, LLMResult
from agents.retriever import RetrievedChunk

SYSTEM = """You are an intelligent research assistant. Answer questions accurately using only the provided context.
If the answer isn't in the context, say so clearly. Be concise but complete."""

class ChatAgent(GroqAgentMixin):
    def answer(self, question: str, chunks: List[RetrievedChunk], mode: str = "pdf") -> LLMResult:
        if mode == "pdf":
            context = "\n\n".join(
                f"[Chunk {c.chunk_id}, relevance: {c.score:.2f}]\n{c.text}" for c in chunks
            )
            prompt = f"""Answer this question using only the PDF context below.
If the answer is not in the context, say: "This information is not found in the uploaded document."

Question: {question}

PDF Context:
{context[:7000]}"""
        else:
            context = "\n\n".join(f"[Source {i+1}]\n{c.text}" for i, c in enumerate(chunks))
            prompt = f"""Answer this question using the web research context below.

Question: {question}

Web Research Context:
{context[:7000]}"""
        return self.generate(prompt, system=SYSTEM, fallback="I could not generate a response. Please try again.")

    def answer_stream(self, question: str, chunks: List[RetrievedChunk], mode: str = "pdf"):
        context = "\n\n".join(
            f"[Source {i+1}, relevance: {c.score:.2f}]\n{c.text}" for i, c in enumerate(chunks)
        )
        source = "the PDF document" if mode == "pdf" else "web research results"
        prompt = f"""Answer this question using only {source} context below.

Question: {question}

Context:
{context[:7000]}"""
        yield from self.stream_generate(prompt, system=SYSTEM)
