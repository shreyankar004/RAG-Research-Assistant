from dataclasses import dataclass
from typing import Dict
from agents.llm import GroqAgentMixin

SYSTEM = """You are a senior research analyst and technical writer. 
Write detailed, evidence-based research sections. 
Ground all claims in the provided context. Be thorough and insightful."""
@dataclass
class ResearchResult:
    ok: bool
    sections: Dict[str, str]
    error: str = ""
SECTIONS = ["Introduction", "Core Concepts", "Applications & Use Cases", "Advantages", "Challenges & Limitations", "Future Directions"]
class ResearchAgent(GroqAgentMixin):
    def research(self, topic: str, plan: str, context: str, mode: str = "general") -> ResearchResult:
        source = "web sources" if mode == "general" else "the uploaded PDF"
        prompt = f"""Write a detailed, well-structured research report on: **{topic}**

Research Plan:
{plan}

Context from {source}:
{context[:8000]}

Write exactly these six sections with rich content (each 150-250 words):
## Introduction
## Core Concepts  
## Applications & Use Cases
## Advantages
## Challenges & Limitations
## Future Directions

Ground every claim in the context. Be analytical and insightful, not just descriptive."""
        result = self.generate(prompt, system=SYSTEM)
        if not result.ok:
            return ResearchResult(False, {}, result.text)
        return ResearchResult(True, self._parse_sections(result.text, topic))

    def _parse_sections(self, text: str, topic: str) -> Dict[str, str]:
        sections = {s: "" for s in SECTIONS}
        current = None
        for line in text.splitlines():
            clean = line.strip().lstrip("#").strip()
            matched = next((s for s in SECTIONS if clean.lower().startswith(s.lower())), None)
            if matched:
                current = matched
                remainder = clean[len(matched):].strip(" :-")
                if remainder:
                    sections[current] += remainder + " "
            elif current and clean:
                sections[current] += clean + " "
        # If parsing failed, put everything in Introduction
        if not any(v.strip() for v in sections.values()):
            sections["Introduction"] = text
        return {k: v.strip() for k, v in sections.items() if v.strip()}
