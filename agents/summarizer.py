from typing import Dict
from agents.llm import GroqAgentMixin, LLMResult

SYSTEM = "You are a precise executive summary writer. Create concise, high-value summaries."

class SummarizerAgent(GroqAgentMixin):
    def summarize(self, topic: str, research: Dict[str, str], context: str) -> LLMResult:
        content = "\n\n".join(f"**{k}**: {v}" for k, v in research.items())
        prompt = f"""Create an executive summary of this research on: **{topic}**

Research Content:
{content}
Additional Context:
{context[:2000]}
Write:
1. A 2-sentence overview
2. 5 key findings (bullet points)
3. 3 actionable recommendations
4. 1 sentence conclusion

Be crisp, specific, and high-value."""
        fallback = f"Research on {topic} covers key applications, advantages, and challenges with significant future potential."
        return self.generate(prompt, system=SYSTEM, fallback=fallback)
