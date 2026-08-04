from agents.llm import GroqAgentMixin, LLMResult

SYSTEM = """You are a senior research strategist. Create precise, actionable research plans.
Always structure output with clear numbered sections."""

class PlannerAgent(GroqAgentMixin):
    def create_plan(self, topic: str, context: str = "", mode: str = "general") -> LLMResult:
        source_note = "web search results" if mode == "general" else "the uploaded PDF document"
        prompt = f"""Create a comprehensive research plan for: **{topic}**

{"Context from " + source_note + ":" if context else ""}
{context[:5000] if context else ""}

Return a structured plan with these exact sections:
## 1. Research Objective
## 2. Key Concepts to Explore  
## 3. Methodology
## 4. Expected Findings
## 5. Research Questions (5 specific questions)
## 6. Suggested Structure for Final Report

Be specific, practical, and grounded in the context if provided."""
        fallback = f"## Research Plan: {topic}\n1. Introduction\n2. Key Concepts\n3. Methodology\n4. Analysis\n5. Conclusion"
        return self.generate(prompt, system=SYSTEM, fallback=fallback)
