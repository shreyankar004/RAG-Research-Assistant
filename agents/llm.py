"""
LLM layer using Groq.
Direct API calls — no ThreadPoolExecutor (causes issues on Railway).
"""
from dataclasses import dataclass
from time import sleep

import streamlit as st
from config import GROQ_API_KEY, GROQ_MODEL, LLM_MAX_TOKENS, LLM_RETRIES, LLM_TIMEOUT_SECONDS


@dataclass
class LLMResult:
    ok: bool
    text: str


@st.cache_resource(show_spinner=False)
def load_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
    except Exception:
        return None


class GroqAgentMixin:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = load_groq_client()
        return self._client

    def generate(self, prompt: str, system: str = "", fallback: str = "") -> LLMResult:
        if self.client is None:
            return LLMResult(
                False,
                "GROQ_API_KEY is not set. Add it in Railway/Render → Variables."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error = ""
        for attempt in range(LLM_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=0.7,
                )
                text = response.choices[0].message.content or fallback
                return LLMResult(True, text.strip())
            except Exception as exc:
                err = str(exc).lower()
                if "401" in err or "auth" in err or "invalid" in err:
                    last_error = "Invalid GROQ_API_KEY. Check your environment variables."
                elif "429" in err or "rate" in err:
                    last_error = "Groq rate limit hit. Wait a moment and try again."
                elif "timeout" in err or "timed out" in err:
                    last_error = f"Groq request timed out. Try again."
                elif "connection" in err or "network" in err:
                    last_error = "Cannot reach Groq API. Check GROQ_API_KEY in Variables."
                else:
                    last_error = f"LLM error: {exc}"
                if attempt < LLM_RETRIES:
                    sleep(1.5 * (attempt + 1))

        return LLMResult(False, last_error or fallback)

    def stream_generate(self, prompt: str, system: str = ""):
        if self.client is None:
            yield "GROQ_API_KEY is missing. Add it in environment variables."
            return
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            stream = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.7,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            yield f"\n\n[Error: {exc}]"
