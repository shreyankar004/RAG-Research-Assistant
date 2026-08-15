"""
LLM layer with multi-provider fallback: Groq -> Mistral -> Gemini.
All three are used on free tiers, no OpenAI dependency.
Direct API calls — no ThreadPoolExecutor (causes issues on Railway).
"""
from dataclasses import dataclass
from time import sleep

import streamlit as st
from config import (
    GROQ_API_KEY, GROQ_MODEL,
    MISTRAL_API_KEY, MISTRAL_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    LLM_MAX_TOKENS, LLM_RETRIES, LLM_TIMEOUT_SECONDS,
)


@dataclass
class LLMResult:
    ok: bool
    text: str
    provider: str = ""


# ---------------------------------------------------------------------------
# Client loaders — each cached, each optional (only used if a key is set)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_mistral_client():
    if not MISTRAL_API_KEY:
        return None
    try:
        from mistralai import Mistral
        return Mistral(api_key=MISTRAL_API_KEY)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_gemini_client():
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        return genai
    except Exception:
        return None


class MultiProviderAgentMixin:
    """
    Tries Groq first (fastest free tier), then Mistral, then Gemini.
    Any single provider being down, rate-limited, or missing a key
    doesn't take the app down — it just falls through to the next one.
    """

    def __init__(self):
        self._groq = None
        self._mistral = None
        self._gemini = None

    @property
    def groq_client(self):
        if self._groq is None:
            self._groq = load_groq_client()
        return self._groq

    @property
    def mistral_client(self):
        if self._mistral is None:
            self._mistral = load_mistral_client()
        return self._mistral

    @property
    def gemini_client(self):
        if self._gemini is None:
            self._gemini = load_gemini_client()
        return self._gemini

    # -- individual provider calls -----------------------------------------

    def _try_groq(self, messages):
        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def _try_mistral(self, messages):
        response = self.mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def _try_gemini(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        model = self.gemini_client.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=system or None,
        )
        response = model.generate_content(
            user,
            generation_config={"max_output_tokens": LLM_MAX_TOKENS, "temperature": 0.7},
        )
        return response.text

    # -- public API -----------------------------------------------------------

    def generate(self, prompt: str, system: str = "", fallback: str = "") -> LLMResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        providers = [
            ("Groq", self.groq_client, self._try_groq),
            ("Mistral", self.mistral_client, self._try_mistral),
            ("Gemini", self.gemini_client, self._try_gemini),
        ]

        last_error = ""
        for name, client, call in providers:
            if client is None:
                continue
            for attempt in range(LLM_RETRIES + 1):
                try:
                    text = call(messages) or fallback
                    return LLMResult(True, text.strip(), provider=name)
                except Exception as exc:
                    err = str(exc).lower()
                    if "401" in err or "auth" in err or "invalid" in err:
                        last_error = f"Invalid API key for {name}."
                    elif "429" in err or "rate" in err:
                        last_error = f"{name} rate limit hit."
                    elif "timeout" in err or "timed out" in err:
                        last_error = f"{name} request timed out."
                    else:
                        last_error = f"{name} error: {exc}"
                    if attempt < LLM_RETRIES:
                        sleep(1.5 * (attempt + 1))
            # this provider exhausted its retries — fall through to the next one

        if not any(c for _, c, _ in providers):
            return LLMResult(
                False,
                "No LLM provider configured. Set GROQ_API_KEY, MISTRAL_API_KEY, "
                "or GEMINI_API_KEY in your environment.",
            )
        return LLMResult(False, last_error or fallback)

    def stream_generate(self, prompt: str, system: str = ""):
        """Streaming is only wired up for Groq; falls back to a single
        non-streamed call through the other providers if Groq is unavailable."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self.groq_client is not None:
            try:
                stream = self.groq_client.chat.completions.create(
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
                return
            except Exception:
                pass  # fall through to non-streaming providers below

        result = self.generate(prompt, system=system)
        yield result.text if result.ok else f"\n\n[Error: {result.text}]"


# Backward-compatible alias so existing agent modules that import
# GroqAgentMixin keep working without changes.
GroqAgentMixin = MultiProviderAgentMixin
