# -*- coding: utf-8 -*-
"""
nlu_router.py

Rule-based NLU (intent classification + year extraction) + optional safe LLM refinement.

Behavior:
 - Rule-based classification and year extraction unchanged.
 - If SEARCH_API_KEY or BING_SUBSCRIPTION_KEY is set, a lightweight web search is performed
   when the rule-based agent can't answer; snippets are passed to the LLM.
 - LLM refinement used only when OPENAI_API_KEY is set and 'openai' package is importable.
 - Any LLM or search error is logged and the original rule-based answer is returned.
 - Non-invasive: does not modify other modules or files.
"""
from typing import Optional, List
import re
import os
import logging
from dotenv import load_dotenv

# external dependency for HTTP calls; safe if not used
import requests

# Load .env (optional) for OPENAI_API_KEY, OPENAI_MODEL, SEARCH_API_KEY, etc.
load_dotenv()

# -------------------------
# Logging
# -------------------------
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# -------------------------
# Config (declare before use)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-5.1"
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "10"))  # seconds
LLM_MIN_WORDS = int(os.getenv("LLM_MIN_WORDS", "5"))  # skip LLM for tiny answers

# Search config: prefer SerpAPI if SEARCH_API_KEY present; otherwise allow Bing/Azure
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "serpapi").lower()  # 'serpapi' or 'bing'
BING_SUBSCRIPTION_KEY = os.getenv("BING_SUBSCRIPTION_KEY")
BING_ENDPOINT = os.getenv("BING_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")

# -------------------------
# Rule-based NLU (unchanged)
# -------------------------
INTENT_KEYWORDS = {
    "labour_overview": [
        "labour", "labor", "employment", "unemployment", "workforce",
        "jobs", "workers", "labour market", "labor market",
    ],
    "top_occupations": [
        "top occupation", "most common jobs", "most common occupations",
        "top jobs", "popular jobs", "occupation",
    ],
    "households": [
        "household", "households", "family", "families",
    ],
    "density": [
        "population density", "densely populated", "density",
    ],
    "housing_units": [
        "housing units", "dwellings", "apartments", "houses",
        "residential units",
    ],
    "students": [
        "students", "school enrollment", "pupils", "enrolment",
    ],
    "teachers": [
        "teachers", "teaching staff", "instructors",
    ],
    "higher_education": [
        "higher education", "university", "universities", "college",
        "tertiary education",
    ],
}


def classify_intent(question: str) -> str:
    q = (question or "").lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return intent
    if "population" in q:
        return "density"
    return "labour_overview" if "unemployment" in q or "employment" in q else "unknown"


def extract_year(question: str, default_year: Optional[int] = None) -> Optional[int]:
    years = re.findall(r"\b(19[0-9]{2}|20[0-9]{2}|2100)\b", question or "")
    if years:
        try:
            return int(years[0])
        except Exception:
            pass
    return default_year


# -------- Hybrid intent fallback (non-destructive) ----------
def classify_intent_hybrid(question: str, use_llm_fallback: bool = True) -> str:
    """
    First run the rule-based classifier. If it returns 'unknown' and
    use_llm_fallback is True, attempt a low-cost LLM call as a fallback.
    Returns one of the known intent labels or 'unknown'.
    Safe behavior:
      - If OPENAI_API_KEY or openai package not available -> returns rule result.
      - LLM is only used when rule-based returns 'unknown'.
      - LLM response is validated against known intents.
      - Any exceptions -> return rule result.
    """
    # Primary: rule-based classifier
    rule_intent = classify_intent(question)
    if rule_intent != "unknown" or not use_llm_fallback:
        return rule_intent

    # If LLM fallback disabled or unavailable, return unknown
    if not use_llm_fallback or not OPENAI_API_KEY:
        return "unknown"

    # Lazy-detect OpenAI availability here (keeps module import light)
    try:
        import openai  # type: ignore
        _OPENAI_AVAILABLE = True
    except Exception:
        LOG.debug("openai package not importable; LLM disabled for classify_intent_hybrid.")
        return "unknown"

    # Build a tiny deterministic prompt asking for a single label
    prompt = (
        "You are a classifier. Choose one label from the list exactly (no extra text): "
        f"{list(INTENT_KEYWORDS.keys())}\n\n"
        f"Question: {question}\n\nLabel:"
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        # try old-style ChatCompletion if present
        if hasattr(openai, "ChatCompletion"):
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL, messages=messages, max_tokens=8, temperature=0.0, request_timeout=OPENAI_TIMEOUT
            )
            label = None
            if resp and "choices" in resp and len(resp["choices"]) > 0:
                label = resp["choices"][0].get("message", {}).get("content", "").strip().split()[0]
        else:
            # try new-style client if exposed
            try:
                from openai import OpenAI as _OpenAIClass  # type: ignore
                client = _OpenAIClass(api_key=OPENAI_API_KEY)
                resp2 = client.chat.completions.create(
                    model=OPENAI_MODEL, messages=messages, max_tokens=8, temperature=0.0
                )
                choices = getattr(resp2, "choices", None) or (resp2.get("choices") if isinstance(resp2, dict) else None)
                label = None
                if choices and len(choices) > 0:
                    first = choices[0]
                    if isinstance(first, dict):
                        label = first.get("message", {}).get("content", "").strip().split()[0]
                    elif hasattr(first, "message") and hasattr(first.message, "content"):
                        label = first.message.content.strip().split()[0]
            except Exception:
                LOG.debug("No compatible OpenAI client found for classify_intent_hybrid.", exc_info=True)
                return "unknown"

        # Normalize and validate label
        if label:
            label = label.strip().strip('"').strip("'").lower()
            for known in INTENT_KEYWORDS.keys():
                if label == known or label.replace("-", "_") == known:
                    return known
        return "unknown"
    except Exception as e:
        LOG.exception("LLM fallback in classify_intent_hybrid failed: %s", e)
        return "unknown"


# -------------------------
# Helpers: web search snippets
# -------------------------
def search_web_snippets(query: str, num_results: int = 3) -> Optional[str]:
    """
    Return a short concatenated string of top search snippets (title + snippet + link).
    Supports SerpAPI (via SEARCH_API_KEY) and Bing (via BING_SUBSCRIPTION_KEY).
    Returns None on failure or if no API key available.
    """
    # Prefer SerpAPI if SEARCH_API_KEY present and engine configured
    if SEARCH_API_KEY and SEARCH_ENGINE == "serpapi":
        try:
            params = {
                "q": query,
                "api_key": SEARCH_API_KEY,
                "engine": "google",
                "num": num_results,
            }
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=6)
            r.raise_for_status()
            data = r.json()
            snippets = []
            for item in data.get("organic_results", [])[:num_results]:
                title = item.get("title", "").strip()
                snippet = item.get("snippet") or item.get("rich_snippet", {}).get("top", {}).get("text", "")
                link = item.get("link", "")
                if title or snippet or link:
                    snippets.append(f"{title}\n{snippet}\n{link}")
            if not snippets:
                return None
            return "\n\n".join(snippets)
        except Exception:
            LOG.exception("SerpAPI search failed")
            return None

    # Fallback to Bing (Azure) if key present
    if BING_SUBSCRIPTION_KEY:
        try:
            headers = {"Ocp-Apim-Subscription-Key": BING_SUBSCRIPTION_KEY}
            params = {"q": query, "count": num_results, "textDecorations": False, "textFormat": "Raw"}
            r = requests.get(BING_ENDPOINT, headers=headers, params=params, timeout=6)
            r.raise_for_status()
            data = r.json()
            snippets = []
            for item in data.get("webPages", {}).get("value", [])[:num_results]:
                title = item.get("name", "").strip()
                snippet = item.get("snippet", "")
                link = item.get("url", "")
                if title or snippet or link:
                    snippets.append(f"{title}\n{snippet}\n{link}")
            if not snippets:
                return None
            return "\n\n".join(snippets)
        except Exception:
            LOG.exception("Bing search failed")
            return None

    # No search key available
    return None


# -------------------------
# OpenAI compatibility wrappers
# -------------------------
# Try to import openai once; presence is checked at call-time as well
try:
    import openai  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    openai = None
    _OPENAI_AVAILABLE = False
    LOG.debug("openai package not importable; LLM disabled.")

# Detect new client class if possible
_have_old_chat = False
_have_new_client = False
_new_client_cls = None

if _OPENAI_AVAILABLE:
    _have_old_chat = hasattr(openai, "ChatCompletion")
    try:
        from openai import OpenAI as _OpenAI  # type: ignore
        _have_new_client = True
        _new_client_cls = _OpenAI
    except Exception:
        _have_new_client = False


def _call_openai_old(messages, model, timeout=OPENAI_TIMEOUT, max_tokens=400, temperature=0.2):
    """Old style openai.ChatCompletion.create call wrapper."""
    openai.api_key = OPENAI_API_KEY
    resp = openai.ChatCompletion.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, request_timeout=timeout
    )
    return resp


def _call_openai_new(messages, model, timeout=OPENAI_TIMEOUT, max_tokens=400, temperature=0.2):
    """New-style OpenAI client wrapper."""
    client = _new_client_cls(api_key=OPENAI_API_KEY)
    # some SDKs accept request_timeout in method
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return resp


# -------------------------
# LLM refinement (safe)
# -------------------------
def summarize_with_llm(text: str, system_prompt: Optional[str] = None, max_tokens: int = 400) -> str:
    """
    Use new OpenAI client if available; fall back to older API shapes if present.
    Always safe: returns original text on any error.
    """
    try:
        if not OPENAI_API_KEY:
            LOG.debug("OPENAI_API_KEY not set; skipping LLM.")
            return text
        if not _OPENAI_AVAILABLE:
            LOG.debug("openai package not available; skipping LLM.")
            return text
        if not text or len(text.split()) < LLM_MIN_WORDS:
            LOG.debug("Answer too short; skipping LLM refinement.")
            return text

        system_msg = system_prompt or (
            "You are a helpful statistician and data analyst. "
            "Edit the answer to be concise, clear and conversational while preserving factual numbers exactly."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Edit this answer to be concise and clear while preserving numerical facts exactly:\n\n{text}"}
        ]

        # Prefer new-style client (openai.OpenAI)
        try:
            if _have_new_client and _new_client_cls is not None:
                resp = _call_openai_new(messages, model=OPENAI_MODEL, max_tokens=max_tokens, temperature=0.2)
                choices = getattr(resp, "choices", None) or (resp.get("choices") if isinstance(resp, dict) else None)
                if choices and len(choices) > 0:
                    first = choices[0]
                    if hasattr(first, "message") and hasattr(first.message, "content"):
                        return first.message.content.strip()
                    if isinstance(first, dict):
                        msg = first.get("message", {})
                        if isinstance(msg, dict) and "content" in msg:
                            return msg["content"].strip()
        except Exception:
            LOG.debug("New-style OpenAI client not available or failed; will try old-style if present.", exc_info=True)

        # Try old-style ChatCompletion if installed and available
        try:
            if _have_old_chat and openai is not None:
                openai.api_key = OPENAI_API_KEY
                resp2 = _call_openai_old(messages, model=OPENAI_MODEL, max_tokens=max_tokens, temperature=0.2)
                if resp2 and "choices" in resp2 and len(resp2["choices"]) > 0:
                    content = resp2["choices"][0].get("message", {}).get("content")
                    if content:
                        return content.strip()
        except Exception:
            LOG.exception("Old-style OpenAI call failed (or removed).")

        LOG.debug("LLM did not produce revised text; returning original.")
        return text

    except Exception:
        LOG.exception("Unexpected error in summarize_with_llm; returning original text.")
        return text


# -------------------------
# Heuristic: canned help detection
# -------------------------
def _is_canned_help(text: Optional[str]) -> bool:
    """
    Heuristic to detect the agent's canned help / example message.
    If the text looks like 'I can help with the following:' or contains 'Examples:' or short bullet list,
    treat it as 'no meaningful answer' so we fallback to LLM.
    """
    if not text:
        return True
    t = text.strip().lower()
    canned_prefixes = [
        "i can help with the following",
        "examples:",
        "feel free to ask",
        "you can ask me about",
        "ask about"
    ]
    for p in canned_prefixes:
        if p in t[:120]:
            return True
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 3 and all(l.startswith(("-", "*")) or l.endswith("?") or len(l) < 80 for l in lines[:4]):
        return True
    if len(text.split()) <= 6 and any(k in t for k in ("help", "examples", "summarize", "overview")):
        return True
    return False


# -------------------------
# Public helper combining agent + LLM (non-invasive)
# -------------------------
def route_and_answer(agent, user_text: str, use_llm: bool = True) -> str:
    """
    Produce an answer via the agent, optionally refine it with LLM.
    Behavior:
      - call agent.answer_question(user_text)
      - if agent returns an actual factual answer -> optionally refine with LLM
      - if agent returns empty / canned help / errors -> call LLM fallback to generate an intelligent answer
    """
    raw = None
    try:
        raw = agent.answer_question(user_text)
    except Exception as e:
        LOG.exception("agent.answer_question raised an exception; falling back to LLM: %s", e)

    # If agent returned nothing meaningful or returned a canned help message -> fallback
    if raw is None or (isinstance(raw, str) and (raw.strip() == "" or _is_canned_help(raw))):
        LOG.debug("Agent returned no usable answer (or canned help). Using LLM fallback.")
        return llm_fallback_answer(user_text, agent_text=raw)

    # If LLM use not requested, return raw
    if not use_llm:
        return raw

    # Otherwise try to refine agent text with LLM; if refinement fails, return raw
    try:
        refined = summarize_with_llm(raw)
        return refined or raw
    except Exception:
        LOG.exception("LLM refinement unexpectedly failed; returning original.")
        return raw


# -------------------------
# LLM fallback (now with optional web snippets)
# -------------------------
def llm_fallback_answer(user_text: str, agent_text: Optional[str] = None) -> str:
    """
    Produce a direct ChatGPT answer when the rule-based agent can't answer.
    - If possible, include brief context (agent_text) for better continuity.
    - If SEARCH_API_KEY or BING_SUBSCRIPTION_KEY is configured, perform a lightweight web search and
      include top snippets in the prompt so the LLM can answer from the internet.
    - Robust handling of both new-style and old-style OpenAI python clients.
    - Safe: returns user-friendly message instead of raising on error.
    """
    if not OPENAI_API_KEY:
        return "I couldn't find an answer in the Bahrain data and no OpenAI API key is configured."

    # attempt to get web snippets (optional)
    web_snips = None
    try:
        web_snips = search_web_snippets(user_text, num_results=3)
    except Exception:
        LOG.exception("web search attempt failed")

    # build messages with optional short context and web snippets
    system_prompt = (
        "You are a helpful statistics and market segmentation assistant focused on Bahrain. "
        "If data is available, answer precisely. If no data, indicate limitations and offer sensible reasoning and possible data sources. "
        "When web snippets are provided, use them as source material and cite them inline (title or URL). Do NOT hallucinate facts."
    )
    messages = [{"role": "system", "content": system_prompt}]
    if agent_text and isinstance(agent_text, str) and agent_text.strip():
        messages.append({"role": "system", "content": f"Context from rule-based agent: {agent_text.strip()[:800]}"})
    if web_snips:
        messages.append({"role": "system", "content": f"Web search snippets (top results):\n{web_snips[:4000]}"})
    messages.append({"role": "user", "content": user_text})

    # Try new-style client first
    try:
        if _have_new_client and _new_client_cls is not None:
            client = _new_client_cls(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=800,
                temperature=0.2,
            )
            # parse safely
            choices = getattr(resp, "choices", None) or (resp.get("choices") if isinstance(resp, dict) else None)
            if choices and len(choices) > 0:
                first = choices[0]
                if hasattr(first, "message") and hasattr(first.message, "content"):
                    return first.message.content.strip()
                if isinstance(first, dict):
                    msg = first.get("message", {})
                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"].strip()

        # Next try old-style if present (some environments)
        if _have_old_chat and openai is not None and hasattr(openai, "ChatCompletion"):
            openai.api_key = OPENAI_API_KEY
            resp2 = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=800,
                temperature=0.2,
                request_timeout=OPENAI_TIMEOUT,
            )
            if resp2 and "choices" in resp2 and len(resp2["choices"]) > 0:
                content = resp2["choices"][0].get("message", {}).get("content")
                if content:
                    return content.strip()
    except Exception as e:
        LOG.exception("LLM fallback failed: %s", e)
        return "I tried to ask the language model for help but it failed. Please try again or check server logs."

    # If both attempts didn't return a valid string:
    return "I couldn't generate an answer at this time."
