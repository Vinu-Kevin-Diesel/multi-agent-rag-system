"""LLM client construction.

The single place in the codebase that knows *which* provider we are talking to. Everything
upstream sees an OpenAI-compatible client and nothing else — swapping NVIDIA NIM for a local
Ollama or vLLM server is a config change, not a code change.
"""

import logging
from functools import lru_cache
from urllib.parse import urlparse

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Local inference servers ignore the API key, but the OpenAI SDK refuses to construct a
# client without one. This is the placeholder we hand it.
_PLACEHOLDER_KEY = "not-needed"

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal", "ollama", "vllm"}


def _is_local(base_url: str) -> bool:
    return (urlparse(base_url).hostname or "") in _LOCAL_HOSTS


def _build_client(base_url: str, api_key: str | None, label: str) -> AsyncOpenAI:
    if not api_key:
        if not _is_local(base_url):
            # A remote endpoint with no key will fail at request time with an opaque 401.
            # Say so now, while the cause is still legible.
            logger.warning(
                "[%s] no API key configured for remote endpoint %s — requests will likely 401",
                label,
                base_url,
            )
        api_key = _PLACEHOLDER_KEY

    logger.info(
        "[%s] using %s (timeout=%.0fs, retries=%d)",
        label, base_url, settings.llm_timeout_seconds, settings.llm_max_retries,
    )
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


@lru_cache(maxsize=1)
def get_llm_client() -> AsyncOpenAI:
    """Client for the model under test — the one that answers user queries."""
    return _build_client(settings.llm_base_url, settings.llm_api_key, "llm")


@lru_cache(maxsize=1)
def get_judge_client() -> AsyncOpenAI:
    """Client for the evaluation judge. Never serves user traffic.

    Deliberately distinct from get_llm_client(): the system under test will soon be a local
    8B, and scoring a model's answers with that same model measures self-consistency, not
    correctness. The judge stays on a stronger, independent model.
    """
    return _build_client(settings.judge_base_url, settings.judge_api_key, "judge")
