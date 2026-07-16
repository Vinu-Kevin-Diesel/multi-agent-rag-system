"""Config resolution and LLM client construction.

The property under test: the model that answers queries and the model that judges those
answers are configured independently. That separation is what makes the RAGAS results
meaningful once queries move to a local 8B — a model grading its own output measures
self-consistency, not correctness.
"""

import pytest

from app import dependencies
from app.config import Settings

# Env vars that would otherwise leak in from the container's .env and skew resolution.
_LLM_ENV = [
    "NVIDIA_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "JUDGE_API_KEY",
    "JUDGE_BASE_URL",
    "JUDGE_MODEL",
]


@pytest.fixture
def clean_env(monkeypatch):
    """Settings() reads the real environment; clear the LLM vars so tests are hermetic."""
    for var in _LLM_ENV:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def clear_client_cache():
    """get_llm_client/get_judge_client are lru_cached; drop the cache around each test."""
    dependencies.get_llm_client.cache_clear()
    dependencies.get_judge_client.cache_clear()
    yield
    dependencies.get_llm_client.cache_clear()
    dependencies.get_judge_client.cache_clear()


def test_nvidia_key_backfills_both_clients(clean_env):
    """An existing single-key .env keeps working: NVIDIA_API_KEY feeds llm and judge."""
    settings = Settings(nvidia_api_key="nvapi-abc", _env_file=None)

    assert settings.llm_api_key == "nvapi-abc"
    assert settings.judge_api_key == "nvapi-abc"


def test_explicit_keys_win_over_the_nvidia_fallback(clean_env):
    """The Day 4 shape: local model under test, judge still on NIM."""
    settings = Settings(
        nvidia_api_key="nvapi-abc",
        llm_api_key="ollama",
        _env_file=None,
    )

    assert settings.llm_api_key == "ollama"
    assert settings.judge_api_key == "nvapi-abc"  # falls back independently


def test_no_keys_at_all_is_valid(clean_env):
    """A fully local run has no API key anywhere. This must not raise.

    nvidia_api_key used to be required, so importing the app without it was impossible.
    """
    settings = Settings(_env_file=None)

    assert settings.llm_api_key is None
    assert settings.judge_api_key is None


def test_llm_and_judge_clients_are_independently_targeted(
    monkeypatch, clean_env, clear_client_cache
):
    """The point of the whole day: local model serves queries, hosted model judges them."""
    monkeypatch.setattr(
        dependencies.settings, "llm_base_url", "http://host.docker.internal:11434/v1"
    )
    monkeypatch.setattr(dependencies.settings, "llm_api_key", None)
    monkeypatch.setattr(
        dependencies.settings, "judge_base_url", "https://integrate.api.nvidia.com/v1"
    )
    monkeypatch.setattr(dependencies.settings, "judge_api_key", "nvapi-abc")

    llm = dependencies.get_llm_client()
    judge = dependencies.get_judge_client()

    assert llm is not judge
    assert "11434" in str(llm.base_url)
    assert "integrate.api.nvidia.com" in str(judge.base_url)

    # The local server ignores the key, but the SDK will not build a client without one.
    assert llm.api_key == dependencies._PLACEHOLDER_KEY
    assert judge.api_key == "nvapi-abc"


def test_client_is_cached(clean_env, clear_client_cache):
    """One client per process — not one per request, as the old module-global did."""
    assert dependencies.get_llm_client() is dependencies.get_llm_client()


def test_clients_bound_by_timeout_and_retries(monkeypatch, clean_env, clear_client_cache):
    """The SDK default is a 600s timeout — one hung call would hold a worker for 10 minutes."""
    monkeypatch.setattr(dependencies.settings, "llm_timeout_seconds", 90.0)
    monkeypatch.setattr(dependencies.settings, "llm_max_retries", 3)

    for client in (dependencies.get_llm_client(), dependencies.get_judge_client()):
        assert client.timeout == 90.0
        assert client.max_retries == 3


def test_missing_key_on_a_remote_endpoint_warns(monkeypatch, clean_env, clear_client_cache, caplog):
    """A keyless remote endpoint 401s at request time. Warn while the cause is still legible."""
    monkeypatch.setattr(dependencies.settings, "llm_base_url", "https://api.example.com/v1")
    monkeypatch.setattr(dependencies.settings, "llm_api_key", None)

    with caplog.at_level("WARNING"):
        dependencies.get_llm_client()

    assert "no API key configured for remote endpoint" in caplog.text


def test_missing_key_on_a_local_endpoint_is_silent(
    monkeypatch, clean_env, clear_client_cache, caplog
):
    """Local servers legitimately have no key. Warning there would be noise every startup."""
    monkeypatch.setattr(dependencies.settings, "llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(dependencies.settings, "llm_api_key", None)

    with caplog.at_level("WARNING"):
        dependencies.get_llm_client()

    assert "no API key configured" not in caplog.text
