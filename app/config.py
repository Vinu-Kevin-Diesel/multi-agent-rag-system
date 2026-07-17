from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── The model under test ───────────────────────────────────────────────
    # Serves user queries. Any OpenAI-compatible endpoint: NVIDIA NIM today, a local
    # Ollama/vLLM server tomorrow. Nothing outside dependencies.py knows which.
    llm_model: str = "deepseek-ai/deepseek-v4-flash"
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_api_key: str | None = None

    # Model for the router/decompose classification calls. Defaults to the main model, which
    # is correct for a non-reasoning hosted model. For local qwen3, point this at the
    # thinking-disabled variant (ROUTER_MODEL=qwen3-router, built by scripts/build-router-model.ps1):
    # reasoning turns a one-word classification into a 17-70s, non-deterministic generation.
    router_model: str = ""

    @property
    def effective_router_model(self) -> str:
        return self.router_model or self.llm_model

    # ── The evaluation judge ───────────────────────────────────────────────
    # Used only by the RAGAS harness; never serves user traffic. Kept deliberately
    # separate from the model under test: once queries run on a local 8B, scoring its
    # output with itself yields noise, not data. The judge stays on a stronger model.
    judge_model: str = "deepseek-ai/deepseek-v4-flash"
    judge_base_url: str = "https://integrate.api.nvidia.com/v1"
    judge_api_key: str | None = None

    # Compatibility shim: the original single-provider key. Still honoured as the
    # fallback for both keys above, so existing .env files keep working unchanged.
    nvidia_api_key: str | None = None

    # Per-request ceiling for an LLM call. The OpenAI SDK defaults to 600s, which lets one
    # hung request tie up a worker for ten minutes. Measured worst case against a local 8B
    # is ~45s for a 10k-token prompt, so this leaves generous headroom while still failing
    # in a bounded time.
    llm_timeout_seconds: float = 120.0
    # Retries apply to connection errors, 429s and 5xx — not to a successful response. The
    # free hosted tier rate-limits under load, where this earns its keep.
    llm_max_retries: int = 2

    database_url: str = "postgresql+asyncpg://docagent:docagent@db:5432/docagent"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    chunk_size: int = 512
    chunk_overlap: int = 64

    # How `unstructured` parses a document. OCR happens inside the library, not as a
    # separate stage of ours:
    #   auto     — extract embedded text where present, fall back to OCR where it isn't
    #   fast     — text extraction only; no OCR (scanned pages come back empty)
    #   hi_res   — layout model + OCR; slowest, best on scanned/complex PDFs
    #   ocr_only — force OCR over every page, ignoring any embedded text
    ingestion_strategy: str = "auto"

    critic_similarity_threshold: float = 0.78
    max_retrieval_attempts: int = 3

    # Token budget for the source chunks handed to an agent. A safety rail, not an
    # optimisation: sized to sit well clear of a 16k-context local model once the system
    # prompt, question and generated answer are accounted for. The multi-hop path can
    # retrieve 12+ chunks (per_sub x sub-questions), and top_k is caller-supplied, so
    # without a cap the prompt size is effectively unbounded — and a model that silently
    # truncates drops the source chunks and then invents an answer.
    max_context_tokens: int = 8000

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "extra": "ignore"}

    @model_validator(mode="after")
    def _default_api_keys_to_nvidia(self) -> "Settings":
        """Fall back to NVIDIA_API_KEY when a specific key isn't set.

        Both may end up None — that is the valid local case (Ollama and vLLM ignore the
        key entirely). dependencies.py handles the empty-key case; config does not guess.
        """
        if self.llm_api_key is None:
            self.llm_api_key = self.nvidia_api_key
        if self.judge_api_key is None:
            self.judge_api_key = self.nvidia_api_key
        return self


settings = Settings()
