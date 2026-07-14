from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── The model under test ───────────────────────────────────────────────
    # Serves user queries. Any OpenAI-compatible endpoint: NVIDIA NIM today, a local
    # Ollama/vLLM server tomorrow. Nothing outside dependencies.py knows which.
    llm_model: str = "deepseek-ai/deepseek-v4-flash"
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_api_key: str | None = None

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
