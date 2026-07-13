from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    nvidia_api_key: str
    llm_model: str = "moonshotai/kimi-k2.5"
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
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


settings = Settings()
