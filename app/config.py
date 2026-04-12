from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str = ""
    database_url: str = "postgresql+asyncpg://docagent:docagent@db:5432/docagent"

    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    chunk_size: int = 512
    chunk_overlap: int = 64

    critic_similarity_threshold: float = 0.78
    max_retrieval_attempts: int = 3

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
