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

    critic_similarity_threshold: float = 0.78
    max_retrieval_attempts: int = 3

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
