"""Application configuration via Pydantic Settings."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Upstash Vector
    upstash_vector_url: str = Field(default="")
    upstash_vector_token: str = Field(default="")

    # Upstash QStash
    qstash_token: str = Field(default="")
    qstash_current_signing_key: str = Field(default="")
    qstash_next_signing_key: str = Field(default="")

    # Public URL for QStash callbacks
    app_public_url: str = Field(default="http://localhost:8000")

    # LLM
    groq_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    llm_model: str = Field(default="llama-3.3-70b-versatile")
    llm_provider: str = Field(default="groq")  # "groq" or "openai"

    # Langfuse
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # GitHub
    github_token: str = Field(default="")

    # App
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # Ingestion limits (free tier safety)
    max_files_per_repo: int = Field(default=500)
    max_file_size_bytes: int = Field(default=102_400)  # 100 KB

    # Query
    default_top_k: int = Field(default=8)

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/codebase_assistant.db"
    )

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_langfuse(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def has_qstash(self) -> bool:
        return bool(self.qstash_token)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
