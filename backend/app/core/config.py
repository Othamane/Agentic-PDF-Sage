# config.py
import json
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic import field_validator  # v2
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_ignore_empty=False,  # keep empty values if present
    )

    # Environment
    ENVIRONMENT: str = Field(default="development", description="Environment")
    DEBUG: bool = Field(default=False, description="Debug mode")

    # Database
    DATABASE_URL: str = Field(..., description="PostgreSQL database URL")
    DATABASE_POOL_SIZE: int = Field(default=10)
    DATABASE_MAX_OVERFLOW: int = Field(default=20)

    # Security & CORS
    SECRET_KEY: str = Field(..., description="Secret key for signing")
    ALLOWED_HOSTS: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    ALLOWED_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"]
    )
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    FRONTEND_HOST: Optional[str] = Field(default=None, description="Frontend base URL")

    # AI / LLM
    HUGGINGFACE_TOKEN: Optional[str] = None
    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    LLM_PROVIDER: str = Field(default="ollama")
    LLM_MODEL: str = Field(default="llama2")
    LLM_TEMPERATURE: float = Field(default=0.1)
    MAX_TOKENS: int = Field(default=2000)

    # Embeddings
    EMBEDDING_PROVIDER: str = Field(default="sentence-transformers")
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    VECTOR_DIMENSION: int = Field(default=384)

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100)
    RATE_LIMIT_WINDOW: int = Field(default=3600)

    # File upload / chunking (these were missing)
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, description="Max upload bytes")
    UPLOAD_DIR: str = Field(default="uploads", description="Upload directory")
    CHUNK_SIZE: int = Field(default=1000, description="Document chunk size")
    CHUNK_OVERLAP: int = Field(default=200, description="Chunk overlap")

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="logs/app.log")

    # -------- Validators (v2) --------
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {sorted(allowed)}")
        return v

    @field_validator("ALLOWED_ORIGINS", "ALLOWED_HOSTS", "CORS_ORIGINS", mode="before")
    @classmethod
    def parse_list_fields(cls, v):
        """
        Accept either:
          - JSON arrays: '["a","b"]'  (preferred; parsed automatically by Pydantic)
          - Comma strings: "a,b"      (fallback)
        """
        if v is None:
            return []
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                # fallback: comma-separated
                return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        valid_prefixes = ("postgresql://", "postgresql+asyncpg://")
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(f"DATABASE_URL must start with one of {valid_prefixes}")
        return v

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
