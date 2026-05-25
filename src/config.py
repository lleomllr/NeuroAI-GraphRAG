from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    QDRANT_URL: str = Field(default="http://localhost:6333", env="QDRANT_URL")

    ARXIV_MAX_RES: int = Field(default=50)

    LLM_API_BASE: str = Field(default="http://localhost:11434/v1")
    LLM_API_KEY: str = Field(default="ollama")
    LLM_MODEL: str = Field(default="llama3")

    EXTRACTION_BACKEND: str = Field(default="ollama")

    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

config = Config()