from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    APP_NAME: str = "SLM Data Validation Service"
    APP_VERSION: str = "1.0.0"

    # URL for the downstream data generation service
    DATA_GENERATION_SERVICE_URL: str = "http://localhost:8001/api/v1/generate-qa/?llm_provider=groq"
    
    # Timeout in seconds when calling the generation service
    DATA_GENERATION_SERVICE_TIMEOUT: int = 600 # 10 minutes

    # Allowed file extensions for upload
    ALLOWED_EXTENSIONS: list[str] = ["csv", "json", "pdf", "docx"]
    
    # For structured files, these columns are mandatory
    EXPECTED_COLUMNS: list[str] = ["question", "answer"]

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

@lru_cache()
def get_settings() -> Settings:
    """Returns a cached instance of the Settings object."""
    return Settings()
