from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Document Analysis Pipeline"
    environment: str = "local"

settings = Settings()
