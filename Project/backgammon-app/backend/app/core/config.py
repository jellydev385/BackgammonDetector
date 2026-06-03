from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    MAX_VIDEO_SIZE_MB: int = 500
    UPLOAD_DIR: str = "/tmp/backgammon_uploads"

    class Config:
        env_file = ".env"


settings = Settings()
