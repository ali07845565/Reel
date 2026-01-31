"""
Configuration module for the Viral Reel Generator.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    
    # Storage Paths
    download_path: Path = Field(default=Path("./downloads"), alias="DOWNLOAD_PATH")
    output_path: Path = Field(default=Path("./outputs"), alias="OUTPUT_PATH")
    temp_path: Path = Field(default=Path("./temp"), alias="TEMP_PATH")
    
    # Processing Configuration
    max_concurrent_downloads: int = Field(default=3, alias="MAX_CONCURRENT_DOWNLOADS")
    max_video_duration_minutes: int = Field(default=60, alias="MAX_VIDEO_DURATION_MINUTES")
    default_reel_duration_seconds: int = Field(default=60, alias="DEFAULT_REEL_DURATION_SECONDS")
    
    # FFmpeg Configuration
    ffmpeg_path: str = Field(default="ffmpeg", alias="FFMPEG_PATH")
    ffprobe_path: str = Field(default="ffprobe", alias="FFPROBE_PATH")
    
    # Whisper Configuration
    whisper_model: str = Field(default="large-v3", alias="WHISPER_MODEL")
    
    # LLM Configuration
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def ensure_directories():
    """Ensure all required directories exist."""
    settings = get_settings()
    settings.download_path.mkdir(parents=True, exist_ok=True)
    settings.output_path.mkdir(parents=True, exist_ok=True)
    settings.temp_path.mkdir(parents=True, exist_ok=True)
