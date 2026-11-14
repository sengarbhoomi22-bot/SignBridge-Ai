import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SignBridge AI Runtime Configuration Settings"""

    PROJECT_NAME: str = "SignBridge AI Backend"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    # CORS Configuration for local frontend environments
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Temporal Preprocessing & Video Ingestion Limits
    MAX_VIDEO_SIZE_MB: int = 25
    MIN_VIDEO_DURATION_SEC: float = 1.0
    MAX_VIDEO_DURATION_SEC: float = 8.0  # Prototype upper bound for phrase signing
    TARGET_SAMPLE_FPS: int = 20  # Balanced temporal sampling rate
    MAX_SEQUENCE_LENGTH: int = 160  # 8.0s @ 20 FPS = 160 frames max

    # Calibrated Confidence & Quality Gating
    CALIBRATED_CONFIDENCE_THRESHOLD: float = 0.50
    MIN_LANDMARK_DETECTION_RATIO: float = 0.60
    HAND_DETECTION_MODE: str = "any_hand"
    TEMPERATURE_SCALING_FACTOR: float = 1.0

    # Model Artifact Configuration
    MODEL_WEIGHTS_PATH: str = "models/weights/baseline_isl.pth"
    DEVICE: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

