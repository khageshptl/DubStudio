"""Application configuration and environment settings."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


def _default_base() -> Path:
    return Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths (override via DUBBING_* env vars)
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    uploads_dir: Optional[str] = None
    outputs_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    models_dir: Optional[str] = None

    @property
    def uploads_path(self) -> Path:
        return Path(self.uploads_dir) if self.uploads_dir else self.base_dir / "uploads"

    @property
    def outputs_path(self) -> Path:
        return Path(self.outputs_dir) if self.outputs_dir else self.base_dir / "outputs"

    @property
    def temp_path(self) -> Path:
        return Path(self.temp_dir) if self.temp_dir else self.base_dir / "temp"

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir) if self.models_dir else self.base_dir.parent / "models"

    # Wav2Lip
    @property
    def wav2lip_dir(self) -> Path:
        return self.base_dir / "Wav2Lip"
    wav2lip_checkpoint: str = "checkpoints/wav2lip_gan.pth"

    # GFPGAN
    gfpgan_model_path: Optional[str] = None  # Auto-download if not set

    # Model IDs (HuggingFace)
    whisper_model: str = "large-v3"
    # Swap default to NLLB as it is native to Transformers and requires no external C++ processors.
    translation_model: str = "facebook/nllb-200-distilled-600M"
    hf_token: Optional[str] = None  # For gated models; or set HF_TOKEN env
    tts_model: str = "facebook/mms-tts-hin"

    # Validation
    max_video_duration_seconds: int = 600  # 10 minutes default
    max_file_size_mb: int = 500
    allowed_video_formats: tuple = ("mp4", "avi", "mov", "mkv", "webm")
    allowed_video_codecs: tuple = ("h264", "hevc", "vp9", "mpeg4", "avc1")

    # Processing
    default_sample_rate: int = 16000
    wav2lip_fps: float = 25.0
    gpu_batch_size: int = 16
    cpu_batch_size: int = 4

    # Database
    database_url: str = "sqlite:///./video_dubbing.db"

    # API
    api_prefix: str = "/api/v1"

    class Config:
        """Pydantic config."""

        env_prefix = "DUBBING_"
        env_file = ".env"
        extra = "ignore"


def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    s = Settings()
    for dir_path in [
        s.uploads_path,
        s.outputs_path,
        s.temp_path,
        s.models_path,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


settings = Settings()
