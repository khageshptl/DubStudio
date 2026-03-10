"""Video and input validation utilities."""

import subprocess
from pathlib import Path
from typing import Optional

from ..config.settings import settings
from .logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


class VideoValidator:
    """Validates video files for the dubbing pipeline."""

    MAX_SIZE_BYTES = settings.max_file_size_mb * 1024 * 1024

    @classmethod
    def validate_file_exists(cls, path: Path) -> None:
        """Ensure the file exists."""
        if not path.exists():
            raise ValidationError(f"File not found: {path}")
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")

    @classmethod
    def validate_format(cls, path: Path) -> None:
        """Validate video format."""
        suffix = path.suffix.lower().lstrip(".")
        if suffix not in settings.allowed_video_formats:
            raise ValidationError(
                f"Invalid format: {suffix}. Allowed: {settings.allowed_video_formats}"
            )

    @classmethod
    def validate_file_size(cls, path: Path) -> None:
        """Validate file size."""
        size = path.stat().st_size
        if size > cls.MAX_SIZE_BYTES:
            raise ValidationError(
                f"File too large: {size / (1024**2):.1f}MB. Max: {settings.max_file_size_mb}MB"
            )
        if size == 0:
            raise ValidationError("File is empty")

    @classmethod
    def get_video_info(cls, path: Path) -> dict:
        """Get video metadata using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=True
            )
            import json

            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error("ffprobe_failed", path=str(path), stderr=e.stderr)
            raise ValidationError("Failed to read video metadata") from e
        except FileNotFoundError:
            raise ValidationError("ffmpeg/ffprobe not found. Please install ffmpeg.")

    @classmethod
    def validate_video_stream(cls, info: dict) -> None:
        """Validate video stream exists and has valid codec."""
        video_streams = [
            s for s in info.get("streams", []) if s.get("codec_type") == "video"
        ]
        if not video_streams:
            raise ValidationError("No video stream found")

        codec = video_streams[0].get("codec_name", "").lower()
        if codec not in settings.allowed_video_codecs and codec != "unknown":
            logger.warning("unusual_codec", codec=codec, allowed=settings.allowed_video_codecs)

    @classmethod
    def validate_audio_presence(cls, info: dict) -> bool:
        """Check if video has audio stream. Returns True if present."""
        audio_streams = [
            s for s in info.get("streams", []) if s.get("codec_type") == "audio"
        ]
        return len(audio_streams) > 0

    @classmethod
    def get_duration(cls, info: dict) -> float:
        """Get video duration in seconds."""
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video" and "duration" in stream:
                return float(stream["duration"])
        format_info = info.get("format", {})
        return float(format_info.get("duration", 0))

    @classmethod
    def get_fps(cls, info: dict) -> float:
        """Get video FPS."""
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get("r_frame_rate", "25/1")
                if "/" in fps_str:
                    num, den = map(int, fps_str.split("/"))
                    return num / den if den else 25.0
                return float(fps_str)
        return 25.0

    @classmethod
    def validate_duration(cls, duration: float) -> None:
        """Validate video duration."""
        if duration <= 0:
            raise ValidationError("Invalid or zero duration")
        if duration > settings.max_video_duration_seconds:
            raise ValidationError(
                f"Duration {duration:.0f}s exceeds max {settings.max_video_duration_seconds}s"
            )

    @classmethod
    def validate_timestamps(
        cls, start: float, end: float, duration: float
    ) -> None:
        """Validate segment timestamps."""
        if start < 0 or end <= start:
            raise ValidationError(f"Invalid timestamps: start={start}, end={end}")
        if end > duration:
            raise ValidationError(
                f"End time {end}s exceeds video duration {duration}s"
            )
        segment_duration = end - start
        if segment_duration < 0.5:
            raise ValidationError("Segment must be at least 0.5 seconds")

    @classmethod
    def validate_full(cls, path: Path) -> dict:
        """
        Run full validation and return video info.
        Raises ValidationError on failure.
        """
        cls.validate_file_exists(path)
        cls.validate_format(path)
        cls.validate_file_size(path)
        info = cls.get_video_info(path)
        cls.validate_video_stream(info)
        has_audio = cls.validate_audio_presence(info)
        if not has_audio:
            logger.warning("no_audio_stream", path=str(path))
        duration = cls.get_duration(info)
        cls.validate_duration(duration)
        return {
            "info": info,
            "duration": duration,
            "fps": cls.get_fps(info),
            "has_audio": has_audio,
        }
