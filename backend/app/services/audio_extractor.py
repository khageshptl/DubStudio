"""Audio extraction service using ffmpeg."""

import subprocess
from pathlib import Path
from typing import Optional

from ..config.settings import settings
from ..utils.logging import get_logger
from ..utils.validation import ValidationError

logger = get_logger(__name__)


class AudioExtractor:
    """Extract and segment audio from video files."""

    def __init__(self, sample_rate: int = settings.default_sample_rate):
        """Initialize the extractor."""
        self.sample_rate = sample_rate

    def extract_audio(
        self,
        video_path: Path,
        output_path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Path:
        """
        Extract audio from video, optionally for a specific segment.

        Args:
            video_path: Path to input video.
            output_path: Path for output WAV file.
            start_time: Optional start time in seconds.
            end_time: Optional end time in seconds.

        Returns:
            Path to extracted WAV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
        ]
        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            cmd.extend(["-ss", str(start_time), "-t", str(duration)])
        cmd.append(str(output_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            if not output_path.exists():
                raise ValidationError("Audio extraction produced no output")
            logger.info(
                "audio_extracted",
                video=str(video_path),
                output=str(output_path),
                segment=f"{start_time}-{end_time}" if start_time is not None else "full",
            )
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(
                "audio_extraction_failed",
                video=str(video_path),
                stderr=e.stderr,
            )
            raise ValidationError(f"Audio extraction failed: {e.stderr}") from e

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds."""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=True
        )
        return float(result.stdout.strip())
