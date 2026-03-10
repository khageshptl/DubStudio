"""Audio duration alignment - stretch/compress TTS to match segment length."""

import subprocess
from pathlib import Path

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AudioAligner:
    """Align generated audio duration to match target segment duration."""

    def __init__(self, sample_rate: int = settings.default_sample_rate):
        """Initialize aligner."""
        self.sample_rate = sample_rate

    def align_duration(
        self,
        audio_path: Path,
        output_path: Path,
        target_duration: float,
    ) -> Path:
        """
        Stretch or compress audio to match target duration.

        Uses ffmpeg atempo filter for time scaling.

        Args:
            audio_path: Input audio path.
            output_path: Output path.
            target_duration: Target duration in seconds.

        Returns:
            Path to aligned audio.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get current duration
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
        current_duration = float(result.stdout.strip())

        if current_duration <= 0:
            logger.warning("zero_duration_audio", path=str(audio_path))
            return audio_path

        ratio = target_duration / current_duration

        # atempo accepts 0.5-2.0; chain for larger ratios
        atempo_filters = []
        r = ratio
        while r > 2.0:
            atempo_filters.append("atempo=2.0")
            r /= 2.0
        while r < 0.5:
            atempo_filters.append("atempo=0.5")
            r /= 0.5
        atempo_filters.append(f"atempo={r}")

        filter_str = ",".join(atempo_filters)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-filter:a",
            filter_str,
            "-ar",
            str(self.sample_rate),
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return output_path
