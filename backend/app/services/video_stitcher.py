"""Video stitching and merging utilities."""

import subprocess
from pathlib import Path
from typing import Optional

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VideoStitcher:
    """Merge audio and video, align durations, and export final output."""

    def __init__(self):
        """Initialize the stitcher."""
        pass

    def merge_audio_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        replace_audio: bool = True,
    ) -> Path:
        """
        Merge video and audio into final output.

        Args:
            video_path: Video file path.
            audio_path: Audio file path (WAV preferred).
            output_path: Output file path.
            replace_audio: If True, replace video's audio; else add as new track.

        Returns:
            Path to merged file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get video duration to ensure alignment
        probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=10, check=True
        )
        video_duration = float(result.stdout.strip())

        # Extend or trim audio to match video
        temp_audio = output_path.parent / "merged_audio.wav"
        probe_audio = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        ar = subprocess.run(
            probe_audio, capture_output=True, text=True, timeout=10
        )
        audio_duration = float(ar.stdout.strip()) if ar.returncode == 0 else 0

        # Use filter to pad/trim audio to match video
        if abs(audio_duration - video_duration) > 0.1:
            # Pad or trim
            filter_complex = (
                f"[0:a]atrim=0:{video_duration},apad=whole_dur={video_duration}[a]"
                if audio_duration < video_duration
                else f"[0:a]atrim=0:{video_duration}[a]"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-filter_complex",
                filter_complex,
                "-map",
                "[a]",
                "-ar",
                "48000",
                "-ac",
                "1",
                str(temp_audio),
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            audio_input = temp_audio
        else:
            audio_input = audio_path

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_input),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            if temp_audio.exists():
                temp_audio.unlink()
            logger.info("merge_complete", output=str(output_path))
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error("merge_failed", stderr=e.stderr)
            raise

    def segment_video(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> Path:
        """
        Extract a segment from video.

        Args:
            input_path: Input video path.
            output_path: Output segment path.
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            Path to segment.
        """
        duration = end_time - start_time
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            str(input_path),
            "-t",
            str(duration),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "1",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return output_path
