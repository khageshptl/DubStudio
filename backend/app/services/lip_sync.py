"""Lip sync service using Wav2Lip."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LipSyncService:
    """Apply lip sync using Wav2Lip on video + audio."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        wav2lip_dir: Optional[Path] = None,
    ):
        """Initialize with paths."""
        self.wav2lip_dir = wav2lip_dir or settings.wav2lip_dir
        checkpoint = checkpoint_path or (self.wav2lip_dir / settings.wav2lip_checkpoint)
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = self.wav2lip_dir / checkpoint

    def process(
        self,
        face_video_path: Path,
        audio_path: Path,
        output_path: Path,
        fps: float = 25.0,
        resize_factor: int = 1,
        pads: tuple[int, int, int, int] = (0, 10, 0, 0),
    ) -> Path:
        """
        Run Wav2Lip to lip-sync face video with audio.

        Args:
            face_video_path: Video containing face.
            audio_path: Audio to lip-sync to (WAV preferred).
            output_path: Output video path.
            fps: Video FPS.
            resize_factor: Resolution divisor.
            pads: Face padding (top, bottom, left, right).

        Returns:
            Path to output video.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        (self.wav2lip_dir / "temp").mkdir(exist_ok=True)

        # Ensure audio is WAV for Wav2Lip
        audio_input = audio_path
        if audio_path.suffix.lower() != ".wav":
            wav_temp = output_path.parent / "temp_audio.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(wav_temp),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            audio_input = wav_temp

        # Use Wav2Lip inference via subprocess (avoids import path issues)
        inference_script = self.wav2lip_dir / "inference.py"
        if not inference_script.exists():
            raise FileNotFoundError(
                f"Wav2Lip inference.py not found at {inference_script}"
            )

        cmd = [
            sys.executable,
            str(inference_script),
            "--checkpoint_path",
            str(self.checkpoint_path),
            "--face",
            str(face_video_path),
            "--audio",
            str(audio_input),
            "--outfile",
            str(output_path),
            "--fps",
            str(fps),
            "--resize_factor",
            str(resize_factor),
            "--pads",
            str(pads[0]),
            str(pads[1]),
            str(pads[2]),
            str(pads[3]),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.wav2lip_dir),
                capture_output=True,
                text=True,
                timeout=600,  # 10 min
            )
            if result.returncode != 0:
                logger.error(
                    "wav2lip_failed",
                    stderr=result.stderr,
                    stdout=result.stdout,
                )
                raise RuntimeError(f"Wav2Lip failed: {result.stderr}")
            logger.info("lip_sync_complete", output=str(output_path))
            return output_path
        except subprocess.TimeoutExpired:
            raise RuntimeError("Wav2Lip timed out after 10 minutes") from None
