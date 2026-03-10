"""
Central dubbing pipeline orchestrator.

Orchestrates: extract -> transcribe -> translate -> TTS -> align -> lip sync -> restore -> merge.
"""

import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from ..config.settings import settings
from ..services.audio_aligner import AudioAligner
from ..services.audio_extractor import AudioExtractor
from ..services.face_restoration import FaceRestorationService
from ..services.lip_sync import LipSyncService
from ..services.transcription import TranscriptionService
from ..services.translation import TranslationService
from ..services.video_stitcher import VideoStitcher
from ..services.voice_clone import VoiceCloneService
from ..utils.logging import get_logger
from ..utils.validation import VideoValidator

logger = get_logger(__name__)


class PipelineError(Exception):
    """Pipeline execution error."""

    pass


class DubbingPipeline:
    """
    Production-ready AI video dubbing pipeline.

    Converts English video to Hindi dubbed video with lip sync.
    """

    def __init__(
        self,
        job_id: Optional[str] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        """
        Initialize pipeline.

        Args:
            job_id: Optional job identifier.
            on_progress: Optional callback (step_name, current, total).
        """
        self.job_id = job_id or str(uuid.uuid4())
        self.on_progress = on_progress or (lambda *a: None)
        self._step_total = 14
        self._current_step = 0

        self.audio_extractor = AudioExtractor()
        self.transcription = TranscriptionService()
        self.translation = TranslationService()
        self.voice_clone = VoiceCloneService()
        self.audio_aligner = AudioAligner()
        self.video_stitcher = VideoStitcher()
        self.lip_sync = LipSyncService()
        self.face_restore = FaceRestorationService()

    def _progress(self, step_name: str) -> None:
        """Report progress."""
        self._current_step += 1
        self.on_progress(step_name, self._current_step, self._step_total)
        logger.info("pipeline_step", step=step_name, n=self._current_step, total=self._step_total)

    def run(
        self,
        video_path: Path,
        output_dir: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        reference_audio_path: Optional[Path] = None,
    ) -> Path:
        """
        Run the full dubbing pipeline.

        Args:
            video_path: Input English video path.
            output_dir: Directory for outputs.
            start_time: Optional segment start (seconds).
            end_time: Optional segment end (seconds).
            reference_audio_path: Optional reference for voice cloning.

        Returns:
            Path to final dubbed video.
        """
        job_dir = output_dir / self.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = job_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Step 1-2: Validate
            self._progress("validate")
            info = VideoValidator.validate_full(video_path)
            duration = info["duration"]
            fps = info["fps"]

            if start_time is None:
                start_time = 0.0
            if end_time is None:
                end_time = duration
            VideoValidator.validate_timestamps(start_time, end_time, duration)

            if not info["has_audio"]:
                raise PipelineError("Video has no audio stream. Cannot transcribe.")

            # Step 3: Extract audio
            self._progress("extract_audio")
            segment_audio = temp_dir / "segment_audio.wav"
            self.audio_extractor.extract_audio(
                video_path, segment_audio, start_time, end_time
            )

            # Step 4: Segment video
            self._progress("segment_video")
            segment_video = temp_dir / "segment_video.mp4"
            self.video_stitcher.segment_video(
                video_path, segment_video, start_time, end_time
            )

            # Step 5: Transcribe
            self._progress("transcribe")
            segments = self.transcription.transcribe(str(segment_audio), language="en")
            if not segments:
                raise PipelineError("No speech detected in segment.")

            # Step 6: Clean transcript
            self._progress("clean_transcript")
            segments = self._clean_transcript(segments)

            # Step 7: Translate
            self._progress("translate")
            segments = self.translation.translate_segments(segments)

            # Step 8-9: Voice clone / Generate Hindi speech
            self._progress("generate_speech")
            voice = VoiceCloneService(reference_audio=reference_audio_path)
            segment_audios: list[Path] = []
            for i, seg in enumerate(segments):
                seg_audio = temp_dir / f"seg_{i}.wav"
                voice.synthesize(seg["hindi_text"], seg_audio)
                segment_audios.append(seg_audio)

            # Step 10: Align audio duration with video
            self._progress("align_audio")
            combined_audio = self._combine_and_align_segments(
                segments, segment_audios, temp_dir
            )

            # Step 11: Lip sync (Skipped by User Request)
            self._progress("lip_sync")
            # lip_sync_out = job_dir / "lipsync_output.mp4"
            # self.lip_sync.process(
            #     segment_video,
            #     combined_audio,
            #     lip_sync_out,
            #     fps=fps,
            # )

            # Step 12: Face restoration (Skipped by User Request)
            self._progress("face_restore")
            # restored = job_dir / "restored.mp4"
            # self.face_restore.process_video(lip_sync_out, restored, fps=fps)

            # Step 13-14: Merge and export
            self._progress("merge_export")
            final_path = job_dir / "final_dubbed.mp4"
            # Since lip sync is skipped, we merge the translated audio directly onto the original segment video!
            self.video_stitcher.merge_audio_video(
                segment_video, combined_audio, final_path
            )

            self._progress("complete")
            logger.info("pipeline_complete", job_id=self.job_id, output=str(final_path))
            return final_path

        except Exception as e:
            logger.error("pipeline_failed", job_id=self.job_id, error=str(e))
            raise PipelineError(str(e)) from e

    def _clean_transcript(self, segments: list[dict]) -> list[dict]:
        """Clean transcript segments (filter silent, merge short)."""
        cleaned = []
        for s in segments:
            text = (s.get("text") or "").strip()
            if text and len(text) > 1:
                cleaned.append(s)
        return cleaned if cleaned else segments

    def _combine_and_align_segments(
        self,
        segments: list[dict],
        segment_audios: list[Path],
        temp_dir: Path,
    ) -> Path:
        """Combine TTS segments and align total duration to sum of segment lengths."""
        import subprocess

        segment_durations = [s["end"] - s["start"] for s in segments]
        total_duration = sum(segment_durations)

        # Align each segment to its target duration, then concat
        aligned = []
        for i, (seg, audio_path) in enumerate(zip(segments, segment_audios)):
            dur = seg["end"] - seg["start"]
            out = temp_dir / f"aligned_{i}.wav"
            self.audio_aligner.align_duration(audio_path, out, dur)
            aligned.append(out)

        # Concat demux
        list_file = temp_dir / "concat_list.txt"
        with open(list_file, "w") as f:
            for p in aligned:
                f.write(f"file '{p.absolute()}'\n")

        combined = temp_dir / "combined_audio.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(combined),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return combined
