"""Transcription service using OpenAI Whisper."""

from pathlib import Path
from typing import Any, Optional

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Lazy load to handle GPU unavailable
_whisper_model = None


def _get_device() -> str:
    """Detect available compute device."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_whisper_model():
    """Lazy load Whisper model with fallback on network errors."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    import whisper

    device = _get_device()
    download_root = str(settings.models_path / "whisper")
    # Fallback order if primary fails (e.g. getaddrinfo/network error)
    models_to_try = [settings.whisper_model, "base", "tiny"]

    for model_name in models_to_try:
        try:
            logger.info("loading_whisper", model=model_name, device=device)
            _whisper_model = whisper.load_model(
                model_name,
                device=device,
                download_root=download_root,
            )
            if model_name != settings.whisper_model:
                logger.warning("using_fallback_model", requested=settings.whisper_model, loaded=model_name)
            return _whisper_model
        except OSError as e:
            if "getaddrinfo failed" in str(e) or "11001" in str(e):
                logger.warning("whisper_network_error", model=model_name, error=str(e))
            else:
                raise
        except Exception as e:
            logger.warning("whisper_load_failed", model=model_name, error=str(e))

    raise RuntimeError(
        "Could not load any Whisper model. Check internet connection and run "
        "'python -c \"import whisper; whisper.load_model(\\\"base\\\")\"' to pre-download when online."
    )


class TranscriptionService:
    """Transcribe audio using Whisper."""

    def __init__(self, model_id: Optional[str] = None):
        """Initialize with optional model override."""
        self.model_id = model_id or settings.whisper_model

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = "en",
        initial_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Transcribe audio and return segments with timestamps.

        Returns:
            List of dicts with keys: start, end, text
        """
        model = _load_whisper_model()
        try:
            result = model.transcribe(
                str(audio_path),
                language=language,
                initial_prompt=initial_prompt or "",
                word_timestamps=False,
                verbose=False,
            )
        except Exception as e:
            logger.error("transcription_failed", path=str(audio_path), error=str(e))
            raise

        segments = []
        for seg in result.get("segments", []):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = (seg.get("text") or "").strip()
            if text:
                segments.append({"start": start, "end": end, "text": text})

        # Merge very short segments, filter silent
        segments = self._clean_segments(segments)
        logger.info("transcription_complete", segments=len(segments))
        return segments

    def _clean_segments(self, segments: list[dict]) -> list[dict]:
        """Clean and merge short segments."""
        if not segments:
            return []
        merged = []
        buf = list(segments[0].values())
        for s in segments[1:]:
            if s["end"] - buf[0] < 2.0 and buf[2]:  # Merge if < 2s apart
                buf[1] = s["end"]
                buf[2] = (buf[2] + " " + s["text"]).strip()
            else:
                merged.append({"start": buf[0], "end": buf[1], "text": buf[2]})
                buf = [s["start"], s["end"], s["text"]]
        merged.append({"start": buf[0], "end": buf[1], "text": buf[2]})
        return [m for m in merged if m["text"]]
