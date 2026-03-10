"""Voice cloning and TTS service - MMS-TTS (Hindi) primary, gTTS fallback."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

_mms_tts_model = None
_mms_tts_tokenizer = None


def _get_device() -> str:
    """Detect available compute device."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_mms_tts():
    """Lazy load facebook/mms-tts-hin for Hindi TTS (transformers-native, no TTS lib)."""
    global _mms_tts_model, _mms_tts_tokenizer
    if _mms_tts_model is not None:
        return _mms_tts_model, _mms_tts_tokenizer
    try:
        from transformers import AutoTokenizer, VitsModel

        model_id = getattr(settings, "tts_model", "facebook/mms-tts-hin") or "facebook/mms-tts-hin"
        cache_dir = str(settings.models_path / "mms_tts")
        import torch
        device = _get_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("loading_mms_tts", model=model_id, device=device)
        _mms_tts_tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, token=token
        )
        _mms_tts_model = VitsModel.from_pretrained(
            model_id, cache_dir=cache_dir, token=token, torch_dtype=dtype
        ).to(device)
        return _mms_tts_model, _mms_tts_tokenizer
    except Exception as e:
        logger.error("mms_tts_load_failed", error=str(e))
        return None, None


class VoiceCloneService:
    """
    Generate Hindi speech from text.
    Uses facebook/mms-tts-hin (transformers); gTTS fallback.
    """

    def __init__(
        self,
        reference_audio: Optional[Path] = None,
        use_voice_cloning: bool = True,
    ):
        """
        Initialize the service.

        Args:
            reference_audio: Unused (kept for API compat; MMS-TTS has no cloning).
            use_voice_cloning: Unused (kept for API compat).
        """
        self.reference_audio = reference_audio
        self.use_voice_cloning = use_voice_cloning

    def synthesize(
        self,
        text: str,
        output_path: Path,
        sample_rate: int = 16000,
    ) -> Path:
        """
        Synthesize Hindi speech from text.

        Args:
            text: Hindi text to speak.
            output_path: Path for output WAV file.
            sample_rate: Output sample rate.

        Returns:
            Path to generated WAV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = (text or "").strip()
        if not text:
            wavfile.write(
                str(output_path),
                sample_rate,
                np.zeros(int(sample_rate * 0.1), dtype=np.int16),
            )
            return output_path

        model, tokenizer = _load_mms_tts()
        if model is not None and tokenizer is not None:
            try:
                import torch

                device = next(model.parameters()).device
                inputs = tokenizer(text, return_tensors="pt").to(device)
                
                # Check if the text translated to any valid tokens (e.g. shape is [1, 0] or empty)
                if inputs.get("input_ids") is None or inputs["input_ids"].shape[1] == 0:
                    raise ValueError("Input resulted in 0 tokens after tokenization")

                with torch.no_grad():
                    output = model(**inputs).waveform
                # output shape: (1, num_samples), float32 in [-1, 1]
                sr = model.config.sampling_rate
                wav = (output.squeeze().cpu().float().numpy() * 32767).astype(np.int16)
                wavfile.write(str(output_path), sr, wav)
                self._resample_if_needed(output_path, sample_rate)
                return output_path
            except Exception as e:
                logger.warning("mms_tts_synthesize_failed", error=str(e))

        # Fallback: gTTS
        self._fallback_tts(text, output_path, sample_rate)
        return output_path

    def _resample_if_needed(self, path: Path, target_sr: int) -> None:
        """Resample WAV to target sample rate if needed."""
        sr, data = wavfile.read(str(path))
        if sr != target_sr:
            import librosa

            data_float = data.astype(np.float32) / 32768.0
            resampled = librosa.resample(
                data_float, orig_sr=sr, target_sr=target_sr
            )
            wavfile.write(
                str(path),
                target_sr,
                (resampled * 32767).astype(np.int16),
            )

    def _fallback_tts(self, text: str, output_path: Path, sample_rate: int) -> None:
        """Fallback using gTTS for Hindi."""
        try:
            from gtts import gTTS

            fd, tmp = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            try:
                gTTS(text=text, lang="hi").save(tmp)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        tmp,
                        "-ar",
                        str(sample_rate),
                        "-ac",
                        "1",
                        str(output_path),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            finally:
                Path(tmp).unlink(missing_ok=True)
        except ImportError:
            raise RuntimeError("Install gtts: pip install gtts") from None
