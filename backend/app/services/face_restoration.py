"""Face restoration service using GFPGAN."""

from pathlib import Path
from typing import Optional

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

_gfpgan_model = None


def _get_device() -> str:
    """Detect available compute device."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_gfpgan():
    """Lazy load GFPGAN model."""
    global _gfpgan_model
    if _gfpgan_model is not None:
        return _gfpgan_model
    try:
        from gfpgan import GFPGANer

        model_path = settings.gfpgan_model_path
        if not model_path:
            # Use default GFPGANv1.4
            import os

            model_path = os.path.join(
                str(settings.models_path / "gfpgan"),
                "GFPGANv1.4.pth",
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if not os.path.exists(model_path):
                import urllib.request

                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
                logger.info("downloading_gfpgan", url=url)
                urllib.request.urlretrieve(url, model_path)

        device = _get_device()
        _gfpgan_model = GFPGANer(
            model_path=model_path,
            upscale=1,  # No upscale, just restoration
            arch="clean",
            channel_multiplier=2,
            device=device,
        )
        return _gfpgan_model
    except ImportError as e:
        logger.warning("gfpgan_not_available", error=str(e))
        return None
    except Exception as e:
        logger.error("gfpgan_load_failed", error=str(e))
        return None


class FaceRestorationService:
    """Enhance face quality using GFPGAN."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize with optional model path."""
        if model_path:
            settings.gfpgan_model_path = model_path

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        fps: Optional[float] = None,
    ) -> Path:
        """
        Apply GFPGAN face restoration to video frames.

        Args:
            input_path: Input video path.
            output_path: Output video path.
            fps: Override FPS (default: from input).

        Returns:
            Path to restored video.
        """
        restorer = _load_gfpgan()
        if restorer is None:
            logger.warning("gfpgan_skipped", reason="model_not_available")
            import shutil

            shutil.copy(input_path, output_path)
            return output_path

        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        vid_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(output_path), fourcc, vid_fps, (width, height))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # GFPGAN expects BGR
                _, _, restored = restorer.enhance(
                    frame, has_aligned=False, only_center_face=False, paste_back=True
                )
                if restored is not None:
                    out.write(restored)
                else:
                    out.write(frame)
                frame_count += 1
        finally:
            cap.release()
            out.release()

        logger.info("face_restoration_complete", frames=frame_count)
        return output_path
