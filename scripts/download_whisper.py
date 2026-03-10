#!/usr/bin/env python3
"""Pre-download Whisper model so it works offline. Run when you have internet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.config.settings import settings

def main():
    import whisper
    download_root = str(settings.models_path / "whisper")
    print("Downloading Whisper large-v3 (this may take a few minutes)...")
    whisper.load_model("large-v3", download_root=download_root)
    print("Done. Model cached at:", download_root)

if __name__ == "__main__":
    main()
