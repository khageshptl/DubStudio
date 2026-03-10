#!/usr/bin/env python3
"""Download required AI model checkpoints."""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

WAV2LIP_URL = "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?download=1"


def download_wav2lip():
    """Download Wav2Lip checkpoint."""
    checkpoint_dir = Path(__file__).parent.parent / "backend" / "Wav2Lip" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / "wav2lip_gan.pth"
    if target.exists():
        print(f"Wav2Lip checkpoint exists: {target}")
        return
    print("Downloading Wav2Lip checkpoint...")
    try:
        import urllib.request
        urllib.request.urlretrieve(WAV2LIP_URL, target)
        print(f"Saved to {target}")
    except Exception as e:
        print(f"Manual download required. Save wav2lip_gan.pth to {checkpoint_dir}")
        print(f"Get it from: https://github.com/Rudrabha/Wav2Lip")
        raise e


def main():
    download_wav2lip()
    print("Models ready.")


if __name__ == "__main__":
    main()
