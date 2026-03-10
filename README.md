# AI Video Dubbing Pipeline

Production-ready pipeline that converts English videos to Hindi with accurate translation, voice cloning, and lip synchronization.

## Features

- **Accurate translation**: IndicTrans2 (English → Hindi)
- **Voice cloning**: XTTS for Hindi TTS with optional reference audio
- **Lip sync**: Wav2Lip for precise lip movements
- **Face restoration**: GFPGAN (optional) for enhanced quality
- **Modular design**: Clean architecture, scalable for batch processing

## Architecture

```
Input Video → Extract Audio → Whisper → Translate → TTS → Align → Wav2Lip → GFPGAN → Merge → Output
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Frontend | Next.js (React + Tailwind) |
| Database | SQLite |
| Queue | FastAPI BackgroundTasks |
| Video | ffmpeg |
| Transcription | OpenAI Whisper Large V3 |
| Translation | IndicTrans2 |
| TTS / Voice | XTTS, gTTS fallback |
| Lip Sync | Wav2Lip |
| Face Restoration | GFPGAN |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- ffmpeg
- CUDA (optional, for GPU)
- ~16GB RAM, 8GB+ VRAM recommended for GPU

### 1. Backend Setup

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

Download Wav2Lip checkpoint:

```bash
# Place wav2lip_gan.pth in backend/Wav2Lip/checkpoints/
# Get from: https://github.com/Rudrabha/Wav2Lip
```

### 2. Frontend Setup

```bash
cd frontend/video-dubbing
npm install
npm run dev
```

### 3. Run Backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open UI

Navigate to `http://localhost:3000`.

---

## Docker

```bash
# Build and run (requires NVIDIA Docker for GPU)
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/upload-video` | Upload video + optional timestamps, start processing |
| POST | `/api/v1/process-video` | Restart processing for a job |
| GET | `/api/v1/status/{job_id}` | Get job status and progress |
| GET | `/api/v1/download/{job_id}` | Download dubbed video |

### Upload Example

```bash
curl -X POST http://localhost:8000/api/v1/upload-video \
  -F "file=@video.mp4" \
  -F "start_time=15" \
  -F "end_time=30"
```

---

## Pipeline Steps

1. **Validate** – Check format, codec, duration, audio presence  
2. **Extract audio** – ffmpeg, 16kHz mono WAV  
3. **Segment video** – Extract 00:15–00:30 (or specified range)  
4. **Transcribe** – Whisper Large V3  
5. **Clean transcript** – Merge short segments, filter silence  
6. **Translate** – IndicTrans2 (English → Hindi)  
7. **Generate speech** – XTTS (Hindi TTS, optional voice clone)  
8. **Align audio** – Stretch/compress to match segment length  
9. **Lip sync** – Wav2Lip  
10. **Face restore** – GFPGAN (optional)  
11. **Merge** – Combine video + audio → final MP4  

---

## Scaling to 500+ Hours

For overnight batch processing:

1. **Job queue**: Replace BackgroundTasks with Celery + Redis (or similar)
2. **Workers**: Run multiple workers, each on a GPU
3. **Parallel GPU inference**: Batch transcriptions, translations, TTS
4. **Chunking**: Split long videos into ~30s segments, process in parallel
5. **Memory control**: Unload models between jobs, limit concurrent jobs per GPU

### Example Celery Setup

```python
# Replace BackgroundTasks with Celery
@celery_app.task
def run_dubbing_pipeline(job_id: str):
    ...
```

---

## Cost Estimation (per minute of video)

| Component | GPU (A100) | CPU |
|-----------|------------|-----|
| Whisper | ~0.02 min | ~2 min |
| IndicTrans2 | ~0.01 min | ~0.5 min |
| XTTS | ~0.05 min | ~3 min |
| Wav2Lip | ~0.1 min | ~5 min |
| GFPGAN | ~0.05 min | ~2 min |
| **Total** | **~0.25 min** | **~12 min** |

Rough cloud cost: ~$0.10–0.30 per minute on GPU.

---

## Edge Cases Handled

- Silent segments → Skipped or short TTS
- Multiple speakers → Single-voice output (extend for multi-speaker)
- Background noise → Whisper robust to moderate noise
- FPS mismatch → Preserved via ffmpeg
- Audio/video duration mismatch → Alignment step
- GPU unavailable → Fallback to CPU
- Model init delays → Lazy loading, caching

---

## Folder Structure

```
backend/
  app/
    api/          # FastAPI routes
    config/       # Settings
    models/       # SQLAlchemy models
    pipeline/     # DubbingPipeline
    services/     # audio, transcription, translation, etc.
    utils/        # logging, validation
  Wav2Lip/        # Lip sync model

frontend/
  video-dubbing/  # Next.js app

models/           # Cached AI models
outputs/          # Final dubbed videos
uploads/          # Uploaded videos
scripts/          # Helper scripts
```

---

## License

MIT
