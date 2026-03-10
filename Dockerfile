# AI Video Dubbing Pipeline - Production Dockerfile
# Supports CUDA for GPU acceleration
# Build from project root: docker build -f Dockerfile .

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Backend
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Wav2Lip
RUN mkdir -p Wav2Lip/checkpoints Wav2Lip/temp

# Create directories for uploads/outputs (mounted at runtime)
RUN mkdir -p /app/uploads /app/outputs /app/temp

EXPOSE 8000

ENV DUBBING_UPLOADS_DIR=/app/uploads
ENV DUBBING_OUTPUTS_DIR=/app/outputs
ENV DUBBING_TEMP_DIR=/app/temp

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
