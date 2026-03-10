
# DubStudio

DubStudio is an AI-powered Video Dubbing application that automates the translation and speech synthesis of English video content into Hindi. It extracts the audio, transcribes it using Whisper, translates the transcript to Hindi using Meta's NLLB model, generates synthesized Hindi speech using MMS-TTS, and effortlessly merges the audio back onto your original video.
DubStudio is an AI-powered Video Dubbing application that automates the translation and speech synthesis of English video content into Hindi. It extracts the audio, transcribes it using Whisper, translates the transcript to Hindi using Meta's NLLB model, generates synthesized Hindi speech using MMS-TTS, and effortlessly merges the audio back onto your original video.

## Features
- **Accurate Transcription**: Powered by OpenAI's Whisper model (`large-v3`).
- **Seamless Translation**: Translates English transcriptions to Hindi using Meta's `nllb-200-distilled-600M` model.
- **Natural Dubbing**: Employs Meta's `mms-tts-hin` model to generate realistic spoken Hindi audio. 
- **AI Hardware Optimization**: Dynamic scaling down to `float16` to comfortably run AI inference pipelines on lower-tier GPUs (4GB VRAM).

## Setup Instructions
- **Accurate Transcription**: Powered by OpenAI's Whisper model (`large-v3`).
- **Seamless Translation**: Translates English transcriptions to Hindi using Meta's `nllb-200-distilled-600M` model.
- **Natural Dubbing**: Employs Meta's `mms-tts-hin` model to generate realistic spoken Hindi audio. 
- **AI Hardware Optimization**: Dynamic scaling down to `float16` to comfortably run AI inference pipelines on lower-tier GPUs (4GB VRAM).

## Setup Instructions

### Prerequisites
- **Python 3.10+**
- **Node.js** (v18+)
- **FFmpeg**: Must be installed and accessible via your system's PATH.

### 1. Backend Setup (FastAPI)
Navigate to the backend directory and set up your Python virtual environment.
- **Python 3.10+**
- **Node.js** (v18+)
- **FFmpeg**: Must be installed and accessible via your system's PATH.

### 1. Backend Setup (FastAPI)
Navigate to the backend directory and set up your Python virtual environment.
```bash
cd backend
python -m venv venv
# Activate the environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
# Activate the environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the root directory (or `backend` directory depending on your setup path) and add your Hugging Face API token if you plan to use gated models:
```env
HF_TOKEN=your_hugging_face_token
DUBBING_TRANSLATION_MODEL=facebook/nllb-200-distilled-600M
```

Run the FastAPI backend server:
Create a `.env` file in the root directory (or `backend` directory depending on your setup path) and add your Hugging Face API token if you plan to use gated models:
```env
HF_TOKEN=your_hugging_face_token
DUBBING_TRANSLATION_MODEL=facebook/nllb-200-distilled-600M
```

Run the FastAPI backend server:
```bash
uvicorn app.main:app --reload --port 8000
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup (Next.js)
Open a new terminal, navigate to the frontend directory, and start the development server.
### 2. Frontend Setup (Next.js)
Open a new terminal, navigate to the frontend directory, and start the development server.
```bash
cd frontend/video-dubbing
npm install
npm run dev
```
The frontend should now be accessible at `http://localhost:3000`.
The frontend should now be accessible at `http://localhost:3000`.

## Architecture & Dependencies
- **Frontend**: Next.js, React, TailwindCSS.
- **Backend API**: FastAPI, Uvicorn, Python 3.10+.
- **AI / Machine Learning**: PyTorch, Transformers, Hugging Face Hub, Librosa, SciPy.
- **Media Processing**: FFmpeg.

## Estimated Cost if Scaled (Per Minute of Video)
If deployed in a cloud environment (e.g., AWS, GCP, RunPod, or Lambda GPU instances):
- **Transcription (Whisper)**: ~$0.005 per minute.
- **Translation (NLLB)**: ~$0.001 per minute.
- **Text-to-Speech (MMS)**: ~$0.01 per minute.
- **Compute (e.g., Nvidia T4 / A10G Instance)**: ~$0.02 - $0.05 per minute of processing time.
- **Total Estimated Cost**: Roughly **$0.03 - $0.07 per minute** of video depending on the target GPU architecture and cloud provider scaling policies.

## Known Limitations
- **Processing Time**: Rendering on CPUs is significantly slower than on dedicated GPUs with CUDA enabled.
- **Hardware Bottlenecks**: Loading multiple Transformer models into memory sequentially requires strict memory management (implemented as `float16` precision) for GPUs with low VRAM (e.g., 4GB).
- **Lip-Sync**: Direct lip-sync orchestration (`Wav2Lip`) is incredibly resource intensive and prone to timeouts on non-CUDA machines. It is currently bypassed in the pipeline to dramatically improve processing time.
- **Translation Fallbacks**: Highly localized slang or heavily accented English might lead to slight misinterpretations during the transcription phase, cascading into translation inaccuracies.

## Future Improvements
If given more time, the following enhancements would be prioritized:
1. **Cloud Offloading (Serverless GPUs)**: Offload the heaviest AI inference tasks (Transcription, TTS) to serverless GPU endpoints (like Replicate or Modal) to decouple hardware requirements from the host machine.
2. **Re-integrate Lip Syncing**: Fix dependencies and performance ceilings to cleanly reintegrate `Wav2Lip` or experiment with newer, faster models like `SadTalker` for realistic facial alignment with the dubbed audio.
3. **Voice Cloning Integration**: Substitute standard TTS with Zero-Shot Voice Cloning (e.g., `XTTSv2` or `ElevenLabs API`) so the generated Hindi payload accurately mimics the original speaker's vocal tone and cadence.
4. **Interactive Subtitles**: Generate and overlay burned-in Hindi subtitles onto the final video using the aligned timestamps.
5. **Streaming/Chunked Processing**: Instead of waiting for the entire video to process before finalizing, pipeline the chunks so audio is dynamically stitched on the fly.
## Architecture & Dependencies
- **Frontend**: Next.js, React, TailwindCSS.
- **Backend API**: FastAPI, Uvicorn, Python 3.10+.
- **AI / Machine Learning**: PyTorch, Transformers, Hugging Face Hub, Librosa, SciPy.
- **Media Processing**: FFmpeg.

## Estimated Cost if Scaled (Per Minute of Video)
If deployed in a cloud environment (e.g., AWS, GCP, RunPod, or Lambda GPU instances):
- **Transcription (Whisper)**: ~$0.005 per minute.
- **Translation (NLLB)**: ~$0.001 per minute.
- **Text-to-Speech (MMS)**: ~$0.01 per minute.
- **Compute (e.g., Nvidia T4 / A10G Instance)**: ~$0.02 - $0.05 per minute of processing time.
- **Total Estimated Cost**: Roughly **$0.03 - $0.07 per minute** of video depending on the target GPU architecture and cloud provider scaling policies.

## Known Limitations
- **Processing Time**: Rendering on CPUs is significantly slower than on dedicated GPUs with CUDA enabled.
- **Hardware Bottlenecks**: Loading multiple Transformer models into memory sequentially requires strict memory management (implemented as `float16` precision) for GPUs with low VRAM (e.g., 4GB).
- **Lip-Sync**: Direct lip-sync orchestration (`Wav2Lip`) is incredibly resource intensive and prone to timeouts on non-CUDA machines. It is currently bypassed in the pipeline to dramatically improve processing time.
- **Translation Fallbacks**: Highly localized slang or heavily accented English might lead to slight misinterpretations during the transcription phase, cascading into translation inaccuracies.

## Future Improvements
If given more time, the following enhancements would be prioritized:
1. **Cloud Offloading (Serverless GPUs)**: Offload the heaviest AI inference tasks (Transcription, TTS) to serverless GPU endpoints (like Replicate or Modal) to decouple hardware requirements from the host machine.
2. **Re-integrate Lip Syncing**: Fix dependencies and performance ceilings to cleanly reintegrate `Wav2Lip` or experiment with newer, faster models like `SadTalker` for realistic facial alignment with the dubbed audio.
3. **Voice Cloning Integration**: Substitute standard TTS with Zero-Shot Voice Cloning (e.g., `XTTSv2` or `ElevenLabs API`) so the generated Hindi payload accurately mimics the original speaker's vocal tone and cadence.
4. **Interactive Subtitles**: Generate and overlay burned-in Hindi subtitles onto the final video using the aligned timestamps.
5. **Streaming/Chunked Processing**: Instead of waiting for the entire video to process before finalizing, pipeline the chunks so audio is dynamically stitched on the fly.
