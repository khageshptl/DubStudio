"""FastAPI application entry point."""

from pathlib import Path

# Load .env from project root so HF_TOKEN is available for gated HuggingFace models
_dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _dotenv_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv_path)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .config.settings import settings, ensure_dirs
from .models import init_db
from .utils.logging import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    ensure_dirs()
    init_db()
    yield
    # cleanup if needed


app = FastAPI(
    title="AI Video Dubbing Pipeline",
    description="English to Hindi video dubbing with lip sync",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}
