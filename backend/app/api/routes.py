"""FastAPI route definitions."""

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config.settings import settings
from ..models.job import Job, JobStatus, SessionLocal, init_db
from ..pipeline.dubbing_pipeline import DubbingPipeline, PipelineError
from ..utils.logging import get_logger
from ..utils.validation import ValidationError, VideoValidator

logger = get_logger(__name__)

router = APIRouter(prefix=settings.api_prefix, tags=["dubbing"])


def _run_pipeline(job_id: str) -> None:
    """Background task to run the dubbing pipeline."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job or job.status != JobStatus.PENDING.value:
            return

        job.status = JobStatus.PROCESSING.value
        db.commit()

        video_path = settings.uploads_path / job_id / job.video_filename
        output_dir = settings.outputs_path

        def on_progress(step: str, current: int, total: int) -> None:
            job.current_step = step
            job.progress_percent = (current / total) * 100
            db.merge(job)
            db.commit()

        pipeline = DubbingPipeline(job_id=job_id, on_progress=on_progress)
        final_path = pipeline.run(
            video_path=video_path,
            output_dir=output_dir,
            start_time=job.start_time,
            end_time=job.end_time,
        )
        job.status = JobStatus.COMPLETED.value
        job.output_path = str(final_path)
        job.progress_percent = 100
        job.current_step = "complete"
        db.commit()
    except (PipelineError, ValidationError) as e:
        job.status = JobStatus.FAILED.value
        job.error_message = str(e)
        db.commit()
        logger.error("pipeline_job_failed", job_id=job_id, error=str(e))
    except Exception as e:
        job.status = JobStatus.FAILED.value
        job.error_message = str(e)
        db.commit()
        logger.exception("pipeline_job_error", job_id=job_id)
    finally:
        db.close()


@router.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    start_time: Optional[float] = Form(None),
    end_time: Optional[float] = Form(None),
):
    """
    Upload video and optionally start processing.

    Returns job_id for status polling.
    """
    init_db()
    if not file.filename:
        raise HTTPException(400, "No filename")
    if not any(
        file.filename.lower().endswith(f".{fmt}")
        for fmt in settings.allowed_video_formats
    ):
        raise HTTPException(
            400,
            f"Invalid format. Allowed: {settings.allowed_video_formats}",
        )

    import uuid

    job_id = str(uuid.uuid4())
    job_dir = settings.uploads_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / file.filename

    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}") from e

    # Validate
    try:
        VideoValidator.validate_full(dest)
    except ValidationError as e:
        dest.unlink(missing_ok=True)
        job_dir.rmdir()
        raise HTTPException(400, str(e)) from e

    info = VideoValidator.get_video_info(dest)
    duration = VideoValidator.get_duration(info)
    if start_time is not None and end_time is not None:
        VideoValidator.validate_timestamps(start_time, end_time, duration)

    db = SessionLocal()
    try:
        job = Job(
            id=job_id,
            status=JobStatus.PENDING.value,
            video_filename=file.filename,
            start_time=start_time,
            end_time=end_time,
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    background_tasks.add_task(_run_pipeline, job_id)
    return {"job_id": job_id, "message": "Video uploaded. Processing started."}


@router.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
):
    """
    Start or restart processing for an uploaded video.
    """
    init_db()
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        video_path = settings.uploads_path / job_id / job.video_filename
        if not video_path.exists():
            raise HTTPException(404, "Video file not found")
        if job.status == JobStatus.PROCESSING.value:
            return {"job_id": job_id, "message": "Already processing"}
        job.status = JobStatus.PENDING.value
        job.error_message = None
        db.commit()
    finally:
        db.close()
    background_tasks.add_task(_run_pipeline, job_id)
    return {"job_id": job_id, "message": "Processing started"}


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status and progress."""
    init_db()
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        return job.to_dict()
    finally:
        db.close()


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download the dubbed video."""
    init_db()
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status != JobStatus.COMPLETED.value:
            raise HTTPException(400, "Job not completed")
        path = Path(job.output_path)
        if not path.exists():
            raise HTTPException(404, "Output file not found")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"dubbed_{job.video_filename}",
        )
    finally:
        db.close()
