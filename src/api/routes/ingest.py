"""Ingestion routes for YouTube playlist processing."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from uuid import UUID
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..models import IngestJobRequest, IngestJobResponse, JobStatus
from .. import jobs
from youtube_ingest.config import YouTubeIngestConfig

router = APIRouter(prefix="/api", tags=["ingest"])


@router.post("/ingest/youtube", response_model=IngestJobResponse, status_code=202)
async def ingest_youtube(request: IngestJobRequest, background_tasks: BackgroundTasks):
    """Start a YouTube playlist ingestion job."""
    config = YouTubeIngestConfig(
        playlist_url=str(request.playlist_url),
        use_captions=request.use_captions,
        force_whisper=request.force_whisper,
        whisper_model=request.whisper_model,
        max_words=request.max_words,
        overlap_words=request.overlap_words,
    )

    job_id = jobs.job_manager.create_job(config)
    background_tasks.add_task(jobs.job_manager.start_job, job_id, config)

    return IngestJobResponse(job_id=job_id, status="pending", message="Job queued")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of an ingestion job."""
    try:
        UUID(job_id)  # Validate UUID format
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    status = jobs.job_manager.get_job(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.websocket("/jobs/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress updates."""
    try:
        UUID(job_id)  # Validate UUID format
    except ValueError:
        await websocket.close(code=4000, reason="Invalid job ID")
        return

    job = jobs.job_manager.get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()
    jobs.job_manager.add_websocket(job_id, websocket)

    try:
        # Send current status immediately
        await websocket.send_json(job.model_dump())
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        jobs.job_manager.remove_websocket(job_id, websocket)