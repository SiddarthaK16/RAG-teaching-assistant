"""Background job manager with WebSocket support for ingestion jobs."""
import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
from datetime import datetime

from .models import JobStatus, JobProgress, JobResults


@dataclass
class IngestionJob:
    """Internal representation of an ingestion job."""
    job_id: str
    status: str = "pending"
    progress: JobProgress = field(default_factory=JobProgress)
    results: Optional[JobResults] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    websockets: Set = field(default_factory=set)
    _future: Optional[asyncio.Future] = None


class JobManager:
    """Manages background ingestion jobs."""

    def __init__(self, max_workers: int = 2):
        self.jobs: Dict[str, IngestionJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for WebSocket notifications."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
        return self._loop

    def create_job(self, config) -> str:
        """Create a new ingestion job and return its ID."""
        job_id = str(uuid.uuid4())
        job = IngestionJob(job_id=job_id)
        self.jobs[job_id] = job
        return job_id

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Get job status by ID."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        return JobStatus(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            results=job.results,
            error=job.error,
        )

    def start_job(self, job_id: str, config):
        """Start the ingestion job in background."""
        job = self.jobs.get(job_id)
        if not job:
            return

        job.status = "running"
        job.updated_at = datetime.now()

        # Run in thread pool
        def run_ingestion():
            try:
                # Import here to avoid circular imports
                from youtube_ingest.pipeline import run_ingestion
                result = run_ingestion(config)
                job.status = "completed"
                job.results = JobResults(
                    videos_processed=result.videos_processed,
                    captions_used=result.captions_used,
                    whisper_transcribed=result.whisper_transcribed,
                    chunks_created=result.chunks_created,
                )
                job.error = None
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
            finally:
                job.updated_at = datetime.now()
                # Notify WebSocket clients
                self._notify_websockets(job)

        self.executor.submit(run_ingestion)

    def add_websocket(self, job_id: str, websocket):
        """Register a WebSocket for job updates."""
        job = self.jobs.get(job_id)
        if job:
            job.websockets.add(websocket)

    def remove_websocket(self, job_id: str, websocket):
        """Unregister a WebSocket."""
        job = self.jobs.get(job_id)
        if job:
            job.websockets.discard(websocket)

    def _notify_websockets(self, job: IngestionJob):
        """Notify all WebSocket clients of job update."""
        # This will be called from a thread, so we need to schedule on the event loop
        loop = self._get_loop()
        status = JobStatus(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            results=job.results,
            error=job.error,
        )
        for ws in job.websockets:
            try:
                asyncio.run_coroutine_threadsafe(ws.send_json(status.model_dump()), loop)
            except Exception:
                pass  # Ignore errors sending to closed connections


# Global job manager instance
job_manager = JobManager()