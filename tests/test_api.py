"""Tests for FastAPI backend."""
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_youtube_creates_job():
    from uuid import uuid4
    test_job_id = str(uuid4())
    with patch("src.api.jobs.job_manager") as mock_job_manager:
        mock_job_manager.create_job.return_value = test_job_id
        resp = client.post("/api/ingest/youtube", json={"playlist_url": "https://youtube.com/playlist?list=test"})
        assert resp.status_code == 202
        assert "job_id" in resp.json()


def test_job_status_endpoint():
    from uuid import uuid4
    from src.api.models import JobStatus, JobProgress
    test_job_id = str(uuid4())
    with patch("src.api.jobs.job_manager") as mock_job_manager:
        mock_job_manager.get_job.return_value = JobStatus(
            job_id=test_job_id,
            status="running",
            progress=JobProgress(total_videos=3, processed=1, current_video="Video 1", current_step="transcribing")
        )
        resp = client.get(f"/api/jobs/{test_job_id}")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == test_job_id
        assert resp.json()["status"] == "running"


def test_ask_endpoint_returns_answer():
    with patch("answer_question.retrieve_passages") as mock_retrieve, \
         patch("answer_question.generate_answer") as mock_generate:

        mock_retrieve.return_value = []
        mock_generate.return_value = "Test answer"

        resp = client.post("/api/ask", json={"question": "What is deep learning?", "mode": "answer"})
        assert resp.status_code == 200
        assert "answer" in resp.json()


def test_lectures_endpoint_lists_indexed():
    with patch("src.api.routes.lectures.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"lecture_title": "Lecture 1", "video_file": "lec1.mp4"},
                {"lecture_title": "Lecture 2", "video_file": "lec2.mp4"},
            ]
        }
        mock_client.return_value.get_collection.return_value = mock_collection

        resp = client.get("/api/lectures")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)