"""Pydantic models for FastAPI request/response validation."""
from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import UUID


class IngestJobRequest(BaseModel):
    """Request to start a YouTube playlist ingestion job."""
    playlist_url: str = Field(..., description="YouTube playlist URL")
    use_captions: bool = Field(default=True, description="Prefer YouTube captions over Whisper")
    force_whisper: bool = Field(default=False, description="Force Whisper transcription even if captions exist")
    whisper_model: str = Field(default="medium", description="Whisper model size (tiny/base/small/medium/large)")
    max_words: int = Field(default=280, description="Maximum words per chunk")
    overlap_words: int = Field(default=40, description="Word overlap between chunks")


class JobProgress(BaseModel):
    """Progress information for an ingestion job."""
    total_videos: int = 0
    processed: int = 0
    current_video: Optional[str] = None
    current_step: Optional[str] = None  # transcribing|downloading|chunking|embedding


class JobResults(BaseModel):
    """Results of a completed ingestion job."""
    videos_processed: int = 0
    captions_used: int = 0
    whisper_transcribed: int = 0
    chunks_created: int = 0


class JobStatus(BaseModel):
    """Status of an ingestion job."""
    job_id: UUID
    status: str = Field(..., description="pending|running|completed|failed")
    progress: JobProgress = Field(default_factory=JobProgress)
    results: Optional[JobResults] = None
    error: Optional[str] = None


class IngestJobResponse(BaseModel):
    """Response when creating an ingestion job."""
    job_id: UUID
    status: str = "pending"
    message: str = "Job queued"


class SourceCitation(BaseModel):
    """Citation for a source passage in an answer."""
    lecture_title: str
    start_time: float
    end_time: float
    text: str


class AskRequest(BaseModel):
    """Request to ask a question."""
    question: str = Field(..., description="Question or topic")
    mode: str = Field(default="answer", description="answer|explain|summarize|quiz")
    top_k: int = Field(default=8, description="Number of passages to retrieve")
    quiz_count: int = Field(default=5, description="Number of quiz questions (quiz mode only)")


class AskResponse(BaseModel):
    """Response from asking a question."""
    answer: str
    mode: str
    sources: List[SourceCitation] = Field(default_factory=list)


class LectureListItem(BaseModel):
    """Summary of a lecture in the index."""
    lecture_title: str
    video_file: str
    chunk_count: int


class LectureDetail(BaseModel):
    """Detailed lecture information with chunks."""
    lecture_title: str
    video_file: str
    chunks: List[SourceCitation]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    ollama_connected: bool = False
    chroma_collections: int = 0