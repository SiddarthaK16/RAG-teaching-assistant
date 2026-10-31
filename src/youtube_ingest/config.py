"""Configuration for YouTube playlist ingestion."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class YouTubeIngestConfig:
    """Configuration for YouTube playlist ingestion pipeline."""

    playlist_url: str
    use_captions: bool = True
    force_whisper: bool = False
    whisper_model: str = "medium"
    max_words: int = 280
    overlap_words: int = 40
    collection: str = "lecture_chunks_bge_small_en_v1_5"
    store: Path = field(default_factory=lambda: Path("chroma_db"))
    videos_dir: Path = field(default_factory=lambda: Path("videos"))
    transcripts_dir: Path = field(default_factory=lambda: Path("transcripts"))
    chunks_dir: Path = field(default_factory=lambda: Path("chunks"))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "playlist_url": self.playlist_url,
            "use_captions": self.use_captions,
            "force_whisper": self.force_whisper,
            "whisper_model": self.whisper_model,
            "max_words": self.max_words,
            "overlap_words": self.overlap_words,
            "collection": self.collection,
            "store": str(self.store),
            "videos_dir": str(self.videos_dir),
            "transcripts_dir": str(self.transcripts_dir),
            "chunks_dir": str(self.chunks_dir),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "YouTubeIngestConfig":
        """Create config from dictionary."""
        path_fields = {"store", "videos_dir", "transcripts_dir", "chunks_dir"}
        processed = {}
        for key, value in data.items():
            if key in path_fields and isinstance(value, str):
                processed[key] = Path(value)
            else:
                processed[key] = value
        return cls(**processed)