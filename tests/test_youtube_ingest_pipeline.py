"""Integration tests for YouTube ingestion pipeline."""
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.youtube_ingest.pipeline import run_ingestion
from src.youtube_ingest.config import YouTubeIngestConfig
from src.youtube_ingest.extractor import VideoInfo, PlaylistInfo


def test_run_ingestion_processes_playlist(tmp_path, monkeypatch):
    """Test full pipeline with mocked external calls."""
    cfg = YouTubeIngestConfig(
        playlist_url="https://youtube.com/playlist?list=test",
        videos_dir=tmp_path / "videos",
        transcripts_dir=tmp_path / "transcripts",
        chunks_dir=tmp_path / "chunks",
        store=tmp_path / "chroma_db",
        whisper_model="tiny",
    )

    # Mock playlist with 3 videos: 2 with captions, 1 without
    mock_playlist = PlaylistInfo(
        title="Test Playlist",
        videos=[
            VideoInfo(url="https://youtube.com/watch?v=vid1", title="Video 1", duration=100, video_id="vid1"),
            VideoInfo(url="https://youtube.com/watch?v=vid2", title="Video 2", duration=200, video_id="vid2"),
            VideoInfo(url="https://youtube.com/watch?v=vid3", title="Video 3", duration=150, video_id="vid3"),
        ]
    )

    # Track which video gets which treatment
    caption_results = {
        "https://youtube.com/watch?v=vid1": "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nCaption 1",
        "https://youtube.com/watch?v=vid2": "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nCaption 2",
        "https://youtube.com/watch?v=vid3": None,  # No captions
    }

    # Mock all external dependencies
    with patch("src.youtube_ingest.pipeline.extract_playlist_info") as mock_extract, \
         patch("src.youtube_ingest.pipeline.get_video_captions") as mock_captions, \
         patch("src.youtube_ingest.pipeline.download_video") as mock_download, \
         patch("src.youtube_ingest.pipeline.transcribe_video") as mock_transcribe, \
         patch("src.youtube_ingest.pipeline.convert_captions_to_whisper_format") as mock_convert, \
         patch("chunking.make_chunks") as mock_chunk:

        mock_extract.return_value = mock_playlist
        mock_captions.side_effect = lambda url, lang="en": caption_results.get(url)
        mock_download.return_value = tmp_path / "videos" / "Video 3.mp4"
        mock_transcribe.return_value = {
            "video_file": "Video 3.mp4",
            "segments": [{"start": 0, "end": 10, "text": "Transcribed content"}],
            "text": "Transcribed content",
            "language": "en",
        }
        mock_convert.side_effect = lambda caps, vid, title: {
            "video_file": f"{vid}.mp4",
            "segments": [{"start": 1, "end": 4, "text": caps.split("\n")[-1]}],
            "text": caps.split("\n")[-1],
            "language": "en",
        }
        mock_chunk.return_value = [{"chunk_id": f"vid{i}-0001", "text": f"Chunk {i}"} for i in range(1, 4)]

        result = run_ingestion(cfg)

    assert result.videos_processed == 3
    assert result.captions_used == 2
    assert result.whisper_transcribed == 1
    assert result.chunks_created == 3  # 3 videos = 3 chunks from mock


def test_run_ingestion_handles_empty_playlist(tmp_path):
    """Test pipeline handles empty playlist gracefully."""
    cfg = YouTubeIngestConfig(
        playlist_url="https://youtube.com/playlist?list=empty",
        videos_dir=tmp_path / "videos",
        transcripts_dir=tmp_path / "transcripts",
        chunks_dir=tmp_path / "chunks",
        store=tmp_path / "chroma_db",
    )

    mock_playlist = PlaylistInfo(title="Empty Playlist", videos=[])

    with patch("src.youtube_ingest.pipeline.extract_playlist_info", return_value=mock_playlist):
        result = run_ingestion(cfg)

    assert result.videos_processed == 0
    assert result.captions_used == 0
    assert result.whisper_transcribed == 0
    assert result.chunks_created == 0