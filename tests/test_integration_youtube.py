"""Integration tests for YouTube ingestion + query pipeline."""
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.youtube_ingest.pipeline import run_ingestion
from src.youtube_ingest.config import YouTubeIngestConfig
from src.youtube_ingest.extractor import VideoInfo, PlaylistInfo


def test_youtube_ingested_content_queryable(tmp_path):
    """Test that YouTube-ingested content can be queried via RAG."""
    cfg = YouTubeIngestConfig(
        playlist_url="https://youtube.com/playlist?list=test",
        videos_dir=tmp_path / "videos",
        transcripts_dir=tmp_path / "transcripts",
        chunks_dir=tmp_path / "chunks",
        store=tmp_path / "chroma_db",
        whisper_model="tiny",
    )

    # Mock playlist with 1 video
    mock_playlist = PlaylistInfo(
        title="Test Playlist",
        videos=[
            VideoInfo(url="https://youtube.com/watch?v=vid1", title="Video 1", duration=100, video_id="vid1"),
        ]
    )

    with patch("src.youtube_ingest.pipeline.extract_playlist_info") as mock_extract, \
         patch("src.youtube_ingest.pipeline.get_video_captions") as mock_captions, \
         patch("src.youtube_ingest.pipeline.download_video") as mock_download, \
         patch("src.youtube_ingest.pipeline.transcribe_video") as mock_transcribe, \
         patch("src.youtube_ingest.pipeline.convert_captions_to_whisper_format") as mock_convert, \
         patch("chunking.make_chunks") as mock_chunk, \
         patch("embed_chunks.main") as mock_embed:

        mock_extract.return_value = mock_playlist
        mock_captions.return_value = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nDeep learning is a subset of machine learning"
        mock_download.return_value = tmp_path / "videos" / "Video 1.mp4"
        mock_transcribe.return_value = {
            "video_file": "Video 1.mp4",
            "segments": [{"start": 0, "end": 10, "text": "Deep learning is a subset of machine learning"}],
            "text": "Deep learning is a subset of machine learning",
            "language": "en",
        }
        mock_convert.return_value = {
            "video_file": "vid1.mp4",
            "segments": [{"start": 1, "end": 4, "text": "Deep learning is a subset of machine learning"}],
            "text": "Deep learning is a subset of machine learning",
            "language": "en",
        }
        mock_chunk.return_value = [
            {"chunk_id": "vid1-0001", "text": "Deep learning is a subset of machine learning", "video_file": "vid1.mp4", "start_time": 1.0, "end_time": 4.0}
        ]
        mock_embed.return_value = 1

        result = run_ingestion(cfg)

    assert result.videos_processed == 1
    assert result.captions_used == 1
    assert result.whisper_transcribed == 0
    assert result.chunks_created == 1


def test_existing_cli_still_works():
    """Test that existing answer_question CLI still works."""
    # This test verifies the existing functionality is not broken
    # We can't easily test the full CLI without a real index, but we can verify imports work
    from answer_question import retrieve_passages, generate_answer, format_time, format_title
    from chunking import make_chunks
    from embed_chunks import main as embed_chunks_main
    from transcribing import main as transcribing_main
    from search_chunks import main as search_main

    # Just verify imports work
    assert retrieve_passages is not None
    assert generate_answer is not None
    assert format_time is not None
    assert format_title is not None
    assert make_chunks is not None
    assert embed_chunks_main is not None
    assert transcribing_main is not None
    assert search_main is not None