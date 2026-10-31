"""Tests for YouTube ingestion configuration."""
from src.youtube_ingest.config import YouTubeIngestConfig


def test_config_defaults():
    cfg = YouTubeIngestConfig(playlist_url="https://youtube.com/playlist?list=ABC")
    assert cfg.use_captions is True
    assert cfg.force_whisper is False
    assert cfg.whisper_model == "medium"
    assert cfg.max_words == 280
    assert cfg.overlap_words == 40


def test_config_custom_values():
    cfg = YouTubeIngestConfig(
        playlist_url="https://youtube.com/playlist?list=XYZ",
        use_captions=False,
        force_whisper=True,
        whisper_model="large",
        max_words=200,
        overlap_words=30,
    )
    assert cfg.use_captions is False
    assert cfg.force_whisper is True
    assert cfg.whisper_model == "large"
    assert cfg.max_words == 200
    assert cfg.overlap_words == 30