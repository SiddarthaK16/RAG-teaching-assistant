"""Tests for yt-dlp playlist extractor."""
from unittest.mock import MagicMock, patch
from src.youtube_ingest.extractor import extract_playlist_info, get_video_captions
from src.youtube_ingest.config import YouTubeIngestConfig
from src.youtube_ingest.extractor import VideoInfo, PlaylistInfo


def test_extract_playlist_info_returns_videos():
    cfg = YouTubeIngestConfig(playlist_url="https://youtube.com/playlist?list=PL_test")

    # Mock yt-dlp to return fake playlist
    mock_result = {
        "title": "Test Playlist",
        "entries": [
            {"id": "vid1", "title": "Video 1", "duration": 100},
            {"id": "vid2", "title": "Video 2", "duration": 200},
        ],
    }

    with patch("src.youtube_ingest.extractor.yt_dlp.YoutubeDL") as mock_ydl:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_result
        mock_ydl.return_value.__enter__.return_value = mock_instance

        info = extract_playlist_info(cfg.playlist_url)

    assert isinstance(info, PlaylistInfo)
    assert info.title == "Test Playlist"
    assert len(info.videos) == 2
    assert info.videos[0].url == "https://youtube.com/watch?v=vid1"
    assert info.videos[0].title == "Video 1"
    assert info.videos[0].duration == 100
    assert info.videos[0].video_id == "vid1"
    assert info.videos[1].video_id == "vid2"


def test_extract_playlist_info_skips_none_entries():
    cfg = YouTubeIngestConfig(playlist_url="https://youtube.com/playlist?list=PL_test")

    mock_result = {
        "title": "Test Playlist",
        "entries": [
            {"id": "vid1", "title": "Video 1", "duration": 100},
            None,  # Some entries can be None
            {"id": "vid2", "title": "Video 2", "duration": 200},
        ],
    }

    with patch("src.youtube_ingest.extractor.yt_dlp.YoutubeDL") as mock_ydl:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_result
        mock_ydl.return_value.__enter__.return_value = mock_instance

        info = extract_playlist_info(cfg.playlist_url)

    assert len(info.videos) == 2


def test_get_video_captions_returns_text_when_available():
    mock_result = {
        "subtitles": {
            "en": [
                {"data": "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nHello world\n"}
            ]
        }
    }

    with patch("src.youtube_ingest.extractor.yt_dlp.YoutubeDL") as mock_ydl:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_result
        mock_ydl.return_value.__enter__.return_value = mock_instance

        captions = get_video_captions("https://youtube.com/watch?v=has_captions")

    assert captions is not None
    assert "Hello world" in captions


def test_get_video_captions_returns_none_when_missing():
    mock_result = {"subtitles": {}, "automatic_captions": {}}

    with patch("src.youtube_ingest.extractor.yt_dlp.YoutubeDL") as mock_ydl:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_result
        mock_ydl.return_value.__enter__.return_value = mock_instance

        captions = get_video_captions("https://youtube.com/watch?v=no_captions")

    assert captions is None


def test_get_video_captions_prefers_manual_subtitles():
    """Manual subtitles should be preferred over auto-generated."""
    mock_result = {
        "subtitles": {
            "en": [{"data": "MANUAL SUBTITLES"}]
        },
        "automatic_captions": {
            "en": [{"data": "AUTO CAPTIONS"}]
        }
    }

    with patch("src.youtube_ingest.extractor.yt_dlp.YoutubeDL") as mock_ydl:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_result
        mock_ydl.return_value.__enter__.return_value = mock_instance

        captions = get_video_captions("https://youtube.com/watch?v=both")

    assert captions == "MANUAL SUBTITLES"