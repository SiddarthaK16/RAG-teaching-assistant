"""Tests for Whisper transcription fallback."""
import subprocess
from pathlib import Path
from src.youtube_ingest.transcriber import transcribe_video
from src.youtube_ingest.config import YouTubeIngestConfig


def test_transcribe_video_returns_whisper_format(tmp_path):
    """Test that transcribe_video returns Whisper JSON format."""
    # Create a tiny silent audio file for testing (2 seconds)
    audio = tmp_path / "test.wav"
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "2", str(audio)
    ], check=True, capture_output=True)

    cfg = YouTubeIngestConfig(playlist_url="x", whisper_model="tiny")
    result = transcribe_video(audio, cfg)

    assert "segments" in result
    assert isinstance(result["segments"], list)
    assert "text" in result
    assert "language" in result
    # For silent audio, text should be empty or minimal
    assert isinstance(result["text"], str)


def test_transcribe_video_includes_video_file(tmp_path):
    """Test that result includes video_file field."""
    audio = tmp_path / "test.wav"
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "1", str(audio)
    ], check=True, capture_output=True)

    cfg = YouTubeIngestConfig(playlist_url="x", whisper_model="tiny")
    result = transcribe_video(audio, cfg)

    assert "video_file" in result
    assert result["video_file"] == "test.wav"