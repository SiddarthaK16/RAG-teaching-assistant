"""Tests for caption (VTT/SRT) to Whisper JSON format converter."""
from src.youtube_ingest.caption_converter import convert_captions_to_whisper_format


def test_convert_vtt_to_whisper_format():
    vtt = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello world

00:00:05.000 --> 00:00:08.000
This is a test
"""
    result = convert_captions_to_whisper_format(vtt, "vid123", "Test Video")
    assert result["video_file"] == "vid123.mp4"
    assert len(result["segments"]) == 2
    assert result["segments"][0]["start"] == 1.0
    assert result["segments"][0]["end"] == 4.0
    assert result["segments"][0]["text"] == "Hello world"
    assert result["segments"][1]["start"] == 5.0
    assert result["segments"][1]["end"] == 8.0
    assert result["segments"][1]["text"] == "This is a test"
    assert result["text"] == "Hello world This is a test"
    assert result["language"] == "en"


def test_convert_vtt_with_milliseconds():
    vtt = """WEBVTT

00:01:30.500 --> 00:01:35.250
Hello world
"""
    result = convert_captions_to_whisper_format(vtt, "vid123", "Test Video")
    assert result["segments"][0]["start"] == 90.5
    assert result["segments"][0]["end"] == 95.25


def test_convert_srt_to_whisper_format():
    srt = """1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:00:08,000
This is a test
"""
    result = convert_captions_to_whisper_format(srt, "vid123", "Test Video")
    assert result["video_file"] == "vid123.mp4"
    assert len(result["segments"]) == 2
    assert result["segments"][0]["start"] == 1.0
    assert result["segments"][0]["end"] == 4.0
    assert result["segments"][0]["text"] == "Hello world"


def test_convert_empty_captions():
    result = convert_captions_to_whisper_format("", "vid123", "Test Video")
    assert result["video_file"] == "vid123.mp4"
    assert result["segments"] == []
    assert result["text"] == ""


def test_convert_vtt_skips_empty_lines():
    vtt = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello world

00:00:05.000 --> 00:00:08.000

00:00:09.000 --> 00:00:12.000
Test
"""
    result = convert_captions_to_whisper_format(vtt, "vid123", "Test Video")
    # Should skip the empty caption at 5-8 seconds
    assert len(result["segments"]) == 2
    assert result["segments"][0]["text"] == "Hello world"
    assert result["segments"][1]["text"] == "Test"