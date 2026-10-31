"""Convert VTT/SRT captions to Whisper JSON format."""
import re
from typing import List, Dict, Any


def _parse_vtt_time(time_str: str) -> float:
    """Parse VTT timestamp (HH:MM:SS.mmm) to seconds."""
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def _parse_srt_time(time_str: str) -> float:
    """Parse SRT timestamp (HH:MM:SS,mmm) to seconds."""
    time_str = time_str.replace(",", ".")
    return _parse_vtt_time(time_str)


def _parse_vtt(content: str) -> List[Dict[str, Any]]:
    """Parse VTT format captions."""
    segments = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for timestamp line
        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                start_str = parts[0].strip()
                end_str = parts[1].strip()
                # Collect text lines until next timestamp or end
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                    text_lines.append(lines[i].strip())
                    i += 1
                text = " ".join(text_lines).strip()
                if text:  # Skip empty captions
                    start = _parse_vtt_time(start_str)
                    end = _parse_vtt_time(end_str)
                    segments.append({"start": start, "end": end, "text": text})
                continue
        i += 1
    return segments


def _parse_srt(content: str) -> List[Dict[str, Any]]:
    """Parse SRT format captions."""
    segments = []
    # Pattern: number + newline + HH:MM:SS,mmm --> HH:MM:SS,mmm + newline + text
    pattern = re.compile(r"\d+\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n([\s\S]*?)(?=\n\d+\s*\n|\Z)")

    for match in pattern.finditer(content):
        start_str, end_str, text = match.groups()
        start = _parse_srt_time(start_str)
        end = _parse_srt_time(end_str)
        text = text.strip().replace("\n", " ")
        if text:  # Skip empty captions
            segments.append({"start": start, "end": end, "text": text})
    return segments


def convert_captions_to_whisper_format(captions: str, video_id: str, title: str) -> Dict[str, Any]:
    """Convert VTT or SRT captions to Whisper-compatible JSON format."""
    if not captions or not captions.strip():
        return {
            "video_file": f"{video_id}.mp4",
            "segments": [],
            "text": "",
            "language": "en",
        }

    # Detect format
    content = captions.strip()
    if content.startswith("WEBVTT"):
        segments = _parse_vtt(content)
    else:
        # Assume SRT format
        segments = _parse_srt(content)

    # Build full text
    full_text = " ".join(seg["text"] for seg in segments)

    return {
        "video_file": f"{video_id}.mp4",
        "segments": segments,
        "text": full_text,
        "language": "en",
    }