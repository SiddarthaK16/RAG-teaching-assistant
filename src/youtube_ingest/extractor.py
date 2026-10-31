"""yt-dlp wrapper for playlist metadata and caption extraction."""
from dataclasses import dataclass
from typing import Optional
import yt_dlp


@dataclass
class VideoInfo:
    """Information about a single video in a playlist."""
    url: str
    title: str
    duration: float
    video_id: str


@dataclass
class PlaylistInfo:
    """Information about a YouTube playlist."""
    title: str
    videos: list[VideoInfo]


def extract_playlist_info(url: str) -> PlaylistInfo:
    """Extract playlist metadata without downloading videos."""
    ydl_opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=False)

    videos = [
        VideoInfo(
            url=f"https://youtube.com/watch?v={entry['id']}",
            title=entry.get("title", "Unknown"),
            duration=entry.get("duration", 0),
            video_id=entry["id"],
        )
        for entry in result.get("entries", [])
        if entry
    ]
    return PlaylistInfo(title=result.get("title", "Playlist"), videos=videos)


def get_video_captions(video_url: str, lang: str = "en") -> Optional[str]:
    """Extract captions/subtitles for a video, preferring manual over auto-generated."""
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(video_url, download=False)

    # Prefer manual subtitles over automatic captions
    for key in ("subtitles", "automatic_captions"):
        subs = result.get(key, {})
        if lang in subs:
            # yt-dlp returns list of dicts; we want the VTT data
            for sub in subs[lang]:
                if "data" in sub:
                    return sub["data"]
                # If no 'data' key, yt-dlp may have written to file
                # In that case, we'd need to read the file, but with the
                # above options it should return the data directly
    return None