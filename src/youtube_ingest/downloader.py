"""Video downloader with retry/backoff using yt-dlp."""
from pathlib import Path
from typing import Optional
import yt_dlp
from yt_dlp.utils import ExtractorError

from .extractor import VideoInfo
from .config import YouTubeIngestConfig


def download_video(video: VideoInfo, output_dir: Path, config: YouTubeIngestConfig) -> Optional[Path]:
    """Download a video using yt-dlp with retry/backoff for rate limits."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output template: videos/Video Title.ext
    outtmpl = str(output_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": outtmpl,
        "retries": 3,
        "fragment_retries": 3,
        "retry_sleep_functions": {
            "http": lambda n: min(2 ** n, 60),  # Exponential backoff: 2, 4, 8... max 60s
            "fragment": lambda n: min(2 ** n, 60),
        },
        "ignoreerrors": False,  # We handle errors ourselves
    }

    # Skip age-restricted videos that require login
    ydl_opts["age_limit"] = 99

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video.url])
    except ExtractorError as e:
        error_msg = str(e).lower()
        if "private" in error_msg or "age" in error_msg or "restricted" in error_msg:
            # Skip private/age-restricted videos silently (warning logged by caller)
            return None
        if "rate limit" in error_msg or "429" in error_msg:
            # Rate limited after retries exhausted
            return None
        # Re-raise other extractor errors
        raise
    except Exception:
        # Any other error (network, etc.)
        return None

    # Find the downloaded file (yt-dlp may have changed extension)
    downloaded_files = list(output_dir.glob(f"{video.title}.*"))
    if not downloaded_files:
        return None

    # Return the first matching file (should be only one)
    return downloaded_files[0]