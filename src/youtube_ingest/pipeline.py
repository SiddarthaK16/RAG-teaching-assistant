"""Pipeline orchestrator for YouTube playlist ingestion."""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from .config import YouTubeIngestConfig
from .extractor import extract_playlist_info, get_video_captions, VideoInfo, PlaylistInfo
from .downloader import download_video
from .transcriber import transcribe_video
from .caption_converter import convert_captions_to_whisper_format

# Import chunking at module level for patching
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from chunking import make_chunks


@dataclass
class IngestionResult:
    """Result of a playlist ingestion run."""
    videos_processed: int = 0
    captions_used: int = 0
    whisper_transcribed: int = 0
    chunks_created: int = 0


def _save_transcript(transcript: Dict[str, Any], transcripts_dir: Path, video_id: str) -> Path:
    """Save transcript to JSON file."""
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcripts_dir / f"{video_id}.json"
    with transcript_path.open("w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    return transcript_path


def _process_video_with_captions(
    video: VideoInfo,
    captions: str,
    transcripts_dir: Path,
) -> Dict[str, Any]:
    """Process a video that has captions."""
    transcript = convert_captions_to_whisper_format(captions, video.video_id, video.title)
    _save_transcript(transcript, transcripts_dir, video.video_id)
    return transcript


def _process_video_without_captions(
    video: VideoInfo,
    config: YouTubeIngestConfig,
    videos_dir: Path,
    transcripts_dir: Path,
) -> Dict[str, Any]:
    """Process a video without captions (download + transcribe)."""
    # Download video
    video_path = download_video(video, videos_dir, config)
    if video_path is None:
        raise RuntimeError(f"Failed to download video: {video.title}")

    # Transcribe with Whisper
    transcript = transcribe_video(video_path, config)
    _save_transcript(transcript, transcripts_dir, video.video_id)
    return transcript


def run_ingestion(config: YouTubeIngestConfig) -> IngestionResult:
    """Run the full YouTube playlist ingestion pipeline."""
    result = IngestionResult()

    # Extract playlist info
    playlist_info = extract_playlist_info(config.playlist_url)
    if not playlist_info.videos:
        return result

    # Ensure directories exist
    config.videos_dir.mkdir(parents=True, exist_ok=True)
    config.transcripts_dir.mkdir(parents=True, exist_ok=True)
    config.chunks_dir.mkdir(parents=True, exist_ok=True)

    all_transcripts = []

    # Process each video
    for video in playlist_info.videos:
        result.videos_processed += 1

        # Try to get captions
        captions = get_video_captions(video.url)

        if captions and not config.force_whisper:
            # Use captions
            transcript = _process_video_with_captions(video, captions, config.transcripts_dir)
            result.captions_used += 1
        else:
            # Download and transcribe with Whisper
            try:
                transcript = _process_video_without_captions(
                    video, config, config.videos_dir, config.transcripts_dir
                )
                result.whisper_transcribed += 1
            except Exception as e:
                # Log error and continue with next video
                print(f"Warning: Failed to process {video.title}: {e}")
                continue

        all_transcripts.append(transcript)

    # Create chunks from all transcripts
    if all_transcripts:
        all_chunks = []
        for transcript in all_transcripts:
            chunks = make_chunks(
                transcript,
                source_file=transcript["video_file"],
                max_words=config.max_words,
                overlap_words=config.overlap_words,
            )
            all_chunks.extend(chunks)

        # Write chunks to JSONL
        config.chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = config.chunks_dir / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        # Embed and index chunks
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from embed_chunks import main as embed_chunks_main

        # Temporarily replace sys.argv for embed_chunks
        old_argv = sys.argv
        sys.argv = [
            "embed_chunks",
            "--input", str(chunks_path),
            "--store", str(config.store),
            "--collection", config.collection,
            "--model", "BAAI/bge-small-en-v1.5",
        ]
        try:
            embed_chunks_main()
        finally:
            sys.argv = old_argv

        result.chunks_created = len(all_chunks)

    return result