"""CLI entry point for YouTube playlist ingestion."""
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from youtube_ingest.config import YouTubeIngestConfig
from youtube_ingest.pipeline import run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a YouTube playlist into the RAG teaching assistant."
    )
    parser.add_argument("playlist_url", help="YouTube playlist URL")
    parser.add_argument(
        "--use-captions",
        action="store_true",
        default=True,
        help="Prefer YouTube captions over Whisper transcription (default: True)",
    )
    parser.add_argument(
        "--force-whisper",
        action="store_true",
        default=False,
        help="Force Whisper transcription even if captions exist",
    )
    parser.add_argument(
        "--whisper-model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for transcription fallback (default: medium)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=280,
        help="Maximum words per chunk (default: 280)",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=40,
        help="Word overlap between chunks (default: 40)",
    )
    parser.add_argument(
        "--collection",
        default="lecture_chunks_bge_small_en_v1_5",
        help="Chroma collection name (default: lecture_chunks_bge_small_en_v1_5)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("chroma_db"),
        help="Chroma database directory (default: chroma_db)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("videos"),
        help="Directory for downloaded videos (default: videos)",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("transcripts"),
        help="Directory for transcripts (default: transcripts)",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("chunks"),
        help="Directory for chunk files (default: chunks)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.max_words < 1:
        parser.error("--max-words must be positive")
    if args.overlap_words < 0 or args.overlap_words >= args.max_words:
        parser.error("--overlap-words must be between 0 and max-words - 1")

    config = YouTubeIngestConfig(
        playlist_url=args.playlist_url,
        use_captions=args.use_captions,
        force_whisper=args.force_whisper,
        whisper_model=args.whisper_model,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
        collection=args.collection,
        store=args.store,
        videos_dir=args.videos_dir,
        transcripts_dir=args.transcripts_dir,
        chunks_dir=args.chunks_dir,
    )

    print(f"Starting ingestion for playlist: {config.playlist_url}")
    print(f"Config: use_captions={config.use_captions}, force_whisper={config.force_whisper}, whisper_model={config.whisper_model}")

    try:
        result = run_ingestion(config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("\nIngestion complete!")
    print(f"  Videos processed: {result.videos_processed}")
    print(f"  Captions used: {result.captions_used}")
    print(f"  Whisper transcribed: {result.whisper_transcribed}")
    print(f"  Chunks created: {result.chunks_created}")

    return 0


if __name__ == "__main__":
    sys.exit(main())