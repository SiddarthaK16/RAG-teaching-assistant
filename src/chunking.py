"""Turn Whisper JSON transcripts into timestamped, overlapping text chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRANSCRIPT_DIR = PROJECT_DIR / "transcripts"
DEFAULT_OUTPUT = PROJECT_DIR / "chunks" / "chunks.jsonl"


def make_chunks(transcript: dict[str, Any], source_file: str, max_words: int, overlap_words: int) -> list[dict[str, Any]]:
    """Chunk segment words while retaining the segment time range for each word."""
    words: list[tuple[str, float, float]] = []
    for segment in transcript.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        words.extend((word, start, end) for word in text.split())

    if not words:
        return []

    video_file = transcript.get("video_file") or f"{Path(source_file).stem}.mp4"
    lecture_title = Path(video_file).stem
    step = max_words - overlap_words
    chunks = []
    for chunk_number, offset in enumerate(range(0, len(words), step), start=1):
        group = words[offset : offset + max_words]
        if not group:
            break
        chunks.append(
            {
                "chunk_id": f"{Path(source_file).stem}-{chunk_number:04d}",
                "source_file": source_file,
                "video_file": video_file,
                "lecture_title": lecture_title,
                "start_time": group[0][1],
                "end_time": group[-1][2],
                "text": " ".join(word for word, _, _ in group),
                "word_count": len(group),
            }
        )
        if offset + max_words >= len(words):
            break
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create timestamped, overlapping chunks from Whisper transcript JSON files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-words", type=int, default=280, help="Maximum words per chunk (default: 280).")
    parser.add_argument("--overlap-words", type=int, default=40, help="Words repeated between chunks (default: 40).")
    args = parser.parse_args()

    if args.max_words < 1 or args.overlap_words < 0 or args.overlap_words >= args.max_words:
        parser.error("--max-words must be positive and --overlap-words must be between 0 and max-words - 1")
    if not args.input_dir.is_dir():
        parser.error(f"Transcript directory not found: {args.input_dir}")

    transcript_paths = sorted(args.input_dir.glob("*.json"))
    if not transcript_paths:
        parser.error(f"No JSON transcripts found in: {args.input_dir}")

    all_chunks = []
    for path in transcript_paths:
        try:
            transcript = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"Could not read {path.name}: {error}")
        if not isinstance(transcript, dict) or not isinstance(transcript.get("segments"), list):
            parser.error(f"{path.name} does not contain a Whisper 'segments' list")
        all_chunks.extend(make_chunks(transcript, path.name, args.max_words, args.overlap_words))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for chunk in all_chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Created {len(all_chunks)} chunks from {len(transcript_paths)} transcript(s): {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
