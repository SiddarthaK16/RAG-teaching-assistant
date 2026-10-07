"""Transcribe every video in ./videos with OpenAI Whisper."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import whisper


PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = PROJECT_DIR / "videos"
TRANSCRIPT_DIR = PROJECT_DIR / "transcripts"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe all videos in the videos folder into timestamped JSON files."
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model to use (default: medium).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace transcript JSON files that already exist.",
    )
    args = parser.parse_args()

    if not VIDEO_DIR.is_dir():
        print(f"Video folder not found: {VIDEO_DIR}")
        return 1

    videos = sorted(
        path
        for path in VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"No supported video files found in: {VIDEO_DIR}")
        return 1

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Found {len(videos)} video(s). Device: {device}. Model: {args.model}.", flush=True)
    print("Loading Whisper model (the first run may download model weights)...", flush=True)
    model = whisper.load_model(args.model, device=device)

    total_started = time.perf_counter()
    failures: list[tuple[str, str]] = []
    completed = 0

    for index, video_path in enumerate(videos, start=1):
        transcript_path = TRANSCRIPT_DIR / f"{video_path.stem}.json"
        if transcript_path.exists() and not args.overwrite:
            print(f"[{index}/{len(videos)}] Skipping existing transcript: {transcript_path.name}")
            continue

        print(f"[{index}/{len(videos)}] Transcribing: {video_path.name}", flush=True)
        video_started = time.perf_counter()
        try:
            result = model.transcribe(
                str(video_path),
                language="en",
                fp16=(device == "cuda"),
            )
            result["video_file"] = video_path.name
            result["model"] = args.model

            temporary_path = transcript_path.with_suffix(".json.tmp")
            with temporary_path.open("w", encoding="utf-8") as output_file:
                json.dump(result, output_file, ensure_ascii=False, indent=2)
                output_file.write("\n")
            temporary_path.replace(transcript_path)

            elapsed = time.perf_counter() - video_started
            completed += 1
            print(f"Saved {transcript_path.name} ({elapsed / 60:.1f} min)", flush=True)
        except Exception as error:  # Keep processing the remaining videos.
            failures.append((video_path.name, str(error)))
            print(f"Failed on {video_path.name}: {error}", flush=True)

    total_elapsed = time.perf_counter() - total_started
    print(
        f"Finished. New transcripts: {completed}; skipped: "
        f"{len(videos) - completed - len(failures)}; failed: {len(failures)}. "
        f"Elapsed: {total_elapsed / 3600:.2f} hours.",
        flush=True,
    )

    if failures:
        print("Failed videos:")
        for filename, message in failures:
            print(f"- {filename}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
