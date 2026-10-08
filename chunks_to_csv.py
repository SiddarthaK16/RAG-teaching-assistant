"""Export chunk JSONL records to a spreadsheet-friendly CSV file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "chunks" / "chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "chunks" / "chunks.csv"
FIELDNAMES = [
    "chunk_id",
    "source_file",
    "video_file",
    "lecture_title",
    "start_time",
    "end_time",
    "word_count",
    "text",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export chunk JSONL records to CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Chunk JSONL file not found: {args.input}")

    rows = []
    try:
        with args.input.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        parser.error(f"Line {line_number} is not a JSON object")
                    rows.append(row)
    except json.JSONDecodeError as error:
        parser.error(f"Invalid JSON in {args.input}, line {error.lineno}: {error.msg}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows, columns=FIELDNAMES)
    dataframe.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Exported {len(rows)} chunk(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
