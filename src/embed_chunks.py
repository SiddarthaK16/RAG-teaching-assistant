"""Embed transcript chunks locally and store them in a persistent Chroma collection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "chunks" / "chunks.jsonl"
DEFAULT_STORE = PROJECT_DIR / "chroma_db"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_COLLECTION = "lecture_chunks_bge_small_en_v1_5"
METADATA_FIELDS = (
    "chunk_id", "source_file", "video_file", "lecture_title",
    "start_time", "end_time", "word_count",
)


def read_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("chunk_id") or not row.get("text"):
                raise ValueError(f"Line {line_number} must have chunk_id and text fields")
            chunks.append(row)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Create local BGE embeddings for transcript chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Chunk file not found: {args.input}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        chunks = read_chunks(args.input)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"Could not read chunks: {error}")
    if not chunks:
        parser.error(f"No chunks found in: {args.input}")

    print(f"Loading local embedding model: {args.model}", flush=True)
    model = SentenceTransformer(args.model, device="cpu")
    max_tokens = model.max_seq_length
    long_chunks = []
    for chunk in chunks:
        token_count = len(
            model.tokenizer(chunk["text"], add_special_tokens=True, truncation=False)["input_ids"]
        )
        if token_count > max_tokens:
            long_chunks.append((chunk["chunk_id"], token_count))
    if long_chunks:
        examples = ", ".join(f"{chunk_id} ({count} tokens)" for chunk_id, count in long_chunks[:5])
        print(
            f"Warning: {len(long_chunks)} chunk(s) exceed the model's {max_tokens}-token limit and will be truncated. "
            f"Examples: {examples}",
            flush=True,
        )

    client = chromadb.PersistentClient(path=str(args.store))
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine", "embedding_model": args.model},
    )
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.get("source_file", "unknown")].append(chunk)

    for source_file, source_chunks in sorted(by_source.items()):
        ids = [str(chunk["chunk_id"]) for chunk in source_chunks]
        documents = [str(chunk["text"]) for chunk in source_chunks]
        embeddings = model.encode(
            documents,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Replace prior records for this lecture so reruns cannot leave stale chunk IDs behind.
        collection.delete(where={"source_file": source_file})
        metadatas = [
            {key: chunk[key] for key in METADATA_FIELDS if key in chunk}
            for chunk in source_chunks
        ]
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        print(f"Indexed {len(source_chunks)} chunk(s) from {source_file}", flush=True)

    print(f"Done. Collection '{args.collection}' has {collection.count()} chunk(s) in {args.store}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
