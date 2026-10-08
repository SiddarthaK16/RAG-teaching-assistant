"""Search the local Chroma collection with the BGE embedding model."""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_STORE = PROJECT_DIR / "chroma_db"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_COLLECTION = "lecture_chunks_bge_small_en_v1_5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def main() -> int:
    parser = argparse.ArgumentParser(description="Search locally indexed lecture chunks.")
    parser.add_argument("query", help="Question or topic to search for")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not args.store.is_dir():
        parser.error(f"Vector store not found: {args.store}. Run embed_chunks.py first.")

    client = chromadb.PersistentClient(path=str(args.store))
    try:
        collection = client.get_collection(args.collection)
    except Exception as error:
        parser.error(f"Could not open collection '{args.collection}': {error}")
    if collection.count() == 0:
        parser.error("The collection is empty. Run embed_chunks.py first.")

    model = SentenceTransformer(args.model, device="cpu")
    query_vector = model.encode(
        [QUERY_INSTRUCTION + args.query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(args.top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    for rank, (document, metadata, distance) in enumerate(
        zip(result["documents"][0], result["metadatas"][0], result["distances"][0]),
        start=1,
    ):
        start = float(metadata.get("start_time", 0))
        end = float(metadata.get("end_time", 0))
        title = metadata.get("lecture_title", metadata.get("video_file", "Lecture"))
        print(f"\n{rank}. {title} [{start:.1f}s–{end:.1f}s]  distance={distance:.4f}")
        print(document)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
