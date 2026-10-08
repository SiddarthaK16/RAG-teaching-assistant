"""Answer, explain, summarize, or quiz from locally retrieved lecture excerpts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = PROJECT_ROOT / "chroma_db"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHAT_MODEL = "qwen3:4b"
DEFAULT_COLLECTION = "lecture_chunks_bge_small_en_v1_5"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
SYSTEM_PROMPT = (
    "You are a teaching assistant whose only source of facts is the supplied lecture transcript excerpts. "
    "Do not use outside knowledge, fill gaps from memory, or invent details. "
    "If the excerpts do not support an answer, say that the provided lecture resources do not contain enough information. "
    "Cite every factual claim with one or more excerpt numbers, such as [1] or [2]. "
    "Use only the citation numbers supplied in the excerpts."
)


def format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_title(metadata: dict[str, Any]) -> str:
    title = str(metadata.get("lecture_title", metadata.get("video_file", "Lecture")))
    return Path(title).stem.replace("-", " ").title()


def retrieve_passages(
    question: str,
    *,
    top_k: int = 8,
    store: Path = DEFAULT_STORE,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: SentenceTransformer | None = None,
) -> list[dict[str, Any]]:
    if not store.is_dir():
        raise FileNotFoundError(f"Vector store not found: {store}. Run src/embed_chunks.py first.")

    client = chromadb.PersistentClient(path=str(store))
    collection = client.get_collection(collection_name)
    count = collection.count()
    if count == 0:
        raise ValueError("The local vector collection is empty. Run src/embed_chunks.py first.")

    model = embedding_model or SentenceTransformer(embedding_model_name, device="cpu")
    query_vector = model.encode(
        [QUERY_INSTRUCTION + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"document": document, "metadata": metadata, "distance": distance}
        for document, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def clean_model_answer(answer: str) -> str:
    """Remove Qwen reasoning blocks sometimes returned inline by Ollama."""
    answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", answer, flags=re.IGNORECASE):
        answer = re.split(r"</think>", answer, maxsplit=1, flags=re.IGNORECASE)[-1]
    if re.search(r"<think>", answer, flags=re.IGNORECASE):
        answer = re.split(r"<think>", answer, maxsplit=1, flags=re.IGNORECASE)[0]
    return answer.strip()


def generate_answer(
    question: str,
    passages: list[dict[str, Any]],
    *,
    mode: str = "answer",
    quiz_count: int = 5,
    chat_model: str = DEFAULT_CHAT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    task_instructions = {
        "answer": "Answer the question directly and concisely.",
        "explain": (
            "Teach the topic step by step in clear language. Define important terms and use examples only "
            "when the excerpts provide them."
        ),
        "summarize": (
            "Summarize the topic using a short heading and organized bullet points. Keep the main ideas and "
            "relationships from the excerpts; do not add background facts from elsewhere."
        ),
        "quiz": (
            f"Create exactly {quiz_count} multiple-choice questions about the topic, each with four options "
            "(A–D). Base every question and answer only on the excerpts. Use a '## Questions' heading, then "
            "a separate '## Answer Key' heading after all questions. Include a brief explanation and citation "
            "for each correct answer. Do not reveal answers in the question section."
        ),
    }
    if mode not in task_instructions:
        raise ValueError(f"Unknown mode: {mode}")

    source_sections = []
    for number, passage in enumerate(passages, start=1):
        metadata = passage["metadata"]
        start = format_time(float(metadata.get("start_time", 0)))
        end = format_time(float(metadata.get("end_time", 0)))
        title = format_title(metadata)
        source_sections.append(f"[{number}] {title} ({start}–{end})\n{passage['document']}")

    user_prompt = (
        f"Task: {task_instructions[mode]}\n"
        f"Topic or question: {question}\n\n"
        "Relevant lecture transcript excerpts:\n\n"
        + "\n\n".join(source_sections)
    )
    payload = json.dumps(
        {
            "model": chat_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "think": False,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")
    request = Request(
        ollama_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Could not reach Ollama at {ollama_url}: {error}. "
            f"Start Ollama and download the model with: ollama pull {chat_model}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Ollama returned invalid JSON: {error}") from error

    message = result.get("message", {})
    answer = clean_model_answer(str(message.get("content", "")))
    if not answer:
        raise RuntimeError("Ollama returned an empty answer")
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Answer a question from the local lecture index using Ollama."
    )
    parser.add_argument("question", help="Question or topic from the lecture transcripts")
    parser.add_argument("--mode", choices=("answer", "explain", "summarize", "quiz"), default="answer")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--quiz-count", type=int, default=5)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    args = parser.parse_args()

    if args.top_k < 1 or args.quiz_count < 1:
        parser.error("--top-k and --quiz-count must be positive")
    try:
        passages = retrieve_passages(
            args.question,
            top_k=args.top_k,
            store=args.store,
            embedding_model_name=args.embedding_model,
            collection_name=args.collection,
        )
        answer = generate_answer(
            args.question,
            passages,
            mode=args.mode,
            quiz_count=args.quiz_count,
            chat_model=args.chat_model,
            ollama_url=args.ollama_url,
        )
    except Exception as error:
        parser.error(str(error))

    print(answer)
    print("\nSources:")
    for number, passage in enumerate(passages, start=1):
        metadata = passage["metadata"]
        start = format_time(float(metadata.get("start_time", 0)))
        end = format_time(float(metadata.get("end_time", 0)))
        print(f"[{number}] {format_title(metadata)} — {start}–{end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
