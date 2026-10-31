"""Query routes for asking questions."""
from fastapi import APIRouter, HTTPException
import sys
from pathlib import Path

from ..models import AskRequest, AskResponse, SourceCitation

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question and get an answer from the lecture transcripts."""
    try:
        from answer_question import retrieve_passages, generate_answer
        from sentence_transformers import SentenceTransformer

        # Load embedding model
        model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

        # Retrieve relevant passages
        passages = retrieve_passages(
            request.question,
            top_k=request.top_k,
            embedding_model_name="BAAI/bge-small-en-v1.5",
            embedding_model=model,
        )

        # Generate answer
        answer = generate_answer(
            request.question,
            passages,
            mode=request.mode,
            quiz_count=request.quiz_count,
        )

        # Format sources
        sources = []
        for passage in passages:
            metadata = passage.get("metadata", {})
            sources.append(SourceCitation(
                lecture_title=metadata.get("lecture_title", metadata.get("video_file", "Unknown")),
                start_time=float(metadata.get("start_time", 0)),
                end_time=float(metadata.get("end_time", 0)),
                text=passage.get("document", "")[:500],  # Truncate for response
            ))

        return AskResponse(answer=answer, mode=request.mode, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))