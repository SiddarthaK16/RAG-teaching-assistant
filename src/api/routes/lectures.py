"""Lecture routes for browsing indexed content."""
from fastapi import APIRouter, HTTPException
import chromadb
from pathlib import Path
import sys

from ..models import LectureListItem, LectureDetail, SourceCitation

router = APIRouter(prefix="/api/lectures", tags=["lectures"])

# Default paths
DEFAULT_STORE = Path("chroma_db")
DEFAULT_COLLECTION = "lecture_chunks_bge_small_en_v1_5"


def get_chroma_collection():
    """Get the Chroma collection."""
    client = chromadb.PersistentClient(path=str(DEFAULT_STORE))
    return client.get_collection(DEFAULT_COLLECTION)


@router.get("", response_model=list[LectureListItem])
async def list_lectures():
    """List all indexed lectures."""
    try:
        collection = get_chroma_collection()
        # Get all documents to extract unique lectures
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])

        # Group by lecture_title/video_file
        lectures = {}
        for metadata in metadatas:
            key = metadata.get("lecture_title", metadata.get("video_file", "Unknown"))
            if key not in lectures:
                lectures[key] = {"lecture_title": key, "video_file": metadata.get("video_file", ""), "chunk_count": 0}
            lectures[key]["chunk_count"] += 1

        return list(lectures.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{lecture_id}", response_model=LectureDetail)
async def get_lecture_detail(lecture_id: str):
    """Get detailed information about a specific lecture."""
    try:
        collection = get_chroma_collection()
        # Query for chunks belonging to this lecture
        result = collection.get(
            where={"lecture_title": lecture_id},
            include=["documents", "metadatas"]
        )

        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        chunks = []
        for doc, meta in zip(documents, metadatas):
            chunks.append(SourceCitation(
                lecture_title=meta.get("lecture_title", lecture_id),
                start_time=float(meta.get("start_time", 0)),
                end_time=float(meta.get("end_time", 0)),
                text=doc[:500],
            ))

        video_file = metadatas[0].get("video_file", "") if metadatas else ""

        return LectureDetail(
            lecture_title=lecture_id,
            video_file=video_file,
            chunks=chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))