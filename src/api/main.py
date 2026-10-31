"""FastAPI application for RAG teaching assistant."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import ingest, query, lectures
from .models import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Starting RAG Teaching Assistant API...")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="RAG Teaching Assistant API",
    description="API for YouTube playlist ingestion and lecture Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(lectures.router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        ollama_connected=False,  # Could add actual check
        chroma_collections=0,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)