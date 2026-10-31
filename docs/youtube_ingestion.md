# YouTube Playlist Ingestion

This guide covers how to ingest YouTube playlists into the RAG teaching assistant.

## Overview

The YouTube ingestion pipeline:
1. Fetches playlist metadata using yt-dlp
2. For each video, attempts to extract captions/subtitles
3. If captions exist: converts them to Whisper JSON format
4. If no captions: downloads the video and transcribes with Whisper
5. Processes transcripts through existing chunking → embedding → indexing pipeline
6. Makes content queryable via RAG modes (ask, explain, summarize, quiz)

## CLI Usage

### Basic Usage

```bash
# Ingest a playlist (uses captions when available, falls back to Whisper)
python src/ingest_youtube.py "https://youtube.com/playlist?list=PLxxxxxx"

# Force Whisper transcription for all videos (skip captions)
python src/ingest_youtube.py "https://youtube.com/playlist?list=PLxxxxxx" --force-whisper

# Use a specific Whisper model for transcription
python src/ingest_youtube.py "https://youtube.com/playlist?list=PLxxxxxx" --whisper-model small

# Custom chunking parameters
python src/ingest_youtube.py "https://youtube.com/playlist?list=PLxxxxxx" --max-words 200 --overlap-words 30
```

### All Options

```bash
python src/ingest_youtube.py --help
```

```
usage: ingest_youtube.py [-h] [--use-captions] [--force-whisper]
                         [--whisper-model {tiny,base,small,medium,large}]
                         [--max-words MAX_WORDS] [--overlap-words OVERLAP_WORDS]
                         [--collection COLLECTION] [--store STORE]
                         [--videos-dir VIDEOS_DIR] [--transcripts-dir TRANSCRIPTS_DIR]
                         [--chunks-dir CHUNKS_DIR]
                         playlist_url

Ingest a YouTube playlist into the RAG teaching assistant.

positional arguments:
  playlist_url          YouTube playlist URL

options:
  -h, --help            show this help message and exit
  --use-captions        Prefer YouTube captions over Whisper transcription (default: True)
  --force-whisper       Force Whisper transcription even if captions exist
  --whisper-model {tiny,base,small,medium,large}
                        Whisper model size for transcription fallback (default: medium)
  --max-words MAX_WORDS
                        Maximum words per chunk (default: 280)
  --overlap-words OVERLAP_WORDS
                        Word overlap between chunks (default: 40)
  --collection COLLECTION
                        Chroma collection name (default: lecture_chunks_bge_small_en_v1_5)
  --store STORE         Chroma database directory (default: chroma_db)
  --videos-dir VIDEOS_DIR
                        Directory for downloaded videos (default: videos)
  --transcripts-dir TRANSCRIPTS_DIR
                        Directory for transcripts (default: transcripts)
  --chunks-dir CHUNKS_DIR
                        Directory for chunk files (default: chunks)
```

## FastAPI Usage

### Start the Server

```bash
# Development
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Start Ingestion Job

```bash
curl -X POST "http://localhost:8000/api/ingest/youtube" \
  -H "Content-Type: application/json" \
  -d '{
    "playlist_url": "https://youtube.com/playlist?list=PLxxxxxx",
    "use_captions": true,
    "force_whisper": false,
    "whisper_model": "medium",
    "max_words": 280,
    "overlap_words": 40
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Job queued"
}
```

#### Check Job Status

```bash
curl "http://localhost:8000/api/jobs/550e8400-e29b-41d4-a716-446655440000"
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": {
    "total_videos": 10,
    "processed": 3,
    "current_video": "Introduction to Neural Networks",
    "current_step": "transcribing"
  },
  "results": null,
  "error": null
}
```

#### Real-time Progress (WebSocket)

```javascript
const ws = new WebSocket("ws://localhost:8000/api/jobs/550e8400-e29b-41d4-a716-446655440000/ws");
ws.onmessage = (event) => {
  const status = JSON.parse(event.data);
  console.log(`Progress: ${status.progress.processed}/${status.progress.total_videos}`);
  console.log(`Current: ${status.progress.current_video} (${status.progress.current_step})`);
};
```

#### Ask a Question

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is backpropagation?",
    "mode": "explain",
    "top_k": 8,
    "quiz_count": 5
  }'
```

Response:
```json
{
  "answer": "Backpropagation is...",
  "mode": "explain",
  "sources": [
    {
      "lecture_title": "Lecture 3: Backpropagation",
      "start_time": 120.5,
      "end_time": 180.2,
      "text": "Backpropagation computes gradients..."
    }
  ]
}
```

#### List Lectures

```bash
curl "http://localhost:8000/api/lectures"
```

Response:
```json
[
  {
    "lecture_title": "Introduction to Neural Networks",
    "video_file": "01-introduction.mp4",
    "chunk_count": 45
  },
  {
    "lecture_title": "Backpropagation",
    "video_file": "03-backprop.mp4",
    "chunk_count": 38
  }
]
```

#### Get Lecture Details

```bash
curl "http://localhost:8000/api/lectures/Introduction%20to%20Neural%20Networks"
```

#### Health Check

```bash
curl "http://localhost:8000/api/health"
```

## How It Works

### Caption Extraction (Fast Path)

1. yt-dlp fetches available subtitles (manual or auto-generated)
2. VTT/SRT format parsed to extract timestamps and text
3. Converted to Whisper-compatible JSON format
4. Saved to `transcripts/` directory

### Whisper Transcription (Quality Path)

1. yt-dlp downloads video (best quality available)
2. Whisper transcribes audio with timestamps
3. Saved to `transcripts/` directory

### Processing Pipeline

1. All transcripts (captions + Whisper) processed through existing chunking
2. Chunks saved to `chunks/chunks.jsonl`
3. Embedded with BGE model and indexed in ChromaDB
4. Available for querying via existing RAG modes

## Configuration

### Environment Variables

No environment variables required. All configuration via CLI arguments or API request body.

### File Structure

```
project_root/
├── videos/           # Downloaded videos (only when no captions)
├── transcripts/      # JSON transcripts (captions or Whisper)
├── chunks/           # Chunked transcripts (chunks.jsonl)
└── chroma_db/        # Vector database
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Private video" warning | Video is private/unlisted - skipped automatically |
| "Age restricted" warning | Video requires login - skipped automatically |
| Rate limited (HTTP 429) | Automatic retry with exponential backoff (max 3 retries) |
| Download fails | Check network, video may be unavailable in region |
| Whisper OOM | Use smaller model (`--whisper-model tiny`) |

### Logs

The CLI outputs progress for each video:
```
Processing: Video Title [1/10]
  → Captions found, using captions (fast path)
Processing: Another Video [2/10]
  → No captions, downloading...
  → Transcribing with Whisper (medium)...
```

## Performance Tips

1. **Use captions when available**: Much faster, no download needed
2. **Choose appropriate Whisper model**:
   - `tiny`/`base`: Fast, lower accuracy
   - `small`/`medium`: Balanced (default: medium)
   - `large`: Best accuracy, slowest
3. **Batch large playlists**: Process in chunks if >50 videos

## API Reference

See the interactive API documentation at `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` (ReDoc) when the server is running.