# RAG-Based Teaching Assistant

## Problem

Long course lectures are hard to search when you are trying to find one specific topic. A lecture may cover several ideas over an hour or more, and its title or description may not tell you exactly where a concept is explained. Scrubbing through the video takes time.

Video timestamps and chapter markers can help, but they are not always present, precise, or reliable. A timestamp may point near the topic without landing on the relevant explanation. Captions and transcripts can also contain recognition errors, especially for technical terms, names, equations, and code.

The goal of this project is to make course lectures searchable by topic and return a useful pointer to the relevant lecture and moment in the video.

## Intended approach

1. Transcribe each lecture locally with OpenAI Whisper.
2. Keep the transcript text aligned with its lecture and time range.
3. Split transcripts into small, overlapping text chunks while retaining the source video, lecture title, and start/end times as metadata.
4. Create vector embeddings for the chunks and store them in a local vector database.
5. Retrieve relevant chunks for a question and show the lecture title, timestamp, and supporting transcript text.

This should make it easier to locate a topic, but results will depend on transcript quality and chunk boundaries. Timestamps are pointers to review, not guarantees that the exact explanation begins at that second. Visual-only information—such as a diagram, slide text, or a code sample—may not be represented in an audio transcript.

## Current project status

- The project contains 11 downloaded MIT 15.773 lecture videos in `videos/`.
- The videos have consistent, numbered filenames.
- Whisper is installed in the Conda environment named `rag` (Python 3.12).
- Transcript generation, chunking, embeddings, retrieval, and the assistant interface are still to be built.

Large media and generated artifacts are excluded from Git in `.gitignore`.

## Local setup

Activate the Conda environment:

```bash
conda activate rag
```

Create a folder for transcripts:

```bash
mkdir -p transcripts
```

Transcribe one lecture first to check accuracy:

```bash
whisper "videos/01-introduction-to-neural-networks-and-deep-learning-training-deep-nns.mp4" \
  --model small \
  --language English \
  --device cpu \
  --fp16 False \
  --output_format json \
  --output_dir transcripts
```

Whisper's JSON output includes transcript segments with start and end times. Once the transcript quality is acceptable, the same command can be run for each video.

## Planned data for each chunk

Each indexed chunk should retain at least:

- `video_file`: local video filename
- `lecture_title`: readable lecture title
- `start_time` and `end_time`: seconds into the video
- `text`: transcript text for the chunk

Keeping this metadata with the embedding lets search results identify both the relevant lecture and a time range to inspect.

## Notes

- Local Whisper transcription runs on the CPU on this machine and may take a while for all lectures.
- If OpenAI's embeddings API is used, transcript chunks are sent to that API for embedding; the videos and vector database can remain local.
- Review transcript samples before indexing everything, especially where technical vocabulary or lecture audio is unclear.
