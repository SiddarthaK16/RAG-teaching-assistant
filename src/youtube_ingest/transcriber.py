"""Whisper transcription fallback for videos without captions."""
import whisper
from pathlib import Path
from .config import YouTubeIngestConfig


# Cache for loaded Whisper models to avoid reloading
_whisper_models = {}


def _get_whisper_model(model_name: str):
    """Get or load a Whisper model."""
    if model_name not in _whisper_models:
        _whisper_models[model_name] = whisper.load_model(model_name, device="cpu")
    return _whisper_models[model_name]


def transcribe_video(video_path: Path, config: YouTubeIngestConfig) -> dict:
    """Transcribe a video file using Whisper, returning Whisper JSON format."""
    model = _get_whisper_model(config.whisper_model)

    # Transcribe with English language, no fp16 on CPU
    result = model.transcribe(
        str(video_path),
        language="en",
        fp16=False,
    )

    # Add video_file field for compatibility with existing pipeline
    result["video_file"] = video_path.name

    return result