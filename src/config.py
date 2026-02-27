# config.py
import os

# Environment setup
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen")
MODEL_MAP = {
    "CustomVoice": f"{MODEL_PATH}/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "VoiceDesign": f"{MODEL_PATH}/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Base": f"{MODEL_PATH}/Qwen3-TTS-12Hz-1.7B-Base",
}

TASK_TYPE = os.environ.get("QWEN3_TTS_TASK_TYPE", "CustomVoice")
MODEL_NAME = os.environ.get("QWEN3_TTS_MODEL", MODEL_MAP.get(TASK_TYPE, MODEL_MAP["CustomVoice"]))

# Inference and streaming parameters
INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "270"))  # seconds
STREAM_CHUNK_TOKENS = int(os.environ.get("STREAM_CHUNK_TOKENS", "24"))  # ~2s at 12Hz
STREAM_LEFT_CONTEXT_TOKENS = int(os.environ.get("STREAM_LEFT_CONTEXT_TOKENS", "25"))

# Audio format mappings
CONTENT_TYPE_MAP = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/ogg; codecs=opus",
    "mp3": "audio/mpeg",
}

EXT_MAP = {
    "opus": "ogg",
}