# tts_generator.py
from typing import Generator, Dict, Any, Optional
import time
import uuid
import numpy as np
import torch
from config import INFERENCE_TIMEOUT, STREAM_CHUNK_TOKENS, STREAM_LEFT_CONTEXT_TOKENS
from model_loader import _get_native_model
from prefill_builder import _build_prefill
from utils import _sample_next_token, _pcm_s16le_base64
from forced_align import forced_align  # assuming this exists


def generate_tts(params: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    job_id = str(uuid.uuid4())
    start_t = time.time()

    def _fail(msg: str):
        yield {"error": msg, "done": True}

    if "text" not in params:
        yield from _fail("Missing required field: text")
        return

    task_type = params.get("task_type", "CustomVoice")
    stream = bool(params.get("stream", False))

    if not stream:
        # Non-streaming path...
        # (kept concise here; full implementation as in original)
        pass

    # Streaming mode
    try:
        native = _get_native_model()
        qwen_model = native.model
        # ... [rest of streaming logic using components from other modules]
        # Includes token generation, decoding, chunk emission
    except Exception as e:
        yield {"error": str(e), "done": True}