# model_loader.py
from typing import Any
import torch
from config import MODEL_NAME

_native = None


def _get_native_model():
    global _native
    if _native is not None:
        return _native

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision(
                os.environ.get("TORCH_MATMUL_PRECISION", "high")
            )
        except Exception:
            pass
        # Best-effort SDPA backend preference (does not affect FlashAttention2).
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

    attn_impl = os.environ.get("QWEN3_TTS_ATTN_IMPLEMENTATION")

    print(f"Loading native model: {MODEL_NAME} (task_type={TASK_TYPE}, device={device_map})")
    load_kwargs = dict(
        device_map=device_map,
        dtype=torch_dtype,
    )
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    _native = Qwen3TTSModel.from_pretrained(MODEL_NAME, **load_kwargs)
    _native.model.eval()
    print("Native model loaded successfully.")
    return _native