# utils.py
import base64
import io
import subprocess
import numpy as np
import torch


def encode_audio(audio_np: np.ndarray, sample_rate: int, audio_format: str = "opus") -> bytes:
    """Encode audio numpy array to the specified format."""
    if audio_format == "opus":
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_np, samplerate=sample_rate, format="WAV")
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                "pipe:0",
                "-c:a",
                "libopus",
                "-b:a",
                "48k",
                "-ar",
                "48000",
                "-ac",
                "1",
                "-f",
                "ogg",
                "pipe:1",
            ],
            input=wav_buf.getvalue(),
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg opus encode failed: {proc.stderr.decode()}")
        return proc.stdout

    buf = io.BytesIO()
    sf.write(buf, audio_np, samplerate=sample_rate, format=audio_format.upper())
    return buf.getvalue()


def _build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\\n<|im_start|>assistant\n"


def _build_ref_text(text: str) -> str:
    return f"\\n<|im_start|>assistant\n{text}<|im_end|>\\n"


def _build_instruct_text(instruct: str) -> str:
    return f"\\n<|im_start|>user\n{instruct}<|im_end|>\\n"


def _pcm_s16le_base64(wav: torch.Tensor) -> tuple[str, int]:
    """Convert float waveform [-1,1] to PCM s16le base64. Returns (b64, num_samples)."""
    wav = wav.detach().to(torch.float32).clamp(-1, 1).cpu().numpy()
    pcm = (wav * 32767.0).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii"), int(pcm.shape[0])


def _apply_repetition_penalty_(logits: torch.Tensor, token_ids: list[int], penalty: float):
    if penalty is None or penalty == 1.0 or not token_ids:
        return
    unique = set(token_ids)
    for token_id in unique:
        score = logits[token_id]
        if score < 0:
            logits[token_id] = score * penalty
        else:
            logits[token_id] = score / penalty


def _top_k_top_p_filter_(logits: torch.Tensor, top_k: int, top_p: float):
    vocab = logits.numel()

    if top_k is not None and top_k > 0 and top_k < vocab:
        values, _ = torch.topk(logits, top_k)
        min_keep = values[-1]
        logits[logits < min_keep] = -float("inf")

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)

        mask = cum > top_p
        mask[0] = False
        sorted_logits[mask] = -float("inf")

        logits.fill_(-float("inf"))
        logits.scatter_(0, sorted_indices, sorted_logits)


def _sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generated_token_ids: list[int],
    suppress_token_ids: torch.Tensor | None,
) -> int:
    logits = logits.to(torch.float32)

    if suppress_token_ids is not None and suppress_token_ids.numel() > 0:
        logits[suppress_token_ids] = -float("inf")

    _apply_repetition_penalty_(logits, generated_token_ids, repetition_penalty)

    if temperature is not None and temperature > 0 and temperature != 1.0:
        logits = logits / float(temperature)

    _top_k_top_p_filter_(logits, int(top_k or 0), float(top_p if top_p is not None else 1.0))

    if not do_sample:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    return int(next_id.item())