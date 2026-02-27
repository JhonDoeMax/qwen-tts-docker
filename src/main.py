"""Qwen3-TTS server with streaming PCM output (native HF/PyTorch) for Uvicorn deployment.

Streaming mode:
  - Incrementally generates codec tokens from the Talker
  - Incrementally decodes to waveform using the 12Hz speech tokenizer decoder (with left-context)
  - Streams PCM chunks (base64) to the client
"""

import base64
import io
import subprocess
import time
import uuid
from typing import Optional, Dict, Any, Generator, List

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Torch version note (no hard fail; just warn for older runtimes)
# ---------------------------------------------------------------------------

def _torch_version_tuple() -> tuple[int, int, int]:
    # torch.__version__ can be like "2.10.0+cu124"
    v = torch.__version__.split("+", 1)[0]
    parts = v.split(".")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return (0, 0, 0)


if _torch_version_tuple() < (2, 10, 0):
    print(
        f"[WARN] handler_streaming is tuned for torch>=2.10.0; detected torch=={torch.__version__}. "
        "Streaming may still work, but performance/compat may be worse."
    )

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen")
MODEL_MAP = {
    "CustomVoice": f"{MODEL_PATH}/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "VoiceDesign": f"{MODEL_PATH}Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Base": f"{MODEL_PATH}/Qwen3-TTS-12Hz-1.7B-Base",
}

TASK_TYPE = os.environ.get("QWEN3_TTS_TASK_TYPE", "CustomVoice")
MODEL_NAME = os.environ.get("QWEN3_TTS_MODEL", MODEL_MAP.get(TASK_TYPE, MODEL_MAP["CustomVoice"]))

INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "270"))  # seconds
STREAM_CHUNK_TOKENS = int(os.environ.get("STREAM_CHUNK_TOKENS", "24"))  # ~2s at 12Hz
STREAM_LEFT_CONTEXT_TOKENS = int(os.environ.get("STREAM_LEFT_CONTEXT_TOKENS", "25"))

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

CONTENT_TYPE_MAP = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/ogg; codecs=opus",
    "mp3": "audio/mpeg",
}

EXT_MAP = {
    "opus": "ogg",  # opus in ogg container
}


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
    suppress_token_ids: Optional[torch.Tensor],
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


def _build_prefill(
    qwen_model,
    input_id: torch.Tensor,
    *,
    instruct_id: Optional[torch.Tensor],
    ref_id: Optional[torch.Tensor],
    voice_clone_prompt: Optional[dict],
    language: str,
    speaker: Optional[str],
    non_streaming_mode: bool,
):
    """Create Talker prefill embeddings/masks for a single sample, following upstream generate()."""
    talker = qwen_model.talker

    pieces = []

    # Speaker embedding (CustomVoice / VoiceClone)
    voice_clone_spk_embeds = None
    if voice_clone_prompt is not None:
        voice_clone_spk_embeds = qwen_model.generate_speaker_prompt(voice_clone_prompt)

    if instruct_id is not None:
        pieces.append(talker.text_projection(talker.get_text_embeddings()(instruct_id)))

    if voice_clone_spk_embeds is None:
        if speaker == "" or speaker is None:
            speaker_embed = None
        else:
            if speaker.lower() not in qwen_model.config.talker_config.spk_id:
                raise NotImplementedError(f"Speaker {speaker} not implemented")
            spk_id = qwen_model.config.talker_config.spk_id[speaker.lower()]
            speaker_embed = talker.get_input_embeddings()(
                torch.tensor(spk_id, device=talker.device, dtype=input_id.dtype)
            )
    else:
        if bool(voice_clone_prompt["x_vector_only_mode"][0]) or bool(voice_clone_prompt["icl_mode"][0]):
            speaker_embed = voice_clone_spk_embeds[0]
        else:
            speaker_embed = None

    # Language token id / dialect handling
    language_id = None
    if language is None:
        raise ValueError("language must be provided")
    if language.lower() != "auto":
        if language.lower() not in qwen_model.config.talker_config.codec_language_id:
            raise NotImplementedError(f"Language {language} not implemented")
        language_id = qwen_model.config.talker_config.codec_language_id[language.lower()]

    if (
        language.lower() in ["chinese", "auto"]
        and speaker not in ("", None)
        and qwen_model.config.talker_config.spk_is_dialect[speaker.lower()] is not False
    ):
        dialect = qwen_model.config.talker_config.spk_is_dialect[speaker.lower()]
        language_id = qwen_model.config.talker_config.codec_language_id[dialect]

    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [[qwen_model.config.tts_bos_token_id, qwen_model.config.tts_eos_token_id, qwen_model.config.tts_pad_token_id]],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)

    if language_id is None:
        codec_prefill_list = [[
            qwen_model.config.talker_config.codec_nothink_id,
            qwen_model.config.talker_config.codec_think_bos_id,
            qwen_model.config.talker_config.codec_think_eos_id,
        ]]
    else:
        codec_prefill_list = [[
            qwen_model.config.talker_config.codec_think_id,
            qwen_model.config.talker_config.codec_think_bos_id,
            language_id,
            qwen_model.config.talker_config.codec_think_eos_id,
        ]]

    codec_input_embedding_0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=talker.device, dtype=input_id.dtype)
    )
    codec_input_embedding_1 = talker.get_input_embeddings()(
        torch.tensor(
            [[qwen_model.config.talker_config.codec_pad_id, qwen_model.config.talker_config.codec_bos_id]],
            device=talker.device,
            dtype=input_id.dtype,
        )
    )

    if speaker_embed is None:
        codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)
    else:
        codec_input_embedding = torch.cat([codec_input_embedding_0, speaker_embed.view(1, 1, -1), codec_input_embedding_1], dim=1)

    role_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    prefill_embed = torch.cat(
        (
            tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
            tts_bos_embed,
        ),
        dim=1,
    ) + codec_input_embedding[:, :-1]

    talker_input_embed = torch.cat((role_embed, prefill_embed), dim=1)

    # ICL prompt for Base (voice clone)
    if (
        voice_clone_prompt is not None
        and voice_clone_prompt.get("ref_code", None) is not None
        and voice_clone_prompt["ref_code"][0] is not None
        and bool(voice_clone_prompt["icl_mode"][0])
    ):
        if ref_id is None:
            raise ValueError("ref_text/ref_id is required for Base task in ICL mode (x_vector_only_mode=False)")
        icl_input_embed, trailing_text_hidden = qwen_model.generate_icl_prompt(
            text_id=input_id[:, 3:-5],
            ref_id=ref_id[:, 3:-2],
            ref_code=voice_clone_prompt["ref_code"][0].to(talker.device),
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        )
        talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
    else:
        # First text token
        talker_input_embed = torch.cat(
            [
                talker_input_embed,
                talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_embedding[:, -1:],
            ],
            dim=1,
        )

        if non_streaming_mode:
            talker_input_embed = talker_input_embed[:, :-1]
            text_embed = torch.cat(
                (talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:-5])), tts_eos_embed),
                dim=1,
            )
            text_embed = text_embed + talker.get_input_embeddings()(
                torch.tensor(
                    [[qwen_model.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                    device=talker.device,
                    dtype=input_id.dtype,
                )
            )
            bos_embed = tts_pad_embed + talker.get_input_embeddings()(
                torch.tensor([[qwen_model.config.talker_config.codec_bos_id]], device=talker.device, dtype=input_id.dtype)
            )
            talker_input_embed = torch.cat([talker_input_embed, text_embed, bos_embed], dim=1)
            trailing_text_hidden = tts_pad_embed
        else:
            trailing_text_hidden = torch.cat(
                (talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5])), tts_eos_embed),
                dim=1,
            )

    pieces.append(talker_input_embed)
    talker_input_embeds = torch.cat(pieces, dim=1)
    attention_mask = torch.ones((1, talker_input_embeds.shape[1]), device=talker_input_embeds.device, dtype=torch.long)

    return talker_input_embeds, attention_mask, trailing_text_hidden, tts_pad_embed


# ---------------------------------------------------------------------------
# Native model (lazy load)
# ---------------------------------------------------------------------------

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
            torch.set_float32_matmul_precision(os.environ.get("TORCH_MATMUL_PRECISION", "high"))
        except Exception:
            pass
        # Best-effort SDPA backend preference (does not affect FlashAttention2).
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

    attn_impl = os.environ.get("QWEN3_TTS_ATTN_IMPLEMENTATION")  # e.g. "flash_attention_2" | "sdpa" | "eager"

    print(f"Loading native model: {MODEL_PATH} (task_type={TASK_TYPE}, device={device_map})")
    load_kwargs = dict(
        device_map=device_map,
        dtype=torch_dtype,
    )
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    _native = Qwen3TTSModel.from_pretrained(MODEL_PATH, **load_kwargs)
    _native.model.eval()
    print("Native model loaded successfully.")
    return _native


# ---------------------------------------------------------------------------
# Uvicorn-compatible handler
# ---------------------------------------------------------------------------

def generate_tts(params: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Streaming-capable generator for TTS processing.
    
    Input additions:
      - stream: bool (default false)
      - stream_chunk_tokens: int (default env STREAM_CHUNK_TOKENS)
      - stream_left_context_tokens: int (default env STREAM_LEFT_CONTEXT_TOKENS)

    Streaming chunk output:
      {"chunk_index": i, "audio_format":"pcm_s16le", "sample_rate":24000, "audio_base64":"...", "done": false}
    Final output:
      {"chunk_index": i, "duration_seconds":3.52, "done": true}
    """
    job_id = str(uuid.uuid4())
    start_t = time.time()

    def _fail(msg: str):
        yield {"error": msg, "done": True}

    if "text" not in params:
        yield from _fail("Missing required field: text")
        return

    task_type = params.get("task_type", TASK_TYPE)
    if task_type != TASK_TYPE:
        yield from _fail(f"task_type mismatch: request={task_type}, endpoint={TASK_TYPE}")
        return

    stream = bool(params.get("stream", False))

    # Non-streaming fallback (native) for callers that still want a single response.
    if not stream:
        try:
            native = _get_native_model()
            text = params["text"]
            language = params.get("language", "Auto")
            output_format = params.get("output_format", "opus")

            gen_kwargs = dict(
                do_sample=bool(params.get("do_sample", True)),
                top_k=int(params.get("top_k", 50)),
                top_p=float(params.get("top_p", 1.0)),
                temperature=float(params.get("temperature", 0.9)),
                repetition_penalty=float(params.get("repetition_penalty", 1.05)),
                max_new_tokens=int(params.get("max_new_tokens", 2048)),
            )

            if task_type == "CustomVoice":
                speaker = params.get("speaker", "Vivian")
                instruct = params.get("instruct", "")
                wavs, sr = native.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                    non_streaming_mode=True,
                    **gen_kwargs,
                )
            elif task_type == "VoiceDesign":
                instruct = params.get("instruct", "")
                if instruct == "":
                    raise ValueError("Missing required field 'instruct' for VoiceDesign task")
                wavs, sr = native.generate_voice_design(
                    text=text,
                    instruct=instruct,
                    language=language,
                    non_streaming_mode=True,
                    **gen_kwargs,
                )
            else:  # Base
                ref_audio = params.get("ref_audio")
                if not ref_audio:
                    raise ValueError("Missing required field 'ref_audio' for Base (voice clone) task")
                ref_text = params.get("ref_text", "")
                x_vector_only_mode = bool(params.get("x_vector_only_mode", False))
                wavs, sr = native.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    non_streaming_mode=False,
                    **gen_kwargs,
                )

            audio_np = wavs[0].astype(np.float32)
            duration = float(audio_np.shape[0] / max(int(sr), 1))
            
            # Encode audio directly for response
            audio_bytes = encode_audio(audio_np, int(sr), output_format)
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            result = {
                "audio_data": audio_base64,
                "sample_rate": int(sr),
                "format": output_format,
                "duration_seconds": round(duration, 2),
            }

            if bool(params.get("timestamps", False)):
                try:
                    result["words"] = forced_align(audio_np, int(sr), text)
                except Exception as e:
                    result["timestamps_error"] = str(e)

            yield result
            return
        except Exception as e:
            yield from _fail(str(e))
            return

    # Streaming mode (native).
    try:
        native = _get_native_model()
        qwen_model = native.model

        text = params["text"]
        language = params.get("language", "Auto")
        output_format = params.get("output_format", "opus")

        if "seed" in params and params["seed"] is not None:
            torch.manual_seed(int(params["seed"]))

        do_sample = bool(params.get("do_sample", True))
        top_k = int(params.get("top_k", 50))
        top_p = float(params.get("top_p", 1.0))
        temperature = float(params.get("temperature", 0.9))
        repetition_penalty = float(params.get("repetition_penalty", 1.05))
        max_new_tokens = int(params.get("max_new_tokens", 2048))

        subtalker_dosample = bool(params.get("subtalker_dosample", True))
        subtalker_top_k = int(params.get("subtalker_top_k", 50))
        subtalker_top_p = float(params.get("subtalker_top_p", 1.0))
        subtalker_temperature = float(params.get("subtalker_temperature", 0.9))

        chunk_tokens = int(params.get("stream_chunk_tokens", STREAM_CHUNK_TOKENS))
        left_ctx = int(params.get("stream_left_context_tokens", STREAM_LEFT_CONTEXT_TOKENS))
        chunk_tokens = max(1, chunk_tokens)
        left_ctx = max(0, left_ctx)

        # Build ids/prompt
        input_id = native._tokenize_texts([_build_assistant_text(text)])[0]

        instruct_id = None
        speaker = None
        ref_id = None
        voice_clone_prompt = None
        non_streaming_mode = True

        decode_prefix_codes = None  # (T,Q) codes used only as decode context (e.g., Base ICL)

        if task_type == "CustomVoice":
            speaker = params.get("speaker", "Vivian")
            instruct = params.get("instruct", "")
            if instruct:
                instruct_id = native._tokenize_texts([_build_instruct_text(instruct)])[0]
            non_streaming_mode = True
        elif task_type == "VoiceDesign":
            instruct = params.get("instruct", "")
            if instruct == "":
                raise ValueError("Missing required field 'instruct' for VoiceDesign task")
            instruct_id = native._tokenize_texts([_build_instruct_text(instruct)])[0]
            non_streaming_mode = True
        else:  # Base
            ref_audio = params.get("ref_audio")
            if not ref_audio:
                raise ValueError("Missing required field 'ref_audio' for Base (voice clone) task")
            ref_text = params.get("ref_text", "")
            x_vector_only_mode = bool(params.get("x_vector_only_mode", False))
            non_streaming_mode = False

            prompt_items = native.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            voice_clone_prompt = native._prompt_items_to_voice_clone_prompt(prompt_items)
            if prompt_items and prompt_items[0].ref_text:
                ref_id = native._tokenize_texts([_build_ref_text(prompt_items[0].ref_text)])[0]

            ref_code_list = voice_clone_prompt.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[0] is not None:
                decode_prefix_codes = ref_code_list[0].to(qwen_model.talker.device)

        # Prefill
        talker_input_embeds, attention_mask, trailing_text_hidden, tts_pad_embed = _build_prefill(
            qwen_model,
            input_id.to(qwen_model.talker.device),
            instruct_id=instruct_id.to(qwen_model.talker.device) if instruct_id is not None else None,
            ref_id=ref_id.to(qwen_model.talker.device) if ref_id is not None else None,
            voice_clone_prompt=voice_clone_prompt,
            language=language,
            speaker=speaker,
            non_streaming_mode=non_streaming_mode,
        )

        eos_token_id = int(qwen_model.config.talker_config.codec_eos_token_id)
        vocab_size = int(qwen_model.config.talker_config.vocab_size)

        suppress = [
            i
            for i in range(vocab_size - 1024, vocab_size)
            if i not in (eos_token_id,)
        ]
        suppress_ids = torch.tensor(suppress, device=qwen_model.talker.device, dtype=torch.long) if suppress else None
        eos_suppress_ids = None
        if suppress_ids is None:
            eos_suppress_ids = torch.tensor([eos_token_id], device=qwen_model.talker.device, dtype=torch.long)
        else:
            eos_suppress_ids = torch.cat(
                [suppress_ids, torch.tensor([eos_token_id], device=qwen_model.talker.device, dtype=torch.long)],
                dim=0,
            )

        # Access the 12Hz decoder directly for incremental decoding.
        tokenizer_model = qwen_model.speech_tokenizer.model
        decoder = tokenizer_model.decoder
        sample_rate = int(qwen_model.speech_tokenizer.get_output_sample_rate())
        samples_per_code = int(qwen_model.speech_tokenizer.get_decode_upsample_rate())
        if samples_per_code <= 1:
            raise RuntimeError(
                f"Invalid decode_upsample_rate={samples_per_code}. "
                "This streaming handler expects the 12Hz tokenizer (e.g. ~2000 samples/token at 24kHz)."
            )

        # Prefill forward to initialize caches
        with torch.inference_mode():
            out = qwen_model.talker(
                input_ids=None,
                inputs_embeds=talker_input_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                subtalker_dosample=subtalker_dosample,
                subtalker_top_k=subtalker_top_k,
                subtalker_top_p=subtalker_top_p,
                subtalker_temperature=subtalker_temperature,
            )

        past_key_values = out.past_key_values
        past_hidden = out.past_hidden
        generation_step = out.generation_step
        next_logits = out.logits[0, -1, :]

        generated_first_ids: list[int] = []
        generated_codes: list[torch.Tensor] = []

        emitted_tokens = 0
        chunk_index = 0
        wav_segments: list[np.ndarray] = []

        def _maybe_timeout():
            if time.time() - start_t > INFERENCE_TIMEOUT:
                raise TimeoutError(f"Inference timeout after {INFERENCE_TIMEOUT}s")

        def _decode_new_audio(gen_len: int) -> torch.Tensor:
            """Decode only newly available audio (since emitted_tokens) with left-context."""
            nonlocal emitted_tokens

            prefix_len = 0
            if decode_prefix_codes is not None:
                prefix_len = int(decode_prefix_codes.shape[0])
                full_codes = torch.cat([decode_prefix_codes, torch.stack(generated_codes, dim=0)], dim=0)
            else:
                full_codes = torch.stack(generated_codes, dim=0)

            start = max(prefix_len + emitted_tokens - left_ctx, 0)
            end = prefix_len + gen_len
            codes_slice = full_codes[start:end]  # (T, Q)

            codes_t = codes_slice.transpose(0, 1).unsqueeze(0).to(torch.long)  # (1, Q, T)
            wav = decoder(codes_t).squeeze(0).squeeze(0)  # (S,)

            context_tokens = (prefix_len + emitted_tokens) - start
            trim = int(context_tokens * samples_per_code)
            if trim > 0:
                wav = wav[trim:]

            emitted_tokens = gen_len
            return wav

        # Token-by-token generation loop
        with torch.inference_mode():
            for step in range(max_new_tokens):
                _maybe_timeout()

                # Match upstream GenerationMixin behavior: min_new_tokens=2 (suppress EOS early).
                step_suppress_ids = eos_suppress_ids if step < 2 else suppress_ids
                next_id = _sample_next_token(
                    next_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_token_ids=generated_first_ids,
                    suppress_token_ids=step_suppress_ids,
                )

                if next_id == eos_token_id:
                    break

                # Append token and run one generation step to get codec ids + next logits
                tok = torch.tensor([[next_id]], device=qwen_model.talker.device, dtype=torch.long)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                    dim=1,
                )
                cache_position = torch.tensor([past_key_values.get_seq_length()], device=tok.device, dtype=torch.long)

                out = qwen_model.talker(
                    input_ids=tok,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                    past_hidden=past_hidden,
                    generation_step=generation_step,
                    trailing_text_hidden=trailing_text_hidden,
                    tts_pad_embed=tts_pad_embed,
                    cache_position=cache_position,
                    subtalker_dosample=subtalker_dosample,
                    subtalker_top_k=subtalker_top_k,
                    subtalker_top_p=subtalker_top_p,
                    subtalker_temperature=subtalker_temperature,
                )

                codec_ids = out.hidden_states[1]  # (B, Q)
                generated_first_ids.append(next_id)
                generated_codes.append(codec_ids.squeeze(0))

                past_key_values = out.past_key_values
                past_hidden = out.past_hidden
                generation_step = out.generation_step
                next_logits = out.logits[0, -1, :]

                gen_len = len(generated_codes)
                if gen_len - emitted_tokens >= chunk_tokens:
                    wav_new = _decode_new_audio(gen_len)
                    if wav_new.numel() > 0:
                        b64, n = _pcm_s16le_base64(wav_new)
                        wav_segments.append(wav_new.detach().to(torch.float32).cpu().numpy())
                        yield {
                            "chunk_index": chunk_index,
                            "audio_format": "pcm_s16le",
                            "sample_rate": sample_rate,
                            "num_samples": n,
                            "audio_base64": b64,
                            "done": False,
                        }
                        chunk_index += 1

        # Flush remainder
        gen_len = len(generated_codes)
        if gen_len > emitted_tokens:
            wav_new = _decode_new_audio(gen_len)
            if wav_new.numel() > 0:
                b64, n = _pcm_s16le_base64(wav_new)
                wav_segments.append(wav_new.detach().to(torch.float32).cpu().numpy())
                yield {
                    "chunk_index": chunk_index,
                    "audio_format": "pcm_s16le",
                    "sample_rate": sample_rate,
                    "num_samples": n,
                    "audio_base64": b64,
                    "done": False,
                }
                chunk_index += 1

        # Finalize streaming response
        audio_np = np.concatenate(wav_segments) if wav_segments else np.zeros((0,), dtype=np.float32)
        duration = float(audio_np.shape[0] / max(sample_rate, 1))
        
        yield {
            "chunk_index": chunk_index,
            "duration_seconds": round(duration, 2),
            "done": True,
        }
        return
    except Exception as e:
        err = str(e)
        yield {"error": err, "done": True}
        return


# ---------------------------------------------------------------------------
# FastAPI application for Uvicorn
# ---------------------------------------------------------------------------

import json
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import asyncio

# Thread pool for CPU-bound TTS processing (single worker to avoid GPU contention)
executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI()

@app.post("/generate")
async def generate_tts_endpoint(request: Request):
    params = await request.json()
    stream_requested = params.get("stream", False)
    
    if not stream_requested:
        # Non-streaming: run to completion and return single response
        try:
            chunks = []
            for chunk in generate_tts(params):
                chunks.append(chunk)
            
            if not chunks:
                return JSONResponse(
                    {"error": "No response from TTS generator"}, 
                    status_code=500
                )
            
            last_chunk = chunks[-1]
            if "error" in last_chunk:
                return JSONResponse(last_chunk, status_code=500)
            return JSONResponse(last_chunk)
        except Exception as e:
            return JSONResponse(
                {"error": f"Server error: {str(e)}", "done": True},
                status_code=500
            )
    
    # Streaming mode: process in background thread and stream results
    q = Queue()
    
    def run_generator():
        try:
            for chunk in generate_tts(params):
                q.put(chunk)
        except Exception as e:
            q.put({"error": f"Generation error: {str(e)}", "done": True})
        finally:
            q.put(None)  # Sentinel value for stream end

    # Start generator in background thread
    threading.Thread(target=run_generator, daemon=True).start()
    
    async def stream_response():
        while True:
            chunk = await asyncio.get_running_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield json.dumps(chunk) + "\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="application/x-ndjson"  # Newline-delimited JSON
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8188, log_level="info")