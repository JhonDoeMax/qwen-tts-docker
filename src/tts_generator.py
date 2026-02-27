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

