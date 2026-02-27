# Qwen3-TTS Streaming Server

A high-performance, Uvicorn-ready TTS server using the **Qwen3-TTS** model with native PyTorch support for **streaming PCM audio output**. This implementation leverages Hugging Face Transformers and `torchaudio` to provide low-latency text-to-speech synthesis with incremental waveform decoding.

This service supports three task types:
- **CustomVoice**: Predefined speaker voices
- **VoiceDesign**: Voice control via natural language instructions
- **Base (Voice Clone)**: Custom voice cloning from reference audio

Streaming mode enables real-time audio generation by sending PCM chunks as they are synthesized, making it suitable for interactive applications.

---

## Features

- ✅ **Streaming PCM Output** – Incremental audio delivery via base64-encoded PCM (`pcm_s16le`)
- 🚀 **Native HF Integration** – Uses standard Hugging Face/PyTorch stack (no external binaries)
- 🔁 **Left-Context Decoding** – Ensures continuity in streamed audio using context overlap
- 📦 **Multiple Audio Formats** – Final output supports `opus`, `wav`, `flac`, `ogg`, `mp3`
- 🌐 **Uvicorn + FastAPI** – Ready for production deployment with async streaming
- ⏱️ **Timeout Protection** – Prevents hanging inference jobs
- 🔊 **12Hz Speech Tokenizer** – High-fidelity autoregressive TTS at ~2000 samples per token (24kHz)

---

## Deployment

```bash
uvicorn main:app --host 0.0.0.0 --port 8188 --workers 1
```

> ⚠️ Only one worker should be used due to GPU memory constraints.

Ensure required packages are installed:

```bash
pip install torch torchaudio transformers soundfile numpy uvicorn fastapi base64
```

Optional (for Opus encoding):
```bash
apt-get install ffmpeg  # or install via conda/brew
```

Set environment variables if needed:

| Variable | Default | Description |
|--------|--------|-----------|
| `MODEL_PATH` | `"Qwen"` | Base path where models are stored |
| `QWEN3_TTS_TASK_TYPE` | `"CustomVoice"` | Task type: `CustomVoice`, `VoiceDesign`, or `Base` |
| `QWEN3_TTS_ATTN_IMPLEMENTATION` | *(auto)* | Attention backend: `flash_attention_2`, `sdpa`, `eager` |
| `INFERENCE_TIMEOUT` | `270` | Max seconds for inference before timeout |
| `STREAM_CHUNK_TOKENS` | `24` | Number of codec tokens per stream chunk (~2 sec) |
| `STREAM_LEFT_CONTEXT_TOKENS` | `25` | Left context tokens to maintain audio coherence |

Model paths expected under `MODEL_PATH/`:
- `Qwen3-TTS-1.7B-CustomVoice`
- `Qwen3-TTS-1.7B-VoiceDesign`
- `Qwen3-TTS-1.7B-Base`

---

## API Endpoint

POST `/tts/generate`

Accepts JSON input with support for both **streaming** and **non-streaming** modes.

### Request Parameters

| Field | Type | Required | Default | Description |
|------|------|---------|--------|------------|
| `text` | string | ✅ Yes | — | The text to synthesize into speech |
| `stream` | boolean | No | `false` | Enable streaming mode (yields partial audio) |
| `task_type` | string | No | `"CustomVoice"` | One of: `CustomVoice`, `VoiceDesign`, `Base` |
| `language` | string | No | `"Auto"` | Input language (e.g., `"Spanish"`, `"zh"`) |
| `output_format` | string | No | `"opus"` | Output format: `wav`, `flac`, `ogg`, `opus`, `mp3` |
| `speaker` | string | Conditional | `"Vivian"` | Speaker name (used in `CustomVoice`) |
| `instruct` | string | Conditional | `""` | Instruction for voice style (used in `VoiceDesign`) |
| `ref_audio` | bytes (base64) | Conditional | — | Reference audio for voice cloning (`Base`) |
| `ref_text` | string | No | `""` | Transcript of reference audio |
| `x_vector_only_mode` | bool | No | `false` | Use only speaker embedding, not content (voice clone) |
| `do_sample` | bool | No | `true` | Whether to sample during generation |
| `top_k` | int | No | `50` | Top-k filtering |
| `top_p` | float | No | `1.0` | Nucleus sampling threshold |
| `temperature` | float | No | `0.9` | Sampling temperature |
| `repetition_penalty` | float | No | `1.05` | Repetition penalty |
| `max_new_tokens` | int | No | `2048` | Maximum generated tokens |
| `seed` | int | No | `None` | Random seed for reproducibility |
| `subtalker_*` | various | No | — | Advanced sampling params for sub-talkers |
| `stream_chunk_tokens` | int | No | env: `24` | Tokens per audio chunk in streaming |
| `stream_left_context_tokens` | int | No | env: `25` | Context size for smooth decoding |
| `timestamps` | bool | No | `false` | Return word timestamps (if supported) |

> 💡 For `VoiceDesign`, `instruct` is required.  
> For `Base` (voice clone), `ref_audio` is required.

---

## Response Format

### Non-Streaming Mode
Returns a single JSON object:
```json
{
  "audio_data": "base64...",
  "sample_rate": 24000,
  "format": "opus",
  "duration_seconds": 5.23,
  "words": [ { "word": "hello", "start": 0.1, "end": 0.6 }, ... ]
}
```

If `timestamps=true`, forced alignment results may be included.

### Streaming Mode
Streams newline-delimited JSON (`application/x-ndjson`) chunks:

#### Audio Chunks
```json
{
  "chunk_index": 0,
  "audio_format": "pcm_s16le",
  "sample_rate": 24000,
  "num_samples": 48000,
  "audio_base64": "pcm_data_here",
  "done": false
}
```

#### Final Chunk
```json
{
  "chunk_index": 5,
  "duration_seconds": 10.45,
  "done": true
}
```

Each `audio_base64` contains raw signed 16-bit little-endian PCM data. Clients must decode and play incrementally.

---

## Example Requests

### 1. Basic Synthesis (Non-Streaming)
```json
{
  "text": "Hello, how are you today?",
  "speaker": "Diego",
  "language": "en"
}
```

### 2. Streaming with Custom Voice
```json
{
  "text": "Bienvenidos al sistema de voz en tiempo real.",
  "stream": true,
  "speaker": "Luna",
  "language": "es",
  "temperature": 0.85
}
```

### 3. Voice Design via Instruction
```json
{
  "task_type": "VoiceDesign",
  "text": "I'm excited to show you this new feature!",
  "instruct": "a young female voice, cheerful and energetic",
  "language": "en"
}
```

### 4. Voice Cloning (Base Model)
```json
{
  "task_type": "Base",
  "text": "This is my synthesized voice.",
  "ref_audio": "base64_encoded_wav_data",
  "ref_text": "This is my real voice.",
  "x_vector_only_mode": false
}
```

---

## Performance Notes

- **Latency**: Streaming begins within ~2 seconds (configurable via `STREAM_CHUNK_TOKENS`)
- **Memory**: Model loads fully into GPU VRAM; ensure sufficient memory (~10–15 GB for bfloat16)
- **Audio Quality**: 24kHz output with high fidelity via 12Hz residual vector quantization decoder
- **Throughput**: Optimized for single concurrent request; use load balancing for scale

---

## License

Refer to the original [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) repository for licensing details.

--- 

> Developed for seamless integration into real-time voice systems, IVR platforms, and AI agents requiring natural, expressive speech synthesis with minimal latency.