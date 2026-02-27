# Qwen3-TTS Service

A high-performance Text-to-Speech (TTS) service based on the **Qwen3-TTS** model. This service supports streaming audio generation, voice cloning, custom voices, and voice design tasks. It is optimized for NVIDIA GPUs using CUDA and Flash Attention 2.

## Features

- **Streaming Support**: Incremental audio generation with low latency (PCM/Opus).
- **Multiple Task Types**:
  - **CustomVoice**: Predefined speakers with optional instructions.
  - **VoiceDesign**: Generate voices based on text instructions.
  - **Base (Voice Clone)**: Clone voices from reference audio.
- **GPU Acceleration**: Built on PyTorch with CUDA 12.8 and Flash Attention 2 support.
- **Flexible Output**: Supports WAV, FLAC, OGG, Opus, and MP3 formats.
- **Dockerized**: Easy deployment using Docker Compose with NVIDIA runtime.

## Prerequisites

- **Docker** & **Docker Compose**
- **NVIDIA GPU** with compatible drivers
- **NVIDIA Container Toolkit** installed on the host machine
- **Model Weights**: You must download the Qwen3-TTS model weights separately and mount them to the container.

## Quick Start

### 1. Clone and Prepare

Ensure you have the model weights downloaded. The service expects models in the `/app/models` directory inside the container.

```bash
# Create a directory for models
mkdir -p ./models

# Place your Qwen3-TTS model weights here (e.g., Qwen3-TTS-12Hz-1.7B-Base)
# Example structure: ./models/Qwen3-TTS-12Hz-1.7B-Base
```

### 2. Build and Run

Use Docker Compose to build and start the service.

```bash
docker-compose up --build
```

The service will be available at `http://localhost:8188`.

## Configuration

You can configure the service using Environment Variables in `docker-compose.yml` or build arguments.

### Build Arguments (`docker-compose.yml`)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `INSTALL_FLASH_ATTN` | `1` | Set to `1` to enable Flash Attention 2, `0` to disable. |

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_PATH` | `/app/models` | Path inside the container where models are stored. |
| `QWEN3_TTS_TASK_TYPE` | `CustomVoice` | Default task type (`CustomVoice`, `VoiceDesign`, `Base`). |
| `QWEN3_TTS_MODEL` | *Auto* | Specific model path override. |
| `QWEN3_TTS_ATTN_IMPLEMENTATION` | `flash_attention_2` | Attention implementation (`flash_attention_2`, `sdpa`, `eager`). |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU ID to use. |
| `SAMPLING_RATE` | `24000` | Audio sampling rate. |
| `INFERENCE_TIMEOUT` | `270` | Timeout for inference in seconds. |
| `STREAM_CHUNK_TOKENS` | `24` | Number of tokens per streaming chunk. |
| `STREAM_LEFT_CONTEXT_TOKENS` | `25` | Left context tokens for streaming decoding. |
| `TORCH_CUDA_ARCH_LIST` | `8.0 8.6...` | CUDA architectures to support. |

## API Reference

### Endpoint

- **URL**: `/tts/generate`
- **Method**: `POST`
- **Content-Type**: `application/json`

### Request Parameters

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `text` | string | **Yes** | - | The text to synthesize. |
| `stream` | boolean | No | `false` | Enable streaming response (NDJSON). |
| `task_type` | string | No | `CustomVoice` | `CustomVoice`, `VoiceDesign`, or `Base`. |
| `language` | string | No | `Auto` | Language code (e.g., `chinese`, `english`, `auto`). |
| `output_format` | string | No | `opus` | Audio format: `wav`, `flac`, `ogg`, `opus`, `mp3`. |
| `speaker` | string | No | `Vivian` | Speaker name (Required for `CustomVoice`). |
| `instruct` | string | No | `""` | Instruction text (Required for `VoiceDesign`, optional for `CustomVoice`). |
| `ref_audio` | string | **Yes*** | - | Reference audio for voice cloning (Required for `Base` task). |
| `ref_text` | string | No | `""` | Text content of the reference audio (for `Base` task). |
| `x_vector_only_mode` | boolean | No | `false` | Use only x-vector for cloning (for `Base` task). |
| `do_sample` | boolean | No | `true` | Enable sampling during generation. |
| `top_k` | int | No | `50` | Top-K sampling parameter. |
| `top_p` | float | No | `1.0` | Top-P (nucleus) sampling parameter. |
| `temperature` | float | No | `0.9` | Sampling temperature. |
| `repetition_penalty` | float | No | `1.05` | Penalty for token repetition. |
| `max_new_tokens` | int | No | `2048` | Maximum number of tokens to generate. |
| `seed` | int | No | `null` | Random seed for reproducibility. |
| `timestamps` | boolean | No | `false` | Return word timestamps (Non-streaming only). |
| `stream_chunk_tokens` | int | No | `24` | Tokens per chunk in streaming mode. |
| `stream_left_context_tokens`| int | No | `25` | Left context tokens for streaming decoding. |
| `subtalker_dosample` | boolean | No | `true` | Enable sampling for subtalker. |
| `subtalker_top_k` | int | No | `50` | Top-K for subtalker. |
| `subtalker_top_p` | float | No | `1.0` | Top-P for subtalker. |
| `subtalker_temperature` | float | No | `0.9` | Temperature for subtalker. |

*\*Required depending on `task_type`.*

### Response Format

#### Non-Streaming (`stream: false`)
Returns a single JSON object upon completion.

```json
{
  "audio_data": "<base64_encoded_audio>",
  "sample_rate": 24000,
  "format": "opus",
  "duration_seconds": 3.52,
  "words": [...] // Optional if timestamps=true
}
```

#### Streaming (`stream: true`)
Returns `application/x-ndjson` (Newline-delimited JSON). Each line is a JSON object.

**Chunk Example:**
```json
{"chunk_index": 0, "audio_format": "pcm_s16le", "sample_rate": 24000, "num_samples": 4800, "audio_base64": "...", "done": false}
```

**Final Chunk Example:**
```json
{"chunk_index": 1, "duration_seconds": 3.52, "done": true}
```

**Error Example:**
```json
{"error": "Error message", "done": true}
```

## Usage Examples

### 1. Custom Voice (Non-Streaming)

```bash
curl -X POST http://localhost:8188/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the Qwen3 TTS service.",
    "task_type": "CustomVoice",
    "speaker": "Vivian",
    "stream": false,
    "output_format": "wav"
  }'
```

### 2. Voice Cloning (Base Task)

```bash
curl -X POST http://localhost:8188/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Cloning this voice successfully.",
    "task_type": "Base",
    "ref_audio": "<base64_encoded_reference_audio>",
    "ref_text": "Text spoken in the reference audio.",
    "stream": false
  }'
```

### 3. Streaming Generation

```bash
curl -X POST http://localhost:8188/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Streaming audio test.",
    "stream": true,
    "output_format": "pcm_s16le"
  }'
```

## Troubleshooting

- **GPU Errors**: Ensure the NVIDIA Container Toolkit is installed and `nvidia-smi` works on the host. Check `CUDA_VISIBLE_DEVICES` in `docker-compose.yml`.
- **Model Not Found**: Verify that the model weights are correctly mounted to `/app/models` inside the container.
- **Flash Attention**: If you encounter errors related to Flash Attention, set `INSTALL_FLASH_ATTN=0` in build args and `QWEN3_TTS_ATTN_IMPLEMENTATION=sdpa` in environment variables.
- **Timeouts**: Increase `INFERENCE_TIMEOUT` in environment variables for long text inputs.

## License

This service wrapper is provided as-is. Please refer to the original Qwen3-TTS repository for model licensing terms.