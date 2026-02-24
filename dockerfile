FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_FLASH_ATTN=1
ARG INSTALL_FROM_SOURCE=0

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    sox libsox-dev libsox-fmt-all \
    ffmpeg libsndfile1-dev \
    git curl wget \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Установка PyTorch с CUDA
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Установка Flash Attention 2
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
        pip3 install flash-attn --no-build-isolation; \
    fi

# Установка Qwen-TTS и зависимостей
RUN pip3 install transformers accelerate diffusers soundfile librosa \
    gradio huggingface_hub sentencepiece

# Установка Qwen3-TTS
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /app/qwen3-tts && \
    cd /app/qwen3-tts && \
    pip3 install -e .

WORKDIR /app

# Создание директории для весов (будет монтироваться отдельно)
RUN mkdir -p /app/models

# Копирование скриптов для запуска
COPY start_tts.py /app/
COPY requirements.txt /app/

EXPOSE 8000

CMD ["python3", "start_tts.py"]