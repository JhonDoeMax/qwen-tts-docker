# Используем официальный образ PyTorch 2.7.1 с CUDA 12.8
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_FLASH_ATTN=1

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    sox libsox-dev libsox-fmt-all \
    ffmpeg libsndfile1-dev \
    git curl wget \
    build-essential \
    libegl1-mesa libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Обновление pip
RUN pip3 install --upgrade pip setuptools wheel

# Установка зависимостей для Flash Attention 2
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
        pip3 install ninja packaging flash-attn --no-build-isolation; \
    fi

# Установка только нужных для Qwen-TTS зависимостей
RUN pip3 install transformers==4.57.3 accelerate==1.12.0 diffusers==0.29.0 \
    soundfile librosa huggingface-hub fastapi uvicorn numpy sentencepiece==0.2.0
     

# Установка Qwen3-TTS
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /app/qwen3-tts && \
    cd /app/qwen3-tts && \
    pip3 install -e .

WORKDIR /app

# Создание директории для весов (будет монтироваться отдельно)
RUN mkdir -p /app/models

# Копирование скриптов для запуска
COPY start_tts.py /app/

ENV MODEL_NAME=Qwen3-TTS-12Hz-1.7B-Base
ENV MODEL_PATH=/app/models
ENV DEVICE=cuda
ENV SAMPLING_RATE=24000
ENV PYTHONUNBUFFERED=1

ENV USE_FLASH_ATTENTION=${INSTALL_FLASH_ATTN}

EXPOSE 8188

CMD ["python3", "start_tts.py"]