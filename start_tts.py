import torch
import os
from transformers import AutoModelForTextToWaveform, AutoTokenizer
from qwen3_tts import Qwen3TTSModel 
import soundfile as sf
import io
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen-TTS Streaming Service")

# Загрузка конфигурации из переменных окружения
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3-TTS-12Hz-1.7B-Base")
MODEL_BASE_PATH = os.getenv("MODEL_PATH", "/app/models")
DEVICE = os.getenv("DEVICE", "cuda")
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "1") == "1"
SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", "24000"))

# Формирование полного пути к модели
model_path = os.path.join(MODEL_BASE_PATH, MODEL_NAME)

logger.info(f"Loading model from: {model_path}")
logger.info(f"Device: {DEVICE}")
logger.info(f"Flash Attention: {USE_FLASH_ATTENTION}")
logger.info(f"Sampling Rate: {SAMPLING_RATE}")

try:
    # Загрузка модели с весами из внешнего каталога
    if USE_FLASH_ATTENTION:
        logger.info("Using Flash Attention 2")
        model = Qwen3TTSModel.from_pretrained(
            model_path, 
            device=DEVICE, 
            use_flash_attention_2=True,
            torch_dtype=torch.float16
        )
    else:
        model = Qwen3TTSModel.from_pretrained(
            model_path, 
            device=DEVICE,
            torch_dtype=torch.float16
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Model loaded successfully!")
    
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "flash_attention": USE_FLASH_ATTENTION
    }

@app.get("/models")
async def list_models():
    """Список доступных моделей в директории"""
    try:
        models = []
        if os.path.exists(MODEL_BASE_PATH):
            for item in os.listdir(MODEL_BASE_PATH):
                item_path = os.path.join(MODEL_BASE_PATH, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                    models.append(item)
        
        return {
            "available_models": models,
            "current_model": MODEL_NAME,
            "model_path": model_path
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/stream-tts")
async def stream_tts(request: Request):
    """
    Стриминг текста в речь.
    
    Пример запроса:
    {
        "text": "Привет! Это тестовая озвучка.",
        "tokens": [1, 2, 3],  # Опционально: токены вместо текста
        "temperature": 0.7,
        "max_length": 1000
    }
    """
    try:
        data = await request.json()
        text = data.get("text", "")
        tokens = data.get("tokens", [])
        temperature = float(data.get("temperature", 0.7))
        max_length = int(data.get("max_length", 1000))
        
        if not text and not tokens:
            raise HTTPException(status_code=400, detail="Either 'text' or 'tokens' must be provided")
        
        async def generate_audio():
            try:
                # Обработка текста или токенов
                if tokens:
                    logger.info(f"Processing {len(tokens)} tokens")
                    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
                else:
                    logger.info(f"Processing text: {text[:50]}...")
                    inputs = tokenizer(text, return_tensors="pt")
                    input_ids = inputs.input_ids.to(DEVICE)
                
                # ⚠️ ПРОБЛЕМА: Проверка устройства для autocast
                if DEVICE == "cuda":
                    with torch.cuda.amp.autocast():
                        audio_stream = model.generate_stream(
                            input_ids=input_ids,
                            max_length=max_length,
                            do_sample=True,
                            temperature=temperature
                        )
                else:
                    audio_stream = model.generate_stream(
                        input_ids=input_ids,
                        max_length=max_length,
                        do_sample=True,
                        temperature=temperature
                    )
                    
                # Стриминг аудио по частям
                chunk_count = 0
                for audio_chunk in audio_stream:
                    chunk_count += 1
                    logger.debug(f"Generating audio chunk {chunk_count}")
                    
                    # ⚠️ ПРОБЛЕМА: Проверка типа данных
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_data = audio_chunk.cpu().numpy()
                    else:
                        audio_data = audio_chunk
                    
                    # Конвертация в WAV формат
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_data, samplerate=SAMPLING_RATE, format='WAV')
                    buffer.seek(0)
                    yield buffer.read()
                
                logger.info(f"Audio generation complete. Total chunks: {chunk_count}")
                
            except Exception as e:
                logger.error(f"Error during audio generation: {str(e)}")
                raise
        
        return StreamingResponse(generate_audio(), media_type="audio/wav")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stream_tts: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/tts")
async def tts(request: Request):
    """
    Полная генерация аудио (не стриминг).
    
    Пример запроса:
    {
        "text": "Привет! Это тестовая озвучка.",
        "format": "wav"  # wav, mp3, ogg
    }
    """
    try:
        data = await request.json()
        text = data.get("text", "")
        audio_format = data.get("format", "wav")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        logger.info(f"Generating full audio for text: {text[:50]}...")
        
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        
        # ⚠️ ПРОБЛЕМА: Проверка устройства для autocast
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    audio = model.generate(
                        input_ids=inputs.input_ids,
                        max_length=1000,
                        do_sample=True,
                        temperature=0.7
                    )
        else:
            with torch.no_grad():
                audio = model.generate(
                    input_ids=inputs.input_ids,
                    max_length=1000,
                    do_sample=True,
                    temperature=0.7
                )
        
        # ⚠️ ПРОБЛЕМА: Проверка типа данных
        if isinstance(audio, torch.Tensor):
            audio_data = audio.cpu().numpy()
        else:
            audio_data = audio
        
        # Конвертация в нужный формат
        buffer = io.BytesIO()
        
        if audio_format == "wav":
            sf.write(buffer, audio_data, samplerate=SAMPLING_RATE, format='WAV')
            media_type = "audio/wav"
        elif audio_format == "mp3":
            # ⚠️ ПРОБЛЕМА: Нужна конвертация в MP3, а не просто сохранение WAV
            sf.write(buffer, audio_data, samplerate=SAMPLING_RATE, format='WAV')
            media_type = "audio/mpeg"
            # Для настоящего MP3 нужен дополнительный код:
            # import subprocess
            # subprocess.run(['ffmpeg', '-i', '-', '-f', 'mp3', '-'], ...)
        else:
            sf.write(buffer, audio_data, samplerate=SAMPLING_RATE, format='WAV')
            media_type = "audio/wav"
        
        buffer.seek(0)
        
        logger.info("Full audio generation complete")
        
        return StreamingResponse(iter([buffer.read()]), media_type=media_type)
    
    except Exception as e:
        logger.error(f"Error in tts: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8188, log_level="info")