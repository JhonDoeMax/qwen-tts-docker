import torch
import os
from transformers import AutoModelForTextToWaveform, AutoTokenizer
from qwen3_tts import Qwen3TTSModel 
import soundfile as sf
import io
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s'
)
logger = logging.getLogger(__name__)


class RequestContextFilter(logging.Filter):
    """Filter for adding request_id to logs"""
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'N/A')
        return True


logger.addFilter(RequestContextFilter())


# Pydantic models for validation
class VoiceAttributes(BaseModel):
    gender: Optional[str] = Field(None, description="Gender of the speaker (male/female/neutral)")
    age: Optional[int] = Field(None, ge=1, le=120, description="Age of the speaker")
    emotion: Optional[str] = Field(None, description="Emotion of the speech (happy/sad/angry/neutral/etc)")

class TTSRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text to synthesize")
    tokens: Optional[List[int]] = Field(None, description="Tokens instead of text")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_length: int = Field(1000, ge=1, le=4096, description="Maximum length")
    voice_attributes: Optional[VoiceAttributes] = Field(None, description="Voice characteristics")
    
    @validator('text', 'tokens')
    def check_text_or_tokens(cls, v, values):
        if v is None and values.get('text') is None and values.get('tokens') is None:
            raise ValueError('Either text or tokens must be provided')
        return v


class TTSFullRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Текст для озвучки")
    format: str = Field("wav", regex="^(wav|mp3|ogg)$", description="Формат аудио")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    voice_attributes: Optional[VoiceAttributes] = Field(None, description="Voice characteristics")


# Загрузка конфигурации из переменных окружения
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3-TTS-12Hz-1.7B-Base")
MODEL_BASE_PATH = os.getenv("MODEL_PATH", "/app/models")
DEVICE = os.getenv("DEVICE", "cuda")
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "1") == "1"
SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", "24000"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

# Формирование полного пути к модели
model_path = os.path.join(MODEL_BASE_PATH, MODEL_NAME)

# Семафор для ограничения параллельных запросов
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global model, tokenizer
    
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Flash Attention: {USE_FLASH_ATTENTION}")
    logger.info(f"Sampling Rate: {SAMPLING_RATE}")
    
    try:
        # Загрузка модели с весами из внешнего каталога
        load_kwargs = {
            "device": DEVICE,
            "torch_dtype": torch.float16
        }
        
        if USE_FLASH_ATTENTION:
            logger.info("Using Flash Attention 2")
            load_kwargs["use_flash_attention_2"] = True
        
        model = Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    yield
    
    # Очистка при завершении
    logger.info("Shutting down, cleaning up resources...")
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete")


app = FastAPI(
    title="Qwen-TTS Streaming Service",
    lifespan=lifespan
)


def get_autocast_context():
    """Контекстный менеджер для автоматического приведения типов"""
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    return torch.no_grad()


def prepare_audio_data(audio) -> io.BytesIO:
    """Подготовка аудио данных для ответа"""
    if isinstance(audio, torch.Tensor):
        audio_data = audio.cpu().numpy()
    else:
        audio_data = audio
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, samplerate=SAMPLING_RATE, format='WAV')
    buffer.seek(0)
    return buffer


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "flash_attention": USE_FLASH_ATTENTION,
        "model_loaded": model is not None
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
        logger.error(f"Error listing models: {str(e)}")
        return {"error": str(e)}


@app.post("/stream-tts")
async def stream_tts(request: Request):
    """
    Стриминг текста в речь с поддержкой параметров голоса.
    
    Параметры голоса:
    - gender: пол говорящего (male/female/neutral)
    - age: возраст говорящего (1-120)
    - emotion: эмоция речи (happy/sad/angry/neutral/etc.)
    """
    request_id = str(uuid.uuid4())
    
    async with request_semaphore:
        try:
            data = await request.json()
            validated_data = TTSRequest(**data)
            
            if not validated_data.text and not validated_data.tokens:
                raise HTTPException(status_code=400, detail="Either 'text' or 'tokens' must be provided")
            
            async def generate_audio():
                try:
                    # Обработка текста или токенов
                    if validated_data.tokens:
                        logger.info(f"[{request_id}] Processing {len(validated_data.tokens)} tokens")
                        input_ids = torch.tensor(validated_data.tokens).unsqueeze(0).to(DEVICE)
                    else:
                        text_preview = validated_data.text[:50] if validated_data.text else ""
                        logger.info(f"[{request_id}] Processing text: {text_preview}...")
                        inputs = tokenizer(validated_data.text, return_tensors="pt")
                        input_ids = inputs.input_ids.to(DEVICE)
                    
                    # Подготовка параметров голоса для генерации
                    generation_kwargs = {
                        "input_ids": input_ids,
                        "max_length": validated_data.max_length,
                        "do_sample": True,
                        "temperature": validated_data.temperature
                    }
                    
                    # Добавление параметров голоса, если они предоставлены
                    if validated_data.voice_attributes:
                        voice_attrs = validated_data.voice_attributes
                        if voice_attrs.gender:
                            generation_kwargs["gender"] = voice_attrs.gender
                        if voice_attrs.age:
                            generation_kwargs["age"] = voice_attrs.age
                        if voice_attrs.emotion:
                            generation_kwargs["emotion"] = voice_attrs.emotion
                    
                    # Генерация аудио с автоматическим приведением типов
                    with get_autocast_context(), torch.no_grad():
                        audio_stream = model.generate_stream(**generation_kwargs)
                    
                    # Стриминг аудио по частям
                    chunk_count = 0
                    for audio_chunk in audio_stream:
                        chunk_count += 1
                        logger.debug(f"[{request_id}] Generating audio chunk {chunk_count}")
                        
                        # Подготовка аудио данных
                        if isinstance(audio_chunk, torch.Tensor):
                            audio_data = audio_chunk.cpu().numpy()
                        else:
                            audio_data = audio_chunk
                        
                        # Конвертация в WAV формат
                        buffer = io.BytesIO()
                        sf.write(buffer, audio_data, samplerate=SAMPLING_RATE, format='WAV')
                        buffer.seek(0)
                        yield buffer.read()
                    
                    logger.info(f"[{request_id}] Audio generation complete. Total chunks: {chunk_count}")
                    
                except Exception as e:
                    logger.error(f"[{request_id}] Error during audio generation: {str(e)}")
                    # Отправляем ошибку клиенту через специальный формат
                    error_buffer = io.BytesIO()
                    error_buffer.write(f"ERROR: {str(e)}".encode())
                    error_buffer.seek(0)
                    yield error_buffer.read()
            
            return StreamingResponse(
                generate_audio(), 
                media_type="audio/wav",
                headers={"X-Request-ID": request_id}
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Error in stream_tts: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "request_id": request_id}
            )


@app.post("/tts")
async def tts(request: Request):
    """
    Полная генерация аудио (не стриминг) с поддержкой параметров голоса.
    
    Параметры голоса:
    - gender: пол говорящего (male/female/neutral)
    - age: возраст говорящего (1-120)
    - emotion: эмоция речи (happy/sad/angry/neutral/etc.)
    """
    request_id = str(uuid.uuid4())
    
    async with request_semaphore:
        try:
            data = await request.json()
            validated_data = TTSFullRequest(**data)
            
            logger.info(f"[{request_id}] Generating full audio for text: {validated_data.text[:50]}...")
            
            inputs = tokenizer(validated_data.text, return_tensors="pt").to(DEVICE)
            
            # Подготовка параметров голоса для генерации
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "max_length": 1000,
                "do_sample": True,
                "temperature": validated_data.temperature
            }
            
            # Добавление параметров голоса, если они предоставлены
            if validated_data.voice_attributes:
                voice_attrs = validated_data.voice_attributes
                if voice_attrs.gender:
                    generation_kwargs["gender"] = voice_attrs.gender
                if voice_attrs.age:
                    generation_kwargs["age"] = voice_attrs.age
                if voice_attrs.emotion:
                    generation_kwargs["emotion"] = voice_attrs.emotion
            
            # Генерация аудио
            with get_autocast_context(), torch.no_grad():
                audio = model.generate(**generation_kwargs)
            
            # Подготовка аудио
            buffer = prepare_audio_data(audio)
            
            # Определение media_type
            if validated_data.format == "wav":
                media_type = "audio/wav"
            elif validated_data.format == "mp3":
                # TODO: Реализовать конвертацию в MP3 через ffmpeg или pydub
                # Пока возвращаем WAV с предупреждением
                logger.warning(f"[{request_id}] MP3 format requested but not implemented, returning WAV")
                media_type = "audio/wav"
            elif validated_data.format == "ogg":
                # TODO: Реализовать конвертацию в OGG
                logger.warning(f"[{request_id}] OGG format requested but not implemented, returning WAV")
                media_type = "audio/wav"
            else:
                media_type = "audio/wav"
            
            logger.info(f"[{request_id}] Full audio generation complete")
            
            return StreamingResponse(
                iter([buffer.read()]), 
                media_type=media_type,
                headers={"X-Request-ID": request_id}
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Error in tts: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "request_id": request_id}
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Error in tts: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "request_id": request_id}
            )


@app.get("/metrics")
async def metrics():
    """Метрики сервиса"""
    import psutil
    import torch
    
    metrics_data = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "active_requests": MAX_CONCURRENT_REQUESTS - request_semaphore._value,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
    }
    
    if torch.cuda.is_available():
        metrics_data["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
        metrics_data["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
    
    return metrics_data


# Примеры использования API:
# 
# 1. Простой запрос с параметрами голоса:
# POST /stream-tts
# {
#     "text": "Привет, мир!",
#     "voice_attributes": {
#         "gender": "female",
#         "age": 25,
#         "emotion": "happy"
#     }
# }
#
# 2. Полная генерация с параметрами голоса:
# POST /tts
# {
#     "text": "Привет, мир!",
#     "format": "wav",
#     "temperature": 0.7,
#     "voice_attributes": {
#         "gender": "male",
#         "age": 30,
#         "emotion": "neutral"
#     }
# }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8188, log_level="info")
