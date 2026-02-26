from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    voice: str = "Ryan"
    sample_rate: int = 24000
    chunk_size: int = 1024  # Размер чанка в байтах

# Загрузка модели при старте
model = Qwen3TTSModel.from_pretrained(
    "/app/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    """Потоковая генерация TTS через HTTP chunked encoding"""
    if not request.text.strip():
        raise HTTPException(400, "Empty text")
    
    # Определяем функцию для потоковой генерации
    def audio_generator():
        # Генерируем полное аудио (без потоковой передачи)
        with torch.inference_mode():
            wav, sr = model.generate_custom_voice(
                text=request.text,
                speaker=request.voice,
                sample_rate=request.sample_rate
            )
        
        # Конвертируем в PCM S16LE
        audio_np = wav[0].cpu().numpy()
        pcm = (audio_np * 32767.0).astype(np.int16)
        
        # Потоковая передача чанками
        for i in range(0, len(pcm), request.chunk_size // 2):  # 2 байта на семпл
            chunk = pcm[i:i + request.chunk_size // 2]
            yield chunk.tobytes()

    return StreamingResponse(
        audio_generator(),
        media_type="audio/pcm",
        headers={
            "Content-Disposition": "inline",
            "X-Audio-Format": f"pcm; rate={request.sample_rate}; channels=1",
            "Transfer-Encoding": "chunked"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8188, log_level="info")