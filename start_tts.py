import torch
from transformers import AutoModelForTextToWaveform, AutoTokenizer
from qwen3_tts import Qwen3TTS
import soundfile as sf
import io
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

# Загрузка модели с весами из внешнего каталога
model_path = "/app/models/Qwen3-TTS-12Hz-1.7B-Base"
model = Qwen3TTS.from_pretrained(model_path, device="cuda", use_flash_attention_2=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

@app.post("/stream-tts")
async def stream_tts(request: Request):
    """Стриминг текста в речь"""
    data = await request.json()
    text = data.get("text", "")
    tokens = data.get("tokens", [])
    
    async def generate_audio():
        # Обработка текста или токенов
        if tokens:
            input_ids = torch.tensor(tokens).unsqueeze(0).to("cuda")
        else:
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs.input_ids.to("cuda")
        
        # Генерация аудио с использованием стриминга
        with torch.cuda.amp.autocast():
            audio_stream = model.generate_stream(
                input_ids=input_ids,
                max_length=1000,
                do_sample=True,
                temperature=0.7
            )
            
            # Стриминг аудио по частям
            for audio_chunk in audio_stream:
                # Конвертация в WAV формат
                buffer = io.BytesIO()
                sf.write(buffer, audio_chunk.cpu().numpy(), samplerate=24000, format='WAV')
                buffer.seek(0)
                yield buffer.read()
    
    return StreamingResponse(generate_audio(), media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8188)