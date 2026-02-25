# Qwen-TTS Streaming Service

Сервис потокового синтеза речи на основе модели Qwen3-TTS-12Hz-1.7B-CustomVoice с поддержкой параметров голоса (пол, возраст, эмоции).

## Возможности

- 🎙️ **Поддержка параметров голоса**: пол (male/female/neutral), возраст (1-120), эмоции (happy/sad/angry/neutral)
- 🌊 **Потоковая генерация аудио**: низкая задержка, стриминг по частям
- 🚀 **Полная генерация**: один запрос - полный аудиофайл
- 📊 **Метрики**: мониторинг ресурсов и загрузки
- 🔧 **Легкая настройка**: через переменные окружения

## Быстрый старт

### 1. Подготовка моделей

Скачайте модель Qwen3-TTS-12Hz-1.7B-CustomVoice и поместите в директорию `models/`:

```bash
mkdir -p models
cd models
# Скачайте модель с HuggingFace
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

### 2. Запуск через Docker Compose

```bash
docker-compose up -d
```

### 3. Проверка работоспособности

```bash
python test_api.py
```

## API Endpoints

### Health Check
```http
GET /health
```

### Streaming TTS
```http
POST /stream-tts
Content-Type: application/json

{
    "text": "Привет, мир!",
    "temperature": 0.7,
    "voice_attributes": {
        "gender": "female",
        "age": 25,
        "emotion": "happy"
    }
}
```

### Full TTS
```http
POST /tts
Content-Type: application/json

{
    "text": "Привет, мир!",
    "format": "wav",
    "temperature": 0.7,
    "voice_attributes": {
        "gender": "male",
        "age": 30,
        "emotion": "neutral"
    }
}
```

### Metrics
```http
GET /metrics
```

## Параметры запроса

### VoiceAttributes

| Параметр | Тип | Диапазон | Описание |
|----------|-----|----------|----------|
| gender | string | male, female, neutral | Пол говорящего |
| age | int | 1-120 | Возраст говорящего |
| emotion | string | happy, sad, angry, neutral, etc. | Эмоция речи |

### TTSRequest (для /stream-tts)

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| text | string | Да (или tokens) | Текст для озвучки |
| tokens | array | Да (или text) | Токены вместо текста |
| temperature | float | Нет | Температура сэмплирования (0.0-2.0, default: 0.7) |
| max_length | int | Нет | Максимальная длина (1-4096, default: 1000) |
| voice_attributes | object | Нет | Параметры голоса |

### TTSFullRequest (для /tts)

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| text | string | Да | Текст для озвучки |
| format | string | Нет | Формат аудио (wav, mp3, ogg, default: wav) |
| temperature | float | Нет | Температура сэмплирования (0.0-2.0, default: 0.7) |
| voice_attributes | object | Нет | Параметры голоса |

## Конфигурация

### Переменные окружения

| Переменная | Значение по умолчанию | Описание |
|------------|----------------------|----------|
| MODEL_NAME | Qwen3-TTS-12Hz-1.7B-CustomVoice | Имя модели |
| MODEL_PATH | /app/models | Путь к моделям |
| DEVICE | cuda | Устройство (cuda/cpu) |
| SAMPLING_RATE | 24000 | Частота дискретизации |
| MAX_CONCURRENT_REQUESTS | 10 | Максимум параллельных запросов |
| USE_FLASH_ATTENTION | 1 | Использование Flash Attention (1/0) |

## Примеры использования

### curl

```bash
# Streaming TTS
curl -X POST http://localhost:8188/stream-tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Привет, мир!",
    "voice_attributes": {
      "gender": "female",
      "age": 25,
      "emotion": "happy"
    }
  }' \
  --output output.wav

# Full TTS
curl -X POST http://localhost:8188/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Привет, мир!",
    "format": "wav",
    "voice_attributes": {
      "gender": "male",
      "age": 30,
      "emotion": "neutral"
    }
  }' \
  --output output.wav
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8188/stream-tts",
    json={
        "text": "Привет, мир!",
        "voice_attributes": {
            "gender": "female",
            "age": 25,
            "emotion": "happy"
        }
    },
    stream=True
)

with open("output.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

## Лицензия

MIT License