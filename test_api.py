#!/usr/bin/env python3
"""
Тестовый скрипт для проверки Qwen-TTS API с параметрами голоса
"""

import requests
import json
import time

BASE_URL = "http://localhost:8188"

def test_health():
    """Проверка состояния сервиса"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_stream_tts():
    """Тест стримингового TTS с параметрами голоса"""
    print("Testing /stream-tts with voice attributes...")
    
    payload = {
        "text": "Привет! Это тестовое сообщение с параметрами голоса.",
        "temperature": 0.7,
        "voice_attributes": {
            "gender": "female",
            "age": 25,
            "emotion": "happy"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/stream-tts",
        json=payload,
        stream=True
    )
    
    print(f"Status code: {response.status_code}")
    print(f"Request ID: {response.headers.get('X-Request-ID')}")
    
    # Сохраняем аудио
    with open("test_stream_output.wav", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("Audio saved to test_stream_output.wav")
    print()

def test_tts():
    """Тест полного TTS с параметрами голоса"""
    print("Testing /tts with voice attributes...")
    
    payload = {
        "text": "Привет! Это тестовое сообщение с параметрами голоса.",
        "format": "wav",
        "temperature": 0.7,
        "voice_attributes": {
            "gender": "male",
            "age": 30,
            "emotion": "neutral"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/tts",
        json=payload
    )
    
    print(f"Status code: {response.status_code}")
    print(f"Request ID: {response.headers.get('X-Request-ID')}")
    
    # Сохраняем аудио
    with open("test_output.wav", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("Audio saved to test_output.wav")
    print()

def test_metrics():
    """Проверка метрик"""
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Metrics: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("Qwen-TTS API Test Suite")
    print("=" * 50)
    print()
    
    # Ждем немного, чтобы сервис успел загрузиться
    print("Waiting for service to be ready...")
    time.sleep(2)
    
    test_health()
    test_stream_tts()
    test_tts()
    test_metrics()
    
    print("=" * 50)
    print("Tests completed!")
    print("=" * 50)