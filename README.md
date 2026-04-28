# Vosk Smart STT API

Vosk Smart STT API — это небольшой FastAPI-сервис для распознавания речи через Vosk.

## Возможности

- Приём аудио в разных форматах: mp3, ogg, m4a, webm, wav и других
- Конвертация аудио через ffmpeg в WAV 16 kHz mono
- Выбор языка и модели: small / big
- Поддержка word timings и confidence
- Опциональная пунктуация и восстановление регистра
- Кэширование загруженных моделей
- Метрики производительности: RTF, xRT, CPU usage

## Требования

- Python 3.10+
- ffmpeg
- Vosk models

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

```

##Модели

Модели не входят в репозиторий.

Создайте папку: ```models/```

И положите туда модели Vosk, например:
```
models/
├── vosk-model-small-ru-0.22/
├── vosk-model-ru-0.42/
├── vosk-model-small-en-us-0.15/
└── vosk-model-en-us-0.22/
```

##Запуск
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

##Проверка
`curl http://localhost:8000/health`

##Распознавание
`curl -X POST "http://localhost:8000/stt?lang=ru&model=small&punc=false" -F "file=@sample.mp3"`

##Параметры API
| Параметр | Описание                                       |
| -------- | ---------------------------------------------- |
| lang     | Язык модели: ru, en                    		|
| model    | Тип модели: small, big                         |
| words    | Возвращать слова с таймингами и confidence     |
| punc     | Применять пунктуацию и восстановление регистра |
