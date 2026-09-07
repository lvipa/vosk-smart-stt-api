# Vosk Smart STT API

An offline speech-to-text HTTP API built with FastAPI, Vosk and ffmpeg. It accepts common audio formats, normalises them to WAV, and returns transcribed text with optional word timings and punctuation recovery.

## What it does

- accepts `mp3`, `ogg`, `m4a`, `webm`, `wav` and other formats supported by ffmpeg;
- supports Russian and English Vosk models, with model selection per request;
- returns word timestamps and confidence values when requested;
- optionally restores casing and punctuation;
- caches loaded models and reports processing metrics such as RTF, xRT and CPU usage;
- exposes lightweight health and model-list endpoints for deployment checks.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) available in `PATH`
- downloaded Vosk models (they are intentionally not included in the repository)

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `models/` and place downloaded models there, for example:

```text
models/
├── vosk-model-small-ru-0.22/
├── vosk-model-ru-0.42/
├── vosk-model-small-en-us-0.15/
└── vosk-model-en-us-0.22/
```

Start the service:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

For Windows, `run_vosk_api.cmd` provides the same startup pattern with log output and conservative defaults for model memory use.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness check. |
| `GET /models` | Lists available model directories. |
| `POST /stt` | Transcribes the uploaded audio file. |

Example health check:

```bash
curl http://localhost:8000/health
```

Example transcription:

```bash
curl -X POST "http://localhost:8000/stt?lang=ru&model=small&words=true&punc=false" \
  -F "file=@sample.mp3"
```

### `POST /stt` parameters

| Parameter | Values | Description |
| --- | --- | --- |
| `file` | audio file | Multipart upload. |
| `lang` | `ru`, `en` | Recognition language. |
| `model` | `small`, `big`, … | Model variant to use. |
| `words` | `true`, `false` | Include word timings and confidence. |
| `punc` | `true`, `false` | Apply punctuation and casing recovery for supported languages. |

Interactive API documentation is available at `/docs` while the service is running.

## Configuration and privacy

- Copy `.env.example` to `.env` if you want to change model directory, ffmpeg path or limits.
- Model files, audio files, logs, virtual environments and `.env` are excluded from Git.
- The service is designed for local or self-hosted use; deploy it behind appropriate authentication and transport security before accepting untrusted external traffic.
