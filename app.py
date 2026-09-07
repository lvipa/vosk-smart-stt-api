# app.py
# Простой Vosk API:
# - Принимает ЛЮБОЙ аудиоформат (mp3/ogg/m4a/webm/wav/...) через ffmpeg
# - Распознаёт по выбранным параметрам: lang + model (small/big/...)
# - Без auto-режима

import io
import json
import os
import subprocess
import tempfile
import wave
from typing import Dict, List, Tuple
import time
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from vosk import Model, KaldiRecognizer, SetLogLevel
import importlib.util
from functools import lru_cache

import logging
log = logging.getLogger("punc")
metrics_log = logging.getLogger("vosk.metrics")
metrics_log.setLevel(logging.INFO)

APP_VERSION = "0.1.0"


# Отключаем лишние логи Vosk
SetLogLevel(-1)

app = FastAPI(
    title="Vosk Smart STT API",
    version=APP_VERSION,
    description="Multilingual speech-to-text API based on FastAPI, Vosk and ffmpeg"
)

# ----------------------------
# Настройки
# ----------------------------

# Папка с моделями (по умолчанию: ./models рядом с app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(BASE_DIR, "models"))

# ffmpeg path. Can be overridden by FFMPEG_BIN environment variable.
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

# Лимит входного файла (защита от огромных загрузок)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# Целевой формат для Vosk
TARGET_SR = 16000  # 16 kHz
TARGET_CH = 1      # mono

# Реестр моделей: язык -> имя_модели -> папка
# Папки указывай относительно MODELS_DIR или абсолютным путём.
# STT модели (аудио -> текст)
MODEL_REGISTRY = {
    "ru": {"small": "vosk-model-small-ru-0.22", "big": "vosk-model-ru-0.42"},
    "en": {"small": "vosk-model-small-en-us-0.15", "big": "vosk-model-en-us-0.22"},
    "fr": {"small": "vosk-model-small-fr-0.22", "big": "vosk-model-fr-0.22"},
    "kz": {"small": "vosk-model-small-kz-0.42", "big": "vosk-model-kz-0.42"},
}

# PUNC модели (текст -> текст с пунктуацией/регистром)
PUNC_REGISTRY = {
    "ru": {"dir": "vosk-recasepunc-ru-0.22"},
    "en": {"dir": "vosk-recasepunc-en-0.22"},
}

# Кэш загруженных моделей (чтобы не грузить каждый раз с диска)
# Важно: big-модели могут быть тяжёлыми по RAM. Если RAM мало — можно не держать много моделей одновременно.
MODELS_CACHE: Dict[str, Model] = {}


# ----------------------------
# Вспомогательные функции
# ----------------------------

def _punc_model_dir(lang: str) -> str:
    item = PUNC_REGISTRY[lang]
    folder = item["dir"] if isinstance(item, dict) else item
    return folder if os.path.isabs(folder) else os.path.join(MODELS_DIR, folder)

@lru_cache(maxsize=8)
def _load_recasepunc_module(punc_dir: str):
    mod_path = os.path.join(punc_dir, "recasepunc.py")
    if not os.path.isfile(mod_path):
        raise ValueError(f"Не найден recasepunc.py в {punc_dir}")

    spec = importlib.util.spec_from_file_location(f"recasepunc_{hash(punc_dir)}", mod_path)
    if spec is None or spec.loader is None:
        raise ValueError("Не удалось загрузить recasepunc.py (importlib spec)")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # --- FIX для torch.load: чекпойнт ожидает классы в __main__
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        for name, obj in vars(module).items():
            if isinstance(obj, type) and not hasattr(main_mod, name):
                setattr(main_mod, name, obj)

    return module

@lru_cache(maxsize=8)
def _get_punc_predictor(lang: str):
    if lang not in PUNC_REGISTRY:
        raise ValueError(f"Для lang={lang} нет punc-модели (есть: {sorted(PUNC_REGISTRY.keys())})")

    punc_dir = _punc_model_dir(lang)
    if not os.path.isdir(punc_dir):
        raise ValueError(f"Папка punc-модели не найдена: {punc_dir}")

    checkpoint = os.path.join(punc_dir, "checkpoint")

    # checkpoint может быть файлом (норма)
    if os.path.isfile(checkpoint):
        checkpoint_path = checkpoint

    # либо папкой (если кто-то распаковал) — ищем файл внутри
    elif os.path.isdir(checkpoint):
        checkpoint_path = None
        for name in os.listdir(checkpoint):
            p = os.path.join(checkpoint, name)
            ext = os.path.splitext(name)[1].lower()
            if os.path.isfile(p) and ext in (".pt", ".pth", ".bin", ""):
                checkpoint_path = p
                break
        if not checkpoint_path:
            raise ValueError(f"В папке {checkpoint} не найден файл весов (*.pt/*.pth/*.bin)")
    else:
        raise ValueError(f"Не найден checkpoint (ни файл, ни папка): {checkpoint}")

    # 1) загружаем recasepunc.py
    recasepunc = _load_recasepunc_module(punc_dir)

    # 2) патчим torch.load внутри recasepunc:
    #    - torch 2.6+: weights_only -> False
    #    - убираем несовместимый ключ position_ids из state_dict
    import inspect
    import torch

    _orig_load = torch.load

    def _load_compat(*args, **kwargs):
        if "weights_only" in inspect.signature(_orig_load).parameters:
            kwargs.setdefault("weights_only", False)

        obj = _orig_load(*args, **kwargs)

        try:
            sd = obj.get("model_state_dict")
            if isinstance(sd, dict):
                sd.pop("bert.embeddings.position_ids", None)
                for k in list(sd.keys()):
                    if k.endswith("position_ids"):
                        sd.pop(k, None)
        except Exception:
            pass

        return obj

    recasepunc.torch.load = _load_compat

    # 3) создаём предиктор
    cfg = PUNC_REGISTRY[lang]
    flavor = cfg.get("flavor") if isinstance(cfg, dict) else None

    # для benob/recasepunc нужно явно задать flavor, иначе будет None
    if flavor:
        return recasepunc.CasePuncPredictor(checkpoint_path, lang=lang, flavor=flavor)
    else:
        return recasepunc.CasePuncPredictor(checkpoint_path, lang=lang)

def _apply_punc(lang: str, text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text

    predictor = _get_punc_predictor(lang)

    tokens = list(enumerate(predictor.tokenize(text)))
    results = ""

    for token, case_label, punc_label in predictor.predict(tokens, lambda x: x[1]):
        prediction = predictor.map_punc_label(
            predictor.map_case_label(token[1], case_label),
            punc_label
        )

        if token[1][0] == "'" or (len(results) > 0 and results[-1] == "'"):
            results = results + prediction
        elif token[1][0] != "#":
            results = results + " " + prediction
        else:
            results = results + prediction

    return results.strip()

def _model_path(lang: str, model_name: str) -> str:
    """Собираем путь к папке модели."""
    folder = MODEL_REGISTRY[lang][model_name]
    return folder if os.path.isabs(folder) else os.path.join(MODELS_DIR, folder)

def _get_model(lang: str, model_name: str) -> Model:
    if lang not in MODEL_REGISTRY:
        raise ValueError(f"Неизвестный язык lang={lang}")
    if model_name not in MODEL_REGISTRY[lang]:
        raise ValueError(f"Неизвестная модель model={model_name} для lang={lang}")

    key = f"{lang}:{model_name}"
    if key in MODELS_CACHE:
        return MODELS_CACHE[key]

    path = _model_path(lang, model_name)
    if not os.path.isdir(path):
        raise ValueError(f"Папка модели не найдена: {path}")

    try:
        MODELS_CACHE[key] = Model(path)
    except Exception as e:
        raise ValueError(f"Не удалось загрузить модель из {path}: {e}")

    return MODELS_CACHE[key]

def _ffmpeg_to_wav(input_bytes: bytes, suffix: str) -> bytes:
    """
    Конвертируем любой входной файл в WAV 16kHz mono PCM s16le через ffmpeg.
    На Windows проще/надёжнее через временные файлы.
    """
    if not suffix:
        suffix = ".bin"
    if not suffix.startswith("."):
        suffix = "." + suffix

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, f"input{suffix}")
        dst = os.path.join(td, "out.wav")

        with open(src, "wb") as f:
            f.write(input_bytes)

        cmd = [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-i", src,
            "-vn",
            "-ac", str(TARGET_CH),
            "-ar", str(TARGET_SR),
            "-c:a", "pcm_s16le",
            dst,
        ]

        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except FileNotFoundError:
            raise ValueError("ffmpeg не найден. Установи ffmpeg или задай FFMPEG_BIN=полный_путь_к_ffmpeg.exe")

        if p.returncode != 0:
            err = (p.stderr.decode("utf-8", errors="ignore") or "").strip()
            raise ValueError(f"ffmpeg не смог декодировать файл: {err[:400]}")

        with open(dst, "rb") as f:
            return f.read()

def _read_wav_chunks(wav_bytes: bytes) -> Tuple[int, float, List[bytes]]:
    try:
        wf = wave.open(io.BytesIO(wav_bytes), "rb")
    except wave.Error as e:
        raise ValueError(f"WAV повреждён или неверный формат: {e}")

    if wf.getnchannels() != 1:
        raise ValueError("WAV должен быть mono (1 канал).")
    if wf.getsampwidth() != 2:
        raise ValueError("WAV должен быть 16-bit PCM (s16le).")

    sample_rate = wf.getframerate()
    duration_sec = wf.getnframes() / float(sample_rate)

    chunks: List[bytes] = []
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        chunks.append(data)

    return sample_rate, duration_sec, chunks


def _transcribe(model_obj: Model, sample_rate: int, chunks: List[bytes], words: bool) -> Dict:
    """Распознаём аудио чанками."""
    rec = KaldiRecognizer(model_obj, sample_rate)
    if words:
        rec.SetWords(True)

    for ch in chunks:
        rec.AcceptWaveform(ch)

    final = json.loads(rec.FinalResult())

    # Средняя уверенность по словам (если слова включены)
    confs = [w.get("conf", 0.0) for w in final.get("result", [])]
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0

    return {
        "text": final.get("text", ""),
        "result": final.get("result", []),
        "avg_conf": avg_conf,
    }


# ----------------------------
# API
# ----------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "available": {lang: sorted(models.keys()) for lang, models in MODEL_REGISTRY.items()},
        "loaded_cache": sorted(MODELS_CACHE.keys()),
    }

@app.get("/models")
def models():
    """Показываем реестр моделей (что прописано в конфиге)."""
    return MODEL_REGISTRY

@app.post("/stt")
async def stt(
    file: UploadFile = File(...),
    lang: str = Query(..., description="ru|en"),
    model: str = Query("small", description="small|big|..."),
    words: bool = Query(True, description="Включить слова с таймингами/conf"),
    punc: bool = Query(False, description="Постобработка: регистр+пунктуация (ru/en)"),
):
    raw = await file.read()

    # Защита от очень больших файлов
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, detail=f"Файл слишком большой (> {MAX_UPLOAD_MB}MB)")

    filename = file.filename or ""
    suffix = os.path.splitext(filename)[1] or ".bin"

    # 1) Конвертируем любой формат -> WAV 16k mono
    try:
        wav_bytes = await run_in_threadpool(_ffmpeg_to_wav, raw, suffix)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    # 2) Парсим WAV
    try:
        sample_rate, audio_sec, chunks = await run_in_threadpool(_read_wav_chunks, wav_bytes)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    # 3) Загружаем модель
    try:
        model_obj = await run_in_threadpool(_get_model, lang, model)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    # 4) Распознаём
    t0 = time.perf_counter()
    c0 = time.process_time()

    out = await run_in_threadpool(_transcribe, model_obj, sample_rate, chunks, words)

    c1 = time.process_time()
    t1 = time.perf_counter()

    elapsed = max(t1 - t0, 1e-9)          # wall time
    cpu = max(c1 - c0, 0.0)               # cpu time процесса
    cores_used = cpu / elapsed            # "ядер" в среднем
    rtf = elapsed / max(audio_sec, 1e-9)  # real-time factor
    xrt = max(audio_sec, 1e-9) / elapsed  # "попугаи" (x realtime)

    metrics_log.info(
        'STT lang=%s model=%s punc=%s words=%s audio=%.3fs stt=%.3fs rtf=%.4f cores=%.2f xrt=%.2f file="%s"',
        lang, model, punc, words, audio_sec, elapsed, rtf, cores_used, xrt, filename
    )
    
    if punc:
        try:
            out["text_raw"] = out.get("text", "")
            out["text"] = await run_in_threadpool(_apply_punc, lang, out["text_raw"])
        except Exception as e:
            log.exception("PUNC failed")  # полный в консоль/лог uvicorn
            raise HTTPException(400,  detail=f"PUNC failed: {type(e).__name__}: {e}")
    
    out.update({"lang": lang, "model": model, "input_filename": filename, "punc": punc})
    return out
