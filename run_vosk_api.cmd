@echo off
setlocal

rem === Project directory ===
set "APP_DIR=%~dp0"

rem === Python from virtual environment ===
set "PY=%APP_DIR%\.venv\Scripts\python.exe"

rem === Models directory ===
set "MODELS_DIR=%APP_DIR%\models"

rem === ffmpeg ===
rem If ffmpeg is in PATH, use:
set "FFMPEG_BIN=ffmpeg"

rem === Limits/settings ===
set "MAX_UPLOAD_MB=50"
set "MAX_LOADED_MODELS=2"

rem === Logs ===
if not exist "%APP_DIR%\logs" mkdir "%APP_DIR%\logs"

cd /d "%APP_DIR%"

rem IMPORTANT: workers=1 to avoid loading models multiple times into RAM
"%PY%" -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --log-config "%APP_DIR%\log_config.json" ^
  >> "%APP_DIR%\logs\uvicorn.log" 2>&1

endlocal