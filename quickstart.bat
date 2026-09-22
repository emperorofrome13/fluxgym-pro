@echo off
REM fluxgym-pro one-click launcher (Windows)
REM 1) creates/uses a local venv  2) installs deps  3) starts the UI at http://127.0.0.1:7860

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo [fluxgym-pro] The local Python environment is stale. Rebuilding it...
    rmdir /s /q ".venv"
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo [fluxgym-pro] Creating virtual environment...
  where python >nul 2>nul
  if errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 (
    echo [fluxgym-pro] ERROR: Python not found. Install Python 3.10+ and retry.
    pause
    exit /b 1
  )
)

set PY=.venv\Scripts\python.exe

echo [fluxgym-pro] Installing/updating dependencies...
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [fluxgym-pro] ERROR: dependency install failed.
  pause
  exit /b 1
)

echo [fluxgym-pro] Starting UI at http://127.0.0.1:7860 ...
"%PY%" app.py

endlocal
