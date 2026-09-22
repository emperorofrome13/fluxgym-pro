@echo off
REM fluxgym-pro one-time setup launcher (Windows)
REM 1) creates/uses the UI venv  2) installs UI deps  3) runs install.py
REM install.py asks yes/no for each component and for your HF token (Krea 2 only).

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

echo [fluxgym-pro] Installing/updating dependencies (gradio, huggingface_hub)...
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [fluxgym-pro] ERROR: dependency install failed.
  pause
  exit /b 1
)

echo [fluxgym-pro] Running the interactive installer...
"%PY%" install.py
if errorlevel 1 (
  echo.
  echo [fluxgym-pro] Installer finished with an error - see the message above.
  pause
  exit /b 1
)

echo.
echo [fluxgym-pro] Setup done. Run quickstart.bat to open the training UI.
pause
endlocal