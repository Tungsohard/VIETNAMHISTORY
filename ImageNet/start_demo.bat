@echo off
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment was not found.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" demo.py --open-browser
if errorlevel 1 pause
