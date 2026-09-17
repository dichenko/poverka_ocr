@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv environment. See README.md.
  exit /b 2
)
".venv\Scripts\python.exe" -X utf8 test_readings_only.py %*
exit /b %errorlevel%
