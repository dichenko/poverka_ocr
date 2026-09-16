@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-reader\Scripts\python.exe" (
  echo Missing .venv-reader environment. See README.md.
  exit /b 2
)
".venv-reader\Scripts\python.exe" -X utf8 test_readings_only.py %*
exit /b %errorlevel%
