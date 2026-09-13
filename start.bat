@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found: .venv
    echo Follow the installation instructions in README.md.
    pause
    exit /b 2
)
".venv\Scripts\python.exe" main.py %*
set "OCR_EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %OCR_EXIT_CODE%
