@echo off
call "%~dp0start.bat" --overwrite %*
exit /b %ERRORLEVEL%
