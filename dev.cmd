@echo off
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0dev.py" %*
) else (
    py "%~dp0dev.py" %*
)
