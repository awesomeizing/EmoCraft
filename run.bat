@echo off
title EmoCraft — Facial Emotion Recognition
cd /d "%~dp0"

echo ===================================================================
echo    EmoCraft: Facial Emotion Recognition System
echo ===================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
    set "PYTHON_CMD=python"
)

"%PYTHON_CMD%" run.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] An error occurred while running EmoCraft.
    pause
)

