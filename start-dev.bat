@echo off
title Centurion Dev Launcher
echo.
echo   Centurion Capital - Dev Servers
echo   Backend  : http://127.0.0.1:8000
echo   Frontend : http://localhost:3000
echo.

cd /d "%~dp0"

:: Start backend in a new window (activate venv, then run FastAPI)
start "Centurion Backend" cmd /k "call ..\myenv\Scripts\activate.bat && python run_api.py --port 8000 --reload"

:: Start frontend in a new window
start "Centurion Frontend" cmd /k "cd frontend && npm run dev"

echo   Both servers launched in separate windows.
echo   Close those windows to stop the servers.
