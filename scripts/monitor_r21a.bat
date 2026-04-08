@echo off
title R21a Monitor
:loop
cls
echo ──────────────────────────────────────────────────
echo   R21a Monitor — %date% %time%
echo ──────────────────────────────────────────────────
cd /d c:\Users\suraboyi\Videos\dev_algo\centurion_core
..\myenv\Scripts\python.exe optimizer\analyze_r21a.py
timeout /t 120 >nul
goto loop
