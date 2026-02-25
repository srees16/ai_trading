@echo off
REM Quick launch script for Algo Trading Alert System Streamlit UI

REM Prevent torch/Streamlit file-watcher conflict
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

echo ======================================================================
echo  Algo Trading Alert System - Streamlit UI
echo ======================================================================
echo.

REM Activate the mywinenv virtual environment
if exist "%~dp0mywinenv\Scripts\activate.bat" (
    call "%~dp0mywinenv\Scripts\activate.bat"
    echo Virtual environment: mywinenv activated
) else (
    echo WARNING: mywinenv not found. Run: python -m venv mywinenv
    echo Then:    mywinenv\Scripts\activate ^& pip install -r requirements.txt
)

echo.
echo Starting Streamlit web interface...
echo The application will open in your default browser.
echo.
echo Press Ctrl+C to stop the server.
echo ======================================================================
echo.

streamlit run app.py

pause
