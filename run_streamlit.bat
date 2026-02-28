@echo off
REM Quick launch script for Algo Trading Alert System Streamlit UI

REM Prevent torch/Streamlit file-watcher conflict
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

echo ======================================================================
echo  Algo Trading Alert System - Streamlit UI
echo ======================================================================
echo.

REM Activate the myenv virtual environment
if exist "%~dp0myenv\Scripts\activate.bat" (
    call "%~dp0myenv\Scripts\activate.bat"
    echo Virtual environment: myenv activated
) else (
    echo WARNING: myenv not found. Run: python -m venv myenv
    echo Then:    myenv\Scripts\activate ^& pip install -r requirements.txt
)

echo.
echo Starting Streamlit web interface...
echo The application will open in your default browser.
echo.
echo Press Ctrl+C to stop the server.
echo ======================================================================
echo.

streamlit run app.py --server.port 9000

pause
