@echo off
REM Quick launch script for Algo Trading Alert System Streamlit UI

REM Prevent torch/Streamlit file-watcher conflict
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

REM PostgreSQL database credentials
set CENTURION_DB_HOST=localhost
set CENTURION_DB_PORT=9003
set CENTURION_DB_NAME=centurion_rag
set CENTURION_DB_USER=postgres
set CENTURION_DB_PASSWORD=superadmin1
set KITE_DB_HOST=localhost
set KITE_DB_PORT=9003
set KITE_DB_NAME=livestocks_ind
set KITE_DB_USER=postgres
set KITE_DB_PASSWORD=superadmin1

REM MinIO object storage credentials
set MINIO_ENDPOINT=localhost:9004
set MINIO_ACCESS_KEY=minioadmin
set MINIO_SECRET_KEY=minioadmin123
set MINIO_SECURE=false
set MINIO_BUCKET=centurion-backtests
set MINIO_ENABLED=true

REM RAG pipeline ChromaDB path
set CENTURION_RAG_CHROMA_DIR=./chroma_store

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
