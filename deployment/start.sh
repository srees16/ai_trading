#!/bin/bash
# Centurion Core — Container entrypoint
# Starts both the FastAPI server and the APScheduler background process.

set -e

PORT="${PORT:-7860}"

echo "Starting Centurion Core..."
echo "  FastAPI  → 0.0.0.0:${PORT}"
echo "  Scheduler → background process"

# Start the scheduler in the background
python scheduler.py &
SCHEDULER_PID=$!

# Start FastAPI (foreground — container lifecycle tied to this)
python run_api.py --host 0.0.0.0 --port "${PORT}" &
API_PID=$!

# Trap shutdown signals and forward to both processes
cleanup() {
    echo "Shutting down..."
    kill $SCHEDULER_PID 2>/dev/null || true
    kill $API_PID 2>/dev/null || true
    # Run backup before exit
    python -c "
try:
    from infrastructure.backup_service import run_backup
    run_backup()
    print('Shutdown backup complete')
except Exception as e:
    print(f'Shutdown backup skipped: {e}')
" 2>/dev/null || true
    wait
}
trap cleanup SIGTERM SIGINT

# Wait for either process to exit
wait -n $API_PID $SCHEDULER_PID
EXIT_CODE=$?
cleanup
exit $EXIT_CODE
