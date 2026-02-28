"""
Entry point for the Centurion Capital FastAPI server.

Usage:
    python run_api.py
    python run_api.py --port 9001 --reload
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Load .env BEFORE any other module reads os.getenv()
from dotenv import load_dotenv
load_dotenv(Path(_ROOT) / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Centurion Capital Trading API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9001, help="Port (default: 9001)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (default: 1)")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
