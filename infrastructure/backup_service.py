# Centurion Capital LLC - SQLite Backup Service
"""
Nightly backup of SQLite databases to S3-compatible storage (MinIO / Cloudflare R2).

Backs up:
  - data/scheduler_cache.sqlite3
  - data/trade_monitor_state.sqlite3
  - chroma_store/chroma.sqlite3

Usage (from scheduler):
    from infrastructure.backup_service import run_backup

    run_backup()  # uploads all SQLite files to R2 under backups/<date>/
"""

import os
import glob
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SQLite files to back up (relative to centurion_core root)
_BACKUP_TARGETS = [
    "data/scheduler_cache.sqlite3",
    "data/trade_monitor_state.sqlite3",
    "chroma_store/chroma.sqlite3",
]


def run_backup() -> dict:
    """Upload all SQLite databases to S3-compatible storage.
    
    Returns:
        dict with 'succeeded', 'failed', 'skipped' counts and file list.
    """
    try:
        from services.storage.minio_service import get_minio_service
    except ImportError:
        logger.warning("MinIO service unavailable — backup skipped")
        return {"succeeded": 0, "failed": 0, "skipped": 0, "error": "minio_service not importable"}

    minio = get_minio_service()
    if not minio.is_available:
        logger.warning("Object storage not reachable — backup skipped")
        return {"succeeded": 0, "failed": 0, "skipped": 0, "error": "storage_unreachable"}

    date_prefix = datetime.now().strftime("%Y-%m-%d")
    results = {"succeeded": 0, "failed": 0, "skipped": 0, "files": []}

    for rel_path in _BACKUP_TARGETS:
        abs_path = os.path.join(_ROOT, rel_path)
        if not os.path.exists(abs_path):
            results["skipped"] += 1
            continue

        object_name = f"backups/{date_prefix}/{rel_path.replace(os.sep, '/')}"
        ok = minio.upload_file(abs_path, object_name, content_type="application/x-sqlite3")
        if ok:
            results["succeeded"] += 1
            results["files"].append(object_name)
        else:
            results["failed"] += 1

    logger.info(
        "Backup complete: %d succeeded, %d failed, %d skipped",
        results["succeeded"], results["failed"], results["skipped"],
    )
    return results
