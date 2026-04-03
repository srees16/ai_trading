"""
Shared batch-run history persistence for all chapter-runner labs.

Saves completed batch results to:
  1. PostgreSQL (backtest_results table, strategy_id = "<lab>_batch")
  2. MinIO / Cloudflare R2 (figures as PNG, text as .txt)
"""

import base64
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def save_batch_history(
    batch_id: str,
    lab_prefix: str,
    chapter_keys: List[str],
    progress: Dict[str, Any],
    tickers: Optional[List[str]] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> None:
    """Persist a completed batch run to DB + object storage.

    Args:
        batch_id: UUID of the batch run
        lab_prefix: e.g. "fml", "tts", "aronson", "ehlers", "vince"
        chapter_keys: list of chapter keys that were requested
        progress: the full _batch_progress[batch_id] dict (with chapters results)
        tickers: tickers used for the run
        date_start: start date string
        date_end: end date string
    """
    _save_to_database(batch_id, lab_prefix, chapter_keys, progress, tickers, date_start, date_end)
    _save_to_object_storage(batch_id, lab_prefix, progress)


def _save_to_database(
    batch_id: str,
    lab_prefix: str,
    chapter_keys: List[str],
    progress: Dict[str, Any],
    tickers: Optional[List[str]],
    date_start: Optional[str],
    date_end: Optional[str],
) -> None:
    """Save batch summary as a BacktestResult row."""
    try:
        from database.service import DatabaseService

        db = DatabaseService()
        if not db.is_available:
            logger.warning("Database not available — skipping batch history save")
            return

        total = progress.get("total", len(chapter_keys))
        completed = progress.get("completed", 0)
        chapters_data = progress.get("chapters", {})

        done_count = sum(1 for v in chapters_data.values() if isinstance(v, dict) and v.get("status") == "done")
        error_count = sum(1 for v in chapters_data.values() if isinstance(v, dict) and v.get("status") == "error")
        status = "completed" if error_count == 0 else ("error" if done_count == 0 else "partial")
        if progress.get("status") == "aborted":
            status = "aborted"

        # Build a lightweight summary (no base64 figures — those go to R2)
        chapter_summaries = {}
        for k, v in chapters_data.items():
            if not isinstance(v, dict):
                continue
            chapter_summaries[k] = {
                "status": v.get("status", "unknown"),
                "figure_count": len(v.get("figures", [])),
                "text_length": len(v.get("text_output", "")),
                "error_message": v.get("error_message"),
            }

        now = datetime.utcnow()
        result = {
            "strategy_id": f"{lab_prefix}_batch",
            "strategy_name": f"{lab_prefix.upper()} Batch Run",
            "strategy_category": "chapter_runner",
            "tickers": tickers or [],
            "start_date": date_start or (now.isoformat()),
            "end_date": date_end or (now.isoformat()),
            "initial_capital": 0,
            "total_trades": 0,
            "success": status in ("completed", "partial"),
            "parameters": {
                "batch_id": batch_id,
                "lab": lab_prefix,
                "chapters_requested": chapter_keys,
            },
            "metrics": {
                "batch_id": batch_id,
                "lab": lab_prefix,
                "status": status,
                "total": total,
                "completed": completed,
                "done_count": done_count,
                "error_count": error_count,
                "chapters": chapter_summaries,
                "created_at": now.isoformat(),
            },
        }

        db.save_backtest_result(result=result, market="LAB")
        logger.info("Saved %s batch %s to database (status=%s)", lab_prefix, batch_id, status)

    except Exception as e:
        logger.warning("Failed to save batch history to DB: %s", e)


def _save_to_object_storage(
    batch_id: str,
    lab_prefix: str,
    progress: Dict[str, Any],
) -> None:
    """Save figures and text output to MinIO / Cloudflare R2."""
    try:
        from services.storage.minio_service import get_minio_service

        minio_svc = get_minio_service()
        if not minio_svc.is_available:
            logger.info("MinIO/R2 not available — skipping object storage save")
            return

        chapters_data = progress.get("chapters", {})
        run_id = f"{lab_prefix}_{batch_id}"
        saved_count = 0

        for ch_key, ch_data in chapters_data.items():
            if not isinstance(ch_data, dict):
                continue

            # Save figures
            for idx, fig_b64 in enumerate(ch_data.get("figures", [])):
                try:
                    img_bytes = base64.b64decode(fig_b64)
                    minio_svc.save_backtest_image(
                        run_id=run_id,
                        image_data=img_bytes,
                        filename=f"fig_{idx}.png",
                        strategy_name=ch_key,
                        chart_title=f"{lab_prefix} {ch_key} figure {idx}",
                    )
                    saved_count += 1
                except Exception:
                    pass

            # Save text output
            text = ch_data.get("text_output", "")
            if text:
                try:
                    text_bytes = text.encode("utf-8")
                    minio_svc.save_backtest_image(
                        run_id=run_id,
                        image_data=text_bytes,
                        filename="output.txt",
                        strategy_name=ch_key,
                        chart_title=f"{lab_prefix} {ch_key} text output",
                        content_type="text/plain",
                    )
                    saved_count += 1
                except Exception:
                    pass

        logger.info("Saved %d artifacts to R2 for %s batch %s", saved_count, lab_prefix, batch_id)

    except Exception as e:
        logger.warning("Failed to save batch history to R2: %s", e)
