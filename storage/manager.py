"""
Storage Manager Module.

Manages persistence of trading signals to MinIO object storage
with Excel/CSV format, append mode, and deduplication.
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import Config
from models import TradingSignal

logger = logging.getLogger(__name__)


def _get_minio():
    """Lazy accessor for MinIOService singleton."""
    try:
        from storage.minio_service import get_minio_service
        return get_minio_service()
    except Exception:
        return None


class StorageManager:
    """Manages storage of trading signals to MinIO (with local fallback)."""

    # MinIO object path for the signals file
    _MINIO_OBJECT = "signals/daily_stock_news.xlsx"

    def __init__(self, output_file: str = None):
        """
        Initialize the storage manager.
        
        Args:
            output_file: Path to output file (Excel or CSV)
        """
        self.output_file = output_file or Config.OUTPUT_FILE
        self.file_path = Path(self.output_file)
    
    def save_signals(self, signals: List[TradingSignal], append: bool = True) -> Optional[str]:
        """
        Save trading signals to MinIO (preferred) or local file (fallback).
        
        Args:
            signals: List of TradingSignal objects to persist
            append: If True, append to existing data; otherwise overwrite
            
        Returns:
            MinIO object path or local file path, or None if save failed
        """
        if not signals:
            logger.info("No signals to save")
            return None
        
        # Convert signals to DataFrame
        data = [signal.to_dict() for signal in signals]
        new_df = pd.DataFrame(data)

        # Try MinIO first
        minio = _get_minio()
        if minio and minio.is_available:
            return self._save_to_minio(minio, new_df, append)

        # Fallback: save locally
        logger.info("MinIO unavailable — falling back to local file")
        return self._save_to_local(new_df, append)

    # ------------------------------------------------------------------
    # MinIO persistence
    # ------------------------------------------------------------------

    def _save_to_minio(self, minio, new_df: pd.DataFrame, append: bool) -> Optional[str]:
        """Save signals DataFrame to MinIO as xlsx."""
        try:
            minio.ensure_bucket_ready()
            bucket = minio._config.bucket_name
            obj_name = self._MINIO_OBJECT
            df_to_save = new_df

            # Append: download existing file, merge, dedup
            if append:
                try:
                    resp = minio.client.get_object(bucket, obj_name)
                    existing_df = pd.read_excel(io.BytesIO(resp.read()))
                    resp.close()
                    resp.release_conn()
                    combined = pd.concat([existing_df, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=["ticker", "source", "title"], keep="last",
                    )
                    df_to_save = combined
                    logger.info("Appended %d signals to existing MinIO object", len(new_df))
                except Exception:
                    # Object doesn't exist yet — first write
                    logger.info("No existing signals in MinIO — creating new object")

            # Write to in-memory buffer then upload
            buf = io.BytesIO()
            df_to_save.to_excel(buf, index=False, engine="openpyxl")
            buf.seek(0)
            data_bytes = buf.getvalue()

            minio.client.put_object(
                bucket,
                obj_name,
                io.BytesIO(data_bytes),
                length=len(data_bytes),
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            path = f"{bucket}/{obj_name}"
            logger.info("Successfully saved signals to MinIO: %s", path)
            return path

        except Exception as e:
            logger.error("MinIO save failed: %s — falling back to local", e)
            return self._save_to_local(new_df, append=True)

    # ------------------------------------------------------------------
    # Local file fallback
    # ------------------------------------------------------------------

    def _save_to_local(self, new_df: pd.DataFrame, append: bool) -> Optional[str]:
        """Save signals DataFrame to local xlsx/csv file."""
        df_to_save = new_df

        if append and self.file_path.exists():
            try:
                if self.file_path.suffix == '.xlsx':
                    existing_df = pd.read_excel(self.file_path)
                else:
                    existing_df = pd.read_csv(self.file_path)

                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=['ticker', 'source', 'title'], keep='last',
                )
                df_to_save = combined
                logger.info("Appended %d signals to existing local file", len(new_df))
            except Exception as e:
                logger.error("Error reading existing file: %s", e)

        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if self.file_path.suffix == '.xlsx':
                df_to_save.to_excel(self.file_path, index=False)
            else:
                df_to_save.to_csv(self.file_path, index=False)

            absolute_path = self.file_path.absolute()
            logger.info("Successfully saved data to %s", absolute_path)
            return str(absolute_path)
        except Exception as e:
            logger.error("Error saving to file: %s", e)
            return None
