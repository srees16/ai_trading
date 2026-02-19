# Centurion Capital LLC - MinIO Object Storage Service
"""
MinIO-based object storage for backtesting images and artifacts.

Stores chart images (matplotlib/plotly) generated during backtesting
and enables retrieval by run_id for the History page.

Usage:
    from storage.minio_service import get_minio_service

    minio = get_minio_service()
    if minio.is_available:
        # Save a chart image
        url = minio.save_backtest_image(
            run_id="abc-123",
            image_data=base64_bytes,
            filename="equity_curve.png",
            strategy_name="MACD Oscillator"
        )

        # Retrieve all images for a run
        images = minio.get_backtest_images(run_id="abc-123")
"""

import io
import os
import json
import logging
import base64
from datetime import timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

try:
    from minio import Minio
    from minio.error import S3Error
    from minio.deleteobjects import DeleteObject
    MINIO_SDK_AVAILABLE = True
except ImportError:
    MINIO_SDK_AVAILABLE = False
    logger.debug("minio package not installed — object storage disabled")


class MinIOConfig:
    """MinIO configuration from environment variables."""

    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
        self.secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        self.bucket_name = os.getenv("MINIO_BUCKET", "centurion-backtests")
        self.enabled = os.getenv("MINIO_ENABLED", "true").lower() == "true"


class MinIOService:
    """
    Object storage service for backtest chart images.

    Images are stored under:
        <bucket>/<run_id>/<ticker>/<strategy_name>/<filename>

    Each object carries user-metadata (strategy, chart_type, title)
    so it can be rendered correctly on retrieval.
    """

    _instance: Optional["MinIOService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._config = MinIOConfig()
        self._client: Optional[Minio] = None
        self._bucket_ensured = False
        self._initialized = True

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @property
    def client(self) -> Optional[Minio]:
        """Lazy-init MinIO client."""
        if self._client is None and MINIO_SDK_AVAILABLE and self._config.enabled:
            try:
                self._client = Minio(
                    self._config.endpoint,
                    access_key=self._config.access_key,
                    secret_key=self._config.secret_key,
                    secure=self._config.secure,
                )
            except Exception as e:
                logger.error(f"Failed to create MinIO client: {e}")
        return self._client

    @property
    def is_available(self) -> bool:
        """Check if MinIO is reachable."""
        if not self.client:
            return False
        try:
            self.client.list_buckets()
            return True
        except Exception:
            return False

    def _ensure_bucket(self):
        """Create the bucket if it does not exist."""
        if self._bucket_ensured or not self.client:
            return
        try:
            if not self.client.bucket_exists(self._config.bucket_name):
                self.client.make_bucket(self._config.bucket_name)
                logger.info(f"Created MinIO bucket: {self._config.bucket_name}")
            self._bucket_ensured = True
        except S3Error as e:
            logger.error(f"MinIO bucket error: {e}")

    # ------------------------------------------------------------------
    # Save operations
    # ------------------------------------------------------------------

    def save_backtest_image(
        self,
        run_id: str,
        image_data: bytes,
        filename: str,
        strategy_name: str = "",
        ticker: str = "",
        chart_title: str = "",
        chart_type: str = "matplotlib",
        content_type: str = "image/png",
    ) -> Optional[str]:
        """
        Save a single backtest chart image to MinIO.

        Args:
            run_id: Analysis / backtest run identifier
            image_data: Raw image bytes (decoded from base64)
            filename: Object filename (e.g. equity_curve.png)
            strategy_name: Strategy that produced the chart
            ticker: Ticker symbol for the chart
            chart_title: Human-readable chart title
            chart_type: 'matplotlib' or 'plotly'
            content_type: MIME type

        Returns:
            Object path on success, None on failure
        """
        if not self.client:
            return None

        self._ensure_bucket()

        safe_strategy = strategy_name.lower().replace(" ", "_")
        safe_ticker = ticker.upper() if ticker else "unknown"
        object_name = f"{run_id}/{safe_ticker}/{safe_strategy}/{filename}"

        metadata = {
            "x-amz-meta-run-id": run_id,
            "x-amz-meta-strategy": strategy_name,
            "x-amz-meta-ticker": ticker,
            "x-amz-meta-chart-title": chart_title,
            "x-amz-meta-chart-type": chart_type,
        }

        try:
            data_stream = io.BytesIO(image_data)
            self.client.put_object(
                self._config.bucket_name,
                object_name,
                data_stream,
                length=len(image_data),
                content_type=content_type,
                metadata=metadata,
            )
            logger.info(f"Saved image to MinIO: {object_name}")
            return object_name
        except S3Error as e:
            logger.error(f"MinIO put_object failed: {e}")
            return None

    def save_backtest_charts(
        self,
        run_id: str,
        charts: list,
        strategy_name: str = "",
    ) -> List[str]:
        """
        Save all charts from a StrategyResult to MinIO.

        Args:
            run_id: Unique run identifier
            charts: List of ChartData objects from StrategyResult
            strategy_name: Strategy name

        Returns:
            List of saved object paths
        """
        if not charts or not self.client:
            return []

        saved_paths: List[str] = []

        for idx, chart in enumerate(charts):
            try:
                if chart.chart_type == "matplotlib":
                    # chart.data is a base64 string (possibly with data: prefix)
                    raw = chart.data
                    if raw.startswith("data:"):
                        raw = raw.split(",", 1)[1]
                    image_bytes = base64.b64decode(raw)
                    filename = f"chart_{idx}.png"
                    content_type = "image/png"
                elif chart.chart_type == "plotly":
                    # Store plotly JSON as .json
                    image_bytes = (
                        chart.data.encode("utf-8")
                        if isinstance(chart.data, str)
                        else json.dumps(chart.data).encode("utf-8")
                    )
                    filename = f"chart_{idx}.json"
                    content_type = "application/json"
                else:
                    continue

                path = self.save_backtest_image(
                    run_id=run_id,
                    image_data=image_bytes,
                    filename=filename,
                    strategy_name=strategy_name,
                    ticker=getattr(chart, 'ticker', ''),
                    chart_title=chart.title,
                    chart_type=chart.chart_type,
                    content_type=content_type,
                )
                if path:
                    saved_paths.append(path)

            except Exception as e:
                logger.error(f"Failed to save chart {idx}: {e}")

        logger.info(
            f"Saved {len(saved_paths)}/{len(charts)} charts for run {run_id}"
        )
        return saved_paths

    # ------------------------------------------------------------------
    # Retrieve operations
    # ------------------------------------------------------------------

    def get_backtest_images(
        self, run_id: str, strategy_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all stored images/artifacts for a backtest run.

        Args:
            run_id: The run identifier
            strategy_name: Optional filter by strategy

        Returns:
            List of dicts with keys:
                - object_name, chart_title, chart_type, strategy,
                  content_type, data (raw bytes), presigned_url
        """
        if not self.client:
            return []

        prefix = f"{run_id}/"
        if strategy_name:
            safe = strategy_name.lower().replace(" ", "_")
            prefix = f"{run_id}/{safe}/"

        results: List[Dict[str, Any]] = []

        try:
            objects = self.client.list_objects(
                self._config.bucket_name, prefix=prefix, recursive=True
            )

            for obj in objects:
                try:
                    # Get object with metadata
                    stat = self.client.stat_object(
                        self._config.bucket_name, obj.object_name
                    )
                    meta = stat.metadata or {}

                    # Read object data
                    response = self.client.get_object(
                        self._config.bucket_name, obj.object_name
                    )
                    data = response.read()
                    response.close()
                    response.release_conn()

                    # Generate presigned URL (valid 1 hour)
                    presigned_url = self.client.presigned_get_object(
                        self._config.bucket_name,
                        obj.object_name,
                        expires=timedelta(hours=1),
                    )

                    results.append({
                        "object_name": obj.object_name,
                        "chart_title": meta.get("x-amz-meta-chart-title", ""),
                        "chart_type": meta.get("x-amz-meta-chart-type", "matplotlib"),
                        "strategy": meta.get("x-amz-meta-strategy", ""),
                        "content_type": stat.content_type or "image/png",
                        "size": stat.size,
                        "data": data,
                        "presigned_url": presigned_url,
                    })
                except Exception as e:
                    logger.error(f"Failed to read object {obj.object_name}: {e}")

        except S3Error as e:
            logger.error(f"MinIO list_objects failed for run {run_id}: {e}")

        return results

    def get_image_as_base64(self, object_name: str) -> Optional[str]:
        """
        Get a single image as a base64 string.

        Args:
            object_name: Full object path in the bucket

        Returns:
            Base64-encoded string or None
        """
        if not self.client:
            return None
        try:
            response = self.client.get_object(
                self._config.bucket_name, object_name
            )
            data = response.read()
            response.close()
            response.release_conn()
            return base64.b64encode(data).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to get image {object_name}: {e}")
            return None

    def delete_run_images(self, run_id: str) -> int:
        """
        Delete all images for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Number of objects deleted
        """
        if not self.client:
            return 0

        try:
            objects = self.client.list_objects(
                self._config.bucket_name,
                prefix=f"{run_id}/",
                recursive=True,
            )
            delete_list = [DeleteObject(obj.object_name) for obj in objects]

            if delete_list:
                errors = list(
                    self.client.remove_objects(
                        self._config.bucket_name, delete_list
                    )
                )
                if errors:
                    for err in errors:
                        logger.error(f"Delete error: {err}")
                deleted = len(delete_list) - len(errors)
                logger.info(f"Deleted {deleted} objects for run {run_id}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Failed to delete run images: {e}")
            return 0

    def list_runs(self) -> List[str]:
        """
        List all run_ids that have stored images.

        Returns:
            Sorted list of run_id prefixes
        """
        if not self.client:
            return []

        try:
            objects = self.client.list_objects(
                self._config.bucket_name, recursive=False
            )
            return sorted(
                obj.object_name.rstrip("/")
                for obj in objects
                if obj.is_dir
            )
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            return []

    def list_runs_detailed(self) -> List[Dict[str, Any]]:
        """
        List all runs with detailed metadata (size, chart count, strategies, tickers).

        Returns:
            List of dicts with keys: run_id, total_size, chart_count,
            strategies, tickers, created_at
        """
        if not self.client:
            return []

        try:
            runs = self.list_runs()
            details: List[Dict[str, Any]] = []

            for run_id in runs:
                objects = list(self.client.list_objects(
                    self._config.bucket_name,
                    prefix=f"{run_id}/",
                    recursive=True,
                ))

                total_size = sum(obj.size or 0 for obj in objects)
                chart_count = len(objects)
                strategies: set = set()
                tickers: set = set()
                earliest = None

                for obj in objects:
                    # Path: run_id/ticker/strategy/filename
                    parts = obj.object_name.split("/")
                    if len(parts) >= 4:
                        tickers.add(parts[1].upper())
                        strategies.add(parts[2].replace("_", " ").title())
                    elif len(parts) >= 3:
                        strategies.add(parts[1].replace("_", " ").title())

                    if obj.last_modified:
                        if earliest is None or obj.last_modified < earliest:
                            earliest = obj.last_modified

                details.append({
                    "run_id": run_id,
                    "total_size": total_size,
                    "chart_count": chart_count,
                    "strategies": sorted(strategies),
                    "tickers": sorted(tickers),
                    "created_at": earliest,
                })

            return details
        except Exception as e:
            logger.error(f"Failed to list detailed runs: {e}")
            return []


# ------------------------------------------------------------------
# Module-level singleton accessor
# ------------------------------------------------------------------

_service_instance: Optional[MinIOService] = None


def get_minio_service() -> MinIOService:
    """Get the global MinIOService singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = MinIOService()
    return _service_instance
