"""
Model Registry — Versioned catalogue for ML / sentiment models.

Tracks which models are loaded, their versions, and provides
lazy-loading with thread-safe singleton semantics so that heavy
models (transformers, scikit-learn) are loaded exactly once.

Usage::

    from infrastructure.model_registry import model_registry

    # Register
    model_registry.register("sentiment", loader=lambda: load_finbert(), version="1.0")

    # Retrieve (loads on first access)
    pipeline = model_registry.get("sentiment")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    name: str
    version: str
    loader: Callable[[], Any]
    instance: Optional[Any] = None
    loaded_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Thread-safe lazy-loading model catalogue."""

    def __init__(self):
        self._models: Dict[str, ModelEntry] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        *,
        loader: Callable[[], Any],
        version: str = "0.1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._models[name] = ModelEntry(
                name=name,
                version=version,
                loader=loader,
                metadata=metadata or {},
            )
        logger.info("Registered model '%s' v%s", name, version)

    def get(self, name: str) -> Any:
        """Get or lazily load a model by name."""
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                raise KeyError(f"Model '{name}' not registered")
            if entry.instance is not None:
                return entry.instance

        # Load outside lock to avoid blocking other models
        instance = entry.loader()
        with self._lock:
            entry.instance = instance
            entry.loaded_at = datetime.now(timezone.utc)
        logger.info("Loaded model '%s' v%s", name, entry.version)
        return instance

    def is_loaded(self, name: str) -> bool:
        with self._lock:
            entry = self._models.get(name)
            return entry is not None and entry.instance is not None

    def unload(self, name: str) -> None:
        with self._lock:
            entry = self._models.get(name)
            if entry:
                entry.instance = None
                entry.loaded_at = None
        logger.info("Unloaded model '%s'", name)

    def list_models(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name": e.name,
                    "version": e.version,
                    "loaded": e.instance is not None,
                    "loaded_at": e.loaded_at.isoformat() if e.loaded_at else None,
                    **e.metadata,
                }
                for e in self._models.values()
            ]


model_registry = ModelRegistry()
