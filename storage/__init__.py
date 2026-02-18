"""
Storage module.
"""

from storage.manager import StorageManager

try:
    from storage.minio_service import MinIOService, get_minio_service
except ImportError:
    MinIOService = None
    get_minio_service = None

__all__ = ['StorageManager', 'MinIOService', 'get_minio_service']
