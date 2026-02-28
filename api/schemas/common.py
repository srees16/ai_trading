"""
Common Pydantic schemas shared across all API modules.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Generic response wrappers
# ---------------------------------------------------------------------------

class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str = "OK"
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Generic error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """System health check response."""
    status: str = Field(..., examples=["healthy"])
    database: bool = False
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, bool] = Field(default_factory=dict)


class PaginationParams(BaseModel):
    """Common pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(50, ge=1, le=500, description="Items per page")


class PaginatedResponse(BaseModel):
    """Paginated list response."""
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int
