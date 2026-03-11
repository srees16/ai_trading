"""
FastAPI Application Factory for Centurion Capital Trading Platform.

Creates and configures the root FastAPI app with:
    - All module routers (US stocks, Indian stocks, RAG, Crypto)
    - CORS middleware
    - Lifespan for startup/shutdown hooks
    - Exception handlers
    - Auth-gated /docs, /redoc, /openapi.json
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

# Ensure project root is on sys.path so all internal imports resolve
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.auth import (
    LOGIN_PAGE_HTML,
    SESSION_COOKIE,
    authenticate_user,
    create_session_token,
    verify_session_token,
)
from auth.shared_session import (
    SHARED_COOKIE_MAX_AGE,
    SHARED_COOKIE_NAME,
    create_shared_token,
    verify_shared_token,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _get_authenticated_user(request: Request) -> dict | None:
    """Return the decoded session payload or None if unauthenticated."""
    # Check API-specific cookie first
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        result = verify_session_token(token)
        if result:
            return result
    # Fall back to shared SSO cookie (set by Streamlit)
    shared = request.cookies.get(SHARED_COOKIE_NAME)
    if shared:
        return verify_shared_token(shared)
    return None


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise heavyweight singletons at startup; tear down on shutdown.
    """
    logger.info("Centurion API starting up")

    # Pre-warm database connection (optional, fast)
    try:
        from api.dependencies import get_db_service
        db = get_db_service()
        if db:
            logger.info("Database connection OK")
        else:
            logger.warning("Database not available — DB-dependent endpoints will 503")
    except Exception as exc:
        logger.warning("Database init skipped: %s", exc)

    logger.info("Centurion API ready")
    yield
    logger.info("Centurion API shutting down ...")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the FastAPI application."""

    # Disable built-in docs routes — we serve our own auth-gated versions
    app = FastAPI(
        title="Centurion Capital LLC API",
        description=(
            "RESTful API for the Centurion Capital algorithmic trading "
            "platform — US stocks analysis, Indian stocks (Zerodha Kite), "
            "RAG pipeline, and crypto mean-reversion strategies."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],         # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers ---
    from api.routers.health import router as health_router
    from api.routers.us_stocks import router as us_stocks_router
    from api.routers.ind_stocks import router as ind_stocks_router
    from api.routers.rag import router as rag_router
    from api.routers.crypto import router as crypto_router
    from api.routers.streaming import router as streaming_router

    app.include_router(health_router)
    app.include_router(us_stocks_router)
    app.include_router(ind_stocks_router)
    app.include_router(rag_router)
    app.include_router(crypto_router)
    app.include_router(streaming_router)

    # ------------------------------------------------------------------
    # Authentication endpoints
    # ------------------------------------------------------------------

    @app.get("/auth/login", include_in_schema=False)
    async def login_page(request: Request):
        """Serve the login form. If already authenticated, redirect to docs."""
        if _get_authenticated_user(request):
            return RedirectResponse(url="/docs", status_code=302)
        return HTMLResponse(LOGIN_PAGE_HTML)

    @app.post("/auth/login", include_in_schema=False)
    async def login(
        username: str = Form(...),
        password: str = Form(...),
    ):
        """Validate credentials, set a session cookie, redirect to docs."""
        ok, display_name, role = authenticate_user(username, password)
        if not ok:
            return JSONResponse(
                status_code=401,
                content={"success": False, "detail": "Invalid username or password"},
            )
        token = create_session_token(username, role)
        shared_token = create_shared_token(username, role)
        response = RedirectResponse(url="/docs", status_code=302)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=token,
            httponly=True,
            samesite="lax",
            max_age=28800,
        )
        # Shared SSO cookie (readable by Streamlit via JS)
        response.set_cookie(
            key=SHARED_COOKIE_NAME,
            value=shared_token,
            httponly=False,
            samesite="lax",
            path="/",
            max_age=SHARED_COOKIE_MAX_AGE,
        )
        logger.info("API docs login: user=%s role=%s", username, role)
        return response

    @app.get("/auth/logout", include_in_schema=False)
    async def logout():
        """Clear session cookies and redirect to the login page."""
        response = RedirectResponse(url="/auth/login", status_code=302)
        response.delete_cookie(SESSION_COOKIE)
        response.delete_cookie(SHARED_COOKIE_NAME, path="/")
        return response

    # ------------------------------------------------------------------
    # Auth-gated OpenAPI / Swagger / ReDoc routes
    # ------------------------------------------------------------------

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi_json(request: Request):
        if not _get_authenticated_user(request):
            return RedirectResponse(url="/auth/login", status_code=302)
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False)
    async def docs(request: Request):
        if not _get_authenticated_user(request):
            return RedirectResponse(url="/auth/login", status_code=302)
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=app.title + " — Swagger UI",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc(request: Request):
        if not _get_authenticated_user(request):
            return RedirectResponse(url="/auth/login", status_code=302)
        return get_redoc_html(
            openapi_url="/openapi.json",
            title=app.title + " — ReDoc",
        )


    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(exc),
            },
        )

    _FAVICON_PATH = Path(__file__).resolve().parent.parent / "ui" / "assets" / "centurion_logo.png"

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        if _FAVICON_PATH.is_file():
            return FileResponse(_FAVICON_PATH, media_type="image/png")
        return JSONResponse(status_code=204, content=None)


    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": "Centurion Capital LLC API",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
        }

    return app


# Allow `uvicorn api.main:app` to work directly
app = create_app()
