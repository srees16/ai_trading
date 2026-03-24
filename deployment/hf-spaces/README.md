---
title: Centurion Core API
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Centurion Core — Backend API

Algorithmic trading platform backend powered by FastAPI.

## Environment Variables (set in HF Spaces Settings → Secrets)

| Variable | Required | Description |
|----------|----------|-------------|
| `CENTURION_DATABASE_URL` | Yes | Neon PostgreSQL pooler URL |
| `ANTHROPIC_API_KEY` | Yes | Claude API key for RAG |
| `ZERODHA_API_KEY` | Yes | Kite Connect API key |
| `ZERODHA_API_SECRET` | Yes | Kite Connect secret |
| `ZERODHA_USER_ID` | Yes | Kite user ID |
| `ZERODHA_PASSWORD` | Yes | Kite password |
| `ZERODHA_TOTP_SECRET` | Yes | Kite TOTP secret |
| `MINIO_ENDPOINT` | Yes | Cloudflare R2 endpoint |
| `MINIO_ACCESS_KEY` | Yes | R2 access key |
| `MINIO_SECRET_KEY` | Yes | R2 secret key |
| `UPSTASH_REDIS_URL` | Optional | Upstash Redis URL |
| `CENTURION_DEFAULT_ADMIN_PASSWORD` | Yes | Admin login password |
| `CENTURION_DEFAULT_ANALYST_PASSWORD` | Yes | Analyst login password |
| `CENTURION_ALLOWED_ORIGINS` | Yes | Comma-separated frontend URLs for CORS |
