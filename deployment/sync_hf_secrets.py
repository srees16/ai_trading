"""Sync secrets from .env to HuggingFace Spaces."""
import os
from pathlib import Path
from huggingface_hub import add_space_secret

REPO_ID = "srees16/centurion-core"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

REQUIRED_SECRETS = [
    "CENTURION_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "SENTRY_DSN",
    "LOGTAIL_TOKEN",
    "ZERODHA_API_KEY",
    "ZERODHA_API_SECRET",
    "ZERODHA_USER_ID",
    "ZERODHA_PASSWORD",
    "ZERODHA_TOTP_SECRET",
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_SECURE",
    "MINIO_BUCKET",
    "MINIO_ENABLED",
    "CENTURION_DEFAULT_ADMIN_PASSWORD",
    "CENTURION_DEFAULT_ANALYST_PASSWORD",
    "CENTURION_ALLOWED_ORIGINS",
    "CENTURION_RAG_LLM_PROVIDER",
    "CENTURION_DB_ENABLED",
    "UPSTASH_REDIS_REST_URL",
    "UPSTASH_REDIS_REST_TOKEN",
]


def load_env(path: Path) -> dict:
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            idx = line.index("=")
            key = line[:idx].strip()
            val = line[idx + 1 :].strip().strip('"').strip("'")
            if val:
                env[key] = val
    return env


def main():
    if not ENV_FILE.exists():
        print(f"ERROR: .env not found at {ENV_FILE}")
        return

    env = load_env(ENV_FILE)
    count = 0

    for key in REQUIRED_SECRETS:
        if key in env:
            try:
                add_space_secret(REPO_ID, key, env[key])
                print(f"  OK: {key}")
                count += 1
            except Exception as e:
                print(f"  FAIL: {key} — {e}")
        else:
            print(f"  SKIP: {key} (not in .env)")

    print(f"\nSynced {count} / {len(REQUIRED_SECRETS)} secrets to {REPO_ID}")


if __name__ == "__main__":
    main()
