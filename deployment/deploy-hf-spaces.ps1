# Centurion Core — Deploy Backend to Hugging Face Spaces
# Pulls from the MAIN branch of the GitHub repo (not local working directory).
#
# Prerequisites:
#   1. pip install huggingface_hub
#   2. huggingface-cli login (paste your HF token)
#   3. Fill in your HF username below
#   4. auth/credentials.yaml must exist locally (it's in .gitignore on GitHub)
#   5. .env file at project root with all secrets (CENTURION_DATABASE_URL, etc.)

$HF_USERNAME = "srees16"
$SPACE_NAME  = "centurion-core"
$GITHUB_REPO = "https://github.com/srees16/centurion_core.git"
$GITHUB_BRANCH = "main"

# Resolve project root (script lives in deployment/)
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Write-Host "=== Deploying Centurion Core Backend to HF Spaces ===" -ForegroundColor Cyan
Write-Host "  Source: $GITHUB_REPO @ $GITHUB_BRANCH" -ForegroundColor Gray

# Step 1: Create Space repo (Docker SDK) — safe to re-run, ignores if exists
Write-Host "`n[1/7] Creating HF Space..." -ForegroundColor Yellow
huggingface-cli repo create "$SPACE_NAME" --type space --space_sdk docker 2>$null

# Step 2: Clone the Space repo (HF Spaces)
$TEMP_DIR = "$env:TEMP\hf-deploy-centurion"
if (Test-Path $TEMP_DIR) { Remove-Item -Recurse -Force $TEMP_DIR }
git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" $TEMP_DIR
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Failed to clone Space repo. Is huggingface-cli logged in?" -ForegroundColor Red; exit 1 }

# Step 3: Clone the GitHub source repo (main branch, shallow)
Write-Host "[2/7] Cloning GitHub repo ($GITHUB_BRANCH branch)..." -ForegroundColor Yellow
$GITHUB_DIR = "$env:TEMP\hf-github-centurion"
if (Test-Path $GITHUB_DIR) { Remove-Item -Recurse -Force $GITHUB_DIR }
git clone --depth 1 --branch $GITHUB_BRANCH $GITHUB_REPO $GITHUB_DIR
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Failed to clone GitHub repo ($GITHUB_BRANCH branch)." -ForegroundColor Red; exit 1 }

# Step 4: Copy application files from GitHub clone (exclude dev/large artifacts)
Write-Host "[3/7] Copying application files from GitHub $GITHUB_BRANCH..." -ForegroundColor Yellow

# Excluded directories and files (same rationale as before)
robocopy $GITHUB_DIR $TEMP_DIR /S /XD `
    __pycache__ .git myenv node_modules .next frontend chroma_store `
    financial_ML\_cache financial_ML\_output `
    ingest_docs rag_uploads bhavcopy_cache chroma_db event_logs rl_models `
    /XF .env .env.* *.pyc *.pdf *.sqlite3 *_original.png *_original.jpg _test_* _debug_* _compare_* | Out-Null

# Step 5: Inject files not in GitHub (gitignored secrets + HF-specific files)
Write-Host "[4/7] Injecting deployment files..." -ForegroundColor Yellow
# auth/credentials.yaml is in .gitignore on GitHub — copy from local project
$CRED_SRC = Join-Path $PROJECT_ROOT "auth\credentials.yaml"
$CRED_DST = Join-Path $TEMP_DIR "auth"
if (Test-Path $CRED_SRC) {
    if (-not (Test-Path $CRED_DST)) { New-Item -ItemType Directory -Path $CRED_DST -Force | Out-Null }
    Copy-Item $CRED_SRC "$CRED_DST\credentials.yaml" -Force
    Write-Host "  Injected auth/credentials.yaml from local project" -ForegroundColor Green
} else {
    Write-Host "ERROR: auth/credentials.yaml not found at $CRED_SRC" -ForegroundColor Red
    Write-Host "  This file is required for authentication. Create it first." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $TEMP_DIR 2>$null
    Remove-Item -Recurse -Force $GITHUB_DIR 2>$null
    exit 1
}
# HF-specific Dockerfile and README
Copy-Item "$GITHUB_DIR\deployment\Dockerfile.hf" "$TEMP_DIR\Dockerfile" -Force
Copy-Item "$GITHUB_DIR\deployment\hf-spaces\README.md" "$TEMP_DIR\README.md" -Force

# Step 6: Validate — check no file exceeds 10 MB
Write-Host "[5/7] Validating file sizes (HF limit: 10 MB)..." -ForegroundColor Yellow
$largeFiles = Get-ChildItem -Path $TEMP_DIR -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -gt 10MB -and $_.DirectoryName -notlike "*\.git*" }
if ($largeFiles) {
    Write-Host "ERROR: Files exceeding 10 MB found:" -ForegroundColor Red
    foreach ($f in $largeFiles) {
        $sizeMB = [math]::Round($f.Length / 1MB, 1)
        Write-Host "  $sizeMB MB  $($f.FullName)" -ForegroundColor Red
    }
    Write-Host "Add exclusions to the robocopy command above, then re-run." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $TEMP_DIR 2>$null
    Remove-Item -Recurse -Force $GITHUB_DIR 2>$null
    exit 1
}
Write-Host "  All files under 10 MB - OK" -ForegroundColor Green

# Step 7: Push to HF Spaces
Write-Host "[6/7] Pushing to HF Spaces..." -ForegroundColor Yellow
Push-Location $TEMP_DIR

# Minimal .gitignore so auth/credentials.yaml gets included
Set-Content -Path ".gitignore" -Value "__pycache__/`n*.pyc`n.env`n.env.*"

git add -A
git commit -m "Deploy Centurion Core backend (from GitHub $GITHUB_BRANCH)"
git push
$pushResult = $LASTEXITCODE
Pop-Location

if ($pushResult -ne 0) {
    Write-Host "ERROR: git push failed. Check output above." -ForegroundColor Red
    Remove-Item -Recurse -Force $TEMP_DIR 2>$null
    Remove-Item -Recurse -Force $GITHUB_DIR 2>$null
    exit 1
}

Write-Host "`n[7/7] Deployment initiated!" -ForegroundColor Green
Write-Host "  Source:    $GITHUB_REPO @ $GITHUB_BRANCH" -ForegroundColor Cyan
Write-Host "  Space URL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor Cyan
Write-Host "  API URL:   https://$HF_USERNAME-$SPACE_NAME.hf.space" -ForegroundColor Cyan

# Step 8: Sync secrets from local .env to HF Spaces via huggingface_hub API
Write-Host "`n[8/8] Syncing secrets from .env to HF Spaces..." -ForegroundColor Yellow

$ENV_FILE = Join-Path $PROJECT_ROOT ".env"
if (Test-Path $ENV_FILE) {
    # Secrets that must be set on HF Spaces for the app to function
    $REQUIRED_SECRETS = @(
        "CENTURION_DATABASE_URL",
        "DATABASE_URL",
        "ANTHROPIC_API_KEY",
        "SENTRY_DSN",
        "LOGTAIL_TOKEN",
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
        "CENTURION_REDIS_URL",
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN"
    )

    $envVars = @{}
    Get-Content $ENV_FILE | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $eqIdx = $line.IndexOf("=")
            $key = $line.Substring(0, $eqIdx).Trim()
            $val = $line.Substring($eqIdx + 1).Trim().Trim('"').Trim("'")
            $envVars[$key] = $val
        }
    }

    $syncCount = 0
    foreach ($secret in $REQUIRED_SECRETS) {
        if ($envVars.ContainsKey($secret) -and $envVars[$secret]) {
            $val = $envVars[$secret]
            # Use huggingface_hub Python API to set secrets (CLI doesn't support it directly)
            python -c "
from huggingface_hub import add_space_secret
try:
    add_space_secret('$HF_USERNAME/$SPACE_NAME', '$secret', '$val')
    print(f'  Set {\"$secret\"} OK')
except Exception as e:
    print(f'  WARN: Failed to set {\"$secret\"}: {e}')
"
            $syncCount++
        }
    }
    Write-Host "  Synced $syncCount secrets to HF Spaces" -ForegroundColor Green
} else {
    Write-Host "  WARNING: .env file not found at $ENV_FILE" -ForegroundColor Red
    Write-Host "  Secrets must be set manually in HF Spaces Settings > Repository secrets:" -ForegroundColor Yellow
    Write-Host "  CENTURION_DATABASE_URL, ANTHROPIC_API_KEY,"
    Write-Host "  ZERODHA_API_KEY, ZERODHA_API_SECRET, ZERODHA_USER_ID, ZERODHA_PASSWORD, ZERODHA_TOTP_SECRET,"
    Write-Host "  MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE=true, MINIO_BUCKET=centurion-backtests, MINIO_ENABLED=true,"
    Write-Host "  CENTURION_DEFAULT_ADMIN_PASSWORD, CENTURION_DEFAULT_ANALYST_PASSWORD,"
    Write-Host "  CENTURION_ALLOWED_ORIGINS=https://centurion-core-fe.vercel.app,"
    Write-Host "  CENTURION_RAG_LLM_PROVIDER=claude"
}

# Cleanup temp dirs
Remove-Item -Recurse -Force $TEMP_DIR 2>$null
Remove-Item -Recurse -Force $GITHUB_DIR 2>$null
