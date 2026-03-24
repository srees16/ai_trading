# Centurion Core — Deploy Backend to Hugging Face Spaces
# Prerequisites: 
#   1. pip install huggingface_hub
#   2. huggingface-cli login (paste your HF token)
#   3. Fill in your HF username below

$HF_USERNAME = "srees16"  # <-- CHANGE THIS
$SPACE_NAME  = "centurion-core"

Write-Host "=== Deploying Centurion Core Backend to HF Spaces ===" -ForegroundColor Cyan

# Step 1: Create Space repo (Docker SDK)
Write-Host "`n[1/5] Creating HF Space..." -ForegroundColor Yellow
huggingface-cli repo create "$SPACE_NAME" --type space --space_sdk docker 2>$null

# Step 2: Clone the Space repo
$TEMP_DIR = "$env:TEMP\hf-deploy-centurion"
if (Test-Path $TEMP_DIR) { Remove-Item -Recurse -Force $TEMP_DIR }
git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" $TEMP_DIR

# Step 3: Copy application files (exclude dev artifacts)
Write-Host "[2/5] Copying application files..." -ForegroundColor Yellow
$SOURCE = (Get-Location).Path

# Use robocopy to exclude unnecessary dirs
robocopy $SOURCE $TEMP_DIR /S /XD `
    __pycache__ .git myenv node_modules .next frontend chroma_store `
    financial_ML\_cache financial_ML\_output data\bhavcopy_cache `
    /XF .env .env.* *.pyc _test_* _debug_* _compare_* | Out-Null

# Step 4: Copy the HF-specific Dockerfile as the root Dockerfile
Write-Host "[3/5] Setting up HF Spaces Dockerfile..." -ForegroundColor Yellow
Copy-Item "$SOURCE\deployment\Dockerfile.hf" "$TEMP_DIR\Dockerfile" -Force
Copy-Item "$SOURCE\deployment\hf-spaces\README.md" "$TEMP_DIR\README.md" -Force

# Step 5: Push to HF Spaces
Write-Host "[4/5] Pushing to HF Spaces..." -ForegroundColor Yellow
Push-Location $TEMP_DIR
git add -A
git commit -m "Deploy Centurion Core backend"
git push
Pop-Location

Write-Host "`n[5/5] Deployment initiated!" -ForegroundColor Green
Write-Host "  Space URL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor Cyan
Write-Host "  API URL:   https://$HF_USERNAME-$SPACE_NAME.hf.space" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: Set your secrets in HF Spaces Settings:" -ForegroundColor Red
Write-Host "  - CENTURION_DATABASE_URL (Neon PostgreSQL)"
Write-Host "  - ANTHROPIC_API_KEY"
Write-Host "  - ZERODHA_API_KEY, ZERODHA_API_SECRET, ZERODHA_USER_ID, ZERODHA_PASSWORD, ZERODHA_TOTP_SECRET"
Write-Host "  - MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY (Cloudflare R2)"
Write-Host "  - UPSTASH_REDIS_URL"
Write-Host "  - CENTURION_DEFAULT_ADMIN_PASSWORD, CENTURION_DEFAULT_ANALYST_PASSWORD"
Write-Host "  - CENTURION_ALLOWED_ORIGINS (your Vercel URL)"

# Cleanup
Remove-Item -Recurse -Force $TEMP_DIR 2>$null
