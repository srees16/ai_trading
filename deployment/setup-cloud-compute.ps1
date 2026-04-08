<#
.SYNOPSIS
    One-time setup for cloud compute infrastructure.

.DESCRIPTION
    Installs Modal CLI, authenticates, creates persistent volume,
    and uploads local data cache for R21a pipeline cloud execution.

.PARAMETER Platform
    modal  — Setup Modal serverless (recommended, default)
    azure  — Setup Azure ACI compute
    both   — Setup both platforms

.EXAMPLE
    .\setup-cloud-compute.ps1 -Platform modal
    .\setup-cloud-compute.ps1 -Platform azure
    .\setup-cloud-compute.ps1 -Platform both
#>

param(
    [ValidateSet("modal", "azure", "both")]
    [string]$Platform = "modal"
)

$projectRoot = $PSScriptRoot | Split-Path | Split-Path  # centurion_core/deployment/../../
$dataDir = Join-Path $projectRoot "centurion_core" "data"

Write-Host "`n$('='*70)" -ForegroundColor Cyan
Write-Host "  Centurion Cloud Compute Setup" -ForegroundColor Cyan
Write-Host "  Platform: $Platform" -ForegroundColor Cyan
Write-Host "$('='*70)`n" -ForegroundColor Cyan


# ══════════════════════════════════════════════════════════════
#  MODAL SETUP
# ══════════════════════════════════════════════════════════════
if ($Platform -in @("modal", "both")) {
    Write-Host "  ── Modal Setup ──" -ForegroundColor Yellow

    # 1. Install Modal
    Write-Host "  [1/4] Installing Modal CLI..." -ForegroundColor Cyan
    & python -m pip install --upgrade modal 2>&1 | Select-Object -Last 1

    # 2. Authenticate
    Write-Host "  [2/4] Authenticating with Modal..." -ForegroundColor Cyan
    Write-Host "         (Browser will open for login)" -ForegroundColor Gray
    & modal token new

    # 3. Create volume
    Write-Host "  [3/4] Creating persistent volume..." -ForegroundColor Cyan
    & modal volume create centurion-data 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "         Volume 'centurion-data' already exists." -ForegroundColor Gray
    }

    # 4. Upload data
    Write-Host "  [4/4] Uploading data to Modal volume..." -ForegroundColor Cyan

    # Upload bhavcopy cache (largest, ~142 MB)
    $bhavDir = Join-Path $dataDir "bhavcopy_cache"
    if (Test-Path $bhavDir) {
        Write-Host "         Uploading bhavcopy_cache/ (~142 MB)..." -ForegroundColor Gray
        & modal volume put centurion-data $bhavDir bhavcopy_cache/
    }

    # Upload extracted forecasts if available
    $forecastsFile = Join-Path $dataDir "extracted_forecasts.pkl"
    if (Test-Path $forecastsFile) {
        Write-Host "         Uploading extracted_forecasts.pkl (~63 MB)..." -ForegroundColor Gray
        & modal volume put centurion-data $forecastsFile extracted_forecasts.pkl
    }

    # Upload supporting files
    $supportFiles = @(
        "earnings_cache.json",
        "fii_flow_cache.json",
        "nse_sector_map.json",
        "strategy_decay_state.json",
        "calibrated_scalars.json"
    )
    foreach ($f in $supportFiles) {
        $path = Join-Path $dataDir $f
        if (Test-Path $path) {
            Write-Host "         Uploading $f..." -ForegroundColor Gray
            & modal volume put centurion-data $path $f
        }
    }

    Write-Host "`n  ✓ Modal setup complete!" -ForegroundColor Green
    Write-Host "  Run optimizer:  modal run run_cloud_modal.py --step optimize" -ForegroundColor Yellow
    Write-Host "  Full pipeline:  modal run run_cloud_modal.py --step pipeline" -ForegroundColor Yellow
    Write-Host "  Check status:   modal run run_cloud_modal.py --step check" -ForegroundColor Yellow
}


# ══════════════════════════════════════════════════════════════
#  AZURE SETUP
# ══════════════════════════════════════════════════════════════
if ($Platform -in @("azure", "both")) {
    Write-Host "`n  ── Azure Setup ──" -ForegroundColor Yellow

    # 1. Check Azure CLI
    Write-Host "  [1/3] Checking Azure CLI..." -ForegroundColor Cyan
    $azVersion = az version 2>$null | ConvertFrom-Json
    if (-not $azVersion) {
        Write-Host "         Azure CLI not found! Install from: https://aka.ms/installazurecli" -ForegroundColor Red
        exit 1
    }
    Write-Host "         Azure CLI $($azVersion.'azure-cli')" -ForegroundColor Gray

    # 2. Login
    Write-Host "  [2/3] Authenticating with Azure..." -ForegroundColor Cyan
    az login

    # 3. Run deployment with upload
    Write-Host "  [3/3] Running initial deployment + data upload..." -ForegroundColor Cyan
    $deployScript = Join-Path $PSScriptRoot "deploy-compute-azure.ps1"
    & $deployScript -Action upload

    Write-Host "`n  ✓ Azure setup complete!" -ForegroundColor Green
    Write-Host "  Deploy:   .\deployment\deploy-compute-azure.ps1 -Action deploy -Step optimize" -ForegroundColor Yellow
    Write-Host "  Monitor:  .\deployment\deploy-compute-azure.ps1 -Action logs" -ForegroundColor Yellow
    Write-Host "  Download: .\deployment\deploy-compute-azure.ps1 -Action download" -ForegroundColor Yellow
}


Write-Host "`n$('='*70)" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "$('='*70)`n" -ForegroundColor Green
