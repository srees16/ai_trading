<#
.SYNOPSIS
    Deploy R21a compute pipeline to Azure Container Instances.

.DESCRIPTION
    Creates an Azure File Share for persistent data, builds and pushes
    the compute Docker image, and deploys to ACI with 8 CPU / 32 GB RAM.
    Data persists across runs via Azure File Share.

.PARAMETER Step
    Pipeline step to run: extract | optimize | validate | pipeline (default)

.PARAMETER Action
    deploy  — Build image + deploy ACI container (default)
    upload  — Upload local data to Azure File Share
    download — Download results from Azure File Share
    logs    — Stream container logs
    status  — Check container status
    cleanup — Delete container (keeps data)
    destroy — Delete everything (container + storage + RG)

.EXAMPLE
    # First time: deploy full pipeline
    .\deploy-compute-azure.ps1 -Action deploy -Step pipeline

    # Upload data cache before running
    .\deploy-compute-azure.ps1 -Action upload

    # Monitor progress
    .\deploy-compute-azure.ps1 -Action logs

    # Download results after completion
    .\deploy-compute-azure.ps1 -Action download

    # Run optimizer only (after extraction is done)
    .\deploy-compute-azure.ps1 -Action deploy -Step optimize
#>

param(
    [ValidateSet("deploy", "upload", "download", "logs", "status", "cleanup", "destroy")]
    [string]$Action = "deploy",

    [ValidateSet("extract", "optimize", "validate", "pipeline")]
    [string]$Step = "pipeline"
)

# ── Configuration ──
$RESOURCE_GROUP   = "centurion-compute-rg"
$LOCATION         = "centralindia"    # Closest to NSE data sources
$ACR_NAME         = "centurionacr"
$STORAGE_ACCOUNT  = "centuriondata"
$FILE_SHARE       = "centurion-data"
$CONTAINER_NAME   = "centurion-r21a"
$IMAGE_TAG        = "centurion-r21a:latest"

# CPU/RAM per step
$CPU_MAP = @{
    "extract"  = 4;  "optimize" = 8;  "validate" = 4;  "pipeline" = 8
}
$RAM_MAP = @{
    "extract"  = 16; "optimize" = 32; "validate" = 16; "pipeline" = 32
}

$projectRoot = Split-Path -Parent $PSScriptRoot  # centurion_core/../

function Get-StorageKey {
    az storage account keys list `
        --resource-group $RESOURCE_GROUP `
        --account-name $STORAGE_ACCOUNT `
        --query "[0].value" -o tsv
}

# ══════════════════════════════════════════════════════════════
#  DEPLOY
# ══════════════════════════════════════════════════════════════
if ($Action -eq "deploy") {
    Write-Host "`n$('='*70)" -ForegroundColor Cyan
    Write-Host "  Deploying R21a Compute — Step: $Step" -ForegroundColor Cyan
    Write-Host "  CPU: $($CPU_MAP[$Step])  RAM: $($RAM_MAP[$Step]) GB" -ForegroundColor Cyan
    Write-Host "$('='*70)" -ForegroundColor Cyan

    Set-Location $projectRoot

    # 1. Resource Group
    Write-Host "`n  [1/5] Creating resource group..." -ForegroundColor Yellow
    az group create --name $RESOURCE_GROUP --location $LOCATION --output none

    # 2. Container Registry
    Write-Host "  [2/5] Creating container registry..." -ForegroundColor Yellow
    az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME `
        --sku Basic --admin-enabled true --output none 2>$null
    $ACR_USER = az acr credential show --name $ACR_NAME --query username -o tsv
    $ACR_PASS = az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv

    # 3. Storage Account + File Share
    Write-Host "  [3/5] Creating storage account + file share..." -ForegroundColor Yellow
    az storage account create --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP `
        --location $LOCATION --sku Standard_LRS --output none 2>$null
    $STORAGE_KEY = Get-StorageKey
    az storage share create --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY `
        --name $FILE_SHARE --quota 50 --output none 2>$null

    # 4. Build + push image
    Write-Host "  [4/5] Building and pushing Docker image..." -ForegroundColor Yellow
    az acr build --registry $ACR_NAME `
        --file centurion_core/deployment/Dockerfile.compute `
        --image $IMAGE_TAG .

    # 5. Delete old container if exists, deploy new
    Write-Host "  [5/5] Deploying container..." -ForegroundColor Yellow
    az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME `
        --yes --output none 2>$null

    # Load .env vars for container
    $envVars = @()
    if (Test-Path "$projectRoot\.env") {
        Get-Content "$projectRoot\.env" | Where-Object { $_ -match "^[A-Z]" } | ForEach-Object {
            $parts = $_ -split "=", 2
            if ($parts.Count -eq 2) {
                $envVars += "--environment-variables"
                $envVars += "$($parts[0])=$($parts[1])"
            }
        }
    }

    az container create `
        --resource-group $RESOURCE_GROUP `
        --name $CONTAINER_NAME `
        --image "$ACR_NAME.azurecr.io/$IMAGE_TAG" `
        --cpu $CPU_MAP[$Step] `
        --memory $RAM_MAP[$Step] `
        --registry-login-server "$ACR_NAME.azurecr.io" `
        --registry-username $ACR_USER `
        --registry-password $ACR_PASS `
        --azure-file-volume-account-name $STORAGE_ACCOUNT `
        --azure-file-volume-account-key $STORAGE_KEY `
        --azure-file-volume-share-name $FILE_SHARE `
        --azure-file-volume-mount-path /app/centurion_core/data `
        --environment-variables PIPELINE_STEP=$Step PYTHONUNBUFFERED=1 `
        --restart-policy Never `
        --output none

    Write-Host "`n$('='*70)" -ForegroundColor Green
    Write-Host "  Deployed! Container running step: $Step" -ForegroundColor Green
    Write-Host "  Monitor:  .\deploy-compute-azure.ps1 -Action logs" -ForegroundColor Yellow
    Write-Host "  Status:   .\deploy-compute-azure.ps1 -Action status" -ForegroundColor Yellow
    Write-Host "  Download: .\deploy-compute-azure.ps1 -Action download" -ForegroundColor Yellow
    Write-Host "$('='*70)" -ForegroundColor Green
}

# ══════════════════════════════════════════════════════════════
#  UPLOAD LOCAL DATA → AZURE FILE SHARE
# ══════════════════════════════════════════════════════════════
elseif ($Action -eq "upload") {
    Write-Host "`n  Uploading local data to Azure File Share..." -ForegroundColor Yellow
    $STORAGE_KEY = Get-StorageKey

    $dataDir = Join-Path $projectRoot "centurion_core" "data"

    # Upload key files
    $files = @(
        "extracted_forecasts.pkl",
        "earnings_cache.json",
        "fii_flow_cache.json",
        "nse_sector_map.json",
        "strategy_decay_state.json"
    )
    foreach ($f in $files) {
        $localPath = Join-Path $dataDir $f
        if (Test-Path $localPath) {
            Write-Host "  Uploading $f..." -ForegroundColor Cyan
            az storage file upload `
                --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY `
                --share-name $FILE_SHARE --source $localPath `
                --path $f --output none
        }
    }

    # Upload bhavcopy_cache directory
    $bhavDir = Join-Path $dataDir "bhavcopy_cache"
    if (Test-Path $bhavDir) {
        Write-Host "  Uploading bhavcopy_cache/ (this may take a few minutes)..." -ForegroundColor Cyan
        az storage file upload-batch `
            --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY `
            --destination $FILE_SHARE `
            --source $bhavDir `
            --destination-path "bhavcopy_cache" `
            --output none
    }

    Write-Host "`n  Upload complete!" -ForegroundColor Green
}

# ══════════════════════════════════════════════════════════════
#  DOWNLOAD RESULTS ← AZURE FILE SHARE
# ══════════════════════════════════════════════════════════════
elseif ($Action -eq "download") {
    Write-Host "`n  Downloading results from Azure File Share..." -ForegroundColor Yellow
    $STORAGE_KEY = Get-StorageKey
    $dataDir = Join-Path $projectRoot "centurion_core" "data"

    $results = @(
        "r21a_optimization_results.pkl",
        "backtest_checkpoint_r21a.pkl",
        "backtest_checkpoint_extract.pkl"
    )
    foreach ($f in $results) {
        Write-Host "  Downloading $f..." -ForegroundColor Cyan
        az storage file download `
            --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY `
            --share-name $FILE_SHARE --path $f `
            --dest (Join-Path $dataDir $f) `
            --output none 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    ✓ $f" -ForegroundColor Green
        } else {
            Write-Host "    ✗ $f (not found)" -ForegroundColor Red
        }
    }
    Write-Host "`n  Download complete!" -ForegroundColor Green
}

# ══════════════════════════════════════════════════════════════
#  LOGS
# ══════════════════════════════════════════════════════════════
elseif ($Action -eq "logs") {
    Write-Host "  Streaming container logs (Ctrl+C to stop)..." -ForegroundColor Yellow
    az container logs --resource-group $RESOURCE_GROUP `
        --name $CONTAINER_NAME --follow
}

# ══════════════════════════════════════════════════════════════
#  STATUS
# ══════════════════════════════════════════════════════════════
elseif ($Action -eq "status") {
    $state = az container show --resource-group $RESOURCE_GROUP `
        --name $CONTAINER_NAME `
        --query "{state:instanceView.state,startTime:containers[0].instanceView.currentState.startTime,exitCode:containers[0].instanceView.currentState.exitCode,cpu:containers[0].resources.requests.cpu,memoryGB:containers[0].resources.requests.memoryInGB}" `
        -o json 2>$null | ConvertFrom-Json

    if ($state) {
        Write-Host "`n  Container Status:" -ForegroundColor Cyan
        Write-Host "    State:     $($state.state)"
        Write-Host "    Started:   $($state.startTime)"
        Write-Host "    Exit Code: $($state.exitCode)"
        Write-Host "    CPU:       $($state.cpu)"
        Write-Host "    Memory:    $($state.memoryGB) GB"
    } else {
        Write-Host "  Container not found." -ForegroundColor Red
    }
}

# ══════════════════════════════════════════════════════════════
#  CLEANUP (stop container, keep data)
# ══════════════════════════════════════════════════════════════
elseif ($Action -eq "cleanup") {
    Write-Host "  Deleting container (data preserved in File Share)..." -ForegroundColor Yellow
    az container delete --resource-group $RESOURCE_GROUP `
        --name $CONTAINER_NAME --yes --output none
    Write-Host "  Container deleted. Data still in Azure File Share." -ForegroundColor Green
}

# ══════════════════════════════════════════════════════════════
#  DESTROY (delete everything)
# ══════════════════════════════════════════════════════════════
elseif ($Action -eq "destroy") {
    Write-Host "  WARNING: This will delete the resource group and ALL resources!" -ForegroundColor Red
    $confirm = Read-Host "  Type 'yes' to confirm"
    if ($confirm -eq "yes") {
        az group delete --name $RESOURCE_GROUP --yes --no-wait
        Write-Host "  Resource group deletion initiated." -ForegroundColor Yellow
    } else {
        Write-Host "  Cancelled." -ForegroundColor Green
    }
}
