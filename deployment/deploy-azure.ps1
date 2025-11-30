# Azure Container Instances Deployment

# Variables - Update these with your values
$RESOURCE_GROUP = "algo-trading-rg"
$LOCATION = "eastus"
$ACR_NAME = "algotradingacr"
$CONTAINER_NAME = "algo-trading-app"
$DNS_NAME = "algo-trading-system"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Deploying to Azure Container Instances" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Change to parent directory for build context
$parentDir = Split-Path -Parent $PSScriptRoot
Set-Location $parentDir

# Login to Azure
Write-Host "`nLogging in to Azure..." -ForegroundColor Yellow
az login

# Create resource group
Write-Host "`nCreating resource group..." -ForegroundColor Yellow
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
Write-Host "`nCreating Azure Container Registry..." -ForegroundColor Yellow
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Get ACR credentials
$ACR_USERNAME = az acr credential show --name $ACR_NAME --query username --output tsv
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv

# Build and push image to ACR
Write-Host "`nBuilding and pushing image to ACR..." -ForegroundColor Yellow
az acr build --registry $ACR_NAME --file deployment/Dockerfile --image algo-trading-system:latest .

# Deploy to Azure Container Instances
Write-Host "`nDeploying to Azure Container Instances..." -ForegroundColor Yellow
az container create `
    --resource-group $RESOURCE_GROUP `
    --name $CONTAINER_NAME `
    --image "$ACR_NAME.azurecr.io/algo-trading-system:latest" `
    --dns-name-label $DNS_NAME `
    --ports 8501 `
    --cpu 2 `
    --memory 4 `
    --registry-login-server "$ACR_NAME.azurecr.io" `
    --registry-username $ACR_USERNAME `
    --registry-password $ACR_PASSWORD `
    --restart-policy Always

# Get the FQDN
$FQDN = az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.fqdn --output tsv

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "`nYour application is available at:" -ForegroundColor Yellow
Write-Host "http://${FQDN}:8501" -ForegroundColor Cyan
Write-Host "`n==================================================" -ForegroundColor Green
