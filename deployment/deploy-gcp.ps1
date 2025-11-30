# Google Cloud Run Deployment

# Variables - Update these with your values
$PROJECT_ID = "your-project-id"
$REGION = "us-central1"
$SERVICE_NAME = "algo-trading-system"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/algo-trading-system"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Deploying to Google Cloud Run" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Change to parent directory for build context
$parentDir = Split-Path -Parent $PSScriptRoot
Set-Location $parentDir

# Set the project
Write-Host "`nSetting GCP project..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID

# Enable required APIs
Write-Host "`nEnabling required APIs..." -ForegroundColor Yellow
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the container image
Write-Host "`nBuilding container image..." -ForegroundColor Yellow
gcloud builds submit --tag $IMAGE_NAME --file deployment/Dockerfile .

# Deploy to Cloud Run
Write-Host "`nDeploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $SERVICE_NAME `
    --image $IMAGE_NAME `
    --platform managed `
    --region $REGION `
    --allow-unauthenticated `
    --port 8501 `
    --memory 4Gi `
    --cpu 2 `
    --timeout 3600 `
    --max-instances 10 `
    --min-instances 0

# Get the service URL
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format "value(status.url)"

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "`nYour application is available at:" -ForegroundColor Yellow
Write-Host $SERVICE_URL -ForegroundColor Cyan
Write-Host "`n==================================================" -ForegroundColor Green
