# Build and Run Script for AI Trading System (Windows PowerShell)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "AI Trading Alert System - Docker Deployment" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Change to parent directory for build context
$parentDir = Split-Path -Parent $PSScriptRoot
Set-Location $parentDir

# Build the Docker image
Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
docker build -f deployment/Dockerfile -t algo-trading-system:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Docker image built successfully!" -ForegroundColor Green
    
    # Stop and remove existing container if running
    Write-Host "`nStopping existing container (if any)..." -ForegroundColor Yellow
    docker stop algo-trading-system 2>$null
    docker rm algo-trading-system 2>$null
    
    # Run the container
    Write-Host "Starting container..." -ForegroundColor Yellow
    docker run -d `
        --name algo-trading-system `
        -p 8501:8501 `
        -v "${parentDir}/data:/app/data" `
        algo-trading-system:latest
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Container started successfully!" -ForegroundColor Green
        Write-Host "`n==================================================" -ForegroundColor Cyan
        Write-Host "Application is running at: http://localhost:8501" -ForegroundColor Green
        Write-Host "==================================================" -ForegroundColor Cyan
        Write-Host "`nUseful commands:" -ForegroundColor Yellow
        Write-Host "  View logs:     docker logs -f algo-trading-system"
        Write-Host "  Stop:          docker stop algo-trading-system"
        Write-Host "  Restart:       docker restart algo-trading-system"
        Write-Host "  Remove:        docker rm -f algo-trading-system"
    }
    else {
        Write-Host "`n✗ Failed to start container" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "`n✗ Failed to build Docker image" -ForegroundColor Red
    exit 1
}
