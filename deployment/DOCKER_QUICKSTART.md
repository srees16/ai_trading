# Quick Start - Docker Deployment

## Deploy Locally with Docker

### Option 1: Quick Deploy (Recommended)

**Windows PowerShell:**
```powershell
.\deployment\deploy.ps1
```

**macOS / Linux:**
```bash
bash deployment/deploy.sh
```

Access API at: http://localhost:9001 — API docs at http://localhost:9001/docs

### Option 2: Docker Compose
```bash
cd deployment
docker-compose up -d
```

### Option 3: Manual Docker Commands

**Windows PowerShell:**
```powershell
docker build -f deployment/Dockerfile -t algo-trading-system:latest .
docker run -d -p 9001:9001 -e PORT=9001 --name algo-trading-system algo-trading-system:latest
docker logs -f algo-trading-system
```

**macOS / Linux:**
```bash
docker build -f deployment/Dockerfile -t algo-trading-system:latest .
docker run -d -p 9001:9001 -e PORT=9001 --name algo-trading-system algo-trading-system:latest
docker logs -f algo-trading-system
```

## Deploy to Cloud

### HF Spaces (Production)
```bash
# Deploy via GitHub Actions CI/CD (auto-deploys on push to main)
# Or manually:
.\deployment\deploy-hf-spaces.ps1
```
Secrets are managed in HF Space Settings → Repository secrets.

### Azure
```powershell
# Update variables in deployment/deploy-azure.ps1 first
.\deployment\deploy-azure.ps1
```

### Google Cloud
```powershell
# Update PROJECT_ID in deployment/deploy-gcp.ps1 first
.\deployment\deploy-gcp.ps1
```

## Verify Deployment

### Check Container Status
```bash
docker ps
```

### Health Check
```bash
curl http://localhost:9001/api/v1/health
```

### View Logs
```bash
docker logs -f algo-trading-system
```

### Access Application
- Local API: http://localhost:9001
- API Docs: http://localhost:9001/docs
- Azure: http://<dns-name>.<region>.azurecontainer.io:9001
- GCP: Provided after deployment

## Stop and Clean Up

**Windows PowerShell:**
```powershell
docker stop algo-trading-system
docker rm algo-trading-system
docker rmi algo-trading-system:latest
```

**macOS / Linux:**
```bash
docker stop algo-trading-system
docker rm algo-trading-system
docker rmi algo-trading-system:latest
```

## Notes

- First run downloads ~250MB DistilBERT model
- Data persists in ./data directory
- Container runs FastAPI backend + APScheduler in parallel
- SQLite databases are backed up to R2/MinIO on container shutdown
- See DEPLOYMENT.md for detailed documentation
