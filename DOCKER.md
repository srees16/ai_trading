# Docker Deployment Quick Reference

All Docker and cloud deployment files are located in the `deployment/` directory.

## 🚀 Quick Commands

### Local Deployment
```powershell
# Windows
.\deployment\deploy.ps1

# Linux/Mac
./deployment/deploy.sh

# Docker Compose
cd deployment
docker-compose up -d
```

### Cloud Deployment
```powershell
# Azure
.\deployment\deploy-azure.ps1

# Google Cloud Platform
.\deployment\deploy-gcp.ps1
```

## 📁 Deployment Files

- `deployment/Dockerfile` - Container definition
- `deployment/.dockerignore` - Files to exclude from image
- `deployment/docker-compose.yml` - Docker Compose configuration
- `deployment/deploy.ps1` - Local deployment (Windows)
- `deployment/deploy.sh` - Local deployment (Linux/Mac)
- `deployment/deploy-azure.ps1` - Azure Container Instances
- `deployment/deploy-gcp.ps1` - Google Cloud Run
- `deployment/DEPLOYMENT.md` - Comprehensive deployment guide
- `deployment/DOCKER_QUICKSTART.md` - Quick start instructions

## 📖 Documentation

For detailed instructions, see:
- `deployment/DOCKER_QUICKSTART.md` - Quick start guide
- `deployment/DEPLOYMENT.md` - Full deployment documentation

## 🌐 Access URLs

- **Local**: http://localhost:8501
- **Azure**: http://<dns-name>.<region>.azurecontainer.io:8501
- **GCP**: Provided after deployment completes
