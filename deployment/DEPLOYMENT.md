# Cloud Deployment Guide for AI Trading Alert System

This guide covers deploying the AI Trading Alert System to cloud platforms using Docker containers.

## Prerequisites

- Docker installed locally
- Cloud provider account (Azure or GCP)
- Azure CLI or Google Cloud SDK installed

## Local Testing

### Build and Run Locally

**Windows:**
```powershell
.\deployment\deploy.ps1
```

**Linux/Mac:**
```bash
chmod +x deployment/deploy.sh
./deployment/deploy.sh
```

**Using Docker Compose:**
```bash
cd deployment
docker-compose up -d
```

Access at: http://localhost:8501

### Stop Container
```bash
docker stop algo-trading-system
```

## Azure Deployment

### 1. Prerequisites
- Install Azure CLI: https://docs.microsoft.com/cli/azure/install-azure-cli
- Login: `az login`

### 2. Deploy to Azure Container Instances

```powershell
.\deployment\deploy-azure.ps1
```

**Manual Steps:**

1. **Update variables** in `deployment/deploy-azure.ps1`:
   - `$RESOURCE_GROUP`: Your resource group name
   - `$ACR_NAME`: Your Azure Container Registry name (globally unique)
   - `$DNS_NAME`: Your DNS label (globally unique)

2. **Run deployment script**

3. **Access your app** at: `http://<dns-name>.<region>.azurecontainer.io:8501`

### Cost Estimate (Azure)
- Container Instances: ~$30-50/month (2 vCPU, 4GB RAM)
- Container Registry: ~$5/month (Basic tier)

### Cleanup Azure Resources
```powershell
az group delete --name algo-trading-rg --yes
```

## Google Cloud Platform Deployment

### 1. Prerequisites
- Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
- Login: `gcloud auth login`
- Create project: `gcloud projects create your-project-id`

### 2. Deploy to Cloud Run

```powershell
.\deployment\deploy-gcp.ps1
```

**Manual Steps:**

1. **Update variables** in `deployment/deploy-gcp.ps1`:
   - `$PROJECT_ID`: Your GCP project ID

2. **Run deployment script**

3. **Access your app** at the provided Cloud Run URL

### Cost Estimate (GCP)
- Cloud Run: ~$20-40/month (based on usage)
- Container Registry: Storage costs (~$5/month)
- First 2 million requests free per month

### Cleanup GCP Resources
```powershell
gcloud run services delete algo-trading-system --region us-central1
gcloud container images delete gcr.io/your-project-id/algo-trading-system
```

## Environment Variables (Optional)

Create a `.env` file for sensitive configurations:

```env
# API Keys (if needed in future)
# ALPHA_VANTAGE_KEY=your_key
# NEWS_API_KEY=your_key

# Application Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
```

Update Dockerfile to use env file:
```dockerfile
COPY .env .
```

## Monitoring

### Azure
```powershell
# View logs
az container logs --resource-group algo-trading-rg --name algo-trading-app

# Monitor metrics
az monitor metrics list --resource /subscriptions/.../algo-trading-app
```

### GCP
```powershell
# View logs
gcloud run services logs read algo-trading-system --region us-central1

# Monitor metrics
gcloud run services describe algo-trading-system --region us-central1
```

## Docker Commands Reference

```bash
# Build image
docker build -f deployment/Dockerfile -t algo-trading-system:latest .

# Run container
docker run -d -p 8501:8501 --name algo-trading-system algo-trading-system:latest

# View logs
docker logs -f algo-trading-system

# Stop container
docker stop algo-trading-system

# Remove container
docker rm algo-trading-system

# List running containers
docker ps

# Execute command in container
docker exec -it algo-trading-system bash
```

## Troubleshooting

### Container won't start
```bash
docker logs algo-trading-system
```

### Port already in use
```bash
# Windows
netstat -ano | findstr :8501

# Linux/Mac
lsof -i :8501
```

### Out of memory
- Increase memory allocation in cloud deployment scripts
- Azure: Adjust `--memory` parameter
- GCP: Adjust `--memory` parameter

### Application crashes
- Check logs for Python errors
- Verify all dependencies in requirements.txt
- Ensure sufficient memory/CPU resources

## Security Best Practices

1. **Don't commit secrets** - Use environment variables
2. **Use private registries** - ACR or GCR
3. **Enable authentication** - Restrict public access if needed
4. **Regular updates** - Keep base image and dependencies updated
5. **Scan images** - Use Azure Security Center or GCP Artifact Analysis

## Scaling

### Azure
```powershell
# Scale horizontally (not directly supported in ACI)
# Use Azure Kubernetes Service (AKS) for better scaling
```

### GCP Cloud Run
```powershell
# Auto-scales by default
# Configure min/max instances in deploy-gcp.ps1
--min-instances 1
--max-instances 100
```

## Custom Domain (Optional)

### Azure
1. Use Azure Front Door or Application Gateway
2. Configure custom domain in Azure Portal

### GCP
1. Use Cloud Load Balancing
2. Map custom domain in Cloud Console

---

**Note:** First deployment may take 5-10 minutes as it downloads the DistilBERT model (~250MB). Subsequent restarts will be faster.
