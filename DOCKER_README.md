# Finnie AI - Single Container Deployment

This setup runs all Finnie AI services (MCP servers + Streamlit UI) in a single Docker container, making it easy to deploy anywhere.

## Files

- `Dockerfile` - Single container with all services
- `supervisord.conf` - Process manager to run all services
- `docker-compose.yml` - For local development
- `.dockerignore` - Keeps the image lean

## Local Development

```bash
# 1. Copy these files to your finnie_ai repo root

# 2. Make sure FAISS indexes are built (one-time)
python -m src.indexer.build_faiss_index -v
python -m src.indexer.build_faiss_index -v -a articles_bogleheads_detailed.csv -i bogleheads

# 3. Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# 4. Build and run
docker-compose up --build

# 5. Open http://localhost:8501
```

## Deployment Options Comparison

| Platform | Free Tier | RAM | Best For | Difficulty |
|----------|-----------|-----|----------|------------|
| **Render.com** | ✅ 750 hrs/mo | 512MB-2GB | Hobby projects | ⭐ Easy |
| **Railway.app** | ✅ $5 credit/mo | 512MB-8GB | Side projects | ⭐ Easy |
| **Google Cloud Run** | ✅ 2M requests/mo | Up to 4GB | Production | ⭐⭐ Medium |
| **AWS App Runner** | ⚠️ Limited | 1-4GB | AWS ecosystem | ⭐⭐ Medium |
| **Fly.io** | ✅ 3 shared VMs | 256MB-2GB | Global edge | ⭐⭐ Medium |
| **DigitalOcean Apps** | ❌ $5/mo min | 512MB-8GB | Simple PaaS | ⭐ Easy |

---

## 🏆 Recommended: Render.com (Easiest + Free)

**Why Render:**
- Generous free tier (750 hours/month = always on)
- Automatic HTTPS
- Easy GitHub integration
- No credit card required for free tier

### Deploy to Render

1. **Push your code to GitHub** (with the Dockerfile in repo root)

2. **Create Render account** at https://render.com

3. **New Web Service:**
   - Connect your GitHub repo
   - Select "Docker" as environment
   - Set environment variable: `OPENAI_API_KEY`

4. **Configure:**
   ```
   Name: finnie-ai
   Region: Oregon (or closest to you)
   Instance Type: Free (or Starter $7/mo for more RAM)
   ```

5. **Deploy!**

⚠️ **Free tier limitation:** 512MB RAM may be tight. If you hit memory issues, upgrade to Starter ($7/mo) for 2GB RAM.

---

## 🥈 Alternative: Railway.app (Best DX)

**Why Railway:**
- $5 free credit/month
- Excellent developer experience
- Easy environment variables
- Auto-deploys from GitHub

### Deploy to Railway

1. **Sign up** at https://railway.app

2. **New Project → Deploy from GitHub repo**

3. **Add environment variable:**
   ```
   OPENAI_API_KEY=your-key-here
   ```

4. **Railway auto-detects Dockerfile and deploys**

5. **Get your URL** from the deployment dashboard

---

## 🥉 Alternative: Google Cloud Run (Most Scalable)

**Why Cloud Run:**
- Generous free tier (2M requests/month)
- Scales to zero (saves money)
- Up to 4GB RAM
- Production-ready

### Deploy to Cloud Run

```bash
# 1. Install gcloud CLI and authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/finnie-ai

# 3. Deploy to Cloud Run
gcloud run deploy finnie-ai \
  --image gcr.io/YOUR_PROJECT_ID/finnie-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars "OPENAI_API_KEY=your-key-here"
```

⚠️ **Note:** Cloud Run has a request timeout (default 5 min). Long-running LLM calls might need timeout adjustment.

---

## AWS Options

### AWS App Runner

```bash
# Requires AWS CLI configured
aws apprunner create-service \
  --service-name finnie-ai \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "YOUR_ECR_IMAGE_URI",
      "ImageRepositoryType": "ECR"
    }
  }'
```

### AWS Lightsail Containers (Cheapest AWS option: $7/mo)

1. Go to Lightsail Console → Containers
2. Create container service (Nano: $7/mo, 512MB RAM)
3. Push your Docker image
4. Deploy

---

## Memory Optimization Tips

If running on limited RAM (512MB-1GB):

1. **Reduce workers:** Add to supervisord.conf programs that are less critical

2. **Lazy load FAISS:** Modify Q&A agent to load index only when needed

3. **Use smaller models:** If using local embeddings, use smaller ones

4. **Monitor memory:**
   ```bash
   docker stats finnie-ai
   ```

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs finnie-ai

# Check supervisor logs inside container
docker exec -it finnie-ai cat /var/log/supervisor/supervisord.log
```

### Out of memory
- Upgrade to a paid tier with more RAM
- Or reduce number of services running

### MCP connection errors
- All services use localhost inside the container
- Make sure your code doesn't have hardcoded Docker hostnames
- Revert any `mcp_config.py` changes (use localhost defaults)

### Streamlit not accessible
- Ensure port 8501 is exposed
- Check if Streamlit is bound to 0.0.0.0 (not 127.0.0.1)

---

## Quick Reference

```bash
# Build locally
docker build -t finnie-ai .

# Run locally
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key finnie-ai

# Run with mounted data
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/src/data:/app/src/data \
  finnie-ai

# View logs
docker logs -f finnie-ai

# Shell into container
docker exec -it finnie-ai bash

# Check service status
docker exec -it finnie-ai supervisorctl status
```