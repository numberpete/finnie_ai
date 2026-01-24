# Finnie AI - Single Container Deployment

This setup runs all Finnie AI services (MCP servers + Streamlit UI) in a single Docker container, making it easy to deploy anywhere.  This is simply meant to be done for a quick deployment for remote demo purposes, it is NOT meant to be a production-grade deployment.

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

## Deloyment to Hosting service
This repository is integrated with a Railway account such that any committed pull request to the "deploy" branch will result in the deployment of this app.  It can be viewed online at https://finnieai-production.up.railway.app/.

## Deployment Options Comparison

⚠️ **Memory requirement:** Finnie AI uses ~775MB at startup and needs 1-2GB under load.

| Platform | Free Tier | RAM | Will it work? | Cost if not |
|----------|-----------|-----|---------------|-------------|
| **Railway.app** ⭐ | ✅ $5 credit/mo | Flexible | ⚠️ Maybe | ~$5/mo usage-based |
| **Google Cloud Run** | ✅ 2M requests/mo | Up to 4GB | ✅ Yes | Pay per request |
| **Render.com** | ✅ 750 hrs/mo | 512MB | ❌ No | $7/mo (Starter) |
| **OCI Always Free** | ✅ Forever free | Up to 24GB | ⚠️ Maybe* | N/A |
| **Fly.io** | ✅ 3 shared VMs | 256MB | ❌ No | $5-10/mo |
| **DigitalOcean Apps** | ❌ $5/mo min | 512MB-8GB | ✅ Yes | $5-12/mo |

*OCI has caveats — see below.

---

## 🏆 Recommended: Railway.app (Easiest)

**Why Railway:**
- Usage-based pricing ($5 free credit/month)
- Can burst above limits temporarily  
- Dead simple GitHub deploy
- If you exceed free tier, you only pay for what you use
- Best developer experience

### Deploy to Railway

1. **Sign up** at https://railway.app

2. **New Project → Deploy from GitHub repo**

3. **Add environment variables:**
   ```
   OPENAI_API_KEY=your-key-here
   CHART_URL=https://your-app.up.railway.app/chart/
   CHART_PATH=generated_charts
   ```
   (Update CHART_URL after first deploy with your actual Railway URL)

4. **Railway auto-detects Dockerfile and deploys**

5. **Get your URL** from the deployment dashboard, then update CHART_URL

💡 **Tip:** Railway's $5 free credit should cover light usage. If you go over, you'll pay a few dollars based on actual usage.

---

## 🥉 Alternative: Google Cloud Run (Scales to Zero)

**Why Cloud Run:**
- Generous free tier (2M requests/month)
- **Scales to zero** when not in use (saves money)
- Can allocate 2-4GB RAM — no memory issues
- Only pay when it's actually running

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

# 4. Get your URL from the output, then update CHART_URL:
gcloud run services update finnie-ai \
  --set-env-vars "CHART_URL=https://finnie-ai-XXXXX-uc.a.run.app/chart/"
```

⚠️ **Note:** Cloud Run has a request timeout (default 5 min). Long-running LLM calls might need timeout adjustment.

---

## 4️⃣ Alternative: Oracle Cloud (OCI) Always Free

**Why OCI:**
- **Forever free** (not a 12-month trial)
- Up to **24GB RAM** and **4 ARM cores** — overkill for Finnie AI
- 200GB storage included

⚠️ **Important caveats:**
- **Idle instance reclamation:** Oracle may delete your VM if CPU usage stays below 20% for 7 days. You'd need to set up a cron job to keep it active.
- **Capacity lottery:** ARM instances are in high demand — you may get "Out of Capacity" errors and be unable to provision a VM at all.
- **No support:** Always Free users cannot open support tickets.
- **More complex setup:** You manage the VM yourself (not a PaaS).
- **Credit card required:** For verification only, but still required.

### Deploy to OCI Always Free

#### 1. Create OCI Account
1. Go to https://www.oracle.com/cloud/free/
2. Sign up for Always Free tier (credit card required for verification)
3. Wait for account to be provisioned

#### 2. Create a Free VM
1. Go to **Compute → Instances → Create Instance**
2. Configure:
   ```
   Name: finnie-ai
   Image: Oracle Linux 8 (or Ubuntu 22.04)
   Shape: VM.Standard.A1.Flex (Always Free eligible)
   OCPUs: 1 (stay within free tier)
   Memory: 2GB (plenty for Finnie AI)
   ```
3. Download your SSH key or add your public key
4. Create the instance

💡 **Tip:** If you get "Out of Capacity" errors, try a different availability domain or retry later (sometimes for days/weeks).

#### 3. Configure Networking
1. Go to **Networking → Virtual Cloud Networks**
2. Click on your VCN → Security Lists → Default Security List
3. Add **Ingress Rules**:
   ```
   Port 8501 (Streamlit): 0.0.0.0/0, TCP, 8501
   Port 8010 (Images):    0.0.0.0/0, TCP, 8010
   ```

#### 4. Install Docker on the VM
```bash
# SSH into your instance
ssh -i your-key.pem opc@<your-vm-public-ip>

# Install Docker (Oracle Linux 8)
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Log out and back in for group changes
exit
ssh -i your-key.pem opc@<your-vm-public-ip>
```

#### 5. Deploy Finnie AI
```bash
# Clone your repo
git clone https://github.com/yourusername/finnie_ai.git
cd finnie_ai

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your-key-here
CHART_URL=http://<your-vm-public-ip>:8010/chart/
EOF

# Build and run
docker build -t finnie-ai .
docker run -d --name finnie-ai \
  -p 8501:8501 -p 8010:8010 \
  --env-file .env \
  -v $(pwd)/src/data:/app/src/data \
  --restart unless-stopped \
  finnie-ai
```

#### 6. Prevent Idle Reclamation
Add a cron job to keep CPU usage above Oracle's 20% threshold:
```bash
# Add to crontab
crontab -e

# Add this line (runs a short CPU spike every 6 hours)
0 */6 * * * timeout 60 yes > /dev/null &
```

#### 7. Access Your App
Open `http://<your-vm-public-ip>:8501` in your browser.

---

## 5️⃣ Alternative: Render.com (Paid Tier)

Render's free tier (512MB) won't work, but their **Starter tier ($7/mo)** gives you 2GB RAM.

### Deploy to Render

1. **Push your code to GitHub** (with the Dockerfile in repo root)

2. **Create Render account** at https://render.com

3. **New Web Service:**
   - Connect your GitHub repo
   - Select "Docker" as environment
   - Set environment variables:
     - `OPENAI_API_KEY` = your key
     - `CHART_URL` = `https://your-app-name.onrender.com/chart/` (use your actual Render URL)

4. **Configure:**
   ```
   Name: finnie-ai
   Region: Oregon (or closest to you)
   Instance Type: Starter ($7/mo) — Free tier won't work!
   ```

5. **Deploy!**

⚠️ **Important:** After your first deploy, copy your Render URL and update `CHART_URL` to match (e.g., `https://finnie-ai.onrender.com/chart/`). Charts won't display until this is set correctly.

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

### Charts not displaying
- **Check CHART_URL:** Must be set to your public URL (e.g., `https://your-app.onrender.com/chart/`)
- Using `localhost` won't work in production — the browser needs to reach the image server
- Verify image server is running: `docker exec -it finnie-ai supervisorctl status`

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
docker run -p 8501:8501 -p 8010:8010 \
  -e OPENAI_API_KEY=your-key \
  -e CHART_URL=http://localhost:8010/chart/ \
  finnie-ai

# Run with mounted data
docker run -p 8501:8501 -p 8010:8010 \
  -e OPENAI_API_KEY=your-key \
  -e CHART_URL=http://localhost:8010/chart/ \
  -v $(pwd)/src/data:/app/src/data \
  finnie-ai

# View logs
docker logs -f finnie-ai

# Shell into container
docker exec -it finnie-ai bash

# Check service status
docker exec -it finnie-ai supervisorctl status
```
