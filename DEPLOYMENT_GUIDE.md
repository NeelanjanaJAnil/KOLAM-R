# 🚀 Deployment Guide: GitHub + Google Cloud Run

This guide explains how to push your **Kolam-R** codebase to GitHub and deploy it live on **Google Cloud Run** to get a public URL for your faculty evaluation.

---

## Part 1: Push Code to GitHub

Open PowerShell in this project directory (`Kolam/`) and run:

```powershell
# 1. Initialize git if not already initialized
git init

# 2. Add files
git add .

# 3. Commit
git commit -m "feat: complete Kolam-R Streamlit prototype with Docker deployment setup"

# 4. Set main branch and link to your GitHub repository
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git

# 5. Push to GitHub
git push -u origin main
```

---

## Part 2: Deploy to Google Cloud Run (2 Options)

### Option A: 1-Click Web Console (Easiest, No CLI needed)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select your Google Cloud Project.
3. In the top search bar, type **Cloud Run** and click **Cloud Run**.
4. Click **Create Service**.
5. Select **Continuously deploy from a repository** (Connect with Cloud Build).
6. Click **Set up with Cloud Build**:
   - Provider: **GitHub**
   - Repository: Select your GitHub repository (`YOUR_REPO_NAME`).
   - Branch: `^main$`
   - Build Type: **Dockerfile** (Path: `/Dockerfile`).
7. In the Cloud Run settings:
   - Service name: `kolam-r-prototype`
   - Region: Select closest (e.g. `asia-south1` or `us-central1`).
   - Authentication: Choose **Allow unauthenticated invocations** (so your professor can view it without logging in).
   - Memory: **1 GiB** (or 2 GiB).
   - Port: `8080` (configured by default).
8. Click **Create**.
9. Once the build finishes (takes ~2 minutes), Google Cloud will provide a public HTTPS URL (e.g., `https://kolam-r-prototype-xyz-uc.a.run.app`).

---

### Option B: Deploy via Google Cloud SDK (gcloud CLI)

If you have `gcloud` CLI installed:

```powershell
# 1. Login to Google Cloud
gcloud auth login

# 2. Set your Project ID
gcloud config set project YOUR_PROJECT_ID

# 3. Build & Deploy directly to Cloud Run
gcloud run deploy kolam-r-prototype `
    --source . `
    --region asia-south1 `
    --allow-unauthenticated `
    --port 8080 `
    --memory 1Gi
```

When finished, Google Cloud will print the live URL:
```
Service URL: https://kolam-r-prototype-xxxxx-el.a.run.app
```

---

## Part 3: Free Alternative (Streamlit Community Cloud via GitHub)

If you want an instant 30-second live public link while Google Cloud is setting up:
1. Push to GitHub (Part 1 above).
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New App** $\to$ Select your GitHub repo $\to$ Main file path: `app/prototype_app.py`.
4. Click **Deploy**. Your app is live instantly with an HTTPS link (e.g. `https://kolam-r.streamlit.app`).
