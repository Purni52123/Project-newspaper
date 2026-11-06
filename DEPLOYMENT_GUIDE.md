# 🚀 Cloud Deployment Guide (24/7 Free Hosting)

This guide explains how to host your **HIN-GNN News Recommendation System** online so that anyone can access the website and API 24/7, even when your laptop is turned off!

---

## 🌟 Overview of Free Hosting Services

| Component | Recommended Free Host | Free Tier Benefits | Live Example URL |
|---|---|---|---|
| **Frontend UI (`app.py`)** | **Streamlit Community Cloud** | 100% Free, Zero configuration, Auto-reloads on Git push | `https://your-name-news-recommender.streamlit.app` |
| **Backend REST API (`api.py`)** | **Render.com** or **Railway** | Free Web Services, HTTPS included, Auto-builds from GitHub | `https://hin-gnn-news-api.onrender.com` |

---

## Step 1: Create a GitHub Repository

1. Open your browser and go to [GitHub.com](https://github.com).
2. Click the **`+`** icon in the top-right corner and select **New repository**.
3. Fill in:
   - **Repository name:** `hin-gnn-news-recommendation` (or any name you like)
   - **Visibility:** **Public** (required for free Streamlit Community Cloud)
   - **Initialize with README:** **Leave unchecked** (we already have all files initialized!)
4. Click **Create repository**.
5. Copy the repository URL (e.g., `https://github.com/YOUR_USERNAME/hin-gnn-news-recommendation.git`).

---

## Step 2: Push Your Project to GitHub

In your VS Code terminal (or PowerShell inside `PROJECT 5th sem`), run:

```powershell
# 1. Link your local project to your new GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/hin-gnn-news-recommendation.git

# 2. Upload your code to GitHub
git push -u origin main
```
*(If GitHub prompts you to sign in, click "Sign in with your browser" to authorize).*

---

## Step 3: Deploy the Frontend on Streamlit Community Cloud (24/7 Free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and click **Sign in with GitHub**.
2. Click **Create app** (or **New app**).
3. Select your repository: `YOUR_USERNAME/hin-gnn-news-recommendation`.
4. Enter:
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**
6. Within 1-2 minutes, Streamlit installs the requirements and gives you a permanent link (e.g. `https://your-app-name.streamlit.app`).

> 🎉 **Done!** Your website is now live on the internet! You can share this link with your college professors, classmates, or on your resume. It will stay online even when your laptop is turned off.

---

## Step 4 (Optional): Deploy the Backend API on Render.com (24/7 Free)

If you also want your FastAPI backend accessible over the internet 24/7:

1. Go to [Render.com](https://render.com) and click **Get Started for Free** (Sign in with GitHub).
2. Click **New +** $\to$ **Web Service**.
3. Connect your repository: `hin-gnn-news-recommendation`.
4. Render will auto-detect your project using the `render.yaml` and `Procfile` we created:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy Web Service**.
6. Render will provide a free HTTPS URL like `https://hin-gnn-news-api.onrender.com` with Swagger docs at `/docs` and admin panel at `/dashboard`.
