"""
REST API for HIN-GNN Diversified News Recommendation System
Built with FastAPI. Provides endpoints for programmatic news recommendation.
"""

import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

from src.reranker import diversify_recommendations

app = FastAPI(
    title="HIN-GNN News Recommendation API",
    description="REST API to fetch personalized and diversified news recommendations using Graph Neural Networks.",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
news_df = None
embeddings = None
user_map = {}
news_map = {}
popularity = {}
user_histories = {}

def load_data():
    global news_df, embeddings, user_map, news_map, popularity, user_histories
    try:
        news_path = os.path.join(ARTIFACTS_DIR, "news.csv")
        emb_path = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
        user_map_path = os.path.join(ARTIFACTS_DIR, "user_map.json")
        news_map_path = os.path.join(ARTIFACTS_DIR, "news_map.json")
        pop_path = os.path.join(ARTIFACTS_DIR, "popularity.json")

        if not all(os.path.exists(p) for p in [news_path, emb_path, user_map_path, news_map_path, pop_path]):
            return False

        news_df = pd.read_csv(news_path).fillna("")
        embeddings = np.load(emb_path)
        with open(user_map_path, "r", encoding="utf-8") as f:
            user_map = json.load(f)
        with open(news_map_path, "r", encoding="utf-8") as f:
            news_map = json.load(f)
        with open(pop_path, "r", encoding="utf-8") as f:
            popularity = json.load(f)

        # Load history
        train_beh_path = os.path.join(BASE_DIR, "data", "train", "behaviors.tsv")
        if os.path.exists(train_beh_path):
            with open(train_beh_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        uid = parts[1]
                        hist = parts[3].split()
                        user_histories[uid] = hist
        return True
    except Exception as e:
        print(f"[!] Error in load_data: {e}")
        return False

# Initialize on import for serverless environments
load_data()

@app.on_event("startup")
def startup_event():
    load_data()

@app.get("/")
def root():
    if embeddings is None:
        load_data()
    return {
        "system": "HIN-GNN Diversified News Recommendation System API",
        "status": "online" if embeddings is not None else "model_not_trained",
        "docs_url": "/docs",
        "dashboard_ui": "/dashboard",
        "available_users": len(user_map) if user_map else 0,
        "available_articles": len(news_df) if news_df is not None else 0
    }

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_ui():
    """Returns a modern, CSS-styled web dashboard directly from the FastAPI backend."""
    css_content = ""
    css_path = os.path.join(BASE_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HIN-GNN Backend API Dashboard</title>
        <style>
            {css_content}
            body {{
                background-color: #f8fafc;
                margin: 0;
                padding: 2.5rem 10%;
                color: #1e293b;
            }}
            .api-card {{
                background: white;
                border-radius: 14px;
                padding: 1.8rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border: 1px solid #e2e8f0;
                margin-bottom: 1.5rem;
            }}
            .endpoint-pill {{
                display: inline-block;
                padding: 0.35rem 0.8rem;
                background: #e0e7ff;
                color: #3730a3;
                font-family: 'JetBrains Mono', monospace;
                border-radius: 6px;
                font-weight: 600;
                font-size: 0.85rem;
            }}
            .btn-link {{
                display: inline-block;
                padding: 0.6rem 1.2rem;
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 600;
                transition: opacity 0.2s;
            }}
            .btn-link:hover {{ opacity: 0.9; }}
        </style>
    </head>
    <body>
        <div class="hero-banner">
            <h1 class="hero-title">🚀 HIN-GNN Backend API Dashboard</h1>
            <p class="hero-subtitle">Production REST Microservice for Personalized and Diversified News Recommendation.</p>
        </div>

        <div class="api-card">
            <h2>⚡ API Status: <span style="color: #10b981;">Online & Ready</span></h2>
            <p>Active Model: <b>Heterogeneous Information Network Graph Neural Network (HIN-GNN)</b></p>
            <p>Total Registered Users: <b>{len(user_map)}</b> | Total News Articles: <b>{len(news_df) if news_df is not None else 0}</b></p>
            <div style="margin-top: 1.2rem; display: flex; gap: 1rem;">
                <a class="btn-link" href="/docs" target="_blank">📖 Open Interactive Swagger UI</a>
                <a class="btn-link" style="background: #0f172a;" href="http://localhost:8501" target="_blank">📰 Open Streamlit Web App</a>
            </div>
        </div>

        <div class="api-card">
            <h3>📡 Available Endpoints</h3>
            <ul style="line-height: 2.2;">
                <li><span class="endpoint-pill">GET /recommend</span> &mdash; Query with <code>?user_id=U0001&alpha=0.75&top_k=5</code> <a href="/recommend?user_id=U0001&top_k=5&alpha=0.75" target="_blank" style="margin-left: 0.5rem; color: #4f46e5;">[Test Live]</a></li>
                <li><span class="endpoint-pill">GET /users</span> &mdash; Fetch list of all active user IDs <a href="/users?limit=10" target="_blank" style="margin-left: 0.5rem; color: #4f46e5;">[Test Live]</a></li>
                <li><span class="endpoint-pill">GET /metrics</span> &mdash; Benchmark results against Popularity, TF-IDF, and LightGCN <a href="/metrics" target="_blank" style="margin-left: 0.5rem; color: #4f46e5;">[Test Live]</a></li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html

@app.get("/users")
def get_users(limit: int = 20):
    """Returns a list of valid user IDs available in the dataset."""
    if not user_map:
        raise HTTPException(status_code=503, detail="Models not loaded. Run pipeline.py first.")
    users = sorted(list(user_map.keys()))
    return {"total_users": len(users), "users": users[:limit]}

@app.get("/recommend")
def get_recommendations(
    user_id: str = Query(..., description="ID of the user, e.g., U0001"),
    top_k: int = Query(5, ge=1, le=20, description="Number of articles to recommend"),
    alpha: float = Query(0.75, ge=0.0, le=1.0, description="Trade-off parameter: 1.0=pure relevance, 0.0=max diversity"),
    filter_history: bool = Query(True, description="Filter out news already read by the user")
):
    """
    Returns personalized and diversified recommendations for a given user.
    """
    if embeddings is None or news_df is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded. Run 'python pipeline.py' first.")

    if user_id not in user_map:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found. Available sample users: {list(user_map.keys())[:5]}"
        )

    # User embedding
    n_users = len(user_map)
    u_idx = user_map[user_id]
    user_vec = embeddings[u_idx]

    # Calculate raw scores
    candidate_ids = list(news_df["news_id"])
    raw_scores = {}
    for nid in candidate_ids:
        if nid in news_map:
            n_idx = n_users + news_map[nid]
            news_vec = embeddings[n_idx]
            raw_scores[nid] = float(np.dot(user_vec, news_vec))

    # History filtering
    hist = user_histories.get(user_id, [])
    if filter_history and hist:
        pool = [n for n in candidate_ids if n not in hist]
    else:
        pool = candidate_ids

    # Diversify
    rec_ids = diversify_recommendations(pool, raw_scores, news_df, alpha=alpha, top_k=top_k)

    # Format result list
    results = []
    news_lookup = news_df.set_index("news_id").to_dict(orient="index")
    for nid in rec_ids:
        item = news_lookup.get(nid, {})
        results.append({
            "news_id": nid,
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "subcategory": item.get("subcategory", ""),
            "abstract": item.get("abstract", ""),
            "gnn_relevance_score": round(raw_scores.get(nid, 0.0), 4)
        })

    return {
        "user_id": user_id,
        "alpha": alpha,
        "mode": "Pure Relevance" if alpha == 1.0 else ("Balanced" if alpha == 0.5 else ("Relevance Focused" if alpha > 0.5 else "Diversity Focused")),
        "top_k": top_k,
        "click_history_count": len(hist),
        "recommendations": results
    }

@app.get("/metrics")
def get_metrics():
    """Returns baseline comparison and diversity trade-off metrics."""
    results_path = os.path.join(ARTIFACTS_DIR, "results.csv")
    tradeoff_path = os.path.join(ARTIFACTS_DIR, "diversity_tradeoff.csv")

    res = {}
    if os.path.exists(results_path):
        res["model_comparison"] = pd.read_csv(results_path).to_dict(orient="records")
    if os.path.exists(tradeoff_path):
        res["diversity_tradeoff"] = pd.read_csv(tradeoff_path).to_dict(orient="records")

    return res

if __name__ == "__main__":
    import uvicorn
    print("[*] Starting REST API on http://localhost:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

