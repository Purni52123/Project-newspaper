"""
Complete Training and Evaluation Pipeline
Trains HIN-GNN, evaluates baselines, runs diversity trade-off analysis,
and saves model artifacts for the Streamlit dashboard and REST API.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data_loader import (
    load_news,
    load_behaviors,
    parse_impressions,
    extract_training_triplets
)
from src.graph_builder import construct_hin, build_sparse_adjacency
from src.models import HINGNN, train_gnn_model
from src.baselines import (
    PopularityRecommender,
    TFIDFRecommender,
    train_lightgcn_baseline
)
from src.reranker import diversify_recommendations
from src.metrics import compute_ranking_accuracy, compute_novelty_and_diversity

# Configuration & Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"
TOP_K = 10
EMBED_DIM = 64
EPOCHS = 5

# Safe limits for memory; change to None to use full files if running full dataset
MAX_TRAIN_ROWS = 50000
MAX_DEV_ROWS = 10000

def ensure_dataset():
    """Ensures data files exist. Generates sample data if not present."""
    train_news = os.path.join(DATA_DIR, "train", "news.tsv")
    train_behaviors = os.path.join(DATA_DIR, "train", "behaviors.tsv")
    if not os.path.exists(train_news) or not os.path.exists(train_behaviors):
        print("[!] MIND dataset files not detected in data/train.")
        print("[*] Automatically generating high-quality starter dataset using sample_data.py...")
        from data.sample_data import generate_sample_dataset
        generate_sample_dataset()

def evaluate_model_on_dev(name, predict_fn, dev_df, news_df, popularity_counts, news_map, k=10):
    """
    Evaluates a candidate ranking function across dev impression sessions.
    """
    accuracy_records = []
    diversity_records = []

    for row in dev_df.itertuples(index=False):
        parsed = parse_impressions(row.impressions)
        candidates = [nid for nid, _ in parsed if nid in news_map]
        relevant = [nid for nid, lbl in parsed if lbl == 1 and nid in news_map]

        if not candidates or not relevant:
            continue

        # Get scores and rank
        scores = {nid: predict_fn(row.user_id, nid) for nid in candidates}
        ranked = sorted(candidates, key=lambda n: scores.get(n, -float("inf")), reverse=True)[:k]

        acc = compute_ranking_accuracy(ranked, relevant, k=k)
        if acc:
            div = compute_novelty_and_diversity(ranked, news_df, popularity_counts, k=k)
            accuracy_records.append(acc)
            diversity_records.append(div)

    if not accuracy_records:
        return {"Model": name, f"nDCG@{k}": 0.0, "MRR": 0.0, f"Recall@{k}": 0.0, "Novelty": 0.0, "CategoryDiversity": 0.0, "ILD": 0.0}

    avg_acc = pd.DataFrame(accuracy_records).mean().to_dict()
    avg_div = pd.DataFrame(diversity_records).mean().to_dict()
    
    return {
        "Model": name,
        **avg_acc,
        **avg_div
    }

def main():
    print("=" * 70)
    print("  HIN-GNN DIVERSIFIED NEWS RECOMMENDATION SYSTEM PIPELINE")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    ensure_dataset()

    # 1. Load Data
    print("\n[Step 1/6] Loading News and Behaviors...")
    train_news_df = load_news(os.path.join(DATA_DIR, "train", "news.tsv"))
    dev_news_df = load_news(os.path.join(DATA_DIR, "dev", "news.tsv"))
    news_df = pd.concat([train_news_df, dev_news_df]).drop_duplicates(subset=["news_id"]).reset_index(drop=True)

    train_behaviors = load_behaviors(os.path.join(DATA_DIR, "train", "behaviors.tsv"), limit=MAX_TRAIN_ROWS)
    dev_behaviors = load_behaviors(os.path.join(DATA_DIR, "dev", "behaviors.tsv"), limit=MAX_DEV_ROWS)

    news_ids = set(news_df["news_id"])
    print(f"  Total Unique News Articles: {len(news_df):,}")
    print(f"  Training Impression Rows:   {len(train_behaviors):,}")
    print(f"  Validation Impression Rows: {len(dev_behaviors):,}")

    # 2. Build Heterogeneous Graph (HIN)
    print("\n[Step 2/6] Constructing Heterogeneous Information Network (HIN)...")
    hin_graph = construct_hin(news_df, train_behaviors)
    print(f"  HIN Graph Nodes: {hin_graph.number_of_nodes():,}")
    print(f"  HIN Graph Edges: {hin_graph.number_of_edges():,}")

    # Mappings
    user_ids = sorted(list(set(train_behaviors["user_id"])))
    user_map = {u: i for i, u in enumerate(user_ids)}
    news_map = {n: i for i, n in enumerate(news_df["news_id"])}

    triplets, popularity_counts = extract_training_triplets(train_behaviors, news_ids)
    print(f"  Training Triplets (User, Pos, Neg): {len(triplets):,}")

    # Adjacency
    adj = build_sparse_adjacency(user_map, news_map, triplets, news_df, hin_graph, DEVICE)

    # 3. Train Proposed HIN-GNN Model
    print("\n[Step 3/6] Training Proposed HIN-GNN Model...")
    total_nodes = len(user_map) + len(news_map)
    hin_gnn_model = HINGNN(total_nodes, len(user_map), len(news_map), embedding_dim=EMBED_DIM, num_layers=2).to(DEVICE)
    hin_embeddings = train_gnn_model(hin_gnn_model, adj, triplets, user_map, news_map, epochs=EPOCHS, device=DEVICE)

    # 4. Train / Setup Baselines
    print("\n[Step 4/6] Setting Up Baseline Models...")
    pop_recommender = PopularityRecommender(popularity_counts)
    
    tfidf_recommender = TFIDFRecommender(news_df)
    tfidf_recommender.build_user_profiles(train_behaviors)

    lightgcn_embeddings = train_lightgcn_baseline(user_map, news_map, triplets, epochs=EPOCHS, embedding_dim=EMBED_DIM, device=DEVICE)

    # Define predictor functions
    n_users = len(user_map)
    def hin_predict(u, n):
        if u in user_map and n in news_map:
            u_vec = hin_embeddings[user_map[u]]
            n_vec = hin_embeddings[n_users + news_map[n]]
            return float(np.dot(u_vec, n_vec))
        return 0.0

    def lightgcn_predict(u, n):
        if u in user_map and n in news_map:
            u_vec = lightgcn_embeddings[user_map[u]]
            n_vec = lightgcn_embeddings[n_users + news_map[n]]
            return float(np.dot(u_vec, n_vec))
        return 0.0

    # 5. Model Evaluation & Comparison
    print("\n[Step 5/6] Evaluating Models on Held-out Validation Split...")
    eval_results = []
    eval_results.append(evaluate_model_on_dev("Popularity", pop_recommender.predict_score, dev_behaviors, news_df, popularity_counts, news_map, k=TOP_K))
    eval_results.append(evaluate_model_on_dev("TF-IDF Content", tfidf_recommender.predict_score, dev_behaviors, news_df, popularity_counts, news_map, k=TOP_K))
    eval_results.append(evaluate_model_on_dev("LightGCN (CF Graph)", lightgcn_predict, dev_behaviors, news_df, popularity_counts, news_map, k=TOP_K))
    eval_results.append(evaluate_model_on_dev("HIN-GNN (Proposed)", hin_predict, dev_behaviors, news_df, popularity_counts, news_map, k=TOP_K))

    results_df = pd.DataFrame(eval_results)
    results_path = os.path.join(ARTIFACTS_DIR, "results.csv")
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "=" * 70)
    print("                     EVALUATION RESULTS TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    # 6. Accuracy vs Diversity Trade-off Analysis (Alpha Tuning)
    print("\n[Step 6/6] Analyzing Accuracy vs Diversity Trade-off (Alpha sweep)...")
    alpha_levels = [1.00, 0.75, 0.50, 0.25]
    tradeoff_records = []

    for alpha in alpha_levels:
        acc_list = []
        div_list = []
        for row in dev_behaviors.itertuples(index=False):
            parsed = parse_impressions(row.impressions)
            candidates = [nid for nid, _ in parsed if nid in news_map]
            relevant = [nid for nid, lbl in parsed if lbl == 1 and nid in news_map]
            if not candidates or not relevant:
                continue

            scores = {nid: hin_predict(row.user_id, nid) for nid in candidates}
            # Re-rank using MMR diversity
            reranked = diversify_recommendations(candidates, scores, news_df, alpha=alpha, top_k=TOP_K)

            acc = compute_ranking_accuracy(reranked, relevant, k=TOP_K)
            if acc:
                div = compute_novelty_and_diversity(reranked, news_df, popularity_counts, k=TOP_K)
                acc_list.append(acc)
                div_list.append(div)

        if acc_list:
            avg_acc = pd.DataFrame(acc_list).mean().to_dict()
            avg_div = pd.DataFrame(div_list).mean().to_dict()
            tradeoff_records.append({
                "Alpha": alpha,
                "Description": "Pure Relevance" if alpha == 1.0 else ("Balanced" if alpha == 0.5 else ("Relevance Focused" if alpha == 0.75 else "Diversity Focused")),
                **avg_acc,
                **avg_div
            })

    tradeoff_df = pd.DataFrame(tradeoff_records)
    tradeoff_path = os.path.join(ARTIFACTS_DIR, "diversity_tradeoff.csv")
    tradeoff_df.to_csv(tradeoff_path, index=False)

    print("\n" + "=" * 70)
    print("              ACCURACY VS DIVERSITY TRADE-OFF ANALYSIS")
    print("=" * 70)
    print(tradeoff_df.to_string(index=False))
    print("=" * 70)

    # Save Model Artifacts for Streamlit & API
    print("\n[*] Saving model artifacts to artifacts/ ...")
    np.save(os.path.join(ARTIFACTS_DIR, "embeddings.npy"), hin_embeddings)
    news_df.to_csv(os.path.join(ARTIFACTS_DIR, "news.csv"), index=False)
    
    with open(os.path.join(ARTIFACTS_DIR, "user_map.json"), "w") as f:
        json.dump(user_map, f)
    with open(os.path.join(ARTIFACTS_DIR, "news_map.json"), "w") as f:
        json.dump(news_map, f)
    with open(os.path.join(ARTIFACTS_DIR, "popularity.json"), "w") as f:
        json.dump(popularity_counts, f)

    # Plot Trade-off Chart
    try:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()

        alphas = tradeoff_df["Alpha"]
        ndcgs = tradeoff_df[f"nDCG@{TOP_K}"]
        ilds = tradeoff_df["ILD"]

        ax1.plot(alphas, ndcgs, "b-o", linewidth=2, label="Accuracy (nDCG@10)")
        ax2.plot(alphas, ilds, "g-s", linewidth=2, label="Diversity (ILD)")

        ax1.set_xlabel("Relevance-Diversity Trade-off Parameter (Alpha)")
        ax1.set_ylabel("nDCG@10 (Ranking Accuracy)", color="blue")
        ax2.set_ylabel("Intra-List Diversity (ILD)", color="green")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax2.tick_params(axis="y", labelcolor="green")
        plt.title("HIN-GNN: Recommendation Accuracy vs Intra-List Diversity Trade-off")
        plt.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        chart_path = os.path.join(ARTIFACTS_DIR, "tradeoff_plot.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"  Saved trade-off plot: {chart_path}")
    except Exception as e:
        print(f"  Chart plotting skipped: {e}")

    print("\n[SUCCESS] Pipeline execution finished! Ready for Web App and API.")

if __name__ == "__main__":
    main()

