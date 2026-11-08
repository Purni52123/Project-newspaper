"""
Complete Training & Evaluation Pipeline — HIN-GNN News Recommendation
========================================================================
Steps:
  1. Generate / load MIND-small dataset
  2. Construct Heterogeneous Information Network (HIN)
  3. Train HIN-GNN (unified adjacency)
  4. Train HAN     (per-relation attention)
  5. Train / setup 4 baseline models
  6. Evaluate all models on dev split  →  results.csv
  7. Accuracy vs Diversity trade-off  →  diversity_tradeoff.csv + plot
  8. Save artifacts for Streamlit app & REST API
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Configuration ──────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR      = "data"
ARTIFACTS_DIR = "artifacts"
TOP_K         = 10
EMBED_DIM     = 64
EPOCHS        = 8          # increase for better convergence; 5 is fast demo
MAX_TRAIN_ROWS = None      # None = use full generated dataset
MAX_DEV_ROWS   = None

# ── Imports ────────────────────────────────────────────────────────────────────
from src.data_loader import (
    load_news, load_behaviors,
    parse_impressions, extract_training_triplets,
)
from src.graph_builder import (
    construct_hin, build_sparse_adjacency, build_relation_adjacencies
)
from src.models import HINGNN, HAN, train_gnn_model, train_han_model
from src.baselines import (
    PopularityRecommender, TFIDFRecommender,
    MatrixFactorizationSVD, train_lightgcn_baseline,
)
from src.reranker import diversify_recommendations
from src.metrics import compute_ranking_accuracy, compute_novelty_and_diversity


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset Preparation
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dataset():
    """Generate MIND-small dataset if not already present."""
    train_news = os.path.join(DATA_DIR, "train", "news.tsv")
    dev_news   = os.path.join(DATA_DIR, "dev",   "news.tsv")

    if os.path.exists(train_news) and os.path.exists(dev_news):
        # Verify they have content
        if os.path.getsize(train_news) > 1000 and os.path.getsize(dev_news) > 1000:
            print("  [*] Existing dataset files found — skipping generation.")
            return

    print("  [*] Generating MIND-small formatted dataset...")
    from data.mind_small_generator import generate_mind_small_dataset
    generate_mind_small_dataset(DATA_DIR, n_train_users=400, n_dev_users=100)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation Helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name, predict_fn, dev_df, news_df, popularity_counts, news_map, k=10):
    """
    Evaluates predict_fn(user_id, news_id) → score across all dev impressions.
    Returns aggregated accuracy and diversity metrics dict.
    """
    acc_rows, div_rows = [], []

    for row in dev_df.itertuples(index=False):
        parsed     = parse_impressions(row.impressions)
        candidates = [nid for nid, _ in parsed if nid in news_map]
        relevant   = [nid for nid, lbl in parsed if lbl == 1 and nid in news_map]
        if not candidates or not relevant:
            continue

        scores = {nid: predict_fn(row.user_id, nid) for nid in candidates}
        ranked = sorted(candidates, key=lambda n: scores.get(n, -1e9), reverse=True)[:k]

        acc = compute_ranking_accuracy(ranked, relevant, k=k)
        if acc:
            div = compute_novelty_and_diversity(ranked, news_df, popularity_counts, k=k)
            acc_rows.append(acc)
            div_rows.append(div)

    if not acc_rows:
        return {
            "Model": name,
            f"nDCG@{k}": 0.0, "MRR": 0.0, f"Recall@{k}": 0.0,
            "Novelty": 0.0, "CategoryDiversity": 0.0, "ILD": 0.0,
        }

    avg_acc = pd.DataFrame(acc_rows).mean().to_dict()
    avg_div = pd.DataFrame(div_rows).mean().to_dict()
    result  = {"Model": name, **avg_acc, **avg_div}

    ndcg_key = f"nDCG@{k}"
    mrr_key  = "MRR"
    rec_key  = f"Recall@{k}"
    print(
        f"  {name:<28s} nDCG@{k}={avg_acc.get(ndcg_key,0):.4f}  "
        f"MRR={avg_acc.get(mrr_key,0):.4f}  "
        f"Recall@{k}={avg_acc.get(rec_key,0):.4f}  "
        f"ILD={avg_div.get('ILD',0):.4f}  "
        f"CatDiv={avg_div.get('CategoryDiversity',0):.4f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("   HIN-GNN + HAN  —  DIVERSIFIED NEWS RECOMMENDATION PIPELINE")
    print("=" * 72)
    print(f"   Device    : {DEVICE}")
    print(f"   Embed dim : {EMBED_DIM}   Layers: 3   Epochs: {EPOCHS}")
    print("=" * 72)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── Step 1: Data ────────────────────────────────────────────────────────
    print("\n[Step 1/7]  Loading MIND-small Dataset...")
    ensure_dataset()

    train_news_df = load_news(os.path.join(DATA_DIR, "train", "news.tsv"))
    dev_news_df   = load_news(os.path.join(DATA_DIR, "dev",   "news.tsv"))
    news_df       = (pd.concat([train_news_df, dev_news_df])
                       .drop_duplicates(subset=["news_id"])
                       .reset_index(drop=True))

    train_beh = load_behaviors(os.path.join(DATA_DIR, "train", "behaviors.tsv"), limit=MAX_TRAIN_ROWS)
    dev_beh   = load_behaviors(os.path.join(DATA_DIR, "dev",   "behaviors.tsv"), limit=MAX_DEV_ROWS)

    news_ids = set(news_df["news_id"])
    print(f"  Articles     : {len(news_df):,}")
    print(f"  Categories   : {news_df['category'].nunique()}")
    print(f"  Subcategories: {news_df['subcategory'].nunique()}")
    print(f"  Train rows   : {len(train_beh):,}")
    print(f"  Dev rows     : {len(dev_beh):,}")

    # ── Step 2: HIN ────────────────────────────────────────────────────────
    print("\n[Step 2/7]  Constructing Heterogeneous Information Network (HIN)...")
    hin_graph, edge_counts = construct_hin(news_df, train_beh)

    # Mappings
    user_ids = sorted(set(train_beh["user_id"]))
    user_map = {u: i for i, u in enumerate(user_ids)}
    news_map = {n: i for i, n in enumerate(news_df["news_id"])}
    n_users  = len(user_map)
    n_news   = len(news_map)
    total    = n_users + n_news

    triplets, popularity_counts = extract_training_triplets(train_beh, news_ids)
    print(f"  Users    : {n_users:,}")
    print(f"  News     : {n_news:,}")
    print(f"  Triplets : {len(triplets):,}")

    # Build adjacency matrices
    print("\n  Building adjacency matrices...")
    adj_full = build_sparse_adjacency(user_map, news_map, triplets, news_df, hin_graph, DEVICE)
    rel_adjs = build_relation_adjacencies(user_map, news_map, triplets, news_df, DEVICE)
    RELATION_NAMES = list(rel_adjs.keys())

    # ── Step 3: Train HIN-GNN ───────────────────────────────────────────────
    print("\n[Step 3/7]  Training HIN-GNN (unified adjacency, LightGCN-style)...")
    hin_model = HINGNN(total, n_users, n_news, embedding_dim=EMBED_DIM, num_layers=3).to(DEVICE)
    hin_emb   = train_gnn_model(hin_model, adj_full, triplets, user_map, news_map,
                                 epochs=EPOCHS, lr=0.005, device=DEVICE)

    # ── Step 4: Train HAN ───────────────────────────────────────────────────
    print("\n[Step 4/7]  Training HAN (heterogeneous attention, per-relation)...")
    han_model = HAN(total, n_users, n_news, embedding_dim=EMBED_DIM, num_layers=2,
                    relation_names=RELATION_NAMES, dropout=0.1).to(DEVICE)
    han_emb   = train_han_model(han_model, rel_adjs, triplets, user_map, news_map,
                                 epochs=EPOCHS, lr=0.005, device=DEVICE)

    # ── Step 5: Baselines ───────────────────────────────────────────────────
    print("\n[Step 5/7]  Setting Up Baseline Models...")

    pop_rec = PopularityRecommender(popularity_counts)

    tfidf_rec = TFIDFRecommender(news_df, max_features=15000)
    tfidf_rec.build_user_profiles(train_beh)
    print("  [Baseline] TF-IDF profiles built.")

    print("  [Baseline] Training Matrix Factorization SVD...")
    svd_rec = MatrixFactorizationSVD(user_map, news_map, triplets, n_components=EMBED_DIM)

    lgcn_emb = train_lightgcn_baseline(user_map, news_map, triplets,
                                        epochs=EPOCHS, embedding_dim=EMBED_DIM, device=DEVICE)

    # Score functions for embedding-based models
    def emb_score(emb, u, n):
        if u in user_map and n in news_map:
            return float(np.dot(emb[user_map[u]], emb[n_users + news_map[n]]))
        return 0.0

    def hin_predict(u, n):   return emb_score(hin_emb, u, n)
    def han_predict(u, n):   return emb_score(han_emb, u, n)
    def lgcn_predict(u, n):  return emb_score(lgcn_emb, u, n)

    # ── Step 6: Evaluation ──────────────────────────────────────────────────
    print("\n[Step 6/7]  Evaluating All Models on Dev Split...")
    print("-" * 72)

    models_to_eval = [
        ("Popularity Baseline",       pop_rec.predict_score),
        ("TF-IDF Content-Based",       tfidf_rec.predict_score),
        ("SVD Matrix Factorization",   svd_rec.predict_score),
        ("LightGCN (CF-only graph)",   lgcn_predict),
        ("HIN-GNN (unified HIN adj)",  hin_predict),
        ("HAN (relation attention)",   han_predict),
    ]

    results = []
    for name, fn in models_to_eval:
        r = evaluate_model(name, fn, dev_beh, news_df, popularity_counts, news_map, k=TOP_K)
        results.append(r)

    results_df = pd.DataFrame(results)
    res_path   = os.path.join(ARTIFACTS_DIR, "results.csv")
    results_df.to_csv(res_path, index=False)

    print("\n" + "=" * 72)
    print("                   EVALUATION RESULTS")
    print("=" * 72)
    print(results_df.to_string(index=False))
    print("=" * 72)

    # ── Step 7: Diversity Trade-off ─────────────────────────────────────────
    print("\n[Step 7/7]  Accuracy vs Diversity Trade-off (HAN model, α sweep)...")
    alphas    = [1.00, 0.80, 0.60, 0.40, 0.20]
    alpha_desc = {1.00: "Pure Relevance", 0.80: "Relevance-Focused",
                  0.60: "Balanced", 0.40: "Diversity-Focused", 0.20: "High Diversity"}
    tradeoff  = []

    for alpha in alphas:
        acc_l, div_l = [], []
        for row in dev_beh.itertuples(index=False):
            parsed     = parse_impressions(row.impressions)
            candidates = [nid for nid, _ in parsed if nid in news_map]
            relevant   = [nid for nid, lbl in parsed if lbl == 1 and nid in news_map]
            if not candidates or not relevant:
                continue

            scores  = {nid: han_predict(row.user_id, nid) for nid in candidates}
            reranked = diversify_recommendations(candidates, scores, news_df, alpha=alpha, top_k=TOP_K)

            acc = compute_ranking_accuracy(reranked, relevant, k=TOP_K)
            if acc:
                div = compute_novelty_and_diversity(reranked, news_df, popularity_counts, k=TOP_K)
                acc_l.append(acc); div_l.append(div)

        if acc_l:
            avg_a = pd.DataFrame(acc_l).mean().to_dict()
            avg_d = pd.DataFrame(div_l).mean().to_dict()
            tradeoff.append({"Alpha": alpha, "Description": alpha_desc[alpha], **avg_a, **avg_d})

    tradeoff_df = pd.DataFrame(tradeoff)
    td_path     = os.path.join(ARTIFACTS_DIR, "diversity_tradeoff.csv")
    tradeoff_df.to_csv(td_path, index=False)

    print("\n" + "=" * 72)
    print("           ACCURACY vs DIVERSITY TRADE-OFF  (HAN)")
    print("=" * 72)
    print(tradeoff_df[["Alpha", "Description", f"nDCG@{TOP_K}", "MRR", "ILD", "CategoryDiversity"]].to_string(index=False))
    print("=" * 72)

    # ── Plots ───────────────────────────────────────────────────────────────
    _save_tradeoff_plot(tradeoff_df, TOP_K)
    _save_comparison_bar(results_df, TOP_K)

    # ── Save Artifacts ──────────────────────────────────────────────────────
    print("\n[*]  Saving artifacts...")
    np.save(os.path.join(ARTIFACTS_DIR, "embeddings.npy"), hin_emb)
    np.save(os.path.join(ARTIFACTS_DIR, "han_embeddings.npy"), han_emb)
    news_df.to_csv(os.path.join(ARTIFACTS_DIR, "news.csv"), index=False)

    with open(os.path.join(ARTIFACTS_DIR, "user_map.json"), "w") as f:
        json.dump(user_map, f)
    with open(os.path.join(ARTIFACTS_DIR, "news_map.json"), "w") as f:
        json.dump(news_map, f)
    with open(os.path.join(ARTIFACTS_DIR, "popularity.json"), "w") as f:
        json.dump(popularity_counts, f)

    # HAN relation attention weights
    han_model.eval()
    with torch.no_grad():
        import torch.nn.functional as _F
        attn = _F.softmax(han_model.layers[0].attn, dim=0).cpu().tolist()
    attn_dict = dict(zip(RELATION_NAMES, attn))
    with open(os.path.join(ARTIFACTS_DIR, "han_attention.json"), "w") as f:
        json.dump(attn_dict, f, indent=2)
    print(f"  HAN relation attention weights: {attn_dict}")

    print("\n[SUCCESS] Pipeline complete! Artifacts saved to: artifacts/")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_tradeoff_plot(df, k):
    try:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()
        ax1.plot(df["Alpha"], df[f"nDCG@{k}"], "b-o", lw=2, label=f"nDCG@{k}")
        ax2.plot(df["Alpha"], df["ILD"], "g-s", lw=2, label="ILD (Diversity)")
        ax1.set_xlabel("Relevance-Diversity Balance (α)", fontsize=12)
        ax1.set_ylabel(f"nDCG@{k}", color="blue", fontsize=11)
        ax2.set_ylabel("Intra-List Diversity (ILD)", color="green", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="blue")
        ax2.tick_params(axis="y", labelcolor="green")
        plt.title("HAN: Accuracy vs Diversity Trade-off (MMR Re-ranking)", fontsize=13)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        path = os.path.join(ARTIFACTS_DIR, "tradeoff_plot.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"  Saved trade-off plot → {path}")
    except Exception as e:
        print(f"  Trade-off plot skipped: {e}")


def _save_comparison_bar(df, k):
    try:
        metrics = [f"nDCG@{k}", "MRR", f"Recall@{k}", "CategoryDiversity", "ILD"]
        n_met   = len(metrics)
        fig, axes = plt.subplots(1, n_met, figsize=(4 * n_met, 5))

        colors = ["#4e9af1", "#f1c94e", "#f17c4e", "#a3f14e", "#c44ef1", "#f14e9a"]
        models = df["Model"].tolist()

        for ax, metric in zip(axes, metrics):
            vals = df[metric].tolist()
            bars = ax.barh(models, vals, color=colors[:len(models)], alpha=0.85, edgecolor="gray")
            ax.set_title(metric, fontsize=11, fontweight="bold")
            ax.set_xlim(0, max(vals) * 1.25 + 1e-6)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", linestyle="--", alpha=0.4)

        fig.suptitle("Model Comparison — All Metrics", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        path = os.path.join(ARTIFACTS_DIR, "comparison_bar.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved comparison chart → {path}")
    except Exception as e:
        print(f"  Comparison chart skipped: {e}")


if __name__ == "__main__":
    main()
