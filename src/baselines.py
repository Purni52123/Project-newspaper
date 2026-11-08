"""
Baseline Recommendation Models
=================================
Implements four baselines for comparison against HIN-GNN and HAN:

1. PopularityRecommender      – Non-personalized frequency baseline
2. TFIDFRecommender           – Content-based cosine similarity baseline
3. MatrixFactorizationSVD     – Classic collaborative filtering (SVD)
4. LightGCN (CF-only graph)   – GNN without HIN meta-path edges
"""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import lil_matrix

from src.models import HINGNN, train_gnn_model


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline 1: Popularity
# ─────────────────────────────────────────────────────────────────────────────

class PopularityRecommender:
    """Ranks articles by overall click frequency — no personalization."""

    def __init__(self, popularity_counts):
        self.popularity = popularity_counts

    def predict_score(self, user_id, news_id):
        return float(self.popularity.get(news_id, 0))

    def rank(self, user_id, candidate_ids):
        return sorted(candidate_ids, key=lambda n: self.predict_score(user_id, n), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline 2: TF-IDF Content-Based
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFRecommender:
    """
    Content-based recommender.
    Represents each user as the centroid of TF-IDF vectors of their history.
    Scores candidates by cosine similarity.
    """

    def __init__(self, news_df, max_features=15000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.news_df      = news_df
        self.news_ids     = list(news_df["news_id"])
        self.news_idx_map = {nid: i for i, nid in enumerate(self.news_ids)}
        self.tfidf_matrix = self.vectorizer.fit_transform(news_df["text"])
        self.user_profiles: dict = {}

    def build_user_profiles(self, behaviors_df):
        for row in behaviors_df.itertuples(index=False):
            hist = str(row.history).split()
            valid = [self.news_idx_map[n] for n in hist if n in self.news_idx_map]
            if valid:
                centroid = np.asarray(self.tfidf_matrix[valid].mean(axis=0))
                self.user_profiles[row.user_id] = centroid

    def predict_score(self, user_id, news_id):
        if user_id not in self.user_profiles or news_id not in self.news_idx_map:
            return 0.0
        u_vec = self.user_profiles[user_id]
        n_vec = self.tfidf_matrix[self.news_idx_map[news_id]]
        return float(cosine_similarity(u_vec, n_vec)[0][0])

    def rank(self, user_id, candidate_ids):
        return sorted(candidate_ids, key=lambda n: self.predict_score(user_id, n), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline 3: Matrix Factorization (SVD)
# ─────────────────────────────────────────────────────────────────────────────

class MatrixFactorizationSVD:
    """
    Classic collaborative filtering via Truncated SVD on the user-item matrix.
    Scores = u_emb · i_emb (dot product in latent space).
    """

    def __init__(self, user_map, news_map, training_triplets, n_components=64):
        n_users = len(user_map)
        n_news  = len(news_map)

        # Build sparse interaction matrix
        R = lil_matrix((n_users, n_news), dtype=np.float32)
        for uid, pos_nid, _ in training_triplets:
            if uid in user_map and pos_nid in news_map:
                R[user_map[uid], news_map[pos_nid]] = 1.0

        R_csr = R.tocsr()

        # Truncated SVD: R ≈ U·S·Vt
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        U   = svd.fit_transform(R_csr)                 # (n_users, k)
        S   = np.diag(svd.singular_values_)
        Vt  = svd.components_                           # (k, n_news)

        self.user_emb = U @ np.sqrt(S)                  # (n_users, k)
        self.news_emb = (np.sqrt(S) @ Vt).T             # (n_news, k)
        self.user_map = user_map
        self.news_map = news_map

    def predict_score(self, user_id, news_id):
        if user_id not in self.user_map or news_id not in self.news_map:
            return 0.0
        u = self.user_emb[self.user_map[user_id]]
        n = self.news_emb[self.news_map[news_id]]
        return float(np.dot(u, n))

    def rank(self, user_id, candidate_ids):
        return sorted(candidate_ids, key=lambda n: self.predict_score(user_id, n), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline 4: LightGCN (CF-only — no HIN meta-paths)
# ─────────────────────────────────────────────────────────────────────────────

def train_lightgcn_baseline(user_map, news_map, training_triplets,
                             epochs=5, embedding_dim=64, device="cpu"):
    """
    LightGCN Collaborative Filtering baseline.
    Graph contains ONLY User↔News edges — no category or entity meta-paths.
    """
    n_users = len(user_map)
    n_news  = len(news_map)
    total   = n_users + n_news
    edges   = set()

    for uid, pos_nid, _ in training_triplets:
        if uid in user_map and pos_nid in news_map:
            u = user_map[uid]
            n = n_users + news_map[pos_nid]
            edges.add((u, n)); edges.add((n, u))

    if not edges:
        for i in range(total):
            edges.add((i, i))

    arr = np.array(list(edges), dtype=np.int64)
    src = torch.tensor(arr[:, 0], dtype=torch.long, device=device)
    dst = torch.tensor(arr[:, 1], dtype=torch.long, device=device)
    deg = torch.bincount(src, minlength=total).float().clamp(min=1.0)
    w   = 1.0 / torch.sqrt(deg[src] * deg[dst])

    adj = torch.sparse_coo_tensor(
        torch.stack([src, dst]), w, (total, total), device=device
    ).coalesce()

    model = HINGNN(total, n_users, n_news, embedding_dim=embedding_dim, num_layers=3).to(device)
    print("  [Baseline] Training LightGCN (pure collaborative filtering)...")
    return train_gnn_model(model, adj, training_triplets, user_map, news_map,
                           epochs=epochs, device=device)
