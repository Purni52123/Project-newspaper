"""
Baseline Recommendation Models
Implements:
1. Popularity-Based Recommender (Non-personalized frequency baseline)
2. TF-IDF Content-Based Recommender (Textual cosine similarity baseline)
3. LightGCN / Collaborative Filtering (Graph baseline without HIN meta-paths)
"""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.models import HINGNN, train_gnn_model

class PopularityRecommender:
    """
    Ranks articles strictly by overall click popularity.
    """
    def __init__(self, popularity_counts):
        self.popularity = popularity_counts

    def predict_score(self, user_id, news_id):
        return float(self.popularity.get(news_id, 0))

    def rank(self, user_id, candidate_ids):
        return sorted(candidate_ids, key=lambda n: self.predict_score(user_id, n), reverse=True)


class TFIDFRecommender:
    """
    Content-based recommender:
    Represents users by averaging TF-IDF vectors of their read history articles.
    Scores candidates using Cosine Similarity.
    """
    def __init__(self, news_df, max_features=10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english"
        )
        self.news_df = news_df
        self.news_ids = list(news_df["news_id"])
        self.news_idx_map = {nid: i for i, nid in enumerate(self.news_ids)}
        
        # Fit-transform news texts
        self.tfidf_matrix = self.vectorizer.fit_transform(news_df["text"])
        self.user_profiles = {}

    def build_user_profiles(self, behaviors_df):
        """
        Creates user centroid vectors from click history.
        """
        for row in behaviors_df.itertuples(index=False):
            history_nids = row.history.split()
            valid_indices = [self.news_idx_map[nid] for nid in history_nids if nid in self.news_idx_map]
            if valid_indices:
                centroid = self.tfidf_matrix[valid_indices].mean(axis=0)
                self.user_profiles[row.user_id] = np.asarray(centroid)

    def predict_score(self, user_id, news_id):
        if user_id not in self.user_profiles or news_id not in self.news_idx_map:
            return 0.0
        user_vec = self.user_profiles[user_id]
        news_vec = self.tfidf_matrix[self.news_idx_map[news_id]]
        sim = cosine_similarity(user_vec, news_vec)[0][0]
        return float(sim)

    def rank(self, user_id, candidate_ids):
        return sorted(candidate_ids, key=lambda n: self.predict_score(user_id, n), reverse=True)


def train_lightgcn_baseline(user_map, news_map, training_triplets, epochs=5, embedding_dim=64, device="cpu"):
    """
    LightGCN Collaborative Filtering Baseline:
    Graph contains ONLY User-News edges (no Category or Entity meta-paths).
    """
    n_users = len(user_map)
    n_news = len(news_map)
    total_nodes = n_users + n_news
    edges = set()

    for user_id, pos_nid, neg_nid in training_triplets:
        if user_id in user_map and pos_nid in news_map:
            u_idx = user_map[user_id]
            n_idx = n_users + news_map[pos_nid]
            edges.add((u_idx, n_idx))
            edges.add((n_idx, u_idx))

    if not edges:
        for i in range(total_nodes):
            edges.add((i, i))

    edge_list = np.array(list(edges), dtype=np.int64)
    src = torch.tensor(edge_list[:, 0], dtype=torch.long, device=device)
    dst = torch.tensor(edge_list[:, 1], dtype=torch.long, device=device)

    degree = torch.bincount(src, minlength=total_nodes).float().clamp(min=1.0)
    norm_weights = 1.0 / torch.sqrt(degree[src] * degree[dst])

    adj = torch.sparse_coo_tensor(
        torch.stack([src, dst]),
        norm_weights,
        (total_nodes, total_nodes),
        device=device
    ).coalesce()

    model = HINGNN(total_nodes, n_users, n_news, embedding_dim=embedding_dim, num_layers=2).to(device)
    print("  [Baseline] Training LightGCN (Pure Collaborative Filtering Graph)...")
    embeddings = train_gnn_model(model, adj, training_triplets, user_map, news_map, epochs=epochs, device=device)
    return embeddings

