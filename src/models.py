"""
GNN Models Module
===================
Implements three graph neural network architectures:

1. HINGNN   – Multi-layer LightGCN-style propagation over combined HIN adj matrix
2. HAN      – Heterogeneous Attention Network with per-relation message passing
               and learnable semantic attention to fuse relation-specific embeddings
3. BPR helpers – shared training loop

All use Bayesian Personalized Ranking (BPR) loss.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Model 1 : HIN-GNN  (LightGCN-style on combined HIN adjacency)
# ─────────────────────────────────────────────────────────────────────────────

class HINGNN(nn.Module):
    """
    Heterogeneous Information Network Graph Neural Network.
    Performs multi-layer symmetric-normalized graph convolution over the unified
    user-news-category-entity adjacency matrix.

    Architecture follows LightGCN (He et al., 2020) with mean-pooled layer outputs.
    """

    def __init__(self, num_nodes, num_users, num_news, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_nodes     = num_nodes
        self.num_users     = num_users
        self.num_news      = num_news
        self.num_layers    = num_layers
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.05)

    def forward(self, sparse_adj):
        """Multi-layer message passing; returns mean-pooled final embeddings."""
        all_embs = [self.embedding.weight]
        cur = self.embedding.weight
        for _ in range(self.num_layers):
            cur = torch.sparse.mm(sparse_adj, cur)
            all_embs.append(cur)
        return torch.stack(all_embs, dim=0).mean(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Model 2 : HAN  (Heterogeneous Attention Network)
# ─────────────────────────────────────────────────────────────────────────────

class HANLayer(nn.Module):
    """
    Single HAN propagation layer.
    For each relation r, propagate embeddings through adj_r, then fuse with
    learned scalar semantic attention weights α_r (softmax across relations).
    """

    def __init__(self, embedding_dim, n_relations, dropout=0.1):
        super().__init__()
        self.attn = nn.Parameter(torch.ones(n_relations) / n_relations)
        self.dropout = nn.Dropout(dropout)
        self.transform = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in range(n_relations)
        ])
        for lin in self.transform:
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, x, relation_adjs):
        """
        x              : (N, D) current node features
        relation_adjs  : list of K sparse tensors (N×N)
        Returns        : (N, D) updated node features
        """
        attn_w = F.softmax(self.attn, dim=0)          # (K,)
        agg = torch.zeros_like(x)
        for k, adj in enumerate(relation_adjs):
            propagated = torch.sparse.mm(adj, self.dropout(x))  # (N, D)
            agg = agg + attn_w[k] * self.transform[k](propagated)
        return F.leaky_relu(agg, 0.2)


class HAN(nn.Module):
    """
    Heterogeneous Attention Network.
    Uses per-relation adjacency matrices and semantic attention to fuse them.
    """

    def __init__(self, num_nodes, num_users, num_news,
                 embedding_dim=64, num_layers=2,
                 relation_names=None, dropout=0.1):
        super().__init__()
        self.num_nodes     = num_nodes
        self.num_users     = num_users
        self.num_news      = num_news
        self.embedding_dim = embedding_dim
        self.relation_names = relation_names or []

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.05)

        n_rel = len(self.relation_names)
        self.layers = nn.ModuleList([
            HANLayer(embedding_dim, n_rel, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, relation_adjs):
        """
        relation_adjs: list of K sparse (N×N) tensors in same order as relation_names
        Returns: (N, D) final embeddings
        """
        all_embs = [self.embedding.weight]
        x = self.embedding.weight
        for layer in self.layers:
            x = self.norm(layer(x, relation_adjs))
            all_embs.append(x)
        return torch.stack(all_embs, dim=0).mean(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared BPR Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def _bpr_loss(u_emb, pos_emb, neg_emb):
    pos_scores = (u_emb * pos_emb).sum(dim=1)
    neg_scores = (u_emb * neg_emb).sum(dim=1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def train_gnn_model(model, sparse_adj, triplets, user_map, news_map,
                    epochs=5, lr=0.005, batch_size=2048, device="cpu"):
    """
    Train HINGNN with BPR loss using a single combined adjacency matrix.
    Returns numpy embedding array of shape (n_users + n_news, dim).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()

    if not triplets:
        raise ValueError("No training triplets provided.")

    n_users = len(user_map)

    for epoch in range(1, epochs + 1):
        random.shuffle(triplets)
        epoch_loss = 0.0

        for i in range(0, len(triplets), batch_size):
            batch = triplets[i : i + batch_size]
            u_idx = torch.tensor([user_map[u]            for u, p, n in batch], dtype=torch.long, device=device)
            p_idx = torch.tensor([n_users + news_map[p]  for u, p, n in batch], dtype=torch.long, device=device)
            n_idx = torch.tensor([n_users + news_map[n]  for u, p, n in batch], dtype=torch.long, device=device)

            optimizer.zero_grad()
            emb = model(sparse_adj)
            loss = _bpr_loss(emb[u_idx], emb[p_idx], emb[n_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        print(f"  [HIN-GNN] Epoch {epoch:02d}/{epochs} | BPR Loss: {epoch_loss/len(triplets):.4f}")

    model.eval()
    with torch.no_grad():
        return model(sparse_adj).detach().cpu().numpy()


def train_han_model(model, relation_adjs, triplets, user_map, news_map,
                    epochs=5, lr=0.005, batch_size=2048, device="cpu"):
    """
    Train HAN with BPR loss using per-relation adjacency matrices.
    Returns numpy embedding array of shape (n_users + n_news, dim).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    model.train()

    if not triplets:
        raise ValueError("No training triplets provided.")

    n_users = len(user_map)
    rel_adjs_list = [relation_adjs[r] for r in model.relation_names]

    for epoch in range(1, epochs + 1):
        random.shuffle(triplets)
        epoch_loss = 0.0

        for i in range(0, len(triplets), batch_size):
            batch = triplets[i : i + batch_size]
            u_idx = torch.tensor([user_map[u]            for u, p, n in batch], dtype=torch.long, device=device)
            p_idx = torch.tensor([n_users + news_map[p]  for u, p, n in batch], dtype=torch.long, device=device)
            n_idx = torch.tensor([n_users + news_map[n]  for u, p, n in batch], dtype=torch.long, device=device)

            optimizer.zero_grad()
            emb = model(rel_adjs_list)
            loss = _bpr_loss(emb[u_idx], emb[p_idx], emb[n_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        scheduler.step()
        attn = F.softmax(model.layers[0].attn, dim=0).detach().cpu().tolist()
        attn_str = " ".join(f"{model.relation_names[k]}={v:.3f}" for k, v in enumerate(attn))
        print(f"  [HAN]     Epoch {epoch:02d}/{epochs} | BPR Loss: {epoch_loss/len(triplets):.4f} | Attn: [{attn_str}]")

    model.eval()
    with torch.no_grad():
        return model(rel_adjs_list).detach().cpu().numpy()
