"""
GNN Recommender Models Module
Implements Heterogeneous Graph Neural Network (HIN-GNN) and LightGCN propagation architectures.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class HINGNN(nn.Module):
    """
    Heterogeneous Information Network Graph Neural Network (HIN-GNN).
    Performs multi-layer graph convolution over user-news-entity-category graph.
    """
    def __init__(self, num_nodes, num_users, num_news, embedding_dim=64, num_layers=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_users = num_users
        self.num_news = num_news
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Initial node embeddings
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.05)

    def forward(self, sparse_adj):
        """
        Multi-layer message passing with residual mean readout.
        """
        all_embeddings = [self.embedding.weight]
        current_emb = self.embedding.weight

        for _ in range(self.num_layers):
            current_emb = torch.sparse.mm(sparse_adj, current_emb)
            all_embeddings.append(current_emb)

        # Average across all propagation layers (including 0-th layer)
        final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return final_embeddings

def train_gnn_model(model, sparse_adj, triplets, user_map, news_map, epochs=5, lr=0.005, batch_size=2048, device="cpu"):
    """
    Trains the HIN-GNN model using Bayesian Personalized Ranking (BPR) Loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()

    if not triplets:
        raise ValueError("Cannot train GNN: No training triplets provided.")

    n_users = len(user_map)

    for epoch in range(1, epochs + 1):
        random.shuffle(triplets)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i + batch_size]
            
            # Map batch strings to tensor indices
            u_indices = torch.tensor([user_map[u] for u, p, n in batch], dtype=torch.long, device=device)
            p_indices = torch.tensor([n_users + news_map[p] for u, p, n in batch], dtype=torch.long, device=device)
            n_indices = torch.tensor([n_users + news_map[n] for u, p, n in batch], dtype=torch.long, device=device)

            optimizer.zero_grad()
            all_emb = model(sparse_adj)

            u_emb = all_emb[u_indices]
            pos_emb = all_emb[p_indices]
            neg_emb = all_emb[n_indices]

            # Inner product scores
            pos_scores = (u_emb * pos_emb).sum(dim=1)
            neg_scores = (u_emb * neg_emb).sum(dim=1)

            # BPR Loss: -ln(sigmoid(pos_score - neg_score))
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch)
            n_batches += 1

        avg_loss = epoch_loss / len(triplets)
        print(f"  [GNN Training] Epoch {epoch:02d}/{epochs:02d} | BPR Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        final_embeddings = model(sparse_adj).detach().cpu().numpy()
        
    return final_embeddings

