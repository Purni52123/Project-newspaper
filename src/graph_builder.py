"""
Graph Builder Module
Constructs the Heterogeneous Information Network (HIN) and creates normalized adjacency matrices for GNN.
"""

import networkx as nx
import numpy as np
import torch

def construct_hin(news_df, behaviors_df):
    """
    Constructs a NetworkX Heterogeneous Information Network (HIN).
    Node types: 'user', 'news', 'category', 'entity'
    Edge types: 'user-click', 'news-category', 'news-entity'
    """
    G = nx.Graph()
    valid_news = set(news_df["news_id"])

    # 1. Add News, Category, and Entity nodes + edges
    for row in news_df.itertuples(index=False):
        nid = row.news_id
        G.add_node(nid, node_type="news", title=row.title)

        if row.category:
            cid = f"cat:{row.category}"
            G.add_node(cid, node_type="category", label=row.category)
            G.add_edge(nid, cid, relation="news-category")

        for ent in row.entities:
            eid = f"ent:{ent}"
            G.add_node(eid, node_type="entity", wikidata=ent)
            G.add_edge(nid, eid, relation="news-entity")

    # 2. Add User nodes + User-Click edges from reading history
    for row in behaviors_df.itertuples(index=False):
        uid = f"user:{row.user_id}"
        G.add_node(uid, node_type="user")

        # Past reading clicks
        for nid in row.history.split():
            if nid in valid_news:
                G.add_edge(uid, nid, relation="user-click")

    return G

def build_sparse_adjacency(user_map, news_map, training_triplets, news_df, hin_graph, device):
    """
    Creates a PyTorch Sparse COO normalized adjacency matrix for GNN propagation.
    Matrix covers (N_users + N_news) nodes.
    Includes User-News collaborative edges and News-News meta-path edges (via category & entity).
    """
    n_users = len(user_map)
    n_news = len(news_map)
    total_nodes = n_users + n_news
    edges = set()

    def add_bidirectional_edge(u_idx, v_idx):
        if u_idx != v_idx:
            edges.add((u_idx, v_idx))
            edges.add((v_idx, u_idx))

    # 1. User <-> News interaction edges from training triplets
    for user_id, pos_nid, neg_nid in training_triplets:
        if user_id in user_map:
            u_idx = user_map[user_id]
            if pos_nid in news_map:
                n_idx = n_users + news_map[pos_nid]
                add_bidirectional_edge(u_idx, n_idx)

    # 2. User <-> News interaction edges from HIN graph (historical clicks)
    for u, v, data in hin_graph.edges(data=True):
        if data.get("relation") == "user-click":
            # u is 'user:U0001', v is 'N1001'
            raw_uid = u.replace("user:", "")
            if raw_uid in user_map and v in news_map:
                u_idx = user_map[raw_uid]
                n_idx = n_users + news_map[v]
                add_bidirectional_edge(u_idx, n_idx)

    # 3. News <-> News Meta-Path Connections (Shared Category & Shared Entity)
    # Group news articles by category and entity
    meta_groups = {}
    for row in news_df.itertuples(index=False):
        nid = row.news_id
        if nid not in news_map:
            continue
        
        # Category meta-path
        if row.category:
            cat_key = f"cat:{row.category}"
            meta_groups.setdefault(cat_key, []).append(nid)
            
        # Entity meta-path
        for ent in row.entities:
            ent_key = f"ent:{ent}"
            meta_groups.setdefault(ent_key, []).append(nid)

    # Connect articles within the same group (cap group size to prevent dense bloat)
    for key, members in meta_groups.items():
        capped_members = members[:50]
        for i in range(len(capped_members)):
            for j in range(i + 1, len(capped_members)):
                n1_idx = n_users + news_map[capped_members[i]]
                n2_idx = n_users + news_map[capped_members[j]]
                add_bidirectional_edge(n1_idx, n2_idx)

    if not edges:
        # Fallback to self-loops if empty
        for i in range(total_nodes):
            edges.add((i, i))

    edge_list = np.array(list(edges), dtype=np.int64)
    src = torch.tensor(edge_list[:, 0], dtype=torch.long, device=device)
    dst = torch.tensor(edge_list[:, 1], dtype=torch.long, device=device)

    # Symmetric Normalization: D^{-1/2} * A * D^{-1/2}
    degree = torch.bincount(src, minlength=total_nodes).float().clamp(min=1.0)
    norm_weights = 1.0 / torch.sqrt(degree[src] * degree[dst])

    adj = torch.sparse_coo_tensor(
        torch.stack([src, dst]),
        norm_weights,
        (total_nodes, total_nodes),
        device=device
    ).coalesce()

    return adj

