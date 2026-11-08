"""
Enhanced Graph Builder Module
=================================
Constructs a rich Heterogeneous Information Network (HIN) with:
  Node types : user | news | category | subcategory | entity
  Edge types  : user_click | news_category | news_subcategory | news_entity | co_entity | co_category

Also builds typed sparse adjacency matrices for multi-relational GNN propagation.
"""

import networkx as nx
import numpy as np
import torch
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
#  HIN Construction
# ─────────────────────────────────────────────────────────────────────────────

def construct_hin(news_df, behaviors_df):
    """
    Builds a NetworkX MultiGraph HIN with 5 node types and 6 edge relation types.

    Returns
    -------
    G          : nx.MultiGraph
    node_meta  : dict  → {node_id: {'type': str, ...}}
    edge_counts: dict  → {relation_type: count}
    """
    G = nx.MultiGraph()
    valid_news = set(news_df["news_id"])
    edge_counts = defaultdict(int)

    # ── 1. News, Category, Subcategory, Entity nodes + their edges ────────────
    cat_members   = defaultdict(list)   # cat_key   → [news_ids]
    subcat_members= defaultdict(list)   # subcat_key→ [news_ids]
    entity_members= defaultdict(list)   # ent_key   → [news_ids]

    for row in news_df.itertuples(index=False):
        nid = row.news_id
        G.add_node(nid, node_type="news", title=getattr(row, "title", ""))

        # Category
        if row.category:
            cid = f"cat:{row.category}"
            G.add_node(cid, node_type="category", label=row.category)
            G.add_edge(nid, cid, relation="news_category")
            edge_counts["news_category"] += 1
            cat_members[cid].append(nid)

        # Subcategory
        if row.subcategory:
            scid = f"subcat:{row.subcategory}"
            G.add_node(scid, node_type="subcategory", label=row.subcategory)
            G.add_edge(nid, scid, relation="news_subcategory")
            edge_counts["news_subcategory"] += 1
            subcat_members[scid].append(nid)

        # Entities
        for ent in row.entities:
            eid = f"ent:{ent}"
            G.add_node(eid, node_type="entity", wikidata=ent)
            G.add_edge(nid, eid, relation="news_entity")
            edge_counts["news_entity"] += 1
            entity_members[eid].append(nid)

    # ── 2. User nodes + user_click edges ──────────────────────────────────────
    for row in behaviors_df.itertuples(index=False):
        uid = f"user:{row.user_id}"
        G.add_node(uid, node_type="user")
        for nid in str(row.history).split():
            if nid in valid_news:
                G.add_edge(uid, nid, relation="user_click")
                edge_counts["user_click"] += 1

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Print stats
    type_counts = defaultdict(int)
    for _, data in G.nodes(data=True):
        type_counts[data.get("node_type", "unknown")] += 1

    print(f"  HIN Nodes : {n_nodes:,} → " +
          " | ".join(f"{t}:{c}" for t, c in sorted(type_counts.items())))
    print(f"  HIN Edges : {n_edges:,} → " +
          " | ".join(f"{r}:{c}" for r, c in sorted(edge_counts.items())))

    return G, dict(edge_counts)


# ─────────────────────────────────────────────────────────────────────────────
#  Sparse Adjacency Matrix — Full HIN (for HAN / HIN-GNN)
# ─────────────────────────────────────────────────────────────────────────────

def build_sparse_adjacency(user_map, news_map, training_triplets, news_df, hin_graph, device):
    """
    Builds normalized symmetric adjacency matrix over (users + news) nodes.

    Includes:
      • User ↔ News edges from training triplets
      • User ↔ News edges from HIN historical clicks
      • News ↔ News meta-path edges (shared category & shared entity)
    """
    n_users = len(user_map)
    n_news  = len(news_map)
    total   = n_users + n_news
    src_list, dst_list = [], []

    def add_bi(u, v):
        if u != v:
            src_list.append(u); dst_list.append(v)
            src_list.append(v); dst_list.append(u)

    # Triplet interactions
    for user_id, pos_nid, neg_nid in training_triplets:
        if user_id in user_map and pos_nid in news_map:
            add_bi(user_map[user_id], n_users + news_map[pos_nid])

    # HIN historical clicks
    for u, v, data in hin_graph.edges(data=True):
        if data.get("relation") == "user_click":
            raw_uid = u.replace("user:", "") if u.startswith("user:") else v.replace("user:", "")
            nid     = v if u.startswith("user:") else u
            if raw_uid in user_map and nid in news_map:
                add_bi(user_map[raw_uid], n_users + news_map[nid])

    # News–News meta-paths via category and entity (capped groups)
    meta_groups: dict = {}
    for row in news_df.itertuples(index=False):
        nid = row.news_id
        if nid not in news_map:
            continue
        if row.category:
            meta_groups.setdefault(f"cat:{row.category}", []).append(nid)
        if row.subcategory:
            meta_groups.setdefault(f"subcat:{row.subcategory}", []).append(nid)
        for ent in row.entities:
            meta_groups.setdefault(f"ent:{ent}", []).append(nid)

    for members in meta_groups.values():
        capped = members[:20]   # cap to avoid quadratic blowup
        m = np.array([n_users + news_map[n] for n in capped], dtype=np.int64)
        if len(m) < 2:
            continue
        ii, jj = np.triu_indices(len(m), k=1)
        src_list.extend(m[ii].tolist()); dst_list.extend(m[jj].tolist())
        src_list.extend(m[jj].tolist()); dst_list.extend(m[ii].tolist())

    if not src_list:
        for i in range(total):
            src_list.append(i); dst_list.append(i)

    # Deduplicate via unique rows
    edge_arr = np.unique(np.column_stack([src_list, dst_list]), axis=0)
    src = torch.tensor(edge_arr[:, 0], dtype=torch.long, device=device)
    dst = torch.tensor(edge_arr[:, 1], dtype=torch.long, device=device)

    degree = torch.bincount(src, minlength=total).float().clamp(min=1.0)
    norm_w  = 1.0 / torch.sqrt(degree[src] * degree[dst])

    adj = torch.sparse_coo_tensor(
        torch.stack([src, dst]),
        norm_w,
        (total, total),
        device=device,
    ).coalesce()

    print(f"  Adjacency : {len(edge_arr):,} directed edges | size {total}×{total}")
    return adj


# ─────────────────────────────────────────────────────────────────────────────
#  Per-Relation Adjacency Matrices (for multi-relational / HAN model)
# ─────────────────────────────────────────────────────────────────────────────

def build_relation_adjacencies(user_map, news_map, training_triplets, news_df, device,
                                max_meta_group=15):
    """
    Returns a dict of {relation_name: sparse_adj_tensor} for multi-relational GNN.

    Relations:
      'user_news'       : user ↔ news interaction edges
      'news_category'   : news ↔ news via shared category
      'news_subcategory': news ↔ news via shared subcategory
      'news_entity'     : news ↔ news via shared entity
    """
    n_users = len(user_map)
    n_news  = len(news_map)
    total   = n_users + n_news

    def _pairs_to_sparse(src_arr, dst_arr):
        """Build normalized sparse adj from arrays of source/dest indices."""
        if len(src_arr) == 0:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, device=device),
                (total, total),
            ).coalesce()
        src = torch.tensor(src_arr, dtype=torch.long, device=device)
        dst = torch.tensor(dst_arr, dtype=torch.long, device=device)
        deg = torch.bincount(src, minlength=total).float().clamp(min=1.0)
        w   = 1.0 / torch.sqrt(deg[src] * deg[dst])
        return torch.sparse_coo_tensor(
            torch.stack([src, dst]), w, (total, total), device=device
        ).coalesce()

    # --- user_news ---
    un_src, un_dst = [], []
    for uid, pos_nid, _ in training_triplets:
        if uid in user_map and pos_nid in news_map:
            u = user_map[uid]
            n = n_users + news_map[pos_nid]
            un_src.extend([u, n]); un_dst.extend([n, u])

    # --- news-news via shared attributes (use numpy for speed) ---
    groups = {"news_category": {}, "news_subcategory": {}, "news_entity": {}}
    for row in news_df.itertuples(index=False):
        nid = row.news_id
        if nid not in news_map:
            continue
        idx = n_users + news_map[nid]
        if row.category:
            groups["news_category"].setdefault(row.category, []).append(idx)
        if row.subcategory:
            groups["news_subcategory"].setdefault(row.subcategory, []).append(idx)
        for ent in row.entities:
            groups["news_entity"].setdefault(ent, []).append(idx)

    rel_data = {
        "user_news":        (un_src, un_dst),
        "news_category":    ([], []),
        "news_subcategory": ([], []),
        "news_entity":      ([], []),
    }
    for rel, grp_dict in groups.items():
        src_list, dst_list = rel_data[rel]
        for members in grp_dict.values():
            capped = members[:max_meta_group]
            m = np.array(capped, dtype=np.int64)
            if len(m) < 2:
                continue
            # All pairs via meshgrid
            ii, jj = np.triu_indices(len(m), k=1)
            src_pairs = m[ii]; dst_pairs = m[jj]
            src_list.extend(src_pairs.tolist()); dst_list.extend(dst_pairs.tolist())
            src_list.extend(dst_pairs.tolist()); dst_list.extend(src_pairs.tolist())

    result = {}
    for rel, (src_list, dst_list) in rel_data.items():
        result[rel] = _pairs_to_sparse(np.array(src_list, dtype=np.int64),
                                        np.array(dst_list, dtype=np.int64))
        print(f"  Relation '{rel}': {len(src_list):,} directed edges")

    return result
