"""
Evaluation Metrics Module
Calculates recommendation accuracy, novelty, and diversity metrics:
- nDCG@K
- MRR
- Recall@K
- Novelty (Self-Information)
- Category Diversity
- Intra-List Diversity (ILD)
"""

import numpy as np

def compute_ranking_accuracy(ranked_list, ground_truth_relevant, k=10):
    """
    Computes nDCG@K, MRR, and Recall@K for a single user impression list.
    """
    relevant_set = set(ground_truth_relevant)
    if not relevant_set:
        return None

    top_items = ranked_list[:k]
    hits = [1 if item in relevant_set else 0 for item in top_items]

    # 1. DCG@K
    dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))

    # Ideal DCG@K
    ideal_hits_count = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits_count))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    # 2. MRR (Reciprocal Rank)
    reciprocal_rank = 0.0
    for idx, hit in enumerate(hits):
        if hit == 1:
            reciprocal_rank = 1.0 / (idx + 1)
            break

    # 3. Recall@K
    recall = sum(hits) / len(relevant_set)

    return {
        f"nDCG@{k}": float(ndcg),
        "MRR": float(reciprocal_rank),
        f"Recall@{k}": float(recall)
    }

def compute_novelty_and_diversity(ranked_list, news_df, popularity_counts, k=10):
    """
    Computes Novelty, Category Diversity, and Intra-List Diversity (ILD) for the ranked list.
    """
    top_items = ranked_list[:k]
    if not top_items:
        return {"Novelty": 0.0, "CategoryDiversity": 0.0, "ILD": 0.0}

    news_map = news_df.set_index("news_id").to_dict(orient="index")
    total_clicks = max(sum(popularity_counts.values()), 1)

    # 1. Novelty: Mean self-information of items
    novelty_vals = []
    for nid in top_items:
        clicks = popularity_counts.get(nid, 0)
        prob = (clicks + 1.0) / total_clicks
        novelty_vals.append(-np.log2(prob))
    novelty = float(np.mean(novelty_vals)) if novelty_vals else 0.0

    # 2. Category Diversity: Unique categories / total items
    categories = [news_map[nid]["category"] for nid in top_items if nid in news_map and news_map[nid]["category"]]
    cat_diversity = float(len(set(categories)) / len(top_items)) if top_items else 0.0

    # 3. Intra-List Diversity (ILD): 1 - pairwise similarity (Jaccard of entities + categories)
    dissimilarities = []
    for i in range(len(top_items)):
        for j in range(i + 1, len(top_items)):
            nid_a = top_items[i]
            nid_b = top_items[j]
            item_a = news_map.get(nid_a, {})
            item_b = news_map.get(nid_b, {})

            # Category similarity (1 if same, 0 if diff)
            same_cat = 1.0 if item_a.get("category") == item_b.get("category") else 0.0

            # Entity Jaccard
            ents_a = set(item_a.get("entities", []))
            ents_b = set(item_b.get("entities", []))
            ent_union = ents_a | ents_b
            ent_sim = len(ents_a & ents_b) / len(ent_union) if ent_union else 0.0

            pairwise_sim = 0.5 * same_cat + 0.5 * ent_sim
            dissimilarities.append(1.0 - pairwise_sim)

    ild = float(np.mean(dissimilarities)) if dissimilarities else 1.0

    return {
        "Novelty": novelty,
        "CategoryDiversity": cat_diversity,
        "ILD": ild
    }

