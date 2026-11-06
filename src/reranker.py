"""
Diversity-Aware Re-ranking Module
Implements Maximal Marginal Relevance (MMR) style greedy diversification.
Balances relevance score against intra-list redundancy using the trade-off parameter alpha.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_article_similarity(article_a, article_b):
    """
    Computes hybrid similarity between two news articles based on:
    1. Category overlap (weight 0.4)
    2. Entity Jaccard similarity (weight 0.3)
    3. Title word overlap (weight 0.3)
    """
    # 1. Category match
    cat_sim = 1.0 if (article_a.get("category") and article_a.get("category") == article_b.get("category")) else 0.0

    # 2. Entity overlap
    ents_a = set(article_a.get("entities", []))
    ents_b = set(article_b.get("entities", []))
    if ents_a or ents_b:
        union = ents_a | ents_b
        ent_sim = len(ents_a & ents_b) / len(union) if union else 0.0
    else:
        ent_sim = 0.0

    # 3. Lexical Jaccard similarity of titles
    words_a = set(str(article_a.get("title", "")).lower().split())
    words_b = set(str(article_b.get("title", "")).lower().split())
    if words_a or words_b:
        union_words = words_a | words_b
        lex_sim = len(words_a & words_b) / len(union_words) if union_words else 0.0
    else:
        lex_sim = 0.0

    return 0.4 * cat_sim + 0.3 * ent_sim + 0.3 * lex_sim

def diversify_recommendations(candidate_ids, scores_dict, news_df, alpha=0.75, top_k=10):
    """
    Applies Maximal Marginal Relevance (MMR) re-ranking:
    Score(i) = alpha * Relevance(i) - (1 - alpha) * max_{j in Selected} Similarity(i, j)
    
    Parameters:
    - candidate_ids: list of news IDs to rank.
    - scores_dict: mapping of news_id -> raw prediction score.
    - news_df: DataFrame with news metadata.
    - alpha: trade-off coefficient in [0.0, 1.0]. (1.0 = Pure relevance, 0.0 = Pure diversity).
    - top_k: number of recommendations to return.
    """
    if not candidate_ids:
        return []

    # Filter to existing articles in news_df
    news_lookup = news_df.set_index("news_id").to_dict(orient="index")
    valid_candidates = [n for n in candidate_ids if n in news_lookup]

    if not valid_candidates:
        return []

    # Min-max normalization of candidate relevance scores to [0.0, 1.0]
    raw_scores = [scores_dict.get(n, 0.0) for n in valid_candidates]
    min_s, max_s = min(raw_scores), max(raw_scores)
    if max_s > min_s:
        norm_scores = {n: (scores_dict.get(n, 0.0) - min_s) / (max_s - min_s) for n in valid_candidates}
    else:
        norm_scores = {n: 1.0 for n in valid_candidates}

    selected = []
    remaining = list(valid_candidates)

    while remaining and len(selected) < top_k:
        best_item = None
        best_mmr = -float("inf")

        for nid in remaining:
            rel = norm_scores[nid]
            
            # Compute maximum similarity to already selected items
            if not selected:
                max_sim = 0.0
            else:
                curr_item = news_lookup[nid]
                sims = [calculate_article_similarity(curr_item, news_lookup[s]) for s in selected]
                max_sim = max(sims) if sims else 0.0

            mmr_score = alpha * rel - (1.0 - alpha) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_item = nid

        if best_item is None:
            break

        selected.append(best_item)
        remaining.remove(best_item)

    return selected

