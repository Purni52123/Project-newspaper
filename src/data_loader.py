"""
Data Loader Module for MIND Dataset
Loads news metadata and user interaction behaviors.
"""

import os
import json
import pandas as pd

NEWS_COLUMNS = [
    "news_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities"
]

BEHAVIORS_COLUMNS = [
    "impression_id", "user_id", "time", "history", "impressions"
]

def load_news(file_path, max_entities=10):
    """
    Load news metadata from news.tsv.
    Extracts text features and entity Wikidata IDs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"News file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        names=NEWS_COLUMNS,
        header=None,
        quoting=3,
        keep_default_na=False,
        encoding="utf-8"
    )

    for col in NEWS_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    # Combined text for TF-IDF and NLP analysis
    df["text"] = (
        df["title"] + " " + df["abstract"] + " " +
        df["category"] + " " + df["subcategory"]
    ).str.strip()

    def parse_entities(row):
        entities = []
        for col in ["title_entities", "abstract_entities"]:
            val = row[col]
            if val and val.startswith("["):
                try:
                    items = json.loads(val)
                    for it in items:
                        wid = it.get("WikidataId")
                        if wid and wid not in entities:
                            entities.append(wid)
                except Exception:
                    pass
        return entities[:max_entities]

    df["entities"] = df.apply(parse_entities, axis=1)
    return df.drop_duplicates(subset=["news_id"]).reset_index(drop=True)

def load_behaviors(file_path, limit=None):
    """
    Load user behavior logs from behaviors.tsv.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Behaviors file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        names=BEHAVIORS_COLUMNS,
        header=None,
        quoting=3,
        keep_default_na=False,
        encoding="utf-8"
    )

    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    df["history"] = df["history"].fillna("").astype(str)
    df["impressions"] = df["impressions"].fillna("").astype(str)
    return df

def parse_impressions(impressions_str):
    """
    Parse MIND impression string (e.g. 'N1001-1 N1002-0') into list of tuples (news_id, label).
    """
    parsed = []
    for item in str(impressions_str).split():
        if "-" in item:
            parts = item.rsplit("-", 1)
            if len(parts) == 2 and parts[1] in ("0", "1"):
                parsed.append((parts[0], int(parts[1])))
    return parsed

def extract_training_triplets(behaviors_df, valid_news_ids, max_negatives_per_pos=4):
    """
    Generates training triplets (user_id, pos_news_id, neg_news_id) for Bayesian Personalized Ranking (BPR).
    Also counts click frequency per news article.
    """
    triplets = []
    popularity_counts = {}

    for row in behaviors_df.itertuples(index=False):
        # Click history counts
        history_items = row.history.split()
        for nid in history_items:
            if nid in valid_news_ids:
                popularity_counts[nid] = popularity_counts.get(nid, 0) + 1

        # Impression clicks
        candidates = parse_impressions(row.impressions)
        positives = [n for n, label in candidates if label == 1 and n in valid_news_ids]
        negatives = [n for n, label in candidates if label == 0 and n in valid_news_ids]

        for pos in positives:
            popularity_counts[pos] = popularity_counts.get(pos, 0) + 1
            for neg in negatives[:max_negatives_per_pos]:
                triplets.append((row.user_id, pos, neg))

    return triplets, popularity_counts

