"""
Interactive Streamlit Web Dashboard
HIN-GNN Diversified News Recommendation System
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.reranker import diversify_recommendations

st.set_page_config(
    page_title="HIN-GNN News Recommender",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

ARTIFACTS_DIR = "artifacts"

@st.cache_data
def load_all_artifacts():
    news_path = os.path.join(ARTIFACTS_DIR, "news.csv")
    emb_path = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
    user_map_path = os.path.join(ARTIFACTS_DIR, "user_map.json")
    news_map_path = os.path.join(ARTIFACTS_DIR, "news_map.json")
    pop_path = os.path.join(ARTIFACTS_DIR, "popularity.json")

    if not all(os.path.exists(p) for p in [news_path, emb_path, user_map_path, news_map_path, pop_path]):
        return None, None, None, None, None

    news_df = pd.read_csv(news_path).fillna("")
    embeddings = np.load(emb_path)
    with open(user_map_path, "r", encoding="utf-8") as f:
        user_map = json.load(f)
    with open(news_map_path, "r", encoding="utf-8") as f:
        news_map = json.load(f)
    with open(pop_path, "r", encoding="utf-8") as f:
        popularity = json.load(f)

    return news_df, embeddings, user_map, news_map, popularity

# Title Header
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">📰 HIN-GNN Diversified News Recommendation System</h1>
    <p class="hero-subtitle">
        A Graph Neural Network framework combining <b>Heterogeneous Information Networks (HIN)</b>, 
        <b>Wikidata Named Entities</b>, and <b>Maximal Marginal Relevance (MMR)</b> for accurate, novel, and diverse discovery.
    </p>
</div>
""", unsafe_allow_html=True)

news_df, embeddings, user_map, news_map, popularity = load_all_artifacts()

if news_df is None:
    st.warning("⚠️ Model artifacts not found! Please run `python pipeline.py` first to train the models and generate artifacts.")
    st.info("Run in terminal:\n```bash\npython pipeline.py\n```")
    st.stop()

# Load User Behaviors to display history
@st.cache_data
def load_user_histories():
    histories = {}
    train_beh_path = os.path.join("data", "train", "behaviors.tsv")
    if os.path.exists(train_beh_path):
        with open(train_beh_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    uid = parts[1]
                    hist = parts[3].split()
                    histories[uid] = hist
    return histories

user_histories = load_user_histories()

# Sidebar
st.sidebar.header("⚙️ Recommendation Controls")
user_list = sorted(list(user_map.keys()))
selected_user = st.sidebar.selectbox("👤 Select User ID", user_list)

alpha = st.sidebar.slider(
    "🎯 Accuracy vs Diversity (α)",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="α = 1.0 means 100% Relevance (no diversity). α = 0.5 balances relevance and category/entity diversity. α = 0.0 maximizes diversity."
)

top_k = st.sidebar.slider(
    "📊 Number of Recommendations (Top-K)",
    min_value=3,
    max_value=15,
    value=6,
    step=1
)

filter_history = st.sidebar.checkbox("🚫 Filter out already clicked news", value=True)

# User Vector Calculation
n_users = len(user_map)
u_idx = user_map[selected_user]
user_vector = embeddings[u_idx]

# Score all articles
candidate_ids = list(news_df["news_id"])
raw_scores = {}
for nid in candidate_ids:
    if nid in news_map:
        n_idx = n_users + news_map[nid]
        news_vec = embeddings[n_idx]
        raw_scores[nid] = float(np.dot(user_vector, news_vec))

# User History filter
user_history = user_histories.get(selected_user, [])
if filter_history and user_history:
    eligible_candidates = [n for n in candidate_ids if n not in user_history]
else:
    eligible_candidates = candidate_ids

# Diversified Ranking
diversified_ids = diversify_recommendations(
    eligible_candidates,
    raw_scores,
    news_df,
    alpha=alpha,
    top_k=top_k
)

# Raw Relevance Ranking (for comparison)
relevance_ids = sorted(
    eligible_candidates,
    key=lambda nid: raw_scores.get(nid, -float("inf")),
    reverse=True
)[:top_k]

# Main Page Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Personalized Recommendations",
    "📈 Model Comparison (Baselines vs Proposed)",
    "⚖️ Accuracy vs Diversity Trade-off",
    "🕸️ Heterogeneous Graph (HIN) Architecture"
])

with tab1:
    col_hist, col_stats = st.columns([2, 1])
    with col_hist:
        st.subheader(f"📖 Click History for User `{selected_user}`")
        if user_history:
            history_rows = news_df[news_df["news_id"].isin(user_history)]
            hist_html = '<div class="history-container">'
            if not history_rows.empty:
                for _, row in history_rows.iterrows():
                    cat = row['category'].lower()
                    badge_class = f"badge-{cat}" if cat in ["technology", "sports", "finance", "science", "entertainment", "health"] else "badge-default"
                    hist_html += f'<div class="history-item"><span class="badge {badge_class}">{cat}</span> <b>{row["title"]}</b></div>'
            else:
                hist_html += f'<div>Articles: {", ".join(user_history)}</div>'
            hist_html += '</div>'
            st.markdown(hist_html, unsafe_allow_html=True)
        else:
            st.caption("No prior history recorded for this user.")

    with col_stats:
        st.subheader("💡 Current Settings")
        mode_text = "Pure Relevance" if alpha == 1.0 else ("Balanced" if alpha == 0.5 else ("Relevance Focused" if alpha > 0.5 else "Diversity Focused"))
        st.markdown(f'<div class="alpha-pill">🎯 Mode: {mode_text} (α = {alpha:.2f})</div>', unsafe_allow_html=True)
        st.metric(label="Diversity Penalty (1 - α)", value=f"{1.0 - alpha:.2f}")

    st.markdown("---")
    
    st.subheader(f"✨ Top {top_k} Recommended Articles (α = {alpha:.2f})")
    
    # Display in 2 responsive columns with modern CSS news cards
    cols = st.columns(2)
    for idx, nid in enumerate(diversified_ids):
        item = news_df[news_df["news_id"] == nid].iloc[0]
        col = cols[idx % 2]
        cat = item['category'].lower()
        badge_class = f"badge-{cat}" if cat in ["technology", "sports", "finance", "science", "entertainment", "health"] else "badge-default"
        score = raw_scores.get(nid, 0.0)
        
        with col:
            card_html = f"""
            <div class="news-card">
                <div>
                    <span class="badge {badge_class}">{item['category']}</span>
                    <h3 class="news-card-title">{idx + 1}. {item['title']}</h3>
                    <p class="news-card-abstract">{item['abstract'][:180]}...</p>
                </div>
                <div class="card-meta-bar">
                    <span class="subcat-label">📁 {item['subcategory']}</span>
                    <span class="gnn-score-pill">GNN Score: {score:.4f}</span>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    # Expandable comparison
    with st.expander("🔍 Compare side-by-side with Pure Relevance Ranking (α = 1.0)"):
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown("#### 🎯 Pure Relevance (α = 1.0)")
            for i, nid in enumerate(relevance_ids):
                row = news_df[news_df["news_id"] == nid].iloc[0]
                cat = row['category'].lower()
                badge_class = f"badge-{cat}" if cat in ["technology", "sports", "finance", "science", "entertainment", "health"] else "badge-default"
                st.markdown(f'<div style="margin-bottom: 0.5rem;"><b>{i+1}.</b> <span class="badge {badge_class}">{cat}</span> {row["title"]}</div>', unsafe_allow_html=True)
        with comp_col2:
            st.markdown(f"#### 🌈 Diversified (α = {alpha:.2f})")
            for i, nid in enumerate(diversified_ids):
                row = news_df[news_df["news_id"] == nid].iloc[0]
                cat = row['category'].lower()
                badge_class = f"badge-{cat}" if cat in ["technology", "sports", "finance", "science", "entertainment", "health"] else "badge-default"
                st.markdown(f'<div style="margin-bottom: 0.5rem;"><b>{i+1}.</b> <span class="badge {badge_class}">{cat}</span> {row["title"]}</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("📊 Experimental Comparison: Proposed HIN-GNN vs Baseline Models")
    results_path = os.path.join(ARTIFACTS_DIR, "results.csv")
    if os.path.exists(results_path):
        res_df = pd.read_csv(results_path)
        st.dataframe(res_df.style.highlight_max(subset=["nDCG@10", "MRR", "Recall@10", "CategoryDiversity", "ILD"], color="#d4edda"))

        st.markdown("#### Key Observations:")
        st.markdown("""
        - **Popularity Baseline:** Recommends globally trending news; achieves low personalization and low novelty.
        - **TF-IDF Content Baseline:** Recommends articles lexically similar to past reading; high intra-list redundancy (low diversity).
        - **LightGCN (CF Graph):** Uses collaborative filtering without category/entity meta-paths; solid accuracy but suffers on cold items.
        - **HIN-GNN (Proposed):** Propagates representations across user-item interactions and semantic entity/category meta-paths, achieving the best balance of ranking accuracy and semantic discovery!
        """)
    else:
        st.info("Run `python pipeline.py` to view evaluation benchmark tables.")

with tab3:
    st.subheader("⚖️ Accuracy vs Diversity Trade-off (Alpha Parameter Sweep)")
    tradeoff_path = os.path.join(ARTIFACTS_DIR, "diversity_tradeoff.csv")
    plot_path = os.path.join(ARTIFACTS_DIR, "tradeoff_plot.png")

    if os.path.exists(tradeoff_path):
        td_df = pd.read_csv(tradeoff_path)
        st.dataframe(td_df)

    if os.path.exists(plot_path):
        st.image(plot_path, caption="Pareto Trade-off: nDCG@10 (Accuracy) vs ILD (Intra-List Diversity) across Alpha")

    st.markdown("""
    > **Research Insight:** As $\\alpha$ decreases from $1.0 \\to 0.5$, Intra-List Diversity (ILD) and Category Diversity rise substantially with only a marginal reduction in nDCG@10. This demonstrates that intelligent graph-based diversification eliminates filter bubbles while preserving user interest alignment.
    """)

with tab4:
    st.subheader("🕸️ Heterogeneous Information Network (HIN) Schema")
    st.markdown("""
    The system models news recommendation as graph message passing on a heterogeneous graph containing 4 node types:
    
    ```mermaid
    graph TD
        User((👤 User)) -->|clicks| News((📰 News Article))
        News -->|has_category| Category((🏷️ Category))
        News -->|has_entity| Entity((🌐 Named Entity))
        News -.->|News-Entity-News| News2((📰 Related News))
        News -.->|News-Category-News| News3((📰 Category News))
        User -.->|User-News-Category| Category
    ```
    
    ### Meta-Paths Formulated:
    1. **$U \\to N \\to C \\to N$ (User-News-Category-News):** Connects users to news articles sharing high-level category affinities.
    2. **$U \\to N \\to E \\to N$ (User-News-Entity-News):** Recommends articles that mention shared real-world Wikidata entities (e.g. OpenAI, Christopher Nolan, Bitcoin).
    3. **$N \\to E \\to N$ (News-Entity-News):** Establishes fine-grained semantic bridges across different publisher outlets.
    """)

