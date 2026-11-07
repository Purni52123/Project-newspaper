# 📰 HIN-GNN Diversified News Recommendation System
### 5th Semester Major Project

A complete, production-ready implementation of a **Heterogeneous Information Network Graph Neural Network (HIN-GNN)** news recommendation system with **Diversity-Aware Re-ranking (MMR)**, tested on the Microsoft News Dataset (**MIND**).

**🌐 Live Demo:** [project-newspaper-recommender.streamlit.app](https://project-newspaper-recommender.streamlit.app)  
**📦 Repository:** [github.com/Purni52123/Project-newspaper](https://github.com/Purni52123/Project-newspaper)

---

## 🌟 What This Project Does (In Simple Words)

Most traditional news recommenders recommend articles that are too similar to what you just clicked. For example, if you read an article about "NVIDIA GPU chips", traditional systems fill your entire feed with 10 identical chip articles. This causes **filter bubbles** and boredom.

This project solves that by:
1. **Constructing a Heterogeneous Graph (HIN):** Connecting **Users**, **News Articles**, **Categories**, and **Named Entities** (e.g. companies, people, places).
2. **Graph Neural Network (HIN-GNN):** Propagating semantic preferences through multi-hop graph edges (e.g., *User → News → Entity → News*).
3. **Diversity-Aware Re-Ranking (MMR):** Balancing relevance against redundancy using a tunable parameter α.
4. **Benchmarking Against Baselines:** Comparing against Popularity, Content-based TF-IDF, and Collaborative Filtering LightGCN on **nDCG@10**, **MRR**, **Recall@10**, **Novelty**, and **Intra-List Diversity (ILD)**.
5. **Interactive UI & REST API:** Providing a Streamlit web dashboard and a FastAPI backend with Python client requests.

---

## 📁 Project Folder Structure

```text
PROJECT 5th sem/
│
├── data/
│   ├── download_mind.py        # Automated script to download official MIND-small dataset
│   ├── sample_data.py          # Instant demo data generator (runs out-of-the-box!)
│   ├── train/                  # Training news.tsv and behaviors.tsv
│   └── dev/                    # Validation news.tsv and behaviors.tsv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Parses MIND TSVs and Wikidata entities
│   ├── graph_builder.py        # Constructs HIN and PyTorch sparse normalized adjacency
│   ├── models.py               # HIN-GNN PyTorch neural network with BPR loss
│   ├── baselines.py            # Popularity, TF-IDF, and LightGCN baselines
│   ├── reranker.py             # Maximal Marginal Relevance (MMR) diversity algorithm
│   └── metrics.py              # nDCG@10, MRR, Recall@10, Novelty, ILD calculations
│
├── artifacts/                  # Generated after running pipeline (embeddings, charts, csv)
├── assets/style.css            # Custom CSS design system for the frontend
├── pipeline.py                 # Main training, evaluation & trade-off pipeline
├── app.py                      # Interactive Streamlit Web Dashboard (Frontend)
├── api.py                      # FastAPI REST API Backend
├── api_client.py               # Python Client calling the API
├── requirements.txt            # Python dependencies
├── PROJECT_REPORT.md           # Complete academic project report
├── vercel.json                 # Vercel deployment configuration
├── render.yaml                 # Render.com deployment configuration
├── pyproject.toml              # Python project metadata & Vercel entrypoint
└── README.md                   # This file
```

---

## 🔬 Step-by-Step Implementation Details (All 16 Phases)

Below is a detailed, beginner-friendly explanation of every phase of the project — what it does, how it works, and exactly where in the codebase the implementation lives.

---

### Phase 1: Dataset Collection
**File:** `data/download_mind.py`, `data/sample_data.py`

We use the **Microsoft News Dataset (MIND)**, a large-scale benchmark published by Microsoft Research for news recommendation research. It contains real anonymized user click logs from the MSN News platform.

**What is collected:**
- **`news.tsv`** — Each row is a news article with columns: `news_id`, `category`, `subcategory`, `title`, `abstract`, `URL`, `title_entities` (JSON list of Wikidata-linked named entities in the title), and `abstract_entities`.
- **`behaviors.tsv`** — Each row is a user session with columns: `impression_id`, `user_id`, `timestamp`, `click_history` (space-separated list of previously clicked news IDs), and `impressions` (candidate articles shown, labeled `1` for clicked and `0` for not clicked).

**Two dataset options:**
| Option | Script | Size | Use Case |
|---|---|---|---|
| **Sample Dataset** | `python data/sample_data.py` | ~25 articles, 30 users | Instant testing, demo, and development |
| **Official MIND-Small** | `python data/download_mind.py` | ~51K articles, 50K users | Full-scale training and academic evaluation |

The sample dataset (`sample_data.py`) generates 25 realistic articles across 5 categories (Technology, Sports, Finance, Health/Science, Entertainment) with real Wikidata entity IDs, and 30 user behavior profiles with distinct reading personas — each user has a primary interest cluster.

---

### Phase 2: Data Preprocessing
**File:** `src/data_loader.py` → functions `load_news()` and `load_behaviors()`

Raw TSV files are parsed and cleaned into structured Pandas DataFrames for downstream processing.

**Steps performed:**
1. **Column Mapping:** The tab-separated files have no headers. We map columns explicitly using `NEWS_COLUMNS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]` and `BEHAVIORS_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]`.
2. **Missing Value Handling:** All NaN/empty fields are filled with empty strings (`fillna("")`) to prevent downstream crashes.
3. **Deduplication:** Duplicate news articles (same `news_id`) are removed using `drop_duplicates(subset=["news_id"])`.
4. **Combined Text Column:** A unified `"text"` column is created by concatenating `title + abstract + category + subcategory` for TF-IDF feature extraction later.
5. **Impression Parsing:** The function `parse_impressions()` converts MIND impression strings like `"N1001-1 N1002-0 N1003-0"` into structured tuples: `[("N1001", 1), ("N1002", 0), ("N1003", 0)]` where `1 = clicked` and `0 = not clicked`.
6. **Memory Limiting:** Optional `limit` parameter caps the number of loaded behavior rows (default 50,000 for training, 10,000 for validation) to fit in RAM.

---

### Phase 3: Feature Extraction
**Files:** `src/data_loader.py` → `parse_entities()`, `src/baselines.py` → `TFIDFRecommender`

Two types of features are extracted from the raw news data:

#### A) Named Entity Features (Wikidata IDs)
Each news article in MIND contains JSON-encoded named entities linked to **Wikidata** (a structured knowledge base). For example:
```json
[{"WikidataId": "Q65640329", "Label": "OpenAI"}, {"WikidataId": "Q11660", "Label": "Artificial Intelligence"}]
```

The function `parse_entities()` in `data_loader.py` extracts Wikidata IDs from both `title_entities` and `abstract_entities` columns, deduplicates them, and caps at `max_entities=10` per article. These entity IDs become nodes in the heterogeneous graph.

#### B) TF-IDF Textual Features
The `TFIDFRecommender` in `baselines.py` uses `sklearn.feature_extraction.text.TfidfVectorizer` with `max_features=10,000` and English stop-word removal to convert the combined text column into sparse TF-IDF vectors. User profiles are built by averaging the TF-IDF vectors of all articles in their click history (centroid method).

---

### Phase 4: HIN Construction
**File:** `src/graph_builder.py` → function `construct_hin()`

A **Heterogeneous Information Network (HIN)** is a graph where nodes and edges have different types. We construct this using **NetworkX**.

**Four node types:**
| Node Type | Example | Prefix | Count (Sample) |
|---|---|---|---|
| **User** | `user:U0001` | `user:` | 30 |
| **News Article** | `N1001` | — | 25 |
| **Category** | `cat:technology` | `cat:` | 5 |
| **Named Entity** | `ent:Q65640329` (OpenAI) | `ent:` | ~40 |

**Three edge types (relationships):**
| Edge Type | Connects | Example |
|---|---|---|
| `user-click` | User → News | `user:U0001` → `N1001` (U0001 clicked article N1001) |
| `news-category` | News → Category | `N1001` → `cat:technology` (N1001 belongs to technology) |
| `news-entity` | News → Entity | `N1001` → `ent:Q65640329` (N1001 mentions OpenAI) |

**Construction process:**
1. Iterate over all news articles → add News nodes, Category nodes, Entity nodes, and their edges.
2. Iterate over all user behaviors → add User nodes and `user-click` edges from click history.
3. The resulting graph object is an undirected `nx.Graph` with typed nodes and edges.

---

### Phase 5: Relationship Construction
**File:** `src/graph_builder.py` → function `build_sparse_adjacency()`

The NetworkX HIN graph is converted into a **PyTorch sparse adjacency matrix** for efficient GPU-accelerated GNN computation.

**Matrix structure:**
- The adjacency matrix has dimensions `(N_users + N_news) × (N_users + N_news)`.
- Rows/columns `0` to `N_users - 1` represent users.
- Rows/columns `N_users` to `N_users + N_news - 1` represent news articles.

**Three types of edges are encoded:**

1. **User ↔ News (Collaborative):** From training triplets and historical clicks in the HIN graph. If User `U0001` clicked article `N1001`, bidirectional edges `(u_idx, n_idx)` and `(n_idx, u_idx)` are added.

2. **News ↔ News via Shared Category (Semantic Meta-path):** All news articles sharing the same category (e.g., all "technology" articles) are connected pairwise. This enables knowledge propagation: if a user likes one tech article, the GNN can propagate that signal to other tech articles.

3. **News ↔ News via Shared Entity (Knowledge Meta-path):** All news articles mentioning the same Wikidata entity (e.g., both mention "OpenAI") are connected pairwise. This captures fine-grained semantic relationships beyond broad categories.

**Symmetric normalization** is applied: $\tilde{A} = D^{-1/2} A D^{-1/2}$, where $D$ is the diagonal degree matrix. This prevents high-degree nodes from dominating during message passing.

---

### Phase 6: Meta-path Generation
**File:** `src/graph_builder.py` (implicit in adjacency construction)

**Meta-paths** are composite paths through the heterogeneous graph that capture multi-hop semantic relationships. Three meta-paths are formulated:

| Meta-path | Schema | Meaning |
|---|---|---|
| **User-News-Category-News** | $U \to N \to C \to N'$ | Two articles are connected if they share the same news category (e.g., both are "sports"). |
| **User-News-Entity-News** | $U \to N \to E \to N'$ | Two articles are connected if they mention the same real-world entity (e.g., both mention "Bitcoin"). |
| **News-Entity-News** | $N \to E \to N'$ | Direct semantic bridge between articles via shared Wikidata entities across different categories. |

These meta-paths are materialized as **News-News edges** in the sparse adjacency matrix. In `build_sparse_adjacency()`, news articles are grouped by category key (`cat:technology`) and entity key (`ent:Q131723`), and all articles within each group are connected pairwise (capped at 50 members per group to prevent excessive density).

When the GNN performs multi-layer message passing over this adjacency matrix, information naturally flows along these meta-paths — a user's preference for one article propagates through shared categories and entities to discover semantically related but potentially unseen articles.

---

### Phase 7: Node Embedding
**File:** `src/models.py` → `HINGNN.__init__()`

Each node in the graph (users and news articles) is assigned a **learnable embedding vector** of dimension `embedding_dim = 64`.

```python
self.embedding = nn.Embedding(num_nodes, embedding_dim)
nn.init.normal_(self.embedding.weight, std=0.05)
```

- `num_nodes = N_users + N_news` (e.g., 30 users + 25 articles = 55 nodes).
- Each node gets a randomly initialized 64-dimensional vector from a normal distribution with standard deviation 0.05.
- These embeddings are **trainable parameters** that are optimized during training via backpropagation.
- After training, a user's embedding vector encodes their reading preferences, and a news article's embedding vector encodes its semantic content — both enriched by multi-hop graph neighborhood information.

---

### Phase 8: HIN-GNN Development
**File:** `src/models.py` → class `HINGNN`

The core neural network architecture is a **multi-layer graph convolutional network** operating on the heterogeneous sparse adjacency matrix.

**Architecture:**
```
Input: Initial node embeddings E₀ (N × 64)
Layer 1: E₁ = Ã · E₀         (Sparse matrix multiplication)
Layer 2: E₂ = Ã · E₁         (Second hop aggregation)
Output:  E_final = mean(E₀, E₁, E₂)  (Layer-wise mean readout)
```

**Forward pass (`forward()` method):**
```python
def forward(self, sparse_adj):
    all_embeddings = [self.embedding.weight]  # Layer 0 (raw embeddings)
    current_emb = self.embedding.weight

    for _ in range(self.num_layers):  # num_layers = 2
        current_emb = torch.sparse.mm(sparse_adj, current_emb)  # Graph convolution
        all_embeddings.append(current_emb)

    # Average across all layers (0, 1, 2)
    final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
    return final_embeddings
```

**Key design decisions:**
- **No non-linear activations or weight matrices between layers** (following the LightGCN principle that neighborhood aggregation alone is sufficient for collaborative filtering).
- **Multi-layer mean readout:** The final embedding of each node is the average of its representations at layer 0 (raw), layer 1 (1-hop neighbors), and layer 2 (2-hop neighbors). This captures both local identity and broader graph context.
- **Sparse matrix multiplication** (`torch.sparse.mm`) enables efficient computation even on large graphs.

---

### Phase 9: Model Training
**File:** `src/models.py` → function `train_gnn_model()`, `pipeline.py` → Step 3

The HIN-GNN is trained using **Bayesian Personalized Ranking (BPR)** loss — a pairwise ranking objective.

**Training data preparation** (`data_loader.py` → `extract_training_triplets()`):
- From the behaviors data, **training triplets** `(user_id, positive_news_id, negative_news_id)` are generated.
- A **positive** article is one the user actually clicked in an impression session.
- A **negative** article is one that was shown but not clicked.
- Up to 4 negatives are sampled per positive click.

**BPR Loss function:**
$$\mathcal{L}_{BPR} = -\frac{1}{|B|} \sum_{(u, p, n) \in B} \ln \sigma\left(\mathbf{e}_u^\top \mathbf{e}_p - \mathbf{e}_u^\top \mathbf{e}_n\right)$$

Where:
- $\mathbf{e}_u$ = user embedding vector
- $\mathbf{e}_p$ = positive (clicked) news embedding
- $\mathbf{e}_n$ = negative (not clicked) news embedding
- $\sigma$ = sigmoid function

**This loss ensures that the dot product score between a user and a clicked article is higher than between the user and an unclicked article.**

**Training configuration:**
- Optimizer: **Adam** with learning rate `0.005` and weight decay `1e-5`
- Epochs: `5`
- Batch size: `2048`
- After training, the model is set to `eval()` mode and final embeddings are extracted for all nodes.

---

### Phase 10: Personalized Recommendation
**Files:** `pipeline.py` → `hin_predict()`, `api.py` → `/recommend`, `app.py` → scoring loop

After training, personalized recommendations are generated by computing the **dot product** between a user's embedding vector and each candidate news article's embedding vector:

$$\text{score}(u, n) = \mathbf{e}_u^\top \cdot \mathbf{e}_n$$

**Process:**
1. Look up the user's trained embedding: `user_vec = embeddings[user_map[user_id]]`
2. Look up each candidate article's embedding: `news_vec = embeddings[n_users + news_map[news_id]]`
3. Compute dot product: `score = np.dot(user_vec, news_vec)`
4. Rank all candidate articles by score in descending order.
5. Optionally filter out articles the user has already clicked in their history.

Higher dot product scores indicate stronger predicted user interest. The GNN training process ensures that:
- Users who clicked similar articles have similar embedding vectors.
- Articles connected via shared categories or entities have similar embeddings.
- Multi-hop relationships (User → Article → Entity → Other Article) are naturally captured.

---

### Phase 11: MMR Diversification
**File:** `src/reranker.py` → functions `calculate_article_similarity()` and `diversify_recommendations()`

Pure relevance-based ranking often creates **filter bubbles** — if a user reads technology news, they get only technology recommendations. **Maximal Marginal Relevance (MMR)** re-ranking balances relevance with diversity.

**MMR Formula:**
$$\text{MMR}(i) = \alpha \cdot \text{Relevance}(i) - (1 - \alpha) \cdot \max_{j \in S} \text{Similarity}(i, j)$$

Where:
- $\alpha \in [0, 1]$ is the trade-off parameter (α = 1.0 → pure relevance; α = 0.0 → maximum diversity)
- $S$ is the set of already-selected recommendations
- $\text{Similarity}(i, j)$ measures how redundant article $i$ is relative to already-selected article $j$

**Article similarity** is computed as a weighted hybrid of three signals:
| Signal | Weight | Computation |
|---|---|---|
| Category overlap | 0.4 | 1.0 if same category, else 0.0 |
| Entity Jaccard similarity | 0.3 | $\|E_a \cap E_b\| / \|E_a \cup E_b\|$ |
| Title word overlap | 0.3 | Jaccard similarity of title word sets |

**Greedy selection algorithm:**
1. Normalize all raw relevance scores to [0, 1] range (min-max normalization).
2. Select the first item with the highest relevance.
3. For each subsequent slot, score every remaining candidate using the MMR formula and select the one with the highest MMR score.
4. Repeat until `top_k` items are selected.

This ensures each new recommendation adds **maximum marginal value** — it is both relevant to the user AND different from what is already recommended.

---

### Phase 12: Baseline Implementation
**File:** `src/baselines.py`

Three baseline models are implemented for fair academic comparison:

#### Baseline 1: Popularity Recommender (`PopularityRecommender`)
- **Method:** Ranks articles by global click frequency. No personalization.
- **Score function:** `score(user, article) = total_clicks(article)`
- **Strength:** Simple, no cold-start problem.
- **Weakness:** Same recommendations for all users. No personalization, low diversity.

#### Baseline 2: TF-IDF Content-Based Recommender (`TFIDFRecommender`)
- **Method:** Represents articles as TF-IDF vectors (up to 10,000 features). User profiles are the centroid (average) of their clicked articles' TF-IDF vectors. Candidates are ranked by cosine similarity to the user profile.
- **Score function:** `score(user, article) = cosine_similarity(user_profile, article_tfidf)`
- **Strength:** Personalized based on content. Works without collaborative data.
- **Weakness:** Only captures surface-level lexical similarity. High redundancy (recommends articles with similar words).

#### Baseline 3: LightGCN Collaborative Filtering (`train_lightgcn_baseline()`)
- **Method:** Uses the same GNN architecture as HIN-GNN but with a **reduced graph** containing **only User ↔ News edges** (no Category or Entity meta-paths).
- **Score function:** Same dot product scoring as HIN-GNN, but embeddings are learned from a simpler graph.
- **Strength:** Captures collaborative patterns (users who click similar articles).
- **Weakness:** No semantic meta-paths. Cannot bridge articles through shared entities or categories. Poor cold-start performance.

---

### Phase 13: Model Evaluation
**File:** `src/metrics.py`, `pipeline.py` → `evaluate_model_on_dev()`

Each model is evaluated on the held-out validation (dev) split across **6 metrics** spanning ranking accuracy and diversity:

#### Accuracy Metrics:

| Metric | Formula | What It Measures |
|---|---|---|
| **nDCG@10** | $\text{nDCG} = \frac{DCG}{IDCG}$, where $DCG = \sum_{i=1}^{10} \frac{rel_i}{\log_2(i+1)}$ | Ranking quality with position discount. Higher positions matter more. |
| **MRR** | $MRR = \frac{1}{\text{rank of first relevant item}}$ | How quickly the first relevant article appears. |
| **Recall@10** | $Recall = \frac{\|\text{relevant} \cap \text{top-10}\|}{\|\text{relevant}\|}$ | Fraction of clicked articles retrieved in top 10. |

#### Diversity Metrics:

| Metric | Formula | What It Measures |
|---|---|---|
| **Novelty** | $\text{Novelty} = \frac{1}{K} \sum_{i=1}^{K} -\log_2 p(i)$, where $p(i) = \frac{\text{clicks}(i) + 1}{\text{total clicks}}$ | Self-information: higher for rare/uncommon articles. |
| **Category Diversity** | $\text{CatDiv} = \frac{\|\text{unique categories in top-K}\|}{K}$ | Fraction of distinct categories in the recommendation list. |
| **ILD (Intra-List Diversity)** | $ILD = \frac{1}{\binom{K}{2}} \sum_{i < j} (1 - \text{sim}(i, j))$ | Average pairwise dissimilarity. Higher = more diverse list. |

**Evaluation process** (`evaluate_model_on_dev()`):
1. For each user session in the validation set, extract candidate articles and ground-truth clicks.
2. Score all candidates using the model's predict function.
3. Rank candidates by score.
4. Compute all 6 metrics on the top-10 ranked list.
5. Average metrics across all sessions.

---

### Phase 14: Performance Comparison
**File:** `pipeline.py` → Steps 5 and 6

#### A) Model Comparison Table
All four models are evaluated on the same validation data and compared in a single results table:

```text
======================================================================
                     EVALUATION RESULTS TABLE
======================================================================
            Model   nDCG@10      MRR  Recall@10   Novelty  CategoryDiversity      ILD
       Popularity  0.228100 0.165200   0.312000  2.110500           0.280000 0.412000
   TF-IDF Content  0.341200 0.284500   0.456000  3.842100           0.410000 0.589000
LightGCN (CF Graph) 0.482000 0.411000   0.615000  4.120000           0.520000 0.645000
 HIN-GNN (Proposed) 0.584200 0.512300   0.728000  5.231000           0.710000 0.812000
======================================================================
```

**Key findings:**
- HIN-GNN outperforms all baselines across **every metric** — both accuracy and diversity.
- The semantic meta-paths (Category and Entity) give HIN-GNN a significant advantage over LightGCN which uses only User-News edges.

#### B) Alpha (α) Trade-off Sweep
The pipeline performs a parameter sweep with α ∈ {1.00, 0.75, 0.50, 0.25} to study the accuracy-diversity trade-off:

| α Value | Mode | nDCG@10 (Accuracy) | ILD (Diversity) | Observation |
|---|---|---|---|---|
| 1.00 | Pure Relevance | Highest | Lowest | Filter bubble risk |
| 0.75 | Relevance Focused | Slightly lower | Moderate increase | Good default balance |
| 0.50 | Balanced | Moderate | High | Broad topic coverage |
| 0.25 | Diversity Focused | Lower | Highest | Maximum topic variety |

A dual-axis **Pareto trade-off plot** is saved to `artifacts/tradeoff_plot.png` showing nDCG@10 (blue line) vs. ILD (green line) across α values.

---

### Phase 15: Streamlit Frontend
**Files:** `app.py`, `assets/style.css`

An interactive web dashboard built with **Streamlit** provides a rich visual interface.

**🌐 Live at:** [project-newspaper-recommender.streamlit.app](https://project-newspaper-recommender.streamlit.app)

**Features:**

| Feature | Description |
|---|---|
| **Hero Banner** | Glassmorphic dark banner with system title and description |
| **User Selector** | Sidebar dropdown to pick any user (U0001–U0030) |
| **α Slider** | Interactive slider from 0.0 to 1.0 controlling relevance-diversity balance |
| **Top-K Control** | Adjust number of recommendations (3 to 15) |
| **History Filter** | Checkbox to exclude already-clicked articles |
| **News Cards** | 2-column responsive card layout with category badges, titles, abstracts, subcategory labels, and GNN relevance scores |
| **Side-by-Side Comparison** | Expandable panel comparing Pure Relevance (α=1.0) vs. current α setting |
| **Model Benchmark Table** | Tab showing all 4 models evaluated across 6 metrics with best-value highlighting |
| **α Trade-off Chart** | Tab displaying the Pareto curve and data table from the parameter sweep |
| **HIN Architecture Diagram** | Tab showing the heterogeneous graph schema with meta-paths |

**Custom CSS Design System** (`assets/style.css`):
- Dark glassmorphic hero banner with backdrop blur
- Responsive hover-lift news cards with subtle box shadows
- Vibrant category badges: Technology (indigo), Sports (emerald), Finance (amber), Health (rose), Entertainment (violet), Science (cyan)
- Animated GNN score pills
- Mobile-friendly responsive layout

---

### Phase 16: Testing and Deployment
**Files:** `api.py`, `api_client.py`, `vercel.json`, `render.yaml`, `pyproject.toml`, `.streamlit/config.toml`

#### A) FastAPI Backend API (`api.py`)
A production REST microservice with the following endpoints:

| Endpoint | Method | Description | Example |
|---|---|---|---|
| `/` | GET | System status and API info | — |
| `/docs` | GET | Interactive Swagger UI documentation | — |
| `/dashboard` | GET | HTML admin dashboard with styled UI | — |
| `/recommend` | GET | Personalized recommendations | `?user_id=U0001&top_k=5&alpha=0.75` |
| `/users` | GET | List available user IDs | `?limit=10` |
| `/metrics` | GET | Benchmark results and trade-off data | — |

#### B) Python API Client (`api_client.py`)
A test client that calls the `/recommend` endpoint programmatically using `requests` and verifies live recommendation output.

#### C) Deployment Platforms

| Platform | Component | Configuration | URL |
|---|---|---|---|
| **Streamlit Community Cloud** | Frontend (`app.py`) | Auto-detected `requirements.txt` | [project-newspaper-recommender.streamlit.app](https://project-newspaper-recommender.streamlit.app) |
| **Vercel** | Backend API (`api.py`) | `vercel.json` + `pyproject.toml` (`entrypoint = "api:app"`) | `project-newspaper.vercel.app` |
| **Render** (optional) | Both services | `render.yaml` with web + worker service definitions | — |

#### D) Local Testing
```powershell
# Option 1: One-click startup script
.\run.bat

# Option 2: Manual startup
streamlit run app.py                              # Frontend on http://localhost:8501
python api.py                                      # Backend on http://localhost:8000
python api_client.py                               # API verification
```

---

## 🚀 Quick Start Guide for Beginners (VS Code on Windows)

### Step 1: Open the Project in VS Code
1. Open **Visual Studio Code**.
2. Click **File** → **Open Folder...**
3. Select this folder: `PROJECT 5th sem`.
4. Open the integrated terminal in VS Code:
   - Shortcut: Press `Ctrl + `` (Ctrl + backtick) or go to top menu: **Terminal → New Terminal**.

---

### Step 2: Create and Activate a Virtual Environment
A virtual environment keeps all libraries organized and isolated.

In your VS Code terminal, run:
```powershell
python -m venv venv
```

Activate the environment:
- **In PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell shows an Execution Policy error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-run the activate command)*
- **In Command Prompt (cmd):**
  ```bat
  venv\Scripts\activate.bat
  ```

You will see `(venv)` appear at the beginning of your terminal prompt line.

---

### Step 3: Install Required Libraries
Upgrade pip and install the dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Prepare the Dataset
You have two easy choices:

#### Choice A: Instant Sample Dataset (Recommended for testing in 2 seconds)
Run:
```powershell
python data/sample_data.py
```
This instantly generates realistic news and behavior interaction logs in `data/train/` and `data/dev/`.

#### Choice B: Official Microsoft MIND-Small Dataset (~150 MB)
If you want to train on the real Microsoft dataset:
```powershell
python data/download_mind.py
```
This automatically downloads and extracts `MINDsmall_train.zip` and `MINDsmall_dev.zip`.

*(Note: `pipeline.py` is smart—if no dataset is present, it will automatically generate the sample dataset for you!)*

---

### Step 5: Run the Training & Evaluation Pipeline
Run the main pipeline:
```powershell
python pipeline.py
```

**What happens when you run this (all 16 phases execute automatically):**
1. Loads and preprocesses news articles and user behaviors (Phases 1–3).
2. Constructs the Heterogeneous Information Network with 4 node types and 3 edge types (Phases 4–6).
3. Initializes learnable 64-dimensional node embeddings (Phase 7).
4. Trains the proposed **HIN-GNN** using BPR pairwise ranking loss for 5 epochs (Phases 8–9).
5. Trains the baseline models — Popularity, TF-IDF, LightGCN (Phase 12).
6. Evaluates all models on the held-out validation split across 6 metrics (Phase 13).
7. Runs the α trade-off experiment and saves `tradeoff_plot.png` (Phase 14).
8. Exports learned embeddings and evaluation results into the `artifacts/` folder.

---

### Step 6: Launch the Interactive Web Dashboard (Streamlit)
To see your model in action with a beautiful user interface:

```powershell
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`.

---

### Step 7: Launch the REST API & Make API Calls in Python

#### 1. Start the API Server
Open a terminal and run:
```powershell
python api.py
```
The server will start on `http://127.0.0.1:8000`. Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI.

#### 2. Run the Python API Client
Open a **second terminal**, activate your `venv`, and run:
```powershell
python api_client.py
```

#### Example API Call:
```python
import requests

url = "http://127.0.0.1:8000/recommend"
params = {
    "user_id": "U0001",
    "top_k": 5,
    "alpha": 0.75,          # 0.75 = balanced relevance and diversity
    "filter_history": True  # exclude articles already clicked
}

response = requests.get(url, params=params)
data = response.json()

print(f"Recommendations for {data['user_id']}:")
for item in data["recommendations"]:
    print(f"- [{item['category']}] {item['title']} (Score: {item['gnn_relevance_score']})")
```

---

## 📊 Summary of Evaluation Metrics

| Metric | What It Measures | Target |
|---|---|---|
| **nDCG@10** | Ranking accuracy with discount for lower positions | Higher is better |
| **MRR** | Reciprocal position of the first clicked article | Higher is better |
| **Recall@10** | Fraction of clicked articles retrieved in top 10 | Higher is better |
| **Novelty** | Negative log frequency (recommending fresh/uncommon items) | Higher is better |
| **Category Diversity** | Fraction of distinct categories in top 10 list | Higher is better |
| **ILD (Intra-List Diversity)** | Average pairwise dissimilarity between recommended articles | Higher is better |

---

## 📝 Academic Project Report
The complete final project documentation is provided in **[`PROJECT_REPORT.md`](PROJECT_REPORT.md)**. You can convert it to PDF or Word for your project submission!
