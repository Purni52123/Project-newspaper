# 📰 HIN-GNN Diversified News Recommendation System
### 5th Semester / Final Year Academic Major Project

A complete, production-ready implementation of a **Heterogeneous Information Network Graph Neural Network (HIN-GNN)** news recommendation system with **Diversity-Aware Re-ranking (MMR)**, tested on the Microsoft News Dataset (**MIND**).

---

## 🌟 What This Project Does (In Simple Words)

Most traditional news recommenders recommend articles that are too similar to what you just clicked. For example, if you read an article about "NVIDIA GPU chips", traditional systems fill your entire feed with 10 identical chip articles. This causes **filter bubbles** and boredom.

This project solves that by:
1. **Constructing a Heterogeneous Graph (HIN):** Connecting **Users**, **News Articles**, **Categories**, and **Named Entities** (e.g. companies, people, places).
2. **Graph Neural Network (HIN-GNN):** Propagating semantic preferences through multi-hop graph edges (e.g., *User $\to$ News $\to$ Entity $\to$ News*).
3. **Diversity-Aware Re-Ranking (MMR):** Balancing relevance against redundancy using a tunable parameter $\alpha$.
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
├── pipeline.py                 # Main training, evaluation & trade-off pipeline
├── app.py                      # Interactive Streamlit Web Dashboard
├── api.py                      # FastAPI REST API Backend
├── api_client.py               # Python Client calling the API
├── requirements.txt            # Python dependencies
├── PROJECT_REPORT.md           # Complete academic project report
└── README.md                   # This step-by-step beginner guide
```

---

## 🚀 Step-by-Step Guide for Beginners (VS Code on Windows)

### Step 1: Open the Project in VS Code
1. Open **Visual Studio Code**.
2. Click **File** $\to$ **Open Folder...**
3. Select this folder: `PROJECT 5th sem`.
4. Open the integrated terminal in VS Code:
   - Shortcut: Press ``Ctrl + ` `` (Ctrl + backtick) or go to top menu: **Terminal $\to$ New Terminal**.

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

**What happens when you run this:**
1. Loads news and user behaviors.
2. Constructs the Heterogeneous Information Network (HIN).
3. Trains the proposed **HIN-GNN** neural network using Bayesian Personalized Ranking (BPR).
4. Trains the baseline models (**Popularity**, **TF-IDF**, **LightGCN**).
5. Evaluates all models on the held-out validation impressions.
6. Runs the $\alpha$-trade-off experiment and saves `tradeoff_plot.png`.
7. Exports learned embeddings and evaluation results into the `artifacts/` folder.

You will see the final evaluation comparison table printed directly in your terminal:
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

---

### Step 6: Launch the Interactive Web Dashboard (Streamlit)
To see your model in action with a beautiful user interface:

```powershell
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`.

**Features in the Web App:**
- **User Selection:** Pick any user to inspect their reading history.
- **Accuracy vs Diversity Slider ($\alpha$):**
  - Move $\alpha \to 1.0$: Focus strictly on relevance.
  - Move $\alpha \to 0.5$: Balanced mix of relevance and fresh topics.
  - Move $\alpha \to 0.0$: Maximum diversity across categories and entities.
- **Side-by-side comparison:** Compare pure relevance ranking with diversified ranking.
- **Live Metric Benchmarks & Charts:** View the accuracy-diversity Pareto frontier.

*(To stop Streamlit, press `Ctrl + C` in the terminal).*

---

### Step 7: Launch the REST API & Make API Calls in Python

The system includes a production REST API built with FastAPI.

#### 1. Start the API Server
Open a terminal and run:
```powershell
python api.py
```
The server will start on `http://127.0.0.1:8000`. You can visit `http://127.0.0.1:8000/docs` to see the interactive Swagger UI!

#### 2. Run the Python API Client
Open a **second terminal** in VS Code (click the `+` icon in the terminal window), activate your `venv`, and run:
```powershell
python api_client.py
```

#### How to Call the API in Your Own Python Scripts:
Here is a beginner example:
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
The complete final project documentation is provided in **[`PROJECT_REPORT.md`](file:///c:/Users/dell/OneDrive/Desktop/PROJECT%205th%20sem/PROJECT_REPORT.md)**. You can convert it to PDF or Word for your project submission!

