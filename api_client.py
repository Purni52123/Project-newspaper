"""
API Client Demo Script in Python
Demonstrates how to interact with the HIN-GNN News Recommendation API programmatically.
Uses the Python 'requests' library to query endpoints, fetch recommendations, and compare alpha values.
"""

import sys
import json
import requests

API_BASE_URL = "http://127.0.0.1:8000"

def check_api_health():
    """Checks if the FastAPI server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=3)
        if response.status_code == 200:
            print("[+] API Server is online!")
            print(f"    System Info: {response.json()}")
            return True
    except requests.exceptions.ConnectionError:
        print("[-] Could not connect to API server at http://127.0.0.1:8000")
        print("    Please start the API server in a separate terminal using:")
        print("    python api.py")
        return False
    return False

def get_recommendations_from_api(user_id="U0001", top_k=5, alpha=0.75):
    """
    Calls GET /recommend endpoint to fetch personalized and diversified recommendations.
    """
    endpoint = f"{API_BASE_URL}/recommend"
    params = {
        "user_id": user_id,
        "top_k": top_k,
        "alpha": alpha,
        "filter_history": True
    }

    print(f"\n[*] Querying API for User '{user_id}' with alpha={alpha} (top_k={top_k})...")
    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        print("=" * 70)
        print(f" RECOMMENDATION RESULTS FOR {data['user_id']} ({data['mode']}, α={data['alpha']})")
        print("=" * 70)
        
        for idx, item in enumerate(data["recommendations"], start=1):
            print(f"\n{idx}. [{item['category'].upper()}] {item['title']}")
            print(f"   Subcategory: {item['subcategory']} | GNN Score: {item['gnn_relevance_score']}")
            print(f"   Abstract: {item['abstract'][:100]}...")
        print("=" * 70)
        return data
    else:
        print(f"[-] Error {response.status_code}: {response.text}")
        return None

def compare_alpha_diversity(user_id="U0001"):
    """
    Demonstrates the impact of the trade-off parameter alpha:
    Calls the API with alpha=1.0 (Pure Relevance) vs alpha=0.25 (High Diversity).
    """
    print("\n" + "#" * 70)
    print(" DEMO: COMPARING PURE RELEVANCE (α=1.0) VS DIVERSIFIED (α=0.25)")
    print("#" * 70)

    # 1. Pure Relevance
    rec_relevance = get_recommendations_from_api(user_id=user_id, top_k=4, alpha=1.0)

    # 2. High Diversity
    rec_diversity = get_recommendations_from_api(user_id=user_id, top_k=4, alpha=0.25)

if __name__ == "__main__":
    print("=" * 70)
    print(" Python REST API Client for HIN-GNN News Recommender")
    print("=" * 70)

    server_running = check_api_health()
    if not server_running:
        sys.exit(1)

    # Fetch available users
    try:
        u_res = requests.get(f"{API_BASE_URL}/users?limit=5")
        if u_res.status_code == 200:
            sample_users = u_res.json().get("users", ["U0001"])
            test_user = sample_users[0]
        else:
            test_user = "U0001"
    except Exception:
        test_user = "U0001"

    # Make API call for default recommendation
    get_recommendations_from_api(user_id=test_user, top_k=5, alpha=0.75)

    # Demonstrate Alpha trade-off API call
    compare_alpha_diversity(user_id=test_user)

