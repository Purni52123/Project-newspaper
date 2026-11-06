import json
from api import load_data, root, get_users, get_recommendations, get_metrics

# Ensure data and embeddings loaded
load_data()

print("=" * 70)
print("1. CALLING GET / (HEALTH CHECK ENDPOINT)")
print("=" * 70)
r0 = root()
print(json.dumps(r0, indent=2))

print("\n" + "=" * 70)
print("2. CALLING GET /users (FETCHING AVAILABLE USERS)")
print("=" * 70)
r_users = get_users(limit=6)
print(json.dumps(r_users, indent=2))

print("\n" + "=" * 70)
print("3. CALLING GET /recommend (PERSONALIZED RECOMMENDATIONS FOR U0001)")
print("=" * 70)
rec_data = get_recommendations(user_id="U0001", top_k=5, alpha=0.75, filter_history=True)
print(f"User ID: {rec_data['user_id']}")
print(f"Re-ranking Mode: {rec_data['mode']} (alpha={rec_data['alpha']})")
print(f"Number of Recommendations: {rec_data['top_k']}")
print("\nRecommended Articles:")
for idx, item in enumerate(rec_data["recommendations"], start=1):
    print(f"\n{idx}. [{item['category'].upper()} - {item['subcategory']}] {item['title']}")
    print(f"   Abstract: {item['abstract']}")
    print(f"   GNN Relevance Score: {item['gnn_relevance_score']}")

print("\n" + "=" * 70)
print("4. CALLING GET /recommend (ALPHA DIVERSITY EXPERIMENT: 1.0 vs 0.25)")
print("=" * 70)
r_pure = get_recommendations(user_id="U0001", top_k=4, alpha=1.0, filter_history=True)
r_div = get_recommendations(user_id="U0001", top_k=4, alpha=0.25, filter_history=True)

print("--- [Pure Relevance Mode: alpha = 1.0] ---")
for i, item in enumerate(r_pure["recommendations"], 1):
    print(f"  {i}. [{item['category']}] {item['title']}")

print("\n--- [Diversity-Focused Mode: alpha = 0.25] ---")
for i, item in enumerate(r_div["recommendations"], 1):
    print(f"  {i}. [{item['category']}] {item['title']}")

print("\n" + "=" * 70)
print("5. CALLING GET /metrics (BENCHMARK RESULTS & METRICS)")
print("=" * 70)
r_metrics = get_metrics()
print(json.dumps(r_metrics, indent=2))
