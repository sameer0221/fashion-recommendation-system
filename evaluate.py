import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load embeddings, metadata, and mapping
# -----------------------------
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
meta_vectors = np.load('metadata_features.npy')
meta_index_map = pickle.load(open('meta_index_map.pkl','rb'))
meta_map = pickle.load(open('meta_map.pkl','rb'))

# Convert meta_map to DataFrame
meta_df = pd.DataFrame(list(meta_map.values()))

# -----------------------------
# FIX: Make metadata match embeddings
# -----------------------------
valid_ids = [meta_df.iloc[i]['id'] for i in range(len(feature_list))]
filtered_meta_indices = [meta_index_map[idx] for idx in valid_ids]

meta_vectors = meta_vectors[filtered_meta_indices]
meta_df = meta_df.iloc[:len(feature_list)].reset_index(drop=True)

# -----------------------------
# Evaluation function
# -----------------------------
def precision_at_k(query_idx, recs, meta_df, k=5):
    q_cat = meta_df.iloc[query_idx]['subCategory']
    hits = sum(meta_df.iloc[r]['subCategory'] == q_cat for r in recs[:k])
    return hits / k

# -----------------------------
# Evaluate image-only similarity
# -----------------------------
num_queries = min(10, len(feature_list))

print("===== Image-Only Similarity Evaluation =====")
for i in range(num_queries):
    query_feat = feature_list[i].reshape(1, -1)
    sims = cosine_similarity(query_feat, feature_list)[0]
    top_indices = sims.argsort()[::-1][1:6]

    prec = precision_at_k(i, top_indices, meta_df)
    print(f"Query {i}: Precision@5 = {prec:.2f}")

# -----------------------------
# Evaluate hybrid similarity
# -----------------------------
alpha = 0.7
beta = 0.3

print("\n===== Hybrid Similarity Evaluation =====")
for i in range(num_queries):
    query_feat = feature_list[i].reshape(1, -1)
    img_sim = cosine_similarity(query_feat, feature_list)[0]

    # metadata similarity for the same aligned index
    query_meta = meta_vectors[i].reshape(1, -1)
    meta_sim = cosine_similarity(query_meta, meta_vectors)[0]

    combined_sim = alpha * img_sim + beta * meta_sim
    top_indices = combined_sim.argsort()[::-1][1:6]

    prec = precision_at_k(i, top_indices, meta_df)
    print(f"Query {i}: Hybrid Precision@5 = {prec:.2f}")