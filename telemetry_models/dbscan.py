import os
import time
import numpy as np
import pandas as pd
from joblib import dump
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score, 
)

INPUT_DATA = "data/preprocessed_telemetry_data/processed.csv"
SAVE_PATH = "saved_telemetry_models"
CSV_PATH = Path("comparison/telemetry_clustering_comparison.csv")

os.makedirs(SAVE_PATH, exist_ok=True)

data_df = pd.read_csv(INPUT_DATA)
feature_cols = ['co', 'humidity', 'lpg', 'temp']

#model
print("Training ...")
model = DBSCAN(eps = 0.5, min_samples=5)
labels = model.fit_predict(data_df[feature_cols])

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

#clustering quality metrics
silhouette = silhouette_score(data_df, model.labels_)
db_index = davies_bouldin_score(data_df, model.labels_)
ch_index = calinski_harabasz_score(data_df, model.labels_)

#Finding Inference Latency
# Use a realistic batch size (single-sample latency is too noisy for Python/sklearn)
batch_size = min(500, len(data_df))
X_batch = data_df.iloc[:batch_size].to_numpy()

_ = model.predict(X_batch)  # Warm-up (CPU cache, Python overhead)

n_runs = 50
latencies = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    model.predict(X_batch)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)  # Convert to milliseconds

avg_latency = np.mean(latencies)
p95_latency = np.percentile(latencies, 95)

#saving model
dump(model, "saved_telemetry_models/telemetry_dbscan.pkl")

#Finding model save size
model_path = "saved_telemetry_models/telemetry_dbscan.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
size_mb = os.path.getsize(model_path) / (1024 * 1024)

#Save metrics to csv-----------------------------------------------------------------------
row = {
    "Model_name": f"dbscan_{n_clusters}_clusters",
    "Noise": n_noise,
    "Silhouette_score": silhouette,             # closer to 1 better
    "Davies_Bouldin Index": db_index,           # lower better
    "Calinski_Harabasz Index": ch_index,        # more better
    "Inertia": model.inertia_,                  # lower better   
    "Size": size_mb,
    "batch_size_latency": batch_size,
    "avg_inference_latency_ms": round(avg_latency, 3),
    "p95_inference_latency_ms": round(p95_latency, 3)    # SLA standard
}
df_new = pd.DataFrame([row])

if CSV_PATH.exists():
    df = pd.concat([pd.read_csv(CSV_PATH), df_new], ignore_index=True)
else:
    df = df_new
df.to_csv(CSV_PATH, index=False)