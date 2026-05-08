import os
import joblib
import pandas as pd

df = pd.read_csv("data/preprocessed_telemetry_data/processed.csv")
sample_data = df.head(1000)

model = joblib.load("saved_telemetry_models/telemetry_kmeans.pkl")
labels = model.predict(sample_data)
clusters = model.cluster_centers_

print(clusters, model.inertia_, len(clusters))
