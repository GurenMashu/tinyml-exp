import os
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler

DATA_PATH = "datasets/predictive_maintenance_dataset.csv"    # https://www.kaggle.com/datasets/ziya07/iot-integrated-predictive-maintenance-dataset
SAVE_PATH = "data/preprocessed_maintenance_data"

os.makedirs(SAVE_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)
sequence_length = 10

df["timestamp"] = pd.to_datetime(df["timestamp"])
df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

feature_cols = ["vibration", "acoustic", "temperature", "current", "IMF_1", "IMF_2", "IMF_3"]

#normalizing/scaling
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

df.drop(columns=["machine_id", "timestamp"], inplace=True)
df.to_csv(os.path.join(SAVE_PATH, "processed_data.csv"), index=False)
