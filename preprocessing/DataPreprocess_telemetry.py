import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

INPUT_CSV = "datasets/iot_telemetry_data.csv"              #https://www.kaggle.com/datasets/garystafford/environmental-sensor-data-132k
OUTPUT_DIR = "data/preprocessed_telemetry_data"
RANDOM_STATE = 77

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
df.dropna(inplace=True)

df = df.drop(columns=["ts","device"])
df["light"] = df["light"].astype(int)
df["motion"] = df["motion"].astype(int)

feature_cols = ["co", "humidity", "lpg", "temp", "smoke", "light", "motion"]

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

#dimensionality reduction
pca = PCA(n_components=2)
dim_reduced_data = pca.fit_transform(df)

#splitting
#train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
#val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE)

#saving
df.to_csv(os.path.join(OUTPUT_DIR, "processed.csv"), index = False)
pd.DataFrame(dim_reduced_data, columns=[f'PC{i+1}' for i in range(dim_reduced_data.shape[1])]).to_csv(
    os.path.join(OUTPUT_DIR, "dim_reduced_to_2.csv"), index=False
)
#test_df.to_csv(os.path.join(OUTPUT_DIR, "test_df"), index = False)
