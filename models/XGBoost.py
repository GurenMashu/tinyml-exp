import os 
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

DATA_DIR = "data/preprocessed_smartposture_data"
CSV_PATH = "comparison/model_comparison.csv"
MODEL_SAVE_PATH = "models/posture_xgboost.json"
os.makedirs("comparison", exist_ok=True)
os.makedirs("models", exist_ok=True)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

feature_cols = [c for c in train_df.columns if c!= "posture_label"]
target_col = "posture_label"

X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
X_val, y_val = val_df[feature_cols].values, val_df[target_col].values
X_test, y_test = test_df[feature_cols].values, test_df[target_col].values

num_classes = len(np.unique(y_train))

model = xgb.XGBClassifier(
    objective = "mutli:softprob",
    num_classes = num_classes,
    max_depth = 4,
    n_estimators = 150,
    learning_rate = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    tree_method = "hist",
    n_jobs = -1,
    random_state = 77,
    eval_metric = "mlogloss"
)

#training
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

model.save_model(MODEL_SAVE_PATH)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

#model info measuring
size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 *1024)
booster = model.get_booster()
num_params = len(booster.trees_to_dataframe())

sample = X_test[:1]
for _ in range(50): 
    model.predict(sample)  # Warmup 
start = time.perf_counter()
for _ in range(1000): model.predict(sample)
end = time.perf_counter()
latency_ms = ((end - start) / 1000) * 1000

#saving model info
row = {
    "model_name": "XGBoost_150_trees",
    "accuracy": round(accuracy, 4),
    "size_mb": round(size_mb, 4),
    "num_parameters": num_params,
    "latency_ms_per_sample": round(latency_ms, 4)
}

df_new = pd.DataFrame([row])
csv_obj = Path(CSV_PATH)
if csv_obj.exists():
    df = pd.concat([pd.read_csv(csv_obj), df_new], ignore_index=True)
else:
    df = df_new

df.to_csv(CSV_PATH, index=False)

