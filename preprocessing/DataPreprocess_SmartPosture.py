import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

INPUT_CSV = "datasets/smartposture_dataset_100k.csv"       #https://www.kaggle.com/datasets/vigneshj22bec1067/smartposture
OUTPUT_DIR = "data/preprocessed_smartposture_data"
RANDOM_STATE = 77

os.makedirs(OUTPUT_DIR,exist_ok=True)

df = pd.read_csv(INPUT_CSV)
#dropping empty cols
df = df.dropna().reset_index(drop=True)

feature_cols = [
    "sensor_backrest_upper", "sensor_backrest_lower",
    "sensor_seat_left", "sensor_seat_right",
    "sensor_seat_front", "sensor_seat_rear"
    ]
target_cols = "posture_label"
group_cols = "person_id"

#Splitting
splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
train_idx, temp_idx = next(splitter1.split(df, groups = df[group_cols]))

splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
val_idx, test_idx = next(splitter2.split(df.iloc[temp_idx], groups = df.iloc[temp_idx][group_cols]))

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[temp_idx].iloc[val_idx].copy()
test_df = df.iloc[temp_idx].iloc[test_idx].copy()

print(f"Split sizes-> Training data: {len(train_df)} | Validation data: {len(val_df)} | Test data: {len(test_df)}")

#scaling
scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

train_df[feature_cols] = scaler.transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

#removing person_id column and saving to disk
for split_df, split_name in zip([train_df, val_df, test_df], ["train", "val", "test"]):
    split_df = split_df.drop(columns=[group_cols])
    split_df.to_csv(os.path.join(OUTPUT_DIR, f"{split_name}.csv"), index=False)
    print(f"Saved {split_name}.csv")
      
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
print("Saved scaler.pkl")

