import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

INPUT_CSV = "datasets/iot_telemetry_data.csv"              #https://www.kaggle.com/datasets/garystafford/environmental-sensor-data-132k
OUTPUT_DIR = "data/preprocessed_telemetry_data"
RANDOM_STATE = 77

CUSTOM_LABEL_MAPPING = {
    'normal_unoccupied': 0,
    'normal_occupied': 1,
    'warning_uncomfortable': 2,
    'warning_poor_air': 3,
    'critical_gas_leak': 4,
    'critical_fire': 5,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_rule_based_labels(df_raw):
    """Percentile-based labeling - automatically adapts to data distribution"""
    """Use only on the raw dataset!!!"""
    df = df_raw.copy()
    df['label'] = 'normal_unoccupied'
    
    # Define thresholds as percentiles of the actual data
    smoke_95 = df['smoke'].quantile(0.95)  # top 5%
    smoke_90 = df['smoke'].quantile(0.90)  # top 10%
    smoke_75 = df['smoke'].quantile(0.75)  # top 25%
    
    co_95 = df['co'].quantile(0.95)
    co_90 = df['co'].quantile(0.90)
    co_75 = df['co'].quantile(0.75)
    
    lpg_95 = df['lpg'].quantile(0.95)
    lpg_90 = df['lpg'].quantile(0.90)
    lpg_75 = df['lpg'].quantile(0.75)
    
    temp_95 = df['temp'].quantile(0.95)
    temp_05 = df['temp'].quantile(0.05)
    temp_90 = df['temp'].quantile(0.90)
    temp_10 = df['temp'].quantile(0.10)
    
    hum_95 = df['humidity'].quantile(0.95)
    hum_05 = df['humidity'].quantile(0.05)
    
    # CRITICAL: Top 5% of hazardous readings
    fire_mask = (
        (df['smoke'] > smoke_95) | 
        (df['co'] > co_95) | 
        ((df['temp'] > temp_90) & (df['smoke'] > smoke_75))
    )
    df.loc[fire_mask, 'label'] = 'critical_fire'
    
    gas_mask = (
        (df['lpg'] > lpg_95) | 
        ((df['co'] > co_90) & (df['smoke'] <= smoke_75))
    )
    df.loc[gas_mask & (df['label'] == 'normal_unoccupied'), 'label'] = 'critical_gas_leak'
    
    # WARNING: Top 25% of concerning readings
    air_warning = (
        (df['smoke'] > smoke_75) | 
        (df['co'] > co_75) | 
        (df['lpg'] > lpg_75)
    )
    df.loc[air_warning & df['label'].str.startswith('normal'), 'label'] = 'warning_poor_air'
    
    comfort_warning = (
        (df['temp'] < temp_10) | (df['temp'] > temp_90) | 
        (df['humidity'] < hum_05) | (df['humidity'] > hum_95)
    )
    df.loc[comfort_warning & df['label'].str.startswith('normal'), 'label'] = 'warning_uncomfortable'
    
    # OCCUPANCY
    occupied = (df['motion'] == 1) | (df['light'] == 1)
    df.loc[occupied & (df['label'] == 'normal_unoccupied'), 'label'] = 'normal_occupied'
    
    return df

#--------------------main------------------------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df.dropna(inplace=True)

df = apply_rule_based_labels(df)     #comment if the dataset is to be kept unlabled.
df.to_csv(os.path.join("datasets/labeled_iot_telemetry_data.csv"), index = False)  

df = df.drop(columns=["ts","device"])
df["light"] = df["light"].astype(int)
df["motion"] = df["motion"].astype(int)
#convert categorical labels to numeric 
df['label'] = df['label'].map(CUSTOM_LABEL_MAPPING)   

feature_cols = ["co", "humidity", "lpg", "temp", "smoke", "light", "motion"]

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
print("Saved scaler.pkl")

#dimensionality reduction
pca = PCA(n_components=2)
dim_reduced_data = pca.fit_transform(df[feature_cols])

#splitting
#train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
#val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE)

#saving
df.to_csv(os.path.join(OUTPUT_DIR, "labeled_processed.csv"), index = False)    
pd.DataFrame(dim_reduced_data, columns=[f'PC{i+1}' for i in range(dim_reduced_data.shape[1])]).to_csv(
    os.path.join(OUTPUT_DIR, "dim_reduced_to_2.csv"), index=False
)
#test_df.to_csv(os.path.join(OUTPUT_DIR, "test_df"), index = False)
