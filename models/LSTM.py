import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset

DATA_PATH = "data/preprocessed_maintenance_data/processed_data.csv"
MODEL_SAVE_PATH = "saved_models"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

class MaintenanceData(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.FloatTensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()
    
df = pd.read_csv(DATA_PATH)
feature_cols = ["vibration", "acoustic", "temperature", "current", "IMF_1", "IMF_2", "IMF_3"]
sequence_length = 10

X, y = df[feature_cols].values, df["label"].values

#creating sequences from the data
sequences, seq_labels = [], []
for i in range(len(X) - sequence_length):
    seq = X[i: i + sequence_length]
    label = y[i + sequence_length - 1]
    sequences.append(seq)
    seq_labels.append(label)

print(f"Total sequences: {len(sequences)}")
print(f"Label distribution: {Counter(seq_labels)}")

split_idx = int(len(sequences) * 0.8)

X_train, X_test = sequences[:split_idx], sequences[split_idx:]
y_train, y_test = seq_labels[:split_idx], seq_labels[split_idx:]
    
train_dataset = MaintenanceData(X_train, y_train)
test_dataset = MaintenanceData(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

