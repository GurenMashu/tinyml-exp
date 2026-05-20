import os
import joblib
import numpy as np
import torch
import torch.nn as nn


class TelemetryMLP(nn.Module):
    def __init__(self, input_dim=7, num_classes=6, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_model(path):
    model = TelemetryMLP()
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def load_scaler(path=None):
    # prefer a joblib scaler
    if path and os.path.exists(path):
        return joblib.load(path)


def predict(model, scaler, sample):
    # sample: array-like shape (1,7)
    if scaler is not None:
        x = scaler.transform(sample)
    else:
        x = np.array(sample, dtype=np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(x).float()
        out = model(tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy().flatten()
        pred = int(probs.argmax())
        return probs, pred
