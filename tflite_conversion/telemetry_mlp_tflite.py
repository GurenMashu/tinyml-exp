import os
import torch
import torch.nn as nn
import litert_torch

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

            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

model = TelemetryMLP()
model.load_state_dict(torch.load("saved_telemetry_models/telemetry_mlp.pth", map_location="cpu", weights_only=True))
model.eval()

sample_input = (torch.randn(1, 7),)

edge_model = litert_torch.convert(model, sample_input)

os.makedirs(os.path.dirname("saved_telemetry_models/telemetry_mlp.tflite"), exist_ok=True)
edge_model.export("saved_telemetry_models/telemetry_mlp.tflite")