import os
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model.load_state_dict(torch.load("saved_telemetry_models/telemetry_mlp.pth", map_location=device, weights_only=True))
model.eval()

model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
torch.ao.quantization.prepare(model, inplace=True)

#calibration
input_tensor = torch.randn(1, 7)
model(input_tensor)

quantized_model = torch.ao.quantization.convert(model, inplace=True)

torch.save(quantized_model.state_dict(), "saved_telemetry_models/telemetry_mlp_int8.pth")