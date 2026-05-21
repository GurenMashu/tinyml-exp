import os
import torch
import torch.onnx
import torch.nn as nn

SAVE_PATH = "saved_telemetry_models/telemetry_mlp.onnx"
device = torch.device("cpu")               # stick with cpu for conversion to prevent unwanted tracing mechanisms in gpu assisted conversion

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
model.to(device)
model.eval()

sample_input = torch.randn(1, 7).to(device)

torch.onnx.export(
    model,
    sample_input,     # we pass a sample input to activate the model's computation graph | unlike for tensorflow where such metadata is already saved
    SAVE_PATH,
    input_names=["input"],
    output_names=["output"]
)