import onnx
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np

SAVE_PATH = "saved_telemetry_models/telemetry_mlp.onnx"
device = torch.device("cpu")

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

# testing onnx model----------------------------------------------
onnx_model = onnx.load(SAVE_PATH)
onnx.checker.check_model(onnx_model)

sample_input = torch.randn(1, 7)

session = ort.InferenceSession(SAVE_PATH)
inputs = {session.get_inputs()[0].name: sample_input.numpy()}
outputs = session.run(None, inputs)

model = TelemetryMLP().to(device)
model.load_state_dict(torch.load("saved_telemetry_models/telemetry_mlp.pth", map_location="cpu", weights_only=True))
model.eval()
with torch.no_grad():
    torch_out = model(sample_input).numpy()

np.testing.assert_allclose(torch_out, outputs[0], rtol=1e-03, atol=1e-05)
