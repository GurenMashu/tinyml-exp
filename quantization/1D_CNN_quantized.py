import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/preprocessed_smartposture_data"
FEATURE_COLS = [
    "sensor_backrest_upper", "sensor_backrest_lower",
    "sensor_seat_left", "sensor_seat_right",
    "sensor_seat_front", "sensor_seat_rear"
]
TARGET_COL = "posture_label"
BATCH_SIZE = 256
MODEL_PATH = "saved_models/posture_1D_CNN.pth" 
CSV_PATH = "comparison/model_comparison.csv"
os.makedirs("comparison", exist_ok=True)

class PostureSensorDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.x = self.df[FEATURE_COLS].values.astype(np.float32)
        self.y = self.df[TARGET_COL].values.astype(np.int64)
    def __len__(self): 
        return len(self.df)
    def __getitem__(self, idx): 
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
    
class PostureSensor1D_CNN(nn.Module):
    def __init__(self, input_features=6, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_features=8, out_features=num_classes)
        )
    
    def forward(self, x):
        # Input: [batch, 6] → [batch, 1, 6] for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.net(x)

#helpers------------------------------------------------------------------------------------------------------
def load_base_model(model_path, num_classes, device):
    model = PostureSensor1D_CNN(input_features=6, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def log_model_metrics(model, test_loader, model_path, model_name, device, csv_path=CSV_PATH):
    model.eval()
    correct, total = 0, 0
    
    # Check if model is FP16 to auto-cast inputs
    is_fp16 = any(p.dtype == torch.float16 for p in model.parameters())
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            if is_fp16: 
                X = X.half()
            y = y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    accuracy = correct / total

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    num_params = sum(p.numel() for p in model.parameters())

    sample_X, _ = next(iter(test_loader))
    dummy = sample_X[:1].to(device)
    if is_fp16: 
        dummy = dummy.half()
    for _ in range(10): 
        model(dummy)
    if device.type == "cuda": 
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100): 
        model(dummy)
    if device.type == "cuda": 
        torch.cuda.synchronize()
    end = time.perf_counter()
    latency_ms = ((end - start) / 100) * 1000

    row = {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "size_mb": round(size_mb, 4),
        "num_parameters": num_params,
        "latency_ms_per_sample": round(latency_ms, 4)
    }
    df_new = pd.DataFrame([row])
    csv_path_obj = Path(csv_path)
    if csv_path_obj.exists():
        df = pd.concat([pd.read_csv(csv_path_obj), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
    print(f"Logged {model_name} → {csv_path}")
    return row
#-------------------------------------------------------------------------------------------

def main():
    # Quantization in PyTorch is CPU-targeted
    device = torch.device("cpu")

    test_loader = DataLoader(PostureSensorDataset(os.path.join(DATA_DIR, "test.csv")), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    num_classes = train_df[TARGET_COL].nunique()

    #FP16 (Half Precision)
    model_fp16 = load_base_model(MODEL_PATH, num_classes, device).half()
    save_path = MODEL_PATH.replace(".pth", "_fp16.pth")
    torch.save(model_fp16.state_dict(), save_path)
    log_model_metrics(model_fp16, test_loader, save_path, "CNN_FP16", device)

    #INT8 Dynamic Quantization
    #Quantizing BOTH Conv1d AND Linear layers for CNNs
    model_base = load_base_model(MODEL_PATH, num_classes, "cpu")
    model_dyn = torch.ao.quantization.quantize_dynamic(
        model_base, 
        {nn.Conv1d, nn.Linear},  
        dtype=torch.qint8
    )
    save_path = MODEL_PATH.replace(".pth", "_int8_dynamic.pth")
    torch.save(model_dyn.state_dict(), save_path)
    log_model_metrics(model_dyn, test_loader, save_path, "CNN_INT8_DYNAMIC", device)

    print("\nAll quantizations complete. Compare results in:", CSV_PATH)


if __name__ == "__main__":
    main()