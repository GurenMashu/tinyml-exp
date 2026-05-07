import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/preprocessed_smartposture_data"
FEATURE_COLS = [
    "sensor_backrest_upper", "sensor_backrest_lower",
    "sensor_seat_left", "sensor_seat_right",
    "sensor_seat_front", "sensor_seat_rear"
] 
TARGET_COL = "posture_label"

train_df_temp = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
NUM_CLASSES = train_df_temp[TARGET_COL].nunique()

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE =  7    #for early stopping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/posture_1D_CNN.pth"

os.makedirs("models", exist_ok=True)

#dataset
class PostureSensorDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.x = self.df[FEATURE_COLS].values.astype(np.float32)
        self.y = self.df[TARGET_COL].values.astype(np.int64)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
    
#model
class PostureSensor1D_CNN(nn.Module):
    def __init__(self, input_dim = 6, num_classes = 5):
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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.net(x)

#helpers
def log_model_metrics(model, test_loader, model_path, model_name, device, csv_path="comparison/model_comparison.csv"):
    model.eval()
    
    #Accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    accuracy = correct / total

    #Model Size (MB)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    #Number of Parameters
    num_params = sum(p.numel() for p in model.parameters())

    #Inference Latency (ms per sample)
    sample_X, _ = next(iter(test_loader))
    dummy = sample_X[:1].to(device)  # Single sample = real-time edge latency
    
    for _ in range(10): 
        model(dummy)  # Warmup
    if device.type == "cuda": 
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100): 
        model(dummy)
    if device.type == "cuda": 
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    latency_ms = ((end - start) / 100) * 1000

    #Save to CSV
    row = {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "size_mb": round(size_mb, 4),
        "num_parameters": num_params,
        "latency_ms_per_sample": round(latency_ms, 4)
    }
    df_new = pd.DataFrame([row])
    csv_path = Path(csv_path)
    
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
    
    print(f"\nMetrics logged → {csv_path}")
    print(pd.DataFrame([row]).to_string(index=False))
    return row

def compute_class_weights(csv_path):
    df = pd.read_csv(csv_path)
    counts = df[TARGET_COL].value_counts().sort_index().values
    total = len(df)
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

#training
def main():
    torch.manual_seed(77)
    np.random.seed(77)

    train_loader = DataLoader(PostureSensorDataset(os.path.join(DATA_DIR, "train.csv")), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(PostureSensorDataset(os.path.join(DATA_DIR, "val.csv")), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(PostureSensorDataset(os.path.join(DATA_DIR, "test.csv")), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    class_weights = compute_class_weights(os.path.join(DATA_DIR, "train.csv"))
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = PostureSensor1D_CNN(input_dim=6, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y).sum().item()
            train_total += y.size(0)
        
        train_loss /= train_total
        train_acc = train_correct/ train_total

        #validation
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train AC: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        #early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, SAVE_PATH)
            print(f"Latest best model saved with val loss: {val_loss:.4f}")
        
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch: {epoch+1}")
                break
    

    #final testing
    print("\nEvaluating on test set")
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print(f"Best model saved")

    test_loader_final = DataLoader(PostureSensorDataset(os.path.join(DATA_DIR, "test.csv")), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    log_model_metrics(
        model=model,
        test_loader=test_loader_final,
        model_path=SAVE_PATH,
        model_name="1D_CNN",  # Change per experiment
        device=DEVICE
    )
    

if __name__ == "__main__":
    main()
