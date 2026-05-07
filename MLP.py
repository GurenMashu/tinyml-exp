import os
import numpy as np
import pandas as pd
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
SAVE_PATH = "models/posture_mlp.pth"

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
class PostureSensorMLP(nn.Module):
    def __init__(self, input_dim = 6, num_classes = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

#helpers
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

    model = PostureSensorMLP(input_dim=6, num_classes=NUM_CLASSES).to(DEVICE)
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
    

if __name__ == "__main__":
    main()
