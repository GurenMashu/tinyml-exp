import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("data/preprocessed_telemetry_data/labeled_processed.csv")

class TelemetryDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
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
    

#---------------TRAINING-------SETUP---------------------------------
feature_cols = ["co", "humidity", "lpg", "temp", "smoke", "light", "motion"]
X = df[feature_cols].values
y = df["label"].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=77, stratify=y)

#----------computing-class-weights----------------------------------------------
num_classes = 6
classes_present = np.unique(y_train)
weights_array = np.zeros(num_classes)  

weights_present = compute_class_weight(
    class_weight="balanced", 
    classes=classes_present, 
    y=y_train
)

# Map weights back to full 6-class array
for cls, w in zip(classes_present, weights_present):
    weights_array[int(cls)] = w

# Handle any missing classes: assign weight=1.0 (neutral)
weights_array[weights_array == 0] = 1.0

class_weights = torch.tensor(weights_array, dtype=torch.float32)
#-------------------------------------------------------------------------------

train_dataset = TelemetryDataset(X_train, y_train)
val_dataset = TelemetryDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

model = TelemetryMLP()
model = model.to(device)
lr = 1e-3
epochs = 100

#------------TRAINING-------------------------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

print("\n Starting Training...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    #validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    scheduler.step(avg_val_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, "saved_telemetry_models/telemetry_mlp.pth")
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print(f"Early stopping at epoch: {epoch+1}")
            break

print(f"\nTraining complete with best validation loss of: {best_val_loss:.4f}")