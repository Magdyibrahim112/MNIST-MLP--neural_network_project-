#Last update eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
# =========================================================
# MNIST MLP FINAL PROJECT (Neural Networks Course)
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ======================
# Data
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size

train_dataset, val_dataset = random_split(
    full_train,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128
)

# ======================
# Model (MLP)
# ======================
class MLP(nn.Module):

    def __init__(self, activation="relu"):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU()

        elif activation == "tanh":
            act = nn.Tanh()

        else:
            act = nn.ReLU()

        self.model = nn.Sequential(

            nn.Linear(28 * 28, 256),
            nn.BatchNorm1d(256),
            act,
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            act,
            nn.Dropout(0.3),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = x.view(x.size(0), -1)

        return self.model(x)

# ======================
# Train Function
# ======================
def train(model, train_loader, val_loader,
          epochs=10, lr=0.001):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    for epoch in range(epochs):

        # ======================
        # TRAIN
        # ======================
        model.train()

        total = 0
        correct = 0
        running_loss = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = criterion(out, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, pred = torch.max(out, 1)

            total += y.size(0)

            correct += (pred == y).sum().item()

        train_loss = running_loss / len(train_loader)

        train_acc = correct / total

        # ======================
        # VALIDATION
        # ======================
        model.eval()

        total = 0
        correct = 0
        val_loss = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                out = model(x)

                loss = criterion(out, y)

                val_loss += loss.item()

                _, pred = torch.max(out, 1)

                total += y.size(0)

                correct += (pred == y).sum().item()

        val_loss /= len(val_loader)

        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss {train_loss:.4f}, "
            f"Train Acc {train_acc:.4f}, "
            f"Val Loss {val_loss:.4f}, "
            f"Val Acc {val_acc:.4f}"
        )

    return (
        train_losses,
        val_losses,
        train_accs,
        val_accs
    )

# ======================
# Experiments
# ======================
experiments = {
    "ReLU": "relu",
    "Tanh": "tanh"
}

results = {}

for name, act in experiments.items():

    print("\n==============================")
    print("Experiment:", name)
    print("==============================")

    model = MLP(act).to(device)

    train_losses, val_losses, train_accs, val_accs = train(
        model,
        train_loader,
        val_loader
    )

    results[name] = {

        "model": model,

        "train_losses": train_losses,
        "val_losses": val_losses,

        "train_accs": train_accs,
        "val_accs": val_accs,

        "best_acc": max(val_accs)
    }

# ======================
# Best Model
# ======================
best_name = max(
    results,
    key=lambda x: results[x]["best_acc"]
)

model = results[best_name]["model"]

print("\nBEST MODEL:", best_name)

# ======================
# Test Evaluation
# ======================
model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)

        out = model(x)

        pred = torch.argmax(out, 1)

        y_true.extend(y.numpy())

        y_pred.extend(pred.cpu().numpy())

acc = np.mean(
    np.array(y_true) == np.array(y_pred)
)

print("\nTest Accuracy:", acc)


# =========================================================
# LOSS CURVES
# =========================================================

# ======================
# Train Loss Curve
# ======================
plt.figure(figsize=(8,5))

for name in results:

    plt.plot(
        results[name]["train_losses"],
        label=f"{name} Train"
    )

plt.title("Train Loss Curve")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

# ======================
# Validation Loss Curve
# ======================
plt.figure(figsize=(8,5))

for name in results:

    plt.plot(
        results[name]["val_losses"],
        label=f"{name} Validation"
    )

plt.title("Validation Loss Curve")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

# =========================================================
# ACCURACY CURVES
# =========================================================

# ======================
# Train Accuracy Curve
# ======================
plt.figure(figsize=(8,5))

for name in results:

    plt.plot(
        results[name]["train_accs"],
        label=f"{name} Train"
    )

plt.title("Train Accuracy Curve")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# ======================
# Validation Accuracy Curve
# ======================
plt.figure(figsize=(8,5))

for name in results:

    plt.plot(
        results[name]["val_accs"],
        label=f"{name} Validation"
    )

plt.title("Validation Accuracy Curve")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()