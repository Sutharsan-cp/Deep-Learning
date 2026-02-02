import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# -----------------------------
# Load dataset (example: Skin_NonSkin.txt)
# -----------------------------
data = np.loadtxt("Skin_NonSkin.txt")   # columns: B, G, R, label
X = data[:, :3]   # features
y = data[:, 3]    # labels (1=skin, 2=nonskin)

# Convert labels to 0/1
y = (y == 1).astype(int)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# -----------------------------
# Define Neural Network
# -----------------------------
class SkinNN(nn.Module):
    def __init__(self):
        super(SkinNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)   # input 3 → hidden 16
        self.fc2 = nn.Linear(16, 8)   # hidden 16 → hidden 8
        self.fc3 = nn.Linear(8, 1)    # output 1
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = SkinNN()

# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.BCELoss()  # binary cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# -----------------------------
# Training Loop
# -----------------------------
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# -----------------------------
# Evaluation
# -----------------------------
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = (y_pred > 0.5).float()
    acc = (y_pred_cls.eq(y_test).sum() / y_test.shape[0]).item()
    print(f"Test Accuracy: {acc:.4f}")
