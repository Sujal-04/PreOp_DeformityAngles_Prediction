import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import KneeXrayDataset
from model import AnglePredictor

# ✅ Paths (Update these paths for your dataset)
IMAGES_DIR = "/content/drive/MyDrive/OAI_Xrays/"
CSV_FILE = "/content/drive/MyDrive/OAI_Xrays/labels.csv"

# ✅ Load Dataset
dataset = KneeXrayDataset(IMAGES_DIR, CSV_FILE)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ✅ Initialize Model, Loss, Optimizer
model = AnglePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training Loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, angles in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), angles)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# ✅ Save Model
torch.save(model.state_dict(), "knee_angle_model.pth")
print("✅ Training Complete! Model saved as knee_angle_model.pth")

