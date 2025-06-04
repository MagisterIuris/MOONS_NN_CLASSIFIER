import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from model import MOONSNet

X_train_np, y_train_np = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)

model = MOONSNet()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.4)

loss_history = []
for epoch in range(100000):
    output = model(X_train)
    loss = criterion(output, y_train)
    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (BCE)")
plt.grid(True)
plt.show()

with torch.no_grad():
    predictions = model(X_train)
    for i in range(4):
        print(f"Input: {X_train[i].tolist()} → Predicted: {predictions[i].item():.4f} (Target: {y_train[i].item()})")

torch.save(model.state_dict(), "saved_model/moons_model.pth")