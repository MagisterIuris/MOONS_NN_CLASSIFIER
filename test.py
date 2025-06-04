import torch
import matplotlib.pyplot as plt
from model import MOONSNet
from sklearn.datasets import make_moons

model = MOONSNet()
model.load_state_dict(torch.load("saved_model/moons_model.pth"))
model.eval()

X_test_np, y_test_np = make_moons(n_samples=1000, noise=0.1, random_state=123)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    predictions = model(X_test)
    predictions_bin = (predictions > 0.5).float()

accuracy = (predictions_bin == y_test).float().mean().item()
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

plt.figure(figsize=(6, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions_bin.numpy().flatten(), cmap="coolwarm", s=10)
plt.title("Model Predictions: Moons Dataset")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()