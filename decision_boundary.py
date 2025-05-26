import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from model import MOONSNet

model = MOONSNet()
model.load_state_dict(torch.load("saved_model/moons_model.pth"))
model.eval()

X_test_np, y_test_np = make_moons(n_samples=1000, noise=0.1, random_state=123)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

xx, yy = torch.meshgrid(
    torch.linspace(-2, 3, 300),
    torch.linspace(-1.5, 2, 300),
    indexing="ij"
)
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

with torch.no_grad():
    probs = model(grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, probs, levels=100, cmap="coolwarm", alpha=0.8)
plt.colorbar(label="Class 1 Probability")

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.numpy().flatten(), cmap="coolwarm", s=15, edgecolors='k')
plt.title("Moons Dataset - Decision Boundary Heatmap")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()
