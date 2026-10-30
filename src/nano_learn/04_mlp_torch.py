"""
Chapter 04 — Same MLP in PyTorch (CPU or GPU)

Ideas that carry over from NumPy:
  - Forward: linear → ReLU → linear
  - Loss: MSE
  - Update: subtract lr * gradient

What PyTorch adds:
  - autograd records ops and fills .grad via loss.backward()
  - tensors live on a device: cuda or cpu

Run:
  uv run python src/nano_learn/04_mlp_torch.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_spiral(n_per_class: int = 100, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for class_id in (0, 1):
        r = torch.linspace(0.2, 1.0, n_per_class)
        t = torch.linspace(class_id * torch.pi, (class_id + 1.5) * torch.pi, n_per_class)
        t = t + 0.15 * torch.randn(n_per_class, generator=g)
        xs.append(torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=1))
        ys.append(torch.full((n_per_class, 1), float(class_id)))
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], y[perm]


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 32, out_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    device = get_device()
    print(f"torch {torch.__version__}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    X, y = make_spiral()
    X, y = X.to(device), y.to(device)

    model = MLP().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    print("\nTraining PyTorch MLP...")
    for step in range(801):
        pred = model(X)
        loss = loss_fn(pred, y)

        opt.zero_grad()   # clear old .grad
        loss.backward()   # chain rule (autograd)
        opt.step()        # SGD update

        if step % 100 == 0:
            with torch.no_grad():
                acc = ((pred > 0.5) == y).float().mean().item()
            print(f"step {step:4d}  loss={loss.item():.4f}  acc≈{acc:.3f}")

    print("\nSame learning dynamics as NumPy — gradients are automatic now.")
    print("Next: 05_attention.py (the core idea behind transformers).")


if __name__ == "__main__":
    main()
