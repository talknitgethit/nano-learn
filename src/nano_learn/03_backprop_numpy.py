"""
Chapter 03 — Manual backprop on a 2-layer MLP (NumPy only)

Math you need (chain rule on a graph):
  If L depends on z, and z depends on w, then dL/dw = (dL/dz) * (dz/dw).
  We walk the graph backward: each local derivative multiplies the upstream gradient.

Network:
  h = ReLU(X @ W1 + b1)
  y = h @ W2 + b2
  L = mean((y − target)²)

Local pieces (shapes in comments use batch B, hidden H, out O, in D):
  dL/dy = 2(y − target) / B
  dL/dW2 = h.T @ dL/dy          (matmul backward)
  dL/db2 = sum(dL/dy, axis=0)
  dL/dh  = dL/dy @ W2.T
  dL/d(pre-relu) = dL/dh * (pre > 0)   # ReLU'
  dL/dW1 = X.T @ dL/d(pre)
  dL/db1 = sum(dL/d(pre), axis=0)

SGD: parameter ← parameter − lr * gradient
  (gradient points uphill; we go downhill to reduce loss)

Run:
  uv run python src/nano_learn/03_backprop_numpy.py
"""

from __future__ import annotations

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def make_spiral(n_per_class: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """2D spiral toy data → binary labels as 0/1 (we regress a single score)."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for class_id in (0, 1):
        r = np.linspace(0.2, 1.0, n_per_class)
        t = np.linspace(class_id * np.pi, (class_id + 1.5) * np.pi, n_per_class) + rng.normal(
            0, 0.15, n_per_class
        )
        X_list.append(np.stack([r * np.cos(t), r * np.sin(t)], axis=1))
        y_list.append(np.full(n_per_class, float(class_id)))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).reshape(-1, 1)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def main() -> None:
    rng = np.random.default_rng(42)
    X, y = make_spiral()
    B, D = X.shape
    H, O = 32, 1

    # Small random init (too large → exploding activations)
    W1 = rng.normal(scale=0.5, size=(D, H))
    b1 = np.zeros(H)
    W2 = rng.normal(scale=0.5, size=(H, O))
    b2 = np.zeros(O)
    lr = 0.05
    steps = 800

    print("Training 2-layer MLP with hand-written gradients (NumPy)...")
    for step in range(steps + 1):
        # ---- forward ----
        pre = X @ W1 + b1          # (B, H)
        h = relu(pre)              # (B, H)
        pred = h @ W2 + b2         # (B, O)
        loss = float(np.mean((pred - y) ** 2))

        if step % 100 == 0:
            acc = float(np.mean((pred > 0.5) == y))
            print(f"step {step:4d}  loss={loss:.4f}  acc≈{acc:.3f}")

        # ---- backward (chain rule) ----
        # dL/dpred: derivative of mean((pred-y)^2) w.r.t. pred
        dpred = (2.0 / B) * (pred - y)          # (B, O)

        # pred = h @ W2 + b2
        dW2 = h.T @ dpred                       # (H, O)
        db2 = dpred.sum(axis=0)                 # (O,)
        dh = dpred @ W2.T                       # (B, H)

        # h = relu(pre) → pass gradient only where pre > 0
        dpre = dh * (pre > 0.0)                 # (B, H)

        # pre = X @ W1 + b1
        dW1 = X.T @ dpre                        # (D, H)
        db1 = dpre.sum(axis=0)                  # (H,)

        # ---- SGD update ----
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    print("\nLoss should drop and accuracy rise — you just trained with the chain rule.")
    print("Next: 04_mlp_torch.py (same idea; PyTorch computes gradients for you).")


if __name__ == "__main__":
    main()
