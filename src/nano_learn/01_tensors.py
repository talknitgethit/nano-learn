"""
Chapter 01 — Tensors: vectors, matrices, and broadcasting

Math you need:
  - A vector is a list of numbers. A matrix is a grid of numbers.
  - Dot product: sum of element-wise products. Measures "alignment".
  - Matrix multiply (matmul): each output entry is a dot product of a row and a column.
  - Broadcasting: NumPy stretches length-1 axes so shapes can combine without copies.

Run:
  uv run python src/nano_learn/01_tensors.py
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    print("=== Vectors ===")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    # Dot product: a·b = 1*4 + 2*5 + 3*6 = 32
    print(f"a · b = {np.dot(a, b)}")

    print("\n=== Matrices & matmul ===")
    # W is 2×3, x is 3×1 → Wx is 2×1
    W = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    x = np.array([[1.0], [2.0], [3.0]])
    y = W @ x  # same as np.matmul(W, x)
    print(f"W shape {W.shape}, x shape {x.shape}, Wx =\n{y}")

    print("\n=== Batch of inputs (neural-net style) ===")
    # In nets we often store a batch as (batch, features).
    # X: 4 examples, each with 3 features. W_out: maps 3 → 2.
    X = np.random.randn(4, 3)
    W_out = np.random.randn(3, 2)
    # X @ W_out → (4, 2): one 2-d output per example
    Y = X @ W_out
    print(f"X {X.shape} @ W {W_out.shape} → Y {Y.shape}")

    print("\n=== Broadcasting ===")
    # Add a bias vector of length 2 to every row of Y
    bias = np.array([0.5, -0.5])
    Y_biased = Y + bias  # bias is treated as (1, 2) and stretched over batch
    print(f"Y + bias → {Y_biased.shape}")
    print(Y_biased)

    print("\n=== Reshape / view ===")
    flat = np.arange(12)
    grid = flat.reshape(3, 4)
    print(f"flat {flat.shape} → grid {grid.shape}\n{grid}")

    print("\nDone. Next: 02_linear_and_loss.py (y = Wx + b and measuring error).")


if __name__ == "__main__":
    main()
