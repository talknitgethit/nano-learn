"""
Chapter 02 — Linear layer + loss

Math you need:
  - Linear map: y = Wx + b  (affine: weights + bias)
  - MSE (mean squared error): average of (prediction − target)²
    Good for regression. Smooth; large mistakes are penalized more.
  - Cross-entropy intuition (classification):
    Softmax turns scores into probabilities that sum to 1.
    Cross-entropy = −log(p_correct). If the model is sure and wrong, loss is huge.

Run:
  uv run python src/nano_learn/02_linear_and_loss.py
"""

from __future__ import annotations

import numpy as np


def linear(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """X: (batch, in_features), W: (in, out), b: (out,) → (batch, out)."""
    return X @ W + b


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean of squared errors. Scalar."""
    return float(np.mean((pred - target) ** 2))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax along last axis."""
    # Subtract max so exp() does not overflow; same probabilities afterward.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    targets: integer class ids, shape (batch,).
    CE = mean over batch of −log(p[target]).
    """
    batch = probs.shape[0]
    # Gather probability of the correct class for each example
    p_correct = probs[np.arange(batch), targets]
    # Tiny epsilon avoids log(0)
    return float(-np.mean(np.log(p_correct + 1e-12)))


def main() -> None:
    rng = np.random.default_rng(0)

    print("=== Linear layer on toy regression ===")
    # True function: y = 2x0 − x1 + 0.5  (noise-free for clarity)
    X = rng.normal(size=(8, 2))
    true_W = np.array([[2.0], [-1.0]])
    true_b = np.array([0.5])
    y_true = linear(X, true_W, true_b)

    # Random untrained weights → high MSE
    W = rng.normal(size=(2, 1)) * 0.1
    b = np.zeros(1)
    y_pred = linear(X, W, b)
    print(f"Untrained MSE: {mse_loss(y_pred, y_true):.4f}")

    # Perfect weights → MSE ≈ 0
    y_perfect = linear(X, true_W, true_b)
    print(f"Perfect MSE:   {mse_loss(y_perfect, y_true):.6f}")

    print("\n=== Softmax + cross-entropy (classification intuition) ===")
    # 3 examples, 4 classes. Row 0 is confidently correct (class 1).
    logits = np.array(
        [
            [0.1, 5.0, 0.2, 0.0],  # almost sure class 1
            [2.0, 2.1, 2.0, 2.0],  # unsure
            [5.0, 0.0, 0.0, 0.0],  # sure but WRONG if target is 1
        ]
    )
    targets = np.array([1, 1, 1])
    probs = softmax(logits)
    print("Probabilities:\n", np.round(probs, 3))
    print(f"Cross-entropy: {cross_entropy(probs, targets):.4f}")
    print("(Low when confident & correct; high when confident & wrong.)")

    print("\nDone. Next: 03_backprop_numpy.py (train by hand with gradients).")


if __name__ == "__main__":
    main()
