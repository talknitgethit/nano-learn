"""
Chapter 06 — Tiny character-level GPT

Stack of:
  token embedding + position embedding
  → N × (causal self-attention + MLP) with residual connections & LayerNorm
  → linear head over vocabulary

Training objective: next-character prediction (cross-entropy).
Same CE idea as chapter 02; now over every position in the sequence.

Implementation lives in mini_gpt_lib.py so the train script can import it cleanly.

Run a tiny smoke demo:
  uv run python src/nano_learn/06_mini_gpt.py

Full training:
  uv run python scripts/train_mini_gpt.py
"""

from __future__ import annotations

import torch

from nano_learn.mini_gpt_lib import GPTConfig, MiniGPT, CharDataset, count_parameters


def main() -> None:
    """Smoke demo: random-init model over a tiny alphabet."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = "hello nano-learn; train a tiny gpt on characters.\n" * 20
    dataset = CharDataset(text)

    cfg = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model = MiniGPT(cfg).to(device)
    print(f"device={device}  params={count_parameters(model):,}")

    block = cfg.block_size
    x = dataset.data[:block].unsqueeze(0).to(device)
    y = dataset.data[1 : block + 1].unsqueeze(0).to(device)
    logits, loss = model(x, y)
    print(f"demo batch loss={loss.item():.4f}  logits={tuple(logits.shape)}")

    start = torch.tensor([[dataset.stoi["h"]]], dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=40, temperature=1.0)
    sample = dataset.decode(out[0].tolist())
    print(f"untrained sample: {sample!r}")
    print("\nFor real training: uv run python scripts/train_mini_gpt.py")


if __name__ == "__main__":
    main()
