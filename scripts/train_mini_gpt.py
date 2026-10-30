"""
Train the tiny character-level GPT locally (CPU or CUDA).

Examples:
  uv run python scripts/train_mini_gpt.py
  uv run python scripts/train_mini_gpt.py --max-steps 500 --batch-size 64
  uv run python scripts/train_mini_gpt.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Allow running as a script without installing editable package quirks
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_learn.mini_gpt_lib import (  # noqa: E402
    GPTConfig,
    MiniGPT,
    CharDataset,
    count_parameters,
    estimate_loss,
    get_batch,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train nano-learn MiniGPT")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "input.txt")
    p.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--sample-tokens", type=int, default=200)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args.checkpoints.mkdir(parents=True, exist_ok=True)

    if not args.data.exists():
        raise SystemExit(f"Missing data file: {args.data}")

    raw = args.data.read_text(encoding="utf-8")
    dataset = CharDataset(raw)
    n = len(dataset.data)
    split = int(n * 0.9)
    train_data = dataset.data[:split]
    val_data = dataset.data[split:]

    cfg = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device={device}")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}")
    print(f"vocab={dataset.vocab_size}  chars={n:,}  params={count_parameters(model):,}")
    print(f"config={cfg}")

    train_losses: list[float] = []
    val_losses: list[float] = []
    steps_log: list[int] = []
    best_val = float("inf")
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        model.train()
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            losses = estimate_loss(
                model, train_data, val_data, args.block_size, args.batch_size, device, args.eval_iters
            )
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            steps_log.append(step)
            elapsed = time.time() - t0
            print(
                f"step {step:5d}/{args.max_steps}  "
                f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "stoi": dataset.stoi,
                    "itos": dataset.itos,
                    "step": step,
                    "val_loss": best_val,
                }
                path = args.checkpoints / "best.pt"
                torch.save(ckpt, path)
                print(f"  saved {path} (val={best_val:.4f})")

                # Sample from the improved checkpoint
                model.eval()
                start = torch.zeros((1, 1), dtype=torch.long, device=device)
                # Prefer starting from newline if present
                if "\n" in dataset.stoi:
                    start[0, 0] = dataset.stoi["\n"]
                out = model.generate(start, max_new_tokens=args.sample_tokens, temperature=0.8)
                text = dataset.decode(out[0].tolist())
                print("--- sample ---")
                print(text[:500])
                print("--------------")

    # Loss curve
    fig_path = args.checkpoints / "loss_curve.png"
    plt.figure(figsize=(7, 4))
    plt.plot(steps_log, train_losses, label="train")
    plt.plot(steps_log, val_losses, label="val")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title("MiniGPT training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"wrote {fig_path}")
    print("Done. Load checkpoints/best.pt to resume or inspect weights.")


if __name__ == "__main__":
    main()
