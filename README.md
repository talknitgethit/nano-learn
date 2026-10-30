# nano-learn

From-first-principles neural nets you can **read, train locally, and extend**.

Path: **tensors → loss → hand-written backprop → PyTorch MLP → attention → tiny character GPT**.

Designed for an RTX-class GPU (tested with CUDA) and CPU fallback. No agents, RAG, or fine-tune APIs — just the math and the training loop.

## Setup

Requires **Python 3.12** from [python.org](https://www.python.org/) / `winget install Python.Python.3.12` (PyTorch does not support every brand-new CPython yet; use the official 3.12 build for the venv).

```powershell
cd ~/Projects/nano-learn
uv venv --python $(py -3.12 -c "import sys; print(sys.executable)")
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## How to learn (run in order)

```powershell
uv run python src/nano_learn/01_tensors.py
uv run python src/nano_learn/02_linear_and_loss.py
uv run python src/nano_learn/03_backprop_numpy.py
uv run python src/nano_learn/04_mlp_torch.py
uv run python src/nano_learn/05_attention.py
uv run python src/nano_learn/06_mini_gpt.py
```

Train the language model (uses `data/input.txt`, writes `checkpoints/`):

```powershell
uv run python scripts/train_mini_gpt.py
# shorter smoke run:
uv run python scripts/train_mini_gpt.py --max-steps 300 --eval-interval 100 --batch-size 32
```

## Concept map

```text
Input x
   │
   ▼
Linear: y = Wx + b          ← chapter 02
   │
   ▼
Nonlinearity (ReLU / GELU)  ← chapter 03–04
   │
   ▼
Loss L (MSE or cross-entropy)
   │
   ▼
∂L/∂W via chain rule        ← chapter 03 (by hand), 04 (autograd)
   │
   ▼
SGD / Adam: W ← W − lr · ∂L/∂W
```

**Chain rule (one line):** if `L` depends on `z` and `z` depends on `w`, then  
`dL/dw = (dL/dz) · (dz/dw)`. Backprop walks the graph **backward**, multiplying local derivatives.

**Loss:** measures “how wrong.” MSE for numbers; cross-entropy for “which token next?”  
**Softmax:** turns scores into probabilities that sum to 1.  
**Attention:** each position looks at others with weights `softmax(QKᵀ / √d)`, then mixes their values — chapter 05.  
**Language model:** same cross-entropy at every character position — chapter 06.

| Stage | File | Math focus |
|-------|------|------------|
| Tensors | `01_tensors.py` | vectors, matmul, broadcast |
| Linear + loss | `02_linear_and_loss.py` | `Wx+b`, MSE, softmax CE |
| Backprop | `03_backprop_numpy.py` | chain rule, ReLU′, SGD |
| PyTorch MLP | `04_mlp_torch.py` | autograd, CPU/GPU device |
| Attention | `05_attention.py` | Q,K,V, causal mask |
| Mini GPT | `06_mini_gpt.py`, `mini_gpt_lib.py` | transformer block, next-char CE |

## Project layout

```text
nano-learn/
  data/input.txt              # training text (Tiny Shakespeare)
  scripts/train_mini_gpt.py   # full train / sample / checkpoint loop
  src/nano_learn/
    01_…06_….py               # chapters (run as scripts)
    mini_gpt_lib.py           # importable model + data helpers
  checkpoints/                # best.pt + loss_curve.png (after training)
```

## Extending (your base)

- Change `n_layer`, `n_head`, `n_embd`, `block_size` in `train_mini_gpt.py` flags.
- Swap `data/input.txt` for any plain-text corpus.
- Add dropout / learning-rate schedules / word-level tokenization later.
- Keep building on `mini_gpt_lib.py` — that is the model you own.

## What this is not

- Not Hugging Face fine-tuning of a giant pretrained model.
- Not tool/agent frameworks.
- Not a from-scratch autograd engine (NumPy backprop on a small MLP is enough to *understand*; then trust PyTorch).
