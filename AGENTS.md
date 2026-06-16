# MiniGPT — AGENTS.md

Minimal GPT implementation from scratch in Portuguese. Trains on a synthetic Portuguese corpus (`data/corpus.py`), not real data.

## Commands

```bash
python main.py train    # Train model (saves to output/)
python main.py chat     # Interactive chat with trained model
python main.py sample   # Batch text generation
python main.py info     # Inspect saved checkpoint
```

Training order matters: `train` must run before `chat`, `sample`, or `info`.

## Dependencies

```bash
pip install -r requirements.txt   # torch>=2.0.0, numpy>=1.24.0
```

No build step, no test suite, no linter configured.

## Architecture

| File | Role |
|------|------|
| `main.py` | CLI entrypoint (argparse subcommands) |
| `config.py` | All hyperparams in `GPTConfig` dataclass |
| `model.py` | GPT from scratch: LayerNorm, MultiHeadSelfAttention, FeedForward, TransformerBlock, GPTModel |
| `train.py` | Training loop, `TextDataset`, LR scheduler (warmup + cosine decay), saves checkpoints |
| `generate.py` | Autoregressive generation with temperature + top-k sampling |
| `tokenizer.py` | `CharTokenizer` — character-level, no BPE, vocab built from training text |
| `data/corpus.py` | Synthetic Portuguese corpus generator/loader |

- **Tokenizer is char-level** (not subword). `vocab_size` is set dynamically at train time from the corpus.
- **Model uses weight tying**: `lm_head.weight` shares weights with `token_embedding`.
- **Pre-norm** Transformer (LayerNorm before attention/FFN, not after).
- **Checkpoint keys**: `modelo`, `config`, `epoch`, `loss`, `total_tokens`, `tempo_total_seg`.

## Key Constraints

- `d_model` must be divisible by `n_heads` (asserted at runtime).
- `context_len` must match between training and inference (model won't generalize to longer sequences).
- Output dir `output/` and `*.pt` files are gitignored.
- All code comments and strings are in Portuguese.
- Auto-detects device: CUDA → MPS → CPU.