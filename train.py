"""
train.py — Loop de treinamento do MiniGPT

EXPANDIDO com:
- Gradient accumulation: simula batch maior sem estourar VRAM
- Split de validação: avalia o modelo em dados não vistos
- Early stopping: para o treino se não melhorar por N épocas
- Tokenizador BPE (ou char, configurável)
"""

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import random_split

from config import GPTConfig
from data.corpus import augmentar_texto, permutar_frases
from model import GPTModel
from tokenizer import CharTokenizer, BPETokenizer, criar_tokenizer, carregar_tokenizer


class TextDataset(Dataset):
    """Dataset que transforma texto tokenizado em pares (input, target)."""

    def __init__(self, tokens: list[int], context_len: int):
        self.tokens = tokens
        self.context_len = context_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.context_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.context_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_lr(it: int, config: GPTConfig) -> float:
    """
    Scheduler de learning rate: Warmup + Cosine Decay.

    lr │    ___....───────┐
       │   /              ╲
       │  /                ╲
       │ /                  ╲___....──── min_lr
       └───────────────────────────────→ step
           warmup   cosine decay
    """
    if it < config.warmup_steps:
        return config.learning_rate * (it + 1) / config.warmup_steps

    progress = (it - config.warmup_steps) / max(1, 10000 - config.warmup_steps)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def avaliar(
    modelo: GPTModel, dataloader: DataLoader, device: str
) -> float:
    """Avalia o modelo no dataset de validação. Retorna a loss média."""
    modelo.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = modelo(x, y)
        if loss is not None:
            total_loss += loss.item()
            n_batches += 1
    modelo.train()
    return total_loss / max(n_batches, 1)


def treinar(
    config: GPTConfig,
    texto: str,
    saida_dir: str = "output",
    device: str | None = None,
) -> tuple[GPTModel, CharTokenizer | BPETokenizer]:
    """
    Loop de treinamento completo do MiniGPT.

    NOVIDADES:
    - Gradient accumulation (batch efetivo = batch_size * gradient_accum_steps)
    - Split de validação (val_split % dos dados)
    - Early stopping (pára se não melhorar por patience épocas)
    - Suporte a BPE ou char tokenizer
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Dispositivo: {device}")

    saida_path = Path(saida_dir)
    saida_path.mkdir(exist_ok=True)

    # ── Tokenizer ──
    print("Construindo tokenizer...")
    tokenizer = criar_tokenizer(config.tokenizer_type)
    if isinstance(tokenizer, BPETokenizer):
        tokenizer.treinar(texto, vocab_size=config.bpe_vocab_size)
    else:
        tokenizer.treinar(texto)
    print(f"  {tokenizer}")
    config.vocab_size = tokenizer.vocab_size

    # ── Dataset ──
    tokens = tokenizer.codificar(texto)
    print(f"  Corpus: {len(texto):,} caracteres → {len(tokens):,} tokens")
    print(f"  Tokens únicos: {tokenizer.vocab_size}")

    # ── Data augmentation ──
    # Cria variações do corpus para reduzir overfitting
    import random as _random
    _random.seed(42)

    # Permutar frases: embaralha a ordem das frases
    texto_permutado = permutar_frases(texto)
    tokens_permutados = tokenizer.codificar(texto_permutado)

    # Augmentação com dropout de tokens
    texto_augmentado = augmentar_texto(texto, prob_dropout=0.05)
    tokens_augmentados = tokenizer.codificar(texto_augmentado)

    print(f"  Augmentação: +{len(tokens_permutados):,} tokens (permutação) +{len(tokens_augmentados):,} tokens (dropout)")

    # Split treino/validação (apenas nos tokens originais)
    val_size = int(len(tokens) * config.val_split)
    val_tokens = tokens[-val_size:] if val_size > 0 else None

    # Treino = original (sem val) + permutado + augmentado
    train_tokens = tokens[:-val_size] if val_size > 0 else tokens
    train_tokens = train_tokens + tokens_permutados + tokens_augmentados

    train_dataset = TextDataset(train_tokens, config.context_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device != "cpu",
    )

    val_loader = None
    if val_tokens and len(val_tokens) > config.context_len:
        val_dataset = TextDataset(val_tokens, config.context_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device != "cpu",
        )

    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accum_steps}x")
    print(f"  Batch efetivo: {config.batch_size * config.gradient_accum_steps}")
    print(f"  Batches por época: {len(train_loader)}")
    print(f"  Validação: {'Sim' if val_loader else 'Não'}")

    # ── Modelo ──
    modelo = GPTModel(config).to(device)
    print(modelo.resumo())

    # ── Otimizador ──
    param_decay = [p for n, p in modelo.named_parameters() if p.dim() >= 2]
    param_nodecay = [p for n, p in modelo.named_parameters() if p.dim() < 2]

    otimizador = torch.optim.AdamW(
        [
            {"params": param_decay, "weight_decay": config.weight_decay},
            {"params": param_nodecay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )

    # ── Loop de treino ──
    modelo.train()
    step_global = 0
    melhor_val_loss = float("inf")
    melhor_train_loss = float("inf")
    epochs_sem_melhora = 0
    media_loss = 0.0
    epoca = 0

    total_tokens_treinados = 0
    tempo_inicio = time.time()
    log_epocas = []

    print(f"\nIniciando treino por {config.max_epochs} épocas...")
    print("=" * 70)

    for epoca in range(1, config.max_epochs + 1):
        modelo.train()
        epoca_loss = 0.0
        n_batches = 0
        t0 = time.time()
        otimizador.zero_grad(set_to_none=True)
        accum_count = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward
            logits, loss = modelo(x, y)

            # Normalizar loss pelo accumulation steps
            loss_norm = loss / config.gradient_accum_steps
            loss_norm.backward()

            accum_count += 1
            epoca_loss += loss.item()
            n_batches += 1
            total_tokens_treinados += x.shape[0] * x.shape[1]

            # Backward + update a cada gradient_accum_steps
            if accum_count % config.gradient_accum_steps == 0:
                lr = get_lr(step_global, config)
                for param_group in otimizador.param_groups:
                    param_group["lr"] = lr

                torch.nn.utils.clip_grad_norm_(
                    modelo.parameters(), config.max_grad_norm
                )
                otimizador.step()
                otimizador.zero_grad(set_to_none=True)
                step_global += 1

        # Remanescente de accumulation
        if accum_count % config.gradient_accum_steps != 0:
            lr = get_lr(step_global, config)
            for param_group in otimizador.param_groups:
                param_group["lr"] = lr
            torch.nn.utils.clip_grad_norm_(
                modelo.parameters(), config.max_grad_norm
            )
            otimizador.step()
            otimizador.zero_grad(set_to_none=True)
            step_global += 1

        media_loss = epoca_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        perplexidade = math.exp(min(media_loss, 20))

        lr_atual = get_lr(step_global, config)

        # Validação
        val_str = ""
        val_loss = None
        if val_loader:
            val_loss = avaliar(modelo, val_loader, device)
            val_str = f" │ Val Loss: {val_loss:.4f}"

        print(
            f"Época {epoca:3d}/{config.max_epochs} │ "
            f"Loss: {media_loss:.4f} │ "
            f"PPL: {perplexidade:.2f} │ "
            f"LR: {lr_atual:.2e} │ "
            f"Tempo: {elapsed:.1f}s"
            f"{val_str}"
        )

        log_epocas.append({
            "epoca": epoca,
            "loss": round(media_loss, 6),
            "ppl": round(perplexidade, 2),
            "lr": lr_atual,
            "tempo_seg": round(elapsed, 1),
            **({"val_loss": round(val_loss, 6)} if val_loss is not None else {}),
        })

        # Salvar melhor modelo (baseado em val_loss se houver, senão train_loss)
        loss_monitor = val_loss if val_loss is not None else media_loss
        loss_ref = melhor_val_loss if val_loss is not None else melhor_train_loss

        if loss_monitor < loss_ref:
            if val_loss is not None:
                melhor_val_loss = val_loss
            else:
                melhor_train_loss = media_loss
            epochs_sem_melhora = 0

            torch.save(
                {
                    "modelo": modelo.state_dict(),
                    "config": config,
                    "epoch": epoca,
                    "loss": media_loss,
                    **({"val_loss": val_loss} if val_loss is not None else {}),
                    "total_tokens": total_tokens_treinados,
                    "tempo_total_seg": time.time() - tempo_inicio,
                },
                saida_path / "melhor_modelo.pt",
            )
            tokenizer.salvar(str(saida_path / "tokenizer.json"))
        else:
            epochs_sem_melhora += 1

        # Early stopping
        if config.patience > 0 and epochs_sem_melhora >= config.patience:
            print(f"\nEarly stopping! Sem melhora há {config.patience} épocas.")
            break

    tempo_total = time.time() - tempo_inicio
    tokens_por_segundo = total_tokens_treinados / max(tempo_total, 1)

    # Salvar modelo final
    torch.save(
        {
            "modelo": modelo.state_dict(),
            "config": config,
            "epoch": epoca,
            "loss": media_loss,
            "total_tokens": total_tokens_treinados,
            "tempo_total_seg": tempo_total,
        },
        saida_path / "modelo_final.pt",
    )

    melhor = melhor_val_loss if val_loader else melhor_train_loss
    print("=" * 70)
    print(f"Treino concluído! Melhor loss: {melhor:.4f}")
    print(f"Tempo total: {tempo_total:.1f}s ({tempo_total/60:.1f} min)")
    print(f"Tokens treinados: {total_tokens_treinados:,}")
    print(f"Throughput: {tokens_por_segundo:,.0f} tokens/s")
    print(f"Modelo salvo em: {saida_path / 'melhor_modelo.pt'}")
    print(f"Tokenizer salvo em: {saida_path / 'tokenizer.json'}")

    # Salvar log
    log_treino = {
        "config": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "context_len": config.context_len,
            "dropout": config.dropout,
            "batch_size": config.batch_size,
            "gradient_accum_steps": config.gradient_accum_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
            "warmup_steps": config.warmup_steps,
            "min_lr": config.min_lr,
            "max_grad_norm": config.max_grad_norm,
            "vocab_size": config.vocab_size,
            "tokenizer_type": config.tokenizer_type,
            "use_rope": config.use_rope,
        },
        "device": device,
        "n_params": modelo.contar_parametros(),
        "total_tokens_treinados": total_tokens_treinados,
        "tempo_total_seg": round(tempo_total, 1),
        "tempo_total_min": round(tempo_total / 60, 1),
        "tokens_por_segundo": round(tokens_por_segundo, 1),
        "melhor_loss": round(melhor, 6),
        "epocas": log_epocas,
    }
    log_path = saida_path / "log_treino.json"
    log_path.write_text(json.dumps(log_treino, indent=2, ensure_ascii=False))
    print(f"Log salvo em: {log_path}")

    return modelo, tokenizer
