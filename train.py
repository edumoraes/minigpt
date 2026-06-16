"""
train.py — Loop de treinamento do MiniGPT

O TREINAMENTO DE UM LM AUTOREGRESSIVO FUNCIONA ASSIM:

1. Pegamos uma sequência de tokens: [t0, t1, t2, ..., tN]
2. O input é a sequência:           [t0, t1, ..., tN-1]
3. O target é a mesma sequência      [t1, t2, ..., tN]
   deslocada 1 posição pra direita
4. Para CADA posição, o modelo prevê o próximo token
5. A loss (cross-entropy) mede quão errado o modelo estava

IS É CHAMADO DE "CAUSAL LANGUAGE MODELING" ou "NEXT TOKEN PREDICTION".
O GPT, GPT-2, GPT-3, GPT-4 — TODOS são treinados assim.

ANIMAÇÃO MENTAL:
Input:  "O gato comeu"   → modelo prevê "gato comeu o"
Target: "gato comeu o"   → compara com o que realmente veio

O AdamW É O OTIMIZADOR PADRÃO:
- Adam com weight decay (regularização L2 separada)
- Previne que os pesos fiquem grandes demais
- warmup + cosine decay pra learning rate

WARMUP + COSINE DECAY:
- Primeiros `warmup_steps`: lr aumenta linearmente de 0 até lr_max
- Depois: lr decai seguindo uma curva cossenoidal
- Isso evita gradientes instáveis no começo do treino
"""

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from config import GPTConfig
from model import GPTModel
from tokenizer import CharTokenizer


class TextDataset(Dataset):
    """
    Dataset que transforma texto tokenizado em pares (input, target).

    Para uma sequência de tamanho context_len+1:
    - input  = tokens[:-1]  (todos exceto o último)
    - target = tokens[1:]   (todos exceto o primeiro, deslocado +1)

    Assim, para cada posição i, o modelo aprende:
    "dados tokens[0:i], preveja tokens[i+1]"
    """

    def __init__(self, tokens: list[int], context_len: int):
        self.tokens = tokens
        self.context_len = context_len

    def __len__(self) -> int:
        # Quantas janelas completas de tamanho context_len+1 cabem?
        return max(0, len(self.tokens) - self.context_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.context_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_lr(it: int, config: GPTConfig) -> float:
    """
    Scheduler de learning rate: Warmup + Cosine Decay.

    FASE 1 (warmup): lr sobe linearmente de min_lr até learning_rate
    FASE 2 (decay): lr decai cossenoidalmente de learning_rate até min_lr

    Gráfico aproximado:

    lr │    ___....───────┐
       │   /              ╲
       │  /                ╲
       │ /                  ╲___....──── min_lr
       └───────────────────────────────→ step
           warmup   cosine decay
    """
    # Fase 1: warmup linear
    if it < config.warmup_steps:
        return config.learning_rate * (it + 1) / config.warmup_steps

    # Fase 2: cosine decay até min_lr
    progress = (it - config.warmup_steps) / max(1, 10000 - config.warmup_steps)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def treinar(
    config: GPTConfig,
    texto: str,
    saida_dir: str = "output",
    device: str | None = None,
) -> tuple[GPTModel, CharTokenizer]:
    """
    Loop de treinamento completo do MiniGPT.

    Args:
        config: hiperparâmetros do modelo
        texto: corpus de treino em português
        saida_dir: diretório pra salvar modelo e tokenizer
        device: 'cuda', 'mps' ou 'cpu' (auto-detectado se None)

    Returns:
        (modelo_treinado, tokenizer)
    """
    # Detectar dispositivo automaticamente
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Dispositivo: {device}")

    # Criar diretório de saída
    saida_path = Path(saida_dir)
    saida_path.mkdir(exist_ok=True)

    # ── Tokenizer ──
    print("Construindo tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.treinar(texto)
    print(f"  {tokenizer}")
    config.vocab_size = tokenizer.vocab_size

    # ── Dataset ──
    tokens = tokenizer.codificar(texto)
    print(f"  Corpus: {len(texto):,} caracteres → {len(tokens):,} tokens")
    print(f"  Tokens únicos: {tokenizer.vocab_size}")

    dataset = TextDataset(tokens, config.context_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device != "cpu",
    )
    print(f"  Batch size: {config.batch_size}")
    print(f"  Batches por época: {len(dataloader)}")

    # ── Modelo ──
    modelo = GPTModel(config).to(device)
    print(modelo.resumo())

    # ── Otimizador ──
    # Separar parâmetros com e sem weight decay
    # Biases e LayerNorm gains NÃO recebem weight decay
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
    melhor_loss = float("inf")

    total_tokens_treinados = 0
    tempo_inicio = time.time()
    log_epocas = []

    print(f"\nIniciando treino por {config.max_epochs} épocas...")
    print("=" * 60)

    for epoca in range(1, config.max_epochs + 1):
        epoca_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            # Atualizar learning rate
            lr = get_lr(step_global, config)
            for param_group in otimizador.param_groups:
                param_group["lr"] = lr

            # Forward pass
            logits, loss = modelo(x, y)

            # Backward pass
            otimizador.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping: limitar magnitude dos gradientes
            # Previne "explosão de gradientes" em RNNs/Transformers
            torch.nn.utils.clip_grad_norm_(
                modelo.parameters(), config.max_grad_norm
            )

            # Atualizar pesos
            otimizador.step()

            epoca_loss += loss.item()
            n_batches += 1
            step_global += 1
            total_tokens_treinados += x.shape[0] * x.shape[1]

        # Estatísticas da época
        media_loss = epoca_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        perplexidade = math.exp(min(media_loss, 20))  # clamp pra evitar overflow

        print(
            f"Época {epoca:3d}/{config.max_epochs} │ "
            f"Loss: {media_loss:.4f} │ "
            f"PPL: {perplexidade:.2f} │ "
            f"LR: {lr:.2e} │ "
            f"Tempo: {elapsed:.1f}s"
        )

        log_epocas.append({
            "epoca": epoca,
            "loss": round(media_loss, 6),
            "ppl": round(perplexidade, 2),
            "lr": lr,
            "tempo_seg": round(elapsed, 1),
        })

        # Salvar o melhor modelo
        if media_loss < melhor_loss:
            melhor_loss = media_loss
            torch.save(
                {
                    "modelo": modelo.state_dict(),
                    "config": config,
                    "epoch": epoca,
                    "loss": media_loss,
                    "total_tokens": total_tokens_treinados,
                    "tempo_total_seg": time.time() - tempo_inicio,
                },
                saida_path / "melhor_modelo.pt",
            )
            tokenizer.salvar(str(saida_path / "tokenizer.json"))

    tempo_total = time.time() - tempo_inicio
    tokens_por_segundo = total_tokens_treinados / max(tempo_total, 1)

    # Salvar modelo final também
    torch.save(
        {
            "modelo": modelo.state_dict(),
            "config": config,
            "epoch": config.max_epochs,
            "loss": media_loss,
            "total_tokens": total_tokens_treinados,
            "tempo_total_seg": tempo_total,
        },
        saida_path / "modelo_final.pt",
    )

    print("=" * 60)
    print(f"Treino concluído! Melhor loss: {melhor_loss:.4f}")
    print(f"Tempo total: {tempo_total:.1f}s ({tempo_total/60:.1f} min)")
    print(f"Tokens treinados: {total_tokens_treinados:,}")
    print(f"Throughput: {tokens_por_segundo:,.0f} tokens/s")
    print(f"Modelo salvo em: {saida_path / 'melhor_modelo.pt'}")
    print(f"Tokenizer salvo em: {saida_path / 'tokenizer.json'}")

    # Salvar log de treino em JSON
    log_treino = {
        "config": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "context_len": config.context_len,
            "dropout": config.dropout,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
            "warmup_steps": config.warmup_steps,
            "min_lr": config.min_lr,
            "max_grad_norm": config.max_grad_norm,
            "vocab_size": config.vocab_size,
        },
        "device": device,
        "n_params": modelo.contar_parametros(),
        "total_tokens_treinados": total_tokens_treinados,
        "tempo_total_seg": round(tempo_total, 1),
        "tempo_total_min": round(tempo_total / 60, 1),
        "tokens_por_segundo": round(tokens_por_segundo, 1),
        "melhor_loss": round(melhor_loss, 6),
        "epocas": log_epocas,
    }
    log_path = saida_path / "log_treino.json"
    log_path.write_text(json.dumps(log_treino, indent=2, ensure_ascii=False))
    print(f"Log salvo em: {log_path}")

    return modelo, tokenizer