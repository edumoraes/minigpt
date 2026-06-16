"""
sft.py — Supervised Fine-Tuning

SFT transforma um modelo pré-treinado (que só prevê próximo token)
num modelo de instrução que segue comandos.

COMO FUNCIONA:
- Formato: "Instrução: {pergunta} Resposta: {resposta}"
- A loss é computada APENAS nos tokens da resposta
- Tokens da instrução recebem target=-100 (ignorados pelo cross_entropy)
- Learning rate menor que o pré-treinamento (1e-5 vs 3e-4)

INTUIÇÃO:
- Pré-treinamento: modelo aprende a LINGUAGEM (português)
- SFT: modelo aprende a SEGUIR INSTRUÇÕES (Q&A, comandos)
"""

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from config import GPTConfig
from model import GPTModel
from tokenizer import CharTokenizer, BPETokenizer


SEPARATOR = " Resposta: "
IGNORE_INDEX = -100


class SFTDataset(Dataset):
    """
    Dataset para Supervised Fine-Tuning.

    Cada amostra é formatada como:
        Instrução: {instruction} Resposta: {response}

    Targets da instrução são -100 (ignorados).
    Targets da resposta são os tokens reais.
    """

    def __init__(
        self,
        pares: list[tuple[str, str]],
        tokenizer: CharTokenizer | BPETokenizer,
        context_len: int,
    ):
        self.samples = []
        self.context_len = context_len

        for instrucao, resposta in pares:
            # Formato completo
            texto_completo = f"Instrução: {instrucao}{SEPARATOR}{resposta}"
            tokens_completo = tokenizer.codificar(texto_completo)

            # Parte da instrução (para mascarar)
            texto_inst = f"Instrução: {instrucao}{SEPARATOR}"
            tokens_inst = tokenizer.codificar(texto_inst)
            inst_len = len(tokens_inst)

            if len(tokens_completo) > context_len:
                continue

            # Input: tokens_completo, Target: shift right, com mask
            input_ids = tokens_completo
            targets = tokens_completo[1:] + [0]
            # Máscara: ignorar tokens da instrução
            targets = [-100] * min(inst_len, len(targets)) + targets[inst_len:]
            targets = targets[: len(input_ids)]

            if len(input_ids) > 0 and len(targets) > 0:
                self.samples.append(
                    (
                        input_ids[:context_len],
                        targets[:context_len],
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, targets = self.samples[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )


def treinar_sft(
    modelo: GPTModel,
    tokenizer: CharTokenizer | BPETokenizer,
    config: GPTConfig,
    dados_sft: list[tuple[str, str]],
    saida_dir: str = "output",
    device: str | None = None,
) -> GPTModel:
    """
    Treina o modelo com Supervised Fine-Tuning.

    Args:
        modelo: modelo pré-treinado
        tokenizer: tokenizer correspondente
        config: hiperparâmetros (usa sft_lr e sft_epochs)
        dados_sft: lista de (instrução, resposta)
        saida_dir: diretório pra salvar
        device: dispositivo
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\n{'='*60}")
    print(f"SFT — Supervised Fine-Tuning")
    print(f"Dispositivo: {device}")
    print(f"Dados de treino: {len(dados_sft)} pares instrução-resposta")
    print(f"Learning rate: {config.sft_lr}")
    print(f"Épocas: {config.sft_epochs}")
    print(f"{'='*60}\n")

    saida_path = Path(saida_dir)

    # Dataset
    dataset = SFTDataset(dados_sft, tokenizer, config.context_len)
    if len(dataset) == 0:
        print("ERRO: Nenhuma amostra SFT válida. Aumente context_len ou simplifique os dados.")
        return modelo

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device != "cpu",
    )
    print(f"  Amostras SFT válidas: {len(dataset)}")
    print(f"  Batches por época: {len(dataloader)}")

    # Otimizador (LR menor para fine-tuning)
    modelo = modelo.to(device)
    optimizer = torch.optim.AdamW(
        modelo.parameters(),
        lr=config.sft_lr,
        weight_decay=config.weight_decay,
    )

    # Loop de treino
    modelo.train()
    melhor_loss = float("inf")

    for epoca in range(1, config.sft_epochs + 1):
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits, loss = modelo(x, y)
            if loss is None:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), config.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        media_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(
            f"SFT Época {epoca:3d}/{config.sft_epochs} │ "
            f"Loss: {media_loss:.4f} │ Tempo: {elapsed:.1f}s"
        )

        # Salvar melhor modelo SFT
        if media_loss < melhor_loss:
            melhor_loss = media_loss
            torch.save(
                {
                    "modelo": modelo.state_dict(),
                    "config": config,
                    "epoch": epoca,
                    "loss": media_loss,
                    "tipo": "sft",
                },
                saida_path / "melhor_modelo_sft.pt",
            )

    modelo.eval()
    print(f"\nSFT concluído! Melhor loss: {melhor_loss:.4f}")
    return modelo