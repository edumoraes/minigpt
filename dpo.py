"""
dpo.py — Direct Preference Optimization

DPO alinha o modelo com preferências humanas SEM precisar de RL.

COMO FUNCIONA:
- Dado um prompt, uma resposta "escolhida" (boa) e uma "rejeitada" (ruim)
- A loss aumenta a probabilidade da escolhida e diminui da rejeitada
- Usa um modelo de referência (cópia congelada do pré-treinado)
- Não precisa de reward model nem PPO (mais simples que RLHF)

MATEMÁTICA (simplificada):
L = -log(σ(β · (log π(y_w|x) - log π_ref(y_w|x) - log π(y_l|x) + log π_ref(y_l|x))))

Onde:
- π = modelo sendo treinado
- π_ref = modelo de referência (congelado)
- y_w = resposta escolhida (preferred)
- y_l = resposta rejeitada
- β = temperatura (controla força do alinhamento)
- σ = sigmoide

REFERÊNCIA:
- "Direct Preference Optimization" (Rafailov et al., 2023)
"""

import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import GPTConfig
from model import GPTModel
from tokenizer import CharTokenizer, BPETokenizer


class DPODataset(Dataset):
    """
    Dataset para DPO com pares de preferência.

    Cada amostra: (prompt, resposta_escolhida, resposta_rejeitada)
    """

    def __init__(
        self,
        pares: list[tuple[str, str, str]],
        tokenizer: CharTokenizer | BPETokenizer,
        context_len: int,
    ):
        self.samples = []
        self.context_len = context_len

        for prompt, chosen, rejected in pares:
            chosen_ids = tokenizer.codificar(prompt + chosen)
            rejected_ids = tokenizer.codificar(prompt + rejected)
            prompt_ids = tokenizer.codificar(prompt)

            if (
                len(chosen_ids) > context_len
                or len(rejected_ids) > context_len
            ):
                continue

            # Targets = shift right
            chosen_targets = chosen_ids[1:] + [0]
            rejected_targets = rejected_ids[1:] + [0]

            self.samples.append({
                "chosen_ids": chosen_ids,
                "chosen_targets": chosen_targets,
                "rejected_ids": rejected_ids,
                "rejected_targets": rejected_targets,
                "prompt_len": len(prompt_ids),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        max_len = self.context_len

        # Pad to same length within the pair
        chosen_len = len(sample["chosen_ids"])
        rejected_len = len(sample["rejected_ids"])
        pair_max = max(chosen_len, rejected_len, 1)

        def pad_seq(seq, target_len, pad_val=0):
            return seq[:target_len] + [pad_val] * max(0, target_len - len(seq))

        return {
            "chosen_ids": torch.tensor(pad_seq(sample["chosen_ids"], pair_max), dtype=torch.long),
            "chosen_targets": torch.tensor(pad_seq(sample["chosen_targets"], pair_max, -100), dtype=torch.long),
            "rejected_ids": torch.tensor(pad_seq(sample["rejected_ids"], pair_max), dtype=torch.long),
            "rejected_targets": torch.tensor(pad_seq(sample["rejected_targets"], pair_max, -100), dtype=torch.long),
            "prompt_len": sample["prompt_len"],
        }


def _get_log_probs(
    modelo: GPTModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Computa log-probabilidades somadas dos tokens target (ignorando -100)."""
    logits, _ = modelo(input_ids, targets)
    # logits: (B, T, V), targets: (B, T)
    log_probs = F.log_softmax(logits, dim=-1)
    # Pegar log-prob do token target
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    # Máscara: ignorar -100
    mask = targets != -100
    return (token_log_probs * mask).sum(dim=-1)


def treinar_dpo(
    modelo: GPTModel,
    tokenizer: CharTokenizer | BPETokenizer,
    config: GPTConfig,
    dados_dpo: list[tuple[str, str, str]],
    saida_dir: str = "output",
    device: str | None = None,
) -> GPTModel:
    """
    Treina o modelo com Direct Preference Optimization.

    Args:
        modelo: modelo pré-treinado (ou pós-SFT)
        tokenizer: tokenizer correspondente
        config: hiperparâmetros (usa dpo_beta, dpo_lr, dpo_epochs)
        dados_dpo: lista de (prompt, resposta_escolhida, resposta_rejeitada)
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
    print(f"DPO — Direct Preference Optimization")
    print(f"Dispositivo: {device}")
    print(f"Dados: {len(dados_dpo)} pares de preferência")
    print(f"beta: {config.dpo_beta}")
    print(f"Learning rate: {config.dpo_lr}")
    print(f"Épocas: {config.dpo_epochs}")
    print(f"{'='*60}\n")

    saida_path = Path(saida_dir)

    # Criar modelo de referência (cópia congelada)
    modelo_ref = copy.deepcopy(modelo)
    modelo_ref.eval()
    for p in modelo_ref.parameters():
        p.requires_grad = False

    modelo = modelo.to(device)
    modelo_ref = modelo_ref.to(device)

    # Dataset
    dataset = DPODataset(dados_dpo, tokenizer, config.context_len)
    if len(dataset) == 0:
        print("ERRO: Nenhuma amostra DPO válida. Aumente context_len ou simplifique os dados.")
        return modelo

    dataloader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        pin_memory=device != "cpu",
    )
    print(f"  Amostras DPO válidas: {len(dataset)}")

    # Otimizador
    optimizer = torch.optim.AdamW(
        [p for p in modelo.parameters() if p.requires_grad],
        lr=config.dpo_lr,
    )

    # Loop
    modelo.train()
    beta = config.dpo_beta
    media_loss = 0.0

    for epoca in range(1, config.dpo_epochs + 1):
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in dataloader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_targets = batch["chosen_targets"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_targets = batch["rejected_targets"].to(device)

            # Log probs do modelo treinável
            log_pi_chosen = _get_log_probs(modelo, chosen_ids, chosen_targets)
            log_pi_rejected = _get_log_probs(modelo, rejected_ids, rejected_targets)

            # Log probs do modelo de referência (sem gradiente)
            with torch.no_grad():
                log_ref_chosen = _get_log_probs(modelo_ref, chosen_ids, chosen_targets)
                log_ref_rejected = _get_log_probs(modelo_ref, rejected_ids, rejected_targets)

            # DPO loss
            log_ratio_chosen = log_pi_chosen - log_ref_chosen
            log_ratio_rejected = log_pi_rejected - log_ref_rejected

            loss = -F.logsigmoid(
                beta * (log_ratio_chosen - log_ratio_rejected)
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), config.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        media_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Calcular accuracy: quantas vezes chosen > rejected
        with torch.no_grad():
            modelo.eval()
            correct = 0
            total = 0
            for batch in dataloader:
                chosen_ids = batch["chosen_ids"].to(device)
                chosen_targets = batch["chosen_targets"].to(device)
                rejected_ids = batch["rejected_ids"].to(device)
                rejected_targets = batch["rejected_targets"].to(device)

                log_pi_c = _get_log_probs(modelo, chosen_ids, chosen_targets)
                log_pi_r = _get_log_probs(modelo, rejected_ids, rejected_targets)
                correct += (log_pi_c > log_pi_r).sum().item()
                total += log_pi_c.size(0)
            accuracy = correct / max(total, 1)
            modelo.train()

        print(
            f"DPO Época {epoca:3d}/{config.dpo_epochs} │ "
            f"Loss: {media_loss:.4f} │ "
            f"Acc: {accuracy:.1%} │ "
            f"Tempo: {elapsed:.1f}s"
        )

    # Salvar modelo DPO
    torch.save(
        {
            "modelo": modelo.state_dict(),
            "config": config,
            "epoch": config.dpo_epochs,
            "loss": media_loss,
            "tipo": "dpo",
        },
        saida_path / "melhor_modelo_dpo.pt",
    )

    modelo.eval()
    print(f"\nDPO concluído! Loss final: {media_loss:.4f}")
    return modelo