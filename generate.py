"""
generate.py — Geração autoregressiva de texto

EXPANDIDO com:
- Top-p (nucleus sampling): filtra tokens por probabilidade acumulada
- Repetition penalty: penaliza tokens já gerados
- Typical sampling: seçãoo baseada em entropia
- Beam search: busca pela melhor sequência
- Temperatura e top-k (original)
"""

import torch
import torch.nn.functional as F

from model import GPTModel
from tokenizer import CharTokenizer, BPETokenizer


# ──────────────────────────────────────────────────────────
# Funções auxiliares de filtragem
# ──────────────────────────────────────────────────────────

def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Top-k: mantém só os k tokens com maior probabilidade."""
    if top_k is None or top_k <= 0:
        return logits
    kth_vals = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
    logits = logits.masked_fill(logits < kth_vals, float("-inf"))
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Top-p (nucleus sampling): mantém os tokens com probabilidade
    acumulada <= top_p.

    Vantagem sobre top-k: adapta-se à distribuição. Em distribuições
    picudas, mantém poucos tokens; em distribuições planas, mantém mais.
    """
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remover tokens cuja prob. acumulada excede top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shiftar: manter pelo menos 1 token (o mais provável)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Mapear de volta para os índices originais
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    tokens_gerados: list[int],
    penalty: float,
) -> torch.Tensor:
    """
    Repetition penalty: divide logits de tokens já gerados.

    Se penalty=1.0, não faz nada. Valores >1.0 desencorajam repetição.
    Fórmula: se logit > 0: logit /= penalty; se logit < 0: logit *= penalty.
    """
    if penalty == 1.0 or not tokens_gerados:
        return logits
    for token_id in set(tokens_gerados):
        if token_id < logits.size(-1):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
    return logits


def apply_typical_sampling(
    logits: torch.Tensor, mass: float = 0.9
) -> torch.Tensor:
    """
    Typical sampling: seleciona tokens cuja informação está
    próxima da entropia da distribuição (entropia típica).

    Intuição:Tokens " típicos" têm probabilidade próxima do esperado.
    Tokens muito improváveis ou muito certos são removidos.
    """
    if mass >= 1.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # Entropia da distribuição
    entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

    # Informação de cada token: -log(p)
    info_content = -log_probs

    # Desvio da entropia (menor = mais típico)
    deviation = torch.abs(info_content - entropy)

    # Ordenar por tipicidade e acumular probabilidade
    sorted_indices = torch.argsort(deviation, dim=-1)
    sorted_probs = probs.gather(-1, sorted_indices)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Manter tokens até atingir mass
    cutoff_mask = cumulative_probs <= mass
    # Pelo menos 1 token
    cutoff_mask[..., 0] = True

    # Mapear de volta para os índices originais
    indices_to_remove = sorted_mask = ~cutoff_mask
    tokens_to_remove = sorted_indices.gather(
        -1, indices_to_remove.long().argsort(dim=-1)
    )

    # Simplificação: usar scatter para remover
    final_mask = torch.ones_like(logits, dtype=torch.bool)
    for i in range(logits.size(-1)):
        if not cutoff_mask[0, i]:
            idx = sorted_indices[0, i]
            final_mask[0, idx.item()] = False

    logits = logits.masked_fill(~final_mask, float("-inf"))
    return logits


# ──────────────────────────────────────────────────────────
# Geração autoregressiva
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def gerar(
    modelo: GPTModel,
    tokenizer: CharTokenizer | BPETokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    typical_mass: float = 1.0,
    device: str = "cpu",
) -> str:
    """
    Gera texto a partir de um prompt usando o modelo treinado.

    Args:
        modelo: modelo GPT treinado
        tokenizer: tokenizer correspondente
        prompt: texto inicial
        max_tokens: máximo de tokens a gerar
        temperature: controla aleatoriedade (0.1=conservador, 1.0=normal)
        top_k: só considerar os k tokens mais prováveis (None=desativado)
        top_p: nucleus sampling (0.9 = manter 90% da prob. acumulada)
        repetition_penalty: penalizar repetição (>1.0 = desencorajar)
        typical_mass: typical sampling mass (1.0 = desativado, 0.9 = típico)
        device: dispositivo
    """
    modelo.eval()

    tokens = tokenizer.codificar(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    if isinstance(tokenizer, BPETokenizer):
        eos_id = tokenizer.vocab.get("<EOS>", -1)
    else:
        eos_id = tokenizer.char_to_id.get("<EOS>", -1)
    tokens_gerados = list(tokens)

    for _ in range(max_tokens):
        idx_cond = idx[:, -modelo.config.context_len :]
        logits, _ = modelo(idx_cond)
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / max(temperature, 1e-8)

        # Repetition penalty
        logits = apply_repetition_penalty(logits, tokens_gerados, repetition_penalty)

        # Top-k
        logits = apply_top_k(logits, top_k)

        # Top-p (nucleus sampling)
        logits = apply_top_p(logits, top_p)

        # Typical sampling
        if typical_mass < 1.0:
            logits = apply_typical_sampling(logits, typical_mass)

        # Amostrar
        probs = F.softmax(logits, dim=-1)
        proximo_token = torch.multinomial(probs, num_samples=1)

        if proximo_token.item() == eos_id:
            break

        tokens_gerados.append(proximo_token.item())
        idx = torch.cat([idx, proximo_token], dim=1)

    return tokenizer.decodificar(idx[0].tolist())


def gerar_variacoes(
    modelo: GPTModel,
    tokenizer: CharTokenizer | BPETokenizer,
    prompt: str,
    n_variacoes: int = 3,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: str = "cpu",
) -> list[str]:
    """Gera múltiplas variações com o mesmo prompt."""
    resultados = []
    for _ in range(n_variacoes):
        texto = gerar(
            modelo, tokenizer, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        )
        resultados.append(texto)
    return resultados


# ──────────────────────────────────────────────────────────
# Beam Search
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def gerar_beam_search(
    modelo: GPTModel,
    tokenizer: CharTokenizer | BPETokenizer,
    prompt: str,
    max_tokens: int = 100,
    beam_width: int = 5,
    temperature: float = 1.0,
    length_penalty: float = 0.6,
    device: str = "cpu",
) -> str:
    """
    Gera texto usando beam search.

    Ao invés de amostrar 1 token por vez, mantém beam_width hipóteses
    e seleciona a de maior score no final.

    Args:
        beam_width: número de hipóteses mantidas em paralelo
        length_penalty: penaliza sequências longas (0.6 = bom padrão)
    """
    modelo.eval()

    tokens = tokenizer.codificar(prompt)
    if isinstance(tokenizer, BPETokenizer):
        eos_id = tokenizer.vocab.get("<EOS>", -1)
    else:
        eos_id = tokenizer.char_to_id.get("<EOS>", -1)

    # Cada hipótese: (score, lista_de_tokens)
    beams: list[tuple[float, list[int]]] = [(0.0, tokens)]

    for _ in range(max_tokens):
        all_candidates = []

        for score, seq in beams:
            idx = torch.tensor([seq[-modelo.config.context_len:]], dtype=torch.long, device=device)
            logits, _ = modelo(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Pegar top beam_width tokens
            log_probs = F.log_softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(log_probs[0], beam_width)

            for i in range(beam_width):
                token_id = top_indices[i].item()
                token_score = top_probs[i].item()
                new_seq = seq + [token_id]
                new_score = score + token_score
                all_candidates.append((new_score, new_seq))

        # Selecionar beam_width melhores
        # Penalizar por comprimento
        all_candidates.sort(
            key=lambda x: x[0] / ((len(x[1])) ** length_penalty),
            reverse=True,
        )
        beams = all_candidates[:beam_width]

        # Se todos os beams terminaram com EOS, parar
        if all(seq[-1] == eos_id for _, seq in beams):
            break

    # Retornar o melhor beam
    best_score, best_seq = max(
        beams,
        key=lambda x: x[0] / ((len(x[1])) ** length_penalty),
    )
    return tokenizer.decodificar(best_seq)