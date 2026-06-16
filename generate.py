"""
generate.py — Geração autoregressiva de texto

COMO O GPT GERA TEXTO?

O processo é chamado AUTOREGRESSIVO:
1. Dado um prompt (texto inicial), tokenizamos ele
2. O modelo prevê probabilidades pro próximo token
3. Amostramos um token dessas probabilidades
4. Adicionamos o token escolhido ao contexto
5. Repetimos até atingir o tamanho máximo ou um token de parada

METAFORICAMENTE: imagine que você está escrevendo um livro.
Para cada palavra, você olha tudo que escreveu antes e decide
a próxima. O GPT faz exatamente isso, um token de cada vez.

ESTRATÉGIAS DE AMOSTRAGEM:
- Greedy: sempre escolhe o token mais provável (determinístico, repetitivo)
- Temperature: controla "criatividade"
  - T=0.1: muito conservador (quase greedy)
  - T=1.0: segue a distribuição original
  - T=2.0: mais aleatório e criativo (mas pode ser incoerente)
- Top-k: só considera os k tokens mais prováveis
  - k=1: greedy
  - k=50: boa diversão entre qualidade e variedade

FÓRMULA DA TEMPERATURE:
logits_ajustados = logits / temperature
probabilidades = softmax(logits_ajustados)

Temperature alta → probabilidades mais uniformes → mais aleatório
Temperature baixa → probabilidades mais picudas → mais determinístico
"""

import torch
import torch.nn.functional as F

from model import GPTModel
from tokenizer import CharTokenizer


@torch.no_grad()
def gerar(
    modelo: GPTModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    device: str = "cpu",
) -> str:
    """
    Gera texto a partir de um prompt usando o modelo treinado.

    Args:
        modelo: modelo GPT treinado
        tokenizer: tokenizer correspondente
        prompt: texto inicial pra começar a geração
        max_tokens: quantos tokens gerar no máximo
        temperature: controla aleatoriedade (0.1=conservador, 1.0=normal, 2.0=criativo)
        top_k: só considerar os k tokens mais prováveis (None = desativado)
        device: dispositivo pra computação

    Returns:
        Texto gerado (incluindo o prompt)
    """
    modelo.eval()

    # Tokenizar o prompt
    tokens = tokenizer.codificar(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    eos_id = tokenizer.char_to_id.get("<EOS>", -1)

    for _ in range(max_tokens):
        # Truncar o contexto se for maior que a janela do modelo
        idx_cond = idx[:, -modelo.config.context_len :]

        # Forward pass: obter logits do próximo token
        logits, _ = modelo(idx_cond)

        # Pegar só o último token (a previsão mais recente)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # ── Aplicar temperature ──
        # Dividir logits pela temperature ANTES do softmax
        logits = logits / max(temperature, 1e-8)

        # ── Aplicar top-k filtering ──
        # Manter só os k tokens com maior probabilidade
        if top_k is not None and top_k > 0:
            # Encontrar o k-ésimo maior logit
            kth_vals = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
            # Zerar todos os que estão abaixo
            logits = logits.masked_fill(logits < kth_vals, float("-inf"))

        # ── Amostrar ──
        probs = F.softmax(logits, dim=-1)

        # Multinomial: escolhe um token proporcionalmente às probabilidades
        proximo_token = torch.multinomial(probs, num_samples=1)

        # Verificar se gerou EOS
        if proximo_token.item() == eos_id:
            break

        # Adicionar ao contexto
        idx = torch.cat([idx, proximo_token], dim=1)

    # Decodificar tudo de volta pra texto
    return tokenizer.decodificar(idx[0].tolist())


def gerar_variacoes(
    modelo: GPTModel,
    tokenizer: CharTokenizer,
    prompt: str,
    n_variacoes: int = 3,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    device: str = "cpu",
) -> list[str]:
    """
    Gera múltiplas variações com o mesmo prompt.
    Útil pra explorar a diversidade do modelo.
    """
    resultados = []
    for i in range(n_variacoes):
        texto = gerar(
            modelo, tokenizer, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        resultados.append(texto)
    return resultados