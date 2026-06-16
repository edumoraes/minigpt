"""
model.py — O modelo GPT do zero

Este é o arquivo mais importante do projeto. Implementamos cada
componente do Transformer na mão, sem usar nn.Transformer do PyTorch.

A ARQUITETURA GPT (de baixo pra cima):

1. Token Embedding + Position Embedding
   Converte tokens (inteiros) em vetores densos e adiciona
   informação posicional (posição do token na sequência).

2. N× TransformerBlock (empilhados)
   Cada bloco contém:
   ├── LayerNorm
   ├── MultiHeadSelfAttention  ← "o mecanismo que faz o modelo olhar
   │     (com residual)           pra todos os tokens anteriores")
   ├── LayerNorm
   └── FeedForward              ← "processa cada token individualmente"
         (com residual)

3. LayerNorm final + Linear head
   Projeta de volta pro vocabulário para prever o próximo token.

POR QUE ESSA ORDEM (Pre-Norm)?
O artigo original do Transformer usava Post-Norm (normalização depois
da atenção). Mas GPT-2 em diante usa Pre-Norm (normalização ANTES).
Isso estabiliza o treinamento em redes profundas.

A INTUIÇÃO DO TRANSFORMER:
- Self-Attention: "Dado o contexto, quão relevante é cada token
  anterior pra prever o próximo?"
- Feed-Forward: "Processa o que a atenção juntou e transforma."
- Residual: "Não perde informação anterior, soma em cima."
- LayerNorm: "Normaliza pra não explodir/desaparecer os gradientes."

REFERÊNCIAS:
- "Attention Is All You Need" (Vaswani et al., 2017)
- GPT-2: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- nanoGPT de Karpathy: https://github.com/karpathy/nanoGPT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class LayerNorm(nn.Module):
    """
    Layer Normalization — normaliza pra média 0 e variância 1.

    Diferente do BatchNorm (que normaliza pelo batch), o LayerNorm
    normaliza CADA token individualmente pelas suas features.

    Fórmula: y = (x - μ) / sqrt(σ² + ε) * γ + β
    onde γ (gain) e β (bias) são parâmetros aprendíveis.

    PQ PRE-SCALED? Usamos 1+γ ao invés de γ direto pra inicialização
    ficar próxima de identidade no começo do treino.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gain * x_norm + self.bias


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention — o coração do Transformer.

    A IDEIA:
    Em vez de olhar pra todos os tokens anteriores com uma única
    "lente", usamos múltiplas cabeças (heads), cada uma aprendendo
    um tipo diferente de relação.

    EXEMPLO PRÁTICO:
    Na frase "O gato comeu o peixe", diferentes cabeças podem aprender:
    - Head 1: relação sujeito→verbo ("gato"→"comeu")
    - Head 2: relação artigo→substantivo ("o"→"peixe")
    - Head 3: relação temporal, etc.

    MATEMÁTICA:
    1. Projetamos input em Q (query), K (key), V (value)
    2. Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V
    3. A máscara causal garante que só olhamos pra trás

    A MÁSCARA CAUSAL:
    - Tokens SÓ podem olhar pra tokens anteriores
    - Isso torna o modelo autoregressivo (gera um token de cada vez)
    - Implementamos com uma matriz triangular superior de -inf
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Projeções lineares: input → Q, K, V
        # Fazemos TUDO numa matriz grande e depois splittamos
        self.W_qkv = nn.Linear(config.d_model, 3 * config.d_model)
        # Projeção de saída: concat(heads) → output
        self.W_out = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, d_model

        # 1. Projetar pra Q, K, V e dividir entre as cabeças
        qkv = self.W_qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # cada um: (B, T, C)

        # Reshape: (B, T, C) → (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. Calcular scores de atenção
        # Q·K^T / √d_k — o √d_k evita que softmax sature
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. Máscara causal: -inf na parte superior direita
        # Só permitimos olhar pra trás (tokens anteriores)
        mascara_causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(mascara_causal, float("-inf"))

        # 4. Softmax: converte scores em probabilidades
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 5. Multiplicar por V: média ponderada dos valores
        out = attn_probs @ v  # (B, n_heads, T, head_dim)

        # 6. Concatenar as cabeças e projetar
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_out(out)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):
    """
    Feed-Forward Network — processa cada token individualmente.

    ARQUITETURA:
    input → Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model) → output
    
    PQ 4x? O GPT-2 usa 4x a dimensão internamente. Isso dá mais
    capacidade pro modelo aprender relações não-lineares.
    
    PQ GELU e não ReLU? GELU (Gaussian Error Linear Unit) é mais suave
    e funciona melhor em transformers. GELU(0) ≈ 0.5, ReLU(0) = 0.
    Essa suavidade evita "neurônios mortos".
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Bloco Transformer — a unidade básica que empilhamos.

    ESTRUTURA (Pre-Norm):
        x ──→ LayerNorm → Attention → (+) ──→ LayerNorm → FFN → (+) ──→ saída
        |                                       |                      |
        └─────── residual ──────────────────────┘──────────────────────┘

    AS CONEXÕES RESIDUAIS (skip connections) são CRUCIAIS:
    - Permitem que gradientes fluam diretamente pelo caminho curto
    - O modelo pode escolher "ignorar" transformações se necessário
    - Sem elas, redes profundas são intrainteríveis (vanishing gradients)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Atenção com conexão residual
        x = x + self.attn(self.ln1(x))
        # FFN com conexão residual
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """
    GPT — Generative Pre-trained Transformer.

    O MODELO COMPLETO, de ponta a ponta:
    
    tokens (inteiros)
      → Token Embedding (lookup table: token → vetor denso)
      + Position Embedding (informa a posição de cada token)
      → Dropout
      → N× TransformerBlock
      → LayerNorm final
      → Linear head (projeta de volta pro vocabulário)
      → logits (probabilidades não-normalizadas do próximo token)

    TREINAMENTO:
    Dado um batch de sequências como [t0, t1, t2, ..., tn],
    o modelo prevê [t1, t2, t3, ..., tn+1].
    Ou seja: o input é a sequência inteira, o target é a mesma
    sequência deslocada 1 posição pra direita.
    
    ISSO É SUPER EFICIENTE porque processamos TODOS os pares
    input→target em paralelo numa única passada forward!
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Embeddings: convertem tokens (inteiros) em vetores densos
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Position embedding: aprende uma representação pra cada posição
        self.position_embedding = nn.Embedding(config.context_len, config.d_model)

        self.drop = nn.Dropout(config.dropout)

        # Os blocos Transformer empilhados
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Normalização final
        self.ln_f = LayerNorm(config.d_model)

        # Cabeça de classificação: projeta embeddings de volta pro vocabulário
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Compartilhar pesos entre token_embedding e lm_head
        # Isso é chamado "weight tying" e reduz o número de parâmetros
        self.lm_head.weight = self.token_embedding.weight

        # Inicialização dos pesos (seguindo GPT-2)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Inicialização dos pesos seguindo o paper do GPT-2.
        
        Linear layers: Normal(0, 0.02)
        Embeddings: Normal(0, 0.02)
        Bias: zero
        
        Camadas residuais: ganho extra pequeno (0.02 / sqrt(2*n_layers))
        Isso compensa o acúmulo residual ao longo das camadas.
        """
        n_layers = self.config.n_layers
        for nome, modulo in self.named_modules():
            if isinstance(modulo, nn.Linear):
                nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
                if modulo.bias is not None:
                    nn.init.zeros_(modulo.bias)
            elif isinstance(modulo, nn.Embedding):
                nn.init.normal_(modulo.weight, mean=0.0, std=0.02)

            # Ganho extra pra camadas residuais — estabiliza treino profundo
            if nome.endswith(("ffn.net.0", "W_qkv", "W_out")):
                nn.init.normal_(
                    modulo.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * n_layers),
                )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass do modelo.

        Args:
            idx: tensor de IDs de tokens, shape (B, T)
                B = batch_size, T = comprimento da sequência
            targets: IDs esperados, shape (B, T)
                Se fornecido, calcula a cross-entropy loss.

        Returns:
            logits: (B, T, vocab_size) — scores para cada token do vocab
            loss: escalar, ou None se targets não fornecido
        """
        B, T = idx.shape

        # Posição: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)

        # Embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)
        pos_emb = self.position_embedding(pos)  # (1, T, d_model)
        x = self.drop(tok_emb + pos_emb)  # broadcasting: (B, T, d_model)

        # Passar pelos blocos Transformer
        x = self.blocks(x)  # (B, T, d_model)
        x = self.ln_f(x)  # (B, T, d_model)

        # Projetar pro vocabulário
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calcular loss se targets fornecidos
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def contar_parametros(self) -> int:
        """Retorna o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def resumo(self) -> str:
        """Retorna um resumo legível do modelo."""
        n = self.contar_parametros()
        c = self.config
        lines = [
            f"════════════════ MiniGPT — Resumo do Modelo ════════════════",
            f"  Parâmetros:     {n:,}",
            f"  Vocabulário:    {c.vocab_size}",
            f"  d_model:        {c.d_model}",
            f"  n_heads:        {c.n_heads}",
            f"  head_dim:       {c.head_dim}",
            f"  n_layers:       {c.n_layers}",
            f"  context_len:    {c.context_len}",
            f"  dropout:        {c.dropout}",
            f"  FFN dim:        {4 * c.d_model}",
            f"════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)