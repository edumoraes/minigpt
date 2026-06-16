"""
model.py — O modelo GPT do zero

Implementamos cada componente do Transformer na mão, sem usar
nn.Transformer do PyTorch.

NOVIDADES:
- RoPE (Rotary Position Embedding): codifica posição via rotação
  nos vetores Q e K ao invés de soma de embeddings. Generaliza
  melhor para sequências longas.
- Flash Attention: usa F.scaled_dot_product_attention do PyTorch
  2.0+ para atenção ~2-4x mais rápida (automático em CUDA).

ARQUITETURA GPT (de baixo pra cima):

1. Token Embedding (+ Position Embedding se não usar RoPE)
2. N× TransformerBlock (empilhados)
    ├── LayerNorm
    ├── MultiHeadSelfAttention (com RoPE + Flash Attention)
    │     (com residual)
    ├── LayerNorm
    └── FeedForward (com residual)
3. LayerNorm final + Linear head

REFERÊNCIAS:
- "Attention Is All You Need" (Vaswani et al., 2017)
- GPT-2 (Radford et al., 2019)
- RoFormer: "Rotary Position Embedding" (Su et al., 2021)
- Flash Attention: "Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
- nanoGPT de Karpathy: https://github.com/karpathy/nanoGPT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


# ──────────────────────────────────────────────────────────
# Rotary Position Embedding (RoPE)
# ──────────────────────────────────────────────────────────

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotaciona metade das dimensões: [-x2, x1] a partir de [x1, x2]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    AO INVÉS de somar um embedding posicional ao input, RoPE aplica
    uma rotação aos vetores Q e K proporcional à posição.

    INTUIÇÃO: tokens em posições próximas têm ângulos de rotação
    próximos → produto escalar (atenção) é maior. Tokens distantes
    têm rotações diferentes → atenção naturalmente decai com distância.

    MATEMÁTICA:
    Para posição pos e dimensão i:
      θ_i = 1 / (10000^(2i/d))
      q_rot = q * cos(pos·θ) + rotate_half(q) * sin(pos·θ)

    VANTAGEM sobre embedding aprendido:
    - Extrapola para sequências mais longas que o treino
    - Codifica distância relativa entre tokens (não absoluta)
    """

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        # Frequências inversas: 1 / (10000^(2i/d))
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Pré-computa cos/sin para todas as posições até seq_len."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(
        self, x: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        """
        Aplica rotação posicional a x.

        Args:
            x: tensor de shape (..., seq_len, dim)
            offset: deslocamento de posição (útil para KV-cache)
        """
        seq_len = x.shape[-2]
        cos = self.cos_cached[offset : offset + seq_len]  # (seq_len, dim)
        sin = self.sin_cached[offset : offset + seq_len]
        # Broadcast para (1, 1, seq_len, dim) se necessário
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return x * cos + _rotate_half(x) * sin


# ──────────────────────────────────────────────────────────
# Componentes básicos
# ──────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """
    Layer Normalization — normaliza pra média 0 e variância 1.

    Fórmula: y = (x - μ) / sqrt(σ² + ε) * γ + β
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

    NOVIDADES:
    - RoPE: rotação posicional aplicada a Q e K antes da atenção
    - Flash Attention: F.scaled_dot_product_attention quando disponível
      (PyTorch 2.0+), que usa memória O(N) ao invés de O(N²).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.use_rope = config.use_rope

        self.W_qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.W_out = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE (uma instância por bloco de atenção)
        if config.use_rope:
            self.rotary_emb = RotaryEmbedding(config.head_dim, config.context_len)
        else:
            self.rotary_emb = None

        # Flag para usar Flash Attention
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Projetar pra Q, K, V
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Aplicar RoPE a Q e K (ANTES da atenção)
        if self.use_rope and self.rotary_emb is not None:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        # Atenção
        if self._use_flash:
            # Flash Attention: O(N) em memória, automático em CUDA
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            # Fallback: atenção manual (CPU ou PyTorch < 2.0)
            attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            mascara_causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(
                mascara_causal, float("-inf")
            )
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            out = attn_probs @ v

        # Concatenar cabeças e projetar
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_out(out)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):
    """
    Feed-Forward Network — processa cada token individualmente.

    input → Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model) → output
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
    Bloco Transformer (Pre-Norm):

    x → LayerNorm → Attention → (+) → LayerNorm → FFN → (+) → saída
    |                               |                          |
    └──── residual ─────────────────┘──────────────────────────┘
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """
    GPT — Generative Pre-trained Transformer.

    MODELO COMPLETO:
    tokens → Token Embedding
           + Position Embedding (se NÃO usar RoPE)
           → Dropout → N× TransformerBlock → LayerNorm → Linear head → logits

    NOVIDADES:
    - RoPE: se use_rope=True, NÃO usa position_embedding
    - ignore_index=-100 no cross_entropy: suporta SFT (mascarar instrução)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        if config.use_rope:
            # RoPE: posição codificada na atenção, não via embedding
            self.position_embedding = None
        else:
            self.position_embedding = nn.Embedding(
                config.context_len, config.d_model
            )

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        # Weight tying: compartilha pesos entre embedding e classificação
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialização GPT-2: Normal(0, 0.02), camadas residuais com ganho menor."""
        n_layers = self.config.n_layers
        for nome, modulo in self.named_modules():
            if isinstance(modulo, nn.Linear):
                nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
                if modulo.bias is not None:
                    nn.init.zeros_(modulo.bias)
            elif isinstance(modulo, nn.Embedding):
                nn.init.normal_(modulo.weight, mean=0.0, std=0.02)

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
        Forward pass.

        Args:
            idx: IDs de tokens, shape (B, T)
            targets: IDs esperados, shape (B, T).
                     Posições com -100 são ignoradas (SFT masking).
        """
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)

        if self.position_embedding is not None:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = self.drop(tok_emb + self.position_embedding(pos))
        else:
            x = self.drop(tok_emb)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_log_probs(
        self, idx: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Retorna log-probabilidades por token (usado no DPO)."""
        logits, _ = self.forward(idx)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1)
        # Máscara: ignorar -100
        mask = targets != -100
        return (token_log_probs * mask).sum(dim=-1)

    def contar_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def resumo(self) -> str:
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
            f"  RoPE:           {'Sim' if c.use_rope else 'Não'}",
            f"  Flash Attn:     {'Disponível' if hasattr(F, 'scaled_dot_product_attention') else 'Não disponível'}",
            f"════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)
