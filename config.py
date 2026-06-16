"""
config.py — Central de hiperparâmetros do MiniGPT

Todos os hiperparâmetros do modelo em um só lugar.
Facilita experimentar diferentes configurações sem mexer no código-fonte.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- Dimensões do modelo (aumentadas!) ---
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2

    # --- Janela de contexto (aumentada!) ---
    context_len: int = 256

    # --- Regularização ---
    dropout: float = 0.3

    # --- Treinamento ---
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.3
    max_epochs: int = 30

    # --- Otimizador ---
    beta1: float = 0.9
    beta2: float = 0.95

    # --- Gradient clipping ---
    max_grad_norm: float = 1.0

    # --- LR scheduler: warmup + cosine decay ---
    warmup_steps: int = 200
    min_lr: float = 1e-5

    # --- NOVO: Gradient accumulation ---
    # Simula batch_size maior sem estourar VRAM.
    # batch_size_efetivo = batch_size * gradient_accum_steps
    gradient_accum_steps: int = 1

    # --- NOVO: Validação e early stopping ---
    val_split: float = 0.1
    patience: int = 8

    # --- NOVO: Estratégias de geração ---
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    typical_mass: float = 0.9

    # --- NOVO: Tokenizador ---
    tokenizer_type: str = "bpe"  # "bpe" ou "char"
    bpe_vocab_size: int = 500

    # --- NOVO: Rotary Position Embedding (RoPE) ---
    # Substitui position_embedding aprendido por embeddings rotacionais.
    # Generaliza melhor para sequências mais longas.
    use_rope: bool = True

    # --- NOVO: Beam search ---
    # beam_width=1 desativado (decodificação greedy/sample normal)
    beam_width: int = 1

    # --- NOVO: SFT (Supervised Fine-Tuning) ---
    sft_lr: float = 1e-5
    sft_epochs: int = 5

    # --- NOVO: DPO (Direct Preference Optimization) ---
    dpo_beta: float = 0.1
    dpo_lr: float = 1e-6
    dpo_epochs: int = 3

    # --- Setado dinamicamente no treino ---
    vocab_size: int = 0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) deve ser divisível por "
            f"n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads
