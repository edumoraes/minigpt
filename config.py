"""
config.py — Central de hiperparâmetros do MiniGPT

Aqui definimos TODOS os hiperparâmetros do modelo em um só lugar.
Isso facilita experimentar diferentes configurações sem precisar
mexer no código-fonte.

DICAS DIDÁTICAS:
- d_model: dimensão interna do modelo. GPT-2 small usa 768, nós usamos 128
  pq queremos rodar em CPU rapidamente.
- n_heads: número de cabeças de atenção. A dimensão por cabeça é
  d_model // n_heads (aqui: 128/4 = 32). Precisa ser divisível!
- n_layers: quantos blocos Transformer empilhamos. Mais camadas =
  mais capacidade, mas mais lento pra treinar.
- context_len: tamanho da janela de contexto. Quantos tokens o
  modelo consegue "olhar pra trás" de uma vez.
- dropout: probabilidade de "desligar" neurônios aleatoriamente
  durante o treino. Previne overfitting. 0.1 = 10%.
"""


from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- Dimensões do modelo ---
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3

    # --- Janela de contexto ---
    context_len: int = 64

    # --- Regularização ---
    dropout: float = 0.1

    # --- Treinamento ---
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 30

    # --- Otimizador ---
    beta1: float = 0.9
    beta2: float = 0.95

    # --- Gradient clipping ---
    max_grad_norm: float = 1.0

    # --- Taxa de aprendizado com warmup + cosine decay ---
    warmup_steps: int = 100
    min_lr: float = 1e-5

    @property
    def head_dim(self) -> int:
        """Dimensão de cada cabeça de atenção."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) deve ser divisível por "
            f"n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads