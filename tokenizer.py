"""
tokenizer.py — Tokenizadores do MiniGPT

Dois tokenizadores disponíveis:

1. CharTokenizer: cada caractere = 1 token (original, didático)
2. BPETokenizer: Byte Pair Encoding — merge de pares frequentes

POR QUÊ BPE?
- Sequências ficam 3-5x mais curtas que char-level
- Tokens capturam subpalavras comuns ("ção", "que", "mente")
- Mesmo vocabulário pequeno (500) já cobre bem o português
- É o padrão da indústria (GPT-2/3/4 usam BPE)

COMO O BPE FUNCIONA:
1. Começa com vocabulário de caracteres individuais
2. Conta frequência de todos os pares adjacentes no corpus
3. Merge o par mais frequente num novo token
4. Repete até atingir vocab_size desejado

Exemplo com "gato gato":
  ["g","a","t","o"," ","g","a","t","o"]
  → merge ("g","a"): ["ga","t","o"," ","ga","t","o"]
  → merge ("t","o"): ["ga","to"," ","ga","to"]
  → merge ("ga","to"): ["gato"," ","gato"]
"""

import json
import re
from collections import Counter
from pathlib import Path


TOKENS_ESPECIAIS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


# ──────────────────────────────────────────────────────────
# CharTokenizer (original, mantido para compatibilidade)
# ──────────────────────────────────────────────────────────

class CharTokenizer:
    """Tokenizador char-level. Cada caractere = 1 token."""

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.vocab_size: int = 0

    def treinar(self, texto: str) -> None:
        chars_unicos = sorted(set(texto))
        self.char_to_id = {}
        self.id_to_char = {}
        for i, token in enumerate(TOKENS_ESPECIAIS):
            self.char_to_id[token] = i
            self.id_to_char[i] = token
        for i, ch in enumerate(chars_unicos, start=len(TOKENS_ESPECIAIS)):
            self.char_to_id[ch] = i
            self.id_to_char[i] = ch
        self.vocab_size = len(self.char_to_id)

    def codificar(self, texto: str) -> list[int]:
        return [
            self.char_to_id.get(ch, self.char_to_id["<UNK>"])
            for ch in texto
        ]

    def decodificar(self, ids: list[int]) -> str:
        chars = []
        for id_ in ids:
            ch = self.id_to_char.get(id_, "")
            if ch in TOKENS_ESPECIAIS:
                continue
            chars.append(ch)
        return "".join(chars)

    def salvar(self, caminho: str) -> None:
        dados = {
            "tipo": "char",
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
            "vocab_size": self.vocab_size,
        }
        Path(caminho).write_text(json.dumps(dados, ensure_ascii=False, indent=2))

    @classmethod
    def carregar(cls, caminho: str) -> "CharTokenizer":
        dados = json.loads(Path(caminho).read_text())
        tok = cls()
        tok.char_to_id = dados["char_to_id"]
        tok.id_to_char = {int(k): v for k, v in dados["id_to_char"].items()}
        tok.vocab_size = dados["vocab_size"]
        return tok

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


# ──────────────────────────────────────────────────────────
# BPETokenizer (novo, padrão)
# ──────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Tokenizador BPE (Byte Pair Encoding).

    Treina merges a partir do corpus e depois aplica na codificação.
    Pre-tokeniza por palavras (separa espaços como tokens próprios)
    pra que o BPE nunca mescle através de fronteiras de palavras.
    """

    def __init__(self):
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.vocab_size: int = 0

    def treinar(self, texto: str, vocab_size: int = 500) -> None:
        """
        Treina o BPE: iterativamente merge os pares mais frequentes.

        Args:
            texto: corpus de treino
            vocab_size: tamanho alvo do vocabulário
        """
        # Pre-tokenizar: separar em palavras (não-espaço) e espaços
        chunks = re.findall(r"\S+|\s+", texto)
        freq_chunks = Counter(chunks)

        # Inicializar: cada chunk como sequência de caracteres
        chunk_splits: dict[str, list[str]] = {c: list(c) for c in freq_chunks}

        # Vocabulário base: tokens especiais + todos os caracteres únicos
        todos_chars = sorted(
            set(c for pieces in chunk_splits.values() for c in pieces)
        )
        base_tokens = list(TOKENS_ESPECIAIS) + todos_chars

        self.vocab = {t: i for i, t in enumerate(base_tokens)}
        self.id_to_token = {i: t for i, t in enumerate(base_tokens)}
        self.merges = []

        # Loop de merges
        while len(self.vocab) < vocab_size:
            # Contar pares adjacentes ponderados pela frequência do chunk
            pair_freq: Counter[tuple[str, str]] = Counter()
            for chunk, count in freq_chunks.items():
                pieces = chunk_splits[chunk]
                for i in range(len(pieces) - 1):
                    pair_freq[(pieces[i], pieces[i + 1])] += count

            if not pair_freq:
                break

            # Merge o par mais frequente
            melhor_par = pair_freq.most_common(1)[0][0]
            self.merges.append(melhor_par)
            novo_token = melhor_par[0] + melhor_par[1]
            self.vocab[novo_token] = len(self.vocab)
            self.id_to_token[len(self.id_to_token)] = novo_token

            # Aplicar merge em todos os chunks
            for chunk in chunk_splits:
                chunk_splits[chunk] = self._merge_pair(
                    chunk_splits[chunk], melhor_par
                )

        self.vocab_size = len(self.vocab)

    @staticmethod
    def _merge_pair(
        pieces: list[str], par: tuple[str, str]
    ) -> list[str]:
        """Aplica um merge num lista de pieces (esquerda→direita, não-sobreposto)."""
        resultado = []
        i = 0
        while i < len(pieces):
            if (
                i < len(pieces) - 1
                and pieces[i] == par[0]
                and pieces[i + 1] == par[1]
            ):
                resultado.append(par[0] + par[1])
                i += 2
            else:
                resultado.append(pieces[i])
                i += 1
        return resultado

    def codificar(self, texto: str) -> list[int]:
        """Converte texto em lista de IDs usando BPE."""
        if not texto:
            return []
        chunks = re.findall(r"\S+|\s+", texto)
        ids: list[int] = []
        unk_id = self.vocab.get("<UNK>", 1)
        for chunk in chunks:
            pieces = list(chunk)
            for merge in self.merges:
                pieces = self._merge_pair(pieces, merge)
            for piece in pieces:
                ids.append(self.vocab.get(piece, unk_id))
        return ids

    def decodificar(self, ids: list[int]) -> str:
        """Converte lista de IDs de volta pra texto."""
        tokens = []
        for id_ in ids:
            t = self.id_to_token.get(id_, "")
            if t in TOKENS_ESPECIAIS:
                continue
            tokens.append(t)
        return "".join(tokens)

    def salvar(self, caminho: str) -> None:
        """Salva merges e vocabulário em JSON."""
        dados = {
            "tipo": "bpe",
            "merges": [[a, b] for a, b in self.merges],
            "vocab": self.vocab,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "vocab_size": self.vocab_size,
        }
        Path(caminho).write_text(json.dumps(dados, ensure_ascii=False, indent=2))

    @classmethod
    def carregar(cls, caminho: str) -> "BPETokenizer":
        """Carrega tokenizer BPE salvo em JSON."""
        dados = json.loads(Path(caminho).read_text())
        tok = cls()
        tok.merges = [tuple(m) for m in dados["merges"]]
        tok.vocab = dados["vocab"]
        tok.id_to_token = {int(k): v for k, v in dados["id_to_token"].items()}
        tok.vocab_size = dados["vocab_size"]
        return tok

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


# ──────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────

def criar_tokenizer(tipo: str = "bpe"):
    """Cria tokenizer do tipo especificado ('bpe' ou 'char')."""
    if tipo == "bpe":
        return BPETokenizer()
    return CharTokenizer()


def carregar_tokenizer(caminho: str) -> CharTokenizer | BPETokenizer:
    """Carrega tokenizer de arquivo, detectando o tipo automaticamente."""
    dados = json.loads(Path(caminho).read_text())
    tipo = dados.get("tipo", "char")
    if tipo == "bpe":
        return BPETokenizer.carregar(caminho)
    return CharTokenizer.carregar(caminho)
