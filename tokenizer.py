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
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path


TOKENS_ESPECIAIS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
# Os dois tokenizadores põem os tokens especiais primeiro, então os IDs são fixos
EOS_ID = TOKENS_ESPECIAIS.index("<EOS>")

# Pré-tokenização do BPE: palavras (com a pontuação grudada) e espaços
PADRAO_CHUNKS = re.compile(r"\S+|\s+")
# Chunks já codificados guardados pelo BPE. As palavras comuns se repetem
# muito, então o limite cobre quase todo o texto sem crescer sem fim
TAMANHO_CACHE = 500_000


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
        self._preparar_codificacao()

    def treinar(
        self, texto: str | Iterable[str], vocab_size: int = 500, cobertura: float = 1.0
    ) -> None:
        """
        Treina o BPE: iterativamente merge os pares mais frequentes.

        Args:
            texto: corpus de treino, ou vários textos (ex.: documentos)
            vocab_size: tamanho alvo do vocabulário
            cobertura: fração das ocorrências de caracteres que o vocabulário
                base cobre. Com 1.0 entram todos; com 0.9999 os mais raros
                (emoji, outros alfabetos) ficam de fora e viram <UNK>
        """
        # Pre-tokenizar: separar em palavras (não-espaço) e espaços
        freq_chunks: Counter[str] = Counter()
        for t in [texto] if isinstance(texto, str) else texto:
            freq_chunks.update(PADRAO_CHUNKS.findall(t))

        # Vocabulário base: tokens especiais + os caracteres que entram na cobertura
        freq_chars: Counter[str] = Counter()
        for chunk, n in freq_chunks.items():
            for c in chunk:
                freq_chars[c] += n
        total_chars = sum(freq_chars.values())
        chars_base: set[str] = set()
        acumulado = 0
        for c, n in freq_chars.most_common():
            if acumulado >= cobertura * total_chars:
                break
            chars_base.add(c)
            acumulado += n
        base_tokens = list(TOKENS_ESPECIAIS) + sorted(chars_base)

        self.vocab = {t: i for i, t in enumerate(base_tokens)}
        self.id_to_token = {i: t for i, t in enumerate(base_tokens)}
        self.merges = []

        # Cada chunk começa como sequência de caracteres. Os que têm algum
        # caractere fora do vocabulário base não entram nos merges
        palavras = [c for c in freq_chunks if chars_base.issuperset(c)]
        pecas = [list(p) for p in palavras]
        freqs = [freq_chunks[p] for p in palavras]

        # Contagem dos pares adjacentes, ponderada pela frequência do chunk, e
        # em quais chunks cada par aparece. Assim cada merge só recalcula os
        # chunks que contêm o par, em vez de recontar o corpus inteiro
        freq_pares: defaultdict[tuple[str, str], int] = defaultdict(int)
        onde: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
        for i, (ps, f) in enumerate(zip(pecas, freqs)):
            for par in zip(ps, ps[1:]):
                freq_pares[par] += f
                onde[par].add(i)

        def primeira_ocorrencia(par: tuple[str, str]) -> tuple[int, int]:
            # Desempate: o par que aparece primeiro no texto (chunk, posição)
            i = min(onde[par])
            return i, list(zip(pecas[i], pecas[i][1:])).index(par)

        # Loop de merges
        while len(self.vocab) < vocab_size and freq_pares:
            # Merge o par mais frequente
            maior = max(freq_pares.values())
            empatados = [p for p, n in freq_pares.items() if n == maior]
            melhor_par = min(empatados, key=primeira_ocorrencia)
            self.merges.append(melhor_par)
            novo_token = melhor_par[0] + melhor_par[1]
            # Dois merges diferentes podem formar o mesmo texto ("es"+"ta" e
            # "est"+"a"): o token já existe e mantém o ID
            if novo_token not in self.vocab:
                self.id_to_token[len(self.vocab)] = novo_token
                self.vocab[novo_token] = len(self.vocab)

            # Aplicar o merge nos chunks que têm o par e atualizar as contagens
            tocados: set[tuple[str, str]] = set()
            for i in list(onde[melhor_par]):
                antes = pecas[i]
                depois = self._merge_pair(antes, melhor_par)
                pares_antes = list(zip(antes, antes[1:]))
                pares_depois = list(zip(depois, depois[1:]))
                for par in pares_antes:
                    freq_pares[par] -= freqs[i]
                for par in pares_depois:
                    freq_pares[par] += freqs[i]
                for par in set(pares_antes).difference(pares_depois):
                    onde[par].discard(i)
                for par in pares_depois:
                    onde[par].add(i)
                tocados.update(pares_antes)
                pecas[i] = depois
            for par in tocados:
                if freq_pares[par] == 0:
                    del freq_pares[par], onde[par]

        self.vocab_size = len(self.vocab)
        self._preparar_codificacao()

    def _preparar_codificacao(self) -> None:
        """Monta a ordem dos merges e esvazia o cache de chunks já codificados."""
        self._ranks: dict[tuple[str, str], int] = {}
        for rank, par in enumerate(self.merges):
            self._ranks.setdefault(par, rank)
        self._cache: dict[str, list[int]] = {}

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

    def _codificar_chunk(self, chunk: str) -> list[int]:
        """
        Aplica os merges num chunk: junta sempre o par aprendido mais cedo
        que ainda aparece. Dá o mesmo resultado que aplicar os merges um por
        um, na ordem do treino, mas só olha os pares que o chunk tem.
        """
        pieces = list(chunk)
        sem_rank = len(self.merges)
        while len(pieces) > 1:
            par = min(zip(pieces, pieces[1:]), key=lambda p: self._ranks.get(p, sem_rank))
            if par not in self._ranks:
                break
            pieces = self._merge_pair(pieces, par)
        unk_id = self.vocab.get("<UNK>", 1)
        return [self.vocab.get(piece, unk_id) for piece in pieces]

    def codificar(self, texto: str) -> list[int]:
        """Converte texto em lista de IDs usando BPE."""
        ids: list[int] = []
        for chunk in PADRAO_CHUNKS.findall(texto):
            ids_chunk = self._cache.get(chunk)
            if ids_chunk is None:
                ids_chunk = self._codificar_chunk(chunk)
                if len(self._cache) < TAMANHO_CACHE:
                    self._cache[chunk] = ids_chunk
            ids.extend(ids_chunk)
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
        tok._preparar_codificacao()
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
