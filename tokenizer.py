"""
tokenizer.py — Tokenizador char-level do MiniGPT

O QUE É UM TOKENIZER?
Um tokenizer converte texto em números (tokens) que o modelo consegue
processar. Modelos reais usam tokenizers como BPE (Byte Pair Encoding)
que quebram texto em pedaços de palavras. Nós usamos algo muito mais
simples: char-level (cada caractere = 1 token).

POR QUÊ CHAR-LEVEL?
- Didático: você entende exatamente o que está acontecendo
- Simples: zero dependências externas
- Funciona: pra um modelo pequeno, aprender caractere por caractere
  já demonstra os conceitos fundamentais

OS TOKENS ESPECIAIS:
- <PAD>: usado pra preencher sequências menores que o batch
- <UNK>: caractere desconhecido (não deveria aparecer no nosso caso)
- <BOS>: Beginning Of Sequence — marca o início de uma sequência
- <EOS>: End Of Sequence — marca o fim de uma sequência
"""

import json
from pathlib import Path
from typing import Optional


TOKENS_ESPECIAIS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


class CharTokenizer:
    """
    Tokenizador char-level.
    
    Cada caractere único no corpus vira um token.
    Mapeamento: caractere <-> inteiro.
    
    Vantagem: vocabulário pequeno, fácil de entender.
    Desvantagem: sequências ficam longas (1 palavra = N tokens).
    """

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.vocab_size: int = 0

    def treinar(self, texto: str) -> None:
        """
        Constrói o vocabulário a partir do texto de treino.
        
        Passos:
        1. Encontra todos os caracteres únicos no texto
        2. Reserva IDs 0-3 pros tokens especiais
        3. Atribui IDs sequenciais pros caracteres normais
        """
        chars_unicos = sorted(set(texto))

        self.char_to_id = {}
        self.id_to_char = {}

        # Primeiro os tokens especiais
        for i, token in enumerate(TOKENS_ESPECIAIS):
            self.char_to_id[token] = i
            self.id_to_char[i] = token

        # Depois os caracteres reais
        for i, ch in enumerate(chars_unicos, start=len(TOKENS_ESPECIAIS)):
            self.char_to_id[ch] = i
            self.id_to_char[i] = ch

        self.vocab_size = len(self.char_to_id)

    def codificar(self, texto: str) -> list[int]:
        """
        Converte texto em lista de IDs inteiros.
        
        Caracteres desconhecidos viram <UNK> (ID 1).
        """
        return [
            self.char_to_id.get(ch, self.char_to_id["<UNK>"])
            for ch in texto
        ]

    def decodificar(self, ids: list[int]) -> str:
        """
        Converte lista de IDs inteiros de volta pra texto.
        
        Tokens especiais são ignorados na saída.
        """
        chars = []
        for id_ in ids:
            ch = self.id_to_char.get(id_, "")
            if ch in TOKENS_ESPECIAIS:
                continue
            chars.append(ch)
        return "".join(chars)

    def salvar(self, caminho: str) -> None:
        """Salva o vocabulário em JSON pra reuso."""
        dados = {
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
            "vocab_size": self.vocab_size,
        }
        Path(caminho).write_text(json.dumps(dados, ensure_ascii=False, indent=2))

    @classmethod
    def carregar(cls, caminho: str) -> "CharTokenizer":
        """Carrega um vocabulário salvo em JSON."""
        dados = json.loads(Path(caminho).read_text())
        tok = cls()
        tok.char_to_id = dados["char_to_id"]
        tok.id_to_char = {int(k): v for k, v in dados["id_to_char"].items()}
        tok.vocab_size = dados["vocab_size"]
        return tok

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"