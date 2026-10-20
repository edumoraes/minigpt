"""
data/preparar.py — Texto real pro pré-treino: GigaVerbo-v2, subset edu_high

O corpus sintético tem ~150 mil tokens e poucos padrões de frase. Aqui o
modelo aprende com texto da web em português (FineWeb-2 e mC4) que o
GigaVerbo-v2 já classificou por qualidade educacional: o edu_high é a parte
mais didática, com 16,2M documentos em 56 arquivos parquet.

O pré-treino usa só o começo: cada arquivo tem 22 row groups de ~14 mil
documentos (~84M caracteres) e o HfFileSystem baixa só os row groups lidos.
Com o vocabulário de 8192 dá ~2,5 caracteres por token, então 150M tokens
saem de ~5 row groups (~250 MB) do primeiro arquivo.

SAÍDA (na pasta destino):
  tokenizer.json   BPE treinado nos primeiros documentos
  val.bin          primeiros documentos (1% dos tokens de treino)
  treino.bin       documentos seguintes
  meta.json        de onde vieram os dados e quantos tokens tem cada arquivo

Os .bin guardam os IDs dos tokens em uint16, com <EOS> no fim de cada documento.
"""

import json
import time
from array import array
from collections.abc import Iterator
from datetime import datetime
from itertools import chain
from pathlib import Path

from tokenizer import BPETokenizer, EOS_ID

REPO = "Polygl0t/gigaverbo-v2"
SUBSET = "edu_high"
FRACAO_VAL = 0.01
# Fração dos caracteres que o vocabulário base cobre: no edu_high isso deixa
# ~230 caracteres e o resto (emoji, outros alfabetos) vira <UNK>
COBERTURA_CHARS = 0.9999


def documentos() -> Iterator[str]:
    """Lê os textos do edu_high em ordem, um row group por vez."""
    # Já vêm instalados no Colab; localmente: pip install -r requirements.txt
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for caminho in sorted(fs.glob(f"datasets/{REPO}/{SUBSET}/*.parquet")):
        with fs.open(caminho) as arquivo:
            parquet = pq.ParquetFile(arquivo)
            for i in range(parquet.num_row_groups):
                yield from parquet.read_row_group(i, columns=["text"]).column("text").to_pylist()


def preparar(
    destino: str = "dados",
    n_tokens: int = 150_000_000,
    vocab_size: int = 8192,
    chars_tokenizer: int = 10_000_000,
) -> None:
    """
    Treina o tokenizer e grava os tokens de treino e validação em destino/.

    Args:
        n_tokens: tokens de treino (a validação leva 1% a mais)
        vocab_size: tamanho do vocabulário BPE (até 65536, o limite do uint16)
        chars_tokenizer: caracteres usados pra treinar o BPE (~40 s com 10M)
    """
    if vocab_size > 2**16:
        raise ValueError(f"vocab_size {vocab_size} não cabe em uint16 (máximo 65536)")
    pasta = Path(destino)
    pasta.mkdir(parents=True, exist_ok=True)
    # O meta.json é gravado por último e marca que os dados estão completos
    (pasta / "meta.json").unlink(missing_ok=True)
    inicio = time.time()
    docs = documentos()

    # ── Tokenizer: treinado nos primeiros documentos ──
    amostra: list[str] = []
    n_chars = 0
    for doc in docs:
        amostra.append(doc)
        n_chars += len(doc)
        if n_chars >= chars_tokenizer:
            break
    print(f"Treinando BPE (vocab {vocab_size}) em {len(amostra):,} documentos "
          f"({n_chars / 1e6:.0f}M caracteres) do {REPO} ({SUBSET})...")
    tokenizer = BPETokenizer()
    tokenizer.treinar(amostra, vocab_size=vocab_size, cobertura=COBERTURA_CHARS)
    tokenizer.salvar(str(pasta / "tokenizer.json"))
    print(f"  {tokenizer} em {time.time() - inicio:.0f}s")

    # ── Tokens: validação com os primeiros documentos, treino com os seguintes ──
    n_val = int(n_tokens * FRACAO_VAL)
    alvo = n_val + n_tokens
    tokens = {"val": 0, "treino": 0}
    n_docs = {"val": 0, "treino": 0}
    n_chars = 0
    t0 = ultimo_progresso = time.time()
    print(f"Tokenizando {n_val:,} tokens de validação + {n_tokens:,} de treino...")
    arquivos = {nome: open(pasta / f"{nome}.bin", "wb") for nome in tokens}
    try:
        for doc in chain(amostra, docs):
            nome = "val" if tokens["val"] < n_val else "treino"
            ids = tokenizer.codificar(doc)
            ids.append(EOS_ID)
            array("H", ids).tofile(arquivos[nome])
            tokens[nome] += len(ids)
            n_docs[nome] += 1
            n_chars += len(doc)

            if time.time() - ultimo_progresso >= 30:
                ultimo_progresso = time.time()
                feitos = tokens["val"] + tokens["treino"]
                ritmo = feitos / (ultimo_progresso - t0)
                print(f"  {feitos / 1e6:,.1f}M de {alvo / 1e6:,.0f}M tokens │ "
                      f"{ritmo:,.0f} tokens/s │ faltam ~{(alvo - feitos) / ritmo / 60:.0f} min")
            if tokens["treino"] >= n_tokens:
                break
    finally:
        for arquivo in arquivos.values():
            arquivo.close()

    total = tokens["val"] + tokens["treino"]
    meta = {
        "dataset": f"{REPO} ({SUBSET})",
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "vocab_size": tokenizer.vocab_size,
        "tokens": tokens,
        "documentos": n_docs,
        "caracteres_por_token": round(n_chars / total, 2),
        "tempo_seg": round(time.time() - inicio, 1),
    }
    (pasta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Pronto em {meta['tempo_seg'] / 60:.1f} min: {tokens['treino']:,} tokens de treino "
          f"({n_docs['treino']:,} documentos) e {tokens['val']:,} de validação em {pasta}/")
