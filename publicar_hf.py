#!/usr/bin/env python3
"""
publicar_hf.py — Publica o modelo pré-treinado no Hugging Face e cria um Space pra testar

1. converte o checkpoint (output/melhor_modelo.pt) em model.safetensors e
   config.json: o .pt é um pickle, que o Hub marca como inseguro de carregar
2. sobe pro repositório <usuario>/<nome> os pesos, o tokenizer, um model card
   gerado do log_treino.json e o código que carrega o modelo
3. cria ou atualiza o Space estático <usuario>/<nome> com space/: uma página
   que baixa os pesos do repositório do modelo e gera texto no navegador de
   quem acessa, com o modelo reescrito em JavaScript (space/minigpt.js).
   Spaces estáticos são grátis; Gradio e Docker na CPU grátis exigem o PRO

Os dois repositórios são públicos. Rodar de novo atualiza os dois com o
checkpoint atual.

USO:
  python publicar_hf.py --nome devaneio
  python publicar_hf.py --nome devaneio --pasta /tmp/devaneio   # só monta os arquivos, sem subir

PRÉ-REQUISITO (uma vez):
  hf auth login   # com um token de escrita (huggingface.co/settings/tokens)
"""

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_model

from data.preparar import REPO as DATASET, SUBSET
from model import GPTModel

RAIZ = Path(__file__).resolve().parent
# Módulos que carregam o modelo e geram texto em Python: vão pro repositório do modelo
CODIGO = ["config.py", "model.py", "tokenizer.py", "generate.py"]
# A página e o modelo em JavaScript: vão pro Space
PAGINA = ["index.html", "minigpt.js", "worker.js"]
GITHUB = "https://github.com/edumoraes/minigpt"


def numero(valor: float, casas: int = 0) -> str:
    """Número no formato brasileiro: 150.061.312 e 2,65."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def model_card(repo_id: str, config, log: dict) -> str:
    """Model card com os números do log de treino."""
    val_loss = log["melhor_loss"]
    accum = config.gradient_accum_steps
    return f"""---
license: apache-2.0
language:
- pt
datasets:
- {DATASET}
pipeline_tag: text-generation
library_name: pytorch
tags:
- gpt
- from-scratch
---

# {repo_id.split("/")[1]}

GPT pequeno ({numero(log["n_params"] / 1e6, 1)}M parâmetros) treinado do zero em português com o
[MiniGPT]({GITHUB}), uma implementação de GPT feita pra estudo. Escreve com a gramática e
o vocabulário do português, mas não guarda fatos: o texto sai fluente e sem compromisso
com a verdade.

Pra testar no navegador: [Space {repo_id}](https://huggingface.co/spaces/{repo_id}). Ele roda o modelo
em JavaScript, no navegador de quem acessa, com os pesos deste repositório.

## Treino

| | |
|---|---|
| Dados | o começo do subset `{SUBSET}` do [GigaVerbo-v2](https://huggingface.co/datasets/{DATASET}): texto da web em português (FineWeb-2 e mC4) filtrado por qualidade educacional |
| Tokens vistos | {numero(log["total_tokens_treinados"])} ({len(log["epocas"])} época) |
| Otimizador | AdamW, lr {config.learning_rate:.0e} com warmup de {config.warmup_steps} passos e cosine até {config.min_lr:.0e}, weight decay {numero(config.weight_decay, 1)} |
| Batch | {config.batch_size * accum} sequências de {config.context_len} tokens |
| Precisão | {log.get("precisao", "fp32")} |
| Tempo | {numero(log["tempo_total_min"], 1)} min em {"GPU" if log["device"] == "cuda" else log["device"]}, {numero(log["tokens_por_segundo"])} tokens/s |
| Val loss | {numero(val_loss, 3)} (perplexidade {numero(math.exp(val_loss), 1)}) |

## Arquitetura

- Decoder-only com {config.n_layers} blocos pre-norm, d_model {config.d_model}, {config.n_heads} cabeças de atenção e FFN de {4 * config.d_model} com GELU
- RoPE nas cabeças de atenção, `scaled_dot_product_attention` causal
- Weight tying entre o embedding e a camada de saída
- Contexto de {config.context_len} tokens
- Tokenizer BPE próprio com vocabulário de {numero(config.vocab_size)}. Os espaços são tokens separados, o que dá ~2,5 caracteres por token

## Como usar

O modelo usa o código do MiniGPT, que vem junto neste repositório:

```python
import json, sys
from huggingface_hub import snapshot_download
from safetensors.torch import load_model

pasta = snapshot_download("{repo_id}")
sys.path.insert(0, pasta)
from config import GPTConfig
from generate import gerar
from model import GPTModel
from tokenizer import carregar_tokenizer

modelo = GPTModel(GPTConfig(**json.load(open(f"{{pasta}}/config.json"))))
load_model(modelo, f"{{pasta}}/model.safetensors")
tokenizer = carregar_tokenizer(f"{{pasta}}/tokenizer.json")
print(gerar(modelo, tokenizer, "A chuva", max_tokens=100))
```

## Limitações

- É pequeno: inventa fatos e perde o fio depois de algumas frases.
- Só tem pré-treino: continua textos, mas não segue instruções nem responde perguntas.
- Mistura português do Brasil e de Portugal, como os dados de treino.
- Pode reproduzir vieses e erros do texto da web em que foi treinado.
- Caracteres raros (emoji, outros alfabetos) viram `<UNK>` e somem do texto gerado.

## Licença

Apache-2.0 para os pesos e o código. Os dados de treino seguem as licenças das fontes do
GigaVerbo-v2, listadas no card do dataset.
"""


def space_card(repo_id: str, n_params: int) -> str:
    """README do Space: o cabeçalho diz ao Hub que é uma página estática."""
    return f"""---
title: {repo_id.split("/")[1]}
emoji: 💭
colorFrom: indigo
colorTo: pink
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
models:
- {repo_id}
short_description: GPT de {numero(n_params / 1e6, 1)}M parâmetros treinado do zero em português
---

Continua textos em português com o modelo [{repo_id}](https://huggingface.co/{repo_id}).
"""


def montar(checkpoint: Path, repo_id: str, destino: Path) -> tuple[Path, Path]:
    """Monta as pastas do repositório do modelo e do Space em destino/."""
    # O checkpoint sai do train.py daqui e guarda o GPTConfig em pickle (ver main.py)
    dados = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "tipo" in dados:
        sys.exit(f"{checkpoint} é um checkpoint de {dados['tipo']}; o model card descreve o pré-treino")
    log = json.loads((checkpoint.parent / "log_treino.json").read_text())
    config = dados["config"]

    pasta_modelo = destino / "modelo"
    pasta_modelo.mkdir(parents=True)
    modelo = GPTModel(config)
    modelo.load_state_dict(dados["modelo"])
    # save_model guarda uma vez só os pesos compartilhados pelo weight tying
    save_model(modelo, str(pasta_modelo / "model.safetensors"))
    (pasta_modelo / "config.json").write_text(json.dumps(asdict(config), indent=2))
    shutil.copy(checkpoint.parent / "tokenizer.json", pasta_modelo)
    (pasta_modelo / "README.md").write_text(model_card(repo_id, config, log))

    for arquivo in CODIGO:
        shutil.copy(RAIZ / arquivo, pasta_modelo)

    pasta_space = destino / "space"
    pasta_space.mkdir()
    for arquivo in PAGINA:
        shutil.copy(RAIZ / "space" / arquivo, pasta_space)
    # A página baixa os pesos de https://huggingface.co/<repo_id>
    pagina = pasta_space / "index.html"
    pagina.write_text(pagina.read_text().replace("__REPO_MODELO__", repo_id))
    (pasta_space / "README.md").write_text(space_card(repo_id, log["n_params"]))

    print(f"  {modelo.contar_parametros():,} parâmetros, val loss {log['melhor_loss']:.4f}")
    return pasta_modelo, pasta_space


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publica o modelo pré-treinado no Hugging Face e cria um Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USO:")[1],
    )
    parser.add_argument("--nome", required=True, help="Nome do repositório do modelo e do Space")
    parser.add_argument("--checkpoint", default="output/melhor_modelo.pt",
                        help="Checkpoint de pré-treino (padrão: output/melhor_modelo.pt)")
    parser.add_argument("--pasta", help="Só monta os arquivos nesta pasta, sem subir nada")
    args = parser.parse_args()

    api = HfApi()
    try:
        usuario = api.whoami()["name"]
    except Exception:
        if not args.pasta:
            sys.exit("Sem login no Hugging Face. Rode antes: hf auth login (com um token de escrita)")
        usuario = "SEU_USUARIO"
    repo_id = f"{usuario}/{args.nome}"

    print(f"[1/3] Convertendo {args.checkpoint}")
    if args.pasta:
        destino = Path(args.pasta)
        if destino.exists():
            sys.exit(f"{destino} já existe")
        montar(RAIZ / args.checkpoint, repo_id, destino)
        print(f"Arquivos em {destino}/ (nada foi enviado). Pra testar a página com os pesos locais:")
        print(f"  python -m http.server -d {destino}   # e abra http://localhost:8000/space/?modelo=../modelo")
        return 0

    with tempfile.TemporaryDirectory(prefix="minigpt-hf-") as tmp:
        pasta_modelo, pasta_space = montar(RAIZ / args.checkpoint, repo_id, Path(tmp))
        mensagem = f"Checkpoint de {date.today():%d/%m/%Y}"

        print(f"[2/3] Modelo: https://huggingface.co/{repo_id}")
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=pasta_modelo, commit_message=mensagem)

        print(f"[3/3] Space: https://huggingface.co/spaces/{repo_id}")
        api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
        # Apaga a página antiga, incluindo o style.css do template que o Hub cria
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=pasta_space,
                          commit_message=mensagem, delete_patterns=["*.html", "*.js", "*.css"])
    print("Pronto. O Space estático fica no ar em poucos segundos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
