"""
main.py — CLI interativa do MiniGPT

COMANDOS:
  python main.py train     # Pré-treinamento
  python main.py sft       # Supervised Fine-Tuning
  python main.py dpo       # Direct Preference Optimization
  python main.py chat      # Chat interativo
  python main.py sample    # Gerar amostras
  python main.py info      # Info do modelo salvo

FLUXO COMPLETO:
  1. train  → pré-treina no corpus (salva melhor_modelo.pt)
  2. sft    → fine-tune com instruções (salva melhor_modelo_sft.pt)
  3. dpo    → alinha com preferências (salva melhor_modelo_dpo.pt)
  4. chat   → conversa com modelo treinado
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from config import GPTConfig
from generate import gerar, gerar_variacoes, gerar_beam_search
from model import GPTModel
from tokenizer import carregar_tokenizer, CharTokenizer, BPETokenizer
from train import treinar


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def carregar_modelo(
    caminho_modelo: str = "output/melhor_modelo.pt",
    caminho_tokenizer: str = "output/tokenizer.json",
    device: str | None = None,
) -> tuple[GPTModel, CharTokenizer | BPETokenizer, str]:
    """Carrega modelo e tokenizer salvos. Lida com retrocompatibilidade."""
    if device is None:
        device = _detect_device()

    checkpoint = torch.load(caminho_modelo, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Retrocompatibilidade: checkpoints antigos podem não ter novos campos
    if not hasattr(config, "use_rope"):
        config.use_rope = True
    if not hasattr(config, "tokenizer_type"):
        config.tokenizer_type = "char"
    if not hasattr(config, "bpe_vocab_size"):
        config.bpe_vocab_size = 500
    if not hasattr(config, "gradient_accum_steps"):
        config.gradient_accum_steps = 1
    if not hasattr(config, "val_split"):
        config.val_split = 0.1
    if not hasattr(config, "patience"):
        config.patience = 5

    # Se checkpoint tem position_embedding, modelo usava embeddings aprendidos
    state_dict = checkpoint["modelo"]
    if "position_embedding.weight" in state_dict:
        config.use_rope = False

    tokenizer = carregar_tokenizer(caminho_tokenizer)

    modelo = GPTModel(config).to(device)
    modelo.load_state_dict(state_dict, strict=False)
    modelo.eval()

    return modelo, tokenizer, device


def cmd_train(args):
    """Pré-treinamento no corpus."""
    from data.corpus import carregar_corpus, carregar_corpus_externo

    config = GPTConfig()

    # Sobrescrever config com argumentos do CLI
    if args.corpus:
        texto = carregar_corpus_externo(args.corpus)
    else:
        texto = carregar_corpus()

    if args.d_model:
        config.d_model = args.d_model
    if args.n_layers:
        config.n_layers = args.n_layers
    if args.context_len:
        config.context_len = args.context_len
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.accum:
        config.gradient_accum_steps = args.accum
    if args.tokenizer:
        config.tokenizer_type = args.tokenizer
    if args.no_rope:
        config.use_rope = False

    print(f"\nConfiguração:")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}")
    print(f"  context_len={config.context_len}, batch_size={config.batch_size}")
    print(f"  lr={config.learning_rate}, epochs={config.max_epochs}")
    print(f"  tokenizer={config.tokenizer_type}, RoPE={config.use_rope}")
    print(f"  gradient_accum={config.gradient_accum_steps}\n")

    modelo, tokenizer = treinar(config, texto, saida_dir="output")

    # Demo pós-treino
    print("\n" + "=" * 60)
    print("GERAÇÃO PÓS-TREINO:")
    print("=" * 60)
    for prompt in ["O gato", "A menina", "Na cidade", "O professor"]:
        resultado = gerar(modelo, tokenizer, prompt, max_tokens=80, device=_detect_device())
        print(f"\nPrompt: '{prompt}'")
        print(f"Saída:  {resultado}")


def cmd_sft(args):
    """Supervised Fine-Tuning."""
    from sft import treinar_sft
    from data.corpus import gerar_dados_sft

    try:
        modelo, tokenizer, device = carregar_modelo()
    except FileNotFoundError:
        print("Modelo pré-treinado não encontrado! Treine primeiro: python main.py train")
        return

    config = modelo.config
    dados = gerar_dados_sft()

    if args.epochs:
        config.sft_epochs = args.epochs
    if args.lr:
        config.sft_lr = args.lr

    modelo = treinar_sft(modelo, tokenizer, config, dados, saida_dir="output", device=device)

    # Demo
    print("\n" + "=" * 60)
    print("DEMO SFT — Perguntas:")
    print("=" * 60)
    perguntas = ["O que é o sol?", "O que os gatos gostam?", "Como é a praia?"]
    for p in perguntas:
        resultado = gerar(modelo, tokenizer, f"Instrução: {p} Resposta: ", max_tokens=60, device=device)
        print(f"\nPergunta: {p}")
        print(f"Resposta: {resultado}")


def cmd_dpo(args):
    """Direct Preference Optimization."""
    from dpo import treinar_dpo
    from data.corpus import gerar_dados_dpo

    # Carregar modelo (preferir SFT se disponível)
    modelo_path = "output/melhor_modelo_sft.pt"
    if not Path(modelo_path).exists():
        modelo_path = "output/melhor_modelo.pt"

    try:
        modelo, tokenizer, device = carregar_modelo(caminho_modelo=modelo_path)
    except FileNotFoundError:
        print("Modelo não encontrado! Treine primeiro: python main.py train")
        return

    config = modelo.config
    dados = gerar_dados_dpo()

    if args.epochs:
        config.dpo_epochs = args.epochs
    if args.beta:
        config.dpo_beta = args.beta

    modelo = treinar_dpo(modelo, tokenizer, config, dados, saida_dir="output", device=device)


def cmd_chat(args):
    """Chat interativo com o modelo."""
    # Procurar melhor modelo disponível
    modelo_path = "output/melhor_modelo_dpo.pt"
    if not Path(modelo_path).exists():
        modelo_path = "output/melhor_modelo_sft.pt"
    if not Path(modelo_path).exists():
        modelo_path = "output/melhor_modelo.pt"

    try:
        modelo, tokenizer, device = carregar_modelo(caminho_modelo=modelo_path)
    except FileNotFoundError:
        print("Modelo não encontrado! Treine primeiro: python main.py train")
        return

    print(modelo.resumo())
    print(f"Modelo: {modelo_path}")
    print("\nMiniGPT Chat — 'sair' pra encerrar, 'config' pra ajustar parâmetros")
    print("=" * 60)

    temperatura = 0.8
    top_k = 40
    top_p = 0.9
    repetition_penalty = 1.2
    max_tokens = 200

    while True:
        try:
            user_input = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté logo!")
            break

        if not user_input:
            continue
        if user_input.lower() == "sair":
            print("Até logo!")
            break
        if user_input.lower() == "config":
            print(f"Temperatura: {temperatura}")
            print(f"Top-k: {top_k}")
            print(f"Top-p: {top_p}")
            print(f"Repetition penalty: {repetition_penalty}")
            print(f"Max tokens: {max_tokens}")
            try:
                v = input(f"Nova temperatura [{temperatura}]: ").strip()
                if v:
                    temperatura = float(v)
                v = input(f"Novo top-k [{top_k}]: ").strip()
                if v:
                    top_k = int(v)
                v = input(f"Novo top-p [{top_p}]: ").strip()
                if v:
                    top_p = float(v)
                v = input(f"Novo repetition penalty [{repetition_penalty}]: ").strip()
                if v:
                    repetition_penalty = float(v)
                v = input(f"Novo max tokens [{max_tokens}]: ").strip()
                if v:
                    max_tokens = int(v)
            except ValueError:
                print("Valor inválido, mantendo configuração atual.")
            continue

        resultado = gerar(
            modelo, tokenizer, user_input,
            max_tokens=max_tokens,
            temperature=temperatura,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        )
        print(f"\nMiniGPT: {resultado}")


def cmd_sample(args):
    """Gera amostras de texto."""
    modelo_path = "output/melhor_modelo_dpo.pt"
    if not Path(modelo_path).exists():
        modelo_path = "output/melhor_modelo_sft.pt"
    if not Path(modelo_path).exists():
        modelo_path = "output/melhor_modelo.pt"

    try:
        modelo, tokenizer, device = carregar_modelo(caminho_modelo=modelo_path)
    except FileNotFoundError:
        print("Modelo não encontrado! Treine primeiro: python main.py train")
        return

    prompts = [
        "O gato", "A menina", "Na cidade", "O professor",
        "Era uma vez", "A casa", "O dia", "A chuva",
    ]

    print(modelo.resumo())
    print()
    print("Amostras (temp=0.8, top_k=40, top_p=0.9, rep_penalty=1.2):")
    print("=" * 60)

    for prompt in prompts:
        variacoes = gerar_variacoes(
            modelo, tokenizer, prompt,
            n_variacoes=2, max_tokens=100,
            temperature=0.8, top_k=40, top_p=0.9,
            repetition_penalty=1.2, device=device,
        )
        print(f"\nPrompt: '{prompt}'")
        for i, v in enumerate(variacoes, 1):
            print(f"  Variação {i}: {v}")

    # Demo beam search se solicitado
    if args.beam and args.beam > 1:
        print(f"\n--- Beam Search (width={args.beam}) ---")
        for prompt in prompts[:4]:
            resultado = gerar_beam_search(
                modelo, tokenizer, prompt,
                max_tokens=80, beam_width=args.beam, device=device,
            )
            print(f"\nPrompt: '{prompt}'")
            print(f"  Beam: {resultado}")


def cmd_info(args):
    """Informações do modelo salvo."""
    modelo_path = args.model or "output/melhor_modelo.pt"
    try:
        checkpoint = torch.load(
            modelo_path, map_location="cpu", weights_only=False
        )
    except FileNotFoundError:
        print(f"Modelo não encontrado em: {modelo_path}")
        return

    config = checkpoint["config"]

    # Retrocompatibilidade
    if not hasattr(config, "use_rope"):
        config.use_rope = True
    if "position_embedding.weight" in checkpoint["modelo"]:
        config.use_rope = False

    modelo = GPTModel(config)
    modelo.load_state_dict(checkpoint["modelo"], strict=False)

    print(modelo.resumo())
    print(f"Checkpoint: {modelo_path}")
    print(f"Época: {checkpoint.get('epoch', 'N/A')}")
    print(f"Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    if "val_loss" in checkpoint:
        print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    if "tipo" in checkpoint:
        print(f"Tipo: {checkpoint['tipo']}")
    if "total_tokens" in checkpoint:
        print(f"Tokens treinados: {checkpoint['total_tokens']:,}")
    if "tempo_total_seg" in checkpoint:
        t = checkpoint["tempo_total_seg"]
        print(f"Tempo de treino: {t:.1f}s ({t/60:.1f} min)")

    # Log detalhado
    log_path = Path("output/log_treino.json")
    if log_path.exists():
        log = json.loads(log_path.read_text())
        print(f"\n--- Log de Treino ---")
        print(f"Parâmetros: {log.get('n_params', 'N/A'):,}")
        t = log.get("tempo_total_seg", 0)
        print(f"Tempo total: {t:.1f}s ({t/60:.1f} min)")
        print(f"Tokens/s: {log.get('tokens_por_segundo', 'N/A'):,}")
        print(f"Melhor loss: {log.get('melhor_loss', 'N/A')}")

        epochs = log.get("epocas", [])
        has_val = any("val_loss" in ep for ep in epochs)
        header = f"  {'Ep':>3} │ {'Loss':>8} │ {'PPL':>6}"
        sep = f"  {'─'*3} │ {'─'*8} │ {'─'*6}"
        if has_val:
            header += f" │ {'ValLoss':>8}"
            sep += f" │ {'─'*8}"
        header += f" │ {'LR':>10} │ {'Tempo':>8}"
        sep += f" │ {'─'*10} │ {'─'*8}"
        print(f"\nÉpocas:")
        print(header)
        print(sep)
        for ep in epochs:
            row = f"  {ep['epoca']:>3} │ {ep['loss']:>8.4f} │ {ep['ppl']:>6.2f}"
            if has_val and "val_loss" in ep:
                row += f" │ {ep['val_loss']:>8.4f}"
            elif has_val:
                row += f" │ {'':>8}"
            row += f" │ {ep['lr']:>10.2e} │ {ep['tempo_seg']:>7.1f}s"
            print(row)


def main():
    parser = argparse.ArgumentParser(
        description="MiniGPT — LLM do zero em português",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py train                  # Pré-treinamento
  python main.py train --corpus data.txt  # Corpus externo
  python main.py train --d_model 256    # Modelo maior
  python main.py sft                    # Fine-tuning supervisionado
  python main.py dpo                    # Otimização por preferência
  python main.py chat                   # Chat interativo
  python main.py sample --beam 5        # Amostras com beam search
  python main.py info                   # Info do modelo
        """,
    )
    subparsers = parser.add_subparsers(dest="comando", help="Comando disponível")

    # ── train ──
    p_train = subparsers.add_parser("train", help="Pré-treinamento")
    p_train.add_argument("--corpus", type=str, help="Caminho para corpus externo")
    p_train.add_argument("--d_model", type=int, help="Dimensão do modelo")
    p_train.add_argument("--n_layers", type=int, help="Número de camadas")
    p_train.add_argument("--context_len", type=int, help="Janela de contexto")
    p_train.add_argument("--epochs", type=int, help="Épocas de treino")
    p_train.add_argument("--batch_size", type=int, help="Batch size")
    p_train.add_argument("--accum", type=int, help="Gradient accumulation steps")
    p_train.add_argument("--tokenizer", type=str, choices=["bpe", "char"], help="Tipo de tokenizer")
    p_train.add_argument("--no-rope", action="store_true", help="Desativar RoPE")

    # ── sft ──
    p_sft = subparsers.add_parser("sft", help="Supervised Fine-Tuning")
    p_sft.add_argument("--epochs", type=int, help="Épocas SFT")
    p_sft.add_argument("--lr", type=float, help="Learning rate SFT")

    # ── dpo ──
    p_dpo = subparsers.add_parser("dpo", help="Direct Preference Optimization")
    p_dpo.add_argument("--epochs", type=int, help="Épocas DPO")
    p_dpo.add_argument("--beta", type=float, help="Beta DPO")

    # ── chat ──
    subparsers.add_parser("chat", help="Chat interativo")

    # ── sample ──
    p_sample = subparsers.add_parser("sample", help="Gerar amostras")
    p_sample.add_argument("--beam", type=int, help="Beam width para beam search")

    # ── info ──
    p_info = subparsers.add_parser("info", help="Info do modelo salvo")
    p_info.add_argument("--model", type=str, help="Caminho do modelo")

    args = parser.parse_args()

    if args.comando is None:
        parser.print_help()
        return

    comandos = {
        "train": cmd_train,
        "sft": cmd_sft,
        "dpo": cmd_dpo,
        "chat": cmd_chat,
        "sample": cmd_sample,
        "info": cmd_info,
    }

    comandos[args.comando](args)


if __name__ == "__main__":
    main()