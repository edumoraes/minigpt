"""
main.py — CLI interativa do MiniGPT

COMO USAR:

1. Treinar:
   python main.py train

2. Conversar com modelo treinado:
   python main.py chat

3. Gerar amostras:
   python main.py sample

O fluxo completo:
   1. Treina o modelo no corpus sintético
   2. Carrega o melhor modelo salvo
   3. Gera texto interativamente
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from config import GPTConfig
from generate import gerar, gerar_variacoes
from model import GPTModel
from tokenizer import CharTokenizer
from train import treinar


def carregar_modelo(
    caminho_modelo: str = "output/melhor_modelo.pt",
    caminho_tokenizer: str = "output/tokenizer.json",
    device: str | None = None,
) -> tuple[GPTModel, CharTokenizer, str]:
    """Carrega modelo e tokenizer salvos."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    checkpoint = torch.load(caminho_modelo, map_location=device, weights_only=False)
    config = checkpoint["config"]

    tokenizer = CharTokenizer.carregar(caminho_tokenizer)

    modelo = GPTModel(config).to(device)
    modelo.load_state_dict(checkpoint["modelo"])
    modelo.eval()

    return modelo, tokenizer, device


def cmd_train(args):
    """Executa o treinamento completo."""
    from data.corpus import carregar_corpus

    config = GPTConfig()
    texto = carregar_corpus()

    print(f"\nConfiguração:")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}")
    print(f"  context_len={config.context_len}, batch_size={config.batch_size}")
    print(f"  lr={config.learning_rate}, epochs={config.max_epochs}\n")

    modelo, tokenizer = treinar(config, texto, saida_dir="output")

    # Demo rápida pós-treino
    print("\n" + "=" * 60)
    print("GERAÇÃO PÓS-TREINO:")
    print("=" * 60)
    for prompt in ["O gato", "A menina", "Na cidade", "O professor"]:
        resultado = gerar(modelo, tokenizer, prompt, max_tokens=80, device="cpu")
        print(f"\nPrompt: '{prompt}'")
        print(f"Saída:  {resultado}")


def cmd_chat(args):
    """Chat interativo com o modelo."""
    try:
        modelo, tokenizer, device = carregar_modelo()
    except FileNotFoundError:
        print("Modelo não encontrado! Treine primeiro com: python main.py train")
        return

    print(modelo.resumo())
    print("\nMiniGPT Chat — digite 'sair' pra encerrar, 'config' pra ajustar parâmetros")
    print("=" * 60)

    temperatura = 0.8
    top_k = 40
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
            print(f"Max tokens: {max_tokens}")
            try:
                nova_temp = input(f"Nova temperatura [{temperatura}]: ").strip()
                if nova_temp:
                    temperatura = float(nova_temp)
                novo_k = input(f"Novo top-k [{top_k}]: ").strip()
                if novo_k:
                    top_k = int(novo_k)
                novo_max = input(f"Novo max tokens [{max_tokens}]: ").strip()
                if novo_max:
                    max_tokens = int(novo_max)
            except ValueError:
                print("Valor inválido, mantendo configuração atual.")
            continue

        resultado = gerar(
            modelo, tokenizer, user_input,
            max_tokens=max_tokens,
            temperature=temperatura,
            top_k=top_k,
            device=device,
        )
        print(f"\nMiniGPT: {resultado}")


def cmd_sample(args):
    """Gera amostras de texto do modelo."""
    try:
        modelo, tokenizer, device = carregar_modelo()
    except FileNotFoundError:
        print("Modelo não encontrado! Treine primeiro com: python main.py train")
        return

    prompts = [
        "O gato", "A menina", "Na cidade", "O professor",
        "Era uma vez", "A casa", "O dia", "A chuva",
    ]

    print(modelo.resumo())
    print()
    print("Amostras de geração (temperature=0.8, top_k=40):")
    print("=" * 60)

    for prompt in prompts:
        variacoes = gerar_variacoes(
            modelo, tokenizer, prompt,
            n_variacoes=2, max_tokens=100,
            temperature=0.8, top_k=40,
            device=device,
        )
        print(f"\nPrompt: '{prompt}'")
        for i, v in enumerate(variacoes, 1):
            print(f"  Variação {i}: {v}")


def cmd_info(args):
    """Mostra informações sobre um modelo salvo."""
    try:
        checkpoint = torch.load(
            "output/melhor_modelo.pt", map_location="cpu", weights_only=False
        )
    except FileNotFoundError:
        print("Modelo não encontrado! Treine primeiro com: python main.py train")
        return

    config = checkpoint["config"]
    modelo = GPTModel(config)
    modelo.load_state_dict(checkpoint["modelo"])

    print(modelo.resumo())
    print(f"Época do checkpoint: {checkpoint.get('epoch', 'N/A')}")
    print(f"Loss do checkpoint: {checkpoint.get('loss', 'N/A'):.4f}")

    if "total_tokens" in checkpoint:
        print(f"Tokens treinados: {checkpoint['total_tokens']:,}")
    if "tempo_total_seg" in checkpoint:
        t = checkpoint["tempo_total_seg"]
        print(f"Tempo de treino: {t:.1f}s ({t/60:.1f} min)")

    # Log de treino detalhado
    log_path = Path("output/log_treino.json")
    if log_path.exists():
        import json
        log = json.loads(log_path.read_text())
        print(f"\n--- Log de Treino ---")
        print(f"Parâmetros: {log.get('n_params', 'N/A'):,}")
        t = log.get("tempo_total_seg", 0)
        print(f"Tempo total: {t:.1f}s ({t/60:.1f} min)")
        print(f"Tokens/s: {log.get('tokens_por_segundo', 'N/A'):,}")
        print(f"Melhor loss: {log.get('melhor_loss', 'N/A')}")
        print(f"\nÉpocas:")
        print(f"  {'Ep':>3} │ {'Loss':>8} │ {'PPL':>6} │ {'LR':>10} │ {'Tempo':>8}")
        print(f"  {'─'*3} │ {'─'*8} │ {'─'*6} │ {'─'*10} │ {'─'*8}")
        for ep in log.get("epocas", []):
            print(f"  {ep['epoca']:>3} │ {ep['loss']:>8.4f} │ {ep['ppl']:>6.2f} │ "
                  f"{ep['lr']:>10.2e} │ {ep['tempo_seg']:>7.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="MiniGPT — Treinamento experimental de LLM do zero",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py train     # Treinar o modelo
  python main.py chat      # Chat interativo
  python main.py sample    # Gerar amostras
  python main.py info      # Ver info do modelo salvo
        """,
    )
    subparsers = parser.add_subparsers(dest="comando", help="Comando disponível")

    subparsers.add_parser("train", help="Treinar o modelo no corpus")
    subparsers.add_parser("chat", help="Chat interativo com modelo treinado")
    subparsers.add_parser("sample", help="Gerar amostras de texto")
    subparsers.add_parser("info", help="Informações do modelo salvo")

    args = parser.parse_args()

    if args.comando is None:
        parser.print_help()
        return

    comandos = {
        "train": cmd_train,
        "chat": cmd_chat,
        "sample": cmd_sample,
        "info": cmd_info,
    }

    comandos[args.comando](args)


if __name__ == "__main__":
    main()