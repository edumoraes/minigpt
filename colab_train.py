#!/usr/bin/env python3
"""
colab_train.py — Roda o treino do MiniGPT numa máquina do Google Colab

Usa o colab CLI (https://github.com/googlecolab/google-colab-cli):
1. cria uma máquina nova (T4 por padrão)
2. envia os arquivos do repositório como estão no seu computador,
   incluindo mudanças ainda não commitadas (o que está no .gitignore fica de fora)
3. roda as etapas do main.py em segundo plano e mostra o log a cada 20s
4. baixa o que foi gerado em output/ e desliga a máquina

A máquina é desligada mesmo se der erro ou se você apertar Ctrl+C. Nesse
caso, o melhor modelo salvo até ali também é baixado.

USO:
  python colab_train.py                                    # train → sft → dpo → sample numa T4
  python colab_train.py --etapas "train --epochs 3" sft dpo sample
  python colab_train.py --gpu L4
  python colab_train.py --cpu --etapas "train --epochs 1"  # teste barato, sem GPU
  python colab_train.py --manter                           # deixa a máquina ligada no fim

PRÉ-REQUISITO (uma vez):
  uv tool install google-colab-cli
  colab whoami   # faz o login no navegador
"""

import argparse
import json
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
REMOTO = "/content/minigpt"
LOG_REMOTO = f"{REMOTO}/treino.log"
MARCA_FIM = "@@fim "
INTERVALO_SEG = 20
MAX_FALHAS_SEGUIDAS = 6  # ~2 min sem conseguir ler o log
ETAPAS_VALIDAS = {"train", "sft", "dpo", "sample", "info"}

# Roda no kernel: troca o código antigo pelo pacote enviado e mostra o ambiente
CODIGO_PREPARAR = f"""
import os, shutil, tarfile, torch
shutil.rmtree("{REMOTO}", ignore_errors=True)
with tarfile.open("{REMOTO}.tgz") as pacote:
    pacote.extractall("{REMOTO}", **({{"filter": "data"}} if hasattr(tarfile, "data_filter") else {{}}))
os.remove("{REMOTO}.tgz")
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma"
print(f"@@ambiente torch {{torch.__version__}} | GPU: {{gpu}}")
"""

# Roda no kernel: executa as etapas com a saída indo pro log. O kernel fica
# ocupado até o fim, o que mantém a máquina ativa mesmo se a conexão cair.
CODIGO_TREINAR = """
import os, subprocess
with open("{log}", "w") as log:
    r = subprocess.run(
        ["bash", "-c", {script}], cwd="{remoto}",
        stdout=log, stderr=subprocess.STDOUT,
        env={{**os.environ, "PYTHONUNBUFFERED": "1"}},
    )
with open("{log}", "a") as log:
    log.write(f"\\n{marca}{{r.returncode}}\\n")
"""


def colab(*args: str, entrada: str | None = None, capturar: bool = False,
          timeout: float | None = None) -> subprocess.CompletedProcess:
    """Roda um comando do colab CLI. Timeout vira returncode 124."""
    try:
        return subprocess.run(
            ["colab", *args], input=entrada, text=True,
            capture_output=capturar, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", "timeout")


def empacotar(destino: Path) -> Path:
    """Gera um .tgz com os arquivos versionados e os novos que não estão no .gitignore."""
    lista = subprocess.run(
        ["git", "-C", str(RAIZ), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    arquivos = [f for f in lista if f and (RAIZ / f).is_file()]
    pacote = destino / "minigpt.tgz"
    with tarfile.open(pacote, "w:gz") as tar:
        for f in arquivos:
            tar.add(RAIZ / f, arcname=f)
    print(f"  {len(arquivos)} arquivos, {pacote.stat().st_size / 1024:.0f} KB")
    return pacote


def montar_script(etapas: list[str]) -> str:
    """Script bash que roda cada etapa do main.py e para na primeira que falhar."""
    linhas = ["set -e"]
    for etapa in etapas:
        comando = f"python main.py {etapa}"
        linhas.append(f"printf '\\n%s\\n' {shlex.quote('### ' + comando)}")
        linhas.append(comando)
    return "\n".join(linhas)


def acompanhar(sessao: str, pasta_tmp: Path) -> int:
    """Baixa o log periodicamente, mostra as linhas novas e devolve o código de saída."""
    log_local = pasta_tmp / "treino.log"
    impresso = 0
    falhas = 0
    while True:
        time.sleep(INTERVALO_SEG)
        r = colab("download", "-s", sessao, LOG_REMOTO, str(log_local), capturar=True, timeout=120)
        if r.returncode != 0:
            falhas += 1
            if falhas >= MAX_FALHAS_SEGUIDAS:
                raise RuntimeError(
                    "Não consegui ler o log da máquina por ~2 min. "
                    "Ela pode ter sido desligada pelo Colab (veja `colab sessions`)."
                )
            continue
        falhas = 0

        texto = log_local.read_text(encoding="utf-8", errors="replace")
        # Só linhas completas: a marca de fim nunca chega cortada ao meio
        corte = texto.rfind("\n") + 1
        novo = texto[impresso:corte]
        impresso = max(impresso, corte)
        for linha in novo.splitlines(keepends=True):
            if linha.startswith(MARCA_FIM):
                return int(linha[len(MARCA_FIM):])
            sys.stdout.write(linha)
        sys.stdout.flush()


def baixar_resultados(sessao: str, saida: Path) -> None:
    """Baixa output/ e o log. O que já estava na pasta local vai pra anterior-<data>/."""
    r = colab("ls", "-s", sessao, f"{REMOTO}/output", capturar=True, timeout=120)
    # Ignora pastas e as linhas "[colab] ..." (ex.: aviso de versão nova)
    linhas = [l.strip() for l in r.stdout.splitlines()] if r.returncode == 0 else []
    nomes = [l for l in linhas if l and not l.startswith("[") and not l.endswith("/")]
    if not nomes:
        print("  Nada em output/ pra baixar.")
        return

    saida.mkdir(parents=True, exist_ok=True)
    antigos = [p for p in saida.iterdir() if p.is_file()]
    if antigos:
        backup = saida / f"anterior-{datetime.now():%Y%m%d-%H%M%S}"
        backup.mkdir()
        for p in antigos:
            p.rename(backup / p.name)
        print(f"  O que já estava em {saida} foi movido pra {backup.name}/")

    arquivos = [(f"{REMOTO}/output/{n}", saida / n) for n in nomes]
    arquivos.append((LOG_REMOTO, saida / "treino.log"))
    for remoto, local in arquivos:
        r = colab("download", "-s", sessao, remoto, str(local), capturar=True, timeout=300)
        print(f"  {'✓' if r.returncode == 0 else '✗'} {local}")


def verificar_sessao_livre(sessao: str) -> None:
    """Sai com erro se o login falhar ou se já existir uma sessão com esse nome."""
    r = colab("sessions", capturar=True, timeout=120)
    if r.returncode != 0:
        sys.exit(f"`colab sessions` falhou. Já fez o login com `colab whoami`?\n{r.stdout}{r.stderr}")
    # Linhas no formato "[nome] endpoint | Hardware: ..."
    if any(linha.startswith(f"[{sessao}] ") for linha in r.stdout.splitlines()):
        sys.exit(f"Já existe uma sessão '{sessao}'. Use --sessao outro_nome ou `colab stop -s {sessao}`.")


def preparar_maquina(sessao: str, pacote: Path) -> str:
    """Envia o pacote, troca o código antigo por ele e devolve a descrição do ambiente."""
    if colab("upload", "-s", sessao, str(pacote), f"{REMOTO}.tgz", capturar=True).returncode != 0:
        raise RuntimeError("Falha ao enviar o pacote pra máquina")
    r = colab("exec", "-s", sessao, "--timeout", "300", entrada=CODIGO_PREPARAR, capturar=True)
    ambiente = next((l for l in r.stdout.splitlines() if l.startswith("@@ambiente ")), None)
    if r.returncode != 0 or ambiente is None:
        raise RuntimeError(f"Falha ao preparar a máquina:\n{r.stdout}{r.stderr}")
    return ambiente.removeprefix("@@ambiente ")


def mostrar_custo() -> None:
    """Mostra quanto a máquina ligada custa por hora e o saldo de unidades."""
    # Saída do `colab usage`: "Usage rate: 1.84/hr" e "Current balance: 85.69 compute units"
    r = colab("usage", capturar=True, timeout=60)
    uso = dict(l.split(":", 1) for l in r.stdout.splitlines() if ":" in l)
    if "Usage rate" in uso:
        taxa = uso["Usage rate"].strip().removesuffix("/hr")
        saldo = uso.get("Current balance", "?").strip().removesuffix(" compute units")
        print(f"  Custo: {taxa} unidades/hora | saldo: {saldo} unidades")


def iniciar_treino(sessao: str, etapas: list[str]) -> subprocess.Popen:
    """Começa as etapas no kernel e devolve o processo do exec, que fica conectado."""
    codigo = CODIGO_TREINAR.format(
        log=LOG_REMOTO, remoto=REMOTO, marca=MARCA_FIM,
        script=json.dumps(montar_script(etapas)),
    )
    # O exec fica conectado durante todo o treino só pra manter o kernel
    # ocupado; o progresso vem do log, então a saída dele é descartada.
    treino = subprocess.Popen(
        ["colab", "exec", "-s", sessao, "--timeout", "86400"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,
    )
    treino.stdin.write(codigo)
    treino.stdin.close()
    return treino


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Treina o MiniGPT numa máquina do Google Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USO:")[1],
    )
    maquina = parser.add_mutually_exclusive_group()
    maquina.add_argument("--gpu", default="T4", help="T4 (padrão), L4, G4, A100 ou H100")
    maquina.add_argument("--cpu", action="store_true", help="Máquina sem GPU (pra testes rápidos)")
    parser.add_argument("--etapas", nargs="+", default=["train", "sft", "dpo", "sample"],
                        help='Comandos do main.py, em ordem. Ex.: "train --epochs 3" sft')
    parser.add_argument("--sessao", default="minigpt", help="Nome da sessão no Colab")
    parser.add_argument("--saida", default="output",
                        help="Pasta local pros arquivos baixados (relativa ao repositório)")
    parser.add_argument("--manter", action="store_true",
                        help="Se tudo der certo, não desliga a máquina (pra usar `colab console`)")
    args = parser.parse_args()
    # Mantém a ordem das mensagens com a saída dos comandos colab (ex.: com `| tee`)
    sys.stdout.reconfigure(line_buffering=True)

    for etapa in args.etapas:
        comando = etapa.split()[0]
        if comando == "chat":
            parser.error("chat é interativo: use --manter e depois `colab console -s SESSAO`")
        if comando not in ETAPAS_VALIDAS:
            parser.error(f"etapa desconhecida: {comando} (válidas: {', '.join(sorted(ETAPAS_VALIDAS))})")

    if not shutil.which("colab"):
        sys.exit("colab CLI não encontrado. Instale com: uv tool install google-colab-cli")
    verificar_sessao_livre(args.sessao)

    saida = RAIZ / args.saida
    hardware = "CPU" if args.cpu else args.gpu
    pasta_tmp = Path(tempfile.mkdtemp(prefix="minigpt-colab-"))
    criou = False
    treino = None
    codigo = 1
    inicio = time.time()

    try:
        print("[1/4] Empacotando o repositório")
        pacote = empacotar(pasta_tmp)

        print(f"[2/4] Criando a máquina '{args.sessao}' ({hardware})")
        try:
            r = colab("new", "-s", args.sessao, *([] if args.cpu else ["--gpu", args.gpu]))
        except KeyboardInterrupt:
            criou = True  # a máquina pode ter sido criada no meio do caminho
            raise
        if r.returncode != 0:
            print("Não foi possível criar a máquina. Sem cota pra essa GPU? Tente --gpu T4 ou --cpu.")
            return 1
        criou = True

        ambiente = preparar_maquina(args.sessao, pacote)
        print(f"  {ambiente}")
        if not args.cpu and "GPU: nenhuma" in ambiente:
            raise RuntimeError("A máquina veio sem GPU visível pro PyTorch")
        mostrar_custo()

        print(f"[3/4] Rodando: {' → '.join(args.etapas)}  (Ctrl+C interrompe e desliga)")
        treino = iniciar_treino(args.sessao, args.etapas)
        codigo = acompanhar(args.sessao, pasta_tmp)
        print(f"Etapas {'concluídas' if codigo == 0 else f'falharam (código {codigo})'}"
              f" em {(time.time() - inicio) / 60:.1f} min")

    except KeyboardInterrupt:
        print("\nInterrompido.")
        codigo = 130
    except RuntimeError as e:
        print(f"\nErro: {e}")
        codigo = 1
    finally:
        # Um segundo Ctrl+C aqui deixaria a máquina ligada gastando unidades
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if treino and treino.poll() is None:
            treino.terminate()
        if criou:
            print(f"[4/4] Baixando resultados pra {saida}")
            baixar_resultados(args.sessao, saida)
            if args.manter and codigo == 0:
                print(f"\nMáquina '{args.sessao}' continua ligada (gastando unidades). Pra conversar com o modelo:")
                print(f"  colab console -s {args.sessao}   # depois: cd {REMOTO} && python main.py chat")
                print(f"Pra desligar: colab stop -s {args.sessao}")
            elif colab("stop", "-s", args.sessao, capturar=True, timeout=180).returncode == 0:
                print(f"  Máquina '{args.sessao}' desligada.")
            else:
                print(f"\nATENÇÃO: não consegui desligar a máquina. Rode agora: colab stop -s {args.sessao}")
        shutil.rmtree(pasta_tmp, ignore_errors=True)

    return codigo


if __name__ == "__main__":
    sys.exit(main())
