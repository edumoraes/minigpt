"""
data/corpus.py — Gerador de dataset sintético em português

POR QUÊ UM DATASET SINTÉTICO?
- Queremos treinar um modelo mesmo sem um grande corpus
- O objetivo é APRENDER os conceitos, não atingir SOTA
- Textos repetitivos ajudam o modelo a aprender padrões mais rápido
- Focamos em frases curtas e estruturas simples em PT

O MODELO VAI APRENDER:
- Estruturas sintáticas básicas do português
- Padrões de palavras frequentes
- Relações artigo→substantivo, sujeito→verbo
- NÃO vai aprender semântica real (precisaria de muito mais dados)
"""

from pathlib import Path


def gerar_corpus() -> str:
    """
    Gera um corpus sintético em português com padrões repetidos.
    
    Quanto mais repetição, mais fácil pro modelo aprender padrões.
    Quanto mais variado, mais geral o modelo fica (mas precisa de mais dados).
    
    Equilíbrio: frases com estrutura simples, vocabulário limitado,
    mas com variação suficiente pra ser interessante.
    """
    sujeitos = [
        "o gato", "a gata", "o cachorro", "a cachorra", "o pássaro",
        "o menino", "a menina", "o professor", "a professora", "o artista",
    ]

    verbos = [
        "comia", "bebia", "dormia", "corria", "pulava",
        "brincava", "estudava", "trabalhava", "cantava", "dançava",
        "falava", "ouvia", "via", "pensava", "sonhava",
    ]

    objetos = [
        "na praça", "no parque", "na casa", "na escola", "no jardim",
        "na rua", "na praia", "no rio", "na montanha", "na cidade",
    ]

    complementos = [
        "com alegria", "com cuidado", "com calma", "com atenção",
        "com amor", "com paciência", "com vontade", "sem pressa",
    ]

    frases = []

    # Padrão 1: Sujeito + verbo + lugar (combinação parcial)
    for s in sujeitos:
        for v in verbos:
            for o in objetos[:5]:
                frases.append(f"{s.capitalize()} {v} {o}.")

    # Padrão 2: Sujeito + verbo + lugar + complemento
    for s in sujeitos[:5]:
        for v in verbos[:5]:
            for o in objetos[:3]:
                for c in complementos[:4]:
                    frases.append(f"{s.capitalize()} {v} {o} {c}.")

    # Frases mais complexas — storytelling
    historias = [
        "Era uma vez um gato que morava numa casa muito bonita. O gato gostava de dormir no jardim todas as tardes. Um dia, o gato encontrou um pássaro na árvore. O pássaro cantava uma melodia linda. O gato e o pássaro ficaram amigos para sempre.",
        "A menina estudava todas as noites na biblioteca. Ela lia muitos livros sobre ciência e arte. A menina queria ser uma grande cientista. A professora ajudava a menina com as dúvidas. A menina trabalhava com dedicação e alegria.",
        "O cachorro corria no parque todas as manhãs. Ele gostava de brincar com a bola. O cachorro era muito feliz. A dona do cachorro cuidava dele com muito amor. Juntos, eles caminhavam pela cidade.",
        "Na cidade grande, as pessoas corriam para todo lado. O trânsito era intenso e barulhento. Mas no parque, tudo era calmo e tranquilo. As crianças brincavam na praça. Os velhos sentavam no banco e observavam os pássaros.",
        "O professor explicava a lição com paciência. Os estudantes ouviam com atenção. A aula era sobre a natureza e os animais. A menina fez uma pergunta. O professor respondeu com alegria. A turma aprendeu muito naquele dia.",
        "A cozinheira preparava um jantar delicioso. Ela cozinhava com muito cuidado. O jantar tinha arroz, feijão e salada. A família comeu com alegria. Todos elogiaram a comida da cozinheira.",
        "O artista pintava um quadro muito bonito. Ele usava cores vivas e fortes. O quadro mostrava uma paisagem da montanha. As pessoas admiravam a pintura na galeria. O artista era feliz com o seu trabalho.",
        "A cientista trabalhava no laboratório. Ela estudava as estrelas e os planetas. A cientista descobriu uma nova estrela. A descoberta era muito importante. O mundo inteiro celebrava a descoberta da cientista.",
        "No jardim da casa, as flores cresciam com beleza. O sol brilhava e a chuva caía com cuidado. As borboletas voavam entre as flores. O gato dormia na grama verde. Era um dia tranquilo e bonito.",
        "O menino e a menina brincavam na praia. Eles construíam castelos de areia. O mar era azul e calmo. Os pássaros voavam sobre a água. A tarde era quente e ensolarada. As crianças estavam felizes.",
    ]

    # Repetir as histórias pra reforçar os padrões
    frases.extend(historias * 5)

    # Adicionar frases soltas mais diretas
    frases_soltas = [
        "O sol brilha no céu azul.",
        "A chuva cai na cidade.",
        "O vento sopra na montanha.",
        "A lua ilumina a noite.",
        "As estrelas brilham no escuro.",
        "O rio corre para o mar.",
        "A floresta é verde e bonita.",
        "O fogo esquenta a casa.",
        "A água é importante para a vida.",
        "A natureza é maravilhosa.",
        "O dia amanheceu bonito.",
        "O pôr do sol era lindo.",
        "A noite estava estrelada.",
        "A manhã chegou com alegria.",
        "A tarde foi tranquila.",
        "O aprendizado nunca termina.",
        "A leitura abre portas.",
        "O conhecimento transforma vidas.",
        "A amizade é um tesouro.",
        "O amor é a maior força.",
        "A vida é bela e curta.",
        "O tempo passa depressa.",
        "A esperança nunca morre.",
        "O mundo é cheio de surpresas.",
        "A música alegra o coração.",
    ]

    frases.extend(frases_soltas * 10)

    # Juntar tudo com espaços
    texto = " ".join(frases)
    return texto


def salvar_corpus(caminho: str = "data/corpus.txt") -> str:
    """Gera e salva o corpus em arquivo. Retorna o texto."""
    texto = gerar_corpus()
    Path(caminho).write_text(texto, encoding="utf-8")
    print(f"Corpus salvo em '{caminho}': {len(texto):,} caracteres")
    return texto


def carregar_corpus(caminho: str = "data/corpus.txt") -> str:
    """Carrega corpus de arquivo, ou gera se não existir."""
    path = Path(caminho)
    if path.exists():
        texto = path.read_text(encoding="utf-8")
        print(f"Corpus carregado de '{caminho}': {len(texto):,} caracteres")
        return texto
    return salvar_corpus(caminho)


if __name__ == "__main__":
    texto = salvar_corpus()
    print(f"\nAmostra do corpus (primeiros 500 chars):")
    print(texto[:500])