"""
data/corpus.py — Corpus sintético em português + augmentação + dados SFT/DPO

EXPANDIDO com:
- Corpus maior e mais variado
- Augmentação de dados (dropout de tokens, permutação de frases)
- Dataset SFT (instrução → resposta)
- Dataset DPO (preferência: resposta boa vs ruim)
- Suporte a corpus externo (arquivo de texto)
"""

import random
from pathlib import Path


# ──────────────────────────────────────────────────────────
# Corpus de pré-treinamento
# ──────────────────────────────────────────────────────────

def gerar_corpus() -> str:
    """Gera corpus sintético em português com padrões variados."""

    sujeitos = [
        "o gato", "a gata", "o cachorro", "a cachorra", "o pássaro",
        "o menino", "a menina", "o professor", "a professora", "o artista",
        "a cozinheira", "o cientista", "a médica", "o engenheiro", "a escritora",
        "o músico", "a rainha", "o rei", "a bailarina", "o pescador",
    ]

    verbos = [
        "comia", "bebia", "dormia", "corria", "pulava",
        "brincava", "estudava", "trabalhava", "cantava", "dançava",
        "falava", "ouvia", "via", "pensava", "sonhava",
        "escrevia", "lia", "pintava", "cozinhava", "navegava",
    ]

    objetos = [
        "na praça", "no parque", "na casa", "na escola", "no jardim",
        "na rua", "na praia", "no rio", "na montanha", "na cidade",
        "na floresta", "no campo", "na biblioteca", "no teatro", "na cozinha",
    ]

    complementos = [
        "com alegria", "com cuidado", "com calma", "com atenção",
        "com amor", "com paciência", "com vontade", "sem pressa",
        "com determinação", "com entusiasmo", "com gratidão", "com coragem",
    ]

    frases = []

    # Padrão 1: Sujeito + verbo + lugar
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

    # Padrão 3: Frases negativas
    for s in sujeitos[:5]:
        for v in verbos[:5]:
            frases.append(f"{s.capitalize()} não {v}.")

    # Padrão 4: Perguntas
    for s in sujeitos[:5]:
        for v in verbos[:5]:
            frases.append(f"Será que {s} {v}?")
            frases.append(f"Onde {s} {v}?")

    # Histórias mais longas — storytelling
    historias = [
        "Era uma vez um gato que morava numa casa muito bonita. O gato gostava de dormir no jardim todas as tardes. Um dia, o gato encontrou um pássaro na árvore. O pássaro cantava uma melodia linda. O gato e o pássaro ficaram amigos para sempre. Juntos, exploravam o jardim e brincavam na grama verde. O gato aprendeu que a amizade é o maior tesouro.",
        "A menina estudava todas as noites na biblioteca. Ela lia muitos livros sobre ciência e arte. A menina queria ser uma grande cientista. A professora ajudava a menina com as dúvidas. A menina trabalhava com dedicação e alegria. Um dia, a menina descobriu algo incrível e o mundo celebrou a sua descoberta.",
        "O cachorro corria no parque todas as manhãs. Ele gostava de brincar com a bola. O cachorro era muito feliz. A dona do cachorro cuidava dele com muito amor. Juntos, eles caminhavam pela cidade e exploravam novos caminhos. O cachorro aprendeu que a vida é bela quando temos alguém que nos ama.",
        "Na cidade grande, as pessoas corriam para todo lado. O trânsito era intenso e barulhento. Mas no parque, tudo era calma e tranquilo. As crianças brincavam na praça. Os velhos sentavam no banco e observavam os pássaros. O sol brilhava e a brisa sopava suavemente. Era uma tarde perfeita na cidade.",
        "O professor explicava a lição com paciência. Os estudantes ouviam com atenção. A aula era sobre a natureza e os animais. A menina fez uma pergunta interessante. O professor respondeu com alegria. A turma aprendeu muito naquele dia. O conhecimento é uma luz que ilumina o caminho.",
        "A cozinheira preparava um jantar delicioso. Ela cozinhava com muito cuidado. O jantar tinha arroz, feijão e salada. A família comeu com alegria. Todos elogiaram a comida da cozinheira. O segredo dela era colocar amor em cada prato. A cozinha era o coração daquela casa.",
        "O artista pintava um quadro muito bonito. Ele usava cores vivas e fortes. O quadro mostrava uma paisagem da montanha. As pessoas admiravam a pintura na galeria. O artista era feliz com o seu trabalho. Ele dizia que a arte é a janela da alma. Cada pincelada contava uma história.",
        "A cientista trabalhava no laboratório. Ela estudava as estrelas e os planetas. A cientista descobriu uma nova estrela. A descoberta era muito importante. O mundo inteiro celebrava a descoberta da cientista. Ela provou que a curiosidade é o motor da ciência. Um novo capítulo da astronomia começava.",
        "No jardim da casa, as flores cresciam com beleza. O sol brilhava e a chuva caía com cuidado. As borboletas voavam entre as flores. O gato dormia na grama verde. Era um dia tranquilo e bonito. A natureza mostra que a simplicidade é a maior riqueza.",
        "O menino e a menina brincavam na praia. Eles construíam castelos de areia. O mar era azul e calmo. Os pássaros voavam sobre a água. A tarde era quente e ensolarada. As crianças estavam felizes. A praia é um lugar mágico onde sonhos se realizam.",
        "A música enchia a sala de concerto. O violinista tocava com paixão. Cada nota era uma emoção profunda. A plateia segurava a respiração. No final, a oviação foi enorme. A música tem o poder de tocar almas e unir corações.",
        "O navio navegava pelo oceano azul. Os marinheiros trabalhavam com determinação. O vento soprava as velas e o barco avançava. No horizonte, uma ilha apareceu. Era uma ilha misteriosa cheia de árvores e cachoeiras. A tripulação decidiu explorar e encontrou um tesouro escondido.",
        "A chuva caía suavemente sobre o telhado. A mulher sentava na janela com um livro. O som da chuva era relaxante e tranquilo. Ela lia página após página, imersa na história. Quando a chuva parou, um arco-íris apareceu no céu. Era como se a natureza estivesse pintando um quadro.",
        "O avião sobrevoava as montanhas nevadas. Os passageiros olhavam pela janela maravilhados. As nuvens formavam castelos brancos no céu. O piloto anunciou que Logo chegariam ao destino. A viagem era longa, mas a vista compensava cada minuto. A beleza vista de cima é impossível de esquecer.",
        "Na fazenda, o galo cantava ao amanhecer. As vacas pastavam no campo verde. O fazendeiro cuidava dos animais com carinho. A vida no campo era simples mas feliz. O ar era puro e a água cristalina. Tudo na natureza funcionava em harmonia perfeita.",
    ]

    frases.extend(historias * 5)

    # Frases soltas — mais variadas
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
        "A sabedoria vem com os anos.",
        "A paciência é uma virtude.",
        "O respeito é fundamental.",
        "A diversidade nos enriquece.",
        "A criatividade não tem limites.",
        "A simpler explicaçao é frequentemente a melhor.",
        "Todo grande jornada começa com um passo.",
        "A prática leva à perfeição.",
        "Nunca é tarde para aprender.",
        "A coragem não é a ausência de medo.",
        "A imaginação é mais importante que o conhecimento.",
    ]

    frases.extend(frases_soltas * 15)

    # Diálogos simples
    dialogos = [
        "— Bom dia! — disse a menina com um sorriso. — Bom dia! — respondeu o menino alegremente.",
        "— O que você está lendo? — perguntou o professor. — Estou lendo um livro sobre ciência! — respondeu a estudante.",
        "— Vamos brincar no parque? — sugeriu o cachorro. — Sim, vamos! — disseram as crianças.",
        "— A comida está deliciosa! — elogiou o visitante. — Obrigada! — disse a cozinheira satisfeita.",
        "— Qual é o seu sonho? — perguntou a amiga. — Eu quero ser cientista! — respondeu a menina.",
    ]
    frases.extend(dialogos * 10)

    random.seed(42)
    random.shuffle(frases)
    texto = " ".join(frases)
    return texto


# ──────────────────────────────────────────────────────────
# Augmentação de dados
# ──────────────────────────────────────────────────────────

def augmentar_texto(texto: str, prob_dropout: float = 0.1) -> str:
    """
    Augmentação por token dropout: remove caracteres aleatoriamente.

    Simula ruído e força o modelo a aprender representações mais robustas.
    Distribui uniformemente entre os caracteres, evitando caracteres de pontuação.
    """
    random.seed(None)
    resultado = []
    for ch in texto:
        if ch in ".!?,;:" and random.random() < prob_dropout * 0.5:
            continue
        if random.random() < prob_dropout * 0.3:
            continue
        resultado.append(ch)
    return "".join(resultado)


def permutar_frases(texto: str) -> str:
    """Embaralha as frases do corpus para criar variações estruturais."""
    frases = [f.strip() for f in texto.split(".") if f.strip()]
    frases = [f + "." for f in frases]
    random.shuffle(frases)
    return " ".join(frases)


# ──────────────────────────────────────────────────────────
# Dataset SFT (Supervised Fine-Tuning)
# ──────────────────────────────────────────────────────────

def gerar_dados_sft() -> list[tuple[str, str]]:
    """
    Gera pares instrução-resposta para fine-tuning supervisionado.

    Formato: (instrução, resposta)
    Durante o treino, a loss é computada apenas nos tokens da resposta.
    """
    dados = [
        ("O que é o sol?", "O sol é uma estrela que ilumina e aquece a Terra. Ele é essencial para a vida no nosso planeta."),
        ("O que os gatos gostam de fazer?", "Os gatos gostam de dormir, brincar, caçar e receber carinho. São animais curiosos e independentes."),
        ("Como é a praia?", "A praia é um lugar bonito com areia, mar e sol. As pessoas nadam, brincam e relaxam na praia."),
        ("O que é a chuva?", "A chuva é água que cai das nuvens. Ela é importante para as plantas, os rios e a natureza."),
        ("Quem é o professor?", "O professor é uma pessoa que ensina e ajuda os estudantes. Ele compartilha conhecimento com paciência e dedicação."),
        ("O que é a amizade?", "A amizade é um sentimento de afeto e confiança entre pessoas. Amigos se apoiam e compartilham momentos juntos."),
        ("Como é a floresta?", "A floresta é um lugar cheio de árvores, plantas e animais. É verde, fresca e muito importante para o planeta."),
        ("O que é a música?", "A música é uma forma de arte que usa sons organizados de maneira harmoniosa. Ela pode expressar emoções e contar histórias."),
        ("O que você faz na escola?", "Na escola, nós estudamos, aprendemos coisas novas, fazemos amigos e nos preparamos para o futuro."),
        ("O que é a leitura?", "A leitura é o ato de interpretar textos escritos. Ela nos leva a mundos imaginários e nos ensina sobre a vida."),
        ("Como é a cidade?", "A cidade é um lugar com muitas pessoas, prédios, ruas e carros. Tem parques, escolas e muitos lugares para visitar."),
        ("O que é a ciência?", "A ciência é o estudo do mundo natural. Os cientistas fazem experiências para entender como as coisas funcionam."),
        ("Por que o céu é azul?", "O céu parece azul porque a luz do sol se espalha ao passar pela atmosfera. As cores mais azuis se espalham mais."),
        ("O que são as estrelas?", "As estrelas são grandes esferas de gás quente que brilham no céu. O sol é a estrela mais perto da Terra."),
        ("Como é o inverno?", "O inverno é a estação mais fria do ano. Os dias são mais curtos, as pessoas usam roupas quentes e às vezes neva."),
        ("O que é um jardim?", "O jardim é um lugar com flores, plantas e árvores. As pessoas cuidam do jardim regando e podando as plantas."),
        ("Quem é a cozinheira?", "A cozinheira é uma pessoa que prepara comida com habilidade e carinho. Ela cozinha pratos deliciosos para os outros."),
        ("O que é a coragem?", "A coragem é a força de enfrentar o medo e as dificuldades. Coragem não é ausência de medo, mas a decisão de agir apesar dele."),
        ("Como funcionam os sonhos?", "Os sonhos são imagens e histórias que o cérebro cria enquanto dormimos. Eles podem ser bonitos, estranhos ou assustadores."),
        ("O que é a felicidade?", "A felicidade é um sentimento de alegria e contentamento. Cada pessoa encontra a felicidade de um jeito diferente."),
    ]
    return dados


# ──────────────────────────────────────────────────────────
# Dataset DPO (Direct Preference Optimization)
# ──────────────────────────────────────────────────────────

def gerar_dados_dpo() -> list[tuple[str, str, str]]:
    """
    Gera pares de preferência para DPO.

    Formato: (prompt, resposta_escolhida, resposta_rejeitada)
    resposta_escolhida = resposta boa (coerente, correta)
    resposta_rejeitada = resposta ruim (incoerente, errada)
    """
    dados = [
        ("O que é o sol?", "O sol é uma estrela que ilumina e aquece a Terra.", "O sol é uma planeta grande."),
        ("Como é a praia?", "A praia é um lugar bonito com areia e mar.", "A praia é um prédio grande."),
        ("O que os gatos gostam?", "Os gatos gostam de dormir e brincar.", "Os gatos gostam de pilotar aviões."),
        ("O que é a chuva?", "A chuva é água que cai das nuvens.", "A chuva é fogo que cai do céu."),
        ("Quem é o professor?", "O professor é uma pessoa que ensina os estudantes.", "O professor é um animal que vive na floresta."),
        ("O que é a amizade?", "A amizade é um sentimento de carinho e confiança.", "A amizade é uma planta que cresce no jardim."),
        ("O que é a ciência?", "A ciência é o estudo do mundo natural.", "A ciência é um tipo de comida."),
        ("Como é o inverno?", "O inverno é a estação mais fria do ano.", "O inverno é a estação mais quente do ano."),
        ("O que são estrelas?", "As estrelas são esferas de gás que brilham no céu.", "As estrelas são pedras no chão."),
        ("O que é a música?", "A música é uma arte que organiza sons de forma harmoniosa.", "A música é um esporte radical."),
    ]
    return dados


# ──────────────────────────────────────────────────────────
# Carregar/salvar corpus
# ──────────────────────────────────────────────────────────

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


def carregar_corpus_externo(caminho: str) -> str:
    """Carrega corpus de um arquivo de texto externo (ex: OSCAR, brWac)."""
    path = Path(caminho)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    texto = path.read_text(encoding="utf-8")
    print(f"Corpus externo carregado de '{caminho}': {len(texto):,} caracteres")
    return texto