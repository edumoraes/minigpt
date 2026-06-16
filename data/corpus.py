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
        "o peixe", "a borboleta", "o urso", "a águia", "o golfinho",
        "a formiga", "o coelho", "a tartaruga", "o leão", "a zebra",
        "o médico", "a enfermeira", "o piloto", "a jornalista", "o agricultor",
        "a advogada", "o arquiteto", "a dentista", "o bombeiro", "a policial",
        "o carteiro", "a psicóloga", "o mecânico", "a veterinária", "o padeiro",
        "a florista", "o rio", "a montanha", "o mar", "a nuvem",
        "o vento", "a chuva", "o sol", "a lua", "a estrela",
        "a flor", "a árvore", "a pedra", "o lago", "a ilha",
    ]

    verbos = [
        "comia", "bebia", "dormia", "corria", "pulava",
        "brincava", "estudava", "trabalhava", "cantava", "dançava",
        "falava", "ouvia", "via", "pensava", "sonhava",
        "escrevia", "lia", "pintava", "cozinhava", "navegava",
        "correu", "pulou", "voou", "nadou", "cantou", "dançou",
        "escreveu", "leu", "pintou", "cozinhou", "navegou", "construiu",
        "descobriu", "ensinou", "aprendeu", "cresceu", "brilhou", "sorriu",
        "chorou", "dormiu", "acordou", "caminhou", "encontrou", "perdeu",
        "ganhou", "amou", "sonhou", "corre", "pula", "voa",
        "nada", "canta", "dança", "escreve", "lê", "pinta",
        "cozinha", "navega", "constrói", "descobre", "ensina", "aprende",
        "cresce", "brilha", "sorri", "chora", "dorme", "acorda",
        "caminha", "encontra", "perde", "ganha", "ama", "sonha",
        "correrá", "pulará", "voará", "nadará", "cantará", "dançará",
        "escreverá", "lerá", "pintará", "cozinhará", "navegará", "construirá",
        "descobrirá", "ensinará", "aprenderá", "crescerá", "brilhará", "sorrirá",
    ]

    objetos = [
        "na praça", "no parque", "na casa", "na escola", "no jardim",
        "na rua", "na praia", "no rio", "na montanha", "na cidade",
        "na floresta", "no campo", "na biblioteca", "no teatro", "na cozinha",
        "no mercado", "na feira", "no hospital", "na igreja", "no estádio",
        "no museu", "no cinema", "no restaurante", "na padaria", "na farmácia",
        "no escritório", "na universidade", "no aeroporto", "na estação", "no porto",
        "na fazenda", "no acampamento", "na caverna", "no deserto", "na neve",
    ]

    complementos = [
        "com alegria", "com cuidado", "com calma", "com atenção",
        "com amor", "com paciência", "com vontade", "sem pressa",
        "com determinação", "com entusiasmo", "com gratidão", "com coragem",
        "com sabedoria", "com esperança", "com fé", "com orgulho",
        "com humildade", "com generosidade", "com justiça", "com verdade",
        "com liberdade", "com delicadeza",
    ]

    adjetivos = [
        "esperto", "curioso", "tranquilo", "corajoso", "gentil", "criativo",
        "alegre", "cuidadoso", "forte", "paciente", "brilhante", "sereno",
        "rápido", "sábio", "generoso", "persistente", "atento", "calmo",
    ]

    gerundios = [
        "estudando", "brincando", "cozinhando", "cantando", "dançando", "lendo",
        "escrevendo", "pintando", "caminhando", "observando", "trabalhando", "viajando",
    ]

    eventos = [
        "choveu", "o telefone tocou", "a campainha soou", "o sol apareceu",
        "a noite chegou", "o vento aumentou", "o trem partiu", "a música começou",
        "a estrela brilhou", "a festa terminou", "o mercado abriu", "a aula começou",
    ]

    condicoes = [
        "chove", "o vento sopra", "a criança lê", "a semente recebe água",
        "a turma coopera", "a família conversa", "o músico pratica", "a cidade descansa",
        "o sol aparece", "a médica escuta", "o atleta treina", "a cientista observa",
    ]

    consequencias = [
        "a planta cresce", "as folhas dançam", "o conhecimento aumenta", "a flor nasce",
        "o trabalho fica leve", "a casa ganha paz", "a canção melhora", "as ruas ficam calmas",
        "o dia fica claro", "o paciente se sente seguro", "o corpo fica forte", "a descoberta acontece",
    ]

    contrastes = [
        "estivesse cansado", "a estrada fosse longa", "o céu estivesse escuro",
        "o problema parecesse difícil", "a chuva caísse forte", "a sala estivesse cheia",
        "o tempo fosse curto", "a montanha fosse alta", "o mar estivesse agitado",
        "a tarefa exigisse esforço",
    ]

    resultados = [
        "o menino continuou", "a viajante chegou feliz", "a estrela apareceu",
        "a equipe encontrou uma solução", "a família cantou na varanda", "a professora explicou com calma",
        "a turma terminou o projeto", "o alpinista alcançou o topo", "o barco voltou ao porto",
        "todos aprenderam algo novo",
    ]

    dialogos_diretos = [
        "Estudem com dedicação.", "Cuidem da natureza.", "Vamos tentar de novo.",
        "Amanhã será melhor.", "A música começa agora.", "Preparem a mesa, por favor.",
        "Observem o céu com atenção.", "A amizade precisa de cuidado.",
    ]

    acoes = [
        "dormir", "viajar", "cozinhar", "estudar", "plantar", "correr",
        "cantar", "pintar", "nadar", "trabalhar", "ler", "ensinar",
    ]

    acoes_passadas = [
        "estudou", "ensinou", "cantou", "dançou", "cozinhou", "viajou",
        "leu", "escreveu", "plantou", "colheu", "observou", "ajudou",
    ]

    frases = []

    # Padrão 1: Sujeito + verbo + lugar
    for s in sujeitos:
        for v in verbos[:45]:
            for o in objetos[:2]:
                frases.append(f"{s.capitalize()} {v} {o}.")

    # Padrão 2: Sujeito + verbo + lugar + complemento
    for s in sujeitos[:20]:
        for v in verbos[10:30]:
            for o in objetos[2:4]:
                for c in complementos[:4]:
                    frases.append(f"{s.capitalize()} {v} {o} {c}.")

    # Padrão 3: Frases negativas
    for s in sujeitos[:30]:
        for v in verbos[:12]:
            frases.append(f"{s.capitalize()} não {v}.")

    # Padrão 4: Perguntas
    for s in sujeitos[:20]:
        for v in verbos[:10]:
            frases.append(f"Será que {s} {v}?")
            frases.append(f"Onde {s} {v}?")

    # Padrão 5: Sujeito + ser + adjetivo
    for i, s in enumerate(sujeitos):
        frases.append(f"{s.capitalize()} é {adjetivos[i % len(adjetivos)]}.")

    # Padrão 6: Sujeito estava + gerúndio + quando + evento
    for i, s in enumerate(sujeitos[:48]):
        frases.append(f"{s.capitalize()} estava {gerundios[i % len(gerundios)]} quando {eventos[i % len(eventos)]}.")

    # Padrão 7: Condição e consequência
    for i, condicao in enumerate(condicoes):
        frases.append(f"Se {condicao}, então {consequencias[i % len(consequencias)]}.")

    # Padrão 8: Contraste e resultado
    for i, contraste in enumerate(contrastes):
        frases.append(f"Embora {contraste}, {resultados[i % len(resultados)]}.")

    # Padrão 9: Diálogo direto em frase narrativa
    for i, fala in enumerate(dialogos_diretos):
        frases.append(f"{sujeitos[i].capitalize()} disse: '{fala}'")

    # Padrão 10: Depois que evento, resultado
    for i, evento in enumerate(eventos):
        frases.append(f"Depois que {evento}, {resultados[i % len(resultados)]}.")

    # Padrão 11: Antes de ação, sujeito ação
    for i, acao in enumerate(acoes):
        frases.append(f"Antes de {acao}, {sujeitos[i + 5]} {verbos[20 + i]}.")

    # Padrão 12: Não apenas ação, mas também ação
    for i, acao in enumerate(acoes_passadas):
        frases.append(f"Não apenas {acao}, mas também {acoes_passadas[(i + 1) % len(acoes_passadas)]}.")

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
        "No laboratório da escola, a turma misturou água, sal e corante. A professora explicou que cada experiência precisava de observação e registro. Quando os cristais apareceram no copo, todos ficaram admirados. A ciência parecia uma aventura feita de perguntas.",
        "A floresta acordou com o canto dos pássaros. A borboleta passou pelas flores amarelas e o coelho saiu devagar da toca. Perto do lago, a tartaruga tomou sol em silêncio. A manhã mostrou que cada ser vivo tinha seu ritmo.",
        "Na avenida movimentada, o carteiro entregava cartas enquanto a jornalista anotava notícias. Os ônibus paravam na estação e as pessoas caminhavam com pressa. Mesmo assim, um músico tocava violão na esquina. A cidade tinha barulho, trabalho e poesia.",
        "O time treinou no estádio antes do campeonato. A treinadora pediu atenção, respeito e coragem. No último minuto, a menina marcou um gol bonito. A vitória foi celebrada como fruto da união.",
        "A banda ensaiava numa garagem pequena. O baterista marcava o ritmo e a cantora procurava a nota certa. Depois de muitas tentativas, a música finalmente ganhou forma. Os vizinhos aplaudiram pela janela.",
        "Na cozinha da avó, o cheiro de pão quente enchia a casa. O padeiro ensinou a sovar a massa com paciência. As crianças esperaram o forno apitar. Quando o pão ficou pronto, todos comeram com manteiga e alegria.",
        "A família viajou de trem até uma cidade antiga. Pela janela, apareciam rios, plantações e pontes de pedra. No museu, aprenderam histórias sobre outros tempos. A viagem deixou lembranças luminosas.",
        "Duas amigas encontraram uma carteira perdida no parque. Elas procuraram um guarda e entregaram tudo com cuidado. O dono agradeceu emocionado. Naquele dia, as meninas entenderam que honestidade também é amizade.",
        "O inverno chegou com noites frias e cobertores pesados. A família preparou sopa e chocolate quente. Do lado de fora, a chuva batia no telhado. Dentro de casa, as conversas aqueciam o coração.",
        "Na primavera, a florista abriu a loja bem cedo. Rosas, lírios e margaridas coloriam as mesas. Um menino comprou uma flor para a mãe. O pequeno presente deixou a manhã mais doce.",
        "O golfinho nadava perto do barco dos pesquisadores. A equipe anotava sons, movimentos e caminhos no mar. Ninguém tocava no animal, apenas observava com respeito. A natureza ensinava sem precisar falar.",
        "Na escola, a biblioteca ganhou novos livros. O professor organizou uma roda de leitura no pátio. Cada estudante escolheu uma história diferente. Ao final, todos queriam contar o que haviam imaginado.",
        "No hospital, a enfermeira caminhava pelos corredores com calma. Ela conversava com os pacientes e lembrava os horários dos remédios. O médico explicava cada exame com clareza. O cuidado fazia o medo diminuir.",
        "Durante as festas de junho, a praça recebeu bandeirinhas coloridas. Havia milho, música e dança ao redor da fogueira. A comunidade inteira ajudou na organização. A noite terminou com risadas e céu estrelado.",
        "O menino sentia saudade do avô que morava longe. Ele escreveu uma carta contando sobre a escola e o cachorro. Dias depois, recebeu uma resposta cheia de carinho. A saudade ficou menor quando as palavras chegaram.",
        "A arquiteta desenhou uma casa com janelas grandes e jardim aberto. Ela pensou na luz da manhã e na sombra da tarde. A família acompanhou cada detalhe do projeto. Quando a obra terminou, a casa parecia abraçar quem entrava.",
        "Na praia limpa, voluntários recolheram plástico e redes antigas. A tartaruga voltou ao mar sem obstáculos. Crianças aprenderam por que o lixo machuca os animais. Cuidar do oceano virou promessa coletiva.",
        "O mecânico abriu a oficina antes do nascer do sol. Ele ouviu o motor do carro e encontrou o problema. Com ferramentas simples, consertou a peça quebrada. O motorista seguiu viagem agradecido.",
        "A psicóloga organizou uma conversa sobre emoções na escola. Os estudantes falaram de medo, raiva, alegria e vergonha. Ninguém riu das histórias dos colegas. A escuta transformou a sala num lugar mais seguro.",
        "No acampamento, as crianças aprenderam a montar barracas. À noite, observaram estrelas e inventaram constelações. O vento balançava as árvores com suavidade. Dormiram felizes depois de uma aventura simples.",
    ]

    frases.extend(historias * 3)

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
        "A gentileza torna o caminho mais leve.",
        "A curiosidade move a ciência.",
        "O trabalho honesto constrói confiança.",
        "A conversa sincera evita muitos conflitos.",
        "Cada estação tem sua beleza.",
        "O silêncio também pode ensinar.",
        "A família guarda memórias importantes.",
        "O esporte ensina disciplina e cooperação.",
        "Cozinhar é uma forma de cuidado.",
        "Viajar amplia o olhar sobre o mundo.",
    ]

    frases.extend(frases_soltas * 10)

    # Diálogos simples
    dialogos = [
        "— Bom dia! — disse a menina com um sorriso. — Bom dia! — respondeu o menino alegremente.",
        "— O que você está lendo? — perguntou o professor. — Estou lendo um livro sobre ciência! — respondeu a estudante.",
        "— Vamos brincar no parque? — sugeriu o cachorro. — Sim, vamos! — disseram as crianças.",
        "— A comida está deliciosa! — elogiou o visitante. — Obrigada! — disse a cozinheira satisfeita.",
        "— Qual é o seu sonho? — perguntou a amiga. — Eu quero ser cientista! — respondeu a menina.",
        "— Você viu a chuva chegando? — perguntou o carteiro. — Vi, e trouxe um guarda-chuva extra — respondeu a florista.",
        "— O treino foi difícil? — perguntou o pai. — Foi, mas eu melhorei meu tempo — disse a atleta.",
        "— Por que as estrelas brilham? — perguntou a criança. — Porque são bolas enormes de gás quente — explicou a professora.",
        "— Posso ajudar na cozinha? — perguntou o menino. — Pode lavar os legumes com cuidado — respondeu a avó.",
        "— O ônibus já passou? — perguntou a jornalista. — Ainda não, ele chega em cinco minutos — disse o motorista.",
        "— Essa música é nova? — perguntou a vizinha. — Sim, compus ontem à noite — respondeu o músico.",
        "— O paciente está melhor? — perguntou a enfermeira. — Está respirando com mais calma — respondeu o médico.",
        "— Vamos plantar uma árvore? — sugeriu a menina. — Vamos cuidar dela todos os dias — respondeu o amigo.",
        "— O mar está agitado hoje? — perguntou o pescador. — Está, precisamos esperar o vento diminuir — disse o piloto.",
        "— Você terminou o desenho? — perguntou a arquiteta. — Terminei e acrescentei uma janela maior — disse o estudante.",
        "— A feira está cheia? — perguntou a mãe. — Está cheia de frutas maduras — respondeu a filha.",
        "— O livro te emocionou? — perguntou o avô. — Sim, parecia falar comigo — respondeu a neta.",
        "— Podemos observar a borboleta? — perguntou o menino. — Podemos, mas sem tocar nas asas — explicou a veterinária.",
        "— O que faremos nas férias? — perguntou o irmão. — Visitaremos a montanha e o lago — respondeu a irmã.",
        "— Você está triste? — perguntou a amiga. — Estou, mas conversar ajuda bastante — respondeu o colega.",
    ]
    frases.extend(dialogos * 5)

    # Fatos sobre o mundo
    fatos = [
        "A Terra gira em torno do Sol.",
        "A água ferve a cem graus Celsius ao nível do mar.",
        "O Brasil é o maior país da América do Sul.",
        "A Lua é o satélite natural da Terra.",
        "As plantas produzem oxigênio durante a fotossíntese.",
        "O coração bombeia sangue pelo corpo.",
        "Os peixes respiram principalmente por brânquias.",
        "As abelhas ajudam na polinização de muitas plantas.",
        "O arco-íris aparece quando a luz atravessa gotas de água.",
        "O inverno é uma das quatro estações do ano.",
        "O oceano cobre a maior parte da superfície da Terra.",
        "A Amazônia abriga enorme diversidade de seres vivos.",
        "O som precisa de um meio material para se propagar.",
        "A luz viaja mais rápido que o som.",
        "Os livros podem preservar histórias por muitos anos.",
        "A vacinação ajuda a prevenir doenças.",
        "O trânsito fica mais seguro com respeito às regras.",
        "A reciclagem reduz a quantidade de lixo descartado.",
        "O café é uma bebida muito consumida no Brasil.",
        "O futebol é um esporte praticado em muitos países.",
        "A música pode combinar ritmo, melodia e harmonia.",
        "A culinária brasileira mistura influências indígenas, africanas e europeias.",
        "As bibliotecas guardam livros, jornais e outros materiais de consulta.",
        "O calendário organiza dias, semanas, meses e anos.",
        "A bússola ajuda a indicar direções.",
        "O sol fornece luz e calor para a Terra.",
        "Os rios levam água doce por diferentes regiões.",
        "As montanhas podem se formar por movimentos da crosta terrestre.",
        "O corpo humano precisa de água para funcionar bem.",
        "A educação amplia oportunidades e conhecimentos.",
        "As nuvens são formadas por pequenas gotas de água ou cristais de gelo.",
        "Os mapas representam lugares e caminhos.",
        "O mel é produzido pelas abelhas a partir do néctar das flores.",
        "A gravidade atrai os objetos em direção à Terra.",
        "O relógio mede a passagem do tempo.",
    ]
    frases.extend(fatos * 6)

    # Descrições de cenas, objetos e ambientes
    descricoes = [
        "O céu estava azul e sem nuvens.",
        "A casa tinha uma porta vermelha e janelas grandes.",
        "O jardim cheirava a terra molhada e flores recém-abertas.",
        "A rua estreita tinha pedras antigas e postes amarelos.",
        "O mercado estava cheio de frutas coloridas e vozes animadas.",
        "A biblioteca era silenciosa, iluminada por lâmpadas suaves.",
        "O rio passava devagar entre árvores altas e sombras frescas.",
        "A montanha parecia dourada ao receber a luz do fim da tarde.",
        "O quarto era pequeno, mas organizado e cheio de livros.",
        "A cozinha tinha panelas brilhantes e cheiro de bolo assado.",
        "O estádio vibrava com bandeiras, tambores e cantos da torcida.",
        "A praia estava calma, com ondas baixas e areia clara.",
        "A floresta era densa, úmida e repleta de sons escondidos.",
        "O museu tinha corredores amplos e quadros de cores profundas.",
        "A estação parecia apressada, com malas, anúncios e passos rápidos.",
        "O hospital tinha paredes claras e corredores muito limpos.",
        "A fazenda exibia campos verdes, cercas de madeira e um céu imenso.",
        "O escritório tinha mesas alinhadas e plantas perto da janela.",
        "A caverna era fria, escura e cheia de ecos.",
        "O deserto se estendia em ondas de areia sob o sol forte.",
        "A neve cobria telhados, ruas e galhos com silêncio branco.",
        "O porto tinha barcos coloridos, redes secando e cheiro de sal.",
        "A sala de aula estava enfeitada com cartazes e desenhos dos estudantes.",
        "O restaurante tinha luz baixa, música tranquila e mesas bem postas.",
    ]
    frases.extend(descricoes * 8)

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