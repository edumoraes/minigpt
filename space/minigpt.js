// minigpt.js — O MiniGPT em JavaScript, pra gerar texto no navegador
//
// Porta pro JS o forward do model.py (blocos pre-norm, RoPE, atenção causal,
// GELU, weight tying), o BPETokenizer.codificar do tokenizer.py e a amostragem
// do generate.py. Lê os mesmos arquivos do repositório do modelo no Hugging
// Face: config.json, tokenizer.json e model.safetensors.
//
// A diferença pro Python é o KV cache: cada token novo calcula só a própria
// posição e reaproveita as chaves e valores das anteriores, em vez de rodar o
// contexto inteiro de novo. Quando o contexto enche, a janela recomeça com a
// metade mais recente dos tokens.
//
// Não usa DOM: roda num Web Worker ou no Node (pros testes).

const ESPECIAIS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"];
const UNK_ID = 1;
export const EOS_ID = 3;
// Igual ao LayerNorm do model.py
const EPS = 1e-5;

// Pré-tokenização do tokenizer.py (\S+|\s+), com os mesmos espaços do \s do Python
const ESPACOS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const CHUNKS = new RegExp(`[^${ESPACOS}]+|[${ESPACOS}]+`, "gu");

// ──────────────────────────────────────────────────────────
// Tokenizer BPE
// ──────────────────────────────────────────────────────────

export class Tokenizer {
  /** @param dados conteúdo do tokenizer.json */
  constructor(dados) {
    this.vocab = new Map(Object.entries(dados.vocab));
    this.idParaToken = [];
    for (const [id, token] of Object.entries(dados.id_to_token)) this.idParaToken[Number(id)] = token;
    // Ordem dos merges: o par aprendido mais cedo é aplicado primeiro
    this.ranks = new Map();
    dados.merges.forEach(([a, b], rank) => {
      const chave = a + "\u0000" + b;
      if (!this.ranks.has(chave)) this.ranks.set(chave, rank);
    });
    this.cache = new Map();
  }

  codificar(texto) {
    const ids = [];
    for (const chunk of texto.match(CHUNKS) ?? []) {
      let idsChunk = this.cache.get(chunk);
      if (!idsChunk) {
        idsChunk = this.codificarChunk(chunk);
        this.cache.set(chunk, idsChunk);
      }
      ids.push(...idsChunk);
    }
    return ids;
  }

  codificarChunk(chunk) {
    let pecas = Array.from(chunk); // por code point, como o list() do Python
    while (pecas.length > 1) {
      let melhorRank = Infinity;
      let par = null;
      for (let i = 0; i < pecas.length - 1; i++) {
        const rank = this.ranks.get(pecas[i] + "\u0000" + pecas[i + 1]);
        if (rank !== undefined && rank < melhorRank) {
          melhorRank = rank;
          par = [pecas[i], pecas[i + 1]];
        }
      }
      if (!par) break;
      // Junta todas as ocorrências do par, da esquerda pra direita
      const novas = [];
      for (let i = 0; i < pecas.length; i++) {
        if (i < pecas.length - 1 && pecas[i] === par[0] && pecas[i + 1] === par[1]) {
          novas.push(par[0] + par[1]);
          i++;
        } else {
          novas.push(pecas[i]);
        }
      }
      pecas = novas;
    }
    return pecas.map((peca) => this.vocab.get(peca) ?? UNK_ID);
  }

  decodificar(ids) {
    return ids
      .map((id) => this.idParaToken[id] ?? "")
      .filter((token) => !ESPECIAIS.includes(token))
      .join("");
  }
}

// ──────────────────────────────────────────────────────────
// Modelo
// ──────────────────────────────────────────────────────────

/** Lê os tensores float32 de um arquivo .safetensors (cabeçalho JSON + dados). */
function lerSafetensors(buffer) {
  const tamanho = Number(new DataView(buffer).getBigUint64(0, true));
  const cabecalho = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, tamanho)));
  const tensores = {};
  for (const [nome, info] of Object.entries(cabecalho)) {
    if (nome === "__metadata__") continue;
    if (info.dtype !== "F32") throw new Error(`${nome}: esperava F32, veio ${info.dtype}`);
    const inicio = 8 + tamanho + info.data_offsets[0];
    const fim = 8 + tamanho + info.data_offsets[1];
    // Float32Array direto no buffer só se o início estiver alinhado em 4 bytes
    tensores[nome] = inicio % 4 === 0
      ? new Float32Array(buffer, inicio, (fim - inicio) / 4)
      : new Float32Array(buffer.slice(inicio, fim));
  }
  return tensores;
}

/** y = W·x + b, com W no formato do nn.Linear (saída × entrada). */
function linear(y, W, b, x) {
  const nEntrada = x.length;
  for (let o = 0; o < y.length; o++) {
    let soma = b ? b[o] : 0;
    const base = o * nEntrada;
    for (let i = 0; i < nEntrada; i++) soma += W[base + i] * x[i];
    y[o] = soma;
  }
}

function layerNorm(y, x, [ganho, vies]) {
  const n = x.length;
  let media = 0;
  for (let i = 0; i < n; i++) media += x[i];
  media /= n;
  let variancia = 0;
  for (let i = 0; i < n; i++) variancia += (x[i] - media) ** 2;
  const inv = 1 / Math.sqrt(variancia / n + EPS);
  for (let i = 0; i < n; i++) y[i] = ganho[i] * (x[i] - media) * inv + vies[i];
}

/** Aproximação 7.1.26 de Abramowitz e Stegun (erro máximo 1,5e-7). */
function erf(x) {
  const sinal = Math.sign(x);
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return sinal * (1 - p * Math.exp(-a * a));
}

/** GELU exata, como o nn.GELU() padrão. */
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

export class GPT {
  /**
   * @param config conteúdo do config.json
   * @param buffer ArrayBuffer do model.safetensors
   */
  constructor(config, buffer) {
    if (!config.use_rope) throw new Error("Só modelos com RoPE");
    const t = lerSafetensors(buffer);
    this.d = config.d_model;
    this.nCabecas = config.n_heads;
    this.dCabeca = this.d / this.nCabecas;
    this.contexto = config.context_len;
    // Weight tying: o safetensors guarda o embedding uma vez só, com um dos dois nomes
    this.emb = t["token_embedding.weight"] ?? t["lm_head.weight"];
    this.vocab = this.emb.length / this.d;

    this.blocos = Array.from({ length: config.n_layers }, (_, l) => {
      const p = (nome) => t[`blocks.${l}.${nome}`];
      return {
        ln1: [p("ln1.gain"), p("ln1.bias")],
        Wqkv: p("attn.W_qkv.weight"), bqkv: p("attn.W_qkv.bias"),
        Wout: p("attn.W_out.weight"), bout: p("attn.W_out.bias"),
        cos: p("attn.rotary_emb.cos_cached"), sin: p("attn.rotary_emb.sin_cached"),
        ln2: [p("ln2.gain"), p("ln2.bias")],
        W1: p("ffn.net.0.weight"), b1: p("ffn.net.0.bias"),
        W2: p("ffn.net.2.weight"), b2: p("ffn.net.2.bias"),
        // KV cache: chaves e valores de cada posição já processada
        K: new Float32Array(this.contexto * this.d),
        V: new Float32Array(this.contexto * this.d),
      };
    });
    this.lnF = [t["ln_f.gain"], t["ln_f.bias"]];

    // Buffers reaproveitados a cada token
    this.x = new Float32Array(this.d);
    this.h = new Float32Array(this.d);
    this.qkv = new Float32Array(3 * this.d);
    this.atencao = new Float32Array(this.d);
    this.ffn = new Float32Array(4 * this.d);
    this.pesos = new Float64Array(this.contexto);
    this.logits = new Float32Array(this.vocab);
    this.pos = 0;
  }

  contarParametros() {
    let total = this.emb.length + this.lnF[0].length * 2;
    for (const b of this.blocos) {
      for (const tensor of [...b.ln1, b.Wqkv, b.bqkv, b.Wout, b.bout, ...b.ln2, b.W1, b.b1, b.W2, b.b2]) {
        total += tensor.length;
      }
    }
    return total;
  }

  /** Esvazia o KV cache: o próximo token volta a ser a posição 0. */
  reiniciar() {
    this.pos = 0;
  }

  /** RoPE: x·cos + rotate_half(x)·sin em cada cabeça, como no model.py. */
  rope(v, cos, sin) {
    const metade = this.dCabeca / 2;
    const linha = this.pos * this.dCabeca;
    for (let c = 0; c < this.nCabecas; c++) {
      const o = c * this.dCabeca;
      for (let i = 0; i < metade; i++) {
        const x1 = v[o + i];
        const x2 = v[o + i + metade];
        v[o + i] = x1 * cos[linha + i] - x2 * sin[linha + i];
        v[o + i + metade] = x2 * cos[linha + i + metade] + x1 * sin[linha + i + metade];
      }
    }
  }

  /**
   * Processa um token na próxima posição e devolve os logits do próximo.
   * O array devolvido é reaproveitado na chamada seguinte.
   */
  passo(token, comLogits = true) {
    if (this.pos >= this.contexto) throw new Error("Contexto cheio: chame reiniciar()");
    const { d, dCabeca, x, h, qkv, atencao, ffn, pesos, pos } = this;
    const escala = 1 / Math.sqrt(dCabeca);
    x.set(this.emb.subarray(token * d, token * d + d));

    for (const b of this.blocos) {
      // x = x + atenção(ln1(x))
      layerNorm(h, x, b.ln1);
      linear(qkv, b.Wqkv, b.bqkv, h);
      const q = qkv.subarray(0, d);
      const k = qkv.subarray(d, 2 * d);
      this.rope(q, b.cos, b.sin);
      this.rope(k, b.cos, b.sin);
      b.K.set(k, pos * d);
      b.V.set(qkv.subarray(2 * d), pos * d);
      for (let c = 0; c < this.nCabecas; c++) {
        const o = c * dCabeca;
        // Causal: a posição atual só enxerga ela mesma e as anteriores
        let maior = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let s = 0;
          for (let i = 0; i < dCabeca; i++) s += q[o + i] * b.K[t * d + o + i];
          pesos[t] = s * escala;
          if (pesos[t] > maior) maior = pesos[t];
        }
        let soma = 0;
        for (let t = 0; t <= pos; t++) {
          pesos[t] = Math.exp(pesos[t] - maior);
          soma += pesos[t];
        }
        for (let i = 0; i < dCabeca; i++) {
          let acc = 0;
          for (let t = 0; t <= pos; t++) acc += pesos[t] * b.V[t * d + o + i];
          atencao[o + i] = acc / soma;
        }
      }
      linear(h, b.Wout, b.bout, atencao);
      for (let i = 0; i < d; i++) x[i] += h[i];

      // x = x + ffn(ln2(x))
      layerNorm(h, x, b.ln2);
      linear(ffn, b.W1, b.b1, h);
      for (let j = 0; j < ffn.length; j++) ffn[j] = gelu(ffn[j]);
      linear(h, b.W2, b.b2, ffn);
      for (let i = 0; i < d; i++) x[i] += h[i];
    }
    this.pos++;
    if (!comLogits) return null;

    layerNorm(h, x, this.lnF);
    linear(this.logits, this.emb, null, h);
    return this.logits;
  }
}

// ──────────────────────────────────────────────────────────
// Amostragem
// ──────────────────────────────────────────────────────────

/**
 * Probabilidades do próximo token, na ordem do generate.py: temperatura,
 * penalidade de repetição (sobre o prompt e o que já saiu), top-k e top-p.
 */
export function distribuicao(logits, anteriores, { temperatura, topK, topP, penalidade }) {
  const n = logits.length;
  const l = Float64Array.from(logits, (v) => v / Math.max(temperatura, 1e-8));

  if (penalidade !== 1) {
    for (const id of new Set(anteriores)) {
      if (id < n) l[id] = l[id] > 0 ? l[id] / penalidade : l[id] * penalidade;
    }
  }

  // Top-k: fica com os k maiores (e com os empatados com o k-ésimo)
  if (topK > 0 && topK < n) {
    const kEsimo = Float64Array.from(l).sort().reverse()[topK - 1];
    for (let i = 0; i < n; i++) if (l[i] < kEsimo) l[i] = -Infinity;
  }

  const probs = softmax(l);

  // Top-p: do mais provável pro menos, fica enquanto a soma antes dele não passa de p
  if (topP < 1) {
    const ordem = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
    let acumulado = 0;
    for (let j = 0; j < n; j++) {
      const id = ordem[j];
      if (j > 0 && acumulado > topP) l[id] = -Infinity;
      acumulado += probs[id];
    }
    return softmax(l);
  }
  return probs;
}

function softmax(l) {
  let maior = -Infinity;
  for (const v of l) if (v > maior) maior = v;
  const p = new Float64Array(l.length);
  let soma = 0;
  for (let i = 0; i < l.length; i++) {
    p[i] = Math.exp(l[i] - maior);
    soma += p[i];
  }
  for (let i = 0; i < l.length; i++) p[i] /= soma;
  return p;
}

/** Sorteia um índice de acordo com as probabilidades. */
export function sortear(probs, aleatorio = Math.random) {
  let r = aleatorio();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

/** Recomeça o KV cache com os últimos tokens e devolve os logits do próximo. */
function recomecar(modelo, tokens) {
  modelo.reiniciar();
  const janela = tokens.slice(-modelo.contexto);
  for (const token of janela.slice(0, -1)) modelo.passo(token, false);
  return modelo.passo(janela.at(-1));
}

/**
 * Continua o prompt, um token por vez: cada iteração devolve o texto do token novo.
 * Para no <EOS> ou depois de opcoes.maxTokens.
 */
export function* gerar(modelo, tokenizer, prompt, opcoes, aleatorio = Math.random) {
  const tokens = tokenizer.codificar(prompt);
  if (tokens.length === 0) return;
  let logits = recomecar(modelo, tokens);
  for (let i = 0; i < opcoes.maxTokens; i++) {
    const proximo = sortear(distribuicao(logits, tokens, opcoes), aleatorio);
    if (proximo === EOS_ID) return;
    tokens.push(proximo);
    yield tokenizer.decodificar([proximo]);
    logits = modelo.pos === modelo.contexto
      ? recomecar(modelo, tokens.slice(-modelo.contexto / 2))
      : modelo.passo(proximo);
  }
}
