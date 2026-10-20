// worker.js — Roda o modelo fora da thread da página, pra interface não travar
//
// Mensagens da página: {tipo: "carregar", base}, {tipo: "gerar", prompt, opcoes}, {tipo: "parar"}
// Respostas: progresso, pronto, token, fim e erro

import { GPT, Tokenizer, gerar } from "./minigpt.js";

let modelo = null;
let tokenizer = null;
let parar = false;

// Devolve o controle pro event loop sem o atraso mínimo do setTimeout,
// pra uma mensagem "parar" ser atendida entre um token e outro
const canal = new MessageChannel();
const ceder = () => new Promise((resolve) => {
  canal.port1.onmessage = () => resolve();
  canal.port2.postMessage(null);
});

async function baixar(url, aoProgresso) {
  const resposta = await fetch(url);
  if (!resposta.ok) throw new Error(`${url}: HTTP ${resposta.status}`);
  const total = Number(resposta.headers.get("content-length")) || 0;
  const leitor = resposta.body.getReader();
  const partes = [];
  let recebidos = 0;
  for (;;) {
    const { done, value } = await leitor.read();
    if (done) break;
    partes.push(value);
    recebidos += value.length;
    aoProgresso?.(recebidos, total);
  }
  const bytes = new Uint8Array(recebidos);
  let pos = 0;
  for (const parte of partes) {
    bytes.set(parte, pos);
    pos += parte.length;
  }
  return bytes.buffer;
}

const json = async (url) => JSON.parse(new TextDecoder().decode(await baixar(url)));

async function carregar(base) {
  const [config, dadosTokenizer] = await Promise.all([
    json(`${base}/config.json`),
    json(`${base}/tokenizer.json`),
  ]);
  const pesos = await baixar(`${base}/model.safetensors`, (recebidos, total) =>
    postMessage({ tipo: "progresso", recebidos, total }));
  modelo = new GPT(config, pesos);
  tokenizer = new Tokenizer(dadosTokenizer);
  postMessage({ tipo: "pronto", parametros: modelo.contarParametros(), contexto: modelo.contexto });
}

async function gerarTexto(prompt, opcoes) {
  parar = false;
  const inicio = performance.now();
  let n = 0;
  for (const texto of gerar(modelo, tokenizer, prompt, opcoes)) {
    postMessage({ tipo: "token", texto });
    n++;
    await ceder();
    if (parar) break;
  }
  postMessage({ tipo: "fim", tokens: n, segundos: (performance.now() - inicio) / 1000 });
}

onmessage = async ({ data }) => {
  try {
    if (data.tipo === "carregar") await carregar(data.base);
    else if (data.tipo === "gerar") await gerarTexto(data.prompt, data.opcoes);
    else if (data.tipo === "parar") parar = true;
  } catch (erro) {
    postMessage({ tipo: "erro", mensagem: String(erro.message ?? erro) });
  }
};
