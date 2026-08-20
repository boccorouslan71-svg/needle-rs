// Web Worker that owns the NeedleWasm engine.
//
// The engine's `run()` is a synchronous, multi-second wasm call. Running it
// on the main thread freezes the UI (no animation, no input, no scroll).
// Moving it here means the main thread is only blocked when serializing
// arguments and deserializing the result — milliseconds, not seconds.
//
// The DOM walker, tool execution, and UI all stay on the main thread; this
// worker only knows about the model.

import init, { NeedleWasm, NeedleV2Wasm } from "needle-rs";

const HF_V1 = "https://huggingface.co/Abdalrahman/needle-rs-safetensors/resolve/main";
const HF_V2 = "https://huggingface.co/Cactus-Compute/needle2/resolve/main";

export type ModelVersion = "v1" | "v2";

type LoadStage = "init" | "weights" | "vocab" | "engine" | "ready";

type InMsg =
  | { id: number; type: "load"; version: ModelVersion }
  | { id: number; type: "infer"; query: string; toolsJson: string }
  | { id: number; type: "retrieve"; query: string; descriptions: string[]; topK: number }
  | { id: number; type: "hasContrastive" };

type OutMsg =
  | { id: number; type: "progress"; stage: LoadStage; loadedBytes?: number; totalBytes?: number }
  | { id: number; type: "result"; data: unknown }
  | { id: number; type: "error"; message: string };

/** Uniform surface over both versions, so the harness never branches on it. */
interface Engine {
  version: ModelVersion;
  /** The tool-call payload, with v2's markers already stripped. */
  infer(query: string, toolsJson: string): string;
  retrieve(query: string, descriptionsJson: string, topK: number): string;
  contrastiveDim(): number;
}

let engine: Engine | null = null;

function post(msg: OutMsg): void {
  (self as unknown as Worker).postMessage(msg);
}

async function fetchWithProgress(
  url: string,
  hintBytes: number,
  id: number,
  stage: LoadStage,
): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
  const total = parseInt(resp.headers.get("content-length") || "0") || hintBytes;
  const reader = resp.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ id, type: "progress", stage, loadedBytes: loaded, totalBytes: total });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  return buf;
}

/** v2 wraps the payload in `<tool_call>`; v1 emits it bare. */
function stripMarkers(text: string): string {
  const open = text.indexOf("<tool_call>");
  if (open < 0) return text.trim();
  const rest = text.slice(open + "<tool_call>".length);
  const close = rest.indexOf("</tool_call>");
  return (close < 0 ? rest : rest.slice(0, close)).trim();
}

async function loadV2(id: number): Promise<Engine> {
  // One .cact carries weights, geometry and tokenizer, so there is no vocab step.
  const cact = await fetchWithProgress(`${HF_V2}/needle2.cact`, 13_737_807, id, "weights");
  post({ id, type: "progress", stage: "engine" });
  const e = NeedleV2Wasm.load(cact);
  if (!e) throw new Error("NeedleV2Wasm.load returned undefined");
  return {
    version: "v2",
    // Constrained: the harness generates tool names from the DOM, so the model
    // must not invent one. `generate(..., constrain = true)` enforces that.
    infer: (q, t) => stripMarkers(e.generate(q, t, 96, 0, 0, true)),
    retrieve: (q, d, k) => e.retrieve_tools(q, d, k),
    contrastiveDim: () => e.contrastive_dim(),
  };
}

async function loadV1(id: number): Promise<Engine> {
  const weights = await fetchWithProgress(`${HF_V1}/needle.safetensors`, 22 * 1024 * 1024, id, "weights");
  const vocabBytes = await fetchWithProgress(`${HF_V1}/vocab.txt`, 120 * 1024, id, "vocab");
  const vocab = new TextDecoder().decode(vocabBytes);
  post({ id, type: "progress", stage: "engine" });
  const e = NeedleWasm.load(weights, vocab);
  if (!e) throw new Error("NeedleWasm.load returned undefined");
  return {
    version: "v1",
    // v1 is always constrained and already returns the payload bare.
    infer: (q, t) => stripMarkers(e.run(q, t)),
    retrieve: (q, d, k) => e.retrieve_tools(q, d, k),
    contrastiveDim: () => e.contrastive_dim(),
  };
}

async function handleLoad(id: number, version: ModelVersion): Promise<void> {
  post({ id, type: "progress", stage: "init" });
  await init();

  const e = version === "v2" ? await loadV2(id) : await loadV1(id);
  engine = e;

  post({ id, type: "progress", stage: "ready" });
  post({ id, type: "result", data: { contrastiveDim: e.contrastiveDim(), version } });
}

self.addEventListener("message", async (ev: MessageEvent<InMsg>) => {
  const msg = ev.data;
  try {
    switch (msg.type) {
      case "load":
        await handleLoad(msg.id, msg.version);
        return;
      case "infer": {
        if (!engine) throw new Error("model not loaded");
        const result = engine.infer(msg.query, msg.toolsJson);
        post({ id: msg.id, type: "result", data: result });
        return;
      }
      case "retrieve": {
        if (!engine) throw new Error("model not loaded");
        const raw = engine.retrieve(msg.query, JSON.stringify(msg.descriptions), msg.topK);
        let parsed: Array<{ index: number; score: number }> = [];
        try {
          parsed = JSON.parse(raw);
        } catch {
          /* leave empty */
        }
        post({ id: msg.id, type: "result", data: parsed });
        return;
      }
      case "hasContrastive": {
        if (!engine) throw new Error("model not loaded");
        post({ id: msg.id, type: "result", data: engine.contrastiveDim() > 0 });
        return;
      }
    }
  } catch (e) {
    post({ id: msg.id, type: "error", message: (e as Error).message });
  }
});
