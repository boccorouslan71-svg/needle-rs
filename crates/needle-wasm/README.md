<div align="center">
  <img src="https://raw.githubusercontent.com/Geekgineer/needle-rs/main/assets/banner.svg" alt="needle-rs" width="100%"/>
</div>

<div align="center">
  <a href="https://needle-rs.pages.dev"><b>Live demo</b></a> ·
  <a href="https://github.com/Geekgineer/needle-rs">GitHub</a> ·
  <a href="https://github.com/Geekgineer/needle-rs/blob/main/docs/wasm-integration.md">Integration guide</a>
</div>

# needle-rs

A working tool-calling LLM in **414 KB of WebAssembly** — 163 KB gzipped. Runs in
the browser, Node.js, Deno, Bun and Cloudflare Workers. No server, no API key, no
data leaving the device.

This is the WebAssembly build of [needle-rs](https://github.com/Geekgineer/needle-rs),
a pure-Rust runtime for [Cactus Compute's](https://github.com/cactus-compute/needle)
Needle models. Both generations are supported from the same module, and output is
verified token-exact against the upstream JAX reference.

```bash
npm install needle-rs
```

## Quick start — Needle v2

One file carries the weights, the geometry and the tokenizer.

```js
import init, { NeedleV2Wasm } from "needle-rs";

await init();

const res = await fetch("https://huggingface.co/Cactus-Compute/needle2/resolve/main/needle2.cact");
const engine = NeedleV2Wasm.load(new Uint8Array(await res.arrayBuffer()));

const tools = JSON.stringify([{
  name: "get_weather",
  description: "Get current weather for a city",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
}]);

const query = "What's the weather in Paris?";
const out = engine.run(query, tools);
// <tool_call>[{"name":"get_weather","arguments":{"city":"Paris"}}]</tool_call>

const payload = engine.run_json(query, tools);
// [{"name":"get_weather","arguments":{"city":"Paris"}}]

// Gate on the model's confidence in the answer it just gave.
const p = engine.confidence_for(query, tools, out);
if (p >= 0.5) console.log(JSON.parse(payload));
```

`run_json` returns `"[]"` when no tool applies — a deliberate abstention worth
acting on — and `""` only in the degenerate case where no markers were emitted.

## Quick start — Needle v1

Separate weights and vocabulary.

```js
import init, { NeedleWasm } from "needle-rs";

await init();

const HF = "https://huggingface.co/Abdalrahman/needle-rs-safetensors/resolve/main";
const [weights, vocab] = await Promise.all([
  fetch(`${HF}/needle.safetensors`).then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
  fetch(`${HF}/vocab.txt`).then(r => r.text()),
]);

const engine = NeedleWasm.load(weights, vocab);   // undefined on failure
const result = engine.run("Book a flight from London to JFK tomorrow", tools);
```

v1 post-processes its output: the `<tool_call>` marker is stripped and your
original tool-name casing restored. With `run_stream`, the streamed pieces are a
progress view — the **returned** string is the answer.

## API

| | `NeedleV2Wasm` | `NeedleWasm` |
|---|---|---|
| `load` | `load(cactBytes)` | `load(weightsBytes, vocabText)` |
| Generate | `run`, `run_json`, `generate`, `run_stream` | `run`, `run_stream`, `run_batch` |
| Confidence | `confidence_for`, `confidence` | — |
| Retrieval | `contrastive_dim`, `encode_contrastive`, `retrieve_tools` | same |
| Constrained decode | ✓ (`generate(..., constrain=true)`) | always on |
| Sampling | ✓ (`temperature`, `seed`) | greedy only |

```js
// generate(query, tools, maxNewTokens, temperature, seed, constrain)
engine.generate(query, tools, 96, 0.0, 0, true);   // greedy, schema-constrained
engine.generate(query, tools, 96, 0.8, 42, false); // sampled, reproducible

// Narrow a large tool catalogue before routing.
engine.retrieve_tools(query, JSON.stringify(descriptions), 3);
// '[[0,0.9],[2,0.55],[1,0.31]]' — [index, score] pairs, descending
```

`confidence_for(query, tools, completion)` returns a probability in `(0, 1)`. The
confidence head scores a judgement already made, so pass the completion you got
back — feeding it a bare query reads near zero however good the answer is.

## Weights

Weights are **not** bundled; fetch them once and let the browser cache them.

| Version | File | Size | Source |
|---|---|---|---|
| v2 | `needle2.cact` | 13.7 MB | [`Cactus-Compute/needle2`](https://huggingface.co/Cactus-Compute/needle2) |
| v1 | `needle.safetensors` + `vocab.txt` | 22 MB + 122 KB | [`Abdalrahman/needle-rs-safetensors`](https://huggingface.co/Abdalrahman/needle-rs-safetensors) |

A generation session needs roughly 23 MB of WASM linear memory. Linear memory
never shrinks, so keep one engine per tab or isolate and reuse the handle.

## Notes

- Single-threaded in WASM by design; the `parallel` feature is native-only.
- Tool schemas are compacted internally, so pretty-printed and minified JSON give
  byte-identical output.
- Needle is a tool-calling router, not a chat model: one query plus tool
  definitions in, one JSON call out.

Full guide: [docs/wasm-integration.md](https://github.com/Geekgineer/needle-rs/blob/main/docs/wasm-integration.md).

## Credit and license

MIT. The **models** — architecture, training and weights — are the work of
[Cactus Compute](https://github.com/cactus-compute/needle) and are also MIT. If
you publish work using them, please cite Needle 2
([arXiv:2607.18363](https://arxiv.org/abs/2607.18363)); the entry is in the
[repository README](https://github.com/Geekgineer/needle-rs#citation).

This package is the runtime only.
