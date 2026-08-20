# Browser demo

A single self-contained HTML page that runs Needle v2 inference entirely
client-side. No server, no bundler, no API key. This is the source for the
deployed demo at **[needle-rs.pages.dev](https://needle-rs.pages.dev)**.

## What it demonstrates

- **One file to load.** A `.cact` container carries the weights, the architecture
  geometry *and* the tokenizer, so the page fetches 13.7 MB and nothing else —
  no vocabulary, no config side-car.
- **Tool calling**, streamed token by token, with the `<think>` reasoning trace
  shown separately from the parsed call.
- **Grammar-constrained decoding**, toggleable, so the payload cannot name an
  undeclared tool or argument key.
- **The two probe heads** — confidence scoring and contrastive tool retrieval —
  which is what v1 could not do in a browser at all.

The spec cards read measured values: the runtime size comes from the actual
`PerformanceResourceTiming` entry for the wasm module and the model size from the
bytes received, so they cannot drift from reality.

## Running locally

Build the WASM package, then serve:

```bash
wasm-pack build crates/needle-wasm --target web --release --out-dir ../../pkg/
./examples/browser-demo/serve.sh
# → http://localhost:8080
```

`serve.sh` copies `pkg/` next to `index.html` in a temp directory, mirroring the
CI deployment layout. The model streams from HuggingFace, so no local weights are
needed.

## Verifying the bindings

The page only calls methods that are covered by a test:

```bash
wasm-pack build crates/needle-wasm --target nodejs --release --out-dir ../../pkg-nodejs/
node crates/needle-wasm/tests/node_e2e_v2.js
```

That exercises `load`, `run`, `run_json`, `generate` (greedy, constrained and
sampled), `run_stream`, `confidence`, `encode_contrastive` and `retrieve_tools`
against the real container, and runs in CI.

## Notes

- The probe heads attend over the whole sequence rather than a sliding window, so
  they allocate a larger KV cache than generation does. That is why the demo puts
  them behind an **Analyse** button instead of running them on every query.
- Constrained decoding is not streamed: the grammar mask depends on the whole
  payload emitted so far, so that path returns the finished text.

## Deployment

`.github/workflows/wasm-demo.yml` builds and deploys this to Cloudflare Pages on
every push to `main`. A Cloudflare API token is required; see the workflow.
