# Browser demo

A single self-contained HTML page that runs Needle inference entirely
client-side — **both model versions, side by side**. No server, no bundler, no
API key. This is the source for the deployed demo at
**[needle-rs.pages.dev](https://needle-rs.pages.dev)**.

Pick the version with the model cards at the top. The page reads the loaded
version's capabilities and enables only the controls that apply, so switching
between them is not a different code path — it is the same UI over two engines.

## What it demonstrates

- **One file to load (v2).** A `.cact` container carries the weights, the
  architecture geometry *and* the tokenizer, so the page fetches 13.7 MB and
  nothing else — no vocabulary, no config side-car. v1 needs two fetches:
  22 MB of weights plus a vocabulary.
- **Tool calling**, streamed token by token, with the `<think>` reasoning trace
  shown separately from the parsed call. v1 emits no reasoning trace, and
  post-processes its output — so the demo renders streamed pieces as they
  arrive, then settles on the returned string, which is the answer.
- **Grammar-constrained decoding** (v2), toggleable, so the payload cannot name
  an undeclared tool or argument key. v1 is always constrained and greedy-only.
- **The probe heads.** Contrastive tool retrieval works on both versions.
  Confidence scoring is v2-only — v1 carries no confidence head.

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
sampled), `run_stream`, `confidence_for`, `confidence`, `encode_contrastive` and
`retrieve_tools` against the real weights — 36 assertions across both versions,
and it runs in CI.

## Notes

- The probe heads attend over the whole sequence rather than a sliding window, so
  they allocate a larger KV cache than generation does. That is why the demo puts
  them behind an **Analyse** button instead of running them on every query.
- Constrained decoding is not streamed: the grammar mask depends on the whole
  payload emitted so far, so that path returns the finished text.
- The confidence head scores a completed judgement, not a question, so
  **Analyse** scores the prompt together with the model's own output via
  `confidence_for`. Passing the bare query instead reads near zero however good
  the answer is.

## Deployment

`.github/workflows/wasm-demo.yml` builds and deploys this to Cloudflare Pages on
every push to `main`, and `release.yml` does the same on a tag. Both need
`CF_API_TOKEN` and `CF_ACCOUNT_ID` as repository secrets — see
[docs/RELEASING.md](../../docs/RELEASING.md).
