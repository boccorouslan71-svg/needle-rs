# CLI quickstart

Runs the `needle-rs` CLI against a single tool definition, on either model
version. Missing weights are fetched from HuggingFace.

```bash
./run.sh                              # Needle v2 (default)
./run.sh "Email bob@example.com"      # Needle v2, your query
./run.sh --v1 "Weather in Paris?"     # Needle v1
./run.sh --both "Weather in Paris?"   # both, to compare
```

## The two invocations differ

A v2 `.cact` container carries the weights, the architecture geometry **and** the
tokenizer, so it takes one path and no vocabulary argument:

```bash
needle-rs --constrain weights/needle2.cact "$QUERY" "$TOOLS"
```

v1 keeps the tokenizer outside the weights, so it takes two:

```bash
needle-rs weights/needle.safetensors weights/vocab.txt "$QUERY" "$TOOLS"
```

The CLI dispatches on the file extension, so you never pass a version flag.

## Flags worth knowing (v2 only)

| Flag | Effect |
|---|---|
| `--constrain` | Restrict the payload to the declared tool names and argument keys |
| `--json` | Print only the tool-call payload, not the full decoded text |
| `--stream` | Print tokens to stderr as they are generated |
| `--max-tokens N` | Generation cap (default 128) |
| `--temperature T` | `0` is greedy; above that, sampling with `--seed` |
| `--system TEXT` | Prepend a system message |
| `--prefill-chunk N` | Positions per batched-prefill chunk; `0` disables batching |

Passing any of these with v1 weights is rejected rather than ignored — v1 is
always constrained and greedy-only.
