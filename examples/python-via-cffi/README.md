# Python via the C ABI

Runs Needle from Python with **no ML dependencies** — standard library and
`ctypes` only, against the `needle-rs` C shared library. Both model versions are
supported, and the version is inferred from the model path.

```bash
cargo build --release -p needle-c

# Needle v2 — one file, tokenizer included
python infer.py --constrain

# Needle v2 — confidence gating (v1 has retrieval, but no confidence head)
python infer.py --analyse

# Needle v2 — streaming
python infer.py --stream

# Needle v1 — weights plus a separate vocabulary
python infer.py --model ../../weights/needle.safetensors --vocab ../../weights/vocab.txt
```

## What it shows

- Declaring both C surfaces: `needle_*` (v1) and `needle_v2_*` (v2), which share
  `needle_free_str` and `needle_last_error`.
- Correct ownership across the boundary. Every string the library allocates is
  declared `restype = c_void_p`, **not** `c_char_p`: ctypes converts a `c_char_p`
  result into a Python `bytes` and discards the pointer, so passing that to
  `needle_free_str` frees Python-owned memory and aborts the process. Read
  through a cast, free the original pointer — see `take()`.
- A streaming callback via `CFUNCTYPE`, with the reference kept alive.
- Confidence gating done correctly. The head scores a completed judgement, so
  `--analyse` calls `needle_v2_confidence_for` with the prompt *and* the model's
  own output. The lower-level `needle_v2_confidence` takes bare text and reads
  near zero for a query however answerable it is — that is the primitive, not a
  bug.
- Capability differences handled rather than assumed: v2-only flags are ignored
  with a warning on v1 instead of failing.

## Two behaviours worth knowing

**v1 post-processes its output.** The returned text has the `<tool_call>` marker
stripped and the caller's original tool-name casing restored, so a streamed
`get_weather` becomes `getWeather` in the result. The streamed tokens are a
progress view; the return value is the answer. Run `--stream` against v1 to see
both.

**v2 wraps the payload in markers.** `needle_v2_run` returns the full decoded
text including `<think>` and `<tool_call>`; `needle_v2_run_json` returns just the
payload.

## Prefer the native wheel

For real Python use, `pip install needle-rs` gives you `NeedleEngine` and
`V2Engine` directly — no ctypes, no library path. This example exists to
document the C ABI for callers in other languages.
