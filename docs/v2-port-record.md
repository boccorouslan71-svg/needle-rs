# needle-rs 0.2.0 — Needle v2 port record

Upstream `needle` moved from the v1 encoder–decoder SAN to a decoder-only v2
architecture with a new weight container. needle-rs 0.1.0 ported v1; 0.2.0 adds
v2 alongside it. This is the record of that port — what changed upstream, the
geometry the shipped container actually declares, and which upstream files were
treated as the spec. For how the result is structured see
[ARCHITECTURE.md](../ARCHITECTURE.md); for what landed, [CHANGELOG.md](../CHANGELOG.md).

## What actually changed upstream

| | v1 (needle-rs 0.1.0 ports this) | v2 (`Cactus-Compute/needle2`) |
|---|---|---|
| Topology | encoder + decoder w/ cross-attn | decoder-only, 27 layers |
| Residual stream | single | mHC, 4 lanes, Sinkhorn-normalised lane mixing |
| MLP | DReLU / SwiGLU / GeGLU FFN | HadamardMLP (`d3 * silu(d2 * ((d1*z) @ H)) @ H`) |
| Associative memory | — | Engram: n-gram hash tables at layers 2 and 15 |
| Attention | GQA + q/k ZCRMSNorm | GQA + q/k ZCRMSNorm + sigmoid `gate_proj` |
| Weight quant | symmetric INT4, group 32, `absmax/7` | Cactus-Quants: Hadamard-rotated Lloyd-Max codebooks, mixed 4/2 bit, group 128 |
| Container | `.safetensors` + `vocab.txt` | `.cact` (geometry header + nameless positional directory + embedded SP-BPE) |
| Aux heads | contrastive | contrastive + confidence (+ MTP, training only) |

Zero shared weight tensors, zero shared loader, zero shared quant scheme.
0.2.0 therefore added v2 as a **parallel** path; the v1 engine and its tests
stayed untouched.

## Shipped geometry (read from `needle2.cact`, 13,737,807 bytes)

```
vocab 8192   d_model 512   heads 8   kv_heads 4   head_dim 64
layers 27    max_seq_len 2048        rope_theta 100000.0
hada_n 512   mhc_lanes 4
engram: slots 8192  sub_dim 128  tables 4  taps 4  dilation 3
        orders (2,3)  sites (2,15)
kv_window 256  kv_bits 8   codebook_len 28 (cb2|cb3|cb4)
405 tensors: 141 CQ@2bit, 4 CQ@4bit (embedding + 3 mhc_phi), 259 FP16, 1 RAW
```

## The kernel identity that drives the design

A CQ group reconstructs as `w_g = (cb[idx_g] * norm_g) @ H`, `H = Walsh(g)/sqrt(g)`.
`H` is symmetric, so for an activation slice `x_g`:

```
dot(x_g, w_g) = dot(x_g, u_g @ H) = dot(H @ x_g, u_g)      u_g = cb[idx_g] * norm_g
```

So the Hadamard rotation moves off the **weights** (once per output row, per
group — `out * in_pad` work) and onto the **activations** (once per group for
the whole matrix). Two consequences:

1. Weights are never reconstructed. The GEMV inner loop is a dot product of
   codebook-decoded bytes against a pre-transformed activation slice.
2. Use a fast Walsh–Hadamard transform, not a dense matmul. At group 128 that
   is `128*7 = 896` add/sub versus `128*128 = 16384` MACs.

The same FWHT replaces the two dense `512x512` Hadamard matmuls inside
HadamardMLP: `2 * 512*9 = 9216` add/sub versus `2 * 262144` MACs per layer per
token, across 27 layers.

## The check that pins the canon

Per-tensor parity cannot catch a canon error: the directory is nameless, so a
*correct* tensor can still land in the wrong positional slot and every individual
comparison still passes.

`tools/cact_params.py` inverts `export._tensors` and rebuilds the Flax parameter
tree straight from `needle2.cact`, so upstream's own `decode.forward_cached` can
run the shipped weights. Fed the documented v2 prompt template

```
<|im_start|>user\n<tools>{tools_json}</tools>\n{query}<|im_end|>\n<|im_start|>assistant\n
```

it produces, greedily:

```
Q: What's the weather in Paris?
A: <tool_call>[{"name":"get_weather","arguments":{"city":"Paris"}}]</tool_call><|im_end|>
```

Coherent tool calls — and chain-of-thought on harder queries — mean the canon, the
mixed-width dequantisation and the tokenizer are all right end to end. That
reconstruction is also the oracle the Rust forward pass is verified against.

## What landed in 0.2.0

All of it verified against `needle2.cact` unless noted. `cargo test --workspace`
is green and `cargo clippy --workspace --all-targets -- -D warnings` is clean.

- **Container** — `needle-infer::cact`. Header, codebook, and the nameless
  positional directory. The canon is derived from the header geometry and every
  slot's shape is validated against it, so a layout drift fails at load rather
  than producing wrong logits. Checked field for field against
  `export.read_export()`.
- **Cactus-Quants** — `needle-core::cq`. 2/3/4-bit and ternary. All 145 CQ
  tensors match upstream's dequantised product.
- **FWHT** — `needle-core::hadamard`.
- **Tokenizer** — `needle-infer::sp_tokenizer`. Exact token-ID match on a 44-case
  corpus against both `RefTokenizer` and real `sentencepiece`.
- **Forward pass** — `needle-core::v2`. mHC lanes with Sinkhorn mixing, Engram,
  HadamardMLP, gated GQA attention, sliding-window KV cache, tied LM head.
  All 788 capture points across 27 layers match `decode.forward_cached` on the
  same weights; largest relative deviation 1.9e-5. Greedy continuation matches
  token for token.
- **Engine and CLI** — `needle-infer::v2_engine`, `needle-cli`. Prompt template,
  greedy and temperature sampling, streaming, `<tool_call>` extraction.
- **Batched prefill** — `needle-core::v2::batch`. Projections evaluated once per
  chunk instead of once per position, bit-identical to the sequential path.
- **Probe heads** — `needle-core::v2::heads`. Contrastive and confidence, pooled
  over per-layer cells with a streaming softmax so nothing is retained. Verified
  to 1.4e-6 / 7.2e-5 against the reference, including inputs past the attention
  window (the heads run full causal; the LM path does not).
- **Constrained decoding** — the v1 JSON state machine over a v2 token table,
  engaged only inside `<tool_call>`.
- **Bindings** — C ABI (`needle_v2_*`), WASM (`NeedleV2Wasm`), Python
  (`needle_rs.V2Engine`).
- **KV ring** — the cache is sized to the attention window, not `max_seq_len`:
  14.16 MB against 113.25 MB, so a generation session is ~23 MB of working memory.
  The probe heads allocate the full length via `make_state_full_causal`, and
  `set_kv_window` refuses a window wider than the ring rather than reading stale
  slots.
- **Threading** — `parallel` feature (rayon), on for the CLI, C ABI and Python
  wheel, off for WASM and `no_std`. Splits large row ranges and the
  batched-prefill per-position loops; bit-identical, since row splitting leaves
  each row's arithmetic untouched. Decode is deliberately excluded — threading its
  small matvecs measured 3.5x *slower*.
- **End-to-end parity gate** — `tests/v2_e2e_parity.rs`, the v1 equivalent: exact
  generated token ids over 14 prompt/tool combinations, 2482 tokens, longest
  prompt 364 (past the window). This is what the ring and threading were verified
  against, and it caught two real write-ordering bugs the ring introduced.
- **Performance** — 2.9x / 2.8x on the packed dot product from splitting the
  accumulator into lanes; prefill 3.6x from batching and threading. Against the
  JAX reference: decode 1.35x faster, crossover at 48 generated tokens, ~50x
  faster cold start. Numbers, method, and the optimisations that were measured and
  *rejected* (including hand-written NEON, which lost to LLVM) are in
  `BENCHMARKS.md`.

## Still open

- **Prefill is slower than a fused dense-f32 GEMM reference** (1.60 vs 0.51
  ms/token). Structural, not a gap to close: the packed kernel already reaches
  38 Gelem/s threaded, and the deficit is the cost of running from 13.7 MB of
  2-bit weights instead of ~180 MB dequantised. Closing it would mean giving up
  the memory advantage.
- **A tool call can repeat an argument key once every declared key is used.** By
  then the model has emitted the `,` that commits it to another key; forcing the
  object closed would need the grammar extended to the value boundary, which
  `JsonStateMachine` does not model. No repeat occurs while an unused key remains.
- **AVX2 throughput is unmeasured.** Correctness — including that the dispatch is
  actually taken — is asserted on CI's x86_64 runners. No figure is claimed: this
  machine is aarch64, Rosetta masks AVX2, and qemu user-mode does not exist for a
  macOS host.
- **MTP head**: nothing to port. `export.py` writes no MTP tensors, so the
  container carries none; it exists only in the training graph.
- **Confidence is the head only, not upstream's composed score.** `apis.md`
  defines the published `confidence` as the *minimum* of the calibration head
  and the decode probability of the call tokens. needle-rs ports the head — it
  matches `forward_confidence` exactly — and exposes it as `confidence_for`,
  which feeds it the prompt-plus-completion input it was trained on. The
  composition is not replicated: it lives in Cactus's compiled engine, so there
  is no reference to verify a port against, and an unverifiable calibrated
  number is worse than a documented gap. Upstream also notes the calibration
  holds for the base model only.

## Fixtures and how to regenerate them

Everything below needs the checkpoint, which is gitignored:

```
mkdir -p weights
curl -sSLo weights/needle2.cact \
  https://huggingface.co/Cactus-Compute/needle2/resolve/main/needle2.cact
```

The JAX-based generators need a separate interpreter — `jax` has no wheels for
the current system Python:

```
uv venv --python 3.12 .venv-parity
uv pip install --python .venv-parity/bin/python "jax[cpu]" flax numpy sentencepiece
```

| Fixture | Tracked | Generator |
|---|---|---|
| `tests/cact_vectors.json` (651 KB) | yes | `tools/gen_cact_parity.py` |
| `tests/tokenizer_vectors.json` (14 KB) | yes | `tools/gen_tokenizer_parity.py` |
| `tests/v2_forward_vectors.json` + `.f32` (1.8 MB) | **no** | `tools/gen_v2_forward_parity.py` |
| `tests/v2_heads_vectors.json` (45 KB) | yes | `tools/gen_v2_heads_parity.py` |
| `tests/v2_e2e_vectors.json` (15 KB) | yes | `tools/gen_v2_e2e_parity.py` |

The forward-pass ladder is gitignored on size, following the same call already
made for `tests/e2e_vectors.json`. Every parity test skips with a printed notice
when its inputs are absent, so a fresh clone runs `cargo test` clean, and CI runs
the two tracked-fixture suites against a freshly downloaded checkpoint.

```
PYTHONPATH=needle:tools JAX_PLATFORMS=cpu .venv-parity/bin/python \
    tools/gen_v2_forward_parity.py
```

## Reference files (upstream, treat as spec)

- `needle/model/export.py` — `.cact` byte layout, `_cq_unpack`, `RefTokenizer`
- `needle/model/quantize.py` — `_cq_codebook_np`, `_cq_hadamard_np`
- `needle/model/decode.py` — `_forward_cached`, `_engram_kv`, `_mhc`
- `needle/model/architecture.py` — training graph, `kv_budget_window`
