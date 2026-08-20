#!/usr/bin/env python3
"""
Generate v2 forward-pass parity vectors from a real `.cact` checkpoint.

Runs upstream's incremental decode path (`needle/model/decode.py`) on weights
rebuilt from the blob by `tools/cact_params.py`, and dumps a bisection ladder:
per-layer mHC and block intermediates plus the final logits, for a prefill and
several single-token decode steps.

The dump comes from `_instrumented_forward`, which is `decode._forward_cached`
with capture hooks. It calls upstream's own `_layer`, `_mhc`, `_block_cached`,
`_engram_kv`, `_rms_unit` and `_sinkhorn`, so none of the mathematics is
re-derived here — only the ~25-line orchestration loop is restated. To keep that
restatement honest, every step asserts the instrumented logits equal
`decode.forward_cached`'s before anything is written.

Output is split so the fixture stays small:
    tests/v2_forward_vectors.json  — manifest, shapes, offsets, token ids
    tests/v2_forward_vectors.f32   — the arrays, little-endian f32, back to back

Usage (from the repo root):
    PYTHONPATH=needle:tools JAX_PLATFORMS=cpu .venv-parity/bin/python \\
        tools/gen_v2_forward_parity.py
"""

import argparse
import json
import math
import struct
import sys

import numpy as np

DEFAULT_TOOLS = (
    '[{"name":"get_weather","description":"Get current weather for a city",'
    '"parameters":{"type":"object","properties":{"city":{"type":"string"}},'
    '"required":["city"]}}]'
)
DEFAULT_QUERY = "What's the weather in Paris?"


class Blobs:
    """Append-only f32 store; the JSON side records name -> (offset, shape)."""

    def __init__(self):
        self.buf = bytearray()
        self.index = {}

    def put(self, name, arr):
        a = np.ascontiguousarray(np.asarray(arr, np.float32))
        if name in self.index:
            raise KeyError(f"duplicate blob {name}")
        self.index[name] = {"offset": len(self.buf) // 4, "shape": list(a.shape)}
        self.buf += a.tobytes()
        return name


def _instrumented_forward(D, params, cfg, tokens, k_cache, v_cache, pos, cos, sin,
                          hist, hist_valid, sink, capture):
    """`decode._forward_cached` with per-layer capture.

    Structurally identical to upstream; every mathematical step calls upstream's
    own helper. `capture(name, array)` records the last sequence position.
    """
    import jax.numpy as jnp
    import jax

    emb = params["embedding"]["embedding"].astype(jnp.float32)
    x = emb[tokens] * math.sqrt(cfg.d_model)
    B, S = tokens.shape
    cos_s = jax.lax.dynamic_slice_in_dim(cos, pos, S, axis=0)
    sin_s = jax.lax.dynamic_slice_in_dim(sin, pos, S, axis=0)
    capture("embed", x[0, -1])

    ekv = None
    if cfg.engram_layers:
        h = tokens if hist is None else hist
        hv = jnp.ones(h.shape, bool) if hist_valid is None else hist_valid
        ekv = D._engram_kv(params, cfg, h, hv,
                           jnp.asarray(0, jnp.int32) if hist is None else pos, S)
        capture("engram_k0", ekv[0][0, 0, -1])
        capture("engram_v0", ekv[1][0, 0, -1])

    n, C = cfg.mhc_lanes, cfg.d_model
    hc = D._mhc(params, cfg)
    x = jnp.broadcast_to(x[:, :, None, :], (B, S, n, C))

    new_k, new_v = [], []
    for i in range(cfg.num_layers):
        nx = D._rms_unit(x.reshape(B, S, n * C))
        hpre = jax.nn.sigmoid(hc["a_pre"][i] * (nx @ hc["phi_pre"][i])
                              + hc["b_pre"][i] + hc["pre_off"][i])
        u = jnp.einsum("btn,btnc->btc", hpre, x)
        bx = u
        if ekv is not None and i in cfg.engram_layers:
            site = cfg.engram_layers.index(i)
            ek, ev = ekv
            alpha = jax.nn.sigmoid(
                jnp.einsum("btd,btd->bt", D._rms_unit(u), D._rms_unit(ek[site]))
                / math.sqrt(C))
            bx = u + alpha[..., None] * ev[site]
            capture(f"L{i:02d}.alpha", alpha[0, -1:])
        y, kc, vc = D._block_cached(bx, D._layer(params, i), k_cache[i], v_cache[i],
                                    pos, cos_s, sin_s, cfg, None, False, sink)
        y = y - u
        hpost = 2 * jax.nn.sigmoid(hc["a_post"][i] * (nx @ hc["phi_post"][i])
                                   + hc["b_post"][i] + hc["post_off"][i])
        res = nx @ hc["phi_res"][i]
        hres = D._sinkhorn(hc["a_res"][i] * res.reshape(B, S, n, n) + hc["b_res"][i])
        x = (jnp.einsum("btij,btjc->btic", hres, x)
             + hpost[..., None] * y[:, :, None, :])
        capture(f"L{i:02d}.hpre", hpre[0, -1])
        capture(f"L{i:02d}.u", u[0, -1])
        capture(f"L{i:02d}.bx", bx[0, -1])
        capture(f"L{i:02d}.y", y[0, -1])
        capture(f"L{i:02d}.hpost", hpost[0, -1])
        capture(f"L{i:02d}.hres", hres[0, -1])
        capture(f"L{i:02d}.x", x[0, -1])
        new_k.append(kc)
        new_v.append(vc)

    x = jnp.mean(x, axis=2)
    capture("lane_mean", x[0, -1])
    x = D._zcrms(x, params["stack"]["final_norm"]["scale"])
    capture("final_norm", x[0, -1])
    logits = x @ emb.T
    capture("logits", logits[0, -1])
    return logits, jnp.stack(new_k), jnp.stack(new_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cact", default="weights/needle2.cact")
    ap.add_argument("--output", default="tests/v2_forward_vectors")
    ap.add_argument("--tools", default=DEFAULT_TOOLS)
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--decode-steps", type=int, default=3)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from cact_params import load_cact_params
    from needle.model import decode as D
    from needle.model.export import parse_tokenizer_blob, RefTokenizer
    from needle.model.architecture import precompute_rope_freqs
    from needle.model.tokenizer import IM_START, IM_END, TOOLS_START, TOOLS_END, BOS_ID, EOS_ID

    params, config, meta = load_cact_params(args.cact)
    params = D._f32(params)
    tok = RefTokenizer(parse_tokenizer_blob(meta["tokenizer_blob"]))

    prompt = (IM_START + "user\n" + TOOLS_START + args.tools + TOOLS_END + "\n"
              + args.query + IM_END + "\n" + IM_START + "assistant\n")
    prompt_ids = [BOS_ID] + tok.encode(prompt)
    max_len = min(config.max_seq_len, len(prompt_ids) + args.decode_steps + 1)

    head_dim = (config.attn_dim or config.d_model) // config.num_heads
    cos, sin = precompute_rope_freqs(head_dim, max_len, config.rope_theta)
    dcfg = D.decode_cfg(config, kv_window=config.kv_window)
    kc, vc = D.init_kv_cache(config, 1, max_len)

    hist = jnp.zeros((1, max_len), jnp.int32).at[0, :len(prompt_ids)].set(
        jnp.asarray(prompt_ids, jnp.int32))
    valid = jnp.ones((1, max_len), bool)
    # `_doc_prefix_len` returns 0 upstream, so the sink mask is all-false; keep it
    # explicit rather than None so the Rust port has the same shape to match.
    sink = jnp.zeros((1, max_len), bool) if config.kv_window else None

    blobs = Blobs()
    steps = []

    def run_step(step_name, toks, pos):
        nonlocal kc, vc
        captured = {}

        def capture(name, arr):
            captured[name] = np.asarray(arr, np.float32)

        logits, nkc, nvc = _instrumented_forward(
            D, params, dcfg, toks, kc, vc, jnp.asarray(pos, jnp.int32), cos, sin,
            hist, valid, sink, capture)

        # Guard the restated loop: it must agree with upstream's real function.
        ref_logits, ref_kc, ref_vc = D.forward_cached(
            params, dcfg, toks, kc, vc, jnp.asarray(pos, jnp.int32), cos, sin, None,
            False, hist, valid, sink)
        a = np.asarray(logits, np.float64)
        b = np.asarray(ref_logits, np.float64)
        if not np.allclose(a, b, rtol=0, atol=1e-3):
            raise SystemExit(
                f"{step_name}: instrumented forward diverges from "
                f"decode.forward_cached (max |d| = {np.abs(a - b).max():.3e})")
        kc, vc = ref_kc, ref_vc

        entry = {"name": step_name, "pos": int(pos),
                 "tokens": [int(t) for t in np.asarray(toks).reshape(-1)],
                 "blobs": {}}
        for k, v in captured.items():
            entry["blobs"][k] = blobs.put(f"{step_name}/{k}", v)
        argmax = int(np.argmax(np.asarray(logits[0, -1])))
        entry["argmax"] = argmax
        steps.append(entry)
        return argmax

    nxt = run_step("prefill", jnp.asarray([prompt_ids], jnp.int32), 0)
    generated = []
    pos = len(prompt_ids)
    for s in range(args.decode_steps):
        if nxt == EOS_ID or pos >= max_len:
            break
        generated.append(nxt)
        hist = hist.at[0, pos].set(nxt)
        nxt = run_step(f"decode{s}", jnp.asarray([[nxt]], jnp.int32), pos)
        pos += 1

    manifest = {
        "source": args.cact,
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "max_len": max_len,
        "kv_window": int(config.kv_window or 0),
        "generated_ids": generated,
        "generated_text": tok.decode(generated),
        "geometry": {
            "d_model": config.d_model,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "num_layers": config.num_layers,
            "head_dim": head_dim,
            "mhc_lanes": config.mhc_lanes,
            "engram_layers": list(config.engram_layers),
            "engram_orders": list(config.engram_orders),
            "rope_theta": float(config.rope_theta),
            "vocab_size": config.vocab_size,
        },
        "blob_index": blobs.index,
        "steps": steps,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(manifest, f)
    with open(args.output + ".f32", "wb") as f:
        f.write(bytes(blobs.buf))

    print(f"wrote {args.output}.json + .f32: {len(steps)} steps, "
          f"{len(blobs.index)} arrays, {len(blobs.buf) / 1e6:.2f} MB of f32",
          file=sys.stderr)
    print(f"generated: {manifest['generated_text']!r}", file=sys.stderr)
    print("[ok] instrumented forward matches decode.forward_cached on every step",
          file=sys.stderr)


if __name__ == "__main__":
    main()
