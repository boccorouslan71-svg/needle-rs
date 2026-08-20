#!/usr/bin/env python3
"""
Generate probe-head parity vectors from a real `.cact` checkpoint.

Runs upstream's `SimpleAttentionNetwork.encode_contrastive` and
`forward_confidence` on weights rebuilt by `tools/cact_params.py`.

Two notes on what is being compared:

* Both methods default to `window=0` (full causal), while the LM path runs the
  checkpoint's `kv_window`. Below 256 tokens the two agree bit for bit; above it
  they diverge (at 642 tokens, cosine 0.99965 on the embedding and 0.096 on the
  confidence logit). The fixture records `window=0`, which is the shipped
  default, and includes inputs on both sides of 256 so the Rust port is pinned at
  a length where the choice actually matters.
* `log_temp` is a training-only scalar that never reaches the container and is
  discarded by `encode_contrastive`; a placeholder is supplied so flax can bind
  the module.

Usage (from the repo root):
    PYTHONPATH=needle:tools JAX_PLATFORMS=cpu .venv-parity/bin/python \
        tools/gen_v2_heads_parity.py
"""

import argparse
import json
import sys

import numpy as np

CASES = [
    "",
    "a",
    "What's the weather in Paris?",
    "Book a table for four at eight tonight",
    "get_weather: Get current weather for a city",
    "send_email: Send an email to a recipient",
    "search_web: Search the web for a query",
    "你好世界",
    "café naïve résumé",
    '{"name":"get_weather","arguments":{"city":"Paris"}}',
    # Past the 256-token window, where window=0 and window=256 diverge.
    "The quick brown fox jumps over the lazy dog. " * 40,
    "alpha beta gamma delta epsilon zeta eta theta " * 30,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cact", default="weights/needle2.cact")
    ap.add_argument("--output", default="tests/v2_heads_vectors.json")
    args = ap.parse_args()

    import jax.numpy as jnp
    from cact_params import load_cact_params
    from needle.model.export import parse_tokenizer_blob, RefTokenizer
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.tokenizer import BOS_ID

    params, config, meta = load_cact_params(args.cact)
    params["contrastive_head"]["log_temp"] = jnp.asarray(0.0, jnp.float32)
    tok = RefTokenizer(parse_tokenizer_blob(meta["tokenizer_blob"]))
    model = SimpleAttentionNetwork(config)

    out = {
        "kv_window_lm": int(config.kv_window or 0),
        "head_window": 0,
        "contrastive_dim": int(config.contrastive_dim),
        "cases": [],
    }
    for text in CASES:
        ids = [BOS_ID] + tok.encode(text)
        toks = jnp.asarray([ids], jnp.int32)
        emb = np.asarray(
            model.apply({"params": params}, toks, window=0,
                        method=SimpleAttentionNetwork.encode_contrastive),
            np.float64)[0]
        conf = float(np.asarray(
            model.apply({"params": params}, toks, window=0,
                        method=SimpleAttentionNetwork.forward_confidence)).reshape(-1)[0])
        out["cases"].append({
            "text": text,
            "ids": [int(i) for i in ids],
            "n_tokens": len(ids),
            "contrastive": [float(v) for v in emb],
            "confidence": conf,
        })
        print(f"  {len(ids):5d} tok  conf {conf:+.5f}  |e| {np.linalg.norm(emb):.6f}  "
              f"{text[:40]!r}", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.output}: {len(out['cases'])} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
