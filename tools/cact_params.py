"""Rebuild a Flax parameter tree from a `.cact` blob.

`needle/model/export.py` flattens the parameter tree into a nameless positional
directory, pre-transposing every matmul weight to `[out, in]`. This module is
the exact inverse: it reads the blob with upstream's own `read_export()` and
puts the tensors back into the nested dict that `needle/model/decode.py`
consumes.

Why bother, when the Rust loader is already verified tensor by tensor: the
positional canon means a *correct* tensor can still land in the wrong slot.
Feeding real weights through upstream's `_forward_cached` gives logits to compare
Rust against, which is the only check that pins the canon end to end.

Usage:
    from cact_params import load_cact_params
    params, config, meta = load_cact_params("weights/needle2.cact")

Inverse of `export._tensors`, slot by slot:

    embedding        stored as-is                  -> [vocab, d_model]
    q/k/v/gate_proj  stored .T                     -> kernel = stored.T
    out_proj         stored .T                     -> kernel = stored.T
    mhc_phi_*        transpose(0,2,1).reshape      -> reshape then transpose back
    engram tables    reshape(tables*slots, sub)    -> reshape(tables, slots, sub)
    engram k/v_proj  stored .T                     -> kernel = stored.T
    head proj        stored .T                     -> kernel = stored.T
"""

import numpy as np

TENSORS_PER_LAYER = 14
MHC_TENSORS = 9
TENSORS_PER_ENGRAM_SITE = 4
HEAD_NAMES = {1: "contrastive_head", 2: "confidence_head"}


def _f32(a):
    return np.asarray(a, np.float32)


def load_cact_params(path):
    """Return `(params, config, meta)` ready for `decode.forward_cached`."""
    from needle.model.export import read_export
    from needle.model.architecture import TransformerConfig, engram_geometry

    meta, ts = read_export(path)
    L = meta["num_layers"]
    d = meta["d_model"]
    lanes = meta["mhc_lanes"]
    nC = lanes * d
    sites = tuple(meta["engram_layers"])
    orders = tuple(meta["engram_orders"])

    config = TransformerConfig(
        vocab_size=meta["vocab_size"],
        d_model=d,
        attn_dim=meta["num_heads"] * meta["head_dim"],
        num_heads=meta["num_heads"],
        num_kv_heads=meta["num_kv_heads"],
        num_layers=L,
        max_seq_len=meta["max_seq_len"],
        rope_theta=meta["rope_theta"],
        mhc_lanes=lanes,
        engram_orders=orders,
        engram_slots=meta["engram_slots"],
        engram_layers=sites,
        kv_window=meta["kv_window"],
        kv_bits=meta["kv_bits"],
        dtype="float32",
    )

    # The header records the table count directly; cross-check it against the
    # geometry the architecture derives, since a mismatch would silently reshape.
    _, e_heads, e_sub = engram_geometry(config)
    n_tables = len(orders) * e_heads
    if n_tables != meta["num_engram_tables"] or e_sub != meta["engram_sub_dim"]:
        raise ValueError(
            f"engram geometry disagrees: header says {meta['num_engram_tables']} tables "
            f"of {meta['engram_sub_dim']}, architecture derives {n_tables} of {e_sub}"
        )

    i = 0
    embedding = _f32(ts[i]); i += 1

    # Per-layer tensors are stacked along a leading layer axis, because
    # `decode._layer` indexes every leaf with `[i]`.
    stack = {k: [] for k in (
        "norm_in", "q", "k", "v", "q_norm", "k_norm", "gate", "out",
        "post_norm", "attn_gate", "pre_hada", "d1", "d2", "d3")}
    for _ in range(L):
        t = ts[i:i + TENSORS_PER_LAYER]
        i += TENSORS_PER_LAYER
        stack["norm_in"].append(_f32(t[0]))
        stack["q"].append(_f32(t[1]).T)
        stack["k"].append(_f32(t[2]).T)
        stack["v"].append(_f32(t[3]).T)
        stack["q_norm"].append(_f32(t[4]))
        stack["k_norm"].append(_f32(t[5]))
        stack["gate"].append(_f32(t[6]).T)
        stack["out"].append(_f32(t[7]).T)
        stack["post_norm"].append(_f32(t[8]))
        stack["attn_gate"].append(_f32(t[9]).reshape(()))
        stack["pre_hada"].append(_f32(t[10]))
        stack["d1"].append(_f32(t[11]))
        stack["d2"].append(_f32(t[12]))
        stack["d3"].append(_f32(t[13]))
    st = {k: np.stack(v) for k, v in stack.items()}

    mhc = ts[i:i + MHC_TENSORS]
    i += MHC_TENSORS
    a_pre, a_post, a_res, b_pre, b_post, b_res = (_f32(x) for x in mhc[:6])

    def unphi(stored, fan):
        # export did transpose(0, 2, 1).reshape(L * fan, nC)
        return _f32(stored).reshape(L, fan, nC).transpose(0, 2, 1)

    phi_pre = unphi(mhc[6], lanes)
    phi_post = unphi(mhc[7], lanes)
    phi_res = unphi(mhc[8], lanes * lanes)

    engrams = {}
    for s in range(len(sites)):
        t = ts[i:i + TENSORS_PER_ENGRAM_SITE]
        i += TENSORS_PER_ENGRAM_SITE
        engrams[f"engrams_{s}"] = {
            "embedding": _f32(t[0]).reshape(n_tables, meta["engram_slots"], e_sub),
            "key_proj": {"kernel": _f32(t[1]).T},
            "value_proj": {"kernel": _f32(t[2]).T},
            "taps": _f32(t[3]),
        }

    final_norm = _f32(ts[i]); i += 1

    params = {
        "embedding": {"embedding": embedding},
        "stack": {
            "layers": {
                "block": {
                    "ZCRMSNorm_0": {"scale": st["norm_in"]},
                    "self_attn": {
                        "q_proj": {"kernel": st["q"]},
                        "k_proj": {"kernel": st["k"]},
                        "v_proj": {"kernel": st["v"]},
                        "gate_proj": {"kernel": st["gate"]},
                        "out_proj": {"kernel": st["out"]},
                        "q_norm": {"scale": st["q_norm"]},
                        "k_norm": {"scale": st["k_norm"]},
                    },
                    "post_attn_norm": {"scale": st["post_norm"]},
                    "attn_gate": st["attn_gate"],
                    "pre_hada_norm": {"scale": st["pre_hada"]},
                    "hadamard_mlp": {"d1": st["d1"], "d2": st["d2"], "d3": st["d3"]},
                }
            },
            "mhc_a_pre": a_pre,
            "mhc_a_post": a_post,
            "mhc_a_res": a_res,
            "mhc_b_pre": b_pre,
            "mhc_b_post": b_post,
            "mhc_b_res": b_res,
            "mhc_phi_pre": phi_pre,
            "mhc_phi_post": phi_post,
            "mhc_phi_res": phi_res,
            "final_norm": {"scale": final_norm},
        },
        **engrams,
    }

    # Probe heads, when the blob carries them: a manifest of codes followed by
    # fixed-stride (probes, proj, bias) triples.
    tokenizer_blob = None
    if i < len(ts) and isinstance(ts[-1], (bytes, bytearray)):
        tokenizer_blob = bytes(ts[-1])
    tail_end = len(ts) - (1 if tokenizer_blob is not None else 0)
    if i < tail_end:
        codes = [int(round(float(c))) for c in np.asarray(ts[i]).reshape(-1)]
        i += 1
        for code in codes:
            name = HEAD_NAMES.get(code)
            if name is None:
                raise ValueError(f"unknown probe-head code {code}")
            probes, proj, bias = ts[i], ts[i + 1], ts[i + 2]
            i += 3
            params[name] = {
                "probes": _f32(probes),
                "proj": {"kernel": _f32(proj).T, "bias": _f32(bias)},
            }

    if i != tail_end:
        raise ValueError(f"consumed {i} tensors, expected {tail_end}")

    meta = dict(meta)
    meta["tokenizer_blob"] = tokenizer_blob
    return params, config, meta


def shape_report(params, prefix=""):
    """Flat `path -> shape` listing, for eyeballing the reconstruction."""
    out = {}

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}/{k}" if path else k)
        else:
            out[path] = tuple(np.asarray(node).shape)

    walk(params, prefix)
    return out
