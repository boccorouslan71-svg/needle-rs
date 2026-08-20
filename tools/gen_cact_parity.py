#!/usr/bin/env python3
"""
Generate `.cact` loader parity vectors from a real Needle v2 checkpoint.

Reads the blob with upstream's own reader (`needle.model.export.read_export`)
and emits, per tensor, enough of a fingerprint that any disagreement in the
Rust loader — header field, directory offset, bit packing, group norms, Hadamard
rotation, row order — shows up as a failed assertion rather than as bad logits.

For CQ tensors the fingerprint is a matrix-vector product against a
deterministic pseudorandom vector, because that exercises the whole
reconstruction path at once. The generator here and `cact_parity.rs` must build
the same vector: see `_probe_vector` and its Rust twin.

Usage (from the repo root):
    PYTHONPATH=needle .venv-parity/bin/python tools/gen_cact_parity.py \
        --cact weights/needle2.cact \
        --output tests/cact_vectors.json

Requires jax/flax only because `needle.model.export` imports the architecture
module; nothing here runs a model.
"""

import argparse
import hashlib
import json
import sys

import numpy as np

FP16, FP32, CQ, RAW = 1, 2, 3, 4

# Leading output components of the probe product that are compared elementwise.
PROBE_HEAD = 8

# Tensors whose rows are dequantised in full, so `dequantize_row` is checked
# directly and not only through a dot product. Positions in the needle2 canon:
# embedding, layer 0's q/k/v/gate/out projections, the two mHC phi shapes, and
# an engram site's tables and key projection — one of every CQ shape and width
# the blob contains. Indices absent from a blob are skipped.
FULL_ROW_TENSORS = (0, 2, 3, 4, 7, 8, 385, 387, 388, 389)

MASK32 = 0xFFFFFFFF


def _probe_vector(n, tensor_index):
    """Deterministic input vector in [-1, 1).

    xorshift32 seeded from the tensor index. Each value is a 24-bit integer
    scaled by a power of two, so it is exactly representable in f32 and the Rust
    twin of this function produces bit-identical inputs.
    """
    s = (0x2545F491 ^ ((tensor_index * 0x9E3779B9) & MASK32)) & MASK32
    if s == 0:
        s = 1
    out = np.empty(n, np.float32)
    for i in range(n):
        s ^= (s << 13) & MASK32
        s ^= s >> 17
        s ^= (s << 5) & MASK32
        out[i] = np.float32((s >> 8) / 8388608.0 - 1.0)
    return out


def _stats(a):
    a64 = np.asarray(a, np.float64)
    return {
        "numel": int(a64.size),
        "sum": float(a64.sum()),
        "sumsq": float((a64 ** 2).sum()),
        "absmax": float(np.abs(a64).max()) if a64.size else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cact", default="weights/needle2.cact")
    ap.add_argument("--output", default="tests/cact_vectors.json")
    args = ap.parse_args()

    from needle.model.export import read_export, _HDR_FMT, _REC_FMT, REC_SIZE
    import struct

    meta, tensors = read_export(args.cact)
    raw = open(args.cact, "rb").read()

    hdr = struct.unpack_from(_HDR_FMT, raw, 0)
    # The directory sits after the fixed header and the codebook block.
    dir_off = struct.calcsize(_HDR_FMT) + len(meta["codebook"]) * 4

    header = {
        "num_tensors": int(hdr[1]),
        "codebook_len": int(hdr[2]),
        "kv_window": int(hdr[3]),
        "kv_bits": int(hdr[4]),
        "vocab_size": int(hdr[5]),
        "d_model": int(hdr[6]),
        "num_heads": int(hdr[7]),
        "num_kv_heads": int(hdr[8]),
        "num_layers": int(hdr[9]),
        "head_dim": int(hdr[10]),
        "max_seq_len": int(hdr[11]),
        "hada_n": int(hdr[12]),
        "mhc_lanes": int(hdr[13]),
        "engram_slots": int(hdr[14]),
        "engram_sub_dim": int(hdr[15]),
        "num_engram_tables": int(hdr[16]),
        "engram_conv_taps": int(hdr[17]),
        "engram_conv_dilation": int(hdr[18]),
        "engram_orders": list(meta["engram_orders"]),
        "engram_sites": list(meta["engram_layers"]),
        "rope_theta": float(hdr[29]),
    }

    out = {
        "source": args.cact,
        "file_bytes": len(raw),
        "header": header,
        "codebook": [float(v) for v in meta["codebook"]],
        "tensors": [],
    }

    for i, t in enumerate(tensors):
        rec = struct.unpack(_REC_FMT, raw[dir_off + i * REC_SIZE: dir_off + (i + 1) * REC_SIZE])
        dtype, ndim = rec[0], rec[1]
        shape = list(rec[3:3 + ndim])
        entry = {
            "index": i,
            "dtype": int(dtype),
            "shape": shape,
            "offset": int(rec[7]),
            "nbytes": int(rec[8]),
            "group": int(rec[9]),
            "bits": int(rec[10]),
        }

        if dtype == RAW:
            entry["sha256"] = hashlib.sha256(bytes(t)).hexdigest()
            entry["len"] = len(t)
        elif dtype in (FP16, FP32):
            flat = np.asarray(t, np.float64).reshape(-1)
            entry["stats"] = _stats(flat)
            entry["first8"] = [float(v) for v in flat[:8]]
            entry["last4"] = [float(v) for v in flat[-4:]]
        elif dtype == CQ:
            w = np.asarray(t, np.float64)  # already dequantised by read_export
            o, n = w.shape
            entry["stats"] = _stats(w)
            x = _probe_vector(n, i).astype(np.float64)
            y = w @ x
            entry["probe"] = {
                "first": [float(v) for v in y[:PROBE_HEAD]],
                "sum": float(y.sum()),
                # Cancellation bound for the tolerance on the compared entries.
                "row_l1_max": float(np.abs(w[:PROBE_HEAD]).sum(axis=1).max()),
                "all_row_l1_sum": float(np.abs(w).sum()),
            }
            if i in FULL_ROW_TENSORS:
                rows = sorted({0, (o // 3) % o, o - 1})
                entry["rows"] = {str(r): [float(v) for v in w[r]] for r in rows}
        out["tensors"].append(entry)

    with open(args.output, "w") as f:
        json.dump(out, f)
    n_cq = sum(1 for e in out["tensors"] if e["dtype"] == CQ)
    print(f"wrote {args.output}: {len(out['tensors'])} tensors ({n_cq} CQ), "
          f"{len(json.dumps(out)) / 1e6:.2f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
