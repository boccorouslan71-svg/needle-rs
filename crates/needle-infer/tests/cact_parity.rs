//! Parity for the Needle v2 `.cact` loader against upstream's own reader.
//!
//! Requires (both gitignored / generated):
//!   weights/needle2.cact       — `huggingface.co/Cactus-Compute/needle2`
//!   tests/cact_vectors.json    — `tools/gen_cact_parity.py`
//!
//! Skips with a printed notice when either is absent, so a fresh clone still
//! runs `cargo test` clean.
//!
//! Run: cargo test -p needle-infer --test cact_parity -- --nocapture

use needle_infer::cact::{Cact, CactLayout, DT_CQ, DT_FP16, DT_FP32, DT_RAW};

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle2.cact");
const VECTORS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/cact_vectors.json");

fn fixtures() -> Option<(Cact, serde_json::Value)> {
    if !std::path::Path::new(CACT).exists() || !std::path::Path::new(VECTORS).exists() {
        eprintln!(
            "skipping cact parity: need weights/needle2.cact and tests/cact_vectors.json\n  \
             see docs/v2-port-record.md"
        );
        return None;
    }
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(VECTORS).expect("read vectors"))
            .expect("parse vectors");
    Some((Cact::load(CACT).expect("load cact"), v))
}

/// Twin of `gen_cact_parity._probe_vector`. xorshift32 seeded from the tensor
/// index; every value is a 24-bit integer over 2^23, so both languages see
/// bit-identical f32 inputs.
fn probe_vector(n: usize, tensor_index: usize) -> Vec<f32> {
    let mut s = 0x2545_F491u32 ^ (tensor_index as u32).wrapping_mul(0x9E37_79B9);
    if s == 0 {
        s = 1;
    }
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 8) as f32 / 8_388_608.0 - 1.0
        })
        .collect()
}

fn f64s(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn usizes(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn header_matches_reference() {
    let Some((c, v)) = fixtures() else { return };
    let h = &v["header"];
    let g = &c.geom;

    assert_eq!(c.byte_len(), v["file_bytes"].as_u64().unwrap() as usize);
    assert_eq!(c.num_tensors(), h["num_tensors"].as_u64().unwrap() as usize);
    assert_eq!(g.codebook_len, h["codebook_len"].as_u64().unwrap() as usize);
    assert_eq!(g.kv_window, h["kv_window"].as_u64().unwrap() as usize);
    assert_eq!(g.kv_bits, h["kv_bits"].as_u64().unwrap() as u32);
    assert_eq!(g.vocab_size, h["vocab_size"].as_u64().unwrap() as usize);
    assert_eq!(g.d_model, h["d_model"].as_u64().unwrap() as usize);
    assert_eq!(g.num_heads, h["num_heads"].as_u64().unwrap() as usize);
    assert_eq!(g.num_kv_heads, h["num_kv_heads"].as_u64().unwrap() as usize);
    assert_eq!(g.num_layers, h["num_layers"].as_u64().unwrap() as usize);
    assert_eq!(g.head_dim, h["head_dim"].as_u64().unwrap() as usize);
    assert_eq!(g.max_seq_len, h["max_seq_len"].as_u64().unwrap() as usize);
    assert_eq!(g.hada_n, h["hada_n"].as_u64().unwrap() as usize);
    assert_eq!(g.mhc_lanes, h["mhc_lanes"].as_u64().unwrap() as usize);
    assert_eq!(g.engram_slots, h["engram_slots"].as_u64().unwrap() as usize);
    assert_eq!(
        g.engram_sub_dim,
        h["engram_sub_dim"].as_u64().unwrap() as usize
    );
    assert_eq!(
        g.num_engram_tables,
        h["num_engram_tables"].as_u64().unwrap() as usize
    );
    assert_eq!(
        g.engram_conv_taps,
        h["engram_conv_taps"].as_u64().unwrap() as usize
    );
    assert_eq!(
        g.engram_conv_dilation,
        h["engram_conv_dilation"].as_u64().unwrap() as usize
    );
    assert_eq!(g.engram_orders, usizes(&h["engram_orders"]));
    assert_eq!(g.engram_sites, usizes(&h["engram_sites"]));
    assert_eq!(g.rope_theta, h["rope_theta"].as_f64().unwrap() as f32);

    let cb_ref = f64s(&v["codebook"]);
    assert_eq!(c.codebook.len(), cb_ref.len());
    for (i, (&got, want)) in c.codebook.iter().zip(cb_ref).enumerate() {
        assert_eq!(got, want as f32, "codebook[{i}]");
    }
}

#[test]
fn directory_records_match_reference() {
    let Some((c, v)) = fixtures() else { return };
    let refs = v["tensors"].as_array().unwrap();
    assert_eq!(c.num_tensors(), refs.len());
    for (i, r) in refs.iter().enumerate() {
        let rec = c.record(i);
        assert_eq!(
            rec.dtype,
            r["dtype"].as_u64().unwrap() as u8,
            "tensor {i} dtype"
        );
        assert_eq!(
            rec.offset,
            r["offset"].as_u64().unwrap(),
            "tensor {i} offset"
        );
        assert_eq!(
            rec.nbytes,
            r["nbytes"].as_u64().unwrap(),
            "tensor {i} nbytes"
        );
        assert_eq!(
            rec.group,
            r["group"].as_u64().unwrap() as usize,
            "tensor {i} group"
        );
        assert_eq!(
            rec.bits,
            r["bits"].as_u64().unwrap() as u8,
            "tensor {i} bits"
        );
        let shape = usizes(&r["shape"]);
        assert_eq!(rec.ndim as usize, shape.len(), "tensor {i} ndim");
        assert_eq!(&rec.shape[..shape.len()], &shape[..], "tensor {i} shape");
    }
}

/// The whole point of the nameless directory check: the canon must land on the
/// shapes the header geometry predicts, for every layer and site.
#[test]
fn layout_resolves_against_real_blob() {
    let Some((c, _)) = fixtures() else { return };
    let l = c.layout().expect("layout must validate");
    let g = &c.geom;

    assert_eq!(l.layers.len(), g.num_layers);
    assert_eq!(l.engrams.len(), g.engram_sites.len());
    assert!(l.tokenizer.is_some(), "needle2.cact embeds a tokenizer");
    assert_eq!(l.heads.len(), 2, "contrastive + confidence");

    // Every canon index must be distinct and in range.
    let mut all: Vec<usize> = vec![l.embedding, l.final_norm];
    for x in &l.layers {
        all.extend([
            x.norm_in,
            x.q_proj,
            x.k_proj,
            x.v_proj,
            x.q_norm,
            x.k_norm,
            x.gate_proj,
            x.out_proj,
            x.post_norm,
            x.attn_gate,
            x.pre_hada,
            x.d1,
            x.d2,
            x.d3,
        ]);
    }
    let m = l.mhc;
    all.extend([
        m.a_pre, m.a_post, m.a_res, m.b_pre, m.b_post, m.b_res, m.phi_pre, m.phi_post, m.phi_res,
    ]);
    for e in &l.engrams {
        all.extend([e.tables, e.key_proj, e.value_proj, e.taps]);
    }
    all.extend(l.head_manifest);
    for h in &l.heads {
        all.extend([h.probes, h.proj, h.bias]);
    }
    all.extend(l.tokenizer);

    assert_eq!(
        all.len(),
        c.num_tensors(),
        "canon must cover every tensor exactly once"
    );
    let mut sorted = all.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), all.len(), "canon indices must be distinct");
    assert_eq!(sorted, (0..c.num_tensors()).collect::<Vec<_>>());

    // Engram sites must be the layers the header names.
    for (site, &layer) in g.engram_sites.iter().enumerate() {
        assert_eq!(CactLayout::engram_site_of_layer(g, layer), Some(site));
    }
    assert_eq!(CactLayout::engram_site_of_layer(g, g.num_layers - 1), None);
}

#[test]
fn fp16_tensors_decode_to_reference_values() {
    let Some((c, v)) = fixtures() else { return };
    let mut checked = 0;
    for (i, r) in v["tensors"].as_array().unwrap().iter().enumerate() {
        let dtype = r["dtype"].as_u64().unwrap() as u8;
        if dtype != DT_FP16 && dtype != DT_FP32 {
            continue;
        }
        let got = c.floats(i).expect("decode floats");
        let st = &r["stats"];
        assert_eq!(
            got.len(),
            st["numel"].as_u64().unwrap() as usize,
            "tensor {i} numel"
        );
        assert_eq!(got.len(), c.record(i).numel(), "tensor {i} numel vs shape");

        // FP16 -> f32 is lossless, so first/last values must match exactly.
        for (k, want) in f64s(&r["first8"]).iter().enumerate() {
            assert_eq!(got[k] as f64, *want, "tensor {i} first8[{k}]");
        }
        let n = got.len();
        for (k, want) in f64s(&r["last4"]).iter().enumerate() {
            let idx = n - f64s(&r["last4"]).len() + k;
            assert_eq!(got[idx] as f64, *want, "tensor {i} last4[{k}]");
        }

        let sum: f64 = got.iter().map(|&x| x as f64).sum();
        let want_sum = st["sum"].as_f64().unwrap();
        let l1: f64 = got.iter().map(|&x| (x as f64).abs()).sum();
        assert!(
            (sum - want_sum).abs() <= 1e-9 * l1.max(1.0),
            "tensor {i} sum: {sum} vs {want_sum}"
        );
        checked += 1;
    }
    assert!(
        checked > 250,
        "expected the blob's ~259 FP16 tensors, checked {checked}"
    );
}

/// The load-bearing test. A CQ matvec exercises bit unpacking, group norms, the
/// codebook slice for the tensor's width, the Hadamard rotation, padding
/// truncation, and row order — all at once, for every CQ tensor in the blob.
#[test]
fn cq_matvec_matches_reference_dequantized_product() {
    let Some((c, v)) = fixtures() else { return };
    let mut checked = 0;
    for (i, r) in v["tensors"].as_array().unwrap().iter().enumerate() {
        if r["dtype"].as_u64().unwrap() as u8 != DT_CQ {
            continue;
        }
        let w = c.cq(i).unwrap_or_else(|e| panic!("tensor {i}: {e}"));
        let shape = usizes(&r["shape"]);
        assert_eq!(
            (w.out_feat, w.in_feat),
            (shape[0], shape[1]),
            "tensor {i} shape"
        );

        let x = probe_vector(w.in_feat, i);
        let mut y = vec![0.0f32; w.out_feat];
        w.matvec(&x, &mut y);

        let p = &r["probe"];
        // |x| <= 1, so a row's f32 rounding error is bounded by its L1 norm.
        let tol = 3e-6 * p["row_l1_max"].as_f64().unwrap().max(1.0);
        for (o, want) in f64s(&p["first"]).iter().enumerate() {
            assert!(
                (y[o] as f64 - want).abs() <= tol,
                "tensor {i} y[{o}]: {} vs {want} (tol {tol})",
                y[o]
            );
        }
        let sum: f64 = y.iter().map(|&a| a as f64).sum();
        let want_sum = p["sum"].as_f64().unwrap();
        let sum_tol = 3e-6 * p["all_row_l1_sum"].as_f64().unwrap().max(1.0);
        assert!(
            (sum - want_sum).abs() <= sum_tol,
            "tensor {i} y sum: {sum} vs {want_sum} (tol {sum_tol})"
        );
        checked += 1;
    }
    assert_eq!(checked, 145, "needle2.cact has 145 CQ tensors");
}

/// `dequantize_row` is on a separate code path from `matvec` (it rotates the
/// weights instead of the activation), so it needs its own check.
#[test]
fn cq_dequantize_row_matches_reference() {
    let Some((c, v)) = fixtures() else { return };
    let mut checked = 0;
    for (i, r) in v["tensors"].as_array().unwrap().iter().enumerate() {
        let Some(rows) = r.get("rows").and_then(|x| x.as_object()) else {
            continue;
        };
        let w = c.cq(i).unwrap();
        for (key, vals) in rows {
            let o: usize = key.parse().unwrap();
            let want = f64s(vals);
            assert_eq!(want.len(), w.in_feat, "tensor {i} row {o} width");
            let mut got = vec![0.0f32; w.in_feat];
            w.dequantize_row(o, &mut got);
            let scale = want.iter().fold(0.0f64, |a, b| a.max(b.abs())).max(1e-6);
            for k in 0..want.len() {
                assert!(
                    (got[k] as f64 - want[k]).abs() <= 1e-5 * scale,
                    "tensor {i} row {o}[{k}]: {} vs {}",
                    got[k],
                    want[k]
                );
            }
            checked += 1;
        }
    }
    assert!(
        checked >= 20,
        "expected the curated full-row probes, checked {checked}"
    );
}

/// A prepared activation must be reusable across the projections that share it,
/// because the engine relies on that to pay the rotation once per layer.
#[test]
fn prepared_input_is_shared_across_layer_projections() {
    let Some((c, _)) = fixtures() else { return };
    let l = c.layout().unwrap();
    let li = l.layers[0];
    let (q, k, val, gate) = (
        c.cq(li.q_proj).unwrap(),
        c.cq(li.k_proj).unwrap(),
        c.cq(li.v_proj).unwrap(),
        c.cq(li.gate_proj).unwrap(),
    );

    let x = probe_vector(c.geom.d_model, 12345);
    let mut xh = vec![0.0f32; q.prepared_len()];
    q.prepare_input(&x, &mut xh);

    for w in [&q, &k, &val, &gate] {
        assert_eq!(w.in_feat, c.geom.d_model);
        assert_eq!(w.prepared_len(), xh.len());
        let mut shared = vec![0.0f32; w.out_feat];
        let mut standalone = vec![0.0f32; w.out_feat];
        w.matvec_prepared(&xh, &mut shared);
        w.matvec(&x, &mut standalone);
        assert_eq!(shared, standalone);
    }
}

#[test]
fn embedded_tokenizer_blob_is_byte_identical() {
    let Some((c, v)) = fixtures() else { return };
    let idx = c.layout().unwrap().tokenizer.expect("tokenizer present");
    let blob = c.raw_tensor(idx).unwrap();
    let r = &v["tensors"].as_array().unwrap()[idx];
    assert_eq!(r["dtype"].as_u64().unwrap() as u8, DT_RAW);
    assert_eq!(blob.len(), r["len"].as_u64().unwrap() as usize);
    assert_eq!(sha256_hex(blob), r["sha256"].as_str().unwrap());
}

/// Minimal SHA-256, so the test suite does not take a dependency for one hash.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}
