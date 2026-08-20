//! Forward-pass parity for the Needle v2 port, against upstream's incremental
//! decode path running the same real weights.
//!
//! Requires (generated; all three need the 13.7 MB checkpoint):
//!   weights/needle2.cact
//!   tests/v2_forward_vectors.json + .f32   — `tools/gen_v2_forward_parity.py`
//!
//! The fixture generator asserts its instrumented forward equals
//! `decode.forward_cached` before writing anything, so matching it here means
//! matching upstream's real function.
//!
//! Divergences are reported at the earliest capture point rather than only at the
//! logits — through 27 layers of mHC routing, Sinkhorn lane mixing, Engram
//! lookups and gated attention, a wrong logit on its own says nothing.
//!
//! Run: cargo test -p needle-infer --release --test v2_forward_parity -- --nocapture

use needle_core::v2::V2Model;
use needle_infer::cact::Cact;
use needle_infer::v2::V2Bundle;

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle2.cact");
const JSON: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/v2_forward_vectors.json");
const F32: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/v2_forward_vectors.f32");

/// Per-capture-point tolerance, relative to the reference vector's own magnitude.
///
/// The port evaluates one position at a time with scalar loops; the reference
/// evaluates batched einsums in f32 under XLA, so accumulation order differs
/// everywhere and the mHC residual carries each layer's rounding into the next.
///
/// Measured worst case over all 788 points on this checkpoint is 1.9e-5
/// relative, at layer 23's `u` — i.e. f32 noise, with no algorithmic gap
/// anywhere. The bounds below keep roughly 15x headroom over that so a genuine
/// regression fails while a different FMA schedule does not.
fn tol_for(key: &str) -> f32 {
    match key {
        // Straight off the dequantiser; no accumulation depth to speak of.
        "embed" | "engram_k0" | "engram_v0" => 1e-5,
        _ => 3e-4,
    }
}

struct Fixture {
    manifest: serde_json::Value,
    blobs: Vec<f32>,
}

impl Fixture {
    /// `name` is the generator's key, e.g. `"prefill/L07.u"`.
    fn get(&self, name: &str) -> Option<&[f32]> {
        let e = self.manifest["blob_index"].get(name)?;
        let off = e["offset"].as_u64()? as usize;
        let n: usize = e["shape"]
            .as_array()?
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .product::<usize>()
            .max(1);
        Some(&self.blobs[off..off + n])
    }
}

fn fixture() -> Option<(V2Bundle, Fixture)> {
    for p in [CACT, JSON, F32] {
        if !std::path::Path::new(p).exists() {
            eprintln!(
                "skipping v2 forward parity: missing {p}\n  \
                 regenerate with tools/gen_v2_forward_parity.py; see docs/v2-port-record.md"
            );
            return None;
        }
    }
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(JSON).expect("read manifest"))
            .expect("parse manifest");
    let raw = std::fs::read(F32).expect("read blobs");
    let blobs: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let cact = Cact::load(CACT).expect("load cact");
    let bundle = V2Bundle::from_cact(&cact).expect("build v2 model");
    Some((bundle, Fixture { manifest, blobs }))
}

/// Compare against the reference, scaling the bound by its largest magnitude so
/// near-zero components are not held to an absolute bound f32 cannot meet.
fn diff(got: &[f32], want: &[f32]) -> Option<(f32, f32, usize)> {
    if got.len() != want.len() {
        return Some((f32::INFINITY, 1.0, 0));
    }
    let scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..got.len() {
        let d = (got[i] - want[i]).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    Some((worst, scale, at))
}

/// Walk the ladder from the embedding outward and report the earliest divergence.
#[test]
fn forward_matches_reference_at_every_capture_point() {
    let Some((bundle, fx)) = fixture() else { return };
    let model = &bundle.model;
    let cfg = &model.cfg;

    // The fixture's geometry must be the geometry we loaded.
    let g = &fx.manifest["geometry"];
    assert_eq!(cfg.d_model, g["d_model"].as_u64().unwrap() as usize);
    assert_eq!(cfg.num_layers, g["num_layers"].as_u64().unwrap() as usize);
    assert_eq!(cfg.mhc_lanes, g["mhc_lanes"].as_u64().unwrap() as usize);
    assert_eq!(cfg.head_dim, g["head_dim"].as_u64().unwrap() as usize);
    assert_eq!(cfg.num_heads, g["num_heads"].as_u64().unwrap() as usize);
    assert_eq!(cfg.num_kv_heads, g["num_kv_heads"].as_u64().unwrap() as usize);
    assert_eq!(cfg.vocab_size, g["vocab_size"].as_u64().unwrap() as usize);
    assert_eq!(cfg.kv_window, fx.manifest["kv_window"].as_u64().unwrap() as usize);
    assert_eq!(
        cfg.engram.sites,
        g["engram_layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
    );

    let mut state = model.make_state();
    let mut logits = vec![0.0f32; cfg.vocab_size];
    let mut checked = 0usize;
    let mut worst_overall: (f32, String) = (0.0, String::new());

    for step in fx.manifest["steps"].as_array().unwrap() {
        let name = step["name"].as_str().unwrap().to_string();
        let tokens: Vec<u32> = step["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap() as u32)
            .collect();

        // The reference records only a step's final position, so a prefill's
        // earlier tokens are fed untraced.
        let (last, prefix) = tokens.split_last().expect("step has tokens");
        for &t in prefix {
            model.step(t, &mut state, &mut logits).expect("untraced step");
        }

        let mut failures: Vec<String> = Vec::new();
        {
            let fx = &fx;
            let name = &name;
            let failures = &mut failures;
            let checked = &mut checked;
            let worst_overall = &mut worst_overall;
            let mut trace = |key: &str, layer: usize, values: &[f32]| {
                let full = if layer == usize::MAX {
                    format!("{name}/{key}")
                } else {
                    format!("{name}/L{layer:02}.{key}")
                };
                let Some(want) = fx.get(&full) else { return };
                let tol = tol_for(key);
                if let Some((worst, scale, at)) = diff(values, want) {
                    *checked += 1;
                    let rel = worst / scale;
                    if rel > worst_overall.0 {
                        *worst_overall = (rel, full.clone());
                    }
                    if worst > tol * scale {
                        failures.push(format!(
                            "{full}: max |delta| {worst:.3e} at [{at}] ({} vs {}), \
                             tol {:.3e} (rel {rel:.3e}, scale {scale:.3e})",
                            values.get(at).copied().unwrap_or(f32::NAN),
                            want.get(at).copied().unwrap_or(f32::NAN),
                            tol * scale
                        ));
                    }
                }
            };
            model
                .step_traced(*last, &mut state, &mut logits, &mut trace)
                .expect("traced step");
        }

        // Argmax is what actually decides the emitted token.
        let want_argmax = step["argmax"].as_u64().unwrap() as usize;
        let got_argmax = argmax(&logits);
        if got_argmax != want_argmax {
            failures.push(format!(
                "{name}: argmax {got_argmax} (logit {}) vs reference {want_argmax} (logit {})",
                logits[got_argmax], logits[want_argmax]
            ));
        }

        if !failures.is_empty() {
            // Only the first few matter; everything after the first divergence is
            // downstream of it.
            let shown: Vec<String> = failures.into_iter().take(8).collect();
            panic!("earliest divergences:\n  {}", shown.join("\n  "));
        }
    }

    assert!(
        checked > 700,
        "expected the generator's ~788 capture points, compared {checked}"
    );
    eprintln!(
        "v2 forward parity: {checked} capture points matched; \
         largest relative deviation {:.2e} at {}",
        worst_overall.0, worst_overall.1
    );
}

/// End to end: the port must reproduce the reference's greedy continuation token
/// for token, and decode it to the same text through our own tokenizer.
#[test]
fn greedy_continuation_matches_reference() {
    let Some((bundle, fx)) = fixture() else { return };
    let model = &bundle.model;
    let mut state = model.make_state();
    let mut logits = vec![0.0f32; model.cfg.vocab_size];

    let prompt = ids(&fx.manifest["prompt_ids"]);
    let want = ids(&fx.manifest["generated_ids"]);
    assert!(!want.is_empty(), "fixture generated no tokens");

    for &t in &prompt {
        model.step(t, &mut state, &mut logits).expect("prefill");
    }
    let mut got = Vec::new();
    for _ in 0..want.len() {
        let next = argmax(&logits) as u32;
        got.push(next);
        model.step(next, &mut state, &mut logits).expect("decode");
    }
    assert_eq!(got, want, "greedy continuation diverged");

    let tok = bundle.tokenizer.as_ref().expect("embedded tokenizer");
    assert_eq!(
        tok.decode(&got),
        fx.manifest["generated_text"].as_str().unwrap()
    );
}

/// Resetting state must give a clean sequence — the KV cache, the token history
/// and the Engram ring all have to be cleared, or a second prompt inherits the
/// first one's context.
#[test]
fn reset_yields_identical_second_run() {
    let Some((bundle, fx)) = fixture() else { return };
    let model = &bundle.model;
    let prompt = ids(&fx.manifest["prompt_ids"]);

    let mut state = model.make_state();
    let mut a = vec![0.0f32; model.cfg.vocab_size];
    for &t in &prompt {
        model.step(t, &mut state, &mut a).expect("first run");
    }

    state.reset();
    assert_eq!(state.pos(), 0);
    assert!(state.history().is_empty());
    let mut b = vec![0.0f32; model.cfg.vocab_size];
    for &t in &prompt {
        model.step(t, &mut state, &mut b).expect("second run");
    }
    assert_eq!(a, b, "reset did not fully clear per-sequence state");
}

/// A fresh state must reach the same logits as one carried over, i.e. the model
/// holds no hidden mutable state of its own.
#[test]
fn model_is_stateless_across_sequences() {
    let Some((bundle, fx)) = fixture() else { return };
    let model = &bundle.model;
    let prompt = ids(&fx.manifest["prompt_ids"]);

    let mut first = model.make_state();
    let mut a = vec![0.0f32; model.cfg.vocab_size];
    for &t in &prompt {
        model.step(t, &mut first, &mut a).expect("run");
    }

    let mut second = model.make_state();
    let mut b = vec![0.0f32; model.cfg.vocab_size];
    for &t in &prompt {
        model.step(t, &mut second, &mut b).expect("run");
    }
    assert_eq!(a, b);
    assert_eq!(V2Model::last_hidden(&first), V2Model::last_hidden(&second));
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}
