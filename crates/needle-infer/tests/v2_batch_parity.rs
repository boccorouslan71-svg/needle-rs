//! Batched prefill against the sequential path, on real weights.
//!
//! The sequential path is held in place by `v2_forward_parity` (788 capture
//! points against upstream). This suite pins the batched path to *that*, so the
//! two hold each other: a regression in either shows up here.
//!
//! `matmul_rows_prepared` is bit-identical to repeated `matvec_rows_prepared`
//! and nothing else about the arithmetic changes, so these are exact-equality
//! assertions, not tolerances. If a tolerance ever becomes necessary, something
//! has been reassociated and the claim in `v2::batch`'s module docs is no longer
//! true.
//!
//! Requires `weights/needle2.cact`; skips with a notice otherwise.
//!
//! Run: cargo test -p needle-infer --release --test v2_batch_parity -- --nocapture

use needle_core::v2::{V2Batch, V2Model};
use needle_infer::v2::V2Bundle;
use needle_infer::v2_engine::V2Engine;

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle2.cact");

fn bundle() -> Option<V2Bundle> {
    if !std::path::Path::new(CACT).exists() {
        eprintln!("skipping v2 batch parity: missing {CACT}");
        return None;
    }
    Some(V2Bundle::load(CACT).expect("load"))
}

/// A deterministic token sequence of a given length, avoiding the special ids so
/// nothing is treated as a control token.
fn tokens(n: usize, vocab: usize) -> Vec<u32> {
    (0..n)
        .map(|i| (20 + (i * 4099 + i / 7) % (vocab - 40)) as u32)
        .collect()
}

fn sequential(model: &V2Model, toks: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let mut state = model.make_state();
    let mut logits = vec![0.0f32; model.cfg.vocab_size];
    let (last, prefix) = toks.split_last().unwrap();
    for &t in prefix {
        model.step_prefill(t, &mut state).unwrap();
    }
    model.step(*last, &mut state, &mut logits).unwrap();
    (logits, V2Model::last_hidden(&state).to_vec())
}

fn batched(model: &V2Model, toks: &[u32], cap: usize) -> (Vec<f32>, Vec<f32>) {
    let mut state = model.make_state();
    let mut batch = V2Batch::new(model, cap);
    let mut logits = vec![0.0f32; model.cfg.vocab_size];
    model
        .prefill_batch(toks, &mut state, &mut batch, Some(&mut logits))
        .unwrap();
    (logits, Vec::new())
}

/// Exact equality across a range of prompt lengths and chunk sizes, including
/// lengths that straddle a chunk boundary.
#[test]
fn batched_prefill_matches_sequential_exactly() {
    let Some(b) = bundle() else { return };
    let model = &b.model;
    let vocab = model.cfg.vocab_size;

    for &n in &[1usize, 2, 7, 16, 31, 32, 33, 64, 100, 129] {
        let toks = tokens(n, vocab);
        let (want, _) = sequential(model, &toks);
        // Chunk sizes above, below and exactly at the prompt length.
        for &cap in &[1usize, 8, 32, 128] {
            let (got, _) = batched(model, &toks, cap);
            let mismatches = got
                .iter()
                .zip(want.iter())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .take(3)
                .map(|(i, (a, b))| format!("[{i}] {a} vs {b}"))
                .collect::<Vec<_>>();
            assert!(
                mismatches.is_empty(),
                "n={n} cap={cap}: {} of {} logits differ; first: {}",
                got.iter().zip(want.iter()).filter(|(a, b)| a != b).count(),
                got.len(),
                mismatches.join(", ")
            );
        }
    }
}

/// A prompt longer than one chunk must give the same answer as one that fits,
/// which is what proves the cross-chunk KV handoff.
#[test]
fn chunk_boundary_does_not_change_the_result() {
    let Some(b) = bundle() else { return };
    let model = &b.model;
    let toks = tokens(300, model.cfg.vocab_size);
    let (want, _) = batched(model, &toks, 512); // single chunk
    for &cap in &[7usize, 64, 128, 299, 300, 301] {
        let (got, _) = batched(model, &toks, cap);
        assert_eq!(got, want, "cap={cap} changed the result");
    }
}

/// State left behind by a batched prefill must be usable for decoding, and give
/// the same continuation the sequential path would.
#[test]
fn decoding_continues_identically_after_batched_prefill() {
    let Some(b) = bundle() else { return };
    let model = &b.model;
    let toks = tokens(40, model.cfg.vocab_size);

    let greedy = |mut state: needle_core::v2::V2State, mut logits: Vec<f32>| -> Vec<u32> {
        let mut out = Vec::new();
        for _ in 0..12 {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            out.push(next);
            model.step(next, &mut state, &mut logits).unwrap();
        }
        out
    };

    let seq_ids = {
        let mut state = model.make_state();
        let mut logits = vec![0.0f32; model.cfg.vocab_size];
        let (last, prefix) = toks.split_last().unwrap();
        for &t in prefix {
            model.step_prefill(t, &mut state).unwrap();
        }
        model.step(*last, &mut state, &mut logits).unwrap();
        greedy(state, logits)
    };

    let batch_ids = {
        let mut state = model.make_state();
        let mut batch = V2Batch::new(model, 16);
        let mut logits = vec![0.0f32; model.cfg.vocab_size];
        model
            .prefill_batch(&toks, &mut state, &mut batch, Some(&mut logits))
            .unwrap();
        greedy(state, logits)
    };

    assert_eq!(batch_ids, seq_ids);
    assert!(!seq_ids.is_empty());
}

/// The engine's batched path must produce byte-identical output to its
/// sequential one on real prompts.
#[test]
fn engine_batched_and_sequential_agree_on_real_prompts() {
    if !std::path::Path::new(CACT).exists() {
        eprintln!("skipping: missing {CACT}");
        return;
    }
    let engine = V2Engine::load(CACT).expect("load");
    const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}}]"#;

    for q in [
        "What's the weather in Paris?",
        "Is it raining in Tokyo right now?",
        "Email alice@example.com saying the build is green",
    ] {
        let seq = engine.generate_sequential(q, TOOLS, &Default::default(), |_, _| {});
        let bat = engine.run(q, TOOLS);
        assert_eq!(bat.token_ids, seq.token_ids, "token ids differ for {q:?}");
        assert_eq!(bat.text, seq.text, "text differs for {q:?}");
        assert_eq!(bat.stop_reason, seq.stop_reason);
    }
}

/// Batch state must be reusable across prompts without leaking context.
#[test]
fn batch_scratch_is_reusable() {
    let Some(b) = bundle() else { return };
    let model = &b.model;
    let mut batch = V2Batch::new(model, 32);
    let a = tokens(50, model.cfg.vocab_size);
    let c = tokens(37, model.cfg.vocab_size);

    let run = |toks: &[u32], batch: &mut V2Batch| {
        let mut state = model.make_state();
        let mut logits = vec![0.0f32; model.cfg.vocab_size];
        model
            .prefill_batch(toks, &mut state, batch, Some(&mut logits))
            .unwrap();
        logits
    };

    let a1 = run(&a, &mut batch);
    let _ = run(&c, &mut batch);
    let a2 = run(&a, &mut batch);
    assert_eq!(a1, a2, "reusing batch scratch changed the result");
    assert!(batch.bytes() > 0);
}
