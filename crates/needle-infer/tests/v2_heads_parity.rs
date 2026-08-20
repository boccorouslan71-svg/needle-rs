//! Probe-head parity against upstream, on real weights.
//!
//! Requires `weights/needle2.cact` and `tests/v2_heads_vectors.json`
//! (`tools/gen_v2_heads_parity.py`); skips with a notice otherwise.
//!
//! The corpus deliberately straddles the 256-token attention window, because the
//! heads run full causal while the LM path does not. A port that reused the LM
//! path's window would pass on the short cases and fail on the long ones.
//!
//! Run: cargo test -p needle-infer --release --test v2_heads_parity -- --nocapture

use needle_infer::v2_engine::V2Engine;

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle2.cact");
const VECTORS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/v2_heads_vectors.json"
);

fn fixtures() -> Option<(V2Engine, serde_json::Value)> {
    for p in [CACT, VECTORS] {
        if !std::path::Path::new(p).exists() {
            eprintln!(
                "skipping v2 head parity: missing {p}\n  \
                 regenerate with tools/gen_v2_heads_parity.py"
            );
            return None;
        }
    }
    let v = serde_json::from_str(&std::fs::read_to_string(VECTORS).expect("read")).expect("parse");
    Some((V2Engine::load(CACT).expect("load"), v))
}

fn f64s(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn contrastive_embeddings_match_reference() {
    let Some((engine, v)) = fixtures() else {
        return;
    };
    let dim = v["contrastive_dim"].as_u64().unwrap() as usize;
    assert_eq!(engine.contrastive_dim(), dim);

    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    for c in v["cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want = f64s(&c["contrastive"]);
        let got = engine.encode_contrastive(text).expect("contrastive head");
        assert_eq!(got.len(), want.len(), "dim for {text:?}");

        // Unit-norm on both sides, so an absolute bound is the right one.
        for i in 0..got.len() {
            let d = (got[i] as f64 - want[i]).abs();
            if d > worst {
                worst = d;
                worst_at = format!("{:?}[{i}]", &text[..text.len().min(30)]);
            }
        }
        let norm: f32 = got.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "not unit norm for {text:?}: {norm}"
        );

        let cos: f64 = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| *a as f64 * b)
            .sum();
        assert!(
            cos > 0.9999,
            "cosine {cos:.8} too low for {text:?} ({} tokens)",
            c["n_tokens"]
        );
    }
    eprintln!("contrastive: largest component deviation {worst:.3e} at {worst_at}");
    assert!(worst < 2e-3, "largest deviation {worst:.3e}");
}

#[test]
fn confidence_logits_match_reference() {
    let Some((engine, v)) = fixtures() else {
        return;
    };
    let mut worst = 0.0f64;
    for c in v["cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want = c["confidence"].as_f64().unwrap();
        let got = engine.confidence(text).expect("confidence head") as f64;
        let d = (got - want).abs();
        worst = worst.max(d);
        assert!(
            d < 5e-2 * want.abs().max(1.0),
            "confidence for {text:?} ({} tokens): {got} vs {want}",
            c["n_tokens"]
        );
        // The probability form must agree with the logit.
        let p = engine.confidence_probability(text).unwrap() as f64;
        assert!((p - 1.0 / (1.0 + (-got).exp())).abs() < 1e-6);
        assert!((0.0..=1.0).contains(&p));
    }
    eprintln!("confidence: largest logit deviation {worst:.3e}");
}

/// The long cases are the ones that discriminate: the heads must run full causal,
/// not the checkpoint's 256-token window.
#[test]
fn long_inputs_use_the_full_causal_window() {
    let Some((engine, v)) = fixtures() else {
        return;
    };
    assert_eq!(v["head_window"].as_u64().unwrap(), 0);
    assert_eq!(v["kv_window_lm"].as_u64().unwrap(), 256);

    let long: Vec<&serde_json::Value> = v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["n_tokens"].as_u64().unwrap() > 256)
        .collect();
    assert!(long.len() >= 2, "fixture needs inputs past the window");

    for c in long {
        let text = c["text"].as_str().unwrap();
        let want = f64s(&c["contrastive"]);
        let got = engine.encode_contrastive(text).unwrap();
        let cos: f64 = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| *a as f64 * b)
            .sum();
        assert!(
            cos > 0.9999,
            "cosine {cos:.8} at {} tokens — a 256-token window would land near 0.9996",
            c["n_tokens"]
        );
    }
}

/// Retrieval should rank a tool's own description first for a matching query.
#[test]
fn retrieve_tools_ranks_sensibly() {
    let Some((engine, _)) = fixtures() else {
        return;
    };
    let tools = [
        "get_weather: Get current weather for a city",
        "send_email: Send an email to a recipient",
        "book_flight: Book a flight between two airports",
        "play_music: Play a song or playlist",
    ];
    let ranked = engine.retrieve_tools("what is the weather in Paris", &tools, 4);
    assert_eq!(ranked.len(), 4);
    // Descending, and every score a valid cosine.
    for w in ranked.windows(2) {
        assert!(w[0].1 >= w[1].1, "not sorted: {ranked:?}");
    }
    for (_, s) in &ranked {
        assert!((-1.001..=1.001).contains(s), "score out of range: {s}");
    }
    assert_eq!(
        ranked[0].0, 0,
        "weather query should rank get_weather first: {ranked:?}"
    );

    let ranked = engine.retrieve_tools("send a message to my colleague", &tools, 2);
    assert_eq!(ranked.len(), 2);
    assert_eq!(
        ranked[0].0, 1,
        "email query should rank send_email first: {ranked:?}"
    );
}

/// A shared head state must give the same answers as a fresh one per call, and
/// `retrieve_tools` relies on that to avoid churning the full-length cache once
/// per description.
#[test]
fn shared_head_state_matches_per_call_state() {
    let Some((engine, v)) = fixtures() else {
        return;
    };
    let mut state = engine.new_head_state().expect("a probe head");
    for c in v["cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let fresh = engine.encode_contrastive(text).unwrap();
        let shared = engine
            .encode_contrastive_with_state(text, &mut state)
            .unwrap();
        assert_eq!(shared, fresh, "shared head state diverged on {text:?}");

        let cf = engine.confidence(text).unwrap();
        let cs = engine.confidence_with_state(text, &mut state).unwrap();
        assert_eq!(cs, cf, "shared confidence diverged on {text:?}");
    }
}

/// A head run must not disturb the engine: it uses its own state and its own
/// attention window.
#[test]
fn head_runs_do_not_affect_generation() {
    let Some((engine, _)) = fixtures() else {
        return;
    };
    const TOOLS: &str = r#"[{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;
    let before = engine.run("What's the weather in Paris?", TOOLS);
    let _ = engine.encode_contrastive("some unrelated text about flights");
    let _ = engine.confidence("and something else entirely");
    let after = engine.run("What's the weather in Paris?", TOOLS);
    assert_eq!(before.token_ids, after.token_ids);
    assert_eq!(before.text, after.text);
}

/// The confidence head scores a judgement, not a question: it is trained on a
/// formatted prompt followed by a completion. This pins the semantics that
/// `confidence_for` relies on — a bare query is uninformative, and once the
/// call is included a correct call must outrank a wrong one.
///
/// Thresholds are deliberately loose; the ordering is the contract, not the
/// magnitudes (which the parity vectors above already cover exactly).
#[test]
fn confidence_ranks_the_completion_not_the_query() {
    let Some((engine, _)) = fixtures() else {
        return;
    };
    let tools = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;
    let query = "What's the weather in Paris?";

    let result = engine.run(query, tools);
    let right = engine
        .confidence_for(query, tools, &result.text)
        .expect("head present");
    let wrong = engine
        .confidence_for(
            query,
            tools,
            "<tool_call>[{\"name\":\"send_email\",\"arguments\":{\"to\":\"bob\"}}]</tool_call>",
        )
        .expect("head present");
    let bare = engine.confidence_probability(query).expect("head present");

    assert!(
        right > 0.5,
        "the model's own correct call should score high, got {right}"
    );
    assert!(
        wrong < 0.1,
        "a mismatched call should score low, got {wrong}"
    );
    assert!(right > wrong, "correct {right} must outrank wrong {wrong}");
    // The trap this test exists to catch: scoring the query alone looks like a
    // near-zero confidence for an answer the model in fact gets right.
    assert!(
        bare < 0.05,
        "bare query is expected to be uninformative, got {bare}"
    );
    assert!(
        right > bare * 10.0,
        "prompt+completion ({right}) must be far above bare query ({bare})"
    );
}
