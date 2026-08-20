//! End-to-end parity for the v2 path: exact generated token ids against the
//! Python reference, over a spread of prompts.
//!
//! The v2 counterpart of `e2e_parity.rs`. Where `v2_forward_parity` pins
//! intermediates for a single prompt, this pins the whole pipeline — chat
//! template, tokenizer, prefill, greedy decode, detokenise — and is the gate
//! that invasive changes to the cache or the execution strategy are checked
//! against.
//!
//! Requires `weights/needle2.cact` and `tests/v2_e2e_vectors.json`
//! (`tools/gen_v2_e2e_parity.py`); skips with a notice otherwise.
//!
//! Run: cargo test -p needle-infer --release --test v2_e2e_parity -- --nocapture

use needle_infer::v2_engine::{GenerateOptions, V2Engine};

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle2.cact");
const VECTORS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/v2_e2e_vectors.json");

struct Fixture {
    engine: V2Engine,
    v: serde_json::Value,
}

fn fixture() -> Option<Fixture> {
    for p in [CACT, VECTORS] {
        if !std::path::Path::new(p).exists() {
            eprintln!(
                "skipping v2 e2e parity: missing {p}\n  \
                 regenerate with tools/gen_v2_e2e_parity.py"
            );
            return None;
        }
    }
    let v = serde_json::from_str(&std::fs::read_to_string(VECTORS).expect("read"))
        .expect("parse");
    Some(Fixture { engine: V2Engine::load(CACT).expect("load"), v })
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn opts(max_new_tokens: usize, prefill_chunk: usize) -> GenerateOptions {
    GenerateOptions { max_new_tokens, prefill_chunk, ..Default::default() }
}

/// Exact token-for-token match on every case, at the default settings.
#[test]
fn generated_tokens_match_reference_exactly() {
    let Some(f) = fixture() else { return };
    let max_new = f.v["max_new_tokens"].as_u64().unwrap() as usize;
    let cases = f.v["cases"].as_array().unwrap();
    assert!(cases.len() >= 12, "expected a spread of cases, got {}", cases.len());

    let mut total_prompt = 0usize;
    let mut total_gen = 0usize;
    let mut longest = 0usize;

    for c in cases {
        let query = c["query"].as_str().unwrap();
        let tools = c["tools"].as_str().unwrap();
        let want = ids(&c["generated_ids"]);
        let want_prompt = c["prompt_tokens"].as_u64().unwrap() as usize;

        let r = f.engine.generate(query, tools, &opts(max_new, 64), |_, _| {});
        assert_eq!(
            r.prompt_tokens, want_prompt,
            "prompt tokenised differently for {query:?}"
        );
        assert_eq!(
            r.token_ids, want,
            "generated ids differ for {query:?}\n  got  {:?}\n  want {:?}\n  got text  {:?}\n  want text {:?}",
            &r.token_ids[..r.token_ids.len().min(12)],
            &want[..want.len().min(12)],
            r.text,
            c["text"].as_str().unwrap()
        );
        assert_eq!(r.text, c["text"].as_str().unwrap(), "text differs for {query:?}");

        total_prompt += want_prompt;
        total_gen += want.len();
        longest = longest.max(want_prompt);
    }

    // The fixture must actually exercise the sliding window, or a cache-sizing
    // bug would pass unnoticed.
    let window = f.v["kv_window"].as_u64().unwrap() as usize;
    assert!(
        longest > window,
        "longest prompt {longest} does not exceed the {window}-token window"
    );
    eprintln!(
        "v2 e2e parity: {} cases, {total_prompt} prompt + {total_gen} generated tokens, \
         longest prompt {longest} (window {window}) — all exact",
        cases.len()
    );
}

/// The result must not depend on the prefill chunking, including chunks that do
/// not divide the prompt and chunks larger than it.
#[test]
fn result_is_independent_of_prefill_chunk() {
    let Some(f) = fixture() else { return };
    let max_new = f.v["max_new_tokens"].as_u64().unwrap() as usize;
    // The long cases are the ones that cross chunk and window boundaries.
    let mut checked = 0;
    for c in f.v["cases"].as_array().unwrap() {
        let prompt_tokens = c["prompt_tokens"].as_u64().unwrap() as usize;
        if prompt_tokens < 150 {
            continue;
        }
        let query = c["query"].as_str().unwrap();
        let tools = c["tools"].as_str().unwrap();
        let want = ids(&c["generated_ids"]);
        for chunk in [0usize, 1, 7, 32, 64, 128, 512] {
            let r = f.engine.generate(query, tools, &opts(max_new, chunk), |_, _| {});
            assert_eq!(
                r.token_ids, want,
                "chunk={chunk} changed the result for {query:?} ({prompt_tokens} tokens)"
            );
        }
        checked += 1;
    }
    assert!(checked >= 3, "expected several long cases, checked {checked}");
}

/// Streaming must reconstruct exactly the text the non-streaming path returns.
#[test]
fn streaming_reproduces_the_same_output() {
    let Some(f) = fixture() else { return };
    let max_new = f.v["max_new_tokens"].as_u64().unwrap() as usize;
    for c in f.v["cases"].as_array().unwrap().iter().take(6) {
        let query = c["query"].as_str().unwrap();
        let tools = c["tools"].as_str().unwrap();
        let mut streamed = String::new();
        let mut streamed_ids = Vec::new();
        let r = f.engine.generate(query, tools, &opts(max_new, 64), |id, piece| {
            streamed_ids.push(id);
            streamed.push_str(piece);
        });
        assert_eq!(streamed_ids, r.token_ids, "streamed ids differ for {query:?}");
        assert_eq!(streamed, r.text, "streamed text differs for {query:?}");
    }
}

/// Reusing one state across every case must give the same answers as a fresh
/// state per case — the check that `reset` really clears the cache, the token
/// history and the Engram ring.
#[test]
fn shared_state_matches_fresh_state() {
    let Some(f) = fixture() else { return };
    let max_new = f.v["max_new_tokens"].as_u64().unwrap() as usize;
    let mut state = f.engine.new_state();
    for c in f.v["cases"].as_array().unwrap() {
        let query = c["query"].as_str().unwrap();
        let tools = c["tools"].as_str().unwrap();
        let want = ids(&c["generated_ids"]);
        let r = f.engine.generate_with_state(query, tools, &opts(max_new, 64), &mut state, |_, _| {});
        assert_eq!(r.token_ids, want, "shared state diverged on {query:?}");
    }
}
