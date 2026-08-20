//! Repeatable single-query timing for the v2 path, as quoted in BENCHMARKS.md.
//!
//! Separates prefill from decode: a step costs the same either way, so folding a
//! ~100-token prefill into one "tokens per second" figure understates decode
//! several-fold.
//!
//! Run: cargo run --release -p needle-infer --example v2_bench

use needle_infer::v2_engine::{GenerateOptions, V2Engine};
use std::time::Instant;
const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}}]"#;
fn main() {
    let q = "What's the weather in Paris?";
    let t0 = Instant::now();
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/needle2.cact".into());
    let e = V2Engine::load(&path).unwrap();
    println!("load: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    // Compare the batched prefill against the one-position-at-a-time reference.
    for chunk in [0usize, 8, 32, 64, 128, 256] {
        let opts = GenerateOptions {
            max_new_tokens: 64,
            prefill_chunk: chunk,
            ..Default::default()
        };
        let mut best = f64::INFINITY;
        let mut ttft = 0.0;
        let mut r = None;
        for _ in 0..3 {
            let mut first = 0.0;
            let t = Instant::now();
            let res = e.generate(q, TOOLS, &opts, |_, _| {
                if first == 0.0 {
                    first = t.elapsed().as_secs_f64() * 1e3;
                }
            });
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
                ttft = first;
                r = Some(res);
            }
        }
        let r = r.unwrap();
        let label = if chunk == 0 {
            "sequential".to_string()
        } else {
            format!("chunk={chunk}")
        };
        println!(
            "{label:>12}: total {best:.0} ms | prefill({} tok) {ttft:.0} ms = {:.2} ms/tok | decode {} tok {:.0} ms",
            r.prompt_tokens, ttft / r.prompt_tokens as f64, r.token_ids.len(), best - ttft
        );
    }

    let opts = GenerateOptions {
        max_new_tokens: 64,
        ..Default::default()
    };
    // Several runs: no JIT here, but let the page cache and branch predictors settle.
    for i in 0..5 {
        let mut ttft = 0.0;
        let t = Instant::now();
        let r = e.generate(q, TOOLS, &opts, |_, _| {
            if ttft == 0.0 {
                ttft = t.elapsed().as_secs_f64() * 1e3;
            }
        });
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "run {i}: total {ms:.0} ms | prefill({} tok) {ttft:.0} ms | decode {} tok {:.0} ms ({:.0} tok/s)",
            r.prompt_tokens, r.token_ids.len(), ms - ttft,
            (r.token_ids.len().saturating_sub(1)) as f64 / ((ms - ttft) / 1e3)
        );
    }
}
