//! Where this runtime overtakes the JAX reference, as a function of how many
//! tokens are generated.
//!
//! Prefill and decode have opposite standings — the reference has a faster
//! prefill and a slower decode — so a single query time says little on its own.
//! This measures both here and reports the crossover against reference figures
//! taken on the same machine and the same weights.
//!
//! Run: cargo run --release -p needle-infer --features parallel --example v2_crossover

use needle_infer::v2_engine::{GenerateOptions, V2Engine};
use std::time::Instant;

/// Measured with `tools/cact_params.py` feeding `decode.forward_cached`, same
/// machine, same weights, JIT already warm. See BENCHMARKS.md.
const JAX_PREFILL_MS: f64 = 52.9;
const JAX_DECODE_MS_PER_TOKEN: f64 = 11.00;

const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}}]"#;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/needle2.cact".into());
    let engine = match V2Engine::load(&path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("failed to load {path}: {e}");
            std::process::exit(1);
        }
    };

    // A prompt that generates a long reasoning trace, so decode is measurable.
    let query = "Weather in Reykjavik and then email bob@example.com about it";
    let opts = GenerateOptions {
        max_new_tokens: 96,
        ..Default::default()
    };

    let mut prefill_ms = f64::INFINITY;
    let mut decode_ms = f64::INFINITY;
    let mut n_prompt = 0;
    let mut n_gen = 0;
    for _ in 0..5 {
        let mut ttft = 0.0;
        let t = Instant::now();
        let r = engine.generate(query, TOOLS, &opts, |_, _| {
            if ttft == 0.0 {
                ttft = t.elapsed().as_secs_f64() * 1e3;
            }
        });
        let total = t.elapsed().as_secs_f64() * 1e3;
        if total < prefill_ms + decode_ms {
            prefill_ms = ttft;
            decode_ms = total - ttft;
            n_prompt = r.prompt_tokens;
            n_gen = r.token_ids.len();
        }
    }
    let per_tok = decode_ms / (n_gen.saturating_sub(1)).max(1) as f64;

    println!("needle-rs (this machine, parallel):");
    println!(
        "  prefill {n_prompt} tok : {prefill_ms:7.1} ms = {:.2} ms/tok",
        prefill_ms / n_prompt as f64
    );
    println!("  decode  {n_gen} tok : {decode_ms:7.1} ms = {per_tok:.2} ms/tok");
    println!("\nPython/JAX reference (same machine, same weights, warm):");
    println!("  prefill {n_prompt} tok : {JAX_PREFILL_MS:7.1} ms");
    println!("  decode          : {JAX_DECODE_MS_PER_TOKEN:.2} ms/tok");

    let d = JAX_DECODE_MS_PER_TOKEN - per_tok;
    println!(
        "\nper-token decode advantage: {d:.2} ms ({:.2}x)",
        JAX_DECODE_MS_PER_TOKEN / per_tok
    );
    if d <= 0.0 {
        println!("no decode advantage; the reference wins at every length");
        return;
    }
    let crossover = ((prefill_ms - JAX_PREFILL_MS) / d).ceil().max(0.0);
    println!("prefill deficit: {:.1} ms", prefill_ms - JAX_PREFILL_MS);
    println!("crossover: needle-rs is faster once more than {crossover:.0} tokens are generated");

    println!(
        "\n{:>8} {:>12} {:>12} {:>10}",
        "tokens", "needle-rs", "jax", "ratio"
    );
    for n in [8usize, 19, 32, 40, 64, 96, 128, 256, 512] {
        let ours = prefill_ms + per_tok * n as f64;
        let theirs = JAX_PREFILL_MS + JAX_DECODE_MS_PER_TOKEN * n as f64;
        println!(
            "{n:>8} {ours:>10.0} ms {theirs:>10.0} ms {:>9.2}x{}",
            theirs / ours,
            if ours < theirs { "  <-- we win" } else { "" }
        );
    }
}
