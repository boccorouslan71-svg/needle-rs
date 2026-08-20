//! End-to-end v2 demo: load a `.cact`, answer a few tool-calling queries, and
//! report timings.
//!
//! Run: cargo run --release -p needle-infer --example v2_demo -- weights/needle2.cact

use needle_infer::v2_engine::{GenerateOptions, V2Engine};
use std::time::Instant;

const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}}]"#;

const QUERIES: &[&str] = &[
    "What's the weather in Paris?",
    "Is it raining in Tokyo right now?",
    "Email alice@example.com saying the build is green",
];

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/needle2.cact".into());

    let t0 = Instant::now();
    let engine = match V2Engine::load(&path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("failed to load {path}: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;

    let cfg = &engine.model().cfg;
    println!(
        "loaded {path} in {load_ms:.0} ms\n  \
         d_model {} | layers {} | heads {}/{} | vocab {} | kv_window {} | engram sites {:?}",
        cfg.d_model,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.vocab_size,
        cfg.kv_window,
        cfg.engram.sites
    );
    if let Some(t) = engine.tokenizer() {
        println!("  embedded tokenizer: {} pieces", t.vocab_size());
    }

    let opts = GenerateOptions {
        max_new_tokens: 64,
        ..Default::default()
    };
    let mut total_steps = 0usize;
    let mut total_ms = 0.0f64;

    for q in QUERIES {
        // Time prefill and decode separately: a step costs the same either way,
        // so folding a ~100-token prefill into a "tokens per second" figure
        // understates decode by several times.
        let mut first_token_ms = 0.0f64;
        let t = Instant::now();
        let r = engine.generate(q, TOOLS, &opts, |_, _| {
            if first_token_ms == 0.0 {
                first_token_ms = t.elapsed().as_secs_f64() * 1e3;
            }
        });
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let gen = r.token_ids.len();
        let decode_ms = ms - first_token_ms;

        println!("\nQ: {q}");
        println!("A: {}", r.text.replace('\n', "\\n"));
        if let Some(th) = &r.thinking {
            println!("   thinking:  {}", th.trim());
        }
        if let Some(tc) = &r.tool_call {
            println!("   tool_call: {tc}");
        }
        println!(
            "   prefill {} tok in {first_token_ms:.0} ms | decode {gen} tok in {decode_ms:.0} ms \
             ({:.0} tok/s) | total {ms:.0} ms | stop={:?}",
            r.prompt_tokens,
            if gen > 1 {
                (gen - 1) as f64 / (decode_ms / 1e3)
            } else {
                0.0
            },
            r.stop_reason
        );
        total_steps += r.prompt_tokens + gen;
        total_ms += ms;
    }

    println!(
        "\n{total_steps} forward steps in {total_ms:.0} ms = {:.2} ms/step ({:.0} steps/s)",
        total_ms / total_steps as f64,
        total_steps as f64 / (total_ms / 1e3)
    );
}
