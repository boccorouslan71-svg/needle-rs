//! How prefill cost scales with prompt length.
//!
//! Distinguishes the O(T) work (projections, mHC, MLP — all batched) from the
//! O(T^2) work (attention, which every position pays over its whole window), so
//! the remaining gap to a fused-GEMM reference can be attributed rather than
//! guessed.
//!
//! Run: cargo run --release -p needle-infer --features parallel --example v2_prefill_scaling

use needle_core::v2::{V2Batch, DEFAULT_CHUNK};
use needle_infer::v2::V2Bundle;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/needle2.cact".into());
    let b = match V2Bundle::load(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to load {path}: {e}");
            std::process::exit(1);
        }
    };
    let model = &b.model;
    let vocab = model.cfg.vocab_size;
    let mut batch = V2Batch::new(model, DEFAULT_CHUNK);
    let mut logits = vec![0.0f32; vocab];

    println!(
        "window = {}, chunk = {}",
        model.cfg.kv_window, DEFAULT_CHUNK
    );
    println!(
        "{:>7} {:>11} {:>12} {:>14}",
        "tokens", "prefill", "per token", "vs 32-tok rate"
    );
    let mut base = 0.0f64;
    for &n in &[32usize, 64, 128, 256, 512, 1024] {
        if n > model.cfg.max_seq_len {
            break;
        }
        let toks: Vec<u32> = (0..n)
            .map(|i| (20 + (i * 4099 + i / 7) % (vocab - 40)) as u32)
            .collect();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let mut state = model.make_state();
            let t = Instant::now();
            model
                .prefill_batch(&toks, &mut state, &mut batch, Some(&mut logits))
                .expect("prefill");
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let per = best / n as f64;
        if base == 0.0 {
            base = per;
        }
        println!("{n:>7} {best:>8.1} ms {per:>9.3} ms {:>13.2}x", per / base);
    }
    println!(
        "\nA flat per-token column means the O(T^2) attention term is not yet \n\
         dominant; a rising one means it is, and past the {}-token window it \n\
         should flatten again because each position's span stops growing.",
        model.cfg.kv_window
    );
}
