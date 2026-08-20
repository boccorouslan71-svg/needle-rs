//! Reports the memory a v2 session actually uses, so the figures quoted in the
//! docs can be checked rather than estimated.

use needle_core::v2::{V2Batch, DEFAULT_CHUNK};
use needle_infer::v2::V2Bundle;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "weights/needle2.cact".into());
    let b = match V2Bundle::load(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to load {path}: {e}");
            std::process::exit(1);
        }
    };
    let cfg = &b.model.cfg;
    let mb = |n: usize| n as f64 / 1e6;

    // The KV cache dominates per-sequence state. Report what was actually
    // allocated rather than a formula, since it is a ring sized to the window.
    let lm = b.model.make_state();
    let full = b.model.make_state_full_causal();
    println!("model file            {:>8.2} MB", mb(std::fs::metadata(&path).unwrap().len() as usize));
    println!(
        "KV cache, generation  {:>8.2} MB  (ring of {} positions, window {})",
        mb(lm.kv_bytes()), lm.cache_len(), cfg.kv_window
    );
    println!(
        "KV cache, full causal {:>8.2} MB  ({} positions; needed by the probe heads)",
        mb(full.kv_bytes()), full.cache_len()
    );
    for cap in [1usize, 16, DEFAULT_CHUNK, 128, 256] {
        let sc = V2Batch::new(&b.model, cap);
        let tag = if cap == DEFAULT_CHUNK { " (default)" } else { "" };
        println!("prefill scratch cap {cap:>4}  {:>6.2} MB{tag}", mb(sc.bytes()));
    }
    for h in &b.heads {
        // Streaming pool state: probes x d_model accumulators, plus per-probe scalars.
        let pool = h.num_probes * cfg.d_model * 4 + h.num_probes * 8;
        let naive = cfg.max_seq_len * (cfg.num_layers + 1) * cfg.d_model * 4;
        println!(
            "head code {} ({} probes): streaming pool {:>6.1} KB vs {:>6.1} MB of retained cells",
            h.code, h.num_probes, pool as f64 / 1e3, mb(naive)
        );
    }
}
