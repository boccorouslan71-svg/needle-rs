//! Benchmarks for the Cactus-Quants kernels that dominate a v2 decode step.
//!
//! Shapes are the ones `needle2.cact` actually ships, so the numbers map onto
//! per-token cost directly:
//!
//! * `512x512 @ 2bit` — q_proj / gate_proj / out_proj, three per layer
//! * `256x512 @ 2bit` — k_proj / v_proj, two per layer
//! * `8192x512 @ 4bit` — the tied LM head, once per token
//! * `108x2048 @ 4bit` — mHC phi_pre / phi_post (only 4 rows are read per layer)

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use needle_core::cq::{CqWeight, CODEBOOK_LEN};
use needle_core::hadamard::{fwht, fwht_normalized};

/// Build a CQ weight with deterministic contents at a given shape and width.
fn make(out: usize, in_feat: usize, bits: u8, group: usize) -> CqWeight {
    let cb: Vec<f32> = {
        let s = 1.0 / (group as f32).sqrt();
        let mut v = Vec::with_capacity(CODEBOOK_LEN);
        for &b in &[2u8, 3, 4] {
            let levels = 1usize << b;
            for i in 0..levels {
                v.push((i as f32 - (levels as f32 - 1.0) / 2.0) / (levels as f32 / 2.0) * s);
            }
        }
        v
    };
    let in_pad = in_feat.div_ceil(group) * group;
    let row_bytes = in_pad * bits as usize / 8;
    let num_groups = in_pad / group;
    let mut blob = vec![0u8; out * row_bytes];
    for (i, b) in blob.iter_mut().enumerate() {
        *b = (i * 37 + i / 11) as u8;
    }
    // FP16 1.0 for every group norm.
    blob.extend(std::iter::repeat_n(0u8, out * num_groups * 2));
    let n = blob.len();
    for c in blob[n - out * num_groups * 2..].chunks_exact_mut(2) {
        c.copy_from_slice(&0x3C00u16.to_le_bytes());
    }
    CqWeight::from_blob(&blob, out, in_feat, group, bits, &cb).expect("build")
}

fn bench_matvec(c: &mut Criterion) {
    let mut g = c.benchmark_group("cq_matvec");
    for &(out, in_feat, bits, label) in &[
        (512usize, 512usize, 2u8, "512x512_w2"),
        (256, 512, 2, "256x512_w2"),
        (8192, 512, 4, "8192x512_w4_lm_head"),
    ] {
        let w = make(out, in_feat, bits, 128);
        let x: Vec<f32> = (0..in_feat).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut xh = vec![0.0f32; w.prepared_len()];
        w.prepare_input(&x, &mut xh);
        let mut y = vec![0.0f32; out];

        // One multiply-accumulate per weight element.
        g.throughput(Throughput::Elements((out * in_feat) as u64));
        g.bench_function(label, |b| {
            b.iter(|| {
                w.matvec_prepared(black_box(&xh), black_box(&mut y));
            })
        });
    }
    g.finish();
}

/// Batched matmul against exactly the same work done as repeated matvecs.
///
/// Both arms share one `xh` buffer and one `y` buffer and are called identically,
/// so the ratio is attributable to the loop nesting alone. This is the figure
/// that decides whether batched prefill is worth the refactor.
/// Optimised kernel against the simple reference, measured back to back in one
/// group so both see the same clock state. Sustained benchmarking downclocks this
/// machine by ~1.5x, so absolute figures recorded minutes apart are not
/// comparable; a ratio measured inside one group is.
fn bench_vs_reference(c: &mut Criterion) {
    let mut g = c.benchmark_group("vs_reference");
    for &(out, in_feat, bits, label) in &[
        (512usize, 512usize, 2u8, "512x512_w2"),
        (8192, 512, 4, "8192x512_w4_lm_head"),
    ] {
        let w = make(out, in_feat, bits, 128);
        let x: Vec<f32> = (0..in_feat).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut xh = vec![0.0f32; w.prepared_len()];
        w.prepare_input(&x, &mut xh);
        let mut y = vec![0.0f32; out];
        g.throughput(Throughput::Elements((out * in_feat) as u64));
        g.bench_function(format!("{label}/optimised"), |b| {
            b.iter(|| w.matvec_prepared(black_box(&xh), black_box(&mut y)))
        });
        #[cfg(feature = "bench-internals")]
        g.bench_function(format!("{label}/single_acc_lut"), |b| {
            b.iter(|| w.matvec_prepared_single_acc(black_box(&xh), black_box(&mut y)))
        });
        g.bench_function(format!("{label}/reference"), |b| {
            b.iter(|| w.matvec_prepared_reference(black_box(&xh), black_box(&mut y)))
        });
    }
    g.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut g = c.benchmark_group("cq_batch_512x512_w2");
    let w = make(512, 512, 2, 128);
    let pl = w.prepared_len();
    for &batch in &[1usize, 8, 32, 128] {
        let mut xh = vec![0.0f32; batch * pl];
        for b in 0..batch {
            let x: Vec<f32> = (0..512)
                .map(|i| ((i + b * 7) as f32 * 0.13).sin())
                .collect();
            w.prepare_input(&x, &mut xh[b * pl..(b + 1) * pl]);
        }
        let mut y = vec![0.0f32; batch * 512];
        let mut acc = vec![0.0f32; batch];

        g.throughput(Throughput::Elements((batch * 512 * 512) as u64));
        g.bench_function(format!("matmul_b{batch}"), |bb| {
            bb.iter(|| {
                w.matmul_rows_prepared(&xh, batch, 0, 512, &mut y, &mut acc);
                black_box(&y);
            })
        });
        g.bench_function(format!("matvec_x{batch}"), |bb| {
            bb.iter(|| {
                for i in 0..batch {
                    w.matvec_prepared(&xh[i * pl..(i + 1) * pl], &mut y[i * 512..(i + 1) * 512]);
                }
                black_box(&y);
            })
        });
    }
    g.finish();
}

/// Same call, three ways of handing it the output buffer. Isolates how much of
/// the reported throughput is the kernel and how much is what the compiler can
/// prove about the destination slice.
fn bench_harness_effect(c: &mut Criterion) {
    let mut g = c.benchmark_group("harness_512x512_w2");
    let w = make(512, 512, 2, 128);
    let x: Vec<f32> = (0..512).map(|i| (i as f32 * 0.13).sin()).collect();
    let mut xh = vec![0.0f32; w.prepared_len()];
    w.prepare_input(&x, &mut xh);
    g.throughput(Throughput::Elements((512 * 512) as u64));

    let mut exact = vec![0.0f32; 512];
    g.bench_function("dest_exact_vec", |b| {
        b.iter(|| w.matvec_prepared(black_box(&xh), black_box(&mut exact)))
    });

    let mut big = vec![0.0f32; 128 * 512];
    g.bench_function("dest_slice_of_big", |b| {
        b.iter(|| w.matvec_prepared(black_box(&xh), black_box(&mut big[0..512])))
    });

    g.bench_function("dest_slice_in_loop", |b| {
        b.iter(|| {
            for i in 0..1usize {
                w.matvec_prepared(&xh, &mut big[i * 512..(i + 1) * 512]);
            }
            black_box(&big);
        })
    });
    g.finish();
}

fn bench_prepare(c: &mut Criterion) {
    let mut g = c.benchmark_group("cq_prepare_input");
    for &(in_feat, label) in &[(512usize, "d512"), (2048, "mhc2048")] {
        let w = make(64, in_feat, 2, 128);
        let x: Vec<f32> = (0..in_feat).map(|i| (i as f32 * 0.07).cos()).collect();
        let mut xh = vec![0.0f32; w.prepared_len()];
        g.throughput(Throughput::Elements(in_feat as u64));
        g.bench_function(label, |b| {
            b.iter(|| w.prepare_input(black_box(&x), black_box(&mut xh)))
        });
    }
    g.finish();
}

fn bench_fwht(c: &mut Criterion) {
    let mut g = c.benchmark_group("fwht");
    for &n in &[128usize, 512] {
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.21).sin()).collect();
        let mut buf = src.clone();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(format!("n{n}"), |b| {
            b.iter(|| {
                buf.copy_from_slice(&src);
                fwht(black_box(&mut buf));
            })
        });
        g.bench_function(format!("n{n}_normalized"), |b| {
            b.iter(|| {
                buf.copy_from_slice(&src);
                fwht_normalized(black_box(&mut buf));
            })
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_matvec,
    bench_vs_reference,
    bench_matmul,
    bench_harness_effect,
    bench_prepare,
    bench_fwht
);
criterion_main!(benches);
