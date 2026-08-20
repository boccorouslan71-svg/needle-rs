//! Probe-pooling heads: contrastive retrieval and confidence.
//!
//! Port of `architecture.probe_pool`, `ContrastiveHead` and `ConfidenceHead`.
//!
//! # What a cell is
//!
//! These heads do not read the final hidden state. They pool over *cells* —
//! `hidden_cells` returns `stack([x0, *hidden], axis=2)`, so for a `T`-token
//! input there are `T * (L + 1)` cells: the scaled embedding, plus the lane-mean
//! of the residual stream after each of the `L` layers. Every cell of every
//! position competes in one softmax per probe.
//!
//! # Streaming instead of storing
//!
//! The reference materialises all `T * (L + 1)` cells and softmaxes across them:
//! at `T = 2048` that is 2048 x 28 x 512 floats, 117 MB. Since a softmax-weighted
//! average is exactly what an online (running-max) softmax computes, the same
//! result is reachable in `O(probes * d_model)` state — 16 KB for the confidence
//! head — with cells consumed as the forward pass produces them and never
//! retained.
//!
//! [`ProbePool::push`] takes one cell; [`ProbePool::pooled`] finishes. The
//! arithmetic is a reassociation of the reference's, so results agree to f32
//! rounding rather than bit-exactly.
//!
//! # Masking
//!
//! `hidden_cells` runs `make_causal_mask & make_padding_mask`, and `probe_pool`
//! additionally drops cells whose position holds a pad token. The caller is
//! responsible for not pushing cells for pad positions — see
//! `V2Engine::encode_contrastive`.
//!
//! Note also that `encode_contrastive` and `forward_confidence` default to
//! `window=0` (full causal), while the LM path runs the checkpoint's `kv_window`.
//! See [`V2State::set_kv_window`].
//!
//! [`V2State::set_kv_window`]: crate::v2::model::V2State::set_kv_window

use crate::math;
use alloc::vec;
use alloc::vec::Vec;

/// Head codes as stored in the container's `heads.manifest`.
pub const HEAD_CONTRASTIVE: u8 = 1;
pub const HEAD_CONFIDENCE: u8 = 2;

/// Epsilon inside the contrastive L2 normalisation, matching the reference.
pub const L2_EPS: f32 = 1e-12;

/// Streaming softmax-weighted pooling over cells, one accumulator per probe.
pub struct ProbePool {
    num_probes: usize,
    d: usize,
    /// `[num_probes * d]`.
    probes: Vec<f32>,
    /// Running max score per probe.
    m: Vec<f32>,
    /// Running sum of `exp(score - m)` per probe.
    s: Vec<f32>,
    /// Running `sum(exp(score - m) * cell)`, `[num_probes * d]`.
    acc: Vec<f32>,
    /// Divisor on the raw dot product: `sqrt(d_model)`.
    denom: f32,
    cells: usize,
}

impl ProbePool {
    /// `probes` is `[num_probes, d]`, row-major.
    pub fn new(probes: &[f32], num_probes: usize, d: usize) -> Result<Self, &'static str> {
        if num_probes == 0 || d == 0 {
            return Err("probe pool needs a non-empty probe matrix");
        }
        if probes.len() != num_probes * d {
            return Err("probe matrix length does not match num_probes * d");
        }
        Ok(Self {
            num_probes,
            d,
            probes: probes.to_vec(),
            m: vec![f32::NEG_INFINITY; num_probes],
            s: vec![0.0; num_probes],
            acc: vec![0.0; num_probes * d],
            denom: math::sqrt(d as f32),
            cells: 0,
        })
    }

    pub fn num_probes(&self) -> usize {
        self.num_probes
    }

    /// Width of the pooled output, `num_probes * d_model`.
    pub fn pooled_len(&self) -> usize {
        self.num_probes * self.d
    }

    /// Cells consumed so far.
    pub fn cells(&self) -> usize {
        self.cells
    }

    /// Fold one cell into the pool.
    pub fn push(&mut self, cell: &[f32]) {
        debug_assert_eq!(cell.len(), self.d);
        self.cells += 1;
        for k in 0..self.num_probes {
            let probe = &self.probes[k * self.d..(k + 1) * self.d];
            let mut dot = 0.0f32;
            for c in 0..self.d {
                dot += cell[c] * probe[c];
            }
            let z = dot / self.denom;

            let acc = &mut self.acc[k * self.d..(k + 1) * self.d];
            if z > self.m[k] {
                // Rebase onto the new maximum.
                let r = if self.m[k] == f32::NEG_INFINITY {
                    0.0
                } else {
                    math::exp(self.m[k] - z)
                };
                self.s[k] *= r;
                for a in acc.iter_mut() {
                    *a *= r;
                }
                self.m[k] = z;
            }
            let w = math::exp(z - self.m[k]);
            self.s[k] += w;
            for (a, &v) in acc.iter_mut().zip(cell.iter()) {
                *a += w * v;
            }
        }
    }

    /// Write the pooled vector, `[num_probes * d]` with probe-major layout —
    /// the flattening `probe_pool` produces with `reshape(b, -1)`.
    pub fn pooled(&self, out: &mut [f32]) -> Result<(), &'static str> {
        if self.cells == 0 {
            return Err("probe pool received no cells");
        }
        debug_assert_eq!(out.len(), self.pooled_len());
        for k in 0..self.num_probes {
            let inv = 1.0 / self.s[k];
            for c in 0..self.d {
                out[k * self.d + c] = self.acc[k * self.d + c] * inv;
            }
        }
        Ok(())
    }

    /// Clear, keeping the probe matrix and the allocations.
    pub fn reset(&mut self) {
        self.m.fill(f32::NEG_INFINITY);
        self.s.fill(0.0);
        self.acc.fill(0.0);
        self.cells = 0;
    }
}

/// A probe head's projection: `out = proj · pooled + bias`.
pub struct ProbeHeadWeights {
    /// [`HEAD_CONTRASTIVE`] or [`HEAD_CONFIDENCE`].
    pub code: u8,
    /// `[num_probes, d_model]`.
    pub probes: Vec<f32>,
    /// `[out_dim, num_probes * d_model]`, row-major — the container stores this
    /// pre-transposed, which is the orientation needed here.
    pub proj: Vec<f32>,
    /// `[out_dim]`. Zero for the contrastive head, which is bias-free upstream.
    pub bias: Vec<f32>,
    pub num_probes: usize,
    pub out_dim: usize,
    pub d_model: usize,
}

impl ProbeHeadWeights {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.probes.len() != self.num_probes * self.d_model {
            return Err("head probes shape does not match num_probes * d_model");
        }
        if self.proj.len() != self.out_dim * self.num_probes * self.d_model {
            return Err("head proj shape does not match out_dim * num_probes * d_model");
        }
        if self.bias.len() != self.out_dim {
            return Err("head bias length does not match out_dim");
        }
        Ok(())
    }

    pub fn pool(&self) -> Result<ProbePool, &'static str> {
        ProbePool::new(&self.probes, self.num_probes, self.d_model)
    }

    /// Project a pooled vector. `out` is `out_dim` long.
    pub fn project(&self, pooled: &[f32], out: &mut [f32]) {
        debug_assert_eq!(pooled.len(), self.num_probes * self.d_model);
        debug_assert_eq!(out.len(), self.out_dim);
        let n = pooled.len();
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &self.proj[o * n..(o + 1) * n];
            let mut acc = 0.0f32;
            for (r, p) in row.iter().zip(pooled.iter()) {
                acc += r * p;
            }
            *slot = acc + self.bias[o];
        }
    }

    /// Project and L2-normalise, as `ContrastiveHead` does.
    pub fn project_normalized(&self, pooled: &[f32], out: &mut [f32]) {
        self.project(pooled, out);
        let sq: f32 = out.iter().map(|v| v * v).sum();
        let inv = 1.0 / math::sqrt(sq + L2_EPS);
        for v in out.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Batch reference: the softmax-then-average the streaming pool replaces.
    fn pooled_reference(
        cells: &[Vec<f32>],
        probes: &[f32],
        num_probes: usize,
        d: usize,
    ) -> Vec<f32> {
        let denom = math::sqrt(d as f32);
        let mut out = vec![0.0f32; num_probes * d];
        for k in 0..num_probes {
            let probe = &probes[k * d..(k + 1) * d];
            let scores: Vec<f32> = cells
                .iter()
                .map(|c| c.iter().zip(probe).map(|(a, b)| a * b).sum::<f32>() / denom)
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&z| math::exp(z - max)).collect();
            let total: f32 = exps.iter().sum();
            for (w, cell) in exps.iter().zip(cells.iter()) {
                for c in 0..d {
                    out[k * d + c] += (w / total) * cell[c];
                }
            }
        }
        out
    }

    fn synth(n_cells: usize, d: usize, scale: f32) -> Vec<Vec<f32>> {
        (0..n_cells)
            .map(|i| {
                (0..d)
                    .map(|c| ((i * 31 + c * 7) as f32 * 0.017).sin() * scale)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn streaming_pool_matches_batch_softmax() {
        for &(n_cells, d, num_probes, scale) in &[
            (28usize, 16usize, 4usize, 1.0f32),
            (200, 32, 8, 3.0),
            (1, 8, 2, 1.0),
            (2800, 16, 4, 2.0),
        ] {
            let cells = synth(n_cells, d, scale);
            let probes: Vec<f32> = (0..num_probes * d)
                .map(|i| (i as f32 * 0.031).cos() * 0.7)
                .collect();

            let mut pool = ProbePool::new(&probes, num_probes, d).unwrap();
            for c in &cells {
                pool.push(c);
            }
            let mut got = vec![0.0f32; num_probes * d];
            pool.pooled(&mut got).unwrap();
            assert_eq!(pool.cells(), n_cells);

            let want = pooled_reference(&cells, &probes, num_probes, d);
            let mag = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
            for i in 0..got.len() {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-5 * mag,
                    "cells={n_cells} d={d} probes={num_probes} i={i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Large score spreads are where a naive streaming sum would overflow; the
    /// running max has to keep it finite.
    #[test]
    fn streaming_pool_survives_extreme_scores() {
        let d = 16;
        let mut cells = synth(50, d, 1.0);
        // One cell with a huge projection onto the probe, one with a huge negative.
        cells.push(vec![90.0; d]);
        cells.push(vec![-90.0; d]);
        let probes: Vec<f32> = vec![1.0; d];
        let mut pool = ProbePool::new(&probes, 1, d).unwrap();
        for c in &cells {
            pool.push(c);
        }
        let mut got = vec![0.0f32; d];
        pool.pooled(&mut got).unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
        // The dominant cell should win the softmax outright.
        for &v in &got {
            assert!((v - 90.0).abs() < 1e-2, "expected ~90, got {v}");
        }
    }

    /// Cell order must not matter — softmax is symmetric in its inputs.
    #[test]
    fn pooling_is_order_independent() {
        let d = 24;
        let cells = synth(60, d, 2.0);
        let probes: Vec<f32> = (0..3 * d).map(|i| (i as f32 * 0.05).sin()).collect();

        let mut a = ProbePool::new(&probes, 3, d).unwrap();
        for c in &cells {
            a.push(c);
        }
        let mut b = ProbePool::new(&probes, 3, d).unwrap();
        for c in cells.iter().rev() {
            b.push(c);
        }
        let (mut x, mut y) = (vec![0.0f32; 3 * d], vec![0.0f32; 3 * d]);
        a.pooled(&mut x).unwrap();
        b.pooled(&mut y).unwrap();
        for i in 0..x.len() {
            assert!((x[i] - y[i]).abs() < 1e-4, "i={i}: {} vs {}", x[i], y[i]);
        }
    }

    #[test]
    fn reset_restores_a_fresh_pool() {
        let d = 8;
        let cells = synth(10, d, 1.0);
        let probes: Vec<f32> = (0..2 * d).map(|i| i as f32 * 0.1).collect();
        let mut pool = ProbePool::new(&probes, 2, d).unwrap();
        for c in &cells {
            pool.push(c);
        }
        let mut first = vec![0.0f32; 2 * d];
        pool.pooled(&mut first).unwrap();

        pool.reset();
        assert_eq!(pool.cells(), 0);
        assert!(pool.pooled(&mut first.clone()).is_err());
        for c in &cells {
            pool.push(c);
        }
        let mut again = vec![0.0f32; 2 * d];
        pool.pooled(&mut again).unwrap();
        assert_eq!(first, again);
    }

    #[test]
    fn rejects_bad_probe_shapes() {
        assert!(ProbePool::new(&[1.0, 2.0], 0, 2).is_err());
        assert!(ProbePool::new(&[1.0, 2.0], 2, 0).is_err());
        assert!(ProbePool::new(&[1.0, 2.0, 3.0], 2, 2).is_err());
    }

    #[test]
    fn contrastive_projection_is_unit_norm() {
        let (p, d, out_dim) = (4usize, 8usize, 6usize);
        let h = ProbeHeadWeights {
            code: HEAD_CONTRASTIVE,
            probes: (0..p * d).map(|i| i as f32 * 0.01).collect(),
            proj: (0..out_dim * p * d)
                .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.05)
                .collect(),
            bias: vec![0.0; out_dim],
            num_probes: p,
            out_dim,
            d_model: d,
        };
        h.validate().unwrap();
        let pooled: Vec<f32> = (0..p * d).map(|i| (i as f32 * 0.07).sin()).collect();
        let mut out = vec![0.0f32; out_dim];
        h.project_normalized(&pooled, &mut out);
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    /// An all-zero projection must not produce NaN — that is what the epsilon
    /// inside the norm is for.
    #[test]
    fn contrastive_projection_handles_zero_vector() {
        let (p, d, out_dim) = (2usize, 4usize, 3usize);
        let h = ProbeHeadWeights {
            code: HEAD_CONTRASTIVE,
            probes: vec![0.0; p * d],
            proj: vec![0.0; out_dim * p * d],
            bias: vec![0.0; out_dim],
            num_probes: p,
            out_dim,
            d_model: d,
        };
        let mut out = vec![0.0f32; out_dim];
        h.project_normalized(&vec![0.0; p * d], &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn confidence_projection_applies_bias() {
        let (p, d) = (2usize, 4usize);
        let h = ProbeHeadWeights {
            code: HEAD_CONFIDENCE,
            probes: vec![0.5; p * d],
            proj: vec![1.0; p * d],
            bias: vec![-2.5],
            num_probes: p,
            out_dim: 1,
            d_model: d,
        };
        h.validate().unwrap();
        let pooled = vec![1.0f32; p * d];
        let mut out = vec![0.0f32; 1];
        h.project(&pooled, &mut out);
        // sum of eight ones, minus the bias.
        assert!((out[0] - (8.0 - 2.5)).abs() < 1e-6, "{}", out[0]);
    }

    #[test]
    fn validate_rejects_shape_mismatches() {
        let mut h = ProbeHeadWeights {
            code: HEAD_CONFIDENCE,
            probes: vec![0.0; 8],
            proj: vec![0.0; 8],
            bias: vec![0.0; 1],
            num_probes: 2,
            out_dim: 1,
            d_model: 4,
        };
        h.validate().unwrap();
        h.bias.push(0.0);
        assert!(h.validate().is_err());
        h.bias.pop();
        h.probes.push(0.0);
        assert!(h.validate().is_err());
    }
}
