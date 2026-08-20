//! Batched prefill: many prompt positions through one pass over the weights.
//!
//! [`V2Model::step`] is the reference path and processes one position at a time,
//! which means it walks the entire packed weight set once per token. Prefill
//! feeds a whole prompt, so those walks are pure repetition: every position
//! multiplies by the *same* weights.
//!
//! This module keeps the same arithmetic but inverts the loop nesting, so each
//! quantisation group is decoded once and applied to every position in the chunk
//! (see [`CqWeight::matmul_rows_prepared`]). The decode work and the weight
//! traffic are paid once per chunk instead of once per token.
//!
//! # Why it is exact, not merely close
//!
//! `matmul_rows_prepared` is bit-identical to repeated `matvec_rows_prepared`,
//! and everything outside the projections is per-position arithmetic that is
//! untouched. So a batched prefill must reproduce the sequential path's logits
//! exactly, which is what `prefill_batch_matches_sequential` asserts. The
//! sequential path stays under test by the 788-point reference ladder, so the two
//! paths hold each other in place.
//!
//! # What has to stay sequential
//!
//! Only attention. Position `p` attends to keys at `<= p`, including keys from
//! earlier positions *in the same chunk*, so the chunk's K/V are all computed and
//! written to the cache before any position attends. Everything else — the mHC
//! router, the Engram lookups, HadamardMLP — is per-position and order-free.
//!
//! Chunking is therefore also a correctness check: a prompt longer than one chunk
//! exercises the cross-chunk KV handoff, which is the same machinery a multi-turn
//! session needs.

use crate::math;
use crate::norm::zc_rms_norm_vec;
use crate::ops::{sigmoid, softmax_inplace};
use crate::v2::kernels::{hadamard_mlp, rms_unit_to, sinkhorn};
use crate::v2::model::{V2Model, V2State};
use alloc::vec;
use alloc::vec::Vec;

/// Default positions per chunk.
///
/// Measured on the shipped geometry with `parallel` enabled, prefill cost per
/// token: 5.74 ms unbatched, 5.06 at chunk 8, 2.80 at 32, 2.32 at 64, 2.03 at 128
/// and above.
///
/// Batching *loses* below about 16 positions — the per-group weight decode is
/// amortised over too few of them to pay for the extra passes over the
/// activations — so a small chunk is worse than none. The curve flattens at 128,
/// which is where this sits: the earlier default of 64 was chosen when the path
/// was serial and 64 was within 1% of the peak, but with row-split threading the
/// larger chunk is worth 8% and the extra 4 MB of scratch is small next to the
/// 14 MB KV ring.
pub const DEFAULT_CHUNK: usize = 128;

/// Reusable scratch for [`V2Model::prefill_batch`].
///
/// Allocated once and reused across prompts. Separate from [`V2State`] because it
/// is per-call working memory, not per-sequence state: the KV cache, token
/// history and Engram ring stay in `V2State` and are shared with the sequential
/// path.
pub struct V2Batch {
    cap: usize,
    lanes: Vec<f32>,
    lanes_prev: Vec<f32>,
    nx: Vec<f32>,
    nx_prep: Vec<f32>,
    u: Vec<f32>,
    bx: Vec<f32>,
    y: Vec<f32>,
    h: Vec<f32>,
    h_prep: Vec<f32>,
    hpre: Vec<f32>,
    hpost: Vec<f32>,
    hres: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    attn_out: Vec<f32>,
    attn_prep: Vec<f32>,
    attn_result: Vec<f32>,
    mlp_out: Vec<f32>,
    hada: Vec<f32>,
    engram_k: Vec<f32>,
    engram_val: Vec<f32>,
    /// Per-position fetched-entry vectors, and their rotations, for one site.
    e_all: Vec<f32>,
    e_prep_all: Vec<f32>,
    /// Per-position un-convolved Engram values for the chunk.
    e_val_all: Vec<f32>,
    scores: Vec<f32>,
    /// Column accumulators for the batched matmul.
    acc: Vec<f32>,
}

impl V2Batch {
    /// Allocate scratch for up to `cap` positions per chunk. `cap` is clamped to
    /// at least 1.
    pub fn new(model: &V2Model, cap: usize) -> Self {
        let cap = cap.max(1);
        let cfg = &model.cfg;
        let (d, n) = (cfg.d_model, cfg.mhc_lanes);
        let sites = cfg.engram.sites.len();
        let layer0 = &model.layers[0];
        Self {
            cap,
            lanes: vec![0.0; cap * n * d],
            lanes_prev: vec![0.0; cap * n * d],
            nx: vec![0.0; cap * n * d],
            nx_prep: vec![0.0; cap * model.mhc.phi_pre.prepared_len()],
            u: vec![0.0; cap * d],
            bx: vec![0.0; cap * d],
            y: vec![0.0; cap * d],
            h: vec![0.0; cap * d],
            h_prep: vec![0.0; cap * layer0.q_proj.prepared_len()],
            hpre: vec![0.0; cap * n],
            hpost: vec![0.0; cap * n],
            hres: vec![0.0; cap * n * n],
            q: vec![0.0; cap * cfg.attn_dim()],
            k: vec![0.0; cap * cfg.kv_dim()],
            v: vec![0.0; cap * cfg.kv_dim()],
            gate: vec![0.0; cap * cfg.attn_dim()],
            attn_out: vec![0.0; cap * cfg.attn_dim()],
            attn_prep: vec![0.0; cap * layer0.out_proj.prepared_len()],
            attn_result: vec![0.0; cap * d],
            mlp_out: vec![0.0; cap * d],
            hada: vec![0.0; cfg.hada_n],
            engram_k: vec![0.0; cap * sites * d],
            engram_val: vec![0.0; cap * sites * d],
            e_all: vec![0.0; cap * cfg.engram.fetched_dim().max(1)],
            e_prep_all: vec![
                0.0;
                cap * model
                    .engrams
                    .first()
                    .map_or(1, |e| e.value_proj.prepared_len())
            ],
            e_val_all: vec![0.0; cap * d],
            scores: vec![0.0; cfg.max_seq_len],
            acc: vec![0.0; cap],
        }
    }

    /// Positions per chunk.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Approximate scratch footprint in bytes.
    pub fn bytes(&self) -> usize {
        let f = |v: &Vec<f32>| v.len() * 4;
        f(&self.lanes)
            + f(&self.lanes_prev)
            + f(&self.nx)
            + f(&self.nx_prep)
            + f(&self.u)
            + f(&self.bx)
            + f(&self.y)
            + f(&self.h)
            + f(&self.h_prep)
            + f(&self.hpre)
            + f(&self.hpost)
            + f(&self.hres)
            + f(&self.q)
            + f(&self.k)
            + f(&self.v)
            + f(&self.gate)
            + f(&self.attn_out)
            + f(&self.attn_prep)
            + f(&self.attn_result)
            + f(&self.mlp_out)
            + f(&self.hada)
            + f(&self.engram_k)
            + f(&self.engram_val)
            + f(&self.scores)
            + f(&self.acc)
    }
}

impl V2Model {
    /// Feed `tokens` at the current position, batching the projections.
    ///
    /// Chunks internally at [`V2Batch::cap`]. When `logits` is `Some`, it receives
    /// the logits for the **final** token only — the intermediate positions skip
    /// the tied LM head, which is what makes prefill cheaper than the same number
    /// of [`step`] calls even before batching.
    ///
    /// Leaves `state` exactly as the equivalent run of [`step_prefill`] would,
    /// so decoding can continue with [`step`].
    ///
    /// [`step`]: V2Model::step
    /// [`step_prefill`]: V2Model::step_prefill
    pub fn prefill_batch(
        &self,
        tokens: &[u32],
        state: &mut V2State,
        batch: &mut V2Batch,
        mut logits: Option<&mut [f32]>,
    ) -> Result<(), &'static str> {
        let cfg = &self.cfg;
        if tokens.is_empty() {
            return Ok(());
        }
        if state.pos() + tokens.len() > cfg.max_seq_len {
            return Err("sequence exceeds max_seq_len");
        }
        if tokens.iter().any(|&t| t as usize >= cfg.vocab_size) {
            return Err("token id outside vocabulary");
        }
        if logits.as_ref().is_some_and(|l| l.len() != cfg.vocab_size) {
            return Err("logits buffer is not vocab_size long");
        }

        let n_chunks = tokens.len().div_ceil(batch.cap);
        for (ci, chunk) in tokens.chunks(batch.cap).enumerate() {
            let last_chunk = ci + 1 == n_chunks;
            self.prefill_chunk(chunk, state, batch)?;
            if last_chunk {
                if let Some(l) = logits.as_deref_mut() {
                    // The final position's lane state is the last row of `lanes`.
                    self.finish_logits(chunk.len() - 1, batch, l);
                }
            }
        }
        Ok(())
    }

    /// Pool, norm and project the LM head for one position of the chunk.
    fn finish_logits(&self, i: usize, batch: &mut V2Batch, logits: &mut [f32]) {
        let cfg = &self.cfg;
        let (d, n) = (cfg.d_model, cfg.mhc_lanes);
        let lanes = &batch.lanes[i * n * d..(i + 1) * n * d];
        let out = &mut batch.attn_result[..d];
        for (c, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for lane in 0..n {
                acc += lanes[lane * d + c];
            }
            *slot = acc / n as f32;
        }
        zc_rms_norm_vec(out, &self.final_norm);
        // `attn_prep` is at least `d` rounded to the group, so it can hold the
        // rotated LM-head input.
        let prep_len = self.embedding.prepared_len();
        let prep = &mut batch.attn_prep[..prep_len];
        self.embedding.prepare_input(out, prep);
        self.embedding.matvec_prepared(prep, logits);
    }

    fn prefill_chunk(
        &self,
        chunk: &[u32],
        state: &mut V2State,
        batch: &mut V2Batch,
    ) -> Result<(), &'static str> {
        let cfg = &self.cfg;
        let (d, n) = (cfg.d_model, cfg.mhc_lanes);
        let k = chunk.len();
        let p0 = state.pos();
        let sites = cfg.engram.sites.len();

        // ---- per-position setup: embedding and Engram.
        for (i, &tok) in chunk.iter().enumerate() {
            let pos = p0 + i;
            state.set_history(pos, tok);
            let row = &mut batch.u[..d];
            self.embedding.dequantize_row(tok as usize, row);
            let scale = math::sqrt(d as f32);
            for v in row.iter_mut() {
                *v *= scale;
            }
            for lane in 0..n {
                batch.lanes[(i * n + lane) * d..(i * n + lane + 1) * d].copy_from_slice(row);
            }
        }

        // Engram, batched. The per-position form was the last serial matmul left
        // in prefill: two 512x512 projections per site per position, which at a
        // 103-token prompt is as many multiply-adds as all the attention
        // projections put together.
        self.batch_engram(chunk.len(), p0, state, batch);
        let _ = sites;

        for li in 0..cfg.num_layers {
            self.batch_mhc_pre(li, k, batch);
            self.batch_engram_gate(li, k, batch);
            self.batch_attention(li, k, p0, state, batch);
            self.batch_mlp(li, k, batch);
            self.batch_mhc_post(li, k, batch);
        }

        state.advance(k);
        Ok(())
    }

    /// Engram keys and values for a whole chunk.
    ///
    /// Mirrors `V2Model::engram_fill` but hoists the two projections out of the
    /// position loop. The tap convolution still runs per position and still reads
    /// `pos - j * dilation`, which may fall in an earlier chunk — those values are
    /// in the ring, and the ones inside this chunk have just been written, so the
    /// convolution can run after the whole batch of values exists.
    fn batch_engram(&self, k: usize, p0: usize, state: &mut V2State, b: &mut V2Batch) {
        let cfg = &self.cfg;
        if cfg.engram.sites.is_empty() {
            return;
        }
        let d = cfg.d_model;
        let eg = &cfg.engram;
        let sites = eg.sites.len();
        let heads = eg.heads();
        let dil = eg.conv_dilation;
        let fetched = eg.fetched_dim();
        let ring = state.ring();

        for site in 0..self.engrams.len() {
            let weights = &self.engrams[site];
            let prep = weights.value_proj.prepared_len();

            // Fetched entries for every position, then one rotation each.
            for i in 0..k {
                let (e_all, hist) = (&mut b.e_all, state.history_slice());
                self.engram_fetch_into(
                    p0 + i,
                    site,
                    heads,
                    hist,
                    &mut e_all[i * fetched..(i + 1) * fetched],
                );
            }
            for i in 0..k {
                let (e_all, e_prep_all) = (&b.e_all, &mut b.e_prep_all);
                weights.value_proj.prepare_input(
                    &e_all[i * fetched..(i + 1) * fetched],
                    &mut e_prep_all[i * prep..(i + 1) * prep],
                );
            }
            let xh = &b.e_prep_all[..k * prep];

            // Un-convolved values for the whole chunk, into a per-position
            // buffer — NOT straight into the ring.
            //
            // The ring holds only `conv_taps * dilation + 1` positions (13 at the
            // shipped geometry). Writing a 64-position chunk into it before
            // convolving would overwrite each position's own inputs; the values
            // therefore live in `e_val_all` for the duration of the chunk, and
            // the ring is refreshed afterwards for the *next* chunk to read.
            weights.value_proj.matmul_rows_prepared(
                xh,
                k,
                0,
                d,
                &mut b.e_val_all[..k * d],
                &mut b.acc,
            );

            // Keys for the chunk.
            let mut keys = core::mem::take(&mut b.mlp_out);
            weights
                .key_proj
                .matmul_rows_prepared(xh, k, 0, d, &mut keys[..k * d], &mut b.acc);
            for i in 0..k {
                let base = (i * sites + site) * d;
                b.engram_k[base..base + d].copy_from_slice(&keys[i * d..(i + 1) * d]);
            }
            b.mlp_out = keys;

            // Tap convolution. A tap reaching back inside this chunk reads
            // `e_val_all`; one reaching into an earlier chunk reads the ring.
            for i in 0..k {
                let pos = p0 + i;
                let vbase = (i * sites + site) * d;
                b.engram_val[vbase..vbase + d].fill(0.0);
                for j in 0..eg.conv_taps {
                    let shift = j * dil;
                    if shift > pos {
                        continue;
                    }
                    let src_pos = pos - shift;
                    let tap = &weights.taps[j * d..(j + 1) * d];
                    if src_pos >= p0 {
                        let off = (src_pos - p0) * d;
                        let v = &b.e_val_all[off..off + d];
                        for (c, (&t, &vc)) in tap.iter().zip(v.iter()).enumerate() {
                            b.engram_val[vbase + c] += t * vc;
                        }
                    } else {
                        let slot = (site * ring + src_pos % ring) * d;
                        let v = state.engram_v_row(slot, d);
                        for (c, (&t, &vc)) in tap.iter().zip(v.iter()).enumerate() {
                            b.engram_val[vbase + c] += t * vc;
                        }
                    }
                }
            }

            // Refresh the ring so the next chunk's taps can reach back into this
            // one. Only the last `ring` positions survive, which is all a tap can
            // reach.
            for i in 0..k {
                let slot = (site * ring + (p0 + i) % ring) * d;
                state
                    .engram_v_slot(slot, d)
                    .copy_from_slice(&b.e_val_all[i * d..(i + 1) * d]);
            }
        }
    }

    fn batch_mhc_pre(&self, li: usize, k: usize, b: &mut V2Batch) {
        let cfg = &self.cfg;
        let (d, n) = (cfg.d_model, cfg.mhc_lanes);
        let prep = self.mhc.phi_pre.prepared_len();

        // Per position and independent: normalise the lane stream, then rotate it
        // for the three phi projections that follow.
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let phi = &self.mhc.phi_pre;
            b.nx[..k * n * d]
                .par_chunks_mut(n * d)
                .zip(b.nx_prep[..k * prep].par_chunks_mut(prep))
                .zip(b.lanes[..k * n * d].par_chunks(n * d))
                .for_each(|((nx, nx_prep), lanes)| {
                    rms_unit_to(lanes, nx);
                    phi.prepare_input(nx, nx_prep);
                });
        }
        #[cfg(not(feature = "parallel"))]
        for i in 0..k {
            rms_unit_to(
                &b.lanes[i * n * d..(i + 1) * n * d],
                &mut b.nx[i * n * d..(i + 1) * n * d],
            );
            let (nx, nx_prep) = (&b.nx, &mut b.nx_prep);
            self.mhc.phi_pre.prepare_input(
                &nx[i * n * d..(i + 1) * n * d],
                &mut nx_prep[i * prep..(i + 1) * prep],
            );
        }

        self.mhc.phi_pre.matmul_rows_prepared(
            &b.nx_prep[..k * prep],
            k,
            li * n,
            n,
            &mut b.hpre[..k * n],
            &mut b.acc,
        );
        let a = self.mhc.a_pre[li];
        for i in 0..k {
            for lane in 0..n {
                let z = a * b.hpre[i * n + lane]
                    + self.mhc.b_pre[li * n + lane]
                    + cfg.pre_off(li, lane);
                b.hpre[i * n + lane] = sigmoid(z);
            }
            for c in 0..d {
                let mut acc = 0.0f32;
                for lane in 0..n {
                    acc += b.hpre[i * n + lane] * b.lanes[(i * n + lane) * d + c];
                }
                b.u[i * d + c] = acc;
            }
            let (src, dst) = (i * d, i * d);
            for c in 0..d {
                b.bx[dst + c] = b.u[src + c];
            }
        }
    }

    fn batch_engram_gate(&self, li: usize, k: usize, b: &mut V2Batch) {
        let Some(site) = self.cfg.engram.site_of_layer(li) else {
            return;
        };
        let d = self.cfg.d_model;
        let sites = self.cfg.engram.sites.len();
        // Divide, as `V2Model::engram_gate` does. Multiplying by a precomputed
        // reciprocal is a different rounding, and at layers 2 and 15 that
        // difference propagates through the remaining blocks.
        let denom = math::sqrt(d as f32);

        for i in 0..k {
            let base = (i * sites + site) * d;
            // rms_unit of u and of the site key, then their dot.
            let u = &b.u[i * d..(i + 1) * d];
            let ek = &b.engram_k[base..base + d];
            let u_rms = rms_inv(u);
            let k_rms = rms_inv(ek);
            let mut dot = 0.0f32;
            for c in 0..d {
                dot += (u[c] * u_rms) * (ek[c] * k_rms);
            }
            let alpha = sigmoid(dot / denom);
            for c in 0..d {
                b.bx[i * d + c] = b.u[i * d + c] + alpha * b.engram_val[base + c];
            }
        }
    }

    fn batch_attention(
        &self,
        li: usize,
        k: usize,
        p0: usize,
        state: &mut V2State,
        b: &mut V2Batch,
    ) {
        let layer = &self.layers[li];
        let cfg = &self.cfg;
        let (h_n, kv_n, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let (d, attn, kv) = (cfg.d_model, cfg.attn_dim(), cfg.kv_dim());
        let reps = cfg.kv_repeat();
        let prep = layer.q_proj.prepared_len();

        // Normed block input, rotated once per position and shared by q/k/v/gate.
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            b.h[..k * d]
                .par_chunks_mut(d)
                .zip(b.h_prep[..k * prep].par_chunks_mut(prep))
                .zip(b.bx[..k * d].par_chunks(d))
                .for_each(|((h, h_prep), bx)| {
                    h.copy_from_slice(bx);
                    zc_rms_norm_vec(h, &layer.norm_in);
                    layer.q_proj.prepare_input(h, h_prep);
                });
        }
        #[cfg(not(feature = "parallel"))]
        for i in 0..k {
            b.h[i * d..(i + 1) * d].copy_from_slice(&b.bx[i * d..(i + 1) * d]);
            zc_rms_norm_vec(&mut b.h[i * d..(i + 1) * d], &layer.norm_in);
            let (h, h_prep) = (&b.h, &mut b.h_prep);
            layer.q_proj.prepare_input(
                &h[i * d..(i + 1) * d],
                &mut h_prep[i * prep..(i + 1) * prep],
            );
        }
        let xh = &b.h_prep[..k * prep];
        layer
            .q_proj
            .matmul_rows_prepared(xh, k, 0, attn, &mut b.q[..k * attn], &mut b.acc);
        layer
            .k_proj
            .matmul_rows_prepared(xh, k, 0, kv, &mut b.k[..k * kv], &mut b.acc);
        layer
            .v_proj
            .matmul_rows_prepared(xh, k, 0, kv, &mut b.v[..k * kv], &mut b.acc);
        layer
            .gate_proj
            .matmul_rows_prepared(xh, k, 0, attn, &mut b.gate[..k * attn], &mut b.acc);

        // Per-head norms and RoPE for the whole chunk, then write-and-attend in
        // position order.
        //
        // The write and the attention MUST be interleaved, not batched into two
        // passes. The KV cache is a ring of `kv_window` positions, so writing
        // position `p+1` overwrites the slot holding `p+1-window` — which is
        // still inside position `p`'s window. Writing the whole chunk first would
        // clobber the oldest key every earlier position in the chunk still needs.
        // (The projections above are still batched; only this loop is ordered.)
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            b.q[..k * attn]
                .par_chunks_mut(attn)
                .zip(b.k[..k * kv].par_chunks_mut(kv))
                .enumerate()
                .for_each(|(i, (q, kk))| {
                    for head in 0..h_n {
                        zc_rms_norm_vec(&mut q[head * hd..(head + 1) * hd], &layer.q_norm);
                    }
                    for head in 0..kv_n {
                        zc_rms_norm_vec(&mut kk[head * hd..(head + 1) * hd], &layer.k_norm);
                    }
                    self.rope_apply(q, 1, h_n, hd, p0 + i);
                    self.rope_apply(kk, 1, kv_n, hd, p0 + i);
                });
        }
        #[cfg(not(feature = "parallel"))]
        for i in 0..k {
            let pos = p0 + i;
            let q = &mut b.q[i * attn..(i + 1) * attn];
            for head in 0..h_n {
                zc_rms_norm_vec(&mut q[head * hd..(head + 1) * hd], &layer.q_norm);
            }
            let kk = &mut b.k[i * kv..(i + 1) * kv];
            for head in 0..kv_n {
                zc_rms_norm_vec(&mut kk[head * hd..(head + 1) * hd], &layer.k_norm);
            }
            self.rope_apply(q, 1, h_n, hd, pos);
            self.rope_apply(kk, 1, kv_n, hd, pos);
        }

        let scale = 1.0 / math::sqrt(hd as f32);
        for i in 0..k {
            let pos = p0 + i;
            state.write_kv(
                li,
                pos,
                &b.k[i * kv..(i + 1) * kv],
                &b.v[i * kv..(i + 1) * kv],
            );

            let lo = state.window_lo(pos);
            let span = pos - lo + 1;
            for head in 0..h_n {
                let kvh = head / reps;
                let qrow = &b.q[i * attn + head * hd..i * attn + (head + 1) * hd];
                for s in 0..span {
                    let krow = state.k_at(li, kvh, lo + s);
                    let mut acc = 0.0f32;
                    for (t, &kt) in krow.iter().enumerate() {
                        acc += qrow[t] * kt;
                    }
                    b.scores[s] = acc * scale;
                }
                softmax_inplace(&mut b.scores[..span]);
                let obase = i * attn + head * hd;
                b.attn_out[obase..obase + hd].fill(0.0);
                for s in 0..span {
                    let w = b.scores[s];
                    let vrow = state.v_at(li, kvh, lo + s);
                    for (t, &vt) in vrow.iter().enumerate() {
                        b.attn_out[obase + t] += w * vt;
                    }
                }
            }
        }

        // Sigmoid gate, then out_proj for the chunk.
        for i in 0..k * attn {
            b.attn_out[i] *= sigmoid(b.gate[i]);
        }
        let oprep = layer.out_proj.prepared_len();
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            b.attn_prep[..k * oprep]
                .par_chunks_mut(oprep)
                .zip(b.attn_out[..k * attn].par_chunks(attn))
                .for_each(|(ap, ao)| layer.out_proj.prepare_input(ao, ap));
        }
        #[cfg(not(feature = "parallel"))]
        for i in 0..k {
            let (ao, ap) = (&b.attn_out, &mut b.attn_prep);
            layer.out_proj.prepare_input(
                &ao[i * attn..(i + 1) * attn],
                &mut ap[i * oprep..(i + 1) * oprep],
            );
        }
        layer.out_proj.matmul_rows_prepared(
            &b.attn_prep[..k * oprep],
            k,
            0,
            d,
            &mut b.attn_result[..k * d],
            &mut b.acc,
        );
    }

    /// Post-attention residual and HadamardMLP.
    ///
    /// Purely per position — no cross-position term anywhere — so with the
    /// `parallel` feature the chunk's positions run concurrently. This and
    /// [`batch_mhc_post`] are where prefill's remaining serial time sat once the
    /// projections were batched.
    ///
    /// [`batch_mhc_post`]: V2Model::batch_mhc_post
    fn batch_mlp(&self, li: usize, k: usize, b: &mut V2Batch) {
        let layer = &self.layers[li];
        let d = self.cfg.d_model;
        let g = sigmoid(layer.attn_gate);

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let hada_n = self.cfg.hada_n;
            b.y[..k * d]
                .par_chunks_mut(d)
                .zip(b.attn_result[..k * d].par_chunks_mut(d))
                .zip(b.h[..k * d].par_chunks_mut(d))
                .zip(b.mlp_out[..k * d].par_chunks_mut(d))
                .zip(b.bx[..k * d].par_chunks(d))
                .zip(b.u[..k * d].par_chunks(d))
                .for_each_init(
                    || vec![0.0f32; hada_n],
                    |hada, (((((y, ar), h), mlp), bx), u)| {
                        mlp_one(layer, g, y, ar, h, mlp, bx, u, hada);
                    },
                );
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut hada = core::mem::take(&mut b.hada);
            for i in 0..k {
                let (lo, hi) = (i * d, (i + 1) * d);
                // Split the borrows by hand: each of these is a distinct field.
                let (y, ar) = (
                    &mut b.y[lo..hi] as *mut [f32],
                    &mut b.attn_result[lo..hi] as *mut [f32],
                );
                // SAFETY: y, ar, h, mlp, bx and u are disjoint fields of `b`.
                unsafe {
                    mlp_one(
                        layer,
                        g,
                        &mut *y,
                        &mut *ar,
                        &mut b.h[lo..hi],
                        &mut b.mlp_out[lo..hi],
                        &b.bx[lo..hi],
                        &b.u[lo..hi],
                        &mut hada,
                    );
                }
            }
            b.hada = hada;
        }
    }

    fn batch_mhc_post(&self, li: usize, k: usize, b: &mut V2Batch) {
        let cfg = &self.cfg;
        let (d, n) = (cfg.d_model, cfg.mhc_lanes);
        let prep = self.mhc.phi_pre.prepared_len();
        let xh = &b.nx_prep[..k * prep];

        self.mhc
            .phi_post
            .matmul_rows_prepared(xh, k, li * n, n, &mut b.hpost[..k * n], &mut b.acc);
        self.mhc.phi_res.matmul_rows_prepared(
            xh,
            k,
            li * n * n,
            n * n,
            &mut b.hres[..k * n * n],
            &mut b.acc,
        );

        let (a_post, a_res) = (self.mhc.a_post[li], self.mhc.a_res[li]);
        let b_post = &self.mhc.b_post[li * n..(li + 1) * n];
        let b_res = &self.mhc.b_res[li * n * n..(li + 1) * n * n];
        let offs: Vec<f32> = (0..n).map(|lane| cfg.post_off(li, lane)).collect();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            b.lanes[..k * n * d]
                .par_chunks_mut(n * d)
                .zip(b.lanes_prev[..k * n * d].par_chunks_mut(n * d))
                .zip(b.hpost[..k * n].par_chunks_mut(n))
                .zip(b.hres[..k * n * n].par_chunks_mut(n * n))
                .zip(b.y[..k * d].par_chunks(d))
                .for_each(|((((lanes, prev), hpost), hres), y)| {
                    mhc_post_one(
                        n, d, a_post, a_res, b_post, b_res, &offs, lanes, prev, hpost, hres, y,
                    );
                });
        }

        #[cfg(not(feature = "parallel"))]
        for i in 0..k {
            let lanes = &mut b.lanes[i * n * d..(i + 1) * n * d] as *mut [f32];
            let prev = &mut b.lanes_prev[i * n * d..(i + 1) * n * d] as *mut [f32];
            // SAFETY: lanes, lanes_prev, hpost, hres and y are distinct fields.
            unsafe {
                mhc_post_one(
                    n,
                    d,
                    a_post,
                    a_res,
                    b_post,
                    b_res,
                    &offs,
                    &mut *lanes,
                    &mut *prev,
                    &mut b.hpost[i * n..(i + 1) * n],
                    &mut b.hres[i * n * n..(i + 1) * n * n],
                    &b.y[i * d..(i + 1) * d],
                );
            }
        }
    }
}

/// One position's post-attention residual and HadamardMLP.
#[allow(clippy::too_many_arguments)]
#[inline]
fn mlp_one(
    layer: &crate::v2::model::V2Layer,
    g: f32,
    y: &mut [f32],
    attn_result: &mut [f32],
    h: &mut [f32],
    mlp_out: &mut [f32],
    bx: &[f32],
    u: &[f32],
    hada: &mut [f32],
) {
    let d = y.len();
    zc_rms_norm_vec(attn_result, &layer.post_norm);
    for c in 0..d {
        y[c] = bx[c] + g * attn_result[c];
    }
    h.copy_from_slice(y);
    zc_rms_norm_vec(h, &layer.pre_hada);
    hadamard_mlp(h, &layer.d1, &layer.d2, &layer.d3, hada, mlp_out);
    for c in 0..d {
        y[c] += mlp_out[c];
        y[c] -= u[c];
    }
}

/// One position's mHC post-routing and lane update.
#[allow(clippy::too_many_arguments)]
#[inline]
fn mhc_post_one(
    n: usize,
    d: usize,
    a_post: f32,
    a_res: f32,
    b_post: &[f32],
    b_res: &[f32],
    post_off: &[f32],
    lanes: &mut [f32],
    prev: &mut [f32],
    hpost: &mut [f32],
    hres: &mut [f32],
    y: &[f32],
) {
    for lane in 0..n {
        let z = a_post * hpost[lane] + b_post[lane] + post_off[lane];
        hpost[lane] = 2.0 * sigmoid(z);
    }
    for (j, v) in hres.iter_mut().enumerate() {
        *v = a_res * *v + b_res[j];
    }
    sinkhorn(hres, n);

    prev.copy_from_slice(lanes);
    for lane_i in 0..n {
        let hp = hpost[lane_i];
        for c in 0..d {
            let mut acc = 0.0f32;
            for lane_j in 0..n {
                acc += hres[lane_i * n + lane_j] * prev[lane_j * d + c];
            }
            lanes[lane_i * d + c] = acc + hp * y[c];
        }
    }
}

/// `1 / sqrt(mean(x^2) + eps)`, the scale `rms_unit` applies.
#[inline]
fn rms_inv(x: &[f32]) -> f32 {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    1.0 / math::sqrt(mean_sq + crate::v2::kernels::EPS)
}
