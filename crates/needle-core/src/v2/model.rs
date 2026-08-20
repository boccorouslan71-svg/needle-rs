//! The v2 forward pass.
//!
//! Port of `needle/model/decode.py::_forward_cached` and the helpers it calls
//! (`_layer`, `_mhc`, `_block_cached`, `_attn_cached`, `_engram_kv`).
//!
//! One token is processed per call, which covers prefill and decode alike: with
//! the KV cache and the Engram history ring in place, a step at position `p`
//! depends only on state. That removes the reference's batch and sequence axes
//! and, with them, the masking that existed to neutralise padding.
//!
//! # Deriving the Engram window
//!
//! `_engram_kv` builds a window `padded[pos .. pos + W + S]` where
//! `W = conv_taps * conv_dilation` and `padded` is the token history with `W`
//! zeros prepended, then applies `_shift_right` along it. Reading that at the
//! kept output position collapses to plain history lookups:
//!
//! * the hash for an order-`o` table reads tokens `p, p-1, ..., p-(o-1)`
//! * `ngram_ok` zeroes the fetched entry when `p < o - 1`
//! * the value convolution reads `v(p - j * dilation)` for each tap `j`,
//!   contributing nothing when `j * dilation > p`
//!
//! so the only state needed is the token history plus the un-convolved value
//! vectors for the last `W` positions.

use crate::cq::CqWeight;
use crate::hadamard::hada_n;
use crate::math;
use crate::norm::zc_rms_norm_vec;
use crate::ops::{sigmoid, softmax_inplace};
use crate::rope::RopeCache;
use crate::v2::config::V2Config;
use crate::v2::kernels::{
    engram_index, engram_table_order, hadamard_mlp, rms_unit, rms_unit_to, sinkhorn,
};
use alloc::vec;
use alloc::vec::Vec;

/// One transformer layer's weights.
pub struct V2Layer {
    /// ZCRMSNorm scale before attention (`ZCRMSNorm_0`).
    pub norm_in: Vec<f32>,
    pub q_proj: CqWeight,
    pub k_proj: CqWeight,
    pub v_proj: CqWeight,
    /// Produces the sigmoid gate applied to the attention output.
    pub gate_proj: CqWeight,
    pub out_proj: CqWeight,
    /// Per-head ZCRMSNorm scales, `head_dim` long.
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    /// ZCRMSNorm scale after attention (`post_attn_norm`).
    pub post_norm: Vec<f32>,
    /// Pre-sigmoid scalar gating the attention residual.
    pub attn_gate: f32,
    /// ZCRMSNorm scale before HadamardMLP.
    pub pre_hada: Vec<f32>,
    /// HadamardMLP diagonals, `hada_n` long each.
    pub d1: Vec<f32>,
    pub d2: Vec<f32>,
    pub d3: Vec<f32>,
}

/// mHC router weights. Stored as one tall matrix per kind with a per-layer band.
pub struct V2Mhc {
    /// `[num_layers]` each.
    pub a_pre: Vec<f32>,
    pub a_post: Vec<f32>,
    pub a_res: Vec<f32>,
    /// `[num_layers * lanes]`.
    pub b_pre: Vec<f32>,
    pub b_post: Vec<f32>,
    /// `[num_layers * lanes * lanes]`, row-major in `(i, j)`.
    pub b_res: Vec<f32>,
    /// `[num_layers * lanes, mhc_width]` — row `layer * lanes + lane`.
    pub phi_pre: CqWeight,
    pub phi_post: CqWeight,
    /// `[num_layers * lanes^2, mhc_width]` — row `layer * lanes^2 + i * lanes + j`.
    pub phi_res: CqWeight,
}

/// One Engram site's weights.
pub struct V2Engram {
    /// `[num_tables * slots, sub_dim]` — row `table * slots + index`.
    pub tables: CqWeight,
    /// `[d_model, fetched_dim]`.
    pub key_proj: CqWeight,
    /// `[d_model, fetched_dim]`.
    pub value_proj: CqWeight,
    /// `[conv_taps, d_model]`.
    pub taps: Vec<f32>,
}

/// A loaded v2 model. Immutable during inference; per-sequence state lives in
/// [`V2State`].
pub struct V2Model {
    pub cfg: V2Config,
    /// Tied embedding, `[vocab_size, d_model]` — also the LM head, transposed.
    pub embedding: CqWeight,
    pub layers: Vec<V2Layer>,
    pub mhc: V2Mhc,
    pub engrams: Vec<V2Engram>,
    pub final_norm: Vec<f32>,
    rope: RopeCache,
    embed_scale: f32,
}

/// Per-sequence mutable state: KV cache, token history, Engram ring, and every
/// scratch buffer a step needs. Allocated once so a decode step does no heap work.
pub struct V2State {
    /// Per layer, `[kv_heads * cache_len * head_dim]`.
    ///
    /// `cache_len` is the attention window, not `max_seq_len`: with a sliding
    /// window the model never reads further back than `kv_window`, so the cache
    /// is a ring of that width indexed by `pos % cache_len`. At the shipped
    /// geometry that is 14 MB instead of 113 MB.
    ///
    /// Writing slot `pos % cache_len` overwrites position `pos - cache_len`,
    /// which is exactly the one that just fell out of the window — so the
    /// overwrite is only safe while `kv_window <= cache_len`, which
    /// [`V2State::set_kv_window`] enforces.
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    history: Vec<u32>,
    pos: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Positions the KV ring holds.
    cache_len: usize,
    /// Attention window in force. Starts at the model's `kv_window`; the probe
    /// heads run at 0 (full causal) because that is what `encode_contrastive`
    /// defaults to upstream.
    kv_window: usize,

    /// Un-convolved Engram values, `[sites * ring * d_model]`, indexed by
    /// `position % ring`.
    engram_v: Vec<f32>,
    ring: usize,

    // ---- scratch
    lanes: Vec<f32>,      // [lanes * d_model]  the mHC lane stream
    lanes_prev: Vec<f32>, // [lanes * d_model]  lane stream before the update
    nx: Vec<f32>,         // [lanes * d_model]  rms_unit of the lane stream
    nx_prep: Vec<f32>,
    u: Vec<f32>,  // [d_model]  lane-mixed block input
    bx: Vec<f32>, // [d_model]  u plus the Engram contribution
    y: Vec<f32>,  // [d_model]  block output, then block output minus u
    h: Vec<f32>,  // [d_model]  normed sub-block input
    h_prep: Vec<f32>,
    hpre: Vec<f32>,     // [lanes]
    hpost: Vec<f32>,    // [lanes]
    hres: Vec<f32>,     // [lanes * lanes]
    q: Vec<f32>,        // [attn_dim]
    k: Vec<f32>,        // [kv_dim]
    v: Vec<f32>,        // [kv_dim]
    gate: Vec<f32>,     // [attn_dim]
    attn_out: Vec<f32>, // [attn_dim]
    attn_prep: Vec<f32>,
    attn_result: Vec<f32>, // [d_model]  out_proj output
    scores: Vec<f32>,      // [max_seq_len]
    hada: Vec<f32>,        // [hada_n]
    mlp_out: Vec<f32>,     // [d_model]
    e_buf: Vec<f32>,       // [fetched_dim]
    e_prep: Vec<f32>,
    engram_k: Vec<f32>,   // [sites * d_model]
    engram_val: Vec<f32>, // [sites * d_model]  after the tap convolution
    /// Last Engram gate value per site, kept for tracing.
    alpha: Vec<f32>, // [sites]
    lm: Vec<f32>,         // [d_model]  pooled, normed hidden state
    lm_prep: Vec<f32>,
    tmp_a: Vec<f32>, // [d_model]
    tmp_b: Vec<f32>, // [d_model]
}

impl V2Model {
    pub fn new(
        cfg: V2Config,
        embedding: CqWeight,
        layers: Vec<V2Layer>,
        mhc: V2Mhc,
        engrams: Vec<V2Engram>,
        final_norm: Vec<f32>,
    ) -> Result<Self, &'static str> {
        cfg.validate()?;
        if layers.len() != cfg.num_layers {
            return Err("layer count does not match num_layers");
        }
        if engrams.len() != cfg.engram.sites.len() {
            return Err("engram count does not match site count");
        }
        if final_norm.len() != cfg.d_model {
            return Err("final_norm width does not match d_model");
        }
        if hada_n(cfg.d_model) != cfg.hada_n {
            return Err("hada_n is not d_model rounded up to a power of two");
        }
        if embedding.out_feat != cfg.vocab_size || embedding.in_feat != cfg.d_model {
            return Err("embedding shape does not match geometry");
        }
        let n = cfg.mhc_lanes;
        let l = cfg.num_layers;
        if mhc.a_pre.len() != l || mhc.a_post.len() != l || mhc.a_res.len() != l {
            return Err("mhc a_* length does not match num_layers");
        }
        if mhc.b_pre.len() != l * n || mhc.b_post.len() != l * n || mhc.b_res.len() != l * n * n {
            return Err("mhc b_* length does not match num_layers x lanes");
        }
        if mhc.phi_pre.out_feat != l * n
            || mhc.phi_post.out_feat != l * n
            || mhc.phi_res.out_feat != l * n * n
        {
            return Err("mhc phi_* row count does not match num_layers x lanes");
        }
        if mhc.phi_pre.in_feat != cfg.mhc_width() {
            return Err("mhc phi_* width does not match lanes x d_model");
        }
        for layer in &layers {
            if layer.d1.len() != cfg.hada_n
                || layer.d2.len() != cfg.hada_n
                || layer.d3.len() != cfg.hada_n
            {
                return Err("HadamardMLP diagonal width does not match hada_n");
            }
            if layer.q_norm.len() != cfg.head_dim || layer.k_norm.len() != cfg.head_dim {
                return Err("q_norm/k_norm width does not match head_dim");
            }
        }
        for e in &engrams {
            if e.taps.len() != cfg.engram.conv_taps * cfg.d_model {
                return Err("engram taps shape does not match conv_taps x d_model");
            }
            if e.tables.in_feat != cfg.engram.sub_dim {
                return Err("engram table width does not match sub_dim");
            }
        }

        let rope = RopeCache::new(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta);
        let embed_scale = math::sqrt(cfg.d_model as f32);
        Ok(Self {
            cfg,
            embedding,
            layers,
            mhc,
            engrams,
            final_norm,
            rope,
            embed_scale,
        })
    }

    /// Where this port follows the decode reference rather than the training graph.
    ///
    /// `architecture.Block` gates the Engram contribution on the lane-stacked `x`
    /// (`einsum "btd,sbtd->sbt"`, scaled by `site_flags`).
    /// `decode._forward_cached` gates it on `u`, the lane-mixed vector the block
    /// actually consumes, with an explicit per-layer branch. This port follows
    /// decode.py: it is the incremental path, and the one the parity fixtures come
    /// from. Do not "fix" this toward architecture.py.
    pub const ENGRAM_GATE_SOURCE: &'static str = "decode.py (_forward_cached), gate on u";

    /// Allocate per-sequence state sized for the model's own attention window.
    ///
    /// This is what generation wants. The KV cache is a ring of `kv_window`
    /// positions rather than `max_seq_len`, which is the difference between
    /// 14 MB and 113 MB at the shipped geometry.
    ///
    /// Use [`make_state_full_causal`] for the probe heads, which attend over the
    /// whole sequence and therefore need the whole cache.
    ///
    /// [`make_state_full_causal`]: V2Model::make_state_full_causal
    pub fn make_state(&self) -> V2State {
        let cache_len = if self.cfg.kv_window == 0 {
            self.cfg.max_seq_len
        } else {
            self.cfg.kv_window.min(self.cfg.max_seq_len)
        };
        self.make_state_with_cache(cache_len, self.cfg.kv_window)
    }

    /// Allocate per-sequence state for full causal attention.
    ///
    /// Sizes the cache to `max_seq_len` and sets the window to 0. Needed by
    /// `architecture.encode_contrastive` and `forward_confidence`, which default
    /// to full causal even though the LM path is windowed.
    pub fn make_state_full_causal(&self) -> V2State {
        self.make_state_with_cache(self.cfg.max_seq_len, 0)
    }

    fn make_state_with_cache(&self, cache_len: usize, kv_window: usize) -> V2State {
        let cfg = &self.cfg;
        let (d, n) = (cfg.d_model, cfg.mhc_lanes);
        let sites = cfg.engram.sites.len();
        let ring = (cfg.engram.window() + 1).max(1);
        let fetched = cfg.engram.fetched_dim().max(1);

        let engram_prep = self
            .engrams
            .first()
            .map_or(fetched, |e| e.value_proj.prepared_len());

        V2State {
            k_cache: (0..cfg.num_layers)
                .map(|_| vec![0.0f32; cfg.num_kv_heads * cache_len * cfg.head_dim])
                .collect(),
            v_cache: (0..cfg.num_layers)
                .map(|_| vec![0.0f32; cfg.num_kv_heads * cache_len * cfg.head_dim])
                .collect(),
            history: vec![0u32; cfg.max_seq_len],
            pos: 0,
            kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            cache_len,
            kv_window,
            engram_v: vec![0.0; sites * ring * d],
            ring,
            lanes: vec![0.0; n * d],
            lanes_prev: vec![0.0; n * d],
            nx: vec![0.0; n * d],
            nx_prep: vec![0.0; self.mhc.phi_pre.prepared_len()],
            u: vec![0.0; d],
            bx: vec![0.0; d],
            y: vec![0.0; d],
            h: vec![0.0; d],
            h_prep: vec![0.0; self.layers[0].q_proj.prepared_len()],
            hpre: vec![0.0; n],
            hpost: vec![0.0; n],
            hres: vec![0.0; n * n],
            q: vec![0.0; cfg.attn_dim()],
            k: vec![0.0; cfg.kv_dim()],
            v: vec![0.0; cfg.kv_dim()],
            gate: vec![0.0; cfg.attn_dim()],
            attn_out: vec![0.0; cfg.attn_dim()],
            attn_prep: vec![0.0; self.layers[0].out_proj.prepared_len()],
            attn_result: vec![0.0; d],
            scores: vec![0.0; cfg.max_seq_len],
            hada: vec![0.0; cfg.hada_n],
            mlp_out: vec![0.0; d],
            e_buf: vec![0.0; fetched],
            e_prep: vec![0.0; engram_prep],
            engram_k: vec![0.0; sites * d],
            engram_val: vec![0.0; sites * d],
            alpha: vec![0.0; sites.max(1)],
            lm: vec![0.0; d],
            lm_prep: vec![0.0; self.embedding.prepared_len()],
            tmp_a: vec![0.0; d],
            tmp_b: vec![0.0; d],
        }
    }

    /// Process one token at the next position and write logits over the vocabulary.
    ///
    /// Feed a prompt by calling this once per token in order; the logits from the
    /// last call are the ones to sample from.
    pub fn step(
        &self,
        token: u32,
        state: &mut V2State,
        logits: &mut [f32],
    ) -> Result<(), &'static str> {
        self.forward(token, state, Some(logits), &mut |_, _, _| {})
    }

    /// Process one token without evaluating the LM head.
    ///
    /// Prefill needs logits only at the final prompt position, and the tied LM
    /// head is the single widest matmul in a step (`vocab_size x d_model` — 8192
    /// rows against the 512 of any projection). Skipping it on the preceding
    /// tokens removes that cost from every one of them.
    ///
    /// [`last_hidden`] is still valid afterwards.
    ///
    /// [`last_hidden`]: V2Model::last_hidden
    pub fn step_prefill(&self, token: u32, state: &mut V2State) -> Result<(), &'static str> {
        self.forward(token, state, None, &mut |_, _, _| {})
    }

    /// [`step`], with a callback at each of the capture points
    /// `tools/gen_v2_forward_parity.py` records — the ladder used to locate a
    /// divergence instead of only detecting one.
    ///
    /// `trace(name, layer, values)`; `layer` is `usize::MAX` for whole-forward
    /// points. Generic rather than a trait object, so the no-op closure [`step`]
    /// passes inlines away entirely.
    ///
    /// [`step`]: V2Model::step
    pub fn step_traced<F: FnMut(&str, usize, &[f32])>(
        &self,
        token: u32,
        state: &mut V2State,
        logits: &mut [f32],
        trace: &mut F,
    ) -> Result<(), &'static str> {
        self.forward(token, state, Some(logits), trace)
    }

    fn forward<F: FnMut(&str, usize, &[f32])>(
        &self,
        token: u32,
        state: &mut V2State,
        logits: Option<&mut [f32]>,
        trace: &mut F,
    ) -> Result<(), &'static str> {
        const WHOLE: usize = usize::MAX;
        let cfg = &self.cfg;
        if state.pos >= cfg.max_seq_len {
            return Err("sequence exceeds max_seq_len");
        }
        if token as usize >= cfg.vocab_size {
            return Err("token id outside vocabulary");
        }
        if logits.as_ref().is_some_and(|l| l.len() != cfg.vocab_size) {
            return Err("logits buffer is not vocab_size long");
        }

        let pos = state.pos;
        let (d, n) = (cfg.d_model, cfg.mhc_lanes);
        state.history[pos] = token;

        // Lane stream starts as the scaled embedding, broadcast across lanes.
        self.embedding
            .dequantize_row(token as usize, &mut state.tmp_a);
        for v in state.tmp_a.iter_mut() {
            *v *= self.embed_scale;
        }
        trace("embed", WHOLE, &state.tmp_a);
        for lane in 0..n {
            state.lanes[lane * d..(lane + 1) * d].copy_from_slice(&state.tmp_a);
        }

        // `_engram_kv` runs once per forward, before the layer loop.
        self.engram_fill(state, pos);
        if !cfg.engram.sites.is_empty() {
            trace("engram_k0", WHOLE, &state.engram_k[..d]);
            trace("engram_v0", WHOLE, &state.engram_val[..d]);
        }

        for li in 0..cfg.num_layers {
            self.mhc_pre(state, li);
            trace("hpre", li, &state.hpre);
            trace("u", li, &state.u);
            self.engram_gate(state, li);
            trace("bx", li, &state.bx);
            if let Some(site) = cfg.engram.site_of_layer(li) {
                trace("alpha", li, &state.alpha[site..site + 1]);
            }
            self.block(state, li, pos);
            trace("y", li, &state.y);
            self.mhc_post(state, li);
            trace("hpost", li, &state.hpost);
            trace("hres", li, &state.hres);
            trace("x", li, &state.lanes);
        }

        // Mean over lanes, final norm, tied LM head.
        for c in 0..d {
            let mut acc = 0.0f32;
            for lane in 0..n {
                acc += state.lanes[lane * d + c];
            }
            state.lm[c] = acc / n as f32;
        }
        trace("lane_mean", WHOLE, &state.lm);
        zc_rms_norm_vec(&mut state.lm, &self.final_norm);
        trace("final_norm", WHOLE, &state.lm);
        if let Some(logits) = logits {
            self.embedding.prepare_input(&state.lm, &mut state.lm_prep);
            self.embedding.matvec_prepared(&state.lm_prep, logits);
            trace("logits", WHOLE, logits);
        }

        state.pos += 1;
        Ok(())
    }

    /// The hidden state the probe heads pool over, for the position just stepped.
    ///
    /// Valid only immediately after a successful [`step`]; it is the post-final-norm
    /// vector the LM head consumed.
    ///
    /// [`step`]: V2Model::step
    pub fn last_hidden(state: &V2State) -> &[f32] {
        &state.lm
    }

    /// Apply the shared RoPE cache. Exposed to the batched path so both use one
    /// table.
    pub(crate) fn rope_apply(
        &self,
        x: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        offset: usize,
    ) {
        self.rope.apply(x, seq_len, num_heads, head_dim, offset);
    }

    /// mHC pre-routing: `hpre` and the lane mix that yields `u`.
    fn mhc_pre(&self, state: &mut V2State, li: usize) {
        let cfg = &self.cfg;
        let (d, n) = (cfg.d_model, cfg.mhc_lanes);

        rms_unit_to(&state.lanes, &mut state.nx);
        // phi_pre, phi_post and phi_res all read nx with identical geometry, so
        // the Hadamard rotation is paid once here and reused by mhc_post.
        self.mhc
            .phi_pre
            .prepare_input(&state.nx, &mut state.nx_prep);

        self.mhc
            .phi_pre
            .matvec_rows_prepared(&state.nx_prep, li * n, &mut state.hpre);
        let a = self.mhc.a_pre[li];
        for lane in 0..n {
            let z = a * state.hpre[lane] + self.mhc.b_pre[li * n + lane] + cfg.pre_off(li, lane);
            state.hpre[lane] = sigmoid(z);
        }

        for c in 0..d {
            let mut acc = 0.0f32;
            for lane in 0..n {
                acc += state.hpre[lane] * state.lanes[lane * d + c];
            }
            state.u[c] = acc;
        }
        state.bx.copy_from_slice(&state.u);
    }

    /// `bx = u + sigmoid(dot(rms_unit(u), rms_unit(ek)) / sqrt(d_model)) * ev`.
    fn engram_gate(&self, state: &mut V2State, li: usize) {
        let Some(site) = self.cfg.engram.site_of_layer(li) else {
            return;
        };
        let d = self.cfg.d_model;
        let base = site * d;

        rms_unit_to(&state.u, &mut state.tmp_a);
        state.tmp_b.copy_from_slice(&state.engram_k[base..base + d]);
        rms_unit(&mut state.tmp_b);
        let mut dot = 0.0f32;
        for c in 0..d {
            dot += state.tmp_a[c] * state.tmp_b[c];
        }
        let alpha = sigmoid(dot / math::sqrt(d as f32));
        state.alpha[site] = alpha;
        for c in 0..d {
            state.bx[c] = state.u[c] + alpha * state.engram_val[base + c];
        }
    }

    /// `_block_cached`: gated attention residual, then HadamardMLP residual.
    /// Leaves `block(bx) - u` in `state.y`, as the reference does.
    fn block(&self, state: &mut V2State, li: usize, pos: usize) {
        let layer = &self.layers[li];
        let d = self.cfg.d_model;

        state.h.copy_from_slice(&state.bx);
        zc_rms_norm_vec(&mut state.h, &layer.norm_in);
        self.attention(state, li, pos);
        zc_rms_norm_vec(&mut state.attn_result, &layer.post_norm);
        let g = sigmoid(layer.attn_gate);
        for c in 0..d {
            state.y[c] = state.bx[c] + g * state.attn_result[c];
        }

        state.h.copy_from_slice(&state.y);
        zc_rms_norm_vec(&mut state.h, &layer.pre_hada);
        hadamard_mlp(
            &state.h,
            &layer.d1,
            &layer.d2,
            &layer.d3,
            &mut state.hada,
            &mut state.mlp_out,
        );
        for c in 0..d {
            state.y[c] += state.mlp_out[c];
            state.y[c] -= state.u[c];
        }
    }

    /// `_attn_cached` for a single query position. Result lands in `attn_result`.
    fn attention(&self, state: &mut V2State, li: usize, pos: usize) {
        let layer = &self.layers[li];
        let cfg = &self.cfg;
        let (h_n, kv_n, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let reps = cfg.kv_repeat();

        // All four projections read the same normed activation.
        layer.q_proj.prepare_input(&state.h, &mut state.h_prep);
        layer.q_proj.matvec_prepared(&state.h_prep, &mut state.q);
        layer.k_proj.matvec_prepared(&state.h_prep, &mut state.k);
        layer.v_proj.matvec_prepared(&state.h_prep, &mut state.v);
        layer
            .gate_proj
            .matvec_prepared(&state.h_prep, &mut state.gate);

        for head in 0..h_n {
            zc_rms_norm_vec(&mut state.q[head * hd..(head + 1) * hd], &layer.q_norm);
        }
        for head in 0..kv_n {
            zc_rms_norm_vec(&mut state.k[head * hd..(head + 1) * hd], &layer.k_norm);
        }
        self.rope.apply(&mut state.q, 1, h_n, hd, pos);
        self.rope.apply(&mut state.k, 1, kv_n, hd, pos);

        // Cache layout: [kv_head][slot][head_dim], slot = position % cache_len.
        let k_src = core::mem::take(&mut state.k);
        let v_src = core::mem::take(&mut state.v);
        state.write_kv(li, pos, &k_src, &v_src);
        state.k = k_src;
        state.v = v_src;

        // Valid keys: `l <= pos` and, when windowed, `pos - l < kv_window`.
        // The reference also ORs a sink mask, but `_doc_prefix_len` returns 0 on
        // every shipped path, leaving it empty.
        let lo = state.window_lo(pos);
        let span = pos - lo + 1;
        let scale = 1.0 / math::sqrt(hd as f32);

        // Ring geometry pulled out so the loops below can index `k_cache` and
        // `attn_out` as separate fields: an accessor method would borrow all of
        // `state` and collide with the accumulation.
        let (cache_len, ring_hd) = (state.cache_len, state.head_dim);
        let slot = |kv: usize, p: usize| (kv * cache_len + p % cache_len) * ring_hd;

        for head in 0..h_n {
            let kv = head / reps;

            for s in 0..span {
                let off = slot(kv, lo + s);
                let krow = &state.k_cache[li][off..off + hd];
                let qrow = &state.q[head * hd..(head + 1) * hd];
                let mut acc = 0.0f32;
                for i in 0..hd {
                    acc += qrow[i] * krow[i];
                }
                state.scores[s] = acc * scale;
            }
            softmax_inplace(&mut state.scores[..span]);

            let obase = head * hd;
            state.attn_out[obase..obase + hd].fill(0.0);
            for s in 0..span {
                let w = state.scores[s];
                let off = slot(kv, lo + s);
                for i in 0..hd {
                    state.attn_out[obase + i] += w * state.v_cache[li][off + i];
                }
            }
        }

        for (o, &g) in state.attn_out.iter_mut().zip(state.gate.iter()) {
            *o *= sigmoid(g);
        }
        layer
            .out_proj
            .prepare_input(&state.attn_out, &mut state.attn_prep);
        layer
            .out_proj
            .matvec_prepared(&state.attn_prep, &mut state.attn_result);
    }

    /// mHC post-routing: `hpost`, the Sinkhorn lane mix `hres`, and the update
    /// `x[i] = sum_j hres[i][j] * x[j] + hpost[i] * y`.
    fn mhc_post(&self, state: &mut V2State, li: usize) {
        let cfg = &self.cfg;
        let (d, n) = (cfg.d_model, cfg.mhc_lanes);

        self.mhc
            .phi_post
            .matvec_rows_prepared(&state.nx_prep, li * n, &mut state.hpost);
        let a_post = self.mhc.a_post[li];
        for lane in 0..n {
            let z = a_post * state.hpost[lane]
                + self.mhc.b_post[li * n + lane]
                + cfg.post_off(li, lane);
            state.hpost[lane] = 2.0 * sigmoid(z);
        }

        self.mhc
            .phi_res
            .matvec_rows_prepared(&state.nx_prep, li * n * n, &mut state.hres);
        let a_res = self.mhc.a_res[li];
        for k in 0..n * n {
            state.hres[k] = a_res * state.hres[k] + self.mhc.b_res[li * n * n + k];
        }
        sinkhorn(&mut state.hres, n);

        // Every output lane reads every input lane, so snapshot first.
        state.lanes_prev.copy_from_slice(&state.lanes);
        for i in 0..n {
            let hp = state.hpost[i];
            for c in 0..d {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += state.hres[i * n + j] * state.lanes_prev[j * d + c];
                }
                state.lanes[i * d + c] = acc + hp * state.y[c];
            }
        }
    }

    /// `_engram_kv` for one position: fills `engram_k` and `engram_val`.
    ///
    /// Shared with the batched prefill path, which drives it position by position
    /// and copies each result out — the Engram ring is keyed on absolute position,
    /// so in-order feeding is all it requires.
    pub(crate) fn engram_fill(&self, state: &mut V2State, pos: usize) {
        let cfg = &self.cfg;
        if cfg.engram.sites.is_empty() {
            return;
        }
        let d = cfg.d_model;
        let eg = &cfg.engram;
        let heads = eg.heads();
        let dil = eg.conv_dilation;
        let ring = state.ring;

        for site in 0..self.engrams.len() {
            self.engram_fetch(state, pos, site, heads);
            let weights = &self.engrams[site];
            weights
                .value_proj
                .prepare_input(&state.e_buf, &mut state.e_prep);

            // Un-convolved value at this position, into the ring.
            let slot = (site * ring + pos % ring) * d;
            weights
                .value_proj
                .matvec_prepared(&state.e_prep, &mut state.engram_v[slot..slot + d]);

            let kbase = site * d;
            weights
                .key_proj
                .matvec_prepared(&state.e_prep, &mut state.engram_k[kbase..kbase + d]);

            // v_conv = sum_j taps[j] * v(pos - j*dilation), skipping taps whose
            // source position falls before the sequence start.
            let vbase = site * d;
            state.engram_val[vbase..vbase + d].fill(0.0);
            for j in 0..eg.conv_taps {
                let shift = j * dil;
                if shift > pos {
                    continue;
                }
                let src = (site * ring + (pos - shift) % ring) * d;
                let tap = &weights.taps[j * d..(j + 1) * d];
                let v = &state.engram_v[src..src + d];
                for (dst, (&t, &vc)) in state.engram_val[vbase..vbase + d]
                    .iter_mut()
                    .zip(tap.iter().zip(v.iter()))
                {
                    *dst += t * vc;
                }
            }
        }
    }

    /// Build the flattened fetched-entry vector for one position and site.
    fn engram_fetch(&self, state: &mut V2State, pos: usize, site: usize, heads: usize) {
        let (hist, e_buf) = (&state.history, &mut state.e_buf);
        self.engram_fetch_into(pos, site, heads, hist, e_buf);
    }

    /// As [`engram_fetch`], writing into caller-provided storage so the batched
    /// prefill can hold one vector per position.
    ///
    /// [`engram_fetch`]: V2Model::engram_fetch
    pub(crate) fn engram_fetch_into(
        &self,
        pos: usize,
        site: usize,
        heads: usize,
        history: &[u32],
        out: &mut [f32],
    ) {
        let eg = &self.cfg.engram;
        let tables = &self.engrams[site].tables;
        let sub = eg.sub_dim;

        for table in 0..eg.num_tables {
            let order = eg.orders[engram_table_order(table, heads)];
            let idx = engram_index(table, order, eg.slots, |j| {
                if j > pos {
                    0
                } else {
                    history[pos - j]
                }
            });
            let dst = &mut out[table * sub..(table + 1) * sub];
            if order > 0 && order - 1 > pos {
                // `ngram_ok`: the oldest token the hash needs is outside the
                // window, so the entry is zeroed rather than fetched.
                dst.fill(0.0);
            } else {
                tables.dequantize_row(table * eg.slots + idx, dst);
            }
        }
    }
}

impl V2State {
    /// Absolute position of the next token.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Tokens fed so far, oldest first.
    pub fn history(&self) -> &[u32] {
        &self.history[..self.pos]
    }

    /// Lowest key position attention may read for a query at `pos`.
    ///
    /// `l <= pos` always; with a window, also `pos - l < kv_window`.
    #[inline]
    pub(crate) fn window_lo(&self, pos: usize) -> usize {
        if self.kv_window == 0 {
            0
        } else {
            pos + 1 - self.kv_window.min(pos + 1)
        }
    }

    /// The attention window this sequence runs at.
    pub fn kv_window(&self) -> usize {
        self.kv_window
    }

    /// Override the attention window. `None` restores the model's own.
    ///
    /// Needed because the two heads upstream ships disagree: the LM path runs the
    /// checkpoint's `kv_window` (256), while `architecture.encode_contrastive` and
    /// `forward_confidence` default to `window=0`, full causal. Below 256 tokens
    /// the two are bit-identical; above it they diverge — at 642 tokens the
    /// contrastive embeddings differ by cosine 3.5e-4 and the confidence logit by
    /// 0.096 — so this is a real fork, not a rounding detail.
    ///
    /// Takes effect from the next step; it does not rewrite the existing cache.
    ///
    /// Fails when the requested window needs more cached positions than this
    /// state allocated — a window of 0 over a ring-sized cache would read stale
    /// slots and quietly return wrong embeddings. Use
    /// [`V2Model::make_state_full_causal`] for that case.
    pub fn set_kv_window(
        &mut self,
        window: Option<usize>,
        model: &V2Model,
    ) -> Result<(), &'static str> {
        let w = window.unwrap_or(model.cfg.kv_window);
        let needed = if w == 0 { model.cfg.max_seq_len } else { w };
        if needed > self.cache_len {
            return Err(
                "requested attention window exceeds this state's KV capacity; \
                        allocate with make_state_full_causal",
            );
        }
        self.kv_window = w;
        Ok(())
    }

    /// Record a token at an absolute position without advancing.
    pub(crate) fn set_history(&mut self, pos: usize, token: u32) {
        self.history[pos] = token;
    }

    /// Advance the position counter after a batch of positions was processed.
    pub(crate) fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    /// Byte offset of one `(kv_head, position)` slot in a layer's ring.
    #[inline]
    fn slot(&self, kv_head: usize, pos: usize) -> usize {
        (kv_head * self.cache_len + pos % self.cache_len) * self.head_dim
    }

    /// Write one position's K and V into a layer's cache.
    pub(crate) fn write_kv(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let hd = self.head_dim;
        for head in 0..self.kv_heads {
            let off = self.slot(head, pos);
            self.k_cache[layer][off..off + hd].copy_from_slice(&k[head * hd..(head + 1) * hd]);
            self.v_cache[layer][off..off + hd].copy_from_slice(&v[head * hd..(head + 1) * hd]);
        }
    }

    /// The cached K for one `(kv_head, position)`. `pos` must be inside the
    /// window ending at the current position, which `window_lo` guarantees.
    #[inline]
    pub(crate) fn k_at(&self, layer: usize, kv_head: usize, pos: usize) -> &[f32] {
        let off = self.slot(kv_head, pos);
        &self.k_cache[layer][off..off + self.head_dim]
    }

    #[inline]
    pub(crate) fn v_at(&self, layer: usize, kv_head: usize, pos: usize) -> &[f32] {
        let off = self.slot(kv_head, pos);
        &self.v_cache[layer][off..off + self.head_dim]
    }

    /// Positions the Engram value ring holds.
    pub(crate) fn ring(&self) -> usize {
        self.ring
    }

    pub(crate) fn history_slice(&self) -> &[u32] {
        &self.history
    }

    pub(crate) fn engram_v_slot(&mut self, slot: usize, d: usize) -> &mut [f32] {
        &mut self.engram_v[slot..slot + d]
    }

    pub(crate) fn engram_v_row(&self, slot: usize, d: usize) -> &[f32] {
        &self.engram_v[slot..slot + d]
    }

    /// Positions the KV ring holds.
    pub fn cache_len(&self) -> usize {
        self.cache_len
    }

    /// Bytes the KV cache occupies — the dominant part of a session.
    pub fn kv_bytes(&self) -> usize {
        self.k_cache
            .iter()
            .chain(self.v_cache.iter())
            .map(|c| c.len() * 4)
            .sum()
    }

    /// Start a new sequence, keeping the allocations.
    pub fn reset(&mut self) {
        self.pos = 0;
        for c in self.k_cache.iter_mut().chain(self.v_cache.iter_mut()) {
            c.fill(0.0);
        }
        self.engram_v.fill(0.0);
        self.history.fill(0);
    }
}
