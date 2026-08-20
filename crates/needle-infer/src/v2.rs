//! Build a `needle_core::v2::V2Model` from a `.cact` container.
//!
//! This is the only place the container format and the compute kernels meet.
//! [`crate::cact`] has already validated that every canon slot carries the shape
//! the header geometry predicts, so the work here is transposition-free: the
//! `.cact` stores matmul weights pre-transposed to `[out, in]`, which is exactly
//! the orientation [`needle_core::cq::CqWeight`] wants.

use crate::cact::{Cact, CactError, CactGeometry, CactLayout, HEAD_CONFIDENCE, HEAD_CONTRASTIVE};
use crate::sp_tokenizer::{SpTokenizer, TokenizerError};
use needle_core::v2::{
    EngramGeometry, ProbeHeadWeights, V2Config, V2Engram, V2Layer, V2Mhc, V2Model,
};
use std::fmt;
use std::path::Path;

/// A pooling probe head (`contrastive` or `confidence`).
pub type ProbeHead = ProbeHeadWeights;

/// Everything a `.cact` file yields: geometry, weights, tokenizer, probe heads.
pub struct V2Bundle {
    pub model: V2Model,
    /// Present whenever the container embeds a piece table, which every shipped
    /// checkpoint does.
    pub tokenizer: Option<SpTokenizer>,
    pub heads: Vec<ProbeHead>,
}

#[derive(Debug)]
pub enum V2LoadError {
    Cact(CactError),
    Tokenizer(TokenizerError),
    /// A `needle_core` invariant rejected the assembled model.
    Model(&'static str),
}

impl fmt::Display for V2LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cact(e) => write!(f, "{e}"),
            Self::Tokenizer(e) => write!(f, "{e}"),
            Self::Model(m) => write!(f, "model rejected: {m}"),
        }
    }
}

impl std::error::Error for V2LoadError {}

impl From<CactError> for V2LoadError {
    fn from(e: CactError) -> Self {
        Self::Cact(e)
    }
}

impl From<TokenizerError> for V2LoadError {
    fn from(e: TokenizerError) -> Self {
        Self::Tokenizer(e)
    }
}

/// Translate the container's geometry into the compute crate's.
pub fn config_from_geometry(g: &CactGeometry) -> V2Config {
    V2Config {
        vocab_size: g.vocab_size,
        d_model: g.d_model,
        num_heads: g.num_heads,
        num_kv_heads: g.num_kv_heads,
        num_layers: g.num_layers,
        head_dim: g.head_dim,
        max_seq_len: g.max_seq_len,
        hada_n: g.hada_n,
        mhc_lanes: g.mhc_lanes,
        rope_theta: g.rope_theta,
        kv_window: g.kv_window,
        engram: EngramGeometry {
            slots: g.engram_slots,
            sub_dim: g.engram_sub_dim,
            num_tables: g.num_engram_tables,
            conv_taps: g.engram_conv_taps,
            conv_dilation: g.engram_conv_dilation,
            orders: g.engram_orders.clone(),
            sites: g.engram_sites.clone(),
        },
    }
}

impl V2Bundle {
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let cact = Cact::load(path)?;
        Self::from_cact(&cact)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, V2LoadError> {
        let cact = Cact::from_bytes(bytes)?;
        Self::from_cact(&cact)
    }

    pub fn from_cact(cact: &Cact) -> Result<Self, V2LoadError> {
        let layout: CactLayout = cact.layout()?;
        let cfg = config_from_geometry(&cact.geom);
        let g = &cact.geom;

        let embedding = cact.cq(layout.embedding)?;

        let mut layers = Vec::with_capacity(g.num_layers);
        for li in &layout.layers {
            layers.push(V2Layer {
                norm_in: cact.floats(li.norm_in)?,
                q_proj: cact.cq(li.q_proj)?,
                k_proj: cact.cq(li.k_proj)?,
                v_proj: cact.cq(li.v_proj)?,
                gate_proj: cact.cq(li.gate_proj)?,
                out_proj: cact.cq(li.out_proj)?,
                q_norm: cact.floats(li.q_norm)?,
                k_norm: cact.floats(li.k_norm)?,
                post_norm: cact.floats(li.post_norm)?,
                // Stored as a length-1 tensor; it is the pre-sigmoid scalar.
                attn_gate: cact.floats(li.attn_gate)?[0],
                pre_hada: cact.floats(li.pre_hada)?,
                d1: cact.floats(li.d1)?,
                d2: cact.floats(li.d2)?,
                d3: cact.floats(li.d3)?,
            });
        }

        let m = layout.mhc;
        let mhc = V2Mhc {
            a_pre: cact.floats(m.a_pre)?,
            a_post: cact.floats(m.a_post)?,
            a_res: cact.floats(m.a_res)?,
            b_pre: cact.floats(m.b_pre)?,
            b_post: cact.floats(m.b_post)?,
            b_res: cact.floats(m.b_res)?,
            phi_pre: cact.cq(m.phi_pre)?,
            phi_post: cact.cq(m.phi_post)?,
            phi_res: cact.cq(m.phi_res)?,
        };

        let mut engrams = Vec::with_capacity(layout.engrams.len());
        for e in &layout.engrams {
            engrams.push(V2Engram {
                tables: cact.cq(e.tables)?,
                key_proj: cact.cq(e.key_proj)?,
                value_proj: cact.cq(e.value_proj)?,
                taps: cact.floats(e.taps)?,
            });
        }

        let final_norm = cact.floats(layout.final_norm)?;

        let mut heads = Vec::with_capacity(layout.heads.len());
        for h in &layout.heads {
            let probes = cact.floats(h.probes)?;
            let proj = cact.floats(h.proj)?;
            let bias = cact.floats(h.bias)?;
            let num_probes = cact.record(h.probes).shape[0];
            let out_dim = cact.record(h.proj).shape[0];
            let head = ProbeHead {
                code: h.code,
                probes,
                proj,
                bias,
                num_probes,
                out_dim,
                d_model: g.d_model,
            };
            head.validate().map_err(V2LoadError::Model)?;
            heads.push(head);
        }

        let tokenizer = match layout.tokenizer {
            Some(idx) => Some(SpTokenizer::from_blob(cact.raw_tensor(idx)?)?),
            None => None,
        };

        let model = V2Model::new(cfg, embedding, layers, mhc, engrams, final_norm)
            .map_err(V2LoadError::Model)?;

        Ok(Self { model, tokenizer, heads })
    }

    pub fn head(&self, code: u8) -> Option<&ProbeHead> {
        self.heads.iter().find(|h| h.code == code)
    }

    pub fn contrastive_head(&self) -> Option<&ProbeHead> {
        self.head(HEAD_CONTRASTIVE)
    }

    pub fn confidence_head(&self) -> Option<&ProbeHead> {
        self.head(HEAD_CONFIDENCE)
    }
}
