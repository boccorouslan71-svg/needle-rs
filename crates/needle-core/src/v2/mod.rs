//! Needle v2: decoder-only architecture with mHC lanes, Engram memory,
//! HadamardMLP blocks, and gated GQA attention.
//!
//! Ported from `needle/model/decode.py::_forward_cached`, upstream's
//! incremental-decode reference, rather than from the training graph in
//! `architecture.py`. The two agree on the mathematics but differ in how the
//! Engram gate is expressed, and only the decode form carries a KV cache — see
//! the note on [`model::V2Model::engram_gate_note`].
//!
//! Layout: `needle-core` holds the compute and stays `no_std`, taking geometry
//! as data ([`config::V2Config`]) and weights as [`crate::cq::CqWeight`].
//! `needle-infer::v2` builds those from a `.cact` container.

pub mod batch;
pub mod config;
pub mod heads;
pub mod kernels;
pub mod model;

pub use batch::{V2Batch, DEFAULT_CHUNK};
pub use config::{EngramGeometry, V2Config};
pub use heads::{ProbeHeadWeights, ProbePool, HEAD_CONFIDENCE, HEAD_CONTRASTIVE};
pub use model::{V2Engram, V2Layer, V2Mhc, V2Model, V2State};
