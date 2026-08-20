//! Geometry for the Needle v2 architecture.
//!
//! Every field comes from the `.cact` header, so one build runs any
//! configuration of the architecture. `needle-infer` converts its
//! `CactGeometry` into this; `needle-core` stays `no_std` and container-agnostic.

use alloc::vec::Vec;

/// Engram (n-gram associative memory) geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngramGeometry {
    /// Rows per hash table.
    pub slots: usize,
    /// Width of one table's entry.
    pub sub_dim: usize,
    /// Number of hash tables — `orders.len() * heads`.
    pub num_tables: usize,
    /// Taps in the causal convolution over the value stream.
    pub conv_taps: usize,
    /// Stride between taps, `max(orders)`.
    pub conv_dilation: usize,
    /// n-gram orders, one group of `num_tables / orders.len()` tables each.
    pub orders: Vec<usize>,
    /// Layer indices carrying a site.
    pub sites: Vec<usize>,
}

impl EngramGeometry {
    /// Tables per order.
    pub fn heads(&self) -> usize {
        if self.orders.is_empty() {
            0
        } else {
            self.num_tables / self.orders.len()
        }
    }

    /// Flattened width of one position's fetched entries.
    pub fn fetched_dim(&self) -> usize {
        self.num_tables * self.sub_dim
    }

    /// Tokens of history the value convolution reaches back over.
    /// Mirrors `decode._engram_window`.
    pub fn window(&self) -> usize {
        if self.sites.is_empty() {
            0
        } else {
            self.conv_taps * self.conv_dilation
        }
    }

    /// Site index for a layer, if it carries one.
    pub fn site_of_layer(&self, layer: usize) -> Option<usize> {
        self.sites.iter().position(|&l| l == layer)
    }
}

/// Full v2 model geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct V2Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// Hadamard width in HadamardMLP: `d_model` rounded up to a power of two.
    pub hada_n: usize,
    pub mhc_lanes: usize,
    pub rope_theta: f32,
    /// Sliding attention window. `0` disables it (full causal).
    pub kv_window: usize,
    pub engram: EngramGeometry,
}

impl V2Config {
    /// Attention width, `num_heads * head_dim`.
    pub fn attn_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Packed K/V projection width.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Query heads served by one KV head.
    pub fn kv_repeat(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Flattened mHC lane-stream width, `mhc_lanes * d_model`.
    pub fn mhc_width(&self) -> usize {
        self.mhc_lanes * self.d_model
    }

    /// The lane layer `i` is "assigned" to — `eye(n)[i % n]` in
    /// `architecture.Stack`, which drives the `pre_off` / `post_off` biases.
    pub fn active_lane(&self, layer: usize) -> usize {
        layer % self.mhc_lanes
    }

    /// `pre_off[layer][lane]` — `8 * lane_onehot - 4`.
    pub fn pre_off(&self, layer: usize, lane: usize) -> f32 {
        if lane == self.active_lane(layer) {
            4.0
        } else {
            -4.0
        }
    }

    /// `post_off[layer][lane]` — `-4 * (1 - lane_onehot)`.
    pub fn post_off(&self, layer: usize, lane: usize) -> f32 {
        if lane == self.active_lane(layer) {
            0.0
        } else {
            -4.0
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_heads == 0 || self.num_kv_heads == 0 {
            return Err("head counts must be non-zero");
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err("num_heads must be divisible by num_kv_heads");
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err("head_dim must be even (RoPE rotates half-splits)");
        }
        if self.d_model == 0 || self.mhc_lanes == 0 || self.num_layers == 0 {
            return Err("d_model, mhc_lanes and num_layers must be non-zero");
        }
        if !self.hada_n.is_power_of_two() || self.hada_n < self.d_model {
            return Err("hada_n must be a power of two at or above d_model");
        }
        if self.vocab_size == 0 || self.max_seq_len == 0 {
            return Err("vocab_size and max_seq_len must be non-zero");
        }
        if !self.engram.sites.is_empty() {
            let e = &self.engram;
            if e.orders.is_empty() || e.num_tables == 0 || e.sub_dim == 0 || e.slots == 0 {
                return Err("engram geometry incomplete");
            }
            if !e.num_tables.is_multiple_of(e.orders.len()) {
                return Err("num_tables must be divisible by the order count");
            }
            if e.sites.iter().any(|&l| l >= self.num_layers) {
                return Err("engram site beyond num_layers");
            }
            if e.fetched_dim() != self.d_model {
                // Not required by the maths, but every shipped configuration has
                // it, and the projections' input width is derived from it.
                return Err("engram fetched_dim must equal d_model");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn cfg() -> V2Config {
        V2Config {
            vocab_size: 8192,
            d_model: 512,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 27,
            head_dim: 64,
            max_seq_len: 2048,
            hada_n: 512,
            mhc_lanes: 4,
            rope_theta: 100000.0,
            kv_window: 256,
            engram: EngramGeometry {
                slots: 8192,
                sub_dim: 128,
                num_tables: 4,
                conv_taps: 4,
                conv_dilation: 3,
                orders: vec![2, 3],
                sites: vec![2, 15],
            },
        }
    }

    #[test]
    fn shipped_geometry_validates() {
        let c = cfg();
        assert_eq!(c.validate(), Ok(()));
        assert_eq!(c.attn_dim(), 512);
        assert_eq!(c.kv_dim(), 256);
        assert_eq!(c.kv_repeat(), 2);
        assert_eq!(c.mhc_width(), 2048);
        assert_eq!(c.engram.heads(), 2);
        assert_eq!(c.engram.fetched_dim(), 512);
        assert_eq!(c.engram.window(), 12);
        assert_eq!(c.engram.site_of_layer(2), Some(0));
        assert_eq!(c.engram.site_of_layer(15), Some(1));
        assert_eq!(c.engram.site_of_layer(3), None);
    }

    /// `pre_off` / `post_off` must reproduce `8*lane - 4` and `-4*(1 - lane)`.
    #[test]
    fn mhc_lane_offsets_match_reference_formulas() {
        let c = cfg();
        for layer in 0..c.num_layers {
            for lane in 0..c.mhc_lanes {
                let onehot = f32::from(lane == layer % c.mhc_lanes);
                assert_eq!(c.pre_off(layer, lane), 8.0 * onehot - 4.0);
                assert_eq!(c.post_off(layer, lane), -4.0 * (1.0 - onehot));
            }
        }
    }

    #[test]
    fn validate_rejects_broken_geometry() {
        let mut c = cfg();
        c.num_kv_heads = 3;
        assert!(c.validate().is_err());

        let mut c = cfg();
        c.head_dim = 63;
        assert!(c.validate().is_err());

        let mut c = cfg();
        c.hada_n = 256; // below d_model
        assert!(c.validate().is_err());

        let mut c = cfg();
        c.engram.sites = vec![2, 99];
        assert!(c.validate().is_err());

        let mut c = cfg();
        c.engram.sub_dim = 64; // fetched_dim 256 != d_model
        assert!(c.validate().is_err());
    }
}
