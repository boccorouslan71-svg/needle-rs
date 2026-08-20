//! Fast Walsh–Hadamard transform.
//!
//! Both places Needle v2 uses a Hadamard matrix build it the same way — the
//! Sylvester recursion `H_{2n} = [[H_n, H_n], [H_n, -H_n]]`, scaled by
//! `1/sqrt(n)`:
//!
//! - `quantize._cq_hadamard_np(group)` — the rotation a Cactus-Quants group is
//!   reconstructed through (`group = 128` in the shipped checkpoint)
//! - `architecture._walsh_matrix(n)` — the mixing matrix inside HadamardMLP
//!   (`n = 512`, i.e. `hada_n`)
//!
//! Applying that matrix as a dense matmul costs `n²` MACs. The butterfly below
//! costs `n log2(n)` add/sub and no multiplies. At `n = 512` that is 4608
//! against 262144.
//!
//! The matrix is symmetric, which is what lets `cq::CqWeight` move the rotation
//! off the weights and onto the activations — see the module docs there.

/// In-place unnormalised transform: computes `W · x`, where `W` is the ±1
/// Sylvester–Hadamard matrix of order `x.len()`.
///
/// Panics if `x.len()` is not a power of two.
#[inline]
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    assert!(n.is_power_of_two(), "FWHT length must be a power of two");
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h << 1;
        }
        h <<= 1;
    }
}

/// In-place normalised transform: computes `(W / sqrt(n)) · x`.
///
/// This is the matrix upstream calls `H` in both `_cq_hadamard_np` and
/// `_walsh_matrix`.
#[inline]
pub fn fwht_normalized(x: &mut [f32]) {
    let n = x.len();
    fwht(x);
    let inv = 1.0 / crate::math::sqrt(n as f32);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `hada_n` for a given model width: the next power of two at or above
/// `d_model`.
///
/// Upstream writes this as `1 << (d_model - 1).bit_length()`, which is the same
/// value as `d_model.next_power_of_two()` for every `d_model >= 1` — a power of
/// two maps to itself, anything else rounds up.
#[inline]
pub fn hada_n(d_model: usize) -> usize {
    debug_assert!(d_model > 0);
    d_model.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Dense reference: the Sylvester construction, exactly as upstream builds it.
    fn walsh_dense(n: usize) -> Vec<Vec<f32>> {
        let mut h = vec![vec![1.0f32]];
        while h.len() < n {
            let m = h.len();
            let mut next = vec![vec![0.0f32; m * 2]; m * 2];
            for r in 0..m {
                for c in 0..m {
                    let v = h[r][c];
                    next[r][c] = v;
                    next[r][c + m] = v;
                    next[r + m][c] = v;
                    next[r + m][c + m] = -v;
                }
            }
            h = next;
        }
        h
    }

    /// Bit-exact comparison against the dense matmul.
    ///
    /// The inputs are small integers, so every partial sum on both sides is an
    /// integer well inside f32's exact range — summation order cannot matter and
    /// the assertion can be exact. (With fractional inputs the butterfly is
    /// *more* accurate than the sequential reference, since it sums pairwise, so
    /// any disagreement there would be the reference's rounding, not a bug.)
    #[test]
    fn fwht_matches_dense_matmul_exactly() {
        for &n in &[1usize, 2, 4, 8, 16, 128, 512, 1024] {
            let w = walsh_dense(n);
            // Non-symmetric under index reversal, so a transposed or
            // wrongly-ordered matrix would show up.
            let x: Vec<f32> = (0..n).map(|i| ((i * 7 + i / 5) % 19) as f32 - 9.0).collect();

            let mut got = x.clone();
            fwht(&mut got);

            for r in 0..n {
                let want: f32 = (0..n).map(|c| w[r][c] * x[c]).sum();
                assert_eq!(got[r], want, "n={n} row={r}");
            }
        }
    }

    #[test]
    fn fwht_matches_dense_matmul_on_fractional_input() {
        for &n in &[8usize, 128, 512] {
            let w = walsh_dense(n);
            let x: Vec<f32> = (0..n)
                .map(|i| (i as f32 * 0.37).sin() + 0.3 * (i as f32 * 0.11).cos())
                .collect();
            let l1: f32 = x.iter().map(|v| v.abs()).sum();

            let mut got = x.clone();
            fwht(&mut got);

            for r in 0..n {
                let want: f32 = (0..n).map(|c| w[r][c] * x[c]).sum();
                // Cancellation is bounded by the L1 norm of the input, not by
                // the size of the (possibly near-zero) result.
                assert!(
                    (got[r] - want).abs() <= 1e-5 * l1,
                    "n={n} row={r}: got {} want {want} (l1={l1})",
                    got[r]
                );
            }
        }
    }

    #[test]
    fn normalized_is_involutive() {
        // H = W/sqrt(n) is symmetric and orthogonal, so H·H·x == x.
        let n = 128;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut y = x.clone();
        fwht_normalized(&mut y);
        fwht_normalized(&mut y);
        for i in 0..n {
            assert!((y[i] - x[i]).abs() < 1e-4, "i={i}: {} vs {}", y[i], x[i]);
        }
    }

    #[test]
    fn hada_n_matches_bit_length_formula() {
        // Reference: 1 << (d - 1).bit_length()
        fn py(d: usize) -> usize {
            let bit_length = usize::BITS - (d - 1).leading_zeros();
            1usize << bit_length
        }
        for d in 1..=2048usize {
            assert_eq!(hada_n(d), py(d), "d_model={d}");
        }
        assert_eq!(hada_n(512), 512);
        assert_eq!(hada_n(768), 1024);
    }
}
