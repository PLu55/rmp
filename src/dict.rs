//! The block-structured dictionary.
//!
//! A *block* fixes one envelope shape — a FOF's `(alpha, beta)` or a Gaussian's `sigma`. Its atoms
//! are indexed by frame (onset on a hop grid) and frequency bin. Nothing below this line knows which
//! kind a block holds: it works from the rendered envelope alone. Because `E(t)` does not depend on `f`, one FFT of an
//! envelope-windowed frame yields correlations against every bin at once — that is what makes the
//! pursuit tractable, and why the FFT layer is the engine's inner loop.
//!
//! # Hop is set by `alpha`, not by the window length
//!
//! This is the design's central departure from MPTK, which targets symmetric Gabor atoms where a
//! hop of `L/2` costs little. FOF atoms are causal with a sharp attack, and onset capture falls off
//! as `exp(-2*alpha*|delta|)` — a width set by `1/alpha`, independent of `L`. At `hop = L/2` the
//! capture fraction is on the order of `1e-4`, so selection would be close to arbitrary.
//!
//! Hop is therefore derived from a capture tolerance by **measuring** the block's own envelope
//! autocorrelation, which accounts for `beta`, the fade tail, and rfofs's phase-transition rounding
//! without trusting a closed form. Because the loss depends on `alpha`, tying hop to the envelope
//! is a *correctness* property: a fixed hop would bias selection toward small-`alpha` blocks and
//! make cross-block ranking unfair.

use crate::atom::Shape;
use crate::fft::{RealFftPlanner, next_fast_len};
use crate::fof::{Envelope, EnvelopeParams, FofError, ReleasePolicy};
use realfft::num_complex::Complex32;

/// Tuning for block construction.
#[derive(Clone, Copy, Debug)]
pub struct BlockConfig {
    /// Worst-case onset capture fraction the hop must guarantee.
    pub capture_tol: f64,
    /// Lowest carrier frequency the block will represent, Hz.
    pub f_min: f32,
    /// Highest carrier frequency the block will represent, Hz.
    pub f_max: f32,
    /// Bins with `rho^2` above this are ill-conditioned and disabled.
    pub rho_sq_max: f32,
    /// Fixed for the whole analysis, and shared with resynthesis.
    pub release: ReleasePolicy,
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            capture_tol: 0.95,
            f_min: 50.0,
            f_max: 10_000.0,
            rho_sq_max: 1.0 - 1e-4,
            release: ReleasePolicy::default(),
        }
    }
}

/// One envelope shape, with its precomputed Gram inverse per frequency bin.
#[derive(Clone, Debug)]
pub struct Block {
    pub env: Envelope,
    /// Transform length: `support_len` rounded up to an even 5-smooth number.
    pub fft_len: usize,
    /// Frame spacing in samples.
    pub hop: usize,
    /// Inclusive bin range this block represents.
    pub k_lo: usize,
    pub k_hi: usize,
    /// `G^-1` pre-divided by `det`, indexed by `k - k_lo`. Dead bins hold zeros, so they project to
    /// zero energy and can never win — no branch needed in the inner loop.
    inv_uu: Vec<f32>,
    inv_uv: Vec<f32>,
    inv_vv: Vec<f32>,
    /// `rho_k`, the u/v coherence. Recorded for diagnostics.
    rho: Vec<f32>,
}

impl Block {
    /// Build a block for one envelope shape.
    pub fn new(
        params: impl Into<Shape>,
        sample_rate: f32,
        planner: &mut dyn RealFftPlanner,
        cfg: &BlockConfig,
    ) -> Result<Self, FofError> {
        let env = Envelope::render(params, sample_rate)?;
        let fft_len = next_fast_len(env.support_len());
        let hop = measure_hop(&env.samples, fft_len, cfg.capture_tol);

        // Bin range. k = 0 and k = fft_len/2 are excluded unconditionally: at omega = 0 and
        // omega = pi the sine basis vector is identically zero, so G is exactly rank-1 there.
        let bin_hz = sample_rate / fft_len as f32;
        let k_lo = ((cfg.f_min / bin_hz).ceil() as usize).max(1);
        let k_hi = ((cfg.f_max / bin_hz).floor() as usize).min(fft_len / 2 - 1);
        if k_lo > k_hi {
            return Err(FofError::Invalid("empty frequency range for this block"));
        }

        // One FFT of E^2 gives the Gram for every bin.
        let mut fft = planner.plan(fft_len);
        let mut squared = vec![0.0f32; fft_len]; // E^2, zero-padded to fft_len
        for (dst, &e) in squared.iter_mut().zip(&env.samples) {
            *dst = e * e;
        }
        let mut spectrum = vec![Complex32::new(0.0, 0.0); fft.complex_len()];
        fft.forward(&mut squared, &mut spectrum);

        let p = env.energy; // == Y[0].re, but summed in f64
        let n = k_hi - k_lo + 1;
        let (mut inv_uu, mut inv_uv, mut inv_vv, mut rho) =
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);

        for k in k_lo..=k_hi {
            let i = k - k_lo;
            let (c, s) = fold_e2_bin(&spectrum, fft_len, 2 * k);

            let guu = (p - c) / 2.0;
            let gvv = (p + c) / 2.0;
            let guv = s / 2.0;
            let det = guu * gvv - guv * guv;

            let rho_sq = ((c * c + s * s) / (p * p)).clamp(0.0, 1.0);
            rho[i] = (rho_sq as f32).sqrt();

            // Leave dead bins as zeros.
            if rho_sq <= cfg.rho_sq_max as f64 && det > 0.0 && guu > 0.0 {
                inv_uu[i] = (gvv / det) as f32;
                inv_uv[i] = (-guv / det) as f32;
                inv_vv[i] = (guu / det) as f32;
            }
        }

        Ok(Self {
            env,
            fft_len,
            hop,
            k_lo,
            k_hi,
            inv_uu,
            inv_uv,
            inv_vv,
            rho,
        })
    }

    pub fn support_len(&self) -> usize {
        self.env.support_len()
    }

    pub fn sample_rate(&self) -> f32 {
        self.env.sample_rate
    }

    /// Carrier frequency of bin `k`, Hz.
    pub fn bin_hz(&self, k: usize) -> f32 {
        k as f32 * self.env.sample_rate / self.fft_len as f32
    }

    /// Number of frames covering a signal of `n` samples.
    pub fn frame_count(&self, n: usize) -> usize {
        if n == 0 { 0 } else { (n - 1) / self.hop + 1 }
    }

    /// Onset of frame `n`, in samples.
    pub fn frame_onset(&self, n: usize) -> usize {
        n * self.hop
    }

    /// `G^-1` row entries for bin `k`, or `None` if the bin is disabled.
    pub fn gram_inv(&self, k: usize) -> Option<(f32, f32, f32)> {
        if k < self.k_lo || k > self.k_hi {
            return None;
        }
        let i = k - self.k_lo;
        if self.inv_uu[i] == 0.0 && self.inv_uv[i] == 0.0 && self.inv_vv[i] == 0.0 {
            return None;
        }
        Some((self.inv_uu[i], self.inv_uv[i], self.inv_vv[i]))
    }

    /// `G^-1`'s three rows over the whole bin range, indexed by `k - k_lo`.
    ///
    /// For scanning every bin of a frame. A dead bin holds zeros rather than being marked absent,
    /// so it projects to zero energy and can never win — which is the point of storing it that way,
    /// and lets the inner loop run straight down the slices with no per-bin branch. Use
    /// [`Block::gram_inv`] when you need one bin and care whether it is live.
    pub fn gram_rows(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.inv_uu, &self.inv_uv, &self.inv_vv)
    }

    /// The u/v coherence at bin `k`. Near 1.0 means the two basis vectors are nearly parallel and
    /// the projection is ill-conditioned.
    pub fn rho(&self, k: usize) -> Option<f32> {
        (k >= self.k_lo && k <= self.k_hi).then(|| self.rho[k - self.k_lo])
    }

    pub fn live_bins(&self) -> impl Iterator<Item = usize> + '_ {
        (self.k_lo..=self.k_hi).filter(|&k| self.gram_inv(k).is_some())
    }
}

/// Read `C` and `S` for frequency index `m` from the real spectrum of `E^2`.
///
/// A real FFT only returns bins `0..=fft_len/2`, but the Gram needs the spectrum at `2k`, which
/// exceeds that for every `k > fft_len/4` — i.e. every carrier above `sr/4`. Those fold back by
/// conjugate symmetry, which flips the sign of `S`. Getting it wrong is silent and plausible.
///
/// Note the fold is unreachable at the default `f_max` of 10 kHz at 48 kHz (the boundary is
/// 12 kHz), so it only engages at lower sample rates or a wider `f_max`. It is still required for
/// correctness there, and `gram_inverse_is_exact_across_the_fold_boundary` exercises it explicitly.
fn fold_e2_bin(spectrum: &[Complex32], fft_len: usize, m: usize) -> (f64, f64) {
    let m = m % fft_len;
    if m <= fft_len / 2 {
        (spectrum[m].re as f64, -spectrum[m].im as f64)
    } else {
        let mirrored = fft_len - m;
        (spectrum[mirrored].re as f64, spectrum[mirrored].im as f64)
    }
}

/// Largest onset offset still capturing `tol` of an atom's energy, doubled.
///
/// Frames sit on a grid of spacing `hop`, so a true onset is at most `hop/2` from the nearest
/// frame; requiring `capture(hop/2) >= tol` gives `hop = 2 * max{delta : capture(delta) >= tol}`.
///
/// Capture falls monotonically with `delta` — the envelope is a rise followed by a decay, and
/// shifting it further only ever lowers its overlap with itself — so the crossing is found by
/// galloping out in powers of two and bisecting, rather than stepping `delta` one sample at a time.
/// Each probe is an O(N) correlation, and at a loose tolerance on a low-`alpha` block the linear
/// scan took ~16k of them over a 331k window: seconds per block, all of it in a dictionary build
/// that used to take a millisecond. `hop_search_matches_the_linear_scan` pins the two against each
/// other on every block of both dictionaries, which is the assumption made checkable.
fn measure_hop(env: &[f32], fft_len: usize, tol: f64) -> usize {
    let ok = |delta: usize| delta < fft_len && envelope_capture(env, delta + 1, fft_len) >= tol;
    if !ok(0) {
        return 1;
    }
    // Invariant: `ok(lo)` holds and, once set, `ok(hi)` does not. The linear scan stops at the
    // first `delta` that fails and doubles *that*, so the answer is `hi`, not `lo`.
    let mut lo = 0usize;
    let mut hi = 1usize;
    while ok(hi) {
        lo = hi;
        hi *= 2;
        if hi >= fft_len {
            hi = fft_len;
            break;
        }
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if ok(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (2 * hi).max(1)
}

/// The one-sample-at-a-time form of [`measure_hop`]: the definition, kept for the test.
#[cfg(test)]
fn measure_hop_linear(env: &[f32], fft_len: usize, tol: f64) -> usize {
    let mut delta = 0usize;
    while delta < fft_len && envelope_capture(env, delta + 1, fft_len) >= tol {
        delta += 1;
    }
    (2 * delta).max(1)
}

/// Fraction of a true atom's energy recoverable by a frame anchored `delta` samples before it.
///
/// Only the envelope matters: the carrier phase is a free parameter and is absorbed by the
/// projection, so a carrier shift costs nothing. This is the squared normalized correlation of the
/// envelope against itself shifted by `delta`, over the frame window.
fn envelope_capture(env: &[f32], delta: usize, fft_len: usize) -> f64 {
    let (mut dot, mut ea, mut eb) = (0.0f64, 0.0f64, 0.0f64);
    for t in 0..fft_len {
        let a = env.get(t).copied().unwrap_or(0.0) as f64;
        let b = t
            .checked_sub(delta)
            .and_then(|i| env.get(i))
            .copied()
            .unwrap_or(0.0) as f64;
        dot += a * b;
        ea += a * a;
        eb += b * b;
    }
    if ea <= 0.0 || eb <= 0.0 {
        return 0.0;
    }
    dot * dot / (ea * eb)
}

/// A set of blocks, one per envelope shape, of any mix of kinds.
#[derive(Clone, Debug)]
pub struct Dictionary {
    pub blocks: Vec<Block>,
    pub sample_rate: f32,
}

impl Dictionary {
    /// Build from an explicit `(alpha, beta)` FOF list under `cfg`'s release policy, skipping
    /// combinations past the `amax` cliff.
    pub fn from_grid(
        grid: &[(f32, f32)],
        sample_rate: f32,
        planner: &mut dyn RealFftPlanner,
        cfg: &BlockConfig,
    ) -> Result<Self, FofError> {
        let shapes: Vec<Shape> = grid
            .iter()
            .map(|&(alpha, beta)| EnvelopeParams::with_policy(alpha, beta, &cfg.release).into())
            .collect();
        Self::from_shapes(&shapes, sample_rate, planner, cfg)
    }

    /// Build one block per shape, in the order given, skipping FOFs past the `amax` cliff.
    ///
    /// The order is the block index a book records and the tie-break selection uses, so a caller
    /// mixing kinds should keep one kind's blocks where they were — the settings put FOFs first,
    /// which is what leaves a FOF-only decomposition bit-identical when Gaussians are added.
    pub fn from_shapes(
        shapes: &[Shape],
        sample_rate: f32,
        planner: &mut dyn RealFftPlanner,
        cfg: &BlockConfig,
    ) -> Result<Self, FofError> {
        cfg.release.validate()?;
        let mut blocks = Vec::new();
        for &params in shapes {
            match Block::new(params, sample_rate, planner, cfg) {
                Ok(b) => blocks.push(b),
                // A grain that renders silent is simply not a usable atom shape.
                Err(FofError::SilentGrain { .. }) => continue,
                Err(e) => return Err(e),
            }
        }
        if blocks.is_empty() {
            return Err(FofError::Invalid("dictionary has no usable blocks"));
        }
        Ok(Self {
            blocks,
            sample_rate,
        })
    }

    /// The default grid for voice: `alpha` in a 1.6 ratio ladder over 25-680 Hz bandwidth, three
    /// attack durations, capped at `alpha*beta <= 4` for well-conditioned `amax`.
    pub fn voice(
        sample_rate: f32,
        planner: &mut dyn RealFftPlanner,
        cfg: &BlockConfig,
    ) -> Result<Self, FofError> {
        const ALPHAS: [f32; 8] = [80.0, 128.0, 205.0, 328.0, 524.0, 839.0, 1342.0, 2147.0];
        const BETAS: [f32; 3] = [0.0003, 0.001, 0.003];

        let grid: Vec<(f32, f32)> = ALPHAS
            .iter()
            .flat_map(|&a| BETAS.iter().map(move |&b| (a, b)))
            .filter(|&(a, b)| a * b <= 4.0)
            .collect();
        Self::from_grid(&grid, sample_rate, planner, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::Planner;
    use crate::gauss::GaussianParams;
    use std::f64::consts::TAU;

    const SR: f32 = 48_000.0;

    fn gauss_block(sigma: f32) -> Block {
        let mut planner = Planner::new();
        Block::new(GaussianParams::new(sigma), SR, &mut planner, &BlockConfig::default()).unwrap()
    }

    fn block(alpha: f32, beta: f32) -> Block {
        let mut planner = Planner::new();
        Block::new(
            EnvelopeParams::new(alpha, beta),
            SR,
            &mut planner,
            &BlockConfig::default(),
        )
        .unwrap()
    }

    /// Direct O(L) Gram, the oracle.
    fn gram_direct(env: &[f32], fft_len: usize, k: usize) -> (f64, f64, f64) {
        let w = TAU * k as f64 / fft_len as f64;
        let (mut uu, mut vv, mut uv) = (0.0, 0.0, 0.0);
        for t in 0..fft_len {
            let e = env.get(t).copied().unwrap_or(0.0) as f64;
            let (s, c) = ((w * t as f64).sin(), (w * t as f64).cos());
            uu += (e * s) * (e * s);
            vv += (e * c) * (e * c);
            uv += (e * s) * (e * c);
        }
        (uu, vv, uv)
    }

    /// Checks `G_direct * G_inv_stored == I`.
    ///
    /// Comparing the inverse entrywise is the wrong metric: `inv_uv` passes through zero as the
    /// basis vectors become orthogonal at high bins, so its *relative* error is unbounded there
    /// while being numerically irrelevant. The product against the direct Gram is scale-free and
    /// tests exactly the quantity the projection consumes.
    fn assert_gram_inverse_exact(b: &Block, probes: &[usize]) {
        let mut worst = 0.0f64;
        let mut checked = 0;
        for &k in probes {
            let Some((iuu, iuv, ivv)) = b.gram_inv(k) else {
                continue;
            };
            let (uu, vv, uv) = gram_direct(&b.env.samples, b.fft_len, k);
            // [uu uv; uv vv] * [iuu iuv; iuv ivv]
            let prod = [
                uu * iuu as f64 + uv * iuv as f64,
                uu * iuv as f64 + uv * ivv as f64,
                uv * iuu as f64 + vv * iuv as f64,
                uv * iuv as f64 + vv * ivv as f64,
            ];
            for (got, want) in prod.iter().zip([1.0, 0.0, 0.0, 1.0]) {
                worst = worst.max((got - want).abs());
            }
            checked += 1;
        }
        assert!(checked >= 3, "only {checked} live bins probed");
        assert!(worst < 1e-3, "G * G_inv deviates from identity by {worst:.3e}");
    }

    #[test]
    fn gram_inverse_matches_direct_sums() {
        let b = block(251.0, 0.002);
        let probes = [b.k_lo, b.k_lo + 1, b.fft_len / 8, b.fft_len / 5, b.k_hi];
        assert_gram_inverse_exact(&b, &probes);
    }

    #[test]
    fn gram_inverse_is_exact_across_the_fold_boundary() {
        // The fold engages only above sr/4, which the default f_max (10 kHz at 48 kHz) never
        // reaches. Widen f_max so bins land on both sides of fft_len/4 and the folded branch of
        // fold_e2_bin is actually executed.
        let mut planner = Planner::new();
        let cfg = BlockConfig {
            f_max: 20_000.0,
            ..BlockConfig::default()
        };
        let b = Block::new(EnvelopeParams::new(251.0, 0.002), SR, &mut planner, &cfg).unwrap();

        let quarter = b.fft_len / 4;
        let folded: Vec<usize> = (b.k_lo..=b.k_hi).filter(|&k| 2 * k > b.fft_len / 2).collect();
        assert!(
            folded.len() > 10,
            "expected many folded bins, got {}",
            folded.len()
        );

        let probes = [
            quarter - 1,
            quarter,
            quarter + 1,
            folded[folded.len() / 2],
            b.k_hi,
        ];
        assert_gram_inverse_exact(&b, &probes);
    }

    #[test]
    fn degenerate_bins_are_excluded() {
        let b = block(251.0, 0.002);
        assert!(b.k_lo >= 1, "DC must be excluded");
        assert!(b.k_hi < b.fft_len / 2, "Nyquist must be excluded");
        assert!(b.gram_inv(0).is_none());
        assert!(b.gram_inv(b.fft_len / 2).is_none());
    }

    #[test]
    fn coherence_falls_as_frequency_rises() {
        // rho ~ alpha/(2*pi*f), so low bins are ill-conditioned and high bins are nearly orthogonal.
        let b = block(524.0, 0.001);
        let lo = b.rho(b.k_lo).unwrap();
        let hi = b.rho(b.k_hi).unwrap();
        assert!(lo > hi, "rho should fall with frequency: {lo} -> {hi}");
        assert!(hi < 0.05, "high bins should be nearly orthogonal, got {hi}");
    }

    #[test]
    fn hop_shrinks_as_alpha_grows() {
        // Hop tracks the envelope's own decorrelation width, so it must fall as alpha rises.
        //
        // Deliberately not compared against the closed form 0.051*sr/alpha: that formula assumes a
        // pure exponential, while the measurement accounts for the attack softening the onset and
        // for the fade tail. Measured hops run *larger* than the formula (e.g. 6 vs 1.8 at
        // alpha=1342), which is the measurement doing its job rather than disagreeing.
        let mut prev = usize::MAX;
        for alpha in [80.0f32, 251.0, 524.0, 1342.0] {
            let hop = block(alpha, 0.0003).hop;
            assert!(hop < prev, "alpha={alpha}: hop {hop} did not fall below {prev}");
            prev = hop;
        }
    }

    #[test]
    fn hop_is_vastly_smaller_than_half_the_window() {
        // Guards against regressing to an MPTK-style hop, which would capture ~1e-4 of the energy.
        let b = block(251.0, 0.002);
        assert!(
            b.hop * 20 < b.fft_len,
            "hop {} is not far below fft_len/2 = {}",
            b.hop,
            b.fft_len / 2
        );
    }

    /// Galloping-plus-bisection assumes capture is monotone in `delta`. This checks the assumption
    /// rather than trusting it, on every shape either dictionary contains, at the tolerances the
    /// configs use — including the loose ones where the scan was slow enough to matter.
    #[test]
    fn hop_search_matches_the_linear_scan() {
        let mut shapes: Vec<(f32, f32)> = Vec::new();
        for a in [80.0f32, 128.0, 205.0, 328.0, 524.0, 839.0, 1342.0, 2147.0] {
            for b in [0.0003f32, 0.001, 0.003] {
                shapes.push((a, b));
            }
        }
        // The low-alpha grid, at a reduced rate so the test stays quick: the shapes are the same.
        for a in [1.0f32, 2.0, 4.0, 16.0, 64.0, 256.0] {
            for b in [0.0003f32, 0.001, 0.003, 0.012] {
                shapes.push((a, b));
            }
        }
        for (alpha, beta) in shapes {
            if alpha * beta > 4.0 {
                continue;
            }
            let sr = if alpha < 50.0 { 4_000.0 } else { SR };
            let env = Envelope::render(EnvelopeParams::new(alpha, beta), sr).unwrap();
            let fft_len = next_fast_len(env.support_len());
            for tol in [0.95, 0.7, 0.5, 0.3] {
                assert_eq!(
                    measure_hop(&env.samples, fft_len, tol),
                    measure_hop_linear(&env.samples, fft_len, tol),
                    "alpha={alpha} beta={beta} tol={tol}"
                );
            }
        }
    }

    #[test]
    fn measured_hop_meets_its_capture_tolerance() {
        for alpha in [80.0f32, 251.0, 1342.0] {
            let b = block(alpha, 0.0003);
            let worst = envelope_capture(&b.env.samples, b.hop / 2, b.fft_len);
            assert!(
                worst >= 0.94,
                "alpha={alpha}: capture at hop/2 = {worst:.4}, below tolerance"
            );
        }
    }

    // ── gaussian blocks ─────────────────────────────────────────────────────────────────────────

    /// The block machinery never asks which kind of envelope it holds; this is the check that it
    /// does not need to. Same oracle, same tolerance as the FOF blocks.
    #[test]
    fn gram_inverse_is_exact_for_gaussian_blocks() {
        for sigma in [0.001f32, 0.004, 0.02] {
            let b = gauss_block(sigma);
            let probes = [b.k_lo, b.k_lo + 1, b.k_lo + 3, b.fft_len / 8, b.fft_len / 5, b.k_hi];
            assert_gram_inverse_exact(&b, &probes);
        }
    }

    /// Capture of a shifted Gaussian is monotone in the shift, so the galloping search must agree
    /// with the definition here too — checked, not assumed, at every tolerance the configs use.
    #[test]
    fn hop_search_matches_the_linear_scan_for_gaussians() {
        for sigma in [0.0005f32, 0.002, 0.008, 0.03, 0.1] {
            // Wide ones at a reduced rate so the linear scan stays quick: the shape is the same.
            let sr = if sigma > 0.01 { 4_000.0 } else { SR };
            let env = Envelope::render(GaussianParams::new(sigma), sr).unwrap();
            let fft_len = next_fast_len(env.support_len());
            for tol in [0.95, 0.7, 0.5, 0.3] {
                assert_eq!(
                    measure_hop(&env.samples, fft_len, tol),
                    measure_hop_linear(&env.samples, fft_len, tol),
                    "sigma={sigma} tol={tol}"
                );
            }
        }
    }

    /// A Gaussian's hop is set by its width, as a FOF's is by `1/alpha`: capture falls as
    /// `exp(-delta^2 / 2 sigma^2)`, so the hop is proportional to sigma and meets its tolerance.
    #[test]
    fn gaussian_hop_scales_with_sigma_and_meets_its_tolerance() {
        let (narrow, wide) = (gauss_block(0.002), gauss_block(0.008));
        let ratio = wide.hop as f64 / narrow.hop as f64;
        assert!((3.6..4.4).contains(&ratio), "hop {} -> {}: ratio {ratio:.2}", narrow.hop, wide.hop);
        for b in [&narrow, &wide] {
            let worst = envelope_capture(&b.env.samples, b.hop / 2, b.fft_len);
            assert!(worst >= 0.94, "{}: capture at hop/2 = {worst:.4}", b.env.params.describe());
        }
    }

    /// Blocks come out in the order the shapes went in, kinds mixed, because that order is the
    /// block index a book records and the tie-break selection uses.
    #[test]
    fn a_mixed_dictionary_keeps_the_shape_order() {
        let mut planner = Planner::new();
        let shapes: Vec<Shape> = vec![
            EnvelopeParams::new(800.0, 0.001).into(),
            GaussianParams::new(0.003).into(),
            EnvelopeParams::new(2000.0, 0.01).into(), // past the amax cliff: skipped
            GaussianParams::new(0.001).into(),
        ];
        let d = Dictionary::from_shapes(&shapes, SR, &mut planner, &BlockConfig::default()).unwrap();
        let got: Vec<Shape> = d.blocks.iter().map(|b| b.env.params).collect();
        assert_eq!(got, vec![shapes[0], shapes[1], shapes[3]]);
    }

    #[test]
    fn voice_dictionary_covers_the_grid_and_skips_the_cliff() {
        let mut planner = Planner::new();
        let d = Dictionary::voice(SR, &mut planner, &BlockConfig::default()).unwrap();
        // 8 alphas x 3 betas = 24, less two past the alpha*beta <= 4 cap:
        // 1342*0.003 = 4.026 and 2147*0.003 = 6.441.
        assert_eq!(d.blocks.len(), 22);
        let fof = |b: &Block| b.env.params.as_fof().unwrap();
        for b in &d.blocks {
            assert!(fof(b).alpha_beta() <= 4.0);
            assert!(b.hop >= 1 && b.hop < b.fft_len);
            assert!(b.k_lo <= b.k_hi);
        }
        // Support (and so fft_len) must shrink as alpha grows.
        let mut by_alpha: Vec<_> = d.blocks.iter().filter(|b| fof(b).beta == 0.0003).collect();
        by_alpha.sort_by(|a, b| fof(a).alpha.total_cmp(&fof(b).alpha));
        for w in by_alpha.windows(2) {
            assert!(w[0].fft_len >= w[1].fft_len);
        }
    }

    #[test]
    fn frame_grid_covers_the_signal() {
        let b = block(251.0, 0.002);
        let n = 48_000;
        let frames = b.frame_count(n);
        assert!(b.frame_onset(frames - 1) < n);
        assert!(b.frame_onset(frames) >= n);
        assert_eq!(b.frame_count(0), 0);
    }
}
