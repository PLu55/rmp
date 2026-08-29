//! Frame correlation and the closed-form projection.
//!
//! # Correlation
//!
//! Window a residual segment by the block envelope and take one real FFT. Bin `k` then carries both
//! correlations against that bin's carrier:
//!
//! ```text
//! d_u = <R, E*sin(wt)> = -Im X[k]
//! d_v = <R, E*cos(wt)> =  Re X[k]        X = rfft(R[t0..t0+fft_len] * E)
//! ```
//!
//! # Projection
//!
//! There is nothing to optimize over phase. As `phi` sweeps `[0, 2*pi)`, `(cos phi, sin phi)` covers
//! every direction in R^2, so maximizing over `phi` *is* projecting onto the whole 2-D span of
//! `u = E*sin(wt)` and `v = E*cos(wt)`:
//!
//! ```text
//! z      = G^-1 d
//! energy = d . z
//! amp    = hypot(z_x, z_y)      always >= 0
//! phi    = atan2(z_y, z_x)
//! ```
//!
//! `z` is the least-squares coefficient pair for `R ~ z_x*u + z_y*v`, and
//! `z_x*u + z_y*v == amp * E(t) * sin(wt + phi)` — exactly rfofs's carrier. Because `amp` is
//! non-negative by construction there is no sign fixup and no mod-pi ambiguity.
//!
//! The `sqrt(d_u^2 + d_v^2)` shortcut is never used for selection: its relative error is `O(rho_k)`,
//! which reaches 8% for a 500 Hz formant of 80 Hz bandwidth. A greedy algorithm whose top candidates
//! differ by a few percent cannot tolerate that.

use crate::dict::Block;
use crate::fft::RealFft;
use crate::fof::{AtomParams, EnvelopeParams};
use realfft::num_complex::Complex32;

/// The best atom in one (frame, bin) cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    /// Energy removed by subtracting this atom: `d . G^-1 d`.
    pub energy: f64,
    /// Linear amplitude, mapping directly onto `FofParams::amp`.
    pub amp: f32,
    /// Carrier phase in radians, rfofs convention.
    pub phi: f32,
}

impl Projection {
    pub const ZERO: Self = Self {
        energy: 0.0,
        amp: 0.0,
        phi: 0.0,
    };
}

/// Solve the 2-D projection for one bin. Returns [`Projection::ZERO`] for disabled bins.
pub fn project(block: &Block, k: usize, d_u: f32, d_v: f32) -> Projection {
    let Some((inv_uu, inv_uv, inv_vv)) = block.gram_inv(k) else {
        return Projection::ZERO;
    };
    let (du, dv) = (d_u as f64, d_v as f64);
    let z_x = inv_uu as f64 * du + inv_uv as f64 * dv;
    let z_y = inv_uv as f64 * du + inv_vv as f64 * dv;

    let energy = du * z_x + dv * z_y;
    Projection {
        // A negative value can only come from round-off on a near-singular bin.
        energy: energy.max(0.0),
        amp: z_x.hypot(z_y) as f32,
        phi: z_y.atan2(z_x) as f32,
    }
}

/// Per-thread scratch for correlating frames of one block.
///
/// Holds the FFT plan, so it cannot be shared across rayon workers — each builds its own. That is
/// exactly the constraint [`RealFft::forward`] encodes by taking `&mut self`.
pub struct Correlator {
    fft: Box<dyn RealFft>,
    windowed: Vec<f32>,
    spectrum: Vec<Complex32>,
}

impl Correlator {
    pub fn new(block: &Block, planner: &mut dyn crate::fft::RealFftPlanner) -> Self {
        let fft = planner.plan(block.fft_len);
        let windowed = vec![0.0; block.fft_len];
        let spectrum = vec![Complex32::new(0.0, 0.0); fft.complex_len()];
        Self {
            fft,
            windowed,
            spectrum,
        }
    }

    /// Correlate the frame of `signal` starting at `onset` against every bin of `block`.
    ///
    /// Reads past the end of `signal` as zeros, so trailing frames need no padding by the caller.
    pub fn correlate(&mut self, block: &Block, signal: &[f32], onset: usize) {
        let env = &block.env.samples;
        self.windowed.fill(0.0);
        let avail = signal.len().saturating_sub(onset).min(env.len());
        for i in 0..avail {
            self.windowed[i] = signal[onset + i] * env[i];
        }
        self.fft.forward(&mut self.windowed, &mut self.spectrum);
    }

    /// Correlations at bin `k` from the last [`correlate`](Self::correlate) call.
    pub fn at(&self, k: usize) -> (f32, f32) {
        let x = self.spectrum[k];
        (-x.im, x.re) // (d_u, d_v)
    }

    /// Best bin of the last correlated frame, with its projection.
    pub fn best_bin(&self, block: &Block) -> (usize, Projection) {
        let mut best = (block.k_lo, Projection::ZERO);
        for k in block.k_lo..=block.k_hi {
            let (d_u, d_v) = self.at(k);
            let p = project(block, k, d_u, d_v);
            if p.energy > best.1.energy {
                best = (k, p);
            }
        }
        best
    }

    /// Turn a selected cell into replayable atom parameters.
    pub fn atom(&self, block: &Block, onset: usize, k: usize, p: Projection) -> AtomParams {
        AtomParams {
            t0: onset as i64,
            f: block.bin_hz(k),
            env: EnvelopeParams { ..block.env.params },
            phi: p.phi,
            amp: p.amp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict::{BlockConfig, Dictionary};
    use crate::fft::Planner;
    use std::f64::consts::TAU;

    const SR: f32 = 48_000.0;

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

    fn xorshift(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 40) as f32 / 8_388_608.0 - 1.0
    }

    /// Direct O(L) correlations — the oracle.
    fn correlate_direct(env: &[f32], sig: &[f32], onset: usize, fft_len: usize, k: usize) -> (f64, f64) {
        let w = TAU * k as f64 / fft_len as f64;
        let (mut du, mut dv) = (0.0, 0.0);
        for t in 0..fft_len {
            let r = sig.get(onset + t).copied().unwrap_or(0.0) as f64;
            let e = env.get(t).copied().unwrap_or(0.0) as f64;
            du += r * e * (w * t as f64).sin();
            dv += r * e * (w * t as f64).cos();
        }
        (du, dv)
    }

    #[test]
    fn frame_fft_matches_direct_correlation() {
        let b = block(251.0, 0.002);
        let mut planner = Planner::new();
        let mut c = Correlator::new(&b, &mut planner);

        let mut seed = 0xdead_beef_cafe_1234u64;
        let sig: Vec<f32> = (0..b.fft_len * 2).map(|_| xorshift(&mut seed)).collect();
        c.correlate(&b, &sig, 17);

        let scale = (0..b.fft_len)
            .map(|t| b.env.samples.get(t).copied().unwrap_or(0.0).abs() as f64)
            .sum::<f64>()
            .max(1.0);
        let mut worst = 0.0f64;
        for k in [b.k_lo, b.k_lo + 3, b.fft_len / 4, b.fft_len / 3, b.k_hi] {
            if k < b.k_lo || k > b.k_hi {
                continue;
            }
            let (du, dv) = correlate_direct(&b.env.samples, &sig, 17, b.fft_len, k);
            let (gu, gv) = c.at(k);
            worst = worst.max((gu as f64 - du).abs() / scale);
            worst = worst.max((gv as f64 - dv).abs() / scale);
        }
        assert!(worst < 1e-5, "worst correlation error {worst:.3e}");
    }

    /// Setting R = u must return d_u = <u,u> and d_v = <u,v>, which pins the sign of both the
    /// frame FFT's imaginary part and the Gram's S term simultaneously.
    #[test]
    fn self_correlation_pins_both_sign_conventions() {
        let b = block(251.0, 0.002);
        let mut planner = Planner::new();
        let mut c = Correlator::new(&b, &mut planner);

        // A low bin, where <u,v> is substantial. At high bins the basis is nearly orthogonal and
        // <u,v> passes through zero, making any relative comparison against it meaningless.
        let k = b.k_lo;
        let w = TAU * k as f64 / b.fft_len as f64;

        // u(t) = E(t) sin(wt), as a signal.
        let u: Vec<f32> = (0..b.fft_len)
            .map(|t| {
                b.env.samples.get(t).copied().unwrap_or(0.0) * (w * t as f64).sin() as f32
            })
            .collect();
        c.correlate(&b, &u, 0);
        let (d_u, d_v) = c.at(k);

        let (uu, vv, uv) = {
            let (mut uu, mut vv, mut uv) = (0.0f64, 0.0, 0.0);
            for t in 0..b.fft_len {
                let e = b.env.samples.get(t).copied().unwrap_or(0.0) as f64;
                let (s, cs) = ((w * t as f64).sin(), (w * t as f64).cos());
                uu += (e * s) * (e * s);
                vv += (e * cs) * (e * cs);
                uv += (e * s) * (e * cs);
            }
            (uu, vv, uv)
        };
        let _ = vv;

        // Scale both by <u,u>: it is the natural magnitude here, and <u,v> may legitimately be
        // near zero without that indicating any error.
        assert!(
            (d_u as f64 - uu).abs() / uu < 1e-4,
            "d_u {d_u} != <u,u> {uu}"
        );
        assert!(
            (d_v as f64 - uv).abs() / uu < 1e-4,
            "d_v {d_v} != <u,v> {uv} (scaled by <u,u> = {uu})"
        );
    }

    #[test]
    fn projection_matches_brute_force_over_phi() {
        let b = block(251.0, 0.002);
        let mut planner = Planner::new();
        let mut c = Correlator::new(&b, &mut planner);
        let mut seed = 0x0123_4567_89ab_cdefu64;
        let sig: Vec<f32> = (0..b.fft_len).map(|_| xorshift(&mut seed)).collect();
        c.correlate(&b, &sig, 0);

        for k in [b.k_lo, b.k_lo + 2, b.fft_len / 4, b.k_hi] {
            if b.gram_inv(k).is_none() {
                continue;
            }
            let (d_u, d_v) = c.at(k);
            let got = project(&b, k, d_u, d_v);

            // Brute force: maximize <R,w>^2 / <w,w> over phi with w = cos(phi)u + sin(phi)v.
            let w = TAU * k as f64 / b.fft_len as f64;
            let (mut uu, mut vv, mut uv) = (0.0f64, 0.0, 0.0);
            for t in 0..b.fft_len {
                let e = b.env.samples.get(t).copied().unwrap_or(0.0) as f64;
                let (s, cs) = ((w * t as f64).sin(), (w * t as f64).cos());
                uu += (e * s) * (e * s);
                vv += (e * cs) * (e * cs);
                uv += (e * s) * (e * cs);
            }
            let (du, dv) = (d_u as f64, d_v as f64);
            let mut best = 0.0f64;
            const STEPS: usize = 100_000;
            for i in 0..STEPS {
                let p = TAU * i as f64 / STEPS as f64;
                let (c1, c2) = (p.cos(), p.sin());
                let den = c1 * c1 * uu + 2.0 * c1 * c2 * uv + c2 * c2 * vv;
                if den > 1e-18 {
                    best = best.max((c1 * du + c2 * dv).powi(2) / den);
                }
            }
            let rel = (got.energy - best).abs() / best.max(1e-12);
            assert!(rel < 1e-4, "k={k}: closed form {} vs brute force {best}", got.energy);
        }
    }

    /// The reconstruction identity: `amp * E(t) * sin(wt + phi)` must be the least-squares fit, so
    /// subtracting it removes exactly `energy` from the residual.
    #[test]
    fn projection_energy_equals_measured_energy_removed() {
        let b = block(251.0, 0.002);
        let mut planner = Planner::new();
        let mut c = Correlator::new(&b, &mut planner);
        let mut seed = 0xfeed_face_0000_1111u64;
        let sig: Vec<f32> = (0..b.fft_len).map(|_| xorshift(&mut seed)).collect();
        c.correlate(&b, &sig, 0);

        let k = b.fft_len / 6;
        let (d_u, d_v) = c.at(k);
        let p = project(&b, k, d_u, d_v);
        let w = TAU * k as f64 / b.fft_len as f64;

        let before: f64 = sig.iter().map(|&s| (s as f64).powi(2)).sum();
        let after: f64 = (0..b.fft_len)
            .map(|t| {
                let e = b.env.samples.get(t).copied().unwrap_or(0.0) as f64;
                let atom = p.amp as f64 * e * (w * t as f64 + p.phi as f64).sin();
                (sig[t] as f64 - atom).powi(2)
            })
            .sum();

        let removed = before - after;
        let rel = (removed - p.energy).abs() / p.energy;
        assert!(
            rel < 1e-3,
            "energy removed {removed} != projected {} (rel {rel:.2e})",
            p.energy
        );
    }

    #[test]
    fn planted_atom_is_found_at_its_own_bin() {
        let b = block(251.0, 0.002);
        let mut planner = Planner::new();
        let mut c = Correlator::new(&b, &mut planner);

        let k_true = b.fft_len / 8;
        let want = AtomParams {
            t0: 0,
            f: b.bin_hz(k_true),
            env: b.env.params,
            phi: 0.9,
            amp: 0.6,
        };
        let sig = want.render(SR).unwrap();

        c.correlate(&b, &sig, 0);
        let (k, p) = c.best_bin(&b);
        assert_eq!(k, k_true, "selected bin {k}, planted at {k_true}");
        assert!((p.amp - want.amp).abs() < 0.02, "amp {} vs {}", p.amp, want.amp);

        let dphi = (p.phi - want.phi).rem_euclid(std::f32::consts::TAU);
        let dphi = dphi.min(std::f32::consts::TAU - dphi);
        assert!(dphi < 0.05, "phi {} vs {} (diff {dphi})", p.phi, want.phi);
    }

    #[test]
    fn disabled_bins_project_to_zero() {
        let b = block(251.0, 0.002);
        assert_eq!(project(&b, 0, 1.0, 1.0), Projection::ZERO);
        assert_eq!(project(&b, b.fft_len / 2, 1.0, 1.0), Projection::ZERO);
        assert_eq!(project(&b, b.k_hi + 1, 1.0, 1.0), Projection::ZERO);
    }

    #[test]
    fn amplitude_is_never_negative() {
        let b = block(251.0, 0.002);
        let k = b.fft_len / 7;
        for (du, dv) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
            let p = project(&b, k, du, dv);
            assert!(p.amp >= 0.0, "amp {} is negative for d=({du},{dv})", p.amp);
            assert!(p.energy >= 0.0);
        }
    }

    #[test]
    fn correlator_works_for_every_block_in_the_voice_dictionary() {
        let mut planner = Planner::new();
        let d = Dictionary::voice(SR, &mut planner, &BlockConfig::default()).unwrap();
        let mut seed = 0xaaaa_bbbb_cccc_ddddu64;
        let sig: Vec<f32> = (0..8192).map(|_| xorshift(&mut seed)).collect();

        for b in &d.blocks {
            let mut c = Correlator::new(b, &mut planner);
            c.correlate(b, &sig, 0);
            let (k, p) = c.best_bin(b);
            assert!(b.gram_inv(k).is_some(), "winning bin {k} is disabled");
            assert!(p.energy.is_finite() && p.energy >= 0.0);
            assert!(p.amp.is_finite() && p.phi.is_finite());
        }
    }
}
