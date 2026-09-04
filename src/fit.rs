//! Exact quadrature fitting at arbitrary `(alpha, beta, f, t0)`.
//!
//! [`crate::corr`] scores the dictionary's own grid using a Gram table precomputed once per block.
//! Refinement and HRMP both need the same score *off* that grid, where no table exists. This module
//! computes it by direct summation, in f64, with the identical algebra and sign conventions — so a
//! refined candidate's score is comparable with a coarse one to the last bit.
//!
//! It is separate from `refine` because `hrmp` needs [`Quad::energy_at`] without needing an
//! optimizer.
//!
//! # The Gram comes from `E^2`, not from `u` and `v`
//!
//! With `u = E*sin(wt)` and `v = E*cos(wt)`, and
//!
//! ```text
//! P = sum E^2      C = sum E^2 cos(2wt)      S = sum E^2 sin(2wt)
//! ```
//!
//! the product-to-sum identities give `<u,u> = (P-C)/2`, `<v,v> = (P+C)/2`, `<u,v> = S/2`. That is
//! one loop over `E^2` instead of three over `u` and `v`, and it is the same algebra
//! [`crate::dict::Block::new`] already reads out of one FFT — so the sign convention is inherited
//! rather than re-derived, which is the only way to be sure the two paths agree.
//!
//! **`G` does not depend on `t0`** for an atom lying wholly inside the signal. A caller sweeping
//! `t0` at fixed `(alpha, beta, f)` can compute it once.
//!
//! # Local atom time
//!
//! The phase reference is the FOF onset, so the carrier is evaluated at `t = e_start + i`, never at
//! the loop index. When `t0 < 0` the leading `-t0` envelope samples fall outside the signal and are
//! skipped, but the ones that remain keep their original phase. Restarting the phase at the clip
//! point would make `phi` jump discontinuously as a refined `t0` swept through zero, and the
//! rendered atom would no longer match the fit.
//!
//! # Where this deliberately differs from `corr` and `naive`
//!
//! Both of those use a whole-support Gram while reading the residual past the signal end as zeros.
//! For an atom overhanging a boundary that *understates* the energy actually removable. Here `G` is
//! clipped to the same sample range as `d`, so the score is the true energy removed. The two agree
//! to the last bit for interior atoms — but it means a seed and a refined candidate must be
//! compared through *this* function, never one here against one from the block table.

use crate::corr::Projection;
use crate::fof::Envelope;

/// Sufficient statistics for one exact quadrature fit.
///
/// `d` and `G` are accumulated over the same clipped sample range, so [`Quad::solve`] and
/// [`Quad::energy_at`] describe energy actually removable from the residual.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub g_uu: f64,
    pub g_uv: f64,
    pub g_vv: f64,
    pub d_u: f64,
    pub d_v: f64,
    /// Samples overlapped. Equals the envelope length for an interior atom.
    pub n: usize,
    /// First envelope index used; nonzero only when `t0 < 0`.
    pub e_start: usize,
}

/// Samples between exact re-seeds of the carrier recurrence.
///
/// The rotation drifts by about `k * eps` over `k` steps, so 512 costs ~1.1e-13 rad — seven orders
/// below the ~1e-4 ripple rfofs's own polynomial/LUT carrier already carries. It must stay f64: an
/// f32 rotation would drift 3.6e-3 rad across a 30000-sample support.
const RESEED: usize = 512;

impl Quad {
    /// Solve the 2-D projection. Returns `None` for an ill-conditioned or degenerate fit.
    ///
    /// The conditioning gate is [`crate::dict::Block::new`]'s, rewritten without the `E^2` spectrum:
    /// since `trace = P` and `4*det = P^2 - C^2 - S^2`, `rho^2 = 1 - 4*det/trace^2` exactly.
    pub fn solve(&self, rho_sq_max: f64) -> Option<Projection> {
        let (z_x, z_y, energy) = self.solve_z(rho_sq_max)?;
        Some(Projection {
            energy,
            amp: z_x.hypot(z_y) as f32,
            phi: z_y.atan2(z_x) as f32,
        })
    }

    /// Captured energy alone, skipping the amplitude and phase.
    ///
    /// [`crate::refine`]'s 1-D searches evaluate this a few hundred times per candidate and read
    /// nothing but the energy, so deriving `amp` and `phi` there is a `hypot` and an `atan2` thrown
    /// away each time.
    pub fn energy(&self, rho_sq_max: f64) -> Option<f64> {
        self.solve_z(rho_sq_max).map(|(_, _, e)| e)
    }

    /// `z = G^-1 d` and the energy it captures, behind the conditioning gate.
    ///
    /// One implementation so [`Quad::solve`] and [`Quad::energy`] cannot disagree — the search
    /// maximises one and the book records the other.
    #[inline]
    fn solve_z(&self, rho_sq_max: f64) -> Option<(f64, f64, f64)> {
        let det = self.g_uu * self.g_vv - self.g_uv * self.g_uv;
        let tr = self.g_uu + self.g_vv;
        // Written as a positive condition so a NaN falls out as `false` rather than through a
        // negated float comparison.
        let usable = det > 0.0 && self.g_uu > 0.0 && tr > 0.0;
        if !usable {
            return None;
        }
        if 1.0 - 4.0 * det / (tr * tr) > rho_sq_max {
            return None;
        }

        let z_x = (self.g_vv * self.d_u - self.g_uv * self.d_v) / det;
        let z_y = (self.g_uu * self.d_v - self.g_uv * self.d_u) / det;
        // A negative value can only come from round-off on a near-singular fit.
        Some((z_x, z_y, (self.d_u * z_x + self.d_v * z_y).max(0.0)))
    }

    /// Energy removed by subtracting `amp * E * sin(wt + phi)`, for an arbitrary `(amp, phi)`.
    ///
    /// `2A<r,g> - A^2<g,g>` with `g = cos(phi)*u + sin(phi)*v`. At the least-squares `(amp, phi)`
    /// this equals [`Quad::solve`]'s energy; HRMP calls it with the amplitude clamped below that.
    pub fn energy_at(&self, amp: f64, phi: f64) -> f64 {
        let (s, c) = phi.sin_cos();
        let r_g = c * self.d_u + s * self.d_v;
        let g_g = c * c * self.g_uu + 2.0 * c * s * self.g_uv + s * s * self.g_vv;
        2.0 * amp * r_g - amp * amp * g_g
    }
}

/// Accumulate the fit statistics for `env` placed at `t0` with carrier `omega` rad/sample.
///
/// Returns `None` if the atom and the signal do not overlap at all.
pub fn accumulate(residual: &[f32], env: &[f32], t0: i64, omega: f64) -> Option<Quad> {
    accumulate_reseed(residual, env, t0, omega, RESEED)
}

/// [`accumulate`] with the carrier evaluated exactly at every sample — the reference path.
///
/// Kept for the test that pins the recurrence against it, mirroring the `NaiveRef` discipline used
/// throughout this crate.
pub fn accumulate_exact(residual: &[f32], env: &[f32], t0: i64, omega: f64) -> Option<Quad> {
    accumulate_reseed(residual, env, t0, omega, 1)
}

fn accumulate_reseed(
    residual: &[f32],
    env: &[f32],
    t0: i64,
    omega: f64,
    reseed: usize,
) -> Option<Quad> {
    // Shared with `signal::subtract_at`, so the basis spans exactly the samples that would be
    // written when this atom is subtracted.
    let (e_start, r_start, n) = crate::signal::overlap(residual.len(), env.len(), t0)?;

    let (sw1, cw1) = omega.sin_cos();
    let (sw2, cw2) = (2.0 * omega).sin_cos();

    let (mut p, mut cc, mut ss) = (0.0f64, 0.0f64, 0.0f64);
    let (mut d_u, mut d_v) = (0.0f64, 0.0f64);

    let mut i = 0;
    while i < n {
        let chunk = reseed.min(n - i);
        let t = (e_start + i) as f64;
        let (mut s1, mut c1) = (omega * t).sin_cos();
        let (mut s2, mut c2) = (2.0 * omega * t).sin_cos();

        for j in 0..chunk {
            let e = env[e_start + i + j] as f64;
            let r = residual[r_start + i + j] as f64;
            let e2 = e * e;
            p += e2;
            cc += e2 * c2;
            ss += e2 * s2;
            d_u += r * e * s1;
            d_v += r * e * c1;

            let n1 = s1 * cw1 + c1 * sw1;
            c1 = c1 * cw1 - s1 * sw1;
            s1 = n1;
            let n2 = s2 * cw2 + c2 * sw2;
            c2 = c2 * cw2 - s2 * sw2;
            s2 = n2;
        }
        i += chunk;
    }

    Some(Quad {
        g_uu: (p - cc) * 0.5,
        g_uv: ss * 0.5,
        g_vv: (p + cc) * 0.5,
        d_u,
        d_v,
        n,
        e_start,
    })
}

/// Score an atom of shape `env` at onset `t0` and frequency `f` Hz against `residual`.
pub fn score(
    residual: &[f32],
    env: &Envelope,
    t0: i64,
    f: f32,
    rho_sq_max: f64,
) -> Option<Projection> {
    score_slice(residual, &env.samples, env.sample_rate, t0, f, rho_sq_max)
}

/// [`score`] over an explicit envelope slice — used to score the fit region, which is a prefix.
pub fn score_slice(
    residual: &[f32],
    env: &[f32],
    sample_rate: f32,
    t0: i64,
    f: f32,
    rho_sq_max: f64,
) -> Option<Projection> {
    let omega = std::f64::consts::TAU * f as f64 / sample_rate as f64;
    accumulate(residual, env, t0, omega)?.solve(rho_sq_max)
}

/// [`score_slice`]'s captured energy alone — the objective the 1-D searches maximise.
pub fn score_energy(
    residual: &[f32],
    env: &[f32],
    sample_rate: f32,
    t0: i64,
    f: f32,
    rho_sq_max: f64,
) -> Option<f64> {
    let omega = std::f64::consts::TAU * f as f64 / sample_rate as f64;
    accumulate(residual, env, t0, omega)?.energy(rho_sq_max)
}

/// Index one past the last sample of the exponential body — where the linear release begins.
///
/// rfofs enters the release where the *raw* decay reaches `fade_level`, and [`Envelope::render`]
/// divides the whole envelope by `fof_amax(alpha, beta)`. So in rendered units the release starts at
/// exactly `fade_level / amax`, a constant that needs no knowledge of rfofs's internal rounding.
///
/// A fixed threshold relative to the *peak* would not work: `amax` runs from 0.94 down to 0.071
/// across the useful grid, putting the true release entry anywhere between -59.5 dB and -37.1 dB
/// below peak. A single relative gate would cut 20 dB of genuine decay at one end of the grid and
/// admit the whole release at the other.
///
/// Falls back to the full support if no sample reaches the threshold, which is the safe direction:
/// an over-long fit region costs accuracy, a truncated one costs correctness.
pub fn fit_end(env: &Envelope) -> usize {
    let theta = env.params.fade_level / env.params.amax();
    let usable = theta > 0.0 && theta.is_finite();
    if !usable {
        return env.support_len();
    }
    env.samples
        .iter()
        .rposition(|&e| e >= theta)
        .map_or(env.support_len(), |i| i + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corr::{Correlator, project};
    use crate::dict::{Block, BlockConfig, Dictionary};
    use crate::fft::Planner;
    use crate::fof::{AtomParams, EnvelopeParams};

    const SR: f32 = 48_000.0;
    const RHO: f64 = 1.0 - 1e-4;

    fn xorshift(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 40) as f32 / 8_388_608.0 - 1.0
    }

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| xorshift(&mut s)).collect()
    }

    fn voice() -> Dictionary {
        let mut planner = Planner::new();
        Dictionary::voice(SR, &mut planner, &BlockConfig::default()).unwrap()
    }

    /// The whole point of the module: an on-grid score computed here must equal the one the
    /// dictionary's precomputed Gram produces. This is the analogue of
    /// `naive::fast_path_projection_matches_the_oracle`, and it is what licenses trusting the
    /// off-grid path where no oracle exists.
    #[test]
    fn on_grid_score_matches_the_block_gram() {
        let dict = voice();
        let mut planner = Planner::new();
        let signal = noise(24_000, 0x51ce);

        for (bi, block) in dict.blocks.iter().enumerate() {
            let mut corr = Correlator::new(block, &mut planner);
            let frames = block.frame_count(signal.len());
            for n in [0, 1, frames / 3, frames / 2] {
                let onset = block.frame_onset(n);
                // Interior frames only: `corr` uses a whole-support Gram while this module clips
                // it, and the two are equal only where the atom fits inside the signal.
                if onset + block.support_len() > signal.len() {
                    continue;
                }
                corr.correlate(block, &signal, onset);

                for k in [block.k_lo, block.k_lo + 7, (block.k_lo + block.k_hi) / 2, block.k_hi] {
                    let (d_u, d_v) = corr.at(k);
                    let want = project(block, k, d_u, d_v);
                    if want.energy <= 0.0 {
                        continue;
                    }
                    let got = score(&signal, &block.env, onset as i64, block.bin_hz(k), RHO)
                        .unwrap_or_else(|| panic!("block {bi} bin {k} rejected"));

                    let rel = (got.energy - want.energy).abs() / want.energy;
                    assert!(rel < 2e-4, "block {bi} bin {k}: energy {got:?} vs {want:?}");
                    assert!(
                        (got.amp - want.amp).abs() <= 2e-4 * want.amp.abs().max(1e-9),
                        "block {bi} bin {k}: amp {} vs {}",
                        got.amp,
                        want.amp
                    );
                    let dphi = (got.phi - want.phi).abs();
                    let dphi = dphi.min((std::f32::consts::TAU - dphi).abs());
                    assert!(dphi < 2e-3, "block {bi} bin {k}: phi {} vs {}", got.phi, want.phi);
                }
            }
        }
    }

    /// The blocked carrier recurrence must be indistinguishable from evaluating `sin_cos` at every
    /// sample. If this ever fails, `RESEED` is too large.
    #[test]
    fn recurrence_matches_exact_trig() {
        let dict = voice();
        let signal = noise(24_000, 0xf00d);
        for block in dict.blocks.iter().take(6) {
            for f in [55.0, 440.0, 3000.0, 9500.0] {
                let omega = std::f64::consts::TAU * f / SR as f64;
                let fast = accumulate(&signal, &block.env.samples, 1234, omega).unwrap();
                let slow = accumulate_exact(&signal, &block.env.samples, 1234, omega).unwrap();
                for (a, b, what) in [
                    (fast.g_uu, slow.g_uu, "g_uu"),
                    (fast.g_uv, slow.g_uv, "g_uv"),
                    (fast.g_vv, slow.g_vv, "g_vv"),
                    (fast.d_u, slow.d_u, "d_u"),
                    (fast.d_v, slow.d_v, "d_v"),
                ] {
                    let scale = b.abs().max(fast.g_vv.abs()).max(1e-12);
                    assert!((a - b).abs() / scale < 1e-11, "{what}: {a} vs {b} at f={f}");
                }
            }
        }
    }

    #[test]
    fn recovers_a_planted_atom_off_the_grid() {
        let dict = voice();
        let block = &dict.blocks[8];
        // Deliberately between bins, between hop positions, and at the block's own envelope.
        let atom = AtomParams {
            t0: 3_333,
            f: block.bin_hz(block.k_lo + 40) + 0.37 * block.bin_hz(1),
            env: block.env.params,
            phi: 0.9,
            amp: 0.42,
        };
        let mut signal = vec![0.0f32; 20_000];
        crate::signal::add_at(&mut signal, &atom.render(SR).unwrap(), atom.t0);

        let got = score(&signal, &block.env, atom.t0, atom.f, RHO).unwrap();
        assert!(
            (got.amp - atom.amp).abs() / atom.amp < 5e-3,
            "amp {} vs {}",
            got.amp,
            atom.amp
        );
        let dphi = (got.phi - atom.phi).abs();
        assert!(dphi < 5e-3, "phi {} vs {}", got.phi, atom.phi);
        // A perfect fit captures essentially all of the signal's energy.
        let total = crate::signal::energy_of(&signal);
        assert!(got.energy / total > 0.999, "captured {} of {total}", got.energy);
    }

    #[test]
    fn energy_at_the_least_squares_solution_equals_solve() {
        let dict = voice();
        let block = &dict.blocks[5];
        let signal = noise(20_000, 0xbeef);
        let omega = std::f64::consts::TAU * 700.0 / SR as f64;
        let q = accumulate(&signal, &block.env.samples, 900, omega).unwrap();
        let p = q.solve(RHO).unwrap();
        let got = q.energy_at(p.amp as f64, p.phi as f64);
        // 1e-6, not 1e-12: `Projection` stores amp and phi as f32 (they map straight onto
        // `FofParams`), so round-tripping the solution back through `energy_at` costs about 1e-7
        // relative. The algebra is exact; the carrier of it is not.
        assert!(
            (got - p.energy).abs() / p.energy < 1e-6,
            "energy_at {got} vs solve {}",
            p.energy
        );
    }

    /// HRMP clamps the amplitude below the least-squares value while holding the phase. The energy
    /// then follows `E_MP * (2*rho - rho^2)`, which is what lets the HR score be computed without
    /// any new inner products.
    #[test]
    fn clamped_energy_follows_the_two_rho_minus_rho_squared_law() {
        let dict = voice();
        let block = &dict.blocks[5];
        let signal = noise(20_000, 0x1234);
        let omega = std::f64::consts::TAU * 1500.0 / SR as f64;
        let q = accumulate(&signal, &block.env.samples, 512, omega).unwrap();
        let p = q.solve(RHO).unwrap();

        for rho in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let got = q.energy_at(rho * p.amp as f64, p.phi as f64);
            let want = p.energy * (2.0 * rho - rho * rho);
            // Bounded by the f32 amp/phi round-trip, as above.
            assert!(
                (got - want).abs() <= 1e-6 * p.energy.max(1e-12),
                "rho={rho}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn phase_reference_is_the_onset_even_when_the_atom_starts_before_the_signal() {
        // An atom whose onset precedes the excerpt must report the phase it was born with, not the
        // phase at the clip point.
        let dict = voice();
        let block = &dict.blocks[3];
        let env = &block.env;
        let f = 620.0;
        let full = AtomParams { t0: 0, f, env: env.params, phi: 0.7, amp: 0.3 };
        let rendered = full.render(SR).unwrap();

        let cut = env.support_len() / 4;
        let signal: Vec<f32> = rendered[cut..].to_vec();
        let got = score(&signal, env, -(cut as i64), f, RHO).unwrap();

        assert!((got.amp - 0.3).abs() / 0.3 < 1e-2, "amp {}", got.amp);
        let dphi = (got.phi - 0.7).abs();
        assert!(dphi < 1e-2, "phi {} should still be the onset phase", got.phi);
    }

    #[test]
    fn no_overlap_yields_none() {
        let dict = voice();
        let env = &dict.blocks[0].env;
        let signal = noise(1000, 1);
        assert!(accumulate(&signal, &env.samples, 1000, 0.1).is_none());
        assert!(accumulate(&signal, &env.samples, -(env.support_len() as i64), 0.1).is_none());
        // One sample of overlap on either side is still a fit.
        assert!(accumulate(&signal, &env.samples, 999, 0.1).is_some());
        assert!(accumulate(&signal, &env.samples, 1 - env.support_len() as i64, 0.1).is_some());
    }

    #[test]
    fn degenerate_fits_are_rejected() {
        let dict = voice();
        let env = &dict.blocks[0].env;
        let signal = noise(20_000, 7);
        // DC: the sine basis vector is identically zero, so G is exactly rank-1.
        assert!(score(&signal, env, 0, 0.0, RHO).is_none());
        // Nyquist: likewise.
        assert!(score(&signal, env, 0, SR / 2.0, RHO).is_none());
    }

    // ── the fit region ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn fit_end_lands_at_the_start_of_the_linear_release() {
        let dict = voice();
        for block in &dict.blocks {
            let env = &block.env;
            let p = env.params;
            let got = fit_end(env);
            let analytic = {
                let natural =
                    (-(p.fade_level as f64).ln() / (p.alpha as f64 / SR as f64)).ceil() as usize;
                natural.max((p.beta * SR).ceil() as usize)
            };
            assert!(
                got.abs_diff(analytic) <= 2,
                "alpha={} beta={}: fit_end {got} vs release start {analytic}",
                p.alpha,
                p.beta
            );
            // Spec 8.1: the complete attack is always inside the fit region. rfofs clamps the
            // release entry up to the attack end, so this holds even when the attack outlasts the
            // decay.
            assert!(got >= (p.beta * SR).ceil() as usize, "alpha={}", p.alpha);
            assert!(got < env.support_len(), "alpha={}: nothing excluded", p.alpha);
        }
    }

    #[test]
    fn the_excluded_tail_carries_almost_no_energy() {
        // The release is dropped because it says nothing about alpha and beta while adding
        // noise-dominated samples to the objective — not because it would distort the score. This
        // pins the second half of that claim.
        let dict = voice();
        for block in &dict.blocks {
            let env = &block.env;
            let cut = fit_end(env);
            let tail: f64 = env.samples[cut..].iter().map(|&e| (e as f64).powi(2)).sum();
            assert!(
                tail / env.energy < 1e-3,
                "alpha={}: tail carries {:.2e} of the energy",
                env.params.alpha,
                tail / env.energy
            );
        }
    }

    #[test]
    fn fit_region_and_full_support_scores_differ_but_track() {
        // They must not be interchangeable (that is why refinement keeps them apart) yet the fit
        // region must still capture the great majority of a well-matched atom.
        let dict = voice();
        let block = &dict.blocks[6];
        let atom = AtomParams {
            t0: 2_000,
            f: 830.0,
            env: block.env.params,
            phi: 0.2,
            amp: 1.0,
        };
        let mut signal = vec![0.0f32; 20_000];
        crate::signal::add_at(&mut signal, &atom.render(SR).unwrap(), atom.t0);

        let full = score(&signal, &block.env, atom.t0, atom.f, RHO).unwrap();
        let cut = fit_end(&block.env);
        let part =
            score_slice(&signal, &block.env.samples[..cut], SR, atom.t0, atom.f, RHO).unwrap();
        assert!(part.energy < full.energy);
        assert!(part.energy / full.energy > 0.99);
    }

    #[test]
    fn block_gram_and_fit_gram_agree_on_conditioning() {
        // Both paths must disable the same bins, or refinement would happily wander into a region
        // the coarse search treats as dead.
        let mut planner = Planner::new();
        let cfg = BlockConfig { f_min: 20.0, f_max: 20_000.0, ..BlockConfig::default() };
        let block =
            Block::new(EnvelopeParams::new(251.0, 0.001), SR, &mut planner, &cfg).unwrap();
        let signal = noise(20_000, 99);

        for k in [block.k_lo, block.k_lo + 1, block.k_lo + 3, block.k_hi / 2, block.k_hi] {
            let live = block.gram_inv(k).is_some();
            let fit = score(&signal, &block.env, 0, block.bin_hz(k), cfg.rho_sq_max as f64);
            assert_eq!(live, fit.is_some(), "bin {k}: block says {live}");
        }
    }
}
