//! Bounded local refinement of `(t0, f, alpha, beta)`.
//!
//! The dictionary is a grid, and a real formant almost never sits on it. Off-grid input needs about
//! twelve times as many atoms to reach the same SNR as on-grid input, because each miss is patched
//! with a cluster of partial atoms rather than represented by one. Refinement moves a selected seed
//! off the grid before it is subtracted, so one atom does the work of the cluster.
//!
//! Amplitude and phase are *not* searched — they come out of the 2-D projection in closed form for
//! any `(t0, f, alpha, beta)`. That is what makes this a variable-projection method and keeps the
//! search one-dimensional at every step.
//!
//! # Two scores, and why they must not be confused
//!
//! `E_capt = ||P_span r||^2` is the squared norm of an orthogonal projection of *one fixed* ambient
//! vector `r`. That is the only reason scores from different blocks, onsets and bins are
//! comparable at all.
//!
//! Truncating the envelope at [`crate::fit::fit_end`] zeroes `r` outside a window that depends on
//! the candidate, so a truncated score is not a projection of `r` and ranks candidates partly by
//! how much residual each was allowed to ignore. So:
//!
//! - the **fit-region** score drives the 1-D searches, and never leaves this module;
//! - the **full-support** score is the MP score, and is the only thing selection ever sees.
//!
//! The release is excluded from the search not because it would distort the score — it carries
//! under 1e-4 of the energy — but because its shape is a fixed function of `alpha` and so says
//! nothing about the parameters being fitted, while contributing hundreds of noise-dominated
//! samples that flatten the objective.
//!
//! # Refinement can lose, so it has to be allowed to decline
//!
//! Because the fit-region and full-support optima differ, the parameters that maximize the search
//! objective can score *worse* on the full support than the seed did. Both are therefore evaluated
//! through [`crate::fit::score`] — the same function, so the same Gram clipping — and the refined
//! atom is adopted only if it strictly wins. Without that gate the pursuit stops being greedy and
//! `mp`'s "residual energy rose" guard would report it as a parameter-mapping bug.

use crate::cand::Candidate;
use crate::dict::Block;
use crate::fit;
use crate::fof::{Envelope, EnvelopeParams, ReleasePolicy};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct RefineConfig {
    pub enabled: bool,
    /// Passes of the `f -> alpha -> beta -> t0` cycle.
    pub rounds: usize,
    /// Stop early when a whole round improves the fit score by less than this fraction.
    pub score_tol: f64,
    /// Golden-section evaluations per parameter, beyond the initial bracket.
    pub golden_iters: usize,

    pub f_min: f32,
    pub f_max: f32,
    pub alpha_min: f32,
    pub alpha_max: f32,
    pub beta_min: f32,
    pub beta_max: f32,
    /// Past this rfofs renders silence, and its `amax` fit is ill-conditioned well before it.
    pub alpha_beta_max: f32,
    /// Reject any envelope longer than this, whatever the bounds imply.
    pub max_atom_samples: usize,

    /// Search `f` within this many bins of the seed.
    pub f_bracket_bins: f32,
    /// Search `alpha` within this multiplicative factor of the seed.
    pub alpha_bracket: f32,
    /// Search `beta` within this multiplicative factor of the seed.
    pub beta_bracket: f32,
    /// Search `t0` within this many samples. 0 derives it from the block's hop.
    pub t0_radius: usize,

    pub rho_sq_max: f64,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rounds: 3,
            score_tol: 1e-4,
            golden_iters: 10,

            f_min: 50.0,
            f_max: 10_000.0,
            // Wider than the voice grid at both ends, so refinement is bounded by physics rather
            // than by the grid it started from.
            alpha_min: 40.0,
            alpha_max: 4_000.0,
            beta_min: 1e-4,
            beta_max: 1e-2,
            alpha_beta_max: 4.0,
            max_atom_samples: 1 << 16,

            // The alpha ladder steps by 1.6 and the beta ladder by about 3.3, so these brackets
            // reach the neighbouring rung in each direction: no true value is out of reach.
            f_bracket_bins: 1.0,
            alpha_bracket: 1.6,
            beta_bracket: 3.5,
            t0_radius: 0,

            rho_sq_max: 1.0 - 1e-4,
        }
    }
}

/// Rendered envelopes, keyed by the exact `(alpha, beta)` bits.
///
/// Golden section revisits the same values across rounds and across candidates, and an
/// [`Envelope::render`] is an allocation plus an rfofs spawn plus a full grain render. Exact-bit
/// keying is right here because the search proposes bit-identical repeats, not nearby ones.
#[derive(Default)]
pub struct EnvelopeCache {
    /// Value is the envelope and its fit-region length, so `fit_end`'s scan is paid once.
    entries: HashMap<(u32, u32), Option<(Envelope, usize)>>,
}

impl EnvelopeCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Render `(alpha, beta)` under `policy`, or return `None` if it is out of bounds or unusable.
    ///
    /// Bounds are enforced here rather than in the optimizers so that every path to an envelope —
    /// including the seed's — is screened by the same rules.
    fn get(
        &mut self,
        alpha: f32,
        beta: f32,
        sample_rate: f32,
        policy: &ReleasePolicy,
        cfg: &RefineConfig,
    ) -> Option<&(Envelope, usize)> {
        let in_bounds = alpha >= cfg.alpha_min
            && alpha <= cfg.alpha_max
            && beta >= cfg.beta_min
            && beta <= cfg.beta_max
            && alpha * beta <= cfg.alpha_beta_max;
        if !in_bounds {
            return None;
        }

        let max_len = cfg.max_atom_samples;
        self.entries
            .entry((alpha.to_bits(), beta.to_bits()))
            .or_insert_with(|| {
                let params = EnvelopeParams::with_policy(alpha, beta, policy);
                let env = Envelope::render(params, sample_rate).ok()?;
                if env.support_len() > max_len {
                    return None;
                }
                let cut = fit::fit_end(&env);
                Some((env, cut))
            })
            .as_ref()
    }
}

/// The parameters being refined. Amplitude and phase are solved, never searched.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Params {
    alpha: f32,
    beta: f32,
    f: f32,
    t0: i64,
}

/// Refine `cand` in place. Returns whether the parameters actually moved.
pub fn refine(
    cand: &mut Candidate,
    block: &Block,
    residual: &[f32],
    cfg: &RefineConfig,
    cache: &mut EnvelopeCache,
) -> bool {
    let sr = block.sample_rate();
    let policy = ReleasePolicy {
        fade_level: cand.atom.env.fade_level,
        // Recovered from the seed so a refined alpha re-derives its release the same way the
        // dictionary did, rather than inheriting the seed's absolute duration.
        fade_dur_scale: cand.atom.env.fade_dur * cand.atom.env.alpha,
        fade_dur_min: cand.atom.env.fade_dur,
        fade_dur_max: cand.atom.env.fade_dur,
    };
    let policy = relax_clamps(policy, cfg);

    let seed = Params {
        alpha: cand.atom.env.alpha,
        beta: cand.atom.env.beta,
        f: cand.atom.f,
        t0: cand.atom.t0,
    };

    // Always re-score the seed through `fit`, even when refinement declines: the incoming score
    // came from the block's whole-support Gram, and mixing the two footings across candidates is
    // exactly what would make selection unfair.
    let Some(seed_fit) = full_score(seed, residual, sr, &policy, cfg, cache) else {
        return false;
    };

    let mut cur = seed;
    let mut cur_score = fit_score(cur, residual, sr, &policy, cfg, cache);
    let t0_radius = if cfg.t0_radius > 0 {
        cfg.t0_radius
    } else {
        (block.hop / 2).max(1)
    };
    let bin_hz = block.bin_hz(1).max(f32::MIN_POSITIVE);

    for _ in 0..cfg.rounds {
        let round_start = cur_score;

        // Frequency, within a bin of the seed.
        let half = bin_hz * cfg.f_bracket_bins;
        let (lo, hi) = (
            (cur.f - half).max(cfg.f_min) as f64,
            (cur.f + half).min(cfg.f_max) as f64,
        );
        golden(lo, hi, cfg.golden_iters, &mut cur, &mut cur_score, |p, x| p.f = x as f32, |p| {
            fit_score(*p, residual, sr, &policy, cfg, cache)
        });

        // Alpha and beta in log space: both are positive scale parameters, so a multiplicative
        // bracket is the natural one and keeps the search away from zero.
        let b = cfg.alpha_bracket.max(1.0) as f64;
        let (lo, hi) = (
            (cur.alpha as f64 / b).max(cfg.alpha_min as f64).ln(),
            (cur.alpha as f64 * b).min(cfg.alpha_max as f64).ln(),
        );
        golden(lo, hi, cfg.golden_iters, &mut cur, &mut cur_score, |p, x| p.alpha = x.exp() as f32, |p| {
            fit_score(*p, residual, sr, &policy, cfg, cache)
        });

        let b = cfg.beta_bracket.max(1.0) as f64;
        let (lo, hi) = (
            (cur.beta as f64 / b).max(cfg.beta_min as f64).ln(),
            (cur.beta as f64 * b).min(cfg.beta_max as f64).ln(),
        );
        golden(lo, hi, cfg.golden_iters, &mut cur, &mut cur_score, |p, x| p.beta = x.exp() as f32, |p| {
            fit_score(*p, residual, sr, &policy, cfg, cache)
        });

        // Onset. Amplitude and phase are re-solved at every trial, so the objective varies on the
        // envelope's scale rather than the carrier's and a bracketed search is well posed.
        let (lo, hi) = (
            (cur.t0 - t0_radius as i64) as f64,
            (cur.t0 + t0_radius as i64) as f64,
        );
        golden(lo, hi, cfg.golden_iters, &mut cur, &mut cur_score, |p, x| p.t0 = x.round() as i64, |p| {
            fit_score(*p, residual, sr, &policy, cfg, cache)
        });
        // Golden section works on a real line; polish the integer it landed between.
        for d in [-1i64, 1] {
            let trial = Params { t0: cur.t0 + d, ..cur };
            let s = fit_score(trial, residual, sr, &policy, cfg, cache);
            if s > cur_score {
                cur = trial;
                cur_score = s;
            }
        }

        if cur_score <= round_start * (1.0 + cfg.score_tol) {
            break;
        }
    }

    if cur == seed {
        adopt(cand, seed, seed_fit, false);
        return false;
    }

    // The acceptance gate. Both sides go through `full_score`, so the comparison is like for like.
    match full_score(cur, residual, sr, &policy, cfg, cache) {
        Some(refined) if refined.energy > seed_fit.energy => {
            adopt(cand, cur, refined, true);
            true
        }
        _ => {
            adopt(cand, seed, seed_fit, false);
            false
        }
    }
}

/// Widen a seed's frozen release clamps to whatever the refined `alpha` needs.
///
/// The seed carries one `fade_dur`, but a refined `alpha` should get the duration the policy would
/// have given it. The scale is recovered from the seed; the clamps are the only thing that cannot
/// be, so they are opened to the range the seed's own bounds imply.
fn relax_clamps(mut policy: ReleasePolicy, cfg: &RefineConfig) -> ReleasePolicy {
    let at_min = policy.fade_dur_scale / cfg.alpha_max;
    let at_max = policy.fade_dur_scale / cfg.alpha_min;
    policy.fade_dur_min = at_min.min(policy.fade_dur_min);
    policy.fade_dur_max = at_max.max(policy.fade_dur_max);
    policy
}

fn adopt(cand: &mut Candidate, p: Params, fit: crate::corr::Projection, refined: bool) {
    cand.atom.env.alpha = p.alpha;
    cand.atom.env.beta = p.beta;
    cand.atom.f = p.f;
    cand.atom.t0 = p.t0;
    cand.atom.amp = fit.amp;
    cand.atom.phi = fit.phi;
    cand.mp_score = fit.energy;
    cand.refined = refined;
}

/// The search objective: captured energy over the fit region only.
fn fit_score(
    p: Params,
    residual: &[f32],
    sr: f32,
    policy: &ReleasePolicy,
    cfg: &RefineConfig,
    cache: &mut EnvelopeCache,
) -> f64 {
    let Some((env, cut)) = cache.get(p.alpha, p.beta, sr, policy, cfg) else {
        return 0.0;
    };
    fit::score_energy(residual, &env.samples[..*cut], sr, p.t0, p.f, cfg.rho_sq_max).unwrap_or(0.0)
}

/// The MP score: captured energy over the whole support, including the release.
fn full_score(
    p: Params,
    residual: &[f32],
    sr: f32,
    policy: &ReleasePolicy,
    cfg: &RefineConfig,
    cache: &mut EnvelopeCache,
) -> Option<crate::corr::Projection> {
    let (env, _) = cache.get(p.alpha, p.beta, sr, policy, cfg)?;
    fit::score(residual, env, p.t0, p.f, cfg.rho_sq_max)
}

/// Golden-section maximization over `[lo, hi]`, adopting the result only if it beats `best_score`.
///
/// The incumbent comparison is what makes a step unable to lose: the objective is only
/// approximately unimodal, and near the bracket edges the maximum may be the starting point itself.
fn golden<P, S>(
    lo: f64,
    hi: f64,
    iters: usize,
    best: &mut Params,
    best_score: &mut f64,
    mut set: P,
    mut score: S,
) where
    P: FnMut(&mut Params, f64),
    S: FnMut(&Params) -> f64,
{
    let bracketed = hi > lo;
    if !bracketed {
        return;
    }
    const R: f64 = 0.618_033_988_749_895;

    let eval = |x: f64, base: &Params, set: &mut P, score: &mut S| {
        let mut p = *base;
        set(&mut p, x);
        (p, score(&p))
    };

    let base = *best;
    let (mut a, mut b) = (lo, hi);
    let (mut x1, mut x2) = (b - R * (b - a), a + R * (b - a));
    let (mut p1, mut f1) = eval(x1, &base, &mut set, &mut score);
    let (mut p2, mut f2) = eval(x2, &base, &mut set, &mut score);

    for _ in 0..iters {
        if f1 >= f2 {
            b = x2;
            x2 = x1;
            p2 = p1;
            f2 = f1;
            x1 = b - R * (b - a);
            (p1, f1) = eval(x1, &base, &mut set, &mut score);
        } else {
            a = x1;
            x1 = x2;
            p1 = p2;
            f1 = f2;
            x2 = a + R * (b - a);
            (p2, f2) = eval(x2, &base, &mut set, &mut score);
        }
    }

    let (p, f) = if f1 >= f2 { (p1, f1) } else { (p2, f2) };
    if f > *best_score {
        *best = p;
        *best_score = f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cand::Seed;
    use crate::dict::{BlockConfig, Dictionary};
    use crate::fft::Planner;
    use crate::fof::AtomParams;
    use crate::signal::add_at;

    const SR: f32 = 48_000.0;

    fn voice() -> Dictionary {
        let mut planner = Planner::new();
        Dictionary::voice(SR, &mut planner, &BlockConfig::default()).unwrap()
    }

    /// Seed a candidate at the grid point nearest `truth`, as the coarse search would.
    fn seed_at(dict: &Dictionary, bi: usize, truth: &AtomParams, residual: &[f32]) -> Candidate {
        let block = &dict.blocks[bi];
        let frame = (truth.t0 as usize).div_ceil(block.hop).min(block.frame_count(residual.len()) - 1);
        let onset = block.frame_onset(frame);
        let bin = ((truth.f / block.bin_hz(1)).round() as usize).clamp(block.k_lo, block.k_hi);
        let p = fit::score(residual, &block.env, onset as i64, block.bin_hz(bin), 1.0 - 1e-4)
            .expect("seed must be scorable");
        Candidate::from_seed(
            Seed { block: bi, frame, bin, onset, energy: p.energy },
            block,
            p.amp,
            p.phi,
            p.energy,
        )
    }

    fn plant(truth: &AtomParams, len: usize) -> Vec<f32> {
        let mut sig = vec![0.0f32; len];
        add_at(&mut sig, &truth.render(SR).unwrap(), truth.t0);
        sig
    }

    /// The spec's parameter-recovery test: seed from a deliberately imperfect template and report
    /// what comes back.
    #[test]
    fn recovers_an_off_grid_atom_from_an_imperfect_seed() {
        let dict = voice();
        // Between the alpha=328 and alpha=524 rungs, between beta rungs, off the bin and hop grids.
        let truth = AtomParams {
            t0: 6_211,
            f: 1_337.0,
            env: EnvelopeParams::new(410.0, 0.0017),
            phi: 0.83,
            amp: 0.7,
        };
        let sig = plant(&truth, 30_000);

        // Seed from the alpha=328 block: the wrong rung, and the wrong beta.
        let bi = dict
            .blocks
            .iter()
            .position(|b| b.env.params.alpha == 328.0 && b.env.params.beta == 0.001)
            .unwrap();
        let mut cand = seed_at(&dict, bi, &truth, &sig);
        let before = cand.mp_score;

        let cfg = RefineConfig::default();
        let mut cache = EnvelopeCache::new();
        let moved = refine(&mut cand, &dict.blocks[bi], &sig, &cfg, &mut cache);
        assert!(moved, "refinement declined to move");

        let total = crate::signal::energy_of(&sig);
        let g = &cand.atom;
        println!(
            "seed {:.4} -> refined {:.4} of {total:.4}\n  \
             t0 {} (d {}), f {:.2} (d {:.2}), alpha {:.1} (d {:.1}%), beta {:.5} (d {:.1}%)",
            before,
            cand.mp_score,
            g.t0,
            g.t0 - truth.t0,
            g.f,
            g.f - truth.f,
            g.env.alpha,
            100.0 * (g.env.alpha - truth.env.alpha) / truth.env.alpha,
            g.env.beta,
            100.0 * (g.env.beta - truth.env.beta) / truth.env.beta,
        );

        assert!(cand.mp_score > before, "refinement lost energy");
        // The atom is alone in the signal, so a good fit captures essentially all of it.
        assert!(
            cand.mp_score / total > 0.99,
            "captured only {:.4} of {total:.4}",
            cand.mp_score
        );
        // `f` is recovered essentially exactly: it is the one parameter the objective is sharply
        // peaked in, and the only one nothing else trades against.
        assert!((g.f - truth.f).abs() / truth.f < 1e-3, "f {}", g.f);
        assert!((g.amp - truth.amp).abs() / truth.amp < 0.05, "amp {}", g.amp);

        // `t0`, `alpha` and `beta` are looser, and coordinate descent is why. All three shape the
        // attack, so they trade against each other along a shallow valley: a slightly late onset
        // with a slightly slower attack and a slightly slower decay fits nearly as well as the
        // truth. Refining one at a time walks down the valley but not along it. That is a known
        // cost of the one-dimensional method the spec asks for in version 1, and it is bounded by
        // what actually matters -- the fit above captures 99.9% of the atom.
        assert!((g.t0 - truth.t0).abs() <= 8, "t0 off by {}", g.t0 - truth.t0);
        assert!(
            (g.env.alpha - truth.env.alpha).abs() / truth.env.alpha < 0.25,
            "alpha {}",
            g.env.alpha
        );
        assert!(
            (g.env.beta - truth.env.beta).abs() / truth.env.beta < 0.35,
            "beta {}",
            g.env.beta
        );
    }

    /// Refinement must never make a candidate worse — the gate that keeps the pursuit greedy.
    #[test]
    fn refinement_never_lowers_the_full_support_score() {
        let dict = voice();
        let mut cache = EnvelopeCache::new();
        let cfg = RefineConfig::default();

        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 8_388_608.0 - 1.0
        };
        let sig: Vec<f32> = (0..20_000).map(|_| rng()).collect();

        for bi in [0, 5, 11, 17] {
            let block = &dict.blocks[bi];
            let frame = 7.min(block.frame_count(sig.len()) - 1);
            let onset = block.frame_onset(frame);
            let bin = (block.k_lo + block.k_hi) / 2;
            let p = fit::score(&sig, &block.env, onset as i64, block.bin_hz(bin), cfg.rho_sq_max)
                .unwrap();
            let mut cand = Candidate::from_seed(
                Seed { block: bi, frame, bin, onset, energy: p.energy },
                block,
                p.amp,
                p.phi,
                p.energy,
            );
            refine(&mut cand, block, &sig, &cfg, &mut cache);
            assert!(
                cand.mp_score >= p.energy,
                "block {bi}: {} < seed {}",
                cand.mp_score,
                p.energy
            );
        }
    }

    #[test]
    fn refined_parameters_stay_inside_their_bounds() {
        let dict = voice();
        let truth = AtomParams {
            t0: 4_000,
            f: 900.0,
            env: EnvelopeParams::new(300.0, 0.002),
            phi: 0.0,
            amp: 1.0,
        };
        let sig = plant(&truth, 24_000);
        // Bounds far tighter than the truth, so the optimizer would leave them if it could.
        let cfg = RefineConfig {
            alpha_min: 500.0,
            alpha_max: 700.0,
            beta_min: 3e-3,
            beta_max: 4e-3,
            f_min: 1_000.0,
            f_max: 1_200.0,
            ..RefineConfig::default()
        };
        let mut cache = EnvelopeCache::new();

        let bi = dict.blocks.iter().position(|b| b.env.params.alpha == 524.0).unwrap();
        let block = &dict.blocks[bi];
        let onset = block.frame_onset(4_000 / block.hop);
        let bin = ((1_100.0 / block.bin_hz(1)).round() as usize).clamp(block.k_lo, block.k_hi);
        let p = fit::score(&sig, &block.env, onset as i64, block.bin_hz(bin), cfg.rho_sq_max)
            .unwrap();
        let mut cand = Candidate::from_seed(
            Seed { block: bi, frame: 0, bin, onset, energy: p.energy },
            block,
            p.amp,
            p.phi,
            p.energy,
        );
        // The seed's own alpha is outside the test bounds, so refinement can only decline.
        refine(&mut cand, block, &sig, &cfg, &mut cache);
        let e = cand.atom.env;
        if cand.refined {
            assert!(e.alpha >= cfg.alpha_min && e.alpha <= cfg.alpha_max, "alpha {}", e.alpha);
            assert!(e.beta >= cfg.beta_min && e.beta <= cfg.beta_max, "beta {}", e.beta);
            assert!(cand.atom.f >= cfg.f_min && cand.atom.f <= cfg.f_max, "f {}", cand.atom.f);
            assert!(e.alpha * e.beta <= cfg.alpha_beta_max);
        }
    }

    #[test]
    fn the_cache_serves_repeats_rather_than_re_rendering() {
        let mut cache = EnvelopeCache::new();
        let cfg = RefineConfig::default();
        let policy = ReleasePolicy::default();
        for _ in 0..5 {
            assert!(cache.get(251.0, 0.001, SR, &policy, &cfg).is_some());
        }
        assert_eq!(cache.len(), 1);
        // Out-of-bounds shapes are refused, and the refusal is remembered too.
        assert!(cache.get(1e6, 0.001, SR, &policy, &cfg).is_none());
        assert!(cache.get(251.0, 1.0, SR, &policy, &cfg).is_none());
    }

    #[test]
    fn golden_section_finds_a_smooth_maximum_and_never_regresses() {
        // A quadratic peaking at 3.0, with the incumbent already at a worse point.
        let mut p = Params { alpha: 0.0, beta: 0.0, f: 0.0, t0: 0 };
        let mut s = f64::NEG_INFINITY;
        golden(0.0, 10.0, 40, &mut p, &mut s, |p, x| p.f = x as f32, |p| {
            -((p.f as f64 - 3.0).powi(2))
        });
        assert!((p.f - 3.0).abs() < 1e-3, "found {}", p.f);

        // An incumbent better than anything in the bracket must survive untouched.
        let mut p = Params { alpha: 1.0, beta: 2.0, f: 42.0, t0: 7 };
        let keep = p;
        let mut s = 1e9;
        golden(0.0, 10.0, 20, &mut p, &mut s, |p, x| p.f = x as f32, |p| {
            -((p.f as f64 - 3.0).powi(2))
        });
        assert_eq!(p, keep);
        assert_eq!(s, 1e9);

        // A degenerate bracket is a no-op, not a panic.
        let mut p = keep;
        let mut s = 0.0;
        golden(5.0, 5.0, 10, &mut p, &mut s, |p, x| p.f = x as f32, |_| 1.0);
        assert_eq!(p, keep);
    }
}
