//! Bounded local refinement of `(t0, f, alpha, beta)` for a FOF and `(t0, f, sigma)` for a Gaussian.
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
//!
//! # A Gaussian's width is searched about its centre
//!
//! `t0` is the first sample of the support for both kinds, but a Gaussian's support is symmetric
//! about a peak `half_len(sigma)` samples later. Changing `sigma` at a fixed `t0` would move that
//! peak by about 3.7 samples per sample of `sigma`, so the width search would really be a coupled
//! width-and-onset search, badly conditioned in both. The `sigma` stage therefore holds the *centre*
//! fixed and moves `t0` with it; the onset stage that follows is then a pure shift again.
//!
//! # The search scores several trials at once, and finds what a serial one would
//!
//! Golden section is sequential, but its next few probes are known before any is scored: one per
//! path of comparison outcomes. `golden` scores that tree as a batch and follows the path the
//! comparisons actually take, and `Objective` splits scoring from committing so the envelope cache's
//! insertions and clears still happen in the serial order, for the serial search's probes only. The
//! result is bit-identical at every width. An envelope shorter than `PARALLEL_SUPPORT` runs the
//! plain serial search, which is also the reference the tests hold every wider search to.

use crate::atom::Shape;
use crate::cand::Candidate;
use crate::dict::Block;
use crate::fit;
use crate::fof::{Envelope, EnvelopeParams, ReleasePolicy};
use crate::gauss::GaussianParams;
use rayon::prelude::*;
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
    /// Bounds on a refined Gaussian's `sigma`, seconds.
    pub sigma_min: f32,
    pub sigma_max: f32,
    /// Reject any envelope longer than this, whatever the bounds imply.
    pub max_atom_samples: usize,

    /// Search `f` within this many bins of the seed.
    pub f_bracket_bins: f32,
    /// Search `alpha` within this multiplicative factor of the seed.
    pub alpha_bracket: f32,
    /// Search `beta` within this multiplicative factor of the seed.
    pub beta_bracket: f32,
    /// Search `sigma` within this multiplicative factor of the seed.
    pub sigma_bracket: f32,
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
            sigma_min: 0.5e-3,
            sigma_max: 0.2,
            max_atom_samples: 1 << 16,

            // The alpha ladder steps by 1.6 and the beta ladder by about 3.3, so these brackets
            // reach the neighbouring rung in each direction: no true value is out of reach. A
            // sigma ladder of about 2.5 is what the manual suggests, so its bracket matches.
            f_bracket_bins: 1.0,
            alpha_bracket: 1.6,
            beta_bracket: 3.5,
            sigma_bracket: 2.5,
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
///
/// **The reuse is within one atom's search, not across atoms, and the lifetime must match.**
/// Refinement moves `(alpha, beta)` continuously, so the next atom's probes are essentially never
/// bit-identical to this one's: the cross-atom hit rate is nil, but the entries are retained
/// anyway. Each holds a full envelope — at a low `alpha_min` that is hundreds of KB — and a search
/// proposes tens of them, so an unbounded cache grows without limit in *atom count*. Measured on
/// `lux-eterna-1.toml`, which reaches `alpha_min = 4`, it cost about 7 MB per selected atom: 2.3 GB
/// at 300 atoms, and the config's own `max_atoms = 7500` would have needed ~50 GB. That, not the
/// frame tables, is what made long clips impossible.
///
/// So the cache carries a byte budget and clears itself whole when it is exceeded. Clearing is
/// always safe — this is a pure memo, and [`get`](Self::get) recomputes exactly what it evicted —
/// which is why the bound can be crude. Clearing whole rather than evicting one entry keeps the
/// *current* search's working set intact in the common case, where a whole search fits.
#[derive(Default)]
pub struct EnvelopeCache {
    /// Value is the envelope and its fit-region length, so `fit_end`'s scan is paid once.
    entries: HashMap<(u8, u32, u32), Option<(Envelope, usize)>>,
    /// Envelope samples currently held, and the ceiling on them.
    samples: usize,
    budget: usize,
    /// The high-water mark over the cache's life, for `RMP_REFRESH_DETAIL`.
    peak_samples: usize,
    /// Times the cache has been cleared. Entries leave only by a clear and are never overwritten,
    /// so an unchanged count proves an entry read earlier is still the one a lookup returns.
    clears: u64,
}

/// Envelope samples the cache may hold before it clears: 64 MB of `f32`.
///
/// Sized to hold a whole refinement search — the widest `alpha_bracket` over the lowest `alpha_min`
/// in `data/config` is a few hundred envelopes of a few hundred KB — while bounding the total
/// independently of how many atoms the pursuit selects.
pub const ENVELOPE_CACHE_SAMPLES: usize = 16 << 20;

impl EnvelopeCache {
    pub fn new() -> Self {
        Self::with_budget(ENVELOPE_CACHE_SAMPLES)
    }

    pub fn with_budget(budget: usize) -> Self {
        Self { budget, ..Self::default() }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Envelope samples held, and the most ever held at once.
    pub fn usage(&self) -> (usize, usize) {
        (self.samples, self.peak_samples)
    }

    /// Drop everything. Safe at any point: the cache is a memo, not state.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.samples = 0;
        self.clears += 1;
    }

    /// Render `shape`, or return `None` if it is out of bounds or unusable.
    ///
    /// Bounds are enforced here rather than in the optimizers so that every path to an envelope —
    /// including the seed's — is screened by the same rules.
    ///
    /// Keyed on [`Shape::cache_key`], which for a FOF is `(alpha, beta)` alone. That is the key
    /// this cache always had, and keeping it exactly is what keeps a FOF decomposition bit-identical:
    /// a hit serves whatever release the first request rendered.
    fn get(
        &mut self,
        shape: Shape,
        sample_rate: f32,
        cfg: &RefineConfig,
    ) -> Option<&(Envelope, usize)> {
        if !in_bounds(shape, cfg) {
            return None;
        }
        let key = shape.cache_key();
        if !self.entries.contains_key(&key) {
            self.insert(key, render_entry(shape, sample_rate, cfg.max_atom_samples));
        }
        self.entries[&key].as_ref()
    }

    /// Store a render of a key the cache does not hold.
    ///
    /// The budget is enforced *before* the insertion, so the entry just rendered always survives
    /// it: a caller that asked for an envelope gets one back, whatever the ceiling.
    fn insert(&mut self, key: (u8, u32, u32), value: Option<(Envelope, usize)>) {
        let cost = value.as_ref().map_or(0, |(env, _)| env.support_len());
        if self.samples + cost > self.budget {
            self.clear();
        }
        self.samples += cost;
        self.peak_samples = self.peak_samples.max(self.samples);
        self.entries.insert(key, value);
    }
}

/// Whether refinement may consider `shape` at all.
fn in_bounds(shape: Shape, cfg: &RefineConfig) -> bool {
    match shape {
        Shape::Fof(p) => {
            p.alpha >= cfg.alpha_min
                && p.alpha <= cfg.alpha_max
                && p.beta >= cfg.beta_min
                && p.beta <= cfg.beta_max
                && p.alpha * p.beta <= cfg.alpha_beta_max
        }
        Shape::Gaussian(g) => g.sigma >= cfg.sigma_min && g.sigma <= cfg.sigma_max,
    }
}

/// What the cache stores for a key it did not hold: the envelope and its fit-region length, or
/// `None` for a shape that cannot be rendered or is longer than `max_len`.
///
/// A pure function of its arguments, which is what lets a speculative search render on any thread
/// and still hand the cache exactly what [`EnvelopeCache::get`] would have rendered.
fn render_entry(shape: Shape, sample_rate: f32, max_len: usize) -> Option<(Envelope, usize)> {
    let env = Envelope::render(shape, sample_rate).ok()?;
    if env.support_len() > max_len {
        return None;
    }
    let cut = fit::fit_end(&env);
    Some((env, cut))
}

/// The shape parameters being searched, per kind.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    Fof { alpha: f32, beta: f32 },
    Gaussian { sigma: f32 },
}

/// The parameters being refined. Amplitude and phase are solved, never searched.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Params {
    form: Form,
    f: f32,
    t0: i64,
}

/// What a refined shape inherits from its seed rather than searching: a FOF's release policy, a
/// Gaussian's cutoff. A refined atom never changes kind.
#[derive(Clone, Copy, Debug)]
enum Family {
    Fof(ReleasePolicy),
    Gaussian { cutoff_level: f32 },
}

impl Family {
    fn shape(&self, form: Form) -> Shape {
        match (self, form) {
            (Self::Fof(policy), Form::Fof { alpha, beta }) => {
                EnvelopeParams::with_policy(alpha, beta, policy).into()
            }
            (Self::Gaussian { cutoff_level }, Form::Gaussian { sigma }) => {
                GaussianParams { sigma, cutoff_level: *cutoff_level }.into()
            }
            _ => unreachable!("refinement never changes an atom's kind"),
        }
    }
}

/// Envelope samples below which a refinement scores its trials one at a time.
///
/// A rayon dispatch costs 15–20 µs and a trial on a 2000-sample envelope about 7, so below here a
/// batch spends more waking the pool than it saves. Anywhere from 1024 to 4096 measured the same
/// to within noise; the top of that range wastes the least speculative work. Where the line sits
/// cannot change a result — see [`golden`] — only how long one takes.
const PARALLEL_SUPPORT: usize = 1 << 12;

/// The most trials one speculative batch may score.
///
/// A batch of `2^k - 1` trials settles `k` golden-section steps, so doubling the width buys one
/// step per batch; 16 is the width at which the first batch, which carries both interior points,
/// also settles four. Wider is no faster, since a batch waits for its slowest trial and past the
/// machine's fast cores the extra trials land on slow ones or share a core.
const MAX_WIDTH: usize = 16;

/// Refine `cand` in place. Returns whether the parameters actually moved.
pub fn refine(
    cand: &mut Candidate,
    block: &Block,
    residual: &[f32],
    cfg: &RefineConfig,
    cache: &mut EnvelopeCache,
) -> bool {
    let width = if block.support_len() >= PARALLEL_SUPPORT {
        rayon::current_num_threads().clamp(1, MAX_WIDTH)
    } else {
        1
    };
    refine_at_width(cand, block, residual, cfg, cache, width)
}

/// [`refine`], scoring up to `width` trials at once. The result does not depend on `width`.
fn refine_at_width(
    cand: &mut Candidate,
    block: &Block,
    residual: &[f32],
    cfg: &RefineConfig,
    cache: &mut EnvelopeCache,
    width: usize,
) -> bool {
    let sr = block.sample_rate();
    let (family, form) = match cand.atom.env {
        Shape::Fof(env) => {
            let policy = ReleasePolicy {
                fade_level: env.fade_level,
                // Recovered from the seed so a refined alpha re-derives its release the same way
                // the dictionary did, rather than inheriting the seed's absolute duration.
                fade_dur_scale: env.fade_dur * env.alpha,
                fade_dur_min: env.fade_dur,
                fade_dur_max: env.fade_dur,
            };
            (
                Family::Fof(relax_clamps(policy, cfg)),
                Form::Fof { alpha: env.alpha, beta: env.beta },
            )
        }
        Shape::Gaussian(g) => (
            Family::Gaussian { cutoff_level: g.cutoff_level },
            Form::Gaussian { sigma: g.sigma },
        ),
    };
    let family = &family;
    let search = Search { residual, sr, family, cfg, width };

    let seed = Params { form, f: cand.atom.f, t0: cand.atom.t0 };

    // Always re-score the seed through `fit`, even when refinement declines: the incoming score
    // came from the block's whole-support Gram, and mixing the two footings across candidates is
    // exactly what would make selection unfair.
    let Some(seed_fit) = full_score(seed, residual, sr, family, cfg, cache) else {
        return false;
    };

    let mut cur = seed;
    let mut cur_score = fit_score(cur, residual, sr, family, cfg, cache);
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
        golden(lo, hi, cfg.golden_iters, &mut cur, &mut cur_score, |p, x| p.f = x as f32, &mut Trials {
            search,
            cache,
            gram: None,
        });

        // The shape. Every scale parameter is searched in log space: they are all positive, so a
        // multiplicative bracket is the natural one and keeps the search away from zero.
        match *family {
            Family::Fof(_) => {
                let Form::Fof { alpha, .. } = cur.form else { unreachable!() };
                let b = cfg.alpha_bracket.max(1.0) as f64;
                let (lo, hi) = (
                    (alpha as f64 / b).max(cfg.alpha_min as f64).ln(),
                    (alpha as f64 * b).min(cfg.alpha_max as f64).ln(),
                );
                let set = |p: &mut Params, x: f64| {
                    if let Form::Fof { alpha, .. } = &mut p.form {
                        *alpha = x.exp() as f32;
                    }
                };
                golden(lo, hi, cfg.golden_iters, &mut cur, &mut cur_score, set, &mut Trials {
                    search,
                    cache,
                    gram: None,
                });

                let Form::Fof { beta, .. } = cur.form else { unreachable!() };
                let b = cfg.beta_bracket.max(1.0) as f64;
                let (lo, hi) = (
                    (beta as f64 / b).max(cfg.beta_min as f64).ln(),
                    (beta as f64 * b).min(cfg.beta_max as f64).ln(),
                );
                let set = |p: &mut Params, x: f64| {
                    if let Form::Fof { beta, .. } = &mut p.form {
                        *beta = x.exp() as f32;
                    }
                };
                golden(lo, hi, cfg.golden_iters, &mut cur, &mut cur_score, set, &mut Trials {
                    search,
                    cache,
                    gram: None,
                });
            }
            Family::Gaussian { cutoff_level } => {
                let Form::Gaussian { sigma } = cur.form else { unreachable!() };
                let half = |s: f32| GaussianParams { sigma: s, cutoff_level }.half_len(sr) as i64;
                let b = cfg.sigma_bracket.max(1.0) as f64;
                let (lo, hi) = (
                    (sigma as f64 / b).max(cfg.sigma_min as f64).ln(),
                    (sigma as f64 * b).min(cfg.sigma_max as f64).ln(),
                );
                // Centre held fixed: see the module docs.
                let set = |p: &mut Params, x: f64| {
                    if let Form::Gaussian { sigma } = &mut p.form {
                        let centre = p.t0 + half(*sigma);
                        *sigma = x.exp() as f32;
                        p.t0 = centre - half(*sigma);
                    }
                };
                golden(lo, hi, cfg.golden_iters, &mut cur, &mut cur_score, set, &mut Trials {
                    search,
                    cache,
                    gram: None,
                });
            }
        }

        // Onset. Amplitude and phase are re-solved at every trial, so the objective varies on the
        // envelope's scale rather than the carrier's and a bracketed search is well posed.
        //
        // This is the one stage that holds both the envelope and the carrier fixed, so `G` is the
        // same for all dozen-odd trials. Computing it once turns each of them into the data half of
        // the accumulation alone, a little under half the per-sample work.
        let t0_gram = {
            let omega = std::f64::consts::TAU * cur.f as f64 / sr as f64;
            cache
                .get(family.shape(cur.form), sr, cfg)
                .map(|(env, cut)| fit::gram(&env.samples[..*cut], omega))
        };
        let mut onset = Trials { search, cache, gram: t0_gram };
        let (lo, hi) = (
            (cur.t0 - t0_radius as i64) as f64,
            (cur.t0 + t0_radius as i64) as f64,
        );
        golden(lo, hi, cfg.golden_iters, &mut cur, &mut cur_score, |p, x| p.t0 = x.round() as i64, &mut onset);

        // Golden section works on a real line; polish the integer it landed between. One at a time
        // that is `t0 - 1`, then `+ 1` from wherever the first left it — which is back at `t0` if
        // `t0 - 1` won — so all three are scored together and committed in that order.
        let near = [Params { t0: cur.t0 - 1, ..cur }, Params { t0: cur.t0 + 1, ..cur }, cur];
        let mut batch = onset.speculate(&near);
        let mut second = 1;
        let s = onset.commit(&mut batch, &near, 0);
        if s > cur_score {
            (cur, cur_score, second) = (near[0], s, 2);
        }
        let s = onset.commit(&mut batch, &near, second);
        if s > cur_score {
            (cur, cur_score) = (near[second], s);
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
    match full_score(cur, residual, sr, family, cfg, cache) {
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

/// Write `p` back into the candidate.
///
/// Only the searched parameters are written. A FOF keeps its seed's `fade_dur` rather than the one
/// the policy derives for the refined `alpha` — that is what this has always done, and changing it
/// would move every refined book.
fn adopt(cand: &mut Candidate, p: Params, fit: crate::corr::Projection, refined: bool) {
    match (&mut cand.atom.env, p.form) {
        (Shape::Fof(env), Form::Fof { alpha, beta }) => {
            env.alpha = alpha;
            env.beta = beta;
        }
        (Shape::Gaussian(g), Form::Gaussian { sigma }) => g.sigma = sigma,
        _ => unreachable!("refinement never changes an atom's kind"),
    }
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
    family: &Family,
    cfg: &RefineConfig,
    cache: &mut EnvelopeCache,
) -> f64 {
    fit_score_with(p, residual, sr, family, cfg, cache, None)
}

/// [`fit_score`] reusing a Gram already computed for this envelope and carrier.
///
/// Only the onset sweep can supply one: it is the single stage where both the envelope and the
/// carrier are held fixed, so `G` cannot have changed between trials.
fn fit_score_with(
    p: Params,
    residual: &[f32],
    sr: f32,
    family: &Family,
    cfg: &RefineConfig,
    cache: &mut EnvelopeCache,
    gram: Option<fit::Gram>,
) -> f64 {
    let Some((env, cut)) = cache.get(family.shape(p.form), sr, cfg) else {
        return 0.0;
    };
    region_score(p, residual, &env.samples[..*cut], sr, cfg, gram)
}

/// The fit-region score of `p` over an envelope already cut to its fit region.
///
/// The one definition, shared by the one-at-a-time path and the speculative one, so that a score
/// computed ahead on another thread is the score the serial search would have computed.
fn region_score(
    p: Params,
    residual: &[f32],
    region: &[f32],
    sr: f32,
    cfg: &RefineConfig,
    gram: Option<fit::Gram>,
) -> f64 {
    let omega = std::f64::consts::TAU * p.f as f64 / sr as f64;
    fit::accumulate_with(residual, region, p.t0, omega, gram)
        .and_then(|q| q.energy(cfg.rho_sq_max))
        .unwrap_or(0.0)
}

/// The MP score: captured energy over the whole support, including the release.
fn full_score(
    p: Params,
    residual: &[f32],
    sr: f32,
    family: &Family,
    cfg: &RefineConfig,
    cache: &mut EnvelopeCache,
) -> Option<crate::corr::Projection> {
    let (env, _) = cache.get(family.shape(p.form), sr, cfg)?;
    fit::score(residual, env, p.t0, p.f, cfg.rho_sq_max)
}

/// A score golden section can evaluate many trials of at once.
///
/// Scoring and committing are separate so that a batch can be scored in any order and on any
/// thread, while whatever scoring *changes* — the envelope cache's insertions and clears — still
/// happens in the order a one-at-a-time search would have made it, and only for the trials that
/// search would actually have made.
trait Objective {
    type Batch;
    /// Trials one batch may hold. 1 is the one-at-a-time search.
    fn width(&self) -> usize;
    /// Score every trial, changing nothing.
    fn speculate(&self, trials: &[Params]) -> Self::Batch;
    /// The score a one-at-a-time search would have seen for `trials[i]`, making the changes to
    /// shared state it would have made. Called in that search's order.
    fn commit(&mut self, batch: &mut Self::Batch, trials: &[Params], i: usize) -> f64;
}

/// Everything a trial's score depends on besides its parameters and the cache.
#[derive(Clone, Copy)]
struct Search<'a> {
    residual: &'a [f32],
    sr: f32,
    family: &'a Family,
    cfg: &'a RefineConfig,
    width: usize,
}

/// One search stage's objective: [`fit_score_with`] at a fixed Gram, or none.
struct Trials<'a, 'c> {
    search: Search<'a>,
    cache: &'c mut EnvelopeCache,
    gram: Option<fit::Gram>,
}

/// Where a speculative score's envelope came from, which decides whether it can be trusted.
enum Source {
    /// Out of bounds: scores zero, and the cache never sees it.
    Refused,
    /// The entry the cache held when the batch was scored.
    Held,
    /// The trial's own render of a key the cache did not hold — what a lookup would have inserted.
    Rendered(Option<(Envelope, usize)>),
}

struct Batch {
    scores: Vec<f64>,
    /// Emptied to `Refused` as each trial is committed, which hands a render over to the cache.
    sources: Vec<Source>,
    /// The cache's clear count when the batch was scored.
    clears: u64,
}

impl Objective for Trials<'_, '_> {
    type Batch = Batch;

    fn width(&self) -> usize {
        self.search.width
    }

    fn speculate(&self, trials: &[Params]) -> Batch {
        let Search { residual, sr, family, cfg, width } = self.search;
        let cache = &*self.cache;
        if width <= 1 {
            // One at a time, each trial is scored as it is committed — the plain serial search,
            // which is also the reference every wider search is held to.
            return Batch { scores: Vec::new(), sources: Vec::new(), clears: cache.clears };
        }
        let score = |p: Params, entry: Option<&(Envelope, usize)>| {
            entry.map_or(0.0, |(env, cut)| {
                region_score(p, residual, &env.samples[..*cut], sr, cfg, self.gram)
            })
        };

        // Render and score in one task per trial, so a batch waits for its slowest trial once
        // rather than once per phase. Two trials of one key the cache lacks would both render it;
        // within a batch only a clear can bring that about, and it costs time, not correctness.
        let (scores, sources) = spread(trials.len(), |i| {
            let p = trials[i];
            let shape = family.shape(p.form);
            if !in_bounds(shape, cfg) {
                return (0.0, Source::Refused);
            }
            match cache.entries.get(&shape.cache_key()) {
                Some(entry) => (score(p, entry.as_ref()), Source::Held),
                None => {
                    let value = render_entry(shape, sr, cfg.max_atom_samples);
                    (score(p, value.as_ref()), Source::Rendered(value))
                }
            }
        })
        .into_iter()
        .unzip();

        Batch { scores, sources, clears: cache.clears }
    }

    fn commit(&mut self, batch: &mut Batch, trials: &[Params], i: usize) -> f64 {
        let Search { residual, sr, family, cfg, width } = self.search;
        if width <= 1 {
            return fit_score_with(trials[i], residual, sr, family, cfg, self.cache, self.gram);
        }
        let key = family.shape(trials[i].form).cache_key();
        let exact = match std::mem::replace(&mut batch.sources[i], Source::Refused) {
            Source::Refused => true,
            // Entries leave only by a clear and are never overwritten, so with no clear since the
            // batch was scored this is still the entry a lookup returns. After one it may not be:
            // a FOF key leaves out the release, and a re-render need not match what was held.
            Source::Held => self.cache.clears == batch.clears,
            // Either still missing, and this render is what the lookup would insert, or inserted
            // since by an earlier trial of this batch — a render of this same shape either way,
            // because a key is `Held` for every trial of a batch or for none.
            Source::Rendered(value) => {
                if !self.cache.entries.contains_key(&key) {
                    self.cache.insert(key, value);
                }
                true
            }
        };
        if exact {
            return batch.scores[i];
        }
        fit_score_with(trials[i], residual, sr, family, cfg, self.cache, self.gram)
    }
}

/// `(0..n).map(f)` across the pool, in order.
fn spread<R: Send>(n: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    (0..n).into_par_iter().with_max_len(1).map(f).collect()
}

/// A golden-section bracket `[a, b]` and its two interior points.
#[derive(Clone, Copy, Debug)]
struct Section {
    a: f64,
    b: f64,
    x1: f64,
    x2: f64,
}

impl Section {
    const R: f64 = 0.618_033_988_749_895;

    fn new(a: f64, b: f64) -> Self {
        Self { a, b, x1: b - Self::R * (b - a), x2: a + Self::R * (b - a) }
    }

    /// Shrink toward `a` when `left` — `x1` scored at least as well as `x2` — keeping `x1` as the
    /// new `x2`; otherwise toward `b`, keeping `x2` as the new `x1`.
    fn step(self, left: bool) -> Self {
        if left {
            let b = self.x2;
            Self { a: self.a, b, x1: b - Self::R * (b - self.a), x2: self.x1 }
        } else {
            let a = self.x1;
            Self { a, b: self.b, x1: self.x2, x2: a + Self::R * (self.b - a) }
        }
    }
}

/// A probe planned before the comparisons that lead to it were made.
#[derive(Clone, Copy, Debug)]
struct Probe {
    /// The bracket after the step that placed this probe.
    sec: Section,
    /// Which way that step went, and so whether the probe is the new `x1` or the new `x2`.
    left: bool,
    /// The probes after it, indexed by the next comparison's outcome, `f1 >= f2`.
    next: [usize; 2],
}

impl Probe {
    fn x(&self) -> f64 {
        if self.left { self.sec.x1 } else { self.sec.x2 }
    }
}

/// Plan `levels` steps from `sec`, the first going `left`. Returns the first probe's index.
fn plan(sec: Section, left: bool, levels: usize, probes: &mut Vec<Probe>) -> usize {
    let sec = sec.step(left);
    let i = probes.len();
    probes.push(Probe { sec, left, next: [usize::MAX; 2] });
    if levels > 1 {
        let right = plan(sec, false, levels - 1, probes);
        let left = plan(sec, true, levels - 1, probes);
        probes[i].next = [right, left];
    }
    i
}

/// Golden-section maximization over `[lo, hi]`, adopting the result only if it beats `best_score`.
///
/// The incumbent comparison is what makes a step unable to lose: the objective is only
/// approximately unimodal, and near the bracket edges the maximum may be the starting point itself.
///
/// **Speculative, and exactly the one-at-a-time search.** Each step's probe is fixed by the bracket
/// and one comparison, so every probe of the next `k` steps is known in advance — `2^k - 1` of them,
/// one per path of outcomes — and a batch that scores them all at once settles `k` steps. The walk
/// then follows the outcomes that actually occur and commits only the probes on that path, in step
/// order. The points probed and the comparisons made are the serial search's, so the result is too,
/// bit for bit, whatever the width: speculation decides how much is computed, never what is found.
fn golden<P, O>(
    lo: f64,
    hi: f64,
    iters: usize,
    best: &mut Params,
    best_score: &mut f64,
    set: P,
    objective: &mut O,
) where
    P: Fn(&mut Params, f64),
    O: Objective,
{
    let bracketed = hi > lo;
    if !bracketed {
        return;
    }

    let base = *best;
    let at = |x: f64| {
        let mut p = base;
        set(&mut p, x);
        p
    };
    let width = objective.width().max(1);

    let mut sec = Section::new(lo, hi);
    let (mut p1, mut p2) = (at(sec.x1), at(sec.x2));
    let (mut f1, mut f2) = (0.0, 0.0);
    let mut done = 0;
    let mut first = true;
    let mut probes = Vec::new();
    loop {
        // The first batch carries both interior points, and no comparison has been made, so it
        // plans below both outcomes; every later batch plans below the one outcome in hand.
        let room = if first { width.saturating_sub(2) / 2 } else { width };
        let levels = ((room + 1).ilog2() as usize).min(iters - done);
        probes.clear();
        let mut roots = [usize::MAX; 2];
        for left in [false, true] {
            if levels > 0 && (first || (f1 >= f2) == left) {
                roots[left as usize] = plan(sec, left, levels, &mut probes);
            }
        }

        let mut trials = Vec::with_capacity(2 + probes.len());
        if first {
            trials.extend([p1, p2]);
        }
        let offset = trials.len();
        trials.extend(probes.iter().map(|q| at(q.x())));

        let mut batch = objective.speculate(&trials);
        if first {
            f1 = objective.commit(&mut batch, &trials, 0);
            f2 = objective.commit(&mut batch, &trials, 1);
            first = false;
        }
        let mut i = roots[(f1 >= f2) as usize];
        for level in 0..levels {
            let q = probes[i];
            let (p, f) = (trials[offset + i], objective.commit(&mut batch, &trials, offset + i));
            if q.left {
                (p2, f2) = (p1, f1);
                (p1, f1) = (p, f);
            } else {
                (p1, f1) = (p2, f2);
                (p2, f2) = (p, f);
            }
            sec = q.sec;
            done += 1;
            if level + 1 < levels {
                i = q.next[(f1 >= f2) as usize];
            }
        }
        if done >= iters {
            break;
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
        let truth_env = EnvelopeParams::new(410.0, 0.0017);
        let truth = AtomParams {
            t0: 6_211,
            f: 1_337.0,
            env: truth_env.into(),
            phi: 0.83,
            amp: 0.7,
        };
        let sig = plant(&truth, 30_000);

        // Seed from the alpha=328 block: the wrong rung, and the wrong beta.
        let bi = dict
            .blocks
            .iter()
            .position(|b| b.env.params == EnvelopeParams::new(328.0, 0.001).into())
            .unwrap();
        let mut cand = seed_at(&dict, bi, &truth, &sig);
        let before = cand.mp_score;

        let cfg = RefineConfig::default();
        let mut cache = EnvelopeCache::new();
        let moved = refine(&mut cand, &dict.blocks[bi], &sig, &cfg, &mut cache);
        assert!(moved, "refinement declined to move");

        let total = crate::signal::energy_of(&sig);
        let g = &cand.atom;
        let (ge, te) = (g.env.as_fof().unwrap(), truth_env);
        println!(
            "seed {:.4} -> refined {:.4} of {total:.4}\n  \
             t0 {} (d {}), f {:.2} (d {:.2}), alpha {:.1} (d {:.1}%), beta {:.5} (d {:.1}%)",
            before,
            cand.mp_score,
            g.t0,
            g.t0 - truth.t0,
            g.f,
            g.f - truth.f,
            ge.alpha,
            100.0 * (ge.alpha - te.alpha) / te.alpha,
            ge.beta,
            100.0 * (ge.beta - te.beta) / te.beta,
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
        assert!((ge.alpha - te.alpha).abs() / te.alpha < 0.25, "alpha {}", ge.alpha);
        assert!((ge.beta - te.beta).abs() / te.beta < 0.35, "beta {}", ge.beta);
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
            env: EnvelopeParams::new(300.0, 0.002).into(),
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

        let bi = dict
            .blocks
            .iter()
            .position(|b| b.env.params.as_fof().unwrap().alpha == 524.0)
            .unwrap();
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
        let e = cand.atom.env.as_fof().unwrap();
        if cand.refined {
            assert!(e.alpha >= cfg.alpha_min && e.alpha <= cfg.alpha_max, "alpha {}", e.alpha);
            assert!(e.beta >= cfg.beta_min && e.beta <= cfg.beta_max, "beta {}", e.beta);
            assert!(cand.atom.f >= cfg.f_min && cand.atom.f <= cfg.f_max, "f {}", cand.atom.f);
            assert!(e.alpha * e.beta <= cfg.alpha_beta_max);
        }
    }

    // ── gaussian atoms ──────────────────────────────────────────────────────────────────────────

    /// A 2.5-ratio sigma ladder, matching the default `sigma_bracket`.
    fn gaussians() -> Dictionary {
        let mut planner = Planner::new();
        let shapes: Vec<Shape> =
            [0.0025f32, 0.006, 0.015].iter().map(|&s| GaussianParams::new(s).into()).collect();
        Dictionary::from_shapes(&shapes, SR, &mut planner, &BlockConfig::default()).unwrap()
    }

    /// The parameter-recovery test for a Gaussian: seed from the wrong width, and get back the
    /// width, the frequency and — because the sigma stage holds it fixed — the centre.
    #[test]
    fn recovers_an_off_grid_gaussian_about_its_centre() {
        let dict = gaussians();
        let truth_env = GaussianParams::new(0.0041);
        let truth = AtomParams { t0: 6_211, f: 1_337.0, env: truth_env.into(), phi: 0.83, amp: 0.7 };
        let sig = plant(&truth, 30_000);
        let centre = truth.t0 + truth_env.half_len(SR) as i64;

        // Seed from the 2.5 ms rung, centred on the nearest frame — where the coarse search's argmax
        // puts a symmetric atom, rather than where its support would have to start.
        let bi = 0;
        let block = &dict.blocks[bi];
        let h_seed = block.env.params.as_gaussian().unwrap().half_len(SR) as i64;
        let frame = ((centre - h_seed) as usize).div_ceil(block.hop);
        let onset = block.frame_onset(frame);
        let bin = ((truth.f / block.bin_hz(1)).round() as usize).clamp(block.k_lo, block.k_hi);
        let p = fit::score(&sig, &block.env, onset as i64, block.bin_hz(bin), 1.0 - 1e-4).unwrap();
        let mut cand = Candidate::from_seed(
            Seed { block: bi, frame, bin, onset, energy: p.energy },
            block,
            p.amp,
            p.phi,
            p.energy,
        );
        let before = cand.mp_score;

        let moved = refine(&mut cand, block, &sig, &RefineConfig::default(), &mut EnvelopeCache::new());
        assert!(moved, "refinement declined to move");

        let g = cand.atom.env.as_gaussian().expect("refinement changed the atom's kind");
        let got_centre = cand.atom.t0 + g.half_len(SR) as i64;
        let total = crate::signal::energy_of(&sig);
        println!(
            "seed {before:.4} -> refined {:.4} of {total:.4}: sigma {:.3} ms, f {:.2}, centre d {}",
            cand.mp_score,
            g.sigma * 1e3,
            cand.atom.f,
            got_centre - centre
        );
        assert!(cand.mp_score > before, "refinement lost energy");
        assert!(cand.mp_score / total > 0.99, "captured only {:.4} of {total:.4}", cand.mp_score);
        assert!((cand.atom.f - truth.f).abs() / truth.f < 1e-3, "f {}", cand.atom.f);
        assert!((g.sigma - truth_env.sigma).abs() / truth_env.sigma < 0.05, "sigma {}", g.sigma);
        assert!((got_centre - centre).abs() <= 4, "centre off by {}", got_centre - centre);
        assert!((cand.atom.amp - truth.amp).abs() / truth.amp < 0.05, "amp {}", cand.atom.amp);
        assert_eq!(g.cutoff_level, truth_env.cutoff_level, "the cutoff is inherited, not searched");
    }

    /// The acceptance gate and the bounds, for the Gaussian arm.
    #[test]
    fn gaussian_refinement_never_lowers_the_score_and_stays_in_bounds() {
        let dict = gaussians();
        let mut cache = EnvelopeCache::new();
        // The widest rung is outside these bounds, so that seed can only decline.
        let cfg = RefineConfig { sigma_min: 0.002, sigma_max: 0.009, ..RefineConfig::default() };

        let mut s = 0x1234_5678_9abc_def1u64;
        let sig: Vec<f32> = (0..20_000)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / 8_388_608.0 - 1.0
            })
            .collect();

        for (bi, block) in dict.blocks.iter().enumerate() {
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
            assert!(cand.mp_score >= p.energy, "block {bi}: {} < seed {}", cand.mp_score, p.energy);
            if cand.refined {
                let sigma = cand.atom.env.as_gaussian().unwrap().sigma;
                assert!((cfg.sigma_min..=cfg.sigma_max).contains(&sigma), "block {bi}: sigma {sigma}");
            }
        }
    }

    #[test]
    fn the_cache_serves_repeats_rather_than_re_rendering() {
        let mut cache = EnvelopeCache::new();
        let cfg = RefineConfig::default();
        let policy = ReleasePolicy::default();
        let fof = |a, b| EnvelopeParams::with_policy(a, b, &policy).into();
        for _ in 0..5 {
            assert!(cache.get(fof(251.0, 0.001), SR, &cfg).is_some());
        }
        assert_eq!(cache.len(), 1);
        // Out-of-bounds shapes are refused, and the refusal is remembered too.
        assert!(cache.get(fof(1e6, 0.001), SR, &cfg).is_none());
        assert!(cache.get(fof(251.0, 1.0), SR, &cfg).is_none());
        // A Gaussian is bounded by sigma, and never collides with a FOF's key.
        assert!(cache.get(GaussianParams::new(0.004).into(), SR, &cfg).is_some());
        assert!(cache.get(GaussianParams::new(10.0).into(), SR, &cfg).is_none());
        assert_eq!(cache.len(), 2);
    }

    /// A pure function as an [`Objective`], recording the trials committed in order.
    struct Pure<F> {
        f: F,
        width: usize,
        committed: Vec<u64>,
    }

    impl<F: Fn(&Params) -> f64> Objective for Pure<F> {
        type Batch = Vec<f64>;
        fn width(&self) -> usize {
            self.width
        }
        fn speculate(&self, trials: &[Params]) -> Vec<f64> {
            trials.iter().map(&self.f).collect()
        }
        fn commit(&mut self, batch: &mut Vec<f64>, trials: &[Params], i: usize) -> f64 {
            self.committed.push((trials[i].f as f64).to_bits());
            batch[i]
        }
    }

    fn pure<F: Fn(&Params) -> f64>(f: F, width: usize) -> Pure<F> {
        Pure { f, width, committed: Vec::new() }
    }

    /// Golden section as it was written before it could speculate, verbatim: one probe per step,
    /// scored the moment it is placed. The reference [`golden`] is held to at every width.
    fn golden_reference(
        lo: f64,
        hi: f64,
        iters: usize,
        best: &mut Params,
        best_score: &mut f64,
        set: impl Fn(&mut Params, f64),
        mut score: impl FnMut(&Params) -> f64,
    ) {
        if hi <= lo {
            return;
        }
        const R: f64 = 0.618_033_988_749_895;
        let base = *best;
        let mut eval = |x: f64| {
            let mut p = base;
            set(&mut p, x);
            (p, score(&p))
        };
        let (mut a, mut b) = (lo, hi);
        let (mut x1, mut x2) = (b - R * (b - a), a + R * (b - a));
        let (mut p1, mut f1) = eval(x1);
        let (mut p2, mut f2) = eval(x2);
        for _ in 0..iters {
            if f1 >= f2 {
                b = x2;
                x2 = x1;
                p2 = p1;
                f2 = f1;
                x1 = b - R * (b - a);
                (p1, f1) = eval(x1);
            } else {
                a = x1;
                x1 = x2;
                p1 = p2;
                f1 = f2;
                x2 = a + R * (b - a);
                (p2, f2) = eval(x2);
            }
        }
        let (p, f) = if f1 >= f2 { (p1, f1) } else { (p2, f2) };
        if f > *best_score {
            *best = p;
            *best_score = f;
        }
    }

    #[test]
    fn golden_section_finds_a_smooth_maximum_and_never_regresses() {
        // A quadratic peaking at 3.0, with the incumbent already at a worse point.
        let quadratic = |p: &Params| -((p.f as f64 - 3.0).powi(2));
        let mut p = Params { form: Form::Fof { alpha: 0.0, beta: 0.0 }, f: 0.0, t0: 0 };
        let mut s = f64::NEG_INFINITY;
        golden(0.0, 10.0, 40, &mut p, &mut s, |p, x| p.f = x as f32, &mut pure(quadratic, 1));
        assert!((p.f - 3.0).abs() < 1e-3, "found {}", p.f);

        // An incumbent better than anything in the bracket must survive untouched.
        let mut p = Params { form: Form::Fof { alpha: 1.0, beta: 2.0 }, f: 42.0, t0: 7 };
        let keep = p;
        let mut s = 1e9;
        golden(0.0, 10.0, 20, &mut p, &mut s, |p, x| p.f = x as f32, &mut pure(quadratic, 1));
        assert_eq!(p, keep);
        assert_eq!(s, 1e9);

        // A degenerate bracket is a no-op, not a panic.
        let mut p = keep;
        let mut s = 0.0;
        golden(5.0, 5.0, 10, &mut p, &mut s, |p, x| p.f = x as f32, &mut pure(|_: &Params| 1.0, 1));
        assert_eq!(p, keep);
    }

    /// Speculation decides how much is computed, never what is found: at every width the search
    /// commits the same trials in the same order and lands on the same bits.
    ///
    /// The objective is deliberately bumpy, so the walk takes both branches many times rather
    /// than sliding down one side, and the iteration counts cover a batch ending mid-tree.
    #[test]
    fn golden_section_is_the_same_search_at_every_width() {
        let bumpy = |p: &Params| {
            let x = p.f as f64;
            (3.0 * x).sin() + 0.3 * (11.0 * x).cos() - 0.02 * (x - 6.0).powi(2)
        };
        let start = Params { form: Form::Fof { alpha: 1.0, beta: 1.0 }, f: 0.0, t0: 0 };
        let set = |p: &mut Params, x: f64| p.f = x as f32;
        for iters in [0, 1, 2, 3, 7, 10, 23] {
            let reference = {
                let (mut p, mut s, mut probed) = (start, f64::NEG_INFINITY, Vec::new());
                golden_reference(0.5, 12.0, iters, &mut p, &mut s, set, |p| {
                    probed.push((p.f as f64).to_bits());
                    bumpy(p)
                });
                (p, s.to_bits(), probed)
            };
            assert_eq!(reference.2.len(), iters + 2);
            for width in [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 64] {
                let (mut p, mut s) = (start, f64::NEG_INFINITY);
                let mut obj = pure(bumpy, width);
                golden(0.5, 12.0, iters, &mut p, &mut s, set, &mut obj);
                assert_eq!((p, s.to_bits(), obj.committed), reference, "iters {iters}, width {width}");
            }
        }
    }

    /// The one case where a score computed ahead is *not* the serial score: the entry it read was
    /// cleared out before the trial was committed, and a lookup now would render afresh.
    ///
    /// Only a stale entry makes that matter, so the fixture plants one. The key leaves out the
    /// release, so a render under a long release — too long for `max_atom_samples`, and so stored
    /// as `None` — sits under the same key a short-release search then probes. Committing the trial
    /// before it inserts enough to clear the cache, and the serial search, looking up afresh, finds
    /// an envelope where the speculative one read `None`.
    #[test]
    fn a_held_entry_is_not_trusted_across_a_clear() {
        let sig = plant(
            &AtomParams {
                t0: 2_000,
                f: 1_000.0,
                env: EnvelopeParams::new(328.0, 0.001).into(),
                phi: 0.3,
                amp: 1.0,
            },
            12_000,
        );
        let release = |dur: f32| {
            Family::Fof(ReleasePolicy {
                fade_level: 1e-3,
                fade_dur_scale: dur * 328.0,
                fade_dur_min: dur,
                fade_dur_max: dur,
            })
        };
        let (short, long) = (release(0.001), release(0.08));
        let cfg = RefineConfig { max_atom_samples: 3_000, ..RefineConfig::default() };
        let form = |alpha| Form::Fof { alpha, beta: 0.001 };
        let at = |alpha| Params { form: form(alpha), f: 1_000.0, t0: 2_000 };
        let len = |family: &Family, alpha| {
            render_entry(family.shape(form(alpha)), SR, usize::MAX).unwrap().0.support_len()
        };
        assert!(len(&long, 328.0) > cfg.max_atom_samples, "the long release must be refused");

        // Full after one short envelope; the second insertion clears it.
        let budget = len(&short, 300.0) + len(&short, 360.0) - 1;
        let prepared = || {
            let mut cache = EnvelopeCache::with_budget(budget);
            assert!(cache.get(short.shape(form(300.0)), SR, &cfg).is_some());
            assert!(cache.get(long.shape(form(328.0)), SR, &cfg).is_none());
            cache
        };
        let trials = [at(360.0), at(328.0)];

        let run = |width| {
            let mut cache = prepared();
            let search = Search { residual: &sig, sr: SR, family: &short, cfg: &cfg, width };
            let mut obj = Trials { search, cache: &mut cache, gram: None };
            let mut batch = obj.speculate(&trials);
            let scores: Vec<u64> =
                (0..trials.len()).map(|i| obj.commit(&mut batch, &trials, i).to_bits()).collect();
            (scores, cache.len(), cache.usage(), cache.clears)
        };
        let serial = run(1);
        assert_eq!(serial.3, 1, "committing the first trial must clear the cache");
        assert!(f64::from_bits(serial.0[1]) > 0.0, "the serial search must find the envelope");
        assert_eq!(run(2), serial);
    }

    /// The same, end to end: refinement at any width selects the same atom, and leaves the cache
    /// in the same state, as refinement one trial at a time.
    ///
    /// The cache is where speculation could go wrong, so the budgets are small enough to clear
    /// mid-search — which is what exposes a speculative score taken from an entry that a clear has
    /// since removed. The candidates alternate between two releases on one `(alpha, beta)`, because
    /// the key leaves the release out: a stale entry and a fresh render then genuinely differ, the
    /// release decides whether a render fits under `max_atom_samples`, and a wrongly trusted score
    /// shows up as a different atom rather than hiding behind identical envelopes.
    #[test]
    fn refinement_is_the_same_at_every_width() {
        let dict = voice();
        let truth = AtomParams {
            t0: 6_211,
            f: 1_337.0,
            env: EnvelopeParams::new(410.0, 0.0017).into(),
            phi: 0.83,
            amp: 0.7,
        };
        let mut sig = plant(&truth, 30_000);
        let mut s = 0x2545_f491_4f6c_dd1du64;
        for x in &mut sig {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x += 0.05 * ((s >> 40) as f32 / 8_388_608.0 - 1.0);
        }

        let bi = dict
            .blocks
            .iter()
            .position(|b| b.env.params == EnvelopeParams::new(328.0, 0.001).into())
            .unwrap();
        let block = &dict.blocks[bi];
        let short = seed_at(&dict, bi, &truth, &sig);
        let mut long = short;
        if let Shape::Fof(e) = &mut long.atom.env {
            e.fade_dur = 0.08;
        }
        let support = block.support_len();
        let cfg = RefineConfig { max_atom_samples: support + 2_000, ..RefineConfig::default() };

        for budget in [1, support, 3 * support, 20 * support, ENVELOPE_CACHE_SAMPLES] {
            let run = |width| {
                let mut cache = EnvelopeCache::with_budget(budget);
                let mut out = Vec::new();
                for seed in [short, long, short, short, long, long, short] {
                    let mut cand = seed;
                    let moved = refine_at_width(&mut cand, block, &sig, &cfg, &mut cache, width);
                    out.push((moved, cand.atom, cand.mp_score.to_bits()));
                }
                (out, cache.len(), cache.usage(), cache.clears)
            };
            let serial = run(1);
            assert!(serial.0.iter().any(|&(moved, ..)| moved), "budget {budget}: nothing refined");
            for width in [3, 7, 16] {
                assert!(run(width) == serial, "budget {budget}, width {width}");
            }
        }
    }
}
