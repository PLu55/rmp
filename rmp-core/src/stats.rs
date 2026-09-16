//! Statistics over a decomposition book.
//!
//! Aggregation only — nothing here renders. [`crate::book::Book`] carries the parameters and the
//! energy bookkeeping; this module turns them into the distributions and diagnostics that say
//! whether a decomposition went well. The `rmpstat` binary decides whether a result becomes a text
//! table or a chart.
//!
//! Two things about a book are easy to get wrong and are worth stating here, because every
//! consumer of this module hits them:
//!
//! **`block`, `onset` and `bin` are the *seed*, not the atom.** They record the grid point the
//! candidate stage proposed, before `refine` moved it. Refinement moves nearly every atom, so a
//! histogram of `Quantity::Block` and one of `Quantity::Alpha` are two genuinely different
//! pictures — the dictionary the pursuit searched, and the parameters it actually settled on.
//!
//! **`fade_dur` is not a free parameter.** It is `clamp(fade_dur_scale / alpha, min, max)`
//! ([`crate::fof::EnvelopeParams::with_policy`]), so its distribution is a function of alpha's and
//! piles up on the rails. It is exposed because that pile-up says whether the alpha range has
//! escaped the release policy, not because it varies independently.
//!
//! **Most shape quantities belong to one atom kind.** `alpha`, `beta`, `alpha*beta`, `fade_dur` and
//! `rho` describe a FOF and `sigma` a Gaussian; [`Evaluator::eval`] returns `None` for an atom they
//! do not describe, and a histogram counts that mass as [`Histogram::inapplicable`] rather than
//! inventing a value for it. `bandwidth`, `Q`, `support` and everything about placement and energy
//! apply to both.

use crate::atom::{AtomKind, Shape};
use crate::book::Book;
use crate::dict::Dictionary;
use crate::fof::{Envelope, FofError};
use crate::signal::db_fs;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Order statistics of a sample, for the columns a text report prints beside a histogram.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    pub n: usize,
    pub min: f64,
    pub p5: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub p95: f64,
    pub max: f64,
    pub mean: f64,
}

impl Summary {
    /// Order statistics of `xs`. Sorts a copy with `total_cmp`, so NaN cannot make the ordering
    /// inconsistent and panic the sort. An empty sample gives all zeros with `n = 0`.
    pub fn of(xs: &[f64]) -> Self {
        if xs.is_empty() {
            return Self::default();
        }
        let mut v = xs.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let at = |q: f64| -> f64 {
            // Nearest-rank on the sorted sample: no interpolation, so every reported value is one
            // that actually occurs. Enough for a diagnostic, and it cannot invent a figure.
            let i = ((q * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1;
            v[i]
        };
        Self {
            n: v.len(),
            min: v[0],
            p5: at(0.05),
            p25: at(0.25),
            median: at(0.5),
            p75: at(0.75),
            p95: at(0.95),
            max: v[v.len() - 1],
            mean: v.iter().sum::<f64>() / v.len() as f64,
        }
    }
}

/// A scalar derived from one selection.
///
/// Every mapping is one the engine already states somewhere; nothing here invents physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quantity {
    /// Decay coefficient, s^-1.
    Alpha,
    /// The -3 dB bandwidth: `alpha / PI` Hz for a FOF, `sqrt(ln 2) / (PI sigma)` for a Gaussian.
    Bandwidth,
    /// A Gaussian's envelope standard deviation, milliseconds.
    Sigma,
    /// Attack duration, milliseconds.
    Beta,
    /// `alpha * beta`. Walls at 4 (the grid and refinement cap), 6.908 (`ln(1/fade_level)`, where
    /// the attack starts outlasting the decay) and 10 (rfofs's `amax` cliff, i.e. silence).
    AlphaBeta,
    /// Carrier frequency, Hz.
    Freq,
    /// Amplitude in dBFS.
    AmpDb,
    /// `energy_removed` in dB relative to the book's initial energy.
    EnergyDb,
    /// Onset, seconds.
    T0,
    /// Support length, milliseconds. Needs a render unless `fast_support` is set.
    SupportMs,
    /// Fade-out ramp, milliseconds. Expect rails — see the module docs.
    FadeDurMs,
    /// `f / bandwidth`: the formant's Q, `f * PI / alpha` for a FOF.
    Q,
    /// `alpha / (2 * PI * f)`, the u/v coherence the projection sees. Near 1 is ill-conditioned
    /// (`dict::Block::rho`).
    Rho,
    /// Carrier cycles spanned by the atom. HRMP's `min_mask_periods` counts the same thing.
    Periods,
    /// The *seed* block index. Pre-refinement — see the module docs.
    Block,
}

impl Quantity {
    /// Every quantity, for a front end that has to offer them all.
    ///
    /// Beside the enum so the two cannot drift: the only list before this one was hand-written
    /// inside `rmpstat`'s `slugs_are_unique` test, and it had been missing [`Quantity::Sigma`] since
    /// the Gaussian atom was added — so the one quantity describing Gaussians was the one quantity
    /// nothing checked.
    pub const ALL: [Quantity; 15] = [
        Self::Alpha,
        Self::Bandwidth,
        Self::Sigma,
        Self::Beta,
        Self::AlphaBeta,
        Self::Freq,
        Self::AmpDb,
        Self::EnergyDb,
        Self::T0,
        Self::SupportMs,
        Self::FadeDurMs,
        Self::Q,
        Self::Rho,
        Self::Periods,
        Self::Block,
    ];

    /// The column heading and unit a report should print.
    pub fn label(self) -> &'static str {
        match self {
            Self::Alpha => "alpha (1/s)",
            Self::Bandwidth => "bandwidth (Hz)",
            Self::Sigma => "sigma (ms)",
            Self::Beta => "beta (ms)",
            Self::AlphaBeta => "alpha*beta",
            Self::Freq => "f (Hz)",
            Self::AmpDb => "amp (dBFS)",
            Self::EnergyDb => "energy (dB re. initial)",
            Self::T0 => "t0 (s)",
            Self::SupportMs => "support (ms)",
            Self::FadeDurMs => "fade_dur (ms)",
            Self::Q => "Q = f/bandwidth",
            Self::Rho => "rho = alpha/(2*pi*f)",
            Self::Periods => "carrier periods",
            Self::Block => "seed block",
        }
    }

    /// Whether a histogram of this quantity defaults to geometric bins.
    ///
    /// The dictionary is a geometric ladder — alpha steps by 1.6, beta by about 3.3 — so linear
    /// bins would put most of it in the bottom bucket and leave the rest empty.
    pub fn log_by_default(self) -> bool {
        matches!(
            self,
            Self::Alpha
                | Self::Bandwidth
                | Self::Sigma
                | Self::Beta
                | Self::AlphaBeta
                | Self::Freq
                | Self::SupportMs
                | Self::Q
                | Self::Rho
                | Self::Periods
        )
    }
}

impl std::str::FromStr for Quantity {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        // Accept `alpha-beta`, `alpha_beta` and `alphabeta` alike; a CLI list is typed by hand.
        let k: String = s
            .chars()
            .filter(|c| !matches!(c, '-' | '_' | ' '))
            .flat_map(char::to_lowercase)
            .collect();
        Ok(match k.as_str() {
            "alpha" => Self::Alpha,
            "bandwidth" | "bw" => Self::Bandwidth,
            "sigma" | "sigmams" => Self::Sigma,
            "beta" => Self::Beta,
            "alphabeta" | "ab" => Self::AlphaBeta,
            "f" | "freq" | "frequency" => Self::Freq,
            "amp" | "ampdb" | "amplitude" => Self::AmpDb,
            "energy" | "energydb" => Self::EnergyDb,
            "t0" | "onset" | "time" => Self::T0,
            "support" | "supportms" => Self::SupportMs,
            "fadedur" | "fadedurms" | "fade" => Self::FadeDurMs,
            "q" => Self::Q,
            "rho" => Self::Rho,
            "periods" => Self::Periods,
            "block" => Self::Block,
            _ => return Err(format!("unknown quantity '{s}'")),
        })
    }
}

/// Evaluates a [`Quantity`] over a book, rendering envelopes only when one is actually needed.
///
/// The memo is on the exact shape bits ([`Shape::cache_key`]). That is worth nothing on a refined
/// book — the values are continuous, so essentially every atom is distinct — and collapses an
/// unrefined one to the handful of dictionary blocks for free. Quantizing the key would not help:
/// measured on a 5000-atom refined book, 3125 atoms still land in distinct buckets at 5% log
/// quantization.
pub struct Evaluator {
    sample_rate: f32,
    initial_energy: f64,
    /// Skip the render and use [`Shape::approx_support_len`]: `ln(1/fade_level) * sr / alpha` for a
    /// FOF, exact for a Gaussian.
    pub fast_support: bool,
    support: HashMap<(u8, u32, u32), usize>,
}

impl Evaluator {
    pub fn new(book: &Book) -> Self {
        Self {
            sample_rate: book.sample_rate,
            initial_energy: book.initial_energy,
            fast_support: false,
            support: HashMap::new(),
        }
    }

    /// Support in samples, memoized. Falls back to the formula when `fast_support` is set.
    ///
    /// The formula is an estimate, not a definition: rfofs clamps `decay_end` to at least
    /// `attack_end` and its death sample depends on internal rounding, which is why the engine
    /// derives support by rendering. The two agree within a factor of 0.8..1.6.
    fn support_len(&mut self, env: Shape) -> Result<usize, FofError> {
        if self.fast_support {
            return Ok(env.approx_support_len(self.sample_rate));
        }
        let key = env.cache_key();
        if let Some(&n) = self.support.get(&key) {
            return Ok(n);
        }
        let n = Envelope::render(env, self.sample_rate)?.support_len();
        self.support.insert(key, n);
        Ok(n)
    }

    /// `q` evaluated on one selection, or `None` when `q` does not describe this atom's kind.
    pub fn eval(
        &mut self,
        s: &crate::book::Selection,
        q: Quantity,
    ) -> Result<Option<f64>, FofError> {
        let f = s.atom.f as f64;
        let (fof, gauss) = (s.atom.env.as_fof(), s.atom.env.as_gaussian());
        Ok(Some(match q {
            Quantity::Alpha => {
                let Some(p) = fof else { return Ok(None) };
                p.alpha as f64
            }
            Quantity::Bandwidth => match (fof, gauss) {
                (Some(p), _) => p.alpha as f64 / PI,
                (_, Some(g)) => g.bandwidth_hz() as f64,
                _ => unreachable!(),
            },
            Quantity::Sigma => {
                let Some(g) = gauss else { return Ok(None) };
                g.sigma as f64 * 1e3
            }
            Quantity::Beta => {
                let Some(p) = fof else { return Ok(None) };
                p.beta as f64 * 1e3
            }
            Quantity::AlphaBeta => {
                let Some(p) = fof else { return Ok(None) };
                p.alpha as f64 * p.beta as f64
            }
            Quantity::Freq => f,
            Quantity::AmpDb => db_fs(s.atom.amp as f64) as f64,
            Quantity::EnergyDb => {
                if self.initial_energy > 0.0 && s.energy_removed > 0.0 {
                    10.0 * (s.energy_removed / self.initial_energy).log10()
                } else {
                    f64::NEG_INFINITY
                }
            }
            Quantity::T0 => s.atom.t0 as f64 / self.sample_rate as f64,
            Quantity::SupportMs => {
                self.support_len(s.atom.env)? as f64 * 1e3 / self.sample_rate as f64
            }
            Quantity::FadeDurMs => {
                let Some(p) = fof else { return Ok(None) };
                p.fade_dur as f64 * 1e3
            }
            Quantity::Q => {
                // Written per kind so a FOF's Q stays `f * PI / alpha` to the last bit.
                let (num, den) = match (fof, gauss) {
                    (Some(p), _) => (f * PI, p.alpha as f64),
                    (_, Some(g)) => (f, g.bandwidth_hz() as f64),
                    _ => unreachable!(),
                };
                if den > 0.0 { num / den } else { f64::INFINITY }
            }
            Quantity::Rho => {
                let Some(p) = fof else { return Ok(None) };
                if f > 0.0 {
                    p.alpha as f64 / (2.0 * PI * f)
                } else {
                    f64::INFINITY
                }
            }
            Quantity::Periods => {
                self.support_len(s.atom.env)? as f64 * f / self.sample_rate as f64
            }
            Quantity::Block => s.block as f64,
        }))
    }

    /// `q` over every selection, in book order. `None` where `q` does not describe the atom.
    pub fn column(&mut self, book: &Book, q: Quantity) -> Result<Vec<Option<f64>>, FofError> {
        book.selections.iter().map(|s| self.eval(s, q)).collect()
    }
}

/// What each atom contributes to a histogram bin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Weight {
    /// One per atom.
    #[default]
    Count,
    /// `energy_removed`, so the bars say where the signal's energy went rather than where the
    /// atoms are. The two differ sharply: MP spends many weak atoms on the residual's tail.
    Energy,
}

/// A weighted histogram, with the order statistics of the underlying sample.
#[derive(Clone, Debug)]
pub struct Histogram {
    pub quantity: Quantity,
    /// Bin edges, ascending, length `counts.len() + 1`. Geometric when the histogram is log-binned.
    pub edges: Vec<f64>,
    pub counts: Vec<f64>,
    /// Mass below `edges[0]` and above the last edge. Kept rather than clamped into the end bins,
    /// so `counts` never lies about the range it covers and mass still conserves.
    pub below: f64,
    pub above: f64,
    /// Mass that had no finite value — a zero-frequency atom's `Q`, say.
    pub skipped: f64,
    /// Mass from atoms the quantity does not describe — a Gaussian's `alpha`. Not part of `total`:
    /// a histogram of `alpha` is a histogram over the FOFs.
    pub inapplicable: f64,
    pub total: f64,
    pub log: bool,
    pub stats: Summary,
}

impl Histogram {
    /// The largest bin, for scaling a bar.
    pub fn peak(&self) -> f64 {
        self.counts.iter().copied().fold(0.0, f64::max)
    }
}

/// Bin `q` over the book.
///
/// `range` overrides the data's own extent. `log` forces geometric or linear bins; `None` takes
/// [`Quantity::log_by_default`]. A log histogram drops non-positive values into `skipped`, since
/// they have no place on a geometric axis.
pub fn histogram(
    book: &Book,
    ev: &mut Evaluator,
    q: Quantity,
    bins: usize,
    log: Option<bool>,
    range: Option<(f64, f64)>,
    w: Weight,
) -> Result<Histogram, FofError> {
    let bins = bins.max(1);
    let log = log.unwrap_or_else(|| q.log_by_default());
    let xs = ev.column(book, q)?;
    let ws: Vec<f64> = match w {
        Weight::Count => vec![1.0; xs.len()],
        Weight::Energy => book.selections.iter().map(|s| s.energy_removed).collect(),
    };

    let usable: Vec<f64> = xs
        .iter()
        .flatten()
        .copied()
        .filter(|x| x.is_finite() && (!log || *x > 0.0))
        .collect();
    let stats = Summary::of(&usable);

    let (lo, hi) = match range {
        Some(r) => r,
        None if usable.is_empty() => (0.0, 1.0),
        // A degenerate sample (every atom identical, or a single atom) would give a zero-width
        // axis and a division by zero below, so widen it into something a bar chart can draw.
        None if stats.min == stats.max => {
            if log {
                (stats.min * 0.5, stats.max * 2.0)
            } else {
                (stats.min - 0.5, stats.max + 0.5)
            }
        }
        None => (stats.min, stats.max),
    };

    let edges: Vec<f64> = if log {
        let (l0, l1) = (lo.max(f64::MIN_POSITIVE).ln(), hi.max(f64::MIN_POSITIVE).ln());
        (0..=bins)
            .map(|i| (l0 + (l1 - l0) * i as f64 / bins as f64).exp())
            .collect()
    } else {
        (0..=bins)
            .map(|i| lo + (hi - lo) * i as f64 / bins as f64)
            .collect()
    };

    let mut counts = vec![0.0; bins];
    let (mut below, mut above, mut skipped, mut total) = (0.0, 0.0, 0.0, 0.0);
    let mut inapplicable = 0.0;
    for (&x, &wt) in xs.iter().zip(&ws) {
        let Some(x) = x else {
            inapplicable += wt;
            continue;
        };
        total += wt;
        if !x.is_finite() || (log && x <= 0.0) {
            skipped += wt;
        } else if x < lo {
            below += wt;
        } else if x > hi {
            above += wt;
        } else {
            // Position on the binning axis, so a log histogram bins geometrically. The clamp
            // catches `x == hi`, which lands exactly on the last edge.
            let u = if log {
                (x.ln() - edges[0].ln()) / (edges[bins].ln() - edges[0].ln())
            } else {
                (x - lo) / (hi - lo)
            };
            let i = ((u * bins as f64) as usize).min(bins - 1);
            counts[i] += wt;
        }
    }

    Ok(Histogram {
        quantity: q,
        edges,
        counts,
        below,
        above,
        skipped,
        inapplicable,
        total,
        log,
        stats,
    })
}

/// The headline figures for a book.
#[derive(Clone, Debug)]
pub struct BookSummary {
    pub atoms: usize,
    pub sample_rate: f32,
    /// Earliest onset and latest sample any atom reaches, in samples. `t0` is signed, so the first
    /// may be negative.
    pub span: (i64, i64),
    pub initial_energy: f64,
    pub residual_energy: f64,
    pub snr_db: f32,
    /// `(target_db, atoms needed)`, for 10/20/30/40 dB.
    pub atoms_to_reach: Vec<(f32, Option<usize>)>,
    pub refined_frac: f64,
    /// Atoms that carry an `hr_score` — i.e. HRMP ran on them.
    pub hrmp_atoms: usize,
    /// `energy_removed / projected_energy`.
    ///
    /// This, not `hr_score`, is what an HRMP clamp looks like from the book: the projection
    /// predicted one energy and the amplitude actually subtracted was smaller. Slightly *above* 1
    /// is ordinary — the atom subtracted is what rfofs rendered, not the ideal vector projected
    /// onto, so the two differ a little in both directions. A large shortfall on a book where HRMP
    /// did **not** run is a parameter-mapping error instead.
    pub removed_over_projected: Summary,
    /// Atoms that fell more than 1% short of their projection.
    pub shortfall_atoms: usize,
    /// `max |hr_score - energy_removed| / energy_removed`.
    ///
    /// Both fields are post-clamp measures of the same energy, so this is a consistency check on
    /// the book and nothing more — it is not a clamp severity. Measured at 5e-3 on real material;
    /// treating a difference this size as "HRMP clamped this atom" reads pure rounding.
    pub hr_consistency: f64,
    /// `sum(energy_removed) / (initial - residual)`. Greedy MP subtracts each atom from the
    /// running residual, so the parts sum to the whole only when no atom re-removed energy an
    /// earlier one had already taken; below 1 is ordinary and quantifies the overlap.
    pub deposited_frac: f64,
    pub energy_removed: Summary,
}

pub fn summarize(book: &Book, ev: &mut Evaluator) -> Result<BookSummary, FofError> {
    let n = book.len();
    let mut span = (0i64, 0i64);
    let (mut refined, mut hrmp, mut shortfall) = (0usize, 0usize, 0usize);
    let (mut ratios, mut removed) = (Vec::new(), Vec::new());
    let (mut sum_removed, mut hr_consistency) = (0.0f64, 0.0f64);

    for (i, s) in book.selections.iter().enumerate() {
        let end = s.atom.t0 + ev.support_len(s.atom.env)? as i64;
        if i == 0 {
            span = (s.atom.t0, end);
        } else {
            span = (span.0.min(s.atom.t0), span.1.max(end));
        }
        refined += usize::from(s.refined);
        if let Some(hr) = s.hr_score {
            hrmp += 1;
            if s.energy_removed > 0.0 {
                hr_consistency =
                    hr_consistency.max((hr - s.energy_removed).abs() / s.energy_removed);
            }
        }
        if s.projected_energy > 0.0 {
            let r = s.energy_removed / s.projected_energy;
            ratios.push(r);
            if r < 0.99 {
                shortfall += 1;
            }
        }
        removed.push(s.energy_removed);
        sum_removed += s.energy_removed;
    }

    let captured = book.initial_energy - book.residual_energy();
    Ok(BookSummary {
        atoms: n,
        sample_rate: book.sample_rate,
        span,
        initial_energy: book.initial_energy,
        residual_energy: book.residual_energy(),
        snr_db: book.snr_db(),
        atoms_to_reach: [10.0f32, 20.0, 30.0, 40.0]
            .iter()
            .map(|&t| (t, book.atoms_to_reach(t)))
            .collect(),
        refined_frac: frac(refined, n),
        hrmp_atoms: hrmp,
        removed_over_projected: Summary::of(&ratios),
        shortfall_atoms: shortfall,
        hr_consistency,
        deposited_frac: if captured > 0.0 {
            sum_removed / captured
        } else {
            0.0
        },
        energy_removed: Summary::of(&removed),
    })
}

fn frac(k: usize, n: usize) -> f64 {
    if n == 0 { 0.0 } else { k as f64 / n as f64 }
}

/// How one dictionary block was used.
#[derive(Clone, Copy, Debug)]
pub struct BlockUse {
    pub index: usize,
    pub shape: Shape,
    pub support_len: usize,
    pub count: usize,
    /// Share of `sum(energy_removed)` the block's seeds account for.
    pub energy_share: f64,
}

/// Diagnostics that need the dictionary the run used.
///
/// The book records only the seed `block` index, never the grid itself, so the caller supplies the
/// config. A config that does not match the run gives a differently-sized block list — caught here
/// — or, worse, a same-sized one with different values and silently meaningless drift figures. The
/// CLI must say which config it used.
#[derive(Clone, Debug)]
pub struct Diagnostics {
    pub blocks: Vec<BlockUse>,
    /// Set when more than a quarter of a family's seeds sit on its first or last rung.
    pub edge_pileup: Option<String>,
    /// `|ln(alpha / alpha_seed)|` over the FOFs — how far refinement walked, in ladder rungs. The
    /// alpha ladder steps by 1.6, so `ln 1.6 = 0.47` is one rung.
    pub d_ln_alpha: Summary,
    /// `|ln(beta / beta_seed)|`. The beta ladder steps by about 3.3, so one rung is `ln 3.3 = 1.19`.
    pub d_ln_beta: Summary,
    /// `|ln(sigma / sigma_seed)|` over the Gaussians. A 2.5 ladder puts one rung at `ln 2.5 = 0.92`.
    pub d_ln_sigma: Summary,
    /// `|f - bin_hz(bin)|`, Hz.
    pub d_f_hz: Summary,
    /// `|t0 - onset|`, samples.
    pub d_t0: Summary,
    /// Atoms whose seed bin the block had disabled, or whose block index is out of range.
    pub off_grid_seeds: usize,
    /// Atoms whose seed bin was live but ill-conditioned (`rho^2` above the gate).
    pub ill_conditioned: usize,
}

pub fn diagnose(book: &Book, dict: &Dictionary, rho_sq_max: f32) -> Diagnostics {
    let n_blocks = dict.blocks.len();
    let hist = book.block_histogram(n_blocks);
    let mut energy = vec![0.0f64; n_blocks];
    let (mut d_a, mut d_b, mut d_s, mut d_f, mut d_t) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut off_grid, mut ill) = (0usize, 0usize);
    let mut total_energy = 0.0f64;

    for s in &book.selections {
        total_energy += s.energy_removed;
        let Some(block) = dict.blocks.get(s.block) else {
            off_grid += 1;
            continue;
        };
        energy[s.block] += s.energy_removed;

        // `Selection::block` is the seed's provenance, so the block's own envelope is the
        // before-picture and `atom.env` is whatever refinement settled on.
        match (block.env.params, s.atom.env) {
            (Shape::Fof(seed), Shape::Fof(atom)) => {
                d_a.push((atom.alpha as f64 / seed.alpha as f64).ln().abs());
                if seed.beta > 0.0 && atom.beta > 0.0 {
                    d_b.push((atom.beta as f64 / seed.beta as f64).ln().abs());
                }
            }
            (Shape::Gaussian(seed), Shape::Gaussian(atom)) => {
                d_s.push((atom.sigma as f64 / seed.sigma as f64).ln().abs());
            }
            // Refinement never changes kind, so a seed block of the other kind means the config
            // does not describe the run that wrote this book.
            _ => {
                off_grid += 1;
                continue;
            }
        }
        d_f.push((s.atom.f - block.bin_hz(s.bin)).abs() as f64);
        d_t.push((s.atom.t0 - s.onset as i64).abs() as f64);

        match block.rho(s.bin) {
            None => off_grid += 1,
            Some(r) if r * r > rho_sq_max => ill += 1,
            Some(_) => {}
        }
    }

    let blocks: Vec<BlockUse> = dict
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| BlockUse {
            index: i,
            shape: b.env.params,
            support_len: b.support_len(),
            count: hist[i],
            energy_share: if total_energy > 0.0 {
                energy[i] / total_energy
            } else {
                0.0
            },
        })
        .collect();

    // Selections piling up at a ladder edge mean it is mis-sized: the pursuit wants a shape the grid
    // does not reach and settles for the nearest rung. Judged per family, since each is its own
    // ladder, and against that family's own seeds.
    let mut warnings = Vec::new();
    for (kind, rung) in [(AtomKind::Fof, "alpha"), (AtomKind::Gaussian, "sigma")] {
        let family: Vec<usize> =
            (0..n_blocks).filter(|&i| dict.blocks[i].env.kind() == kind).collect();
        let (Some(&first), Some(&last)) = (family.first(), family.last()) else {
            continue;
        };
        let seeds: usize = family.iter().map(|&i| hist[i]).sum();
        if seeds == 0 {
            continue;
        }
        // Each family is generated rung-major (alpha before beta), so its first and last blocks
        // are the extreme rungs whatever the beta count is.
        let edge = hist[first] + hist[last];
        if edge * 4 > seeds {
            warnings.push(format!(
                "{:.0}% of {kind} seeds are on the first or last {rung} rung — the ladder may not \
                 reach far enough",
                100.0 * edge as f64 / seeds as f64
            ));
        }
    }
    let edge_pileup = (!warnings.is_empty()).then(|| warnings.join("; "));

    Diagnostics {
        blocks,
        edge_pileup,
        d_ln_alpha: Summary::of(&d_a),
        d_ln_beta: Summary::of(&d_b),
        d_ln_sigma: Summary::of(&d_s),
        d_f_hz: Summary::of(&d_f),
        d_t0: Summary::of(&d_t),
        off_grid_seeds: off_grid,
        ill_conditioned: ill,
    }
}

#[cfg(test)]
mod tests {

    /// `Quantity::ALL` really is all of them.
    ///
    /// The inner match is exhaustive, so adding a variant stops this compiling until it is listed
    /// there — and the assertion then catches it missing from `ALL`. Without that pairing a new
    /// quantity would simply be unreachable from every front end that enumerates them, which is
    /// exactly how `Sigma` came to be absent from the only list there was.
    #[test]
    fn every_quantity_is_in_all() {
        fn tag(q: Quantity) -> u8 {
            match q {
                Quantity::Alpha => 0,
                Quantity::Bandwidth => 1,
                Quantity::Sigma => 2,
                Quantity::Beta => 3,
                Quantity::AlphaBeta => 4,
                Quantity::Freq => 5,
                Quantity::AmpDb => 6,
                Quantity::EnergyDb => 7,
                Quantity::T0 => 8,
                Quantity::SupportMs => 9,
                Quantity::FadeDurMs => 10,
                Quantity::Q => 11,
                Quantity::Rho => 12,
                Quantity::Periods => 13,
                Quantity::Block => 14,
            }
        }
        let mut tags: Vec<u8> = Quantity::ALL.iter().map(|&q| tag(q)).collect();
        tags.sort_unstable();
        let n = tags.len();
        tags.dedup();
        assert_eq!(tags.len(), n, "a quantity appears twice in ALL");
        assert_eq!(tags, (0..15).collect::<Vec<u8>>(), "a quantity is missing from ALL");
    }

    /// Every quantity says what it is; a blank label would be a blank checkbox.
    #[test]
    fn every_quantity_has_a_label() {
        for q in Quantity::ALL {
            assert!(!q.label().is_empty(), "{q:?} has no label");
        }
    }

    use super::*;
    use crate::book::Selection;
    use crate::fof::{AtomParams, EnvelopeParams};
    use crate::gauss::GaussianParams;

    fn sel(alpha: f32, beta: f32, f: f32, amp: f32, energy: f64) -> Selection {
        Selection {
            atom: AtomParams {
                t0: 0,
                f,
                env: EnvelopeParams::new(alpha, beta).into(),
                phi: 0.0,
                amp,
            },
            block: 0,
            onset: 0,
            bin: 1,
            projected_energy: energy,
            energy_removed: energy,
            residual_energy: 1.0,
            hr_score: None,
            refined: false,
        }
    }

    fn book_of(sels: Vec<Selection>) -> Book {
        let mut b = Book::new(100.0, 48_000.0);
        b.selections = sels;
        b
    }

    #[test]
    fn summary_order_statistics() {
        let s = Summary::of(&[5.0, 1.0, 3.0, 2.0, 4.0]);
        assert_eq!((s.n, s.min, s.max, s.median), (5, 1.0, 5.0, 3.0));
        assert!((s.mean - 3.0).abs() < 1e-12);
        assert_eq!(Summary::of(&[]).n, 0);
    }

    /// Mass conserves whatever falls outside the range, and the bars carry the weight asked for.
    #[test]
    fn histogram_conserves_mass_under_both_weights() {
        let b = book_of(vec![
            sel(100.0, 0.001, 400.0, 1.0, 4.0),
            sel(200.0, 0.001, 800.0, 1.0, 3.0),
            sel(400.0, 0.001, 1600.0, 1.0, 2.0),
            sel(800.0, 0.001, 3200.0, 1.0, 1.0),
        ]);
        for (w, expect) in [(Weight::Count, 4.0), (Weight::Energy, 10.0)] {
            for range in [None, Some((150.0, 500.0))] {
                let mut ev = Evaluator::new(&b);
                let h = histogram(&b, &mut ev, Quantity::Alpha, 8, None, range, w).unwrap();
                let inside: f64 = h.counts.iter().sum();
                assert!(
                    (inside + h.below + h.above + h.skipped - expect).abs() < 1e-12,
                    "{w:?} {range:?}"
                );
                assert!((h.total - expect).abs() < 1e-12);
            }
        }
    }

    /// A log histogram's edges are geometric, and a geometric sample lands one per bin — the
    /// property linear bins would destroy for a 1.6-ratio ladder.
    #[test]
    fn log_bins_are_geometric_and_spread_a_geometric_sample() {
        let b = book_of(vec![
            sel(100.0, 0.001, 400.0, 1.0, 1.0),
            sel(200.0, 0.001, 400.0, 1.0, 1.0),
            sel(400.0, 0.001, 400.0, 1.0, 1.0),
            sel(800.0, 0.001, 400.0, 1.0, 1.0),
        ]);
        let mut ev = Evaluator::new(&b);
        let h = histogram(&b, &mut ev, Quantity::Alpha, 4, Some(true), None, Weight::Count).unwrap();
        assert!(h.log);
        for i in 0..h.edges.len() - 2 {
            let (r0, r1) = (h.edges[i + 1] / h.edges[i], h.edges[i + 2] / h.edges[i + 1]);
            assert!((r0 - r1).abs() < 1e-9, "edges not geometric: {r0} vs {r1}");
        }
        assert_eq!(h.counts, vec![1.0, 1.0, 1.0, 1.0]);

        // The same sample on a linear axis piles into the bottom bin — why log is the default.
        let mut ev = Evaluator::new(&b);
        let lin =
            histogram(&b, &mut ev, Quantity::Alpha, 4, Some(false), None, Weight::Count).unwrap();
        assert_eq!(lin.counts[0], 2.0);
    }

    /// Non-positive and non-finite values have no place on a geometric axis, and must be counted
    /// as skipped rather than silently dropped or folded into the first bin.
    #[test]
    fn log_histogram_skips_non_positive_values() {
        let mut b = book_of(vec![
            sel(100.0, 0.001, 400.0, 1.0, 1.0),
            // A zero-frequency atom has no finite Q.
            sel(100.0, 0.001, 0.0, 1.0, 1.0),
        ]);
        b.selections[1].atom.f = 0.0;
        let mut ev = Evaluator::new(&b);
        let h = histogram(&b, &mut ev, Quantity::Q, 4, None, None, Weight::Count).unwrap();
        assert_eq!(h.skipped, 1.0);
        assert_eq!(h.counts.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn quantities_parse_and_derive() {
        use std::str::FromStr;
        assert_eq!(Quantity::from_str("alpha_beta").unwrap(), Quantity::AlphaBeta);
        assert_eq!(Quantity::from_str("BW").unwrap(), Quantity::Bandwidth);
        assert!(Quantity::from_str("nonsense").is_err());

        let b = book_of(vec![sel(314.159_27, 0.002, 1000.0, 0.5, 1.0)]);
        let mut ev = Evaluator::new(&b);
        let s = &b.selections[0];
        let mut at = |q| ev.eval(s, q).unwrap();
        assert!((at(Quantity::Bandwidth).unwrap() - 100.0).abs() < 1e-3);
        assert!((at(Quantity::Beta).unwrap() - 2.0).abs() < 1e-6);
        assert!((at(Quantity::AlphaBeta).unwrap() - 0.628_318_5).abs() < 1e-6);
        // Q = f*pi/alpha = 1000*pi/314.15927 = 10.
        assert!((at(Quantity::Q).unwrap() - 10.0).abs() < 1e-4);
        assert!((at(Quantity::AmpDb).unwrap() + 6.0206).abs() < 1e-3);
        assert_eq!(at(Quantity::Sigma), None, "a FOF has no sigma");
    }

    /// A Gaussian answers the quantities that describe it, declines the FOF-only ones, and a
    /// histogram over a mixed book keeps the declined mass apart instead of dropping or faking it.
    #[test]
    fn a_gaussian_has_its_own_quantities_and_the_rest_are_inapplicable() {
        let mut g = sel(100.0, 0.001, 1000.0, 1.0, 5.0);
        g.atom.env = GaussianParams::new(0.005).into();
        let b = book_of(vec![sel(100.0, 0.001, 400.0, 1.0, 1.0), g]);
        let mut ev = Evaluator::new(&b);
        let s = &b.selections[1];
        for q in [Quantity::Alpha, Quantity::Beta, Quantity::AlphaBeta, Quantity::FadeDurMs, Quantity::Rho] {
            assert_eq!(ev.eval(s, q).unwrap(), None, "{q:?}");
        }
        assert!((ev.eval(s, Quantity::Sigma).unwrap().unwrap() - 5.0).abs() < 1e-4);
        let bw = ev.eval(s, Quantity::Bandwidth).unwrap().unwrap();
        assert!((bw - 0.2650 / 0.005).abs() < 0.1, "bandwidth {bw}");
        assert!((ev.eval(s, Quantity::Q).unwrap().unwrap() - 1000.0 / bw).abs() < 1e-3);
        let support = GaussianParams::new(0.005).support_len(48_000.0) as f64;
        assert_eq!(ev.eval(s, Quantity::SupportMs).unwrap(), Some(support * 1e3 / 48_000.0));

        for w in [Weight::Count, Weight::Energy] {
            let h = histogram(&b, &mut ev, Quantity::Alpha, 4, None, None, w).unwrap();
            let inside: f64 = h.counts.iter().sum();
            let (want_total, want_other) = if w == Weight::Count { (1.0, 1.0) } else { (1.0, 5.0) };
            assert_eq!((inside, h.total, h.inapplicable), (want_total, want_total, want_other));
        }
    }

    /// The estimate and the render agree within the bound the engine's own test asserts, so
    /// `--fast-support` is a speed knob and not a different answer.
    #[test]
    fn fast_support_tracks_the_render() {
        let b = book_of(vec![
            sel(80.0, 0.003, 400.0, 1.0, 1.0),
            sel(500.0, 0.001, 400.0, 1.0, 1.0),
            sel(2147.0, 0.000_3, 400.0, 1.0, 1.0),
        ]);
        let mut exact = Evaluator::new(&b);
        let mut fast = Evaluator::new(&b);
        fast.fast_support = true;
        for s in &b.selections {
            let e = exact.eval(s, Quantity::SupportMs).unwrap().unwrap();
            let f = fast.eval(s, Quantity::SupportMs).unwrap().unwrap();
            assert!((0.8..1.6).contains(&(e / f)), "{} {e} vs {f}", s.atom.env.describe());
        }
    }

    #[test]
    fn summary_reuses_the_books_own_convergence_figures() {
        let mut b = book_of(vec![
            sel(100.0, 0.001, 400.0, 1.0, 90.0),
            sel(200.0, 0.001, 400.0, 1.0, 9.0),
        ]);
        b.selections[0].residual_energy = 10.0; // 10 dB
        b.selections[1].residual_energy = 1.0; // 20 dB
        b.selections[1].refined = true;
        b.selections[1].hr_score = Some(4.5); // clamped to half

        let mut ev = Evaluator::new(&b);
        let s = summarize(&b, &mut ev).unwrap();
        assert_eq!(s.atoms, 2);
        assert!((s.snr_db - b.snr_db()).abs() < 1e-6);
        assert_eq!(s.atoms_to_reach[0], (10.0, Some(1)));
        assert_eq!(s.atoms_to_reach[3], (40.0, None));
        assert!((s.refined_frac - 0.5).abs() < 1e-12);
        assert_eq!(s.hrmp_atoms, 1);
        // `hr_score` is half `energy_removed` here, which is a *consistency* complaint about the
        // fixture, not a clamp: the two fields measure the same post-clamp energy.
        assert!((s.hr_consistency - 0.5).abs() < 1e-12);
        // Removed matches projected exactly, so nothing fell short.
        assert_eq!(s.shortfall_atoms, 0);
        assert!((s.removed_over_projected.median - 1.0).abs() < 1e-12);
        // 99 of 99 removed.
        assert!((s.deposited_frac - 1.0).abs() < 1e-12);
    }

    #[test]
    fn diagnose_reports_drift_and_edge_pileup() {
        let mut planner = crate::fft::Planner::new();
        let dict = Dictionary::voice(48_000.0, &mut planner, &Default::default()).unwrap();

        // Every seed on block 0 — the bottom alpha rung — so the pile-up warning must fire.
        let mut b = book_of(vec![
            sel(80.0, 0.000_3, 400.0, 1.0, 1.0),
            sel(128.0, 0.000_3, 400.0, 1.0, 1.0),
        ]);
        // The second atom's alpha is one rung (1.6x) above its seed's.
        b.selections[1].refined = true;
        // Block 0 starts well above bin 1 — `f_min` is 50 Hz and its `fft_len` is large — so a
        // seed bin has to be taken from the block rather than made up, or every atom counts as
        // off-grid.
        let live = dict.blocks[0].live_bins().next().unwrap();
        for s in &mut b.selections {
            s.bin = live;
        }

        let d = diagnose(&b, &dict, 0.9999);
        assert_eq!(d.blocks.len(), dict.blocks.len());
        assert_eq!(d.blocks[0].count, 2);
        assert!(d.edge_pileup.is_some(), "expected an edge-pile-up warning");
        assert!(
            (d.d_ln_alpha.max - 1.6f64.ln()).abs() < 1e-3,
            "one rung is ln 1.6: {}",
            d.d_ln_alpha.max
        );
        assert_eq!(d.d_ln_alpha.min, 0.0);
        assert_eq!(d.off_grid_seeds, 0);
    }
}
