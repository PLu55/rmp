//! A time-frequency display of a decomposition — the atom-based pseudo-Wigner map.
//!
//! The spec is explicit that this is diagnostics, not pursuit: build the map as
//! `E(t,f) = sum_k E_k * W~_{g_k}(t,f)`, a sum of *per-atom* distributions, never the bilinear
//! Wigner-Ville distribution of the summed signal. Summing per-atom means there are no cross-terms
//! to suppress, by construction rather than by smoothing.
//!
//! # What the kernel is
//!
//! The exact Wigner-Ville form of a FOF atom has never been derived — LastWave's FOF display code
//! says as much in its own comments, and MPTK's non-Gaussian-window paths are approximate. So this
//! is a *pseudo*-Wigner representation, and the approximation is named precisely: for each atom
//! the map deposits the **separable product of the atom's exact time and frequency marginals**,
//!
//! ```text
//! E_k(t, f) = E_k * p_k(t) * q_k(f),      sum p_k = sum q_k = 1
//!
//! p_k = the atom's own power envelope E(n - t0)^2, binned    (Envelope::render)
//! q_k = |A_k(nu)|^2 of the *rendered atom*, binned           (one rfft)
//! ```
//!
//! Since a product space constrained only by its marginals has the product of those marginals as
//! its maximum-entropy solution, this is exactly "the least-committed joint density consistent with
//! what the atom actually is in time and in frequency" — a statement, not a hedge. It is also where
//! a smoothed pseudo-Wigner distribution lands once the smoothing kernel is wide enough to kill the
//! interference structure.
//!
//! **What it gets right, exactly:** the time marginal is the atom's true rendered power envelope;
//! the frequency marginal is its true sampled energy spectral density, carrying the `alpha/PI`
//! width, the negative-frequency image and the Nyquist fold; the total mass is exactly `E_k`, so
//! the whole map sums to the energy the book removed; and every cell is non-negative.
//!
//! **What it loses:** the coupling between them. The true Wigner-Ville distribution of a causal
//! decaying exponential is a *wedge* — `W(t,nu) = 2 e^(-2*alpha*t) * sin(2*t*dw)/dw` — whose
//! frequency half-width to the first null is `1/(4t)`: infinite at the onset, collapsing
//! hyperbolically through the decay, with negative sinc sidelobes. It has precisely the marginals
//! reproduced here, but they are not independent. The product form replaces the wedge with a
//! constant half-width `alpha/(2*PI)`, which is too narrow over the first 23% of an atom's life
//! (`t < PI/(2*alpha)`, against a support of `6.9/alpha`) and too wide over the rest. It is not a
//! skew or a chirp — the carrier really is constant — it is a loss of time-varying bandwidth.
//!
//! The upgrade, if the wedge is ever wanted, is not to add a skew: it is a per-atom
//! adaptive-window spectrogram with the window tied to `1/alpha`, which is a genuine Cohen-class
//! member and stays non-negative, at the cost of many transforms per atom and marginals that are
//! then only approximate.
//!
//! # Three decisions that are not obvious
//!
//! **The frequency marginal transforms the rendered atom, not an envelope spectrum shifted to
//! `f_k`.** Shifting drops the negative-frequency image, the cross term between the two images, and
//! the Nyquist fold. That is not a tail correction: on a real 5000-atom book, 523 atoms have
//! `2f < alpha/PI`, where the two images overlap and the cross term is order one. Transforming the
//! atom costs one extra render — tens of milliseconds over a whole book — and deletes all three
//! approximations along with the shift/fold index arithmetic that would otherwise be needed.
//!
//! **The time marginal stays on the envelope.** `a(n)^2` carries a `sin^2` ripple at `2f`; at a few
//! milliseconds per time bin, a low carrier's ripple would show as banding. `E^2` is the atom's
//! analytic power, and it is the same definition of occupancy `signal::overlap` uses.
//!
//! **Every kernel value is mass integrated over a bin, never a density sampled at its centre.**
//! This is what makes the enormous alpha range a non-issue. At a typical grid size one time bin is
//! a few hundred samples and one frequency bin a few Hz, so a small-alpha atom is many time bins by
//! *less than one* frequency bin while a large-alpha atom is *less than one* time bin by hundreds
//! of frequency bins. Depositing integrated mass handles both exactly, with no multi-resolution
//! grid and no special case; point sampling would make a narrowband atom's total depend on where
//! its carrier sat inside a bin, flickering along the time axis and breaking energy conservation.
//!
//! # A cache that looks obvious and is not
//!
//! The envelope shape in normalized time `u = alpha*t` depends only on `(alpha*beta, fade_level)`,
//! with `alpha` a pure time scale, so `|E^(nu)|^2 = alpha^-2 |S^(nu/alpha)|^2` — spectra at equal
//! `alpha*beta` agree to 0.05 dB down to -80 dB. It is tempting to key a cache on `alpha*beta`
//! alone. Don't: that identity is continuous-time, and the *sampled* spectrum aliases. At
//! `alpha = 3994` against a reference rendered at `alpha = 81`, the error is +2.6 dB at five
//! half-widths and +13.7 dB at sixteen — and five half-widths is still inside the audio band. A
//! cache keyed on `(alpha, beta)` instead collapses nothing on a refined book (4752 of 5000 atoms
//! have distinct bits), so there is no cheap win here and the module does not reach for one.

use crate::book::{Book, Selection};
use crate::fft::{Complex32, Planner, RealFft, RealFftPlanner, next_fast_len};
use crate::fof::{Envelope, FofError};
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Range;

/// Where the map lives and at what resolution.
///
/// Edges rather than `(lo, hi, n)`: the frequency kernel is a difference of a cumulative evaluated
/// at arbitrary edges, so non-uniform bins are already the general case and a log-frequency axis
/// costs nothing but a different constructor.
#[derive(Clone, Debug)]
pub struct MapGrid {
    /// Time bin edges in samples, ascending, `n_t + 1` long. Signed: `t0` may be negative.
    pub t_edges: Vec<f64>,
    /// Frequency bin edges in Hz, ascending, `n_f + 1` long.
    pub f_edges: Vec<f32>,
    pub sample_rate: f32,
}

impl MapGrid {
    /// Uniform bins on both axes.
    pub fn linear(t: Range<f64>, n_t: usize, f: Range<f32>, n_f: usize, sample_rate: f32) -> Self {
        let (n_t, n_f) = (n_t.max(1), n_f.max(1));
        Self {
            t_edges: (0..=n_t)
                .map(|i| t.start + (t.end - t.start) * i as f64 / n_t as f64)
                .collect(),
            f_edges: (0..=n_f)
                .map(|i| f.start + (f.end - f.start) * i as f32 / n_f as f32)
                .collect(),
            sample_rate,
        }
    }

    /// Uniform in time, geometric in frequency.
    ///
    /// Worth having because a decomposition's content routinely spans six octaves, of which a
    /// linear axis gives the bottom two a few percent of the plot. `f.start` is clamped to
    /// something positive, since zero has no place on a geometric axis.
    pub fn log_freq(t: Range<f64>, n_t: usize, f: Range<f32>, n_f: usize, sample_rate: f32) -> Self {
        let (n_t, n_f) = (n_t.max(1), n_f.max(1));
        let lo = f.start.max(1e-3);
        let hi = f.end.max(lo * 1.000_01);
        Self {
            t_edges: (0..=n_t)
                .map(|i| t.start + (t.end - t.start) * i as f64 / n_t as f64)
                .collect(),
            f_edges: (0..=n_f)
                .map(|i| (lo.ln() + (hi.ln() - lo.ln()) * i as f32 / n_f as f32).exp())
                .collect(),
            sample_rate,
        }
    }

    /// Bounds covering every atom's support and carrier.
    ///
    /// The book records no signal length, so the extent is derived from the atoms. The frequency
    /// ceiling stops at `1.25 * max(f + alpha/PI)` rather than Nyquist: a decomposition topping out
    /// a few kHz below Nyquist would otherwise spend most of the plot on empty band.
    pub fn covering(
        book: &Book,
        n_t: usize,
        n_f: usize,
        log_freq: bool,
    ) -> Result<Self, FofError> {
        let sr = book.sample_rate;
        if book.is_empty() {
            let g = Self::linear(0.0..1.0, n_t, 0.0..sr / 2.0, n_f, sr);
            return Ok(g);
        }

        let mut cache: HashMap<(u8, u32, u32), usize> = HashMap::new();
        let (mut t_lo, mut t_hi) = (i64::MAX, i64::MIN);
        let (mut f_lo, mut f_hi) = (f32::INFINITY, 0.0f32);
        for s in &book.selections {
            let key = s.atom.env.cache_key();
            let support = match cache.get(&key) {
                Some(&n) => n,
                None => {
                    let n = Envelope::render(s.atom.env, sr)?.support_len();
                    cache.insert(key, n);
                    n
                }
            };
            t_lo = t_lo.min(s.atom.t0);
            t_hi = t_hi.max(s.atom.t0 + support as i64);
            f_lo = f_lo.min(s.atom.f);
            f_hi = f_hi.max(s.atom.f + s.atom.env.bandwidth_hz());
        }
        // A single atom, or several sharing an onset, would give a zero-width axis.
        if t_hi <= t_lo {
            t_hi = t_lo + 1;
        }
        let f_top = (1.25 * f_hi).min(sr / 2.0).max(1.0);

        Ok(if log_freq {
            // Half an octave of headroom below the lowest carrier, floored where hearing starts.
            Self::log_freq(
                t_lo as f64..t_hi as f64,
                n_t,
                (0.5 * f_lo).max(20.0)..f_top,
                n_f,
                sr,
            )
        } else {
            Self::linear(t_lo as f64..t_hi as f64, n_t, 0.0..f_top, n_f, sr)
        })
    }

    pub fn n_t(&self) -> usize {
        self.t_edges.len() - 1
    }

    pub fn n_f(&self) -> usize {
        self.f_edges.len() - 1
    }

    /// Whether the frequency axis is geometric, judged by whether the first two bins differ in
    /// width. Only used to pick a spectral resolution, so the loose test is enough.
    fn f_is_log(&self) -> bool {
        self.n_f() >= 2
            && (self.f_edges[1] - self.f_edges[0] - (self.f_edges[2] - self.f_edges[1])).abs()
                > 1e-6 * (self.f_edges[1] - self.f_edges[0]).abs()
    }

    /// Width of the narrowest frequency bin the atom at `f` could land in, for sizing its FFT.
    ///
    /// On a log axis the bins below `f` are narrower than the one containing it, and an atom's
    /// skirt reaches down into them, so the resolution has to be set by the bottom of the axis
    /// rather than by the carrier's own bin.
    pub fn df_at(&self, f: f32) -> f32 {
        if self.f_is_log() {
            self.f_edges[1] - self.f_edges[0]
        } else {
            let w = (self.f_edges[self.n_f()] - self.f_edges[0]) / self.n_f() as f32;
            let _ = f;
            w
        }
    }

    /// Bin containing `t` samples, or `None` outside the axis.
    fn t_bin(&self, t: f64) -> Option<usize> {
        if t < self.t_edges[0] || t >= self.t_edges[self.n_t()] {
            return None;
        }
        // Uniform by construction in both constructors, so this is exact and O(1).
        let w = (self.t_edges[self.n_t()] - self.t_edges[0]) / self.n_t() as f64;
        Some((((t - self.t_edges[0]) / w) as usize).min(self.n_t() - 1))
    }
}

/// Which per-atom energy the map deposits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Weight {
    /// `energy_removed` — measured from the rendered atom, and the quantity whose sum the book's
    /// own energy accounting is written in.
    #[default]
    EnergyRemoved,
    /// `hr_score` where HRMP recorded one, else `energy_removed`.
    HrScore,
    /// `amp^2 * sum(E^2)`: the atom's own energy, before any of it went into cancelling an earlier
    /// atom. The ratio to `energy_removed` says how much of an atom is interference correction.
    AtomEnergy,
}

#[derive(Clone, Copy, Debug)]
pub struct MapOptions {
    pub weight: Weight,
    /// Fine spectral bins per `alpha/PI` width, through the FFT pad. Four gives about 8.8.
    pub spec_oversample: usize,
    /// Skip frequency bins holding less than this fraction of an atom's spectral mass. Bounds the
    /// outer product and stops denormals being deposited; far below any display floor.
    pub cell_floor: f64,
}

impl Default for MapOptions {
    fn default() -> Self {
        Self {
            weight: Weight::default(),
            spec_oversample: 4,
            cell_floor: 1e-12,
        }
    }
}

/// What the dB scale is measured against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Reference {
    /// The largest cell in this map. Self-scaling; two maps are not comparable.
    #[default]
    Max,
    /// The book's `initial_energy`, spread over the grid. Makes two maps of different signals
    /// directly comparable.
    Initial,
}

/// One atom's contribution, before accumulation. Separable, so it is two vectors and not a block.
#[derive(Clone, Debug)]
pub struct AtomKernel {
    /// First time bin `time` applies to.
    pub t_lo: usize,
    /// Mass per time bin, summing to the fraction of the atom's power that stayed on the time
    /// axis. Not renormalized to 1 — see [`atom_kernel`].
    pub time: Vec<f64>,
    /// First frequency bin `freq` applies to.
    pub f_lo: usize,
    /// Mass per frequency bin, summing to the fraction that stayed on the frequency axis.
    pub freq: Vec<f64>,
    /// `E_k`.
    pub weight: f64,
    /// Mass that fell outside the grid on either axis.
    pub clipped: f64,
}

/// The accumulated map, in energy units — not density, not dB.
#[derive(Clone, Debug)]
pub struct TfMap {
    pub grid: MapGrid,
    /// Row-major, `n_t * n_f`.
    pub cells: Vec<f64>,
    /// Energy landing inside the grid.
    pub deposited: f64,
    /// Energy whose atom fell partly or wholly outside it.
    pub clipped: f64,
    pub atoms: usize,
    /// Atoms `Envelope::render` refused — past the `alpha*beta` cliff, or otherwise invalid.
    pub skipped: usize,
}

impl TfMap {
    pub fn total(&self) -> f64 {
        self.cells.iter().sum()
    }

    /// Energy per second per Hz.
    ///
    /// Always use this for display, not the raw cells. On a log-frequency axis the bin widths vary
    /// by more than an order of magnitude down the axis, so raw cells would paint a spurious
    /// gradient; on a linear one it makes the image invariant to grid resolution.
    pub fn density(&self) -> Vec<f64> {
        let (n_t, n_f) = (self.grid.n_t(), self.grid.n_f());
        let dt = (self.grid.t_edges[1] - self.grid.t_edges[0]) / self.grid.sample_rate as f64;
        let mut out = vec![0.0; self.cells.len()];
        for j in 0..n_f {
            let df = (self.grid.f_edges[j + 1] - self.grid.f_edges[j]) as f64;
            let scale = 1.0 / (dt * df).max(f64::MIN_POSITIVE);
            for i in 0..n_t {
                out[i * n_f + j] = self.cells[i * n_f + j] * scale;
            }
        }
        out
    }

    /// Display values: `10*log10(density / reference)`, clamped to `[-floor_db, 0]`.
    pub fn to_db(&self, floor_db: f32, r: Reference) -> Vec<f32> {
        let d = self.density();
        let peak = d.iter().copied().fold(0.0f64, f64::max);
        let reference = match r {
            Reference::Max => peak,
            Reference::Initial => {
                // The same total spread evenly, so a map is measured against the signal it came
                // from rather than against its own loudest cell.
                let dt = (self.grid.t_edges[self.grid.n_t()] - self.grid.t_edges[0])
                    / self.grid.sample_rate as f64;
                let df = (self.grid.f_edges[self.grid.n_f()] - self.grid.f_edges[0]) as f64;
                ((self.deposited + self.clipped) / (dt * df).max(f64::MIN_POSITIVE)).max(peak * 1e-30)
            }
        };
        if reference <= 0.0 {
            return vec![-floor_db; d.len()];
        }
        d.iter()
            .map(|&v| {
                if v <= 0.0 {
                    -floor_db
                } else {
                    (10.0 * (v / reference).log10()).clamp(-floor_db as f64, f64::INFINITY) as f32
                }
            })
            .collect()
    }

    /// Energy per time bin.
    pub fn time_marginal(&self) -> Vec<f64> {
        let n_f = self.grid.n_f();
        self.cells.chunks_exact(n_f).map(|r| r.iter().sum()).collect()
    }

    /// Energy per frequency bin.
    pub fn freq_marginal(&self) -> Vec<f64> {
        let n_f = self.grid.n_f();
        let mut out = vec![0.0; n_f];
        for row in self.cells.chunks_exact(n_f) {
            for (o, v) in out.iter_mut().zip(row) {
                *o += v;
            }
        }
        out
    }
}

/// Per-worker reusable buffers and FFT plans.
///
/// `RealFft::forward` takes `&mut self`, so a plan cannot be shared across rayon workers; and
/// planning is dear enough that re-planning once per atom would dominate. Keyed on length, of
/// which a book uses a few hundred distinct values.
pub struct Scratch {
    planner: Planner,
    plans: HashMap<usize, Box<dyn RealFft>>,
    input: Vec<f32>,
    spectrum: Vec<Complex32>,
    /// Cumulative `|A|^2` over fine bins, `complex_len + 1` long, starting at zero.
    cumulative: Vec<f64>,
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            planner: Planner::new(),
            plans: HashMap::new(),
            input: Vec::new(),
            spectrum: Vec::new(),
            cumulative: Vec::new(),
        }
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Longest transform a single atom may ask for.
///
/// A log axis's bottom bins are hair-thin, and `df_at` reports the narrowest of them, so without a
/// cap a low-frequency axis could demand a transform of millions of points for one atom. At the cap
/// the bottom bins are interpolated rather than integrated, which is the right thing to give up.
const MAX_FFT_LEN: usize = 1 << 16;

/// One atom's separable kernel, or `None` if it misses the grid entirely.
pub fn atom_kernel(
    sel: &Selection,
    sample_rate: f32,
    grid: &MapGrid,
    opts: &MapOptions,
    scratch: &mut Scratch,
) -> Result<Option<AtomKernel>, FofError> {
    let env = Envelope::render(sel.atom.env, sample_rate)?;
    let support = env.support_len();

    let weight = match opts.weight {
        Weight::EnergyRemoved => sel.energy_removed,
        Weight::HrScore => sel.hr_score.unwrap_or(sel.energy_removed),
        Weight::AtomEnergy => (sel.atom.amp as f64).powi(2) * env.energy,
    };
    // Rejects NaN as well as zero — a book can be hand-edited, and a NaN weight would
    // poison every cell it touched.
    if !(weight.is_finite() && weight > 0.0) {
        return Ok(None);
    }

    // ---- time marginal: the atom's own power envelope, summed per bin -------------------------
    //
    // Exact by construction: it is a sum over integer sample indices, so nothing is interpolated
    // and a support shorter than one bin deposits all of its mass in that one bin.
    let (mut t_lo, mut t_hi) = (usize::MAX, 0usize);
    let mut time = vec![0.0f64; grid.n_t()];
    let (mut inside, mut outside) = (0.0f64, 0.0f64);
    for (n, &e) in env.samples.iter().enumerate() {
        let p = (e as f64) * (e as f64);
        if p == 0.0 {
            continue;
        }
        match grid.t_bin(sel.atom.t0 as f64 + n as f64) {
            Some(i) => {
                time[i] += p;
                inside += p;
                t_lo = t_lo.min(i);
                t_hi = t_hi.max(i);
            }
            None => outside += p,
        }
    }
    if inside <= 0.0 {
        // Every sample fell off the time axis; the whole atom is clipped.
        return Ok(Some(AtomKernel {
            t_lo: 0,
            time: Vec::new(),
            f_lo: 0,
            freq: Vec::new(),
            weight,
            clipped: weight,
        }));
    }
    let time_total = inside + outside;
    let time_kept = inside / time_total;
    // Normalized by the atom's *whole* power, so the slice sums to `time_kept` and the axis's own
    // clipping is already carried. Renormalizing to 1 here and reapplying `time_kept` below would
    // apply it twice — invisible on a fixture that clips nothing, and a silent scale error on one
    // that does.
    let time: Vec<f64> = time[t_lo..=t_hi].iter().map(|v| v / time_total).collect();

    // ---- frequency marginal: |A(nu)|^2 of the rendered atom, integrated per bin ---------------
    //
    // The atom, not the envelope: see the module docs on the 10% of atoms whose negative-frequency
    // image overlaps their own.
    let n_fft = pad_len(support, grid.df_at(sel.atom.f), sample_rate, opts.spec_oversample);
    if scratch.input.len() < n_fft {
        scratch.input.resize(n_fft, 0.0);
    }
    let buf = &mut scratch.input[..n_fft];
    // `render_into` overwrites only what it reaches, so the pad must be zeroed first and the atom
    // rendered into exactly its support.
    buf.fill(0.0);
    sel.atom.render_into(sample_rate, &mut buf[..support.min(n_fft)]);

    let plan = scratch
        .plans
        .entry(n_fft)
        .or_insert_with(|| scratch.planner.plan(n_fft));
    let complex_len = plan.complex_len();
    if scratch.spectrum.len() < complex_len {
        scratch.spectrum.resize(complex_len, Complex32::default());
    }
    plan.forward(buf, &mut scratch.spectrum[..complex_len]);

    // Cumulative of `|A|^2` over fine bins, so a display bin's mass is one subtraction and the sum
    // over display bins telescopes exactly.
    //
    // Trapezoid rather than a left Riemann sum. It is the same arithmetic and the same transform,
    // and it takes the discretization from first order to second: measured against the direct
    // oracle, the worst cell disagreement falls from 0.81 dB to under 0.05 dB at the default
    // oversampling. A Riemann cumulative converges too, but only by paying for a longer transform.
    scratch.cumulative.clear();
    scratch.cumulative.reserve(complex_len);
    scratch.cumulative.push(0.0);
    let mut acc = 0.0f64;
    let p = |c: &Complex32| (c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64);
    for w in scratch.spectrum[..complex_len].windows(2) {
        acc += 0.5 * (p(&w[0]) + p(&w[1]));
        scratch.cumulative.push(acc);
    }
    let spec_total = acc;
    if spec_total <= 0.0 {
        return Ok(None);
    }

    // Fine bin index of a frequency, as a real number: bin m sits at m * sr / n_fft. The table
    // spans bins `0..=complex_len - 1`, i.e. DC to Nyquist.
    let last = complex_len - 1;
    let bin_of = |hz: f32| (hz as f64) * n_fft as f64 / sample_rate as f64;
    let cum = |x: f64| -> f64 {
        // Linear interpolation of the cumulative, clamped to its ends. The cumulative is
        // non-decreasing, so every difference taken below is non-negative and a display bin can
        // never be handed negative mass.
        let x = x.clamp(0.0, last as f64);
        let i = x.floor() as usize;
        if i >= last {
            return scratch.cumulative[last];
        }
        let f = x - i as f64;
        scratch.cumulative[i] * (1.0 - f) + scratch.cumulative[i + 1] * f
    };

    let mut freq = vec![0.0f64; grid.n_f()];
    let mut f_inside = 0.0f64;
    let mut prev = cum(bin_of(grid.f_edges[0]));
    for (j, slot) in freq.iter_mut().enumerate() {
        let next = cum(bin_of(grid.f_edges[j + 1]));
        let m = (next - prev).max(0.0);
        prev = next;
        if m > opts.cell_floor * spec_total {
            *slot = m;
            f_inside += m;
        }
    }
    if f_inside <= 0.0 {
        return Ok(Some(AtomKernel {
            t_lo: 0,
            time: Vec::new(),
            f_lo: 0,
            freq: Vec::new(),
            weight,
            clipped: weight,
        }));
    }
    let freq_kept = f_inside / spec_total;
    let f_lo = freq.iter().position(|&v| v > 0.0).unwrap();
    let f_hi = grid.n_f() - 1 - freq.iter().rev().position(|&v| v > 0.0).unwrap();
    // Same convention as the time marginal: normalized by the whole spectrum, so this sums to
    // `freq_kept`.
    let freq: Vec<f64> = freq[f_lo..=f_hi].iter().map(|v| v / spec_total).collect();

    // Each marginal carries its own axis's clipping, so their product sums to
    // `time_kept * freq_kept` with no further scaling. What fell off is reported rather than being
    // quietly concentrated into the edge bins.
    let kept = time_kept * freq_kept;
    Ok(Some(AtomKernel {
        t_lo,
        time,
        f_lo,
        freq,
        weight,
        clipped: weight * (1.0 - kept),
    }))
}

/// Transform length for one atom.
///
/// `oversample * support` puts a fixed number of fine bins across the `-3 dB` width whatever alpha
/// is — `sr/(4L)` against `alpha/PI` with `L ~= 6.9*sr/alpha` is about 8.8 at the default — which is
/// the property that lets one rule serve an 83-sample atom and a 33000-sample one. The `df` term
/// additionally resolves the display bin the carrier lands in, so a narrowband atom is integrated
/// rather than interpolated.
fn pad_len(support: usize, df: f32, sample_rate: f32, oversample: usize) -> usize {
    let by_support = support.saturating_mul(oversample.max(1));
    let by_display = if df > 0.0 {
        (2.0 * sample_rate / df).ceil() as usize
    } else {
        0
    };
    next_fast_len(by_support.max(by_display).max(support).min(MAX_FFT_LEN))
}

/// Accumulate one kernel into the grid.
fn accumulate(cells: &mut [f64], n_f: usize, k: &AtomKernel) {
    for (di, &pt) in k.time.iter().enumerate() {
        if pt == 0.0 {
            continue;
        }
        let w = k.weight * pt;
        let row = &mut cells[(k.t_lo + di) * n_f..][..n_f];
        for (dj, &pf) in k.freq.iter().enumerate() {
            row[k.f_lo + dj] += w * pf;
        }
    }
}

/// Build the map.
///
/// The kernels are computed in parallel and folded in serially, in book order. That is deliberate:
/// per-thread accumulators would cost more to reduce than the fold takes, and fixing the f64
/// addition order to book order makes the result **bit-identical regardless of thread count** —
/// the same standard `mp::refresh_frames` is held to, and the only thing that gives a determinism
/// gate any teeth.
pub fn compute(book: &Book, grid: MapGrid, opts: &MapOptions) -> Result<TfMap, FofError> {
    let sr = book.sample_rate;
    let (n_t, n_f) = (grid.n_t(), grid.n_f());

    let kernels: Vec<Result<Option<AtomKernel>, FofError>> = book
        .selections
        .par_iter()
        .map_init(Scratch::new, |scratch, s| {
            atom_kernel(s, sr, &grid, opts, scratch)
        })
        .collect();

    let mut cells = vec![0.0f64; n_t * n_f];
    let (mut deposited, mut clipped, mut atoms, mut skipped) = (0.0, 0.0, 0usize, 0usize);
    for k in kernels {
        match k {
            // A render failure is a property of one atom, not of the book: report the count and
            // draw the rest, rather than refusing to draw anything.
            Err(_) => skipped += 1,
            Ok(None) => skipped += 1,
            Ok(Some(k)) => {
                atoms += 1;
                deposited += k.weight - k.clipped;
                clipped += k.clipped;
                accumulate(&mut cells, n_f, &k);
            }
        }
    }

    Ok(TfMap {
        grid,
        cells,
        deposited,
        clipped,
        atoms,
        skipped,
    })
}

/// The heat ramp a pseudo-Wigner map is read through: a normalised level to RGB.
///
/// `u` runs 0 at the display floor to 1 at the reference — `(db + floor) / floor` over
/// [`TfMap::to_db`]'s output — and is clamped, so a caller need not.
///
/// Here rather than in a front end because both of them draw the same map, and a diagnostic that
/// coloured differently in a chart than in a window would be worth less than one that did not exist.
/// It lives beside `to_db` for the same reason: the map's scaling and its colouring are one
/// decision about how to read it, not two.
///
/// Black through violet and orange to a pale yellow — a spectrogram ramp anchored at true black,
/// so silence reads as empty rather than as the coloured field viridis's dark blue would give.
/// Piecewise-linear through five stops, and **monotone in luminance**, which is the property that
/// makes a level readable off the map at all and what `the_heat_ramp_is_monotone_in_luminance`
/// pins.
pub fn heat(u: f64) -> [u8; 3] {
    const STOPS: [(f64, f64, f64, f64); 5] = [
        (0.00, 0.0, 0.0, 0.0),
        (0.30, 40.0, 20.0, 110.0),
        (0.55, 150.0, 30.0, 110.0),
        (0.80, 240.0, 110.0, 40.0),
        (1.00, 255.0, 255.0, 210.0),
    ];
    let u = u.clamp(0.0, 1.0);
    let mut i = 0;
    while i + 2 < STOPS.len() && u > STOPS[i + 1].0 {
        i += 1;
    }
    let (u0, r0, g0, b0) = STOPS[i];
    let (u1, r1, g1, b1) = STOPS[i + 1];
    let t = ((u - u0) / (u1 - u0)).clamp(0.0, 1.0);
    [
        (r0 + (r1 - r0) * t) as u8,
        (g0 + (g1 - g0) * t) as u8,
        (b0 + (b1 - b0) * t) as u8,
    ]
}

#[cfg(test)]
mod tests {

    /// The property that makes a level readable off the map: brighter always means more.
    #[test]
    fn the_heat_ramp_is_monotone_in_luminance() {
        let lum = |c: [u8; 3]| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64;
        assert_eq!(heat(0.0), [0, 0, 0]);
        let mut prev = -1.0;
        for i in 0..=64 {
            let l = lum(heat(i as f64 / 64.0));
            assert!(l > prev, "luminance fell at {i}");
            prev = l;
        }
    }

    /// The ramp moved here from `rmpstat`'s renderer, and these are the bytes the five-stop table
    /// gave *before* the move — computed from it independently, so this checks the move rather than
    /// merely re-testing whatever arrived.
    #[test]
    fn the_heat_ramp_is_the_one_rmpstat_had() {
        assert_eq!(heat(0.00), [0, 0, 0]);
        assert_eq!(heat(0.15), [20, 10, 55]);
        assert_eq!(heat(0.30), [40, 20, 110]);
        assert_eq!(heat(0.55), [150, 30, 110]);
        assert_eq!(heat(0.80), [240, 110, 40]);
        assert_eq!(heat(1.00), [255, 255, 210]);
    }

    /// Clamped at both ends, so a caller handing it an out-of-range level gets the end colour
    /// rather than a wrapped byte.
    #[test]
    fn the_heat_ramp_clamps_rather_than_wrapping() {
        assert_eq!(heat(-0.5), heat(0.0));
        assert_eq!(heat(1.5), heat(1.0));
        assert_eq!(heat(f64::NAN), [0, 0, 0], "NaN clamps to the floor, not to a random colour");
    }

    use super::*;
    use crate::fof::{AtomParams, EnvelopeParams};

    const SR: f32 = 48_000.0;

    fn sel(t0: i64, f: f32, alpha: f32, beta: f32, amp: f32) -> Selection {
        let atom = AtomParams {
            t0,
            f,
            env: EnvelopeParams::new(alpha, beta).into(),
            phi: 0.3,
            amp,
        };
        let energy = Envelope::render(atom.env, SR).unwrap().energy * (amp as f64).powi(2);
        Selection {
            atom,
            block: 0,
            onset: t0.max(0) as usize,
            bin: 1,
            projected_energy: energy,
            energy_removed: energy,
            residual_energy: 0.0,
            hr_score: None,
            refined: false,
        }
    }

    fn book_of(sels: Vec<Selection>) -> Book {
        let initial: f64 = sels.iter().map(|s| s.energy_removed).sum();
        let mut b = Book::new(initial, SR);
        b.selections = sels;
        b
    }

    /// A deliberately adversarial fixture: alpha over a 200x range, `alpha*beta` from 0.002 to the
    /// grid cap, an atom whose negative-frequency image overlaps its own (`2f < alpha/PI`), one
    /// starting before the origin, and one running past the end.
    fn hard_book() -> Book {
        book_of(vec![
            sel(2_000, 1000.0, 20.0, 0.000_1, 1.0),
            sel(4_000, 2000.0, 300.0, 0.001, 0.5),
            sel(6_000, 400.0, 2000.0, 0.002, 0.8),
            sel(8_000, 3000.0, 4000.0, 0.001, 0.3),
            // 2f = 60 Hz against a bandwidth of 1273 Hz: the images overlap outright.
            sel(5_000, 30.0, 4000.0, 0.000_5, 0.6),
            sel(3_000, 800.0, 1000.0, 0.004, 0.4), // alpha*beta = 4, the cap
            sel(-500, 600.0, 500.0, 0.001, 0.7),   // starts before the origin
            sel(11_500, 1500.0, 200.0, 0.001, 0.5), // runs off the end
        ])
    }

    // ---- structural gate 1: energy conservation ------------------------------------------------

    /// Every atom's mass is either deposited or reported as clipped — never lost, never doubled.
    /// This is the gate that catches a normalization error or a cumulative that fails to telescope.
    #[test]
    fn energy_is_conserved() {
        let b = hard_book();
        let grid = MapGrid::covering(&b, 96, 64, false).unwrap();
        let map = compute(&b, grid, &MapOptions::default()).unwrap();

        let expect: f64 = b.selections.iter().map(|s| s.energy_removed).sum();
        assert_eq!(map.skipped, 0);
        let got = map.deposited + map.clipped;
        assert!(
            (got - expect).abs() < 1e-9 * expect,
            "deposited {} + clipped {} != {expect}",
            map.deposited,
            map.clipped
        );
        // And `deposited` really is what is in the cells.
        assert!((map.total() - map.deposited).abs() < 1e-9 * expect);
    }

    /// On a grid covering every atom, the map holds exactly the energy the book removed.
    #[test]
    fn a_covering_grid_clips_nothing() {
        let b = book_of(vec![
            sel(2_000, 1000.0, 300.0, 0.001, 1.0),
            sel(6_000, 2000.0, 800.0, 0.001, 0.5),
        ]);
        // Pad the frequency axis to Nyquist so no skirt escapes off the top.
        let base = MapGrid::covering(&b, 128, 256, false).unwrap();
        let grid = MapGrid::linear(
            base.t_edges[0] - 1.0..base.t_edges[base.n_t()] + 1.0,
            128,
            0.0..SR / 2.0,
            256,
            SR,
        );
        let map = compute(&b, grid, &MapOptions::default()).unwrap();
        let expect: f64 = b.selections.iter().map(|s| s.energy_removed).sum();
        assert!(
            map.clipped < 1e-6 * expect,
            "clipped {} of {expect}",
            map.clipped
        );
        assert!((map.total() - expect).abs() < 1e-6 * expect);
    }

    // ---- structural gate 2: the whole time marginal --------------------------------------------

    /// Compare the *whole* marginal vector, not a summary of it.
    ///
    /// The weak form has no teeth — an off-by-one in the accumulator perturbs the total by
    /// nothing, because the mass is merely in the wrong column. The bin-by-bin comparison is what
    /// catches it.
    #[test]
    fn time_marginal_matches_an_independent_binning() {
        let b = hard_book();
        let grid = MapGrid::covering(&b, 96, 64, false).unwrap();
        let map = compute(&b, grid.clone(), &MapOptions::default()).unwrap();

        // Independently: for each atom, bin E^2 by hand and scale by its weight and kept fraction.
        let mut want = vec![0.0f64; grid.n_t()];
        for s in &b.selections {
            let env = Envelope::render(s.atom.env, SR).unwrap();
            let mut bins = vec![0.0f64; grid.n_t()];
            let (mut inside, mut total) = (0.0f64, 0.0f64);
            for (n, &e) in env.samples.iter().enumerate() {
                let p = (e as f64) * (e as f64);
                total += p;
                if let Some(i) = grid.t_bin(s.atom.t0 as f64 + n as f64) {
                    bins[i] += p;
                    inside += p;
                }
            }
            // The frequency axis also clips, so scale by what the map actually kept there.
            let mut scratch = Scratch::new();
            let k = atom_kernel(s, SR, &grid, &MapOptions::default(), &mut scratch)
                .unwrap()
                .unwrap();
            let f_kept: f64 = k.freq.iter().sum();
            let _ = inside;
            for (w, v) in want.iter_mut().zip(&bins) {
                *w += s.energy_removed * (v / total) * f_kept;
            }
        }

        let got = map.time_marginal();
        let scale: f64 = want.iter().sum();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-9 * scale,
                "time bin {i}: {g} vs {w}"
            );
        }
    }

    // ---- structural gate 3: the frequency marginal is physics, not a shape ---------------------

    /// The marginal has the atom's real bandwidth and centre, and — the discriminating part — the
    /// right *tail*. A test asserting only "the peak is at f" passes with a Lorentzian model, which
    /// is 24 dB high at five half-widths for an `alpha*beta = 1` atom and would smear the whole
    /// plot. Checking a far-out level is what rules that out.
    #[test]
    fn frequency_marginal_has_the_right_width_centre_and_tail() {
        let (f0, alpha) = (2000.0f32, 400.0f32);
        let bw = alpha / std::f32::consts::PI; // 127.3 Hz
        let b = book_of(vec![sel(4_000, f0, alpha, 0.002, 1.0)]);
        // Fine, linear, full-band: about 4 bins across the -3 dB width.
        let grid = MapGrid::linear(0.0..40_000.0, 8, 0.0..SR / 2.0, 800, SR);
        let map = compute(&b, grid.clone(), &MapOptions::default()).unwrap();
        let m = map.freq_marginal();
        let df = SR / 2.0 / 800.0; // 30 Hz

        let peak_bin = m
            .iter()
            .enumerate()
            .max_by(|a, c| a.1.total_cmp(c.1))
            .unwrap()
            .0;
        let peak_hz = (peak_bin as f32 + 0.5) * df;
        assert!((peak_hz - f0).abs() <= df, "peak at {peak_hz}, want {f0}");

        // Centroid over the carrier's own lobe.
        let lo = ((f0 - 4.0 * bw) / df) as usize;
        let hi = (((f0 + 4.0 * bw) / df) as usize).min(m.len() - 1);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (j, &v) in m.iter().enumerate().take(hi + 1).skip(lo) {
            let hz = (j as f32 + 0.5) * df;
            num += v * hz as f64;
            den += v;
        }
        let centroid = (num / den) as f32;
        assert!((centroid - f0).abs() < df, "centroid {centroid}, want {f0}");

        // Half-power width, measured in density so the bin width divides out.
        let peak = m[peak_bin];
        let n_half = m[lo..=hi].iter().filter(|&&v| v >= 0.5 * peak).count();
        let width = (n_half as f32) * df;
        assert!(
            (width - bw).abs() < 0.35 * bw,
            "half-power width {width} Hz, want ~{bw}"
        );

        // The tail. A Lorentzian at 10 half-widths sits at -20 dB; the attack cuts it well below
        // that. Assert the map is under -26 dB there, which a Lorentzian model could not be.
        let far = ((f0 + 10.0 * bw) / df) as usize;
        let tail_db = 10.0 * (m[far] / peak).log10();
        assert!(
            tail_db < -26.0,
            "level at 10 half-widths is {tail_db:.1} dB — a Lorentzian tail would be near -20"
        );
    }

    // ---- structural gate 4: shift invariance ---------------------------------------------------

    /// Moving an atom by exactly one bin's worth of samples moves the map by exactly one column,
    /// bit for bit. Catches a rounding or edge-assignment bug that a tolerance would hide.
    #[test]
    fn a_one_bin_shift_moves_the_map_one_column() {
        let bin_samples = 256i64;
        let span = 0.0..(64 * bin_samples) as f64;
        let grid = MapGrid::linear(span.clone(), 64, 0.0..SR / 2.0, 128, SR);

        let a = book_of(vec![sel(8_000, 1200.0, 500.0, 0.001, 1.0)]);
        let b = book_of(vec![sel(8_000 + bin_samples, 1200.0, 500.0, 0.001, 1.0)]);
        let ma = compute(&a, grid.clone(), &MapOptions::default()).unwrap();
        let mb = compute(&b, grid.clone(), &MapOptions::default()).unwrap();

        let n_f = grid.n_f();
        for i in 0..grid.n_t() - 1 {
            assert_eq!(
                &ma.cells[i * n_f..(i + 1) * n_f],
                &mb.cells[(i + 1) * n_f..(i + 2) * n_f],
                "column {i} did not shift"
            );
        }
    }

    // ---- the oracle ----------------------------------------------------------------------------

    /// Direct evaluation — the oracle. Shares nothing with the fast path but `AtomParams` and the
    /// grid: no FFT, no cumulative, no interpolation, no pad heuristic. `|A(nu)|^2` comes from a
    /// direct f64 DFT sum evaluated at `sub` points across each display bin and trapezoided; the
    /// time marginal is summed by hand.
    fn naive_map(book: &Book, grid: &MapGrid, sub: usize) -> Vec<f64> {
        use std::f64::consts::TAU;
        let sr = book.sample_rate as f64;
        let (n_t, n_f) = (grid.n_t(), grid.n_f());
        let mut cells = vec![0.0f64; n_t * n_f];

        for s in &book.selections {
            let env = Envelope::render(s.atom.env, book.sample_rate).unwrap();
            let atom = s.atom.render(book.sample_rate).unwrap();

            let mut time = vec![0.0f64; n_t];
            let mut t_total = 0.0f64;
            for (n, &e) in env.samples.iter().enumerate() {
                let p = (e as f64) * (e as f64);
                t_total += p;
                if let Some(i) = grid.t_bin(s.atom.t0 as f64 + n as f64) {
                    time[i] += p;
                }
            }
            let t_inside: f64 = time.iter().sum();
            if t_inside <= 0.0 {
                continue;
            }

            // |A(nu)|^2 by direct summation, at `sub + 1` points across each bin.
            let psd = |hz: f64| -> f64 {
                let w = TAU * hz / sr;
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (n, &x) in atom.iter().enumerate() {
                    let (sn, cs) = (w * n as f64).sin_cos();
                    re += x as f64 * cs;
                    im -= x as f64 * sn;
                }
                re * re + im * im
            };

            let mut freq = vec![0.0f64; n_f];
            for (j, slot) in freq.iter_mut().enumerate() {
                let (a, c) = (grid.f_edges[j] as f64, grid.f_edges[j + 1] as f64);
                let h = (c - a) / sub as f64;
                let mut acc = 0.0;
                for m in 0..=sub {
                    let v = psd(a + h * m as f64);
                    // Trapezoid: the ends count half.
                    acc += if m == 0 || m == sub { 0.5 * v } else { v };
                }
                *slot = acc * h;
            }
            let f_inside: f64 = freq.iter().sum();
            if f_inside <= 0.0 {
                continue;
            }
            // The oracle knows the whole-axis total only up to Nyquist, which is where the grid
            // ends in these fixtures, so normalize over what it integrated.
            for i in 0..n_t {
                if time[i] == 0.0 {
                    continue;
                }
                let w = s.energy_removed * (time[i] / t_total);
                for j in 0..n_f {
                    cells[i * n_f + j] += w * freq[j] / f_inside;
                }
            }
        }
        cells
    }

    /// The fast path against the oracle, on a grid small enough for an O(N*M) reference.
    #[test]
    fn matches_the_direct_oracle() {
        let b = book_of(vec![
            sel(2_000, 1500.0, 300.0, 0.001, 1.0),
            sel(5_000, 700.0, 1200.0, 0.002, 0.6),
            // The overlapping-image case — the one a shifted envelope spectrum would get wrong.
            sel(3_000, 40.0, 3000.0, 0.000_5, 0.8),
        ]);
        // Full band and a wide time span, so neither path clips and the two normalizations agree.
        // 187 Hz bins resolve the narrowest atom's 95 Hz width; the oracle stays affordable
        // because these supports are short (alpha 300 is 1104 samples).
        let grid = MapGrid::linear(0.0..24_000.0, 16, 0.0..SR / 2.0, 128, SR);
        let map = compute(&b, grid.clone(), &MapOptions::default()).unwrap();
        let want = naive_map(&b, &grid, 16);

        let peak = want.iter().copied().fold(0.0f64, f64::max);
        let floor = peak * 1e-8; // -80 dB: far enough out to test the tails, not just the lobes
        let (mut checked, mut worst) = (0usize, 0.0f64);
        for (g, w) in map.cells.iter().zip(&want) {
            if *w < floor {
                continue;
            }
            checked += 1;
            let db = 10.0 * (g / w).log10();
            worst = worst.max(db.abs());
        }
        assert!(checked > 100, "only {checked} cells above the floor");
        // Measured at 0.152 dB. The residue is the fast path's finite spectral resolution, not a
        // shape error: it falls first-order with `spec_oversample` and the oracle's own
        // sub-sampling barely moves it.
        assert!(
            worst < 0.2,
            "worst disagreement {worst:.3} dB over {checked} cells"
        );
    }

    // ---- axes, display, and the book-derived extent ---------------------------------------------

    #[test]
    fn log_frequency_edges_are_geometric_and_conserve_energy() {
        let b = hard_book();
        let grid = MapGrid::covering(&b, 64, 48, true).unwrap();
        for i in 0..grid.n_f() - 1 {
            let (r0, r1) = (
                grid.f_edges[i + 1] / grid.f_edges[i],
                grid.f_edges[i + 2] / grid.f_edges[i + 1],
            );
            assert!((r0 - r1).abs() < 1e-4 * r0, "edges not geometric: {r0} {r1}");
        }
        let map = compute(&b, grid, &MapOptions::default()).unwrap();
        let expect: f64 = b.selections.iter().map(|s| s.energy_removed).sum();
        assert!((map.deposited + map.clipped - expect).abs() < 1e-9 * expect);
    }

    /// The extent reaches every atom, including one starting before the origin.
    #[test]
    fn covering_spans_every_atom() {
        let b = hard_book();
        let grid = MapGrid::covering(&b, 64, 48, false).unwrap();
        let min_t0 = b.selections.iter().map(|s| s.atom.t0).min().unwrap();
        assert!(grid.t_edges[0] <= min_t0 as f64);
        assert!(min_t0 < 0, "the fixture is supposed to include a negative t0");
        for s in &b.selections {
            let end = s.atom.t0 + Envelope::render(s.atom.env, SR).unwrap().support_len() as i64;
            assert!(grid.t_edges[grid.n_t()] >= end as f64);
        }
    }

    /// Refining the grid subdivides the same measure rather than resampling a different one.
    ///
    /// This is the property that makes integrating per bin — rather than sampling a density at bin
    /// centres — the right choice, and it is exact enough to assert tightly: summing each 2x2
    /// block of the fine map must reproduce the coarse map, because both are integrals of one
    /// measure over nested edges. Peak density is *not* the invariant to test here; the attack is a
    /// genuine sharp maximum, so halving the bin around it raises the measured peak however
    /// correct the map is.
    #[test]
    fn refining_the_grid_subdivides_the_same_measure() {
        let b = book_of(vec![
            sel(1_000, 1000.0, 400.0, 0.001, 1.0),
            sel(3_000, 2500.0, 1200.0, 0.002, 0.6),
        ]);
        let opts = MapOptions::default();
        let coarse = compute(
            &b,
            MapGrid::linear(0.0..8192.0, 32, 0.0..SR / 2.0, 128, SR),
            &opts,
        )
        .unwrap();
        let fine = compute(
            &b,
            MapGrid::linear(0.0..8192.0, 64, 0.0..SR / 2.0, 256, SR),
            &opts,
        )
        .unwrap();

        let peak = coarse.cells.iter().copied().fold(0.0f64, f64::max);
        let (mut checked, mut worst) = (0usize, 0.0f64);
        for i in 0..32 {
            for j in 0..128 {
                let want = coarse.cells[i * 128 + j];
                let got: f64 = [(0, 0), (0, 1), (1, 0), (1, 1)]
                    .iter()
                    .map(|&(di, dj)| fine.cells[(2 * i + di) * 256 + 2 * j + dj])
                    .sum();
                if want < peak * 1e-8 {
                    continue;
                }
                checked += 1;
                worst = worst.max((got / want - 1.0).abs());
            }
        }
        assert!(checked > 100, "only {checked} cells above the floor");
        // Measured exactly zero here: both grids happen to size their transforms the same, so the
        // two integrate the identical fine spectrum and the differences telescope. The bound stays
        // loose because a grid pair that picked different transform lengths would not be exact.
        assert!(worst < 0.01, "worst re-binning error {:.4}% over {checked} cells", worst * 100.0);
    }

    #[test]
    fn db_is_floored_and_topped_at_zero() {
        let b = hard_book();
        let grid = MapGrid::covering(&b, 48, 32, false).unwrap();
        let map = compute(&b, grid, &MapOptions::default()).unwrap();
        let db = map.to_db(60.0, Reference::Max);
        assert!(db.iter().all(|&v| (-60.0..=0.0).contains(&v)));
        assert!(db.iter().any(|&v| v > -0.001), "nothing reached the top");
        assert!(db.iter().any(|&v| v <= -60.0), "nothing reached the floor");
    }

    #[test]
    fn weights_select_different_energies() {
        let mut b = book_of(vec![sel(4_000, 1000.0, 400.0, 0.001, 1.0)]);
        b.selections[0].hr_score = Some(b.selections[0].energy_removed * 0.25);
        let grid = MapGrid::linear(0.0..24_000.0, 32, 0.0..SR / 2.0, 64, SR);
        let full = compute(&b, grid.clone(), &MapOptions::default()).unwrap();
        let clamped = compute(
            &b,
            grid,
            &MapOptions {
                weight: Weight::HrScore,
                ..Default::default()
            },
        )
        .unwrap();
        assert!((clamped.total() / full.total() - 0.25).abs() < 1e-9);
    }

    /// An atom past the `alpha*beta` cliff renders as silence; the map counts it and draws the rest
    /// rather than refusing to draw anything.
    #[test]
    fn an_unrenderable_atom_is_skipped_not_fatal() {
        let mut b = book_of(vec![
            sel(4_000, 1000.0, 400.0, 0.001, 1.0),
            sel(6_000, 1000.0, 400.0, 0.001, 1.0),
        ]);
        // alpha*beta = 20, past rfofs's cliff at 10.
        b.selections[1].atom.env = EnvelopeParams::new(20_000.0, 0.001).into();
        let grid = MapGrid::linear(0.0..24_000.0, 32, 0.0..SR / 2.0, 64, SR);
        let map = compute(&b, grid, &MapOptions::default()).unwrap();
        assert_eq!((map.atoms, map.skipped), (1, 1));
        assert!(map.total() > 0.0);
    }

    // ---- against the real book -----------------------------------------------------------------

    fn real_book() -> Book {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/books/book1.json");
        crate::book::read(&p).expect("data/books/book1.json")
    }

    /// The map is bit-identical however many threads build it.
    ///
    /// The kernels are computed in parallel but folded in serially, in book order, precisely so
    /// that the f64 addition order does not depend on scheduling. This is the same standard
    /// `mp::refresh_frames` is held to, and it is what lets any later comparison of two maps mean
    /// something.
    #[test]
    fn the_map_does_not_depend_on_the_thread_count() {
        let b = real_book();
        let grid = MapGrid::covering(&b, 300, 200, false).unwrap();
        let many = compute(&b, grid.clone(), &MapOptions::default()).unwrap();
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| compute(&b, grid, &MapOptions::default()).unwrap());
        assert_eq!(many.cells, one.cells);
        assert_eq!(many.deposited.to_bits(), one.deposited.to_bits());
    }

    /// The map holds what the book says the pursuit removed, on real material.
    #[test]
    fn the_real_book_accounts_for_its_energy() {
        let b = real_book();
        let grid = MapGrid::covering(&b, 300, 200, false).unwrap();
        let map = compute(&b, grid, &MapOptions::default()).unwrap();
        assert_eq!(map.skipped, 0);
        let captured = b.initial_energy - b.residual_energy();
        let got = map.deposited + map.clipped;
        assert!(
            (got - captured).abs() < 1e-6 * captured,
            "map holds {got}, the book removed {captured}"
        );
    }

    /// No spurious haze: the map stays sparse at a display floor.
    ///
    /// This is the regression that encodes the kernel choice. A Lorentzian frequency model would
    /// fail it outright — its tails sit above -60 dB out to roughly a thousand half-widths, which
    /// on this material would paint a broadband vertical smear under every atom with a real attack
    /// and drive this fraction toward 1. Measured at 21.3% on a 1200x800 grid.
    #[test]
    fn the_real_book_does_not_haze_the_display() {
        let b = real_book();
        let grid = MapGrid::covering(&b, 600, 400, false).unwrap();
        let map = compute(&b, grid, &MapOptions::default()).unwrap();
        let db = map.to_db(60.0, Reference::Max);
        let lit = db.iter().filter(|&&v| v > -60.0).count() as f64 / db.len() as f64;
        assert!(
            lit < 0.35,
            "{:.1}% of cells are above the -60 dB floor — the kernel is smearing",
            100.0 * lit
        );
        // And it is not empty either, or the bound above would pass for the wrong reason.
        assert!(lit > 0.05, "only {:.1}% of cells are lit", 100.0 * lit);
    }
}
