//! The coarse time-frequency energy map every partial is read from (§9–§10).
//!
//! `P(t, f) = Σ E_i K_t(t − t_i) K_f(f − f_i)`: each observation deposits its energy over a small
//! area rather than into one cell, with Gaussian kernels whose widths come from the atom itself.
//!
//! - **Time.** `σ_t = hypot(window/4, duration/2)`: the analysis window, whose `±2σ` spans
//!   `window_ms`, combined with the atom's own spread, whose `±σ` is its effective duration.
//! - **Frequency.** `σ_f` is the atom's −3 dB half-width expressed in the axis's units and converted
//!   from half-width to standard deviation, never less than half a bin — a narrow atom still lands
//!   on a few bins, so a peak between two bins is interpolable.
//!
//! **Each kernel is normalised over the cells it actually covers, so every atom deposits exactly its
//! energy.** That is what makes the map a fair accumulator of evidence (§8): forty weak atoms along
//! one frequency deposit forty times one of them, and nothing is thresholded before they are summed.
//!
//! **Kernels are truncated with their pedestal removed**: `max(0, g(x) − g(edge))` rather than `g`
//! cut off. A hard cut leaves a step at the edge, and a step in one kernel lying on the slope of
//! another is a local maximum the map does not contain — it showed up as a partial between two
//! harmonics with not one atom supporting it. With the pedestal gone each kernel is continuous and
//! unimodal, and its edges can only ever make a minimum. Time is cut at `±2σ`, deliberately tight:
//! the extent of an isolated event on the map is its kernel, and a wider one would carry a 20 ms
//! transient past the minimum duration of a partial by blur alone. Frequency is cut at `±3σ`.
//!
//! **The sums are bit-identical at any thread count.** Frames are cut into fixed chunks and each
//! chunk is one task, which adds the atoms overlapping it in book order. Every cell is therefore the
//! same sequence of f64 additions however the chunks are scheduled.

use crate::config::{FrequencyScale, PartialAnalysisConfig};
use crate::observation::AtomObservation;
use rayon::prelude::*;

/// Kernels are cut at this many standard deviations. See the module docs for why time is tight.
pub const TIME_SIGMAS: f64 = 2.0;
pub const FREQUENCY_SIGMAS: f64 = 3.0;

/// Frames per parallel task.
const CHUNK_FRAMES: usize = 64;

/// Half-width to standard deviation for a Gaussian: `sqrt(2 ln 2)`.
const HWHM_PER_SIGMA: f64 = 1.177_410_022_515_474_6;

/// The frequency axis: a uniform grid in the scale's own unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    pub scale: FrequencyScale,
    /// Units per bin.
    step: f64,
    /// Bin 0's centre, in whole steps from the unit's zero.
    first: i64,
    pub n_bins: usize,
}

impl Axis {
    pub fn new(scale: FrequencyScale, f_min: f64, f_max: f64) -> Self {
        let step = match scale {
            FrequencyScale::LinearHz { bin_width_hz } => bin_width_hz,
            FrequencyScale::LogCents { cents_per_bin, .. } => cents_per_bin,
            FrequencyScale::Erb { bands_per_erb } => 1.0 / bands_per_erb,
        };
        let lo = (unit(scale, f_min) / step).floor() as i64;
        let hi = (unit(scale, f_max) / step).ceil() as i64;
        Self { scale, step, first: lo, n_bins: (hi - lo + 1).max(1) as usize }
    }

    /// Continuous bin position of `f`: bin `b`'s centre is at exactly `b`.
    pub fn position(&self, f: f64) -> f64 {
        unit(self.scale, f) / self.step - self.first as f64
    }

    /// The frequency at a continuous bin position.
    pub fn frequency(&self, position: f64) -> f64 {
        hz(self.scale, (position + self.first as f64) * self.step)
    }

    /// An atom's frequency spread, as a standard deviation in bins.
    fn sigma_bins(&self, f: f64, bandwidth_hz: f64) -> f64 {
        let half = (unit(self.scale, f + 0.5 * bandwidth_hz) - unit(self.scale, f)) / self.step;
        (half / HWHM_PER_SIGMA).max(0.5)
    }
}

/// The scale's unit: cents re `reference_hz`, Hz, or the ERB-rate number (Glasberg & Moore).
fn unit(scale: FrequencyScale, f: f64) -> f64 {
    match scale {
        FrequencyScale::LinearHz { .. } => f,
        FrequencyScale::LogCents { reference_hz, .. } => 1200.0 * (f / reference_hz).log2(),
        FrequencyScale::Erb { .. } => 21.4 * (1.0 + 0.00437 * f).log10(),
    }
}

fn hz(scale: FrequencyScale, u: f64) -> f64 {
    match scale {
        FrequencyScale::LinearHz { .. } => u,
        FrequencyScale::LogCents { reference_hz, .. } => reference_hz * (u / 1200.0).exp2(),
        FrequencyScale::Erb { .. } => (10f64.powf(u / 21.4) - 1.0) / 0.00437,
    }
}

/// The accumulated map, frame-major.
#[derive(Clone, Debug, PartialEq)]
pub struct TfGrid {
    /// Absolute sample of frame 0.
    pub origin: u64,
    /// Samples per frame.
    pub hop: u64,
    pub n_frames: usize,
    pub axis: Axis,
    /// `n_frames * axis.n_bins` energies; frame `k` is `cells[k * n_bins ..][.. n_bins]`.
    pub cells: Vec<f64>,
}

impl TfGrid {
    pub fn frame(&self, k: usize) -> &[f64] {
        let n = self.axis.n_bins;
        &self.cells[k * n..(k + 1) * n]
    }

    /// Absolute sample position of frame `k`.
    pub fn time_of(&self, k: usize) -> u64 {
        self.origin + k as u64 * self.hop
    }

    /// The frame nearest to absolute sample `t`, clamped into the grid.
    pub fn frame_of(&self, t: u64) -> usize {
        let k = ((t.saturating_sub(self.origin)) + self.hop / 2) / self.hop;
        (k as usize).min(self.n_frames.saturating_sub(1))
    }
}

/// One observation's footprint, precomputed so the parallel pass only multiplies and adds.
struct Kernel {
    /// Frames `k_lo..k_hi`.
    k_lo: usize,
    k_hi: usize,
    /// Centre and standard deviation, in frames.
    kc: f64,
    sk: f64,
    /// Bins `b_lo..b_lo + fw.len()`, each carrying `E * w_f / (sum w_f * sum w_t)`.
    b_lo: usize,
    fw: Vec<f64>,
}

/// A Gaussian cut at `±reach` standard deviations with its pedestal removed, so it falls
/// continuously to zero at the cut.
fn gauss(x: f64, s: f64, reach: f64) -> f64 {
    ((-0.5 * (x / s) * (x / s)).exp() - (-0.5 * reach * reach).exp()).max(0.0)
}

/// Cells `lo..hi` of a `±reach` sigma window about `centre`, clipped to `0..n`.
fn span(centre: f64, sigma: f64, reach: f64, n: usize) -> (usize, usize) {
    let lo = (centre - reach * sigma).ceil().max(0.0);
    let hi = ((centre + reach * sigma).floor() + 1.0).min(n as f64);
    if hi <= lo { (0, 0) } else { (lo as usize, hi as usize) }
}

fn kernel(o: &AtomObservation, grid: &TfGrid, window_samples: f64) -> Option<Kernel> {
    let hop = grid.hop as f64;
    let kc = (o.time_center_samples as f64 - grid.origin as f64) / hop;
    let sk = (0.25 * window_samples).hypot(0.5 * o.effective_duration_samples) / hop;
    let (k_lo, k_hi) = span(kc, sk, TIME_SIGMAS, grid.n_frames);

    let bc = grid.axis.position(o.frequency_hz);
    let sb = grid.axis.sigma_bins(o.frequency_hz, o.effective_bandwidth_hz);
    let (b_lo, b_hi) = span(bc, sb, FREQUENCY_SIGMAS, grid.axis.n_bins);
    if k_lo == k_hi || b_lo == b_hi || !bc.is_finite() {
        return None;
    }

    let t_sum: f64 = (k_lo..k_hi).map(|k| gauss(k as f64 - kc, sk, TIME_SIGMAS)).sum();
    let mut fw: Vec<f64> =
        (b_lo..b_hi).map(|b| gauss(b as f64 - bc, sb, FREQUENCY_SIGMAS)).collect();
    let f_sum: f64 = fw.iter().sum();
    // A kernel narrower than a cell can miss every cell centre it spans.
    if !(t_sum > 0.0 && f_sum > 0.0) {
        return None;
    }
    let scale = o.energy / (t_sum * f_sum);
    for w in &mut fw {
        *w *= scale;
    }
    Some(Kernel { k_lo, k_hi, kc, sk, b_lo, fw })
}

/// The grid's time and frequency extent for `observations`.
pub fn layout(
    observations: &[AtomObservation],
    sample_rate: f64,
    start_sample: u64,
    cfg: &PartialAnalysisConfig,
) -> TfGrid {
    let hop = ((cfg.time_step_ms * 1e-3 * sample_rate).round() as u64).max(1);
    let origin = observations.iter().map(|o| o.start_samples).min().unwrap_or(start_sample);
    let origin = origin.min(start_sample);
    let end = observations.iter().map(|o| o.end_samples).max().unwrap_or(origin);
    let n_frames = ((end - origin) / hop + 1) as usize;
    let f_max = cfg.f_max.min(0.5 * sample_rate);
    let axis = Axis::new(cfg.scale(), cfg.f_min, f_max);
    TfGrid { origin, hop, n_frames, axis, cells: Vec::new() }
}

/// Accumulate every observation into `grid`. Returns how many fell entirely outside it.
pub fn accumulate(
    grid: &mut TfGrid,
    observations: &[AtomObservation],
    sample_rate: f64,
    cfg: &PartialAnalysisConfig,
) -> usize {
    let window = cfg.window_ms * 1e-3 * sample_rate;
    let kernels: Vec<Option<Kernel>> =
        observations.par_iter().map(|o| kernel(o, grid, window)).collect();
    let outside = kernels.iter().filter(|k| k.is_none()).count();

    // Each chunk's atoms, in book order.
    let n_chunks = grid.n_frames.div_ceil(CHUNK_FRAMES);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_chunks];
    for (i, k) in kernels.iter().enumerate() {
        if let Some(k) = k {
            for bucket in &mut buckets[k.k_lo / CHUNK_FRAMES..=(k.k_hi - 1) / CHUNK_FRAMES] {
                bucket.push(i as u32);
            }
        }
    }

    let n_bins = grid.axis.n_bins;
    grid.cells = vec![0.0; grid.n_frames * n_bins];
    grid.cells
        .par_chunks_mut(CHUNK_FRAMES * n_bins)
        .zip(buckets.par_iter())
        .enumerate()
        .for_each(|(c, (cells, bucket))| {
            let first = c * CHUNK_FRAMES;
            let last = first + cells.len() / n_bins;
            for &i in bucket {
                let k = kernels[i as usize].as_ref().unwrap();
                for frame in k.k_lo.max(first)..k.k_hi.min(last) {
                    let wt = gauss(frame as f64 - k.kc, k.sk, TIME_SIGMAS);
                    let row = &mut cells[(frame - first) * n_bins..][..n_bins];
                    for (cell, w) in row[k.b_lo..][..k.fw.len()].iter_mut().zip(&k.fw) {
                        *cell += wt * w;
                    }
                }
            }
        });
    outside
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::AtomId;
    use rmp_core::AtomKind;

    fn obs(t: u64, f: f64, e: f64, dur: f64, bw: f64) -> AtomObservation {
        AtomObservation {
            atom_id: AtomId(0),
            channel: None,
            time_center_samples: t,
            start_samples: t.saturating_sub(dur as u64),
            end_samples: t + dur as u64,
            frequency_hz: f,
            energy: e,
            effective_bandwidth_hz: bw,
            effective_duration_samples: dur,
            phase: None,
            atom_kind: AtomKind::Gaussian,
        }
    }

    #[test]
    fn every_scale_maps_its_bins_back_to_the_same_frequencies() {
        for scale in [
            FrequencyScale::LogCents { cents_per_bin: 20.0, reference_hz: 440.0 },
            FrequencyScale::LinearHz { bin_width_hz: 10.0 },
            FrequencyScale::Erb { bands_per_erb: 4.0 },
        ] {
            let axis = Axis::new(scale, 20.0, 20_000.0);
            for f in [25.0, 100.0, 440.0, 1234.5, 19_000.0] {
                let p = axis.position(f);
                assert!(p >= 0.0 && p < axis.n_bins as f64, "{scale:?} {f}");
                assert!((axis.frequency(p) / f - 1.0).abs() < 1e-9, "{scale:?} {f}");
            }
        }
        let cents = Axis::new(FrequencyScale::LogCents { cents_per_bin: 20.0, reference_hz: 440.0 }, 20.0, 20_000.0);
        // 20 Hz .. 20 kHz is 11960 cents: 598 bins, plus the partial bins at each end.
        assert!((598..=601).contains(&cents.n_bins), "{}", cents.n_bins);
    }

    /// Every atom deposits exactly its energy, however it is shaped and wherever it lands.
    #[test]
    fn kernels_deposit_exactly_the_atoms_energy() {
        let cfg = PartialAnalysisConfig::default();
        let sr = 48_000.0;
        let o = vec![
            obs(24_000, 440.0, 1.0, 480.0, 10.0),
            obs(30_000, 3000.0, 0.25, 9600.0, 300.0),
            obs(100, 60.0, 2.0, 48.0, 2.0), // clipped by the start of the grid
        ];
        let mut grid = layout(&o, sr, 0, &cfg);
        assert_eq!(accumulate(&mut grid, &o, sr, &cfg), 0);
        let total: f64 = grid.cells.iter().sum();
        assert!((total - 3.25).abs() < 1e-12, "{total}");
    }

    #[test]
    fn an_atom_outside_the_frequency_range_is_counted_and_deposits_nothing() {
        let cfg = PartialAnalysisConfig { f_min: 100.0, f_max: 1000.0, ..Default::default() };
        let o = vec![obs(24_000, 5000.0, 1.0, 480.0, 10.0)];
        let mut grid = layout(&o, 48_000.0, 0, &cfg);
        assert_eq!(accumulate(&mut grid, &o, 48_000.0, &cfg), 1);
        assert_eq!(grid.cells.iter().sum::<f64>(), 0.0);
    }

    /// The map is the same bits however the chunks are scheduled.
    #[test]
    fn the_map_does_not_depend_on_the_thread_count() {
        let cfg = PartialAnalysisConfig::default();
        let o: Vec<_> = (0..500)
            .map(|i| obs(i * 997 % 480_000, 100.0 + (i * 37 % 4000) as f64, 1.0 + i as f64, 2000.0, 20.0))
            .collect();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                let mut grid = layout(&o, 48_000.0, 0, &cfg);
                accumulate(&mut grid, &o, 48_000.0, &cfg);
                grid
            })
        };
        assert_eq!(run(1), run(7));
    }
}
