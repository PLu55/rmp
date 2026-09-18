//! Persistent partials: from atoms to slowly varying frequency and level trajectories (§8–§17).
//!
//! ```text
//! observations → accumulated map → per-frame peaks → ridges → accepted, simplified partials
//! ```
//!
//! A partial is not an atom. It is a persistent concentration of energy near a continuously evolving
//! frequency, so nothing is thresholded until evidence has been aggregated: every atom is deposited
//! into the map first, and weak atoms that recur along one frequency add up to a ridge that no one of
//! them would make. Rejection happens only at the level of ridges, by duration and persistence.
//!
//! **Supporting atoms** trace a partial back to the book (§14). After the partials are fixed, each
//! atom is given to the partial alive at its centre time whose trajectory passes nearest its
//! frequency, if any passes within `max_jump_cents_per_frame`. At most one partial per atom, and
//! atoms near none stay unassigned — detail the partial level deliberately throws away. A partial's
//! `energy` is the sum of its supporting atoms' energies, so it is measured in the book's own terms.
//!
//! **Ids are assigned after a deterministic sort** by start, then mean frequency, then energy
//! (descending), so they are stable for a given book and configuration (§42).

pub mod accumulation;
pub mod metrics;
pub mod peaks;
pub mod simplify;
pub mod tracking;

use crate::config::PartialAnalysisConfig;
use crate::error::{Result, StructureError};
use crate::ids::{AtomId, PartialId};
use crate::observation::{self, AtomObservation, Skipped};
use crate::output::partial_book::{PartialBook, PartialBookMetadata};
use crate::trajectory::{Trajectory, TrajectoryPoint};
use accumulation::TfGrid;
use metrics::Verdict;
use peaks::Peak;
use rayon::prelude::*;
use rmp_core::book::Book;
use serde::{Deserialize, Serialize};
use tracking::Ridge;

/// One persistent partial.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Partial {
    pub id: PartialId,
    /// Absolute samples of the first and last frame the ridge was found in.
    pub start_samples: u64,
    pub end_samples: u64,
    pub mean_frequency_hz: f64,
    pub geometric_mean_frequency_hz: f64,
    pub frequency_std_cents: f64,
    /// Mean over the ridge's frames of the square root of its frame energy.
    pub mean_amplitude: f64,
    /// Sum of the supporting atoms' `energy_removed`.
    pub energy: f64,
    /// `energy` over the energy of every observation in the book.
    pub normalized_energy: f64,
    pub persistence: f64,
    /// `normalized_energy × persistence`.
    pub significance: f64,
    /// Hz, simplified in cents.
    pub frequency: Trajectory,
    /// dB of the ridge's energy per frame, simplified in dB.
    pub amplitude: Trajectory,
    /// Ascending.
    pub supporting_atoms: Vec<AtomId>,
}

impl Partial {
    pub fn duration_samples(&self) -> u64 {
        self.end_samples - self.start_samples
    }
}

/// What happened, for a front end to report (§39).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PartialDiagnostics {
    pub input_atoms: usize,
    pub observations: usize,
    pub skipped: Skipped,
    /// Observations that fell entirely outside the grid's frequency range.
    pub outside_grid: usize,
    pub frames: usize,
    pub bins: usize,
    pub peaks: usize,
    pub candidate_ridges: usize,
    pub rejected_short: usize,
    pub rejected_sparse: usize,
    pub accepted_partials: usize,
    pub supporting_atoms: usize,
    pub unsupported_atoms: usize,
}

/// The stages' own results, for inspection (§40). Only produced on request.
#[derive(Clone, Debug, PartialEq)]
pub struct Intermediates {
    pub observations: Vec<AtomObservation>,
    pub grid: TfGrid,
    pub peaks: Vec<Vec<Peak>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PartialAnalysis {
    pub book: PartialBook,
    pub diagnostics: PartialDiagnostics,
    pub intermediates: Option<Intermediates>,
}

/// A ridge that passed, before it has an id.
struct Candidate {
    start: u64,
    end: u64,
    metrics: metrics::RidgeMetrics,
    frequency: Trajectory,
    amplitude: Trajectory,
    supporting: Vec<AtomId>,
    energy: f64,
}

/// Extract the persistent partials of `book`.
pub fn analyze_partials(book: &Book, cfg: &PartialAnalysisConfig) -> Result<PartialAnalysis> {
    cfg.validate()?;
    let sr = book.sample_rate as f64;
    if !(sr > 0.0 && sr.is_finite()) {
        return Err(StructureError::InvalidBook(format!("sample rate {sr}")));
    }
    let tracking_cfg = cfg.tracking();

    let (observations, skipped) = observation::observe(book);
    let mut grid = accumulation::layout(&observations, sr, book.start_sample, cfg);
    let outside_grid = accumulation::accumulate(&mut grid, &observations, sr, cfg);
    let peaks = peaks::find_peaks(&grid, tracking_cfg.min_relative_level_db);
    let ridges = tracking::track(&peaks, &tracking_cfg);

    let mut diag = PartialDiagnostics {
        input_atoms: book.selections.len(),
        observations: observations.len(),
        skipped,
        outside_grid,
        frames: grid.n_frames,
        bins: grid.axis.n_bins,
        peaks: peaks.iter().map(Vec::len).sum(),
        candidate_ridges: ridges.len(),
        ..Default::default()
    };

    let mut candidates = Vec::new();
    for ridge in &ridges {
        let m = metrics::measure(ridge, cfg.time_step_ms);
        match metrics::judge(&m, &tracking_cfg) {
            Verdict::Short => diag.rejected_short += 1,
            Verdict::Sparse => diag.rejected_sparse += 1,
            Verdict::Accept => candidates.push(candidate(ridge, m, &grid, cfg)),
        }
    }

    assign_support(&mut candidates, &observations, &grid, cfg.max_jump_cents_per_frame);
    let total_energy: f64 = observations.iter().map(|o| o.energy).sum();

    candidates.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.metrics.mean_frequency_hz.total_cmp(&b.metrics.mean_frequency_hz))
            .then(b.energy.total_cmp(&a.energy))
            .then(a.end.cmp(&b.end))
    });
    let partials: Vec<Partial> = candidates
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let normalized = if total_energy > 0.0 { c.energy / total_energy } else { 0.0 };
            Partial {
                id: PartialId(i as u32),
                start_samples: c.start,
                end_samples: c.end,
                mean_frequency_hz: c.metrics.mean_frequency_hz,
                geometric_mean_frequency_hz: c.metrics.geometric_mean_frequency_hz,
                frequency_std_cents: c.metrics.frequency_std_cents,
                mean_amplitude: c.metrics.mean_amplitude,
                energy: c.energy,
                normalized_energy: normalized,
                persistence: c.metrics.persistence,
                significance: metrics::significance(normalized, c.metrics.persistence),
                frequency: c.frequency,
                amplitude: c.amplitude,
                supporting_atoms: c.supporting,
            }
        })
        .collect();

    diag.accepted_partials = partials.len();
    diag.supporting_atoms = partials.iter().map(|p| p.supporting_atoms.len()).sum();
    diag.unsupported_atoms = observations.len() - diag.supporting_atoms;

    let book_out = PartialBook {
        metadata: PartialBookMetadata::new(sr, book.start_sample, cfg.clone()),
        partials,
    };
    let intermediates = cfg.keep_intermediates.then_some(Intermediates { observations, grid, peaks });
    Ok(PartialAnalysis { book: book_out, diagnostics: diag, intermediates })
}

/// Fill a ridge's missed frames, smooth and simplify it.
fn candidate(ridge: &Ridge, m: metrics::RidgeMetrics, grid: &TfGrid, cfg: &PartialAnalysisConfig) -> Candidate {
    let first = ridge.first_frame();
    let cents: Vec<(usize, f64)> = ridge.points.iter().map(|p| (p.frame, p.cents)).collect();
    // Level in dB of the ridge's frame energy. Energies are positive: a peak is at or above a floor
    // that is a positive fraction of the strongest cell.
    let level: Vec<(usize, f64)> =
        ridge.points.iter().map(|p| (p.frame, 10.0 * p.energy.log10())).collect();

    let cents = simplify::smooth(&simplify::fill(&cents), cfg.smoothing_frames);
    let level = simplify::smooth(&simplify::fill(&level), cfg.smoothing_frames);

    let points = |y: &[f64], tol: f64, map: &dyn Fn(f64) -> f64| Trajectory {
        points: simplify::rdp(y, tol)
            .into_iter()
            .map(|i| TrajectoryPoint { time_samples: grid.time_of(first + i), value: map(y[i]) })
            .collect(),
    };
    Candidate {
        start: grid.time_of(first),
        end: grid.time_of(ridge.last_frame()),
        metrics: m,
        frequency: points(&cents, cfg.frequency_simplify_cents, &peaks::hz),
        amplitude: points(&level, cfg.amplitude_simplify_db, &|v| v),
        supporting: Vec::new(),
        energy: 0.0,
    }
}

/// Frames per bucket of the partial time index.
const INDEX_FRAMES: u64 = 64;

/// Give each observation to at most one candidate — the nearest in frequency among those alive at
/// its centre time, within `tolerance_cents` — and total each candidate's energy.
fn assign_support(
    candidates: &mut [Candidate],
    observations: &[AtomObservation],
    grid: &TfGrid,
    tolerance_cents: f64,
) {
    if candidates.is_empty() {
        return;
    }
    // A partial is considered alive one frame either side of its first and last.
    let span = INDEX_FRAMES * grid.hop;
    let bucket = |t: u64| (t.saturating_sub(grid.origin) / span) as usize;
    let last_end = candidates.iter().map(|c| c.end).max().unwrap() + grid.hop;
    let mut index: Vec<Vec<u32>> = vec![Vec::new(); bucket(last_end) + 1];
    for (i, c) in candidates.iter().enumerate() {
        let (lo, hi) = (bucket(c.start.saturating_sub(grid.hop)), bucket(c.end + grid.hop));
        for list in &mut index[lo..=hi] {
            list.push(i as u32);
        }
    }

    let owner: Vec<Option<u32>> = observations
        .par_iter()
        .map(|o| {
            let t = o.time_center_samples;
            let list = index.get(bucket(t))?;
            let want = peaks::cents(o.frequency_hz);
            let mut best: Option<(f64, u32)> = None;
            for &i in list {
                let c = &candidates[i as usize];
                if t + grid.hop < c.start || t > c.end + grid.hop {
                    continue;
                }
                let at = t.clamp(c.start, c.end);
                let Some(f) = c.frequency.value_at(at) else { continue };
                let d = (peaks::cents(f) - want).abs();
                if d <= tolerance_cents && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, i));
                }
            }
            best.map(|(_, i)| i)
        })
        .collect();

    for (o, owner) in observations.iter().zip(owner) {
        if let Some(i) = owner {
            let c = &mut candidates[i as usize];
            c.supporting.push(o.atom_id);
            c.energy += o.energy;
        }
    }
}

#[cfg(test)]
mod tests;
