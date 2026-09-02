//! The decomposition result.
//!
//! A [`Book`] is the output of a pursuit: the atoms selected, in order, with enough provenance to
//! diagnose a bad decomposition and enough parameters to replay it through rfofs.

use crate::fof::AtomParams;
use crate::signal::snr_db;
use rfofs::fof::FofParams;

/// One selected atom, with where it came from and what it actually removed.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Selection {
    pub atom: AtomParams,
    /// Index into the dictionary's block list.
    pub block: usize,
    /// Onset in samples (the grid position before any refinement).
    pub onset: usize,
    /// Frequency bin.
    pub bin: usize,
    /// Energy the projection predicted.
    pub projected_energy: f64,
    /// Energy actually removed, measured from the rendered atom. Divergence from
    /// `projected_energy` indicates a parameter-mapping error.
    pub energy_removed: f64,
    /// Residual energy after this atom was subtracted.
    pub residual_energy: f64,
    /// Energy this atom removed after HRMP clamped its amplitude, when HRMP ran.
    ///
    /// `None` means ordinary MP: the atom was subtracted at its full projected amplitude.
    #[serde(default)]
    pub hr_score: Option<f64>,
    /// Whether refinement moved the parameters off the seed block's grid point.
    #[serde(default)]
    pub refined: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Book {
    pub selections: Vec<Selection>,
    pub initial_energy: f64,
    pub sample_rate: f32,
}

impl Book {
    pub fn new(initial_energy: f64, sample_rate: f32) -> Self {
        Self {
            selections: Vec::new(),
            initial_energy,
            sample_rate,
        }
    }

    pub fn len(&self) -> usize {
        self.selections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.selections.is_empty()
    }

    /// Residual energy after the last atom, or the initial energy if none were selected.
    pub fn residual_energy(&self) -> f64 {
        self.selections
            .last()
            .map_or(self.initial_energy, |s| s.residual_energy)
    }

    pub fn snr_db(&self) -> f32 {
        snr_db(self.initial_energy, self.residual_energy())
    }

    /// SNR in dB after each atom — the convergence curve.
    pub fn snr_trace(&self) -> Vec<f32> {
        self.selections
            .iter()
            .map(|s| snr_db(self.initial_energy, s.residual_energy))
            .collect()
    }

    /// Atoms needed to first reach `target_db`, if reached.
    pub fn atoms_to_reach(&self, target_db: f32) -> Option<usize> {
        self.snr_trace()
            .iter()
            .position(|&db| db >= target_db)
            .map(|i| i + 1)
    }

    /// Render the book back to a signal — the analysis inverted.
    ///
    /// Uses the same rfofs path the pursuit subtracted with, so a book that reached N dB against
    /// its input reproduces that input to N dB here.
    pub fn resynthesize(&self, len: usize) -> Result<crate::signal::Signal, crate::fof::FofError> {
        let atoms: Vec<AtomParams> = self.selections.iter().map(|s| s.atom).collect();
        crate::signal::Signal::from_atoms(&atoms, len, self.sample_rate)
    }

    /// Replayable parameters, for rendering through rfofs.
    pub fn to_fof_params(&self, origin: u64) -> Vec<FofParams> {
        self.selections
            .iter()
            .map(|s| s.atom.to_fof_params(origin))
            .collect()
    }

    /// How often each block was selected — the best diagnostic of a mis-sized grid. Piling up at an
    /// `alpha` edge means the ladder does not reach far enough.
    pub fn block_histogram(&self, n_blocks: usize) -> Vec<usize> {
        let mut counts = vec![0; n_blocks];
        for s in &self.selections {
            if s.block < n_blocks {
                counts[s.block] += 1;
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fof::EnvelopeParams;

    fn sel(block: usize, residual: f64) -> Selection {
        Selection {
            atom: AtomParams {
                t0: 0,
                f: 1000.0,
                env: EnvelopeParams::new(251.0, 0.001),
                phi: 0.0,
                amp: 1.0,
            },
            block,
            onset: 0,
            bin: 10,
            projected_energy: 1.0,
            energy_removed: 1.0,
            residual_energy: residual,
            hr_score: None,
            refined: false,
        }
    }

    #[test]
    fn snr_trace_and_targets() {
        let mut b = Book::new(100.0, 48_000.0);
        b.selections.push(sel(0, 10.0)); // 10 dB
        b.selections.push(sel(1, 1.0)); // 20 dB
        b.selections.push(sel(0, 0.1)); // 30 dB

        let trace = b.snr_trace();
        assert!((trace[0] - 10.0).abs() < 1e-3);
        assert!((trace[2] - 30.0).abs() < 1e-3);
        assert!((b.snr_db() - 30.0).abs() < 1e-3);

        assert_eq!(b.atoms_to_reach(20.0), Some(2));
        assert_eq!(b.atoms_to_reach(99.0), None);
    }

    #[test]
    fn empty_book_reports_initial_energy() {
        let b = Book::new(50.0, 48_000.0);
        assert_eq!(b.residual_energy(), 50.0);
        assert!((b.snr_db() - 0.0).abs() < 1e-6);
        assert!(b.is_empty());
    }

    #[test]
    fn block_histogram_counts_selections() {
        let mut b = Book::new(1.0, 48_000.0);
        b.selections.push(sel(0, 0.5));
        b.selections.push(sel(2, 0.25));
        b.selections.push(sel(0, 0.1));
        assert_eq!(b.block_histogram(3), vec![2, 0, 1]);
    }
}
