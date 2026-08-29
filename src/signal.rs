//! Signal buffers and energy bookkeeping.
//!
//! Samples are f32 (matching rfofs), but **every accumulator is f64**. Matching pursuit subtracts
//! thousands of times from the same buffer, and a running residual energy in f32 drifts badly:
//! naively summing 8192 f32 already loses ~1.5e-6 relative.

use crate::fof::{AtomParams, FofError};

#[derive(Clone, Debug, PartialEq)]
pub struct Signal {
    pub samples: Vec<f32>,
    pub sample_rate: f32,
}

impl Signal {
    pub fn new(samples: Vec<f32>, sample_rate: f32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    pub fn silence(len: usize, sample_rate: f32) -> Self {
        Self::new(vec![0.0; len], sample_rate)
    }

    /// Render a sum of atoms — the ground-truth generator for recovery tests.
    pub fn from_atoms(
        atoms: &[AtomParams],
        len: usize,
        sample_rate: f32,
    ) -> Result<Self, FofError> {
        let mut out = Self::silence(len, sample_rate);
        for a in atoms {
            let rendered = a.render(sample_rate)?;
            add_at(&mut out.samples, &rendered, a.t0);
        }
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Sum of squares, accumulated in f64.
    pub fn energy(&self) -> f64 {
        energy_of(&self.samples)
    }

    /// `10*log10(self.energy() / residual_energy)`.
    ///
    /// Returns `f32::INFINITY` for an exactly zero residual.
    pub fn snr_db(&self, residual_energy: f64) -> f32 {
        snr_db(self.energy(), residual_energy)
    }
}

pub fn energy_of(samples: &[f32]) -> f64 {
    samples.iter().map(|&s| (s as f64) * (s as f64)).sum()
}

pub fn snr_db(signal_energy: f64, residual_energy: f64) -> f32 {
    if residual_energy <= 0.0 {
        return f32::INFINITY;
    }
    (10.0 * (signal_energy / residual_energy).log10()) as f32
}

/// Add `src` into `dst` at a signed offset, clipping to `dst`'s bounds.
pub fn add_at(dst: &mut [f32], src: &[f32], offset: i64) {
    let (src_start, dst_start) = if offset < 0 {
        ((-offset) as usize, 0usize)
    } else {
        (0usize, offset as usize)
    };
    if src_start >= src.len() || dst_start >= dst.len() {
        return;
    }
    let n = (src.len() - src_start).min(dst.len() - dst_start);
    for i in 0..n {
        dst[dst_start + i] += src[src_start + i];
    }
}

/// Subtract `atom` from `residual` at `offset`, returning the residual's new energy.
///
/// The energy is **measured**, not assumed equal to the projected energy:
/// `||R_new||^2 = ||R_old||^2 - 2<R,a> + ||a||^2`, with both terms taken from the atom actually
/// rendered. rfofs's LUT/polynomial carrier means the rendered atom is not exactly the ideal vector
/// that was projected onto, so assuming the projected energy would let error accumulate silently.
pub fn subtract_at(residual: &mut [f32], atom: &[f32], offset: i64, old_energy: f64) -> f64 {
    let (src_start, dst_start) = if offset < 0 {
        ((-offset) as usize, 0usize)
    } else {
        (0usize, offset as usize)
    };
    if src_start >= atom.len() || dst_start >= residual.len() {
        return old_energy;
    }
    let n = (atom.len() - src_start).min(residual.len() - dst_start);

    let (mut dot, mut norm2) = (0.0f64, 0.0f64);
    for i in 0..n {
        let r = residual[dst_start + i] as f64;
        let a = atom[src_start + i] as f64;
        dot += r * a;
        norm2 += a * a;
        residual[dst_start + i] = (r - a) as f32;
    }
    (old_energy - 2.0 * dot + norm2).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fof::EnvelopeParams;

    const SR: f32 = 48_000.0;

    fn atom(t0: i64, f: f32, amp: f32) -> AtomParams {
        AtomParams {
            t0,
            f,
            env: EnvelopeParams::new(2147.0, 0.0003),
            phi: 0.0,
            amp,
        }
    }

    #[test]
    fn energy_and_snr() {
        let s = Signal::new(vec![3.0, 4.0], SR);
        assert_eq!(s.energy(), 25.0);
        assert!((s.snr_db(25.0) - 0.0).abs() < 1e-6);
        assert!((s.snr_db(0.25) - 20.0).abs() < 1e-4);
        assert_eq!(s.snr_db(0.0), f32::INFINITY);
    }

    #[test]
    fn add_at_clips_at_both_ends() {
        let mut dst = vec![0.0; 4];
        add_at(&mut dst, &[1.0, 2.0, 3.0], -1);
        assert_eq!(dst, vec![2.0, 3.0, 0.0, 0.0]);

        let mut dst = vec![0.0; 4];
        add_at(&mut dst, &[1.0, 2.0, 3.0], 3);
        assert_eq!(dst, vec![0.0, 0.0, 0.0, 1.0]);

        let mut dst = vec![0.0; 4];
        add_at(&mut dst, &[1.0], 99);
        assert_eq!(dst, vec![0.0; 4]);
    }

    #[test]
    fn subtract_measures_energy_exactly() {
        let mut r = vec![1.0f32, 2.0, 3.0, 4.0];
        let before = energy_of(&r);
        let a = [0.5f32, 1.0];
        let after = subtract_at(&mut r, &a, 1, before);
        assert_eq!(r, vec![1.0, 1.5, 2.0, 4.0]);
        // Recomputed from scratch must agree with the incremental figure.
        assert!((after - energy_of(&r)).abs() < 1e-9);
    }

    #[test]
    fn subtracting_a_signal_from_itself_empties_it() {
        let a = atom(0, 2000.0, 0.8);
        let rendered = a.render(SR).unwrap();
        let mut r = rendered.clone();
        let before = energy_of(&r);
        let after = subtract_at(&mut r, &rendered, 0, before);
        assert!(after < before * 1e-12, "residual {after} vs {before}");
        assert!(r.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn from_atoms_superposes() {
        let atoms = [atom(0, 1000.0, 0.5), atom(300, 2000.0, 0.7)];
        let s = Signal::from_atoms(&atoms, 2048, SR).unwrap();
        assert_eq!(s.len(), 2048);
        assert!(s.energy() > 0.0);

        // Sum of parts equals the whole, since rendering is linear.
        let e0 = Signal::from_atoms(&atoms[..1], 2048, SR).unwrap();
        let e1 = Signal::from_atoms(&atoms[1..], 2048, SR).unwrap();
        for i in 0..2048 {
            assert!((s.samples[i] - (e0.samples[i] + e1.samples[i])).abs() < 1e-6);
        }
    }
}
