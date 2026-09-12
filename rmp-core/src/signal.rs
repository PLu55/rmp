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

    /// Root-mean-square amplitude.
    pub fn rms(&self) -> f64 {
        rms_of(&self.samples)
    }

    /// Largest absolute sample.
    pub fn peak(&self) -> f32 {
        peak_of(&self.samples)
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

/// Root-mean-square amplitude. Zero for an empty buffer.
pub fn rms_of(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    (energy_of(samples) / samples.len() as f64).sqrt()
}

/// Largest absolute sample. Zero for an empty buffer.
pub fn peak_of(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()))
}

/// An amplitude as dB relative to full scale, so 1.0 is 0 dBFS.
///
/// Returns `f32::NEG_INFINITY` for exact silence rather than `NaN`.
pub fn db_fs(amplitude: f64) -> f32 {
    if amplitude <= 0.0 {
        return f32::NEG_INFINITY;
    }
    (20.0 * amplitude.log10()) as f32
}

pub fn snr_db(signal_energy: f64, residual_energy: f64) -> f32 {
    if residual_energy <= 0.0 {
        return f32::INFINITY;
    }
    (10.0 * (signal_energy / residual_energy).log10()) as f32
}

/// Clip `src` placed at a signed `offset` into a buffer of `dst_len`.
///
/// Returns `(src_start, dst_start, n)`, or `None` when the two do not overlap at all.
///
/// Every path that places an atom against the signal goes through here — writing it
/// ([`add_at`], [`subtract_at`]), scoring it ([`crate::fit::accumulate`]), and invalidating the
/// frames it touched ([`crate::mp::Mp::stale_range`]). They have to agree on exactly which samples
/// an atom occupies, and the cheapest way to guarantee that is to give them one implementation.
pub fn overlap(dst_len: usize, src_len: usize, offset: i64) -> Option<(usize, usize, usize)> {
    let (src_start, dst_start) = if offset < 0 {
        ((-offset) as usize, 0usize)
    } else {
        (0usize, offset as usize)
    };
    if src_start >= src_len || dst_start >= dst_len {
        return None;
    }
    Some((
        src_start,
        dst_start,
        (src_len - src_start).min(dst_len - dst_start),
    ))
}

/// Add `src` into `dst` at a signed offset, clipping to `dst`'s bounds.
pub fn add_at(dst: &mut [f32], src: &[f32], offset: i64) {
    let Some((src_start, dst_start, n)) = overlap(dst.len(), src.len(), offset) else {
        return;
    };
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
    let len = residual.len();
    subtract_at_core(residual, atom, offset, old_energy, old_energy, len).0
}

/// [`subtract_at`], tracking the energy of a leading *core* range alongside the whole buffer's.
///
/// The windowed pursuit analyses `[0, core_len)` and carries `[core_len, len)` only so that an
/// atom starting near the core's end is scored against real residual instead of an artificial zero
/// edge. Its stopping rule therefore has to read the core's energy — the guard belongs to the next
/// window and will be decomposed there, so counting it would make every window look unfinished.
///
/// Both energies come out of the one pass, over exactly the samples written, so there is one
/// definition of what an atom removed and not two that can drift. [`subtract_at`] is the
/// `core_len == residual.len()` case of this, which is why it delegates rather than repeating it.
///
/// Returns `(whole, core)`. The whole-buffer figure is what the book records as `energy_removed`:
/// an atom's removal is a fact about the atom, not about which window happened to select it.
pub fn subtract_at_core(
    residual: &mut [f32],
    atom: &[f32],
    offset: i64,
    old_energy: f64,
    old_core_energy: f64,
    core_len: usize,
) -> (f64, f64) {
    let Some((src_start, dst_start, n)) = overlap(residual.len(), atom.len(), offset) else {
        return (old_energy, old_core_energy);
    };

    let (mut dot, mut norm2) = (0.0f64, 0.0f64);
    let (mut core_dot, mut core_norm2) = (0.0f64, 0.0f64);
    for i in 0..n {
        let r = residual[dst_start + i] as f64;
        let a = atom[src_start + i] as f64;
        dot += r * a;
        norm2 += a * a;
        if dst_start + i < core_len {
            core_dot += r * a;
            core_norm2 += a * a;
        }
        residual[dst_start + i] = (r - a) as f32;
    }
    (
        (old_energy - 2.0 * dot + norm2).max(0.0),
        (old_core_energy - 2.0 * core_dot + core_norm2).max(0.0),
    )
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
            env: EnvelopeParams::new(2147.0, 0.0003).into(),
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

    /// `mp` invalidates the frames an atom touched using the range `overlap` reports, so that range
    /// has to be exactly the set of samples `add_at` and `subtract_at` actually write. Anything
    /// less and a frame goes silently stale.
    #[test]
    fn overlap_is_exactly_what_add_and_subtract_write() {
        let src = vec![1.0f32; 7];
        for offset in -9i64..14 {
            let mut dst = vec![0.0f32; 10];
            add_at(&mut dst, &src, offset);
            let touched: Vec<usize> =
                dst.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(i, _)| i).collect();

            match overlap(dst.len(), src.len(), offset) {
                None => assert!(touched.is_empty(), "offset {offset} wrote {touched:?}"),
                Some((src_start, dst_start, n)) => {
                    assert_eq!(
                        touched,
                        (dst_start..dst_start + n).collect::<Vec<_>>(),
                        "offset {offset}"
                    );
                    assert!(src_start + n <= src.len(), "offset {offset} reads past src");
                }
            }
        }
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

    #[test]
    fn rms_peak_and_db_fs() {
        let s = Signal::new(vec![0.5, -0.5, 0.5, -0.5], SR);
        assert!((s.rms() - 0.5).abs() < 1e-12);
        assert_eq!(s.peak(), 0.5);
        // Half amplitude is -6.02 dBFS.
        assert!((db_fs(s.rms()) + 6.0206).abs() < 1e-3);
        assert_eq!(db_fs(1.0), 0.0);
        assert_eq!(db_fs(0.0), f32::NEG_INFINITY);

        // Empty is silent, not NaN.
        let empty = Signal::new(Vec::new(), SR);
        assert_eq!(empty.rms(), 0.0);
        assert_eq!(empty.peak(), 0.0);
    }
}
