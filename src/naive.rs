//! Brute-force matching pursuit — the oracle.
//!
//! Obviously correct and far too slow to use: it scans **every integer onset** and every bin of
//! every block, computing each inner product by direct summation. Its purpose is to be the
//! reference that the fast path is diffed against.
//!
//! # Independence
//!
//! This module must not share machinery with the code it validates, or it inherits the same bugs
//! and proves nothing. It therefore computes its own correlations and its own Gram by direct O(L)
//! sums, and never calls [`crate::corr`] or the FFT layer.
//!
//! What it *does* share is deliberate:
//!
//! - [`crate::fof`], which is the *definition* of the atom. Both paths must render identical atoms
//!   or a comparison is meaningless.
//! - The search space — envelopes, `fft_len` (which fixes the frequency grid) and bin ranges — is
//!   taken from a [`Dictionary`] so both paths search exactly the same candidates. That makes bin
//!   *gating* a shared assumption; the mathematics on each candidate stays independent.
//!
//! # Cost
//!
//! `blocks * onsets * bins * L` per iteration. Practical only for short signals and a couple of
//! short-support (high-`alpha`) blocks — a few million operations per iteration. A full voice
//! dictionary over a second of audio is out of the question, by design.

use crate::book::{Book, Selection};
use crate::dict::Dictionary;
use crate::fof::AtomParams;
use crate::signal::{Signal, subtract_at};
use std::f64::consts::TAU;

/// Stopping rules for the oracle.
#[derive(Clone, Copy, Debug)]
pub struct NaiveConfig {
    pub max_atoms: usize,
    /// Stop once residual energy falls to this fraction of the original.
    pub target_residual_fraction: f64,
    /// Stop when the best atom would remove less than this fraction of the current residual.
    pub min_gain_fraction: f64,
}

impl Default for NaiveConfig {
    fn default() -> Self {
        Self {
            max_atoms: 64,
            target_residual_fraction: 1e-6,
            min_gain_fraction: 1e-9,
        }
    }
}

/// Basis vectors and Gram for one bin, all built by direct summation.
struct Basis {
    u: Vec<f32>,
    v: Vec<f32>,
    /// Entries of `G^-1`, or `None` if degenerate.
    inv: Option<(f64, f64, f64)>,
}

struct NaiveBlock {
    env: Vec<f32>,
    params: crate::fof::EnvelopeParams,
    fft_len: usize,
    k_lo: usize,
    bases: Vec<Basis>,
}

impl NaiveBlock {
    fn bin_hz(&self, k: usize, sample_rate: f32) -> f32 {
        k as f32 * sample_rate / self.fft_len as f32
    }
}

pub struct NaiveMp {
    blocks: Vec<NaiveBlock>,
    sample_rate: f32,
}

impl NaiveMp {
    /// Build from a dictionary's search space, computing every Gram directly.
    pub fn new(dict: &Dictionary) -> Self {
        let blocks = dict
            .blocks
            .iter()
            .map(|b| {
                let env = &b.env.samples;
                let bases = (b.k_lo..=b.k_hi)
                    .map(|k| {
                        let w = TAU * k as f64 / b.fft_len as f64;
                        let mut u = vec![0.0f32; env.len()];
                        let mut v = vec![0.0f32; env.len()];
                        let (mut uu, mut vv, mut uv) = (0.0f64, 0.0, 0.0);
                        for (t, &e) in env.iter().enumerate() {
                            let (s, c) = ((w * t as f64).sin(), (w * t as f64).cos());
                            let (us, vc) = (e as f64 * s, e as f64 * c);
                            u[t] = us as f32;
                            v[t] = vc as f32;
                            uu += us * us;
                            vv += vc * vc;
                            uv += us * vc;
                        }
                        let det = uu * vv - uv * uv;
                        let inv = (det > 0.0 && uu > 0.0)
                            .then(|| (vv / det, -uv / det, uu / det));
                        Basis { u, v, inv }
                    })
                    .collect();
                NaiveBlock {
                    env: env.clone(),
                    params: b.env.params,
                    fft_len: b.fft_len,
                    k_lo: b.k_lo,
                    bases,
                }
            })
            .collect();
        Self {
            blocks,
            sample_rate: dict.sample_rate,
        }
    }

    /// Run the pursuit, returning the book and the final residual.
    pub fn run(&self, signal: &Signal, cfg: &NaiveConfig) -> (Book, Signal) {
        let mut residual = signal.clone();
        let initial = residual.energy();
        let mut book = Book::new(initial, self.sample_rate);
        let mut energy = initial;

        for _ in 0..cfg.max_atoms {
            if energy <= initial * cfg.target_residual_fraction {
                break;
            }
            let Some(best) = self.best_atom(&residual.samples) else {
                break;
            };
            if best.energy <= energy * cfg.min_gain_fraction {
                break;
            }

            let atom = AtomParams {
                t0: best.onset as i64,
                f: self.blocks[best.block].bin_hz(best.bin, self.sample_rate),
                env: self.blocks[best.block].params,
                phi: best.phi,
                amp: best.amp,
            };
            let Ok(rendered) = atom.render(self.sample_rate) else {
                break;
            };

            let before = energy;
            energy = subtract_at(&mut residual.samples, &rendered, atom.t0, before);
            book.selections.push(Selection {
                atom,
                block: best.block,
                onset: best.onset,
                bin: best.bin,
                projected_energy: best.energy,
                energy_removed: before - energy,
                residual_energy: energy,
                hr_score: None,
                refined: false,
            });

            // The measured energy must fall. A rise means a parameter-mapping bug, not noise.
            if energy >= before {
                break;
            }
        }

        (book, residual)
    }

    /// Exhaustive scan: every block, every integer onset, every bin.
    fn best_atom(&self, residual: &[f32]) -> Option<Candidate> {
        let mut best: Option<Candidate> = None;
        for (bi, block) in self.blocks.iter().enumerate() {
            for onset in 0..residual.len() {
                for (i, basis) in block.bases.iter().enumerate() {
                    let Some((inv_uu, inv_uv, inv_vv)) = basis.inv else {
                        continue;
                    };
                    let (d_u, d_v) = dot2(residual, onset, &basis.u, &basis.v);
                    let z_x = inv_uu * d_u + inv_uv * d_v;
                    let z_y = inv_uv * d_u + inv_vv * d_v;
                    let energy = d_u * z_x + d_v * z_y;

                    if best.as_ref().is_none_or(|b| energy > b.energy) {
                        best = Some(Candidate {
                            block: bi,
                            onset,
                            bin: block.k_lo + i,
                            energy,
                            amp: z_x.hypot(z_y) as f32,
                            phi: z_y.atan2(z_x) as f32,
                        });
                    }
                }
            }
        }
        best
    }

    /// Projection at one explicit cell — for cross-checking the fast path.
    pub fn project_at(
        &self,
        residual: &[f32],
        block: usize,
        onset: usize,
        bin: usize,
    ) -> Option<(f64, f32, f32)> {
        let b = self.blocks.get(block)?;
        let basis = b.bases.get(bin.checked_sub(b.k_lo)?)?;
        let (inv_uu, inv_uv, inv_vv) = basis.inv?;
        let (d_u, d_v) = dot2(residual, onset, &basis.u, &basis.v);
        let z_x = inv_uu * d_u + inv_uv * d_v;
        let z_y = inv_uv * d_u + inv_vv * d_v;
        Some((
            d_u * z_x + d_v * z_y,
            z_x.hypot(z_y) as f32,
            z_y.atan2(z_x) as f32,
        ))
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn support_len(&self, block: usize) -> usize {
        self.blocks[block].env.len()
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    block: usize,
    onset: usize,
    bin: usize,
    energy: f64,
    amp: f32,
    phi: f32,
}

/// Both correlations in one pass, reading past the end of `signal` as zeros.
fn dot2(signal: &[f32], onset: usize, u: &[f32], v: &[f32]) -> (f64, f64) {
    let n = signal.len().saturating_sub(onset).min(u.len());
    let (mut du, mut dv) = (0.0f64, 0.0f64);
    for i in 0..n {
        let r = signal[onset + i] as f64;
        du += r * u[i] as f64;
        dv += r * v[i] as f64;
    }
    (du, dv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corr::{Correlator, project};
    use crate::dict::BlockConfig;
    use crate::fft::Planner;

    const SR: f32 = 48_000.0;

    /// A deliberately tiny dictionary: high alpha gives short support, which keeps the exhaustive
    /// scan affordable. Two blocks so block selection is actually exercised.
    fn tiny_dict() -> Dictionary {
        let mut planner = Planner::new();
        let cfg = BlockConfig {
            f_min: 500.0,
            f_max: 6000.0,
            ..BlockConfig::default()
        };
        Dictionary::from_grid(&[(2147.0, 0.0003), (1342.0, 0.0003)], SR, &mut planner, &cfg)
            .unwrap()
    }

    /// Build an atom sitting exactly on the dictionary grid.
    ///
    /// Asserts the bin is actually searchable: planting an atom outside `[k_lo, k_hi]` produces a
    /// signal the dictionary cannot represent, which looks like a recovery failure rather than a
    /// bad test.
    fn on_grid_atom(d: &Dictionary, block: usize, bin: usize, t0: i64, amp: f32, phi: f32) -> AtomParams {
        let b = &d.blocks[block];
        assert!(
            bin >= b.k_lo && bin <= b.k_hi,
            "bin {bin} ({} Hz) is outside block {block}'s range {}..={}",
            b.bin_hz(bin),
            b.k_lo,
            b.k_hi
        );
        assert!(b.gram_inv(bin).is_some(), "bin {bin} is disabled");
        AtomParams {
            t0,
            f: b.bin_hz(bin),
            env: b.env.params,
            phi,
            amp,
        }
    }

    #[test]
    fn recovers_a_single_planted_atom() {
        let d = tiny_dict();
        let mp = NaiveMp::new(&d);
        let want = on_grid_atom(&d, 0, 20, 40, 0.8, 0.7);
        let sig = Signal::from_atoms(&[want], 512, SR).unwrap();

        let (book, _) = mp.run(&sig, &NaiveConfig { max_atoms: 1, ..Default::default() });
        assert_eq!(book.len(), 1);
        let got = &book.selections[0];
        assert_eq!(got.block, 0);
        assert_eq!(got.onset, 40);
        assert_eq!(got.bin, 20);
        assert!((got.atom.amp - want.amp).abs() < 0.02);
        assert!(book.snr_db() > 25.0, "SNR only {} dB", book.snr_db());
    }

    #[test]
    fn drives_a_planted_atom_to_near_zero_residual() {
        let d = tiny_dict();
        let mp = NaiveMp::new(&d);
        let want = on_grid_atom(&d, 0, 15, 30, 1.0, 0.0);
        let sig = Signal::from_atoms(&[want], 400, SR).unwrap();

        let (book, residual) = mp.run(&sig, &NaiveConfig { max_atoms: 8, ..Default::default() });
        assert!(book.snr_db() > 40.0, "SNR only {} dB", book.snr_db());
        assert!(residual.energy() < sig.energy() * 1e-4);
    }

    #[test]
    fn recovers_two_separated_atoms() {
        let d = tiny_dict();
        let mp = NaiveMp::new(&d);
        let a = on_grid_atom(&d, 0, 12, 20, 1.0, 0.2);
        let b = on_grid_atom(&d, 0, 22, 260, 0.7, 1.4);
        let sig = Signal::from_atoms(&[a, b], 512, SR).unwrap();

        let (book, _) = mp.run(&sig, &NaiveConfig { max_atoms: 6, ..Default::default() });
        assert!(book.snr_db() > 30.0, "SNR only {} dB", book.snr_db());

        // Both onsets must appear among the first two picks.
        let onsets: Vec<usize> = book.selections.iter().take(2).map(|s| s.onset).collect();
        assert!(onsets.contains(&20), "missed onset 20, got {onsets:?}");
        assert!(onsets.contains(&260), "missed onset 260, got {onsets:?}");
    }

    #[test]
    fn residual_energy_decreases_monotonically() {
        let d = tiny_dict();
        let mp = NaiveMp::new(&d);
        let mut seed = 0x5eed_1234_5678_9abcu64;
        let noise: Vec<f32> = (0..384)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / 8_388_608.0 - 1.0
            })
            .collect();
        let sig = Signal::new(noise, SR);

        let (book, residual) = mp.run(&sig, &NaiveConfig { max_atoms: 12, ..Default::default() });
        assert!(book.len() > 1);
        let mut prev = book.initial_energy;
        for s in &book.selections {
            assert!(s.residual_energy < prev, "energy rose: {prev} -> {}", s.residual_energy);
            prev = s.residual_energy;
        }
        // The incremental figure must match a recomputation from scratch.
        assert!((residual.energy() - book.residual_energy()).abs() < 1e-6 * book.initial_energy);
    }

    #[test]
    fn measured_energy_tracks_the_projection() {
        // Divergence would mean the fitted (amp, phi) do not describe the atom actually rendered.
        let d = tiny_dict();
        let mp = NaiveMp::new(&d);
        let want = on_grid_atom(&d, 0, 18, 50, 0.9, 2.1);
        let sig = Signal::from_atoms(&[want], 512, SR).unwrap();

        let (book, _) = mp.run(&sig, &NaiveConfig { max_atoms: 1, ..Default::default() });
        let s = &book.selections[0];
        let rel = (s.energy_removed - s.projected_energy).abs() / s.projected_energy;
        assert!(
            rel < 5e-3,
            "removed {} vs projected {} (rel {rel:.2e})",
            s.energy_removed,
            s.projected_energy
        );
    }

    /// The payoff: the fast Stage 2 path must agree with this independent implementation at the
    /// same cell. Two separate codebases computing the same projection.
    #[test]
    fn fast_path_projection_matches_the_oracle() {
        let d = tiny_dict();
        let mp = NaiveMp::new(&d);
        let mut planner = Planner::new();

        let mut seed = 0xface_0ff1_ce00_1234u64;
        let sig: Vec<f32> = (0..1024)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / 8_388_608.0 - 1.0
            })
            .collect();

        let mut checked = 0;
        for (bi, block) in d.blocks.iter().enumerate() {
            let mut c = Correlator::new(block, &mut planner);
            for onset in [0usize, 37, 200] {
                c.correlate(block, &sig, onset);
                for k in [block.k_lo, block.k_lo + 5, (block.k_lo + block.k_hi) / 2, block.k_hi] {
                    let (d_u, d_v) = c.at(k);
                    let fast = project(block, k, d_u, d_v);
                    let Some((energy, amp, phi)) = mp.project_at(&sig, bi, onset, k) else {
                        continue;
                    };

                    let scale = energy.max(1e-12);
                    assert!(
                        (fast.energy - energy).abs() / scale < 1e-3,
                        "block {bi} onset {onset} bin {k}: energy {} vs {energy}",
                        fast.energy
                    );
                    assert!(
                        (fast.amp - amp).abs() / amp.abs().max(1e-6) < 1e-2,
                        "block {bi} onset {onset} bin {k}: amp {} vs {amp}",
                        fast.amp
                    );
                    let dphi = (fast.phi - phi).abs().rem_euclid(std::f32::consts::TAU);
                    let dphi = dphi.min(std::f32::consts::TAU - dphi);
                    assert!(dphi < 1e-2, "block {bi} onset {onset} bin {k}: phi {} vs {phi}", fast.phi);
                    checked += 1;
                }
            }
        }
        assert!(checked >= 16, "only {checked} cells compared");
    }

    #[test]
    fn stops_when_no_atom_gains_anything() {
        let d = tiny_dict();
        let mp = NaiveMp::new(&d);
        let sig = Signal::silence(256, SR);
        let (book, _) = mp.run(&sig, &NaiveConfig::default());
        assert!(book.is_empty(), "selected {} atoms from silence", book.len());
    }

    #[test]
    fn book_replays_through_rfofs_parameters() {
        let d = tiny_dict();
        let mp = NaiveMp::new(&d);
        let want = on_grid_atom(&d, 0, 22, 60, 0.75, 1.0);
        let sig = Signal::from_atoms(&[want], 512, SR).unwrap();
        let (book, _) = mp.run(&sig, &NaiveConfig { max_atoms: 4, ..Default::default() });

        // Rendering the book's own parameters must reproduce the analysed signal.
        let atoms: Vec<AtomParams> = book.selections.iter().map(|s| s.atom).collect();
        let resynth = Signal::from_atoms(&atoms, 512, SR).unwrap();
        let err: f64 = sig
            .samples
            .iter()
            .zip(&resynth.samples)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum();
        let snr = crate::signal::snr_db(sig.energy(), err);
        assert!(snr > 30.0, "resynthesis SNR only {snr} dB");

        let params = book.to_fof_params(0);
        assert_eq!(params.len(), book.len());
        assert_eq!(params[0].start_sample, book.selections[0].onset as u64);
        assert_eq!(params[0].gliss, 0.0);
    }
}
