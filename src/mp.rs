//! The pursuit loop, with MPTK's local update.
//!
//! # What is stored
//!
//! Only two values per frame: its best projected energy and the bin that achieved it. Keeping
//! `(d_u, d_v)` for every (block, frame, bin) would cost on the order of hundreds of megabytes per
//! second of audio at these hop sizes. The winning frame's FFT is recomputed on demand when it is
//! selected — one extra transform per iteration, which is free next to the update.
//!
//! # The local update
//!
//! Subtracting an atom over `[tau, tau + L_a)` can only change frames whose read window overlaps
//! it. Frame `n` of block `b` reads `[n*h_b, n*h_b + support_b)`, so the stale range is
//!
//! ```text
//! n_lo = ceil((tau - support_b + 1) / h_b)     clamped at 0
//! n_hi = (tau + L_a - 1) / h_b                 clamped at frame_count - 1
//! ```
//!
//! Note this uses `support_len`, not `fft_len`: the envelope is zero past its support, so residual
//! changes beyond it cannot affect that frame's correlations.
//!
//! # Equivalence
//!
//! [`MpConfig::full_update`] recomputes every frame of every block each iteration. The incremental
//! path must produce a bit-identical book — that is the gate for this stage, and the stale-set
//! arithmetic is where bugs would hide.

use crate::book::{Book, Selection};
use crate::corr::Correlator;
use crate::dict::Dictionary;
use crate::fft::RealFftPlanner;
use crate::fof::AtomParams;
use crate::select::SegTree;
use crate::signal::{Signal, snr_db, subtract_at};

#[derive(Clone, Copy, Debug)]
pub struct MpConfig {
    pub max_atoms: usize,
    /// Stop once this reconstruction SNR is reached.
    pub target_snr_db: f32,
    /// Stop when the best atom would remove less than this fraction of the current residual.
    pub min_gain_fraction: f64,
    /// Recompute every frame each iteration instead of only the stale ones. Debug only — this is
    /// the reference the incremental path is validated against.
    pub full_update: bool,
}

impl Default for MpConfig {
    fn default() -> Self {
        Self {
            max_atoms: 1000,
            target_snr_db: 30.0,
            min_gain_fraction: 1e-9,
            full_update: false,
        }
    }
}

/// Per-block frame table and its max tree.
struct BlockState {
    /// Best projected energy per frame.
    energy: Vec<f64>,
    /// Bin achieving that energy.
    bin: Vec<u32>,
    tree: SegTree,
}

pub struct Mp<'a> {
    dict: &'a Dictionary,
    corrs: Vec<Correlator>,
    states: Vec<BlockState>,
    residual: Vec<f32>,
    energy: f64,
    initial_energy: f64,
    sample_rate: f32,
}

impl<'a> Mp<'a> {
    /// Correlate every frame of every block once, up front.
    pub fn new(dict: &'a Dictionary, signal: &Signal, planner: &mut dyn RealFftPlanner) -> Self {
        let mut corrs: Vec<Correlator> = dict
            .blocks
            .iter()
            .map(|b| Correlator::new(b, planner))
            .collect();

        let mut states = Vec::with_capacity(dict.blocks.len());
        for (bi, block) in dict.blocks.iter().enumerate() {
            let frames = block.frame_count(signal.len());
            let mut energy = vec![f64::NEG_INFINITY; frames];
            let mut bin = vec![0u32; frames];
            for n in 0..frames {
                let (k, p) = scan_frame(&mut corrs[bi], block, &signal.samples, block.frame_onset(n));
                energy[n] = p.energy;
                bin[n] = k as u32;
            }
            let tree = SegTree::new(&energy);
            states.push(BlockState { energy, bin, tree });
        }

        let initial_energy = signal.energy();
        Self {
            dict,
            corrs,
            states,
            residual: signal.samples.clone(),
            energy: initial_energy,
            initial_energy,
            sample_rate: signal.sample_rate,
        }
    }

    pub fn residual(&self) -> &[f32] {
        &self.residual
    }

    pub fn residual_energy(&self) -> f64 {
        self.energy
    }

    pub fn run(&mut self, cfg: &MpConfig) -> Book {
        let mut book = Book::new(self.initial_energy, self.sample_rate);

        for _ in 0..cfg.max_atoms {
            if snr_db(self.initial_energy, self.energy) >= cfg.target_snr_db {
                break;
            }
            let Some((bi, frame)) = self.global_argmax() else {
                break;
            };
            if self.states[bi].energy[frame] <= self.energy * cfg.min_gain_fraction {
                break;
            }

            let block = &self.dict.blocks[bi];
            let onset = block.frame_onset(frame);

            // Recompute the winner to recover amp and phi, which are not stored.
            let (k, p) = scan_frame(&mut self.corrs[bi], block, &self.residual, onset);
            if p.energy <= 0.0 {
                break;
            }

            let atom = AtomParams {
                t0: onset as i64,
                f: block.bin_hz(k),
                env: block.env.params,
                phi: p.phi,
                amp: p.amp,
            };
            let Ok(rendered) = atom.render(self.sample_rate) else {
                break;
            };

            let before = self.energy;
            self.energy = subtract_at(&mut self.residual, &rendered, atom.t0, before);
            book.selections.push(Selection {
                atom,
                block: bi,
                onset,
                bin: k,
                projected_energy: p.energy,
                energy_removed: before - self.energy,
                residual_energy: self.energy,
            });

            if self.energy >= before {
                break; // a rise means a parameter-mapping bug, not noise
            }

            if cfg.full_update {
                self.refresh_all();
            } else {
                self.refresh_stale(onset, rendered.len());
            }
        }

        book
    }

    /// Block and frame of the globally best atom. Ties go to the lowest block, then lowest frame.
    fn global_argmax(&self) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, f64)> = None;
        for (bi, st) in self.states.iter().enumerate() {
            if st.tree.is_empty() {
                continue;
            }
            let m = st.tree.max();
            if m.is_finite() && best.as_ref().is_none_or(|&(_, _, b)| m > b) {
                best = Some((bi, st.tree.argmax(), m));
            }
        }
        best.map(|(b, f, _)| (b, f))
    }

    /// Recompute only frames whose read window overlaps `[tau, tau + atom_len)`.
    ///
    /// The range itself comes from [`Self::stale_range`], so the arithmetic has exactly one
    /// implementation and the test that pins it against the overlap definition guards this path
    /// too.
    fn refresh_stale(&mut self, tau: usize, atom_len: usize) {
        for bi in 0..self.dict.blocks.len() {
            let Some((n_lo, n_hi)) = self.stale_range(bi, tau, atom_len) else {
                continue;
            };
            for n in n_lo..=n_hi {
                self.refresh_frame(bi, n);
            }
        }
    }

    fn refresh_frame(&mut self, bi: usize, n: usize) {
        let block = &self.dict.blocks[bi];
        let onset = block.frame_onset(n);
        let (k, p) = scan_frame(&mut self.corrs[bi], block, &self.residual, onset);
        self.states[bi].energy[n] = p.energy;
        self.states[bi].bin[n] = k as u32;
        self.states[bi].tree.set(n, p.energy);
    }

    /// Recompute every frame — the reference behaviour.
    fn refresh_all(&mut self) {
        for bi in 0..self.dict.blocks.len() {
            for n in 0..self.states[bi].energy.len() {
                self.refresh_frame(bi, n);
            }
        }
    }

    /// Per-frame best energies for one block. Exposed so the incremental and full-recompute paths
    /// can be compared table-for-table, which is far more sensitive than comparing the atoms they
    /// go on to select.
    pub fn frame_energy(&self, bi: usize) -> &[f64] {
        &self.states[bi].energy
    }

    /// Per-frame argmax bins for one block.
    pub fn frame_bin(&self, bi: usize) -> &[u32] {
        &self.states[bi].bin
    }

    /// Frames the incremental path would mark stale — exposed for testing the arithmetic directly.
    pub fn stale_range(&self, bi: usize, tau: usize, atom_len: usize) -> Option<(usize, usize)> {
        let block = &self.dict.blocks[bi];
        let frames = self.states[bi].energy.len();
        if frames == 0 {
            return None;
        }
        let n_lo = tau.saturating_sub(block.support_len() - 1).div_ceil(block.hop);
        let n_hi = ((tau + atom_len).saturating_sub(1) / block.hop).min(frames - 1);
        (n_lo <= n_hi).then_some((n_lo, n_hi))
    }
}

fn scan_frame(
    corr: &mut Correlator,
    block: &crate::dict::Block,
    signal: &[f32],
    onset: usize,
) -> (usize, crate::corr::Projection) {
    corr.correlate(block, signal, onset);
    corr.best_bin(block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict::BlockConfig;
    use crate::fft::Planner;
    use crate::naive::{NaiveConfig, NaiveMp};

    const SR: f32 = 48_000.0;

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

    fn noise(n: usize, seed: u64) -> Signal {
        let mut s = seed;
        let samples = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / 8_388_608.0 - 1.0
            })
            .collect();
        Signal::new(samples, SR)
    }

    fn on_grid(d: &Dictionary, block: usize, bin: usize, t0: i64, amp: f32, phi: f32) -> AtomParams {
        let b = &d.blocks[block];
        assert!(bin >= b.k_lo && bin <= b.k_hi, "bin {bin} out of range");
        AtomParams {
            t0,
            f: b.bin_hz(bin),
            env: b.env.params,
            phi,
            amp,
        }
    }

    fn run(d: &Dictionary, sig: &Signal, cfg: &MpConfig) -> (Book, Vec<f32>) {
        let mut planner = Planner::new();
        let mut mp = Mp::new(d, sig, &mut planner);
        let book = mp.run(cfg);
        (book, mp.residual().to_vec())
    }

    /// The gate for this stage.
    #[test]
    fn incremental_update_is_bit_identical_to_full_recompute() {
        let d = tiny_dict();
        let sig = noise(600, 0xabcd_ef01_2345_6789);
        let cfg = MpConfig {
            max_atoms: 25,
            target_snr_db: f32::INFINITY,
            ..Default::default()
        };

        let (fast, fast_res) = run(&d, &sig, &cfg);
        let (slow, slow_res) = run(
            &d,
            &sig,
            &MpConfig {
                full_update: true,
                ..cfg
            },
        );

        assert_eq!(fast.len(), slow.len(), "different atom counts");
        assert!(fast.len() >= 20, "only {} atoms selected", fast.len());
        for (i, (a, b)) in fast.selections.iter().zip(&slow.selections).enumerate() {
            assert_eq!(a, b, "selection {i} differs:\n  fast {a:?}\n  slow {b:?}");
        }
        assert_eq!(fast_res, slow_res, "residuals differ");
    }

    /// The strong form of the gate: compare the entire frame table after every single atom.
    ///
    /// Comparing only the selected atoms is too weak to catch an off-by-one in the stale range.
    /// The frames at the range's edges overlap the atom where its envelope has already decayed to
    /// `fade_level` (-60 dB), so omitting one barely perturbs its energy and the selection order
    /// survives. Comparing the stored tables detects a missed frame immediately, whether or not it
    /// would ever have been selected.
    #[test]
    fn stale_set_leaves_every_frame_table_identical_to_full_recompute() {
        let d = tiny_dict();
        let sig = noise(600, 0x0fed_cba9_8765_4321);
        let mut planner = Planner::new();
        let mut fast = Mp::new(&d, &sig, &mut planner);
        let mut slow = Mp::new(&d, &sig, &mut planner);

        let one = MpConfig {
            max_atoms: 1,
            target_snr_db: f32::INFINITY,
            ..Default::default()
        };
        let one_full = MpConfig {
            full_update: true,
            ..one
        };

        for step in 0..15 {
            let a = fast.run(&one);
            let b = slow.run(&one_full);
            assert_eq!(a.selections, b.selections, "step {step}: different atom");
            assert_eq!(a.len(), 1, "step {step}: pursuit stopped early");

            for bi in 0..d.blocks.len() {
                let (fe, se) = (fast.frame_energy(bi), slow.frame_energy(bi));
                assert_eq!(fe.len(), se.len());
                for n in 0..fe.len() {
                    assert_eq!(
                        fe[n], se[n],
                        "step {step} block {bi} frame {n}: stale energy {} vs fresh {}",
                        fe[n], se[n]
                    );
                }
                assert_eq!(fast.frame_bin(bi), slow.frame_bin(bi), "step {step} block {bi}");
            }
        }
    }

    #[test]
    fn incremental_matches_full_recompute_on_a_planted_signal() {
        let d = tiny_dict();
        let atoms = [
            on_grid(&d, 0, 10, 30, 1.0, 0.3),
            on_grid(&d, 1, 20, 250, 0.8, 1.2),
            on_grid(&d, 0, 18, 400, 0.6, 2.4),
        ];
        let sig = Signal::from_atoms(&atoms, 700, SR).unwrap();
        let cfg = MpConfig {
            max_atoms: 15,
            target_snr_db: f32::INFINITY,
            ..Default::default()
        };

        let (fast, _) = run(&d, &sig, &cfg);
        let (slow, _) = run(&d, &sig, &MpConfig { full_update: true, ..cfg });
        assert_eq!(fast.selections, slow.selections);
    }

    /// Stale bounds must be exactly the frames that overlap, checked against the definition.
    #[test]
    fn stale_range_matches_the_overlap_definition() {
        let d = tiny_dict();
        let sig = noise(800, 1);
        let mut planner = Planner::new();
        let mp = Mp::new(&d, &sig, &mut planner);

        for bi in 0..d.blocks.len() {
            let block = &d.blocks[bi];
            let frames = block.frame_count(sig.len());
            for &tau in &[0usize, 1, 37, 200, 500, 799] {
                for &atom_len in &[1usize, 50, 300] {
                    let want: Vec<usize> = (0..frames)
                        .filter(|&n| {
                            let start = block.frame_onset(n);
                            let end = start + block.support_len();
                            start < tau + atom_len && end > tau
                        })
                        .collect();
                    let got = mp.stale_range(bi, tau, atom_len);

                    match (want.first(), want.last()) {
                        (Some(&lo), Some(&hi)) => {
                            assert_eq!(
                                got,
                                Some((lo, hi)),
                                "block {bi} tau {tau} len {atom_len}"
                            );
                            // Contiguity: the overlap set must be an unbroken run.
                            assert_eq!(want.len(), hi - lo + 1);
                        }
                        _ => assert_eq!(got, None, "block {bi} tau {tau} len {atom_len}"),
                    }
                }
            }
        }
    }

    #[test]
    fn recovers_planted_atoms() {
        let d = tiny_dict();
        let atoms = [
            on_grid(&d, 0, 12, 40, 1.0, 0.5),
            on_grid(&d, 0, 24, 300, 0.7, 1.9),
        ];
        let sig = Signal::from_atoms(&atoms, 600, SR).unwrap();
        let (book, _) = run(&d, &sig, &MpConfig { max_atoms: 10, target_snr_db: 40.0, ..Default::default() });

        assert!(book.snr_db() > 30.0, "SNR only {} dB", book.snr_db());
        let onsets: Vec<usize> = book.selections.iter().take(2).map(|s| s.onset).collect();
        assert!(onsets.contains(&40), "missed onset 40, got {onsets:?}");
        assert!(onsets.contains(&300), "missed onset 300, got {onsets:?}");
    }

    /// Against the independent brute-force implementation, with hop forced to 1 so both search the
    /// same onsets. This is the check that the fast path is not merely self-consistent.
    #[test]
    fn matches_the_naive_oracle_when_hop_is_one() {
        let mut planner = Planner::new();
        let cfg = BlockConfig {
            f_min: 1000.0,
            f_max: 4000.0,
            ..BlockConfig::default()
        };
        let mut d =
            Dictionary::from_grid(&[(2147.0, 0.0003)], SR, &mut planner, &cfg).unwrap();
        d.blocks[0].hop = 1; // align the search spaces

        let sig = noise(320, 0x1111_2222_3333_4444);
        let (fast, _) = run(
            &d,
            &sig,
            &MpConfig {
                max_atoms: 5,
                target_snr_db: f32::INFINITY,
                ..Default::default()
            },
        );

        let oracle = NaiveMp::new(&d);
        let (slow, _) = oracle.run(
            &sig,
            &NaiveConfig {
                max_atoms: 5,
                target_residual_fraction: 0.0,
                min_gain_fraction: 0.0,
            },
        );

        assert_eq!(fast.len(), slow.len());
        for (i, (a, b)) in fast.selections.iter().zip(&slow.selections).enumerate() {
            assert_eq!(a.onset, b.onset, "atom {i}: onset");
            assert_eq!(a.bin, b.bin, "atom {i}: bin");
            let rel = (a.projected_energy - b.projected_energy).abs()
                / b.projected_energy.abs().max(1e-12);
            assert!(rel < 1e-3, "atom {i}: energy {} vs {}", a.projected_energy, b.projected_energy);
        }
    }

    #[test]
    fn residual_energy_decreases_and_matches_recomputation() {
        let d = tiny_dict();
        let sig = noise(500, 0x9999_8888_7777_6666);
        let mut planner = Planner::new();
        let mut mp = Mp::new(&d, &sig, &mut planner);
        let book = mp.run(&MpConfig {
            max_atoms: 20,
            target_snr_db: f32::INFINITY,
            ..Default::default()
        });

        let mut prev = book.initial_energy;
        for s in &book.selections {
            assert!(s.residual_energy < prev, "energy rose {prev} -> {}", s.residual_energy);
            prev = s.residual_energy;
        }
        let recomputed = crate::signal::energy_of(mp.residual());
        assert!(
            (recomputed - mp.residual_energy()).abs() < 1e-6 * book.initial_energy,
            "incremental {} vs recomputed {recomputed}",
            mp.residual_energy()
        );
    }

    #[test]
    fn stops_at_the_snr_target() {
        let d = tiny_dict();
        let atoms = [on_grid(&d, 0, 15, 50, 1.0, 0.0)];
        let sig = Signal::from_atoms(&atoms, 500, SR).unwrap();
        let (book, _) = run(&d, &sig, &MpConfig { target_snr_db: 20.0, ..Default::default() });
        assert!(book.snr_db() >= 20.0);
        assert!(book.len() < 5, "took {} atoms to reach 20 dB", book.len());
    }

    #[test]
    fn silence_selects_nothing() {
        let d = tiny_dict();
        let sig = Signal::silence(300, SR);
        let (book, _) = run(&d, &sig, &MpConfig::default());
        assert!(book.is_empty(), "selected {} atoms from silence", book.len());
    }
}
