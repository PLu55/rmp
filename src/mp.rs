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
use crate::cand::{Candidate, FrameTable, Seed, top_seeds};
use crate::corr::Correlator;
use crate::dict::Dictionary;
use crate::fft::RealFftPlanner;
use crate::fit;
use crate::hrmp::{self, HrmpConfig};
use crate::refine::{EnvelopeCache, RefineConfig, refine};
use crate::select::SegTree;
use crate::signal::{Signal, overlap, snr_db, subtract_at};

#[derive(Clone, Copy, Debug)]
pub struct MpConfig {
    pub max_atoms: usize,
    /// Stop once this reconstruction SNR is reached.
    pub target_snr_db: f32,
    /// Stop when the best atom would remove less than this fraction of the current residual.
    pub min_gain_fraction: f64,
    /// How many local maxima to promote per iteration.
    ///
    /// 1 reduces the pipeline to the plain global argmax, bit for bit.
    ///
    /// Measured on the off-grid benchmark, raising this buys **nothing**: 1, 4, 8 and 64 all reach
    /// 40 dB in 91 atoms with the same splitting factor, while the candidate stage's cost scales
    /// linearly (5 ms at 1, 210 ms at 64, over the same run). The strongest seed is also the seed
    /// that refines best, so the extra work is discarded. It is kept configurable because a
    /// candidate can be *rejected* rather than merely outscored -- which is what HRMP does -- and
    /// then the loop needs somewhere to fall through to.
    pub candidate_count: usize,
    /// Local refinement of the promoted candidates.
    pub refine: RefineConfig,
    /// Local-support validation of the refined candidates.
    pub hrmp: HrmpConfig,
    /// Give up after this many consecutive iterations in which every candidate was rejected.
    ///
    /// Only reachable under HRMP, and it is the spec's "no candidate passes" stopping condition.
    pub max_stalls: usize,
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
            candidate_count: 1,
            // Off by default so the library's default is still the plain, fully-validated pursuit.
            // `Config` turns it on for the CLI.
            refine: RefineConfig { enabled: false, ..RefineConfig::default() },
            hrmp: HrmpConfig::default(),
            max_stalls: 16,
            full_update: false,
        }
    }
}

/// The outcome of scoring one iteration's candidates.
struct Chosen {
    best: Option<Candidate>,
    /// Frames whose stored energy must be lowered because their candidate was rejected outright.
    demote: Vec<(usize, usize, f64)>,
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
    /// Envelopes rendered during refinement, reused across candidates and iterations.
    cache: EnvelopeCache,
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
            cache: EnvelopeCache::new(),
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
        let mut stalls = 0usize;

        for _ in 0..cfg.max_atoms {
            if snr_db(self.initial_energy, self.energy) >= cfg.target_snr_db {
                break;
            }
            // Steps 1-2: promote local maxima and keep the best few.
            let seeds = self.top_candidates(cfg.candidate_count);
            let Some(top) = seeds.first() else { break };
            debug_assert_eq!(
                self.global_argmax(),
                Some((top.block, top.frame)),
                "the strongest seed must be the global argmax"
            );
            if top.energy <= self.energy * cfg.min_gain_fraction {
                break;
            }

            // Steps 3-5: score each seed exactly, refine it off the grid, validate its local
            // support, and select.
            let chosen = self.best_candidate(&seeds, cfg);

            // A rejected candidate is not a reason to stop: it is a reason to look elsewhere. But
            // it must be *demoted* in the frame table first, or the next iteration recomputes the
            // same argmax and the loop spins forever with no error. The demotion is undone
            // naturally the next time `refresh_stale` touches that frame, which is correct — the
            // residual there will have changed, so the verdict has to be retaken.
            let Some(best) = chosen.best else {
                if chosen.demote.is_empty() {
                    break;
                }
                for (bi, frame, energy) in chosen.demote {
                    self.states[bi].energy[frame] = energy;
                    self.states[bi].tree.set(frame, energy);
                }
                stalls += 1;
                if stalls >= cfg.max_stalls {
                    break;
                }
                continue;
            };
            stalls = 0;
            if best.score() <= 0.0 {
                break;
            }

            // Step 6: synthesize and subtract.
            let Ok(rendered) = best.atom.render(self.sample_rate) else {
                break;
            };
            let Some((_, tau, atom_len)) =
                overlap(self.residual.len(), rendered.len(), best.atom.t0)
            else {
                break;
            };

            let before = self.energy;
            self.energy = subtract_at(&mut self.residual, &rendered, best.atom.t0, before);
            book.selections.push(Selection {
                atom: best.atom,
                block: best.seed.block,
                onset: best.seed.onset,
                bin: best.seed.bin,
                projected_energy: best.mp_score,
                energy_removed: before - self.energy,
                residual_energy: self.energy,
                hr_score: best.hr_score,
                refined: best.refined,
            });

            if self.energy >= before {
                break; // a rise means a parameter-mapping bug, not noise
            }

            // The invalidated range is the one `subtract_at` just wrote, from the same `overlap`
            // call — never the seed's frame onset, which stops equalling the atom's `t0` the moment
            // refinement can move it.
            if cfg.full_update {
                self.refresh_all();
            } else {
                self.refresh_stale(tau, atom_len);
            }
        }

        book
    }

    /// The strongest seeds this iteration, best first.
    fn top_candidates(&self, k: usize) -> Vec<Seed> {
        let tables: Vec<FrameTable<'_>> = self
            .states
            .iter()
            .enumerate()
            .map(|(bi, st)| FrameTable {
                block: bi,
                energy: &st.energy,
                bin: &st.bin,
                hop: self.dict.blocks[bi].hop,
                support_len: self.dict.blocks[bi].support_len(),
            })
            .collect();
        top_seeds(&tables, k)
    }

    /// Score every seed and return the best.
    ///
    /// The frame table stores only energy and bin, so each seed's transform is recomputed here to
    /// recover `amp` and `phi`. That is one extra FFT per candidate, which is free next to the
    /// update — and it is where refinement will attach.
    fn best_candidate(&mut self, seeds: &[Seed], cfg: &MpConfig) -> Chosen {
        let mut out = Chosen { best: None, demote: Vec::new() };
        for &seed in seeds {
            let block = &self.dict.blocks[seed.block];
            let (k, p) = scan_frame(&mut self.corrs[seed.block], block, &self.residual, seed.onset);
            if p.energy <= 0.0 {
                continue;
            }
            let seed = Seed { bin: k, ..seed };
            let mut cand = Candidate::from_seed(seed, block, p.amp, p.phi, p.energy);
            if cfg.refine.enabled {
                refine(&mut cand, block, &self.residual, &cfg.refine, &mut self.cache);
            }
            if cfg.hrmp.enabled && !self.apply_hrmp(&mut cand, &cfg.hrmp) {
                out.demote.push((seed.block, seed.frame, 0.0));
                continue;
            }
            if cand.score() <= 0.0 {
                out.demote.push((seed.block, seed.frame, 0.0));
                continue;
            }
            // Strictly greater, so an exact tie keeps the earlier — and better-seeded — candidate.
            if out.best.as_ref().is_none_or(|b| cand.score() > b.score()) {
                out.best = Some(cand);
            }
        }
        out
    }

    /// Validate a candidate's local support, clamping its amplitude. Returns whether it survived.
    fn apply_hrmp(&mut self, cand: &mut Candidate, cfg: &HrmpConfig) -> bool {
        let sr = self.sample_rate;
        let Ok(env) = crate::fof::Envelope::render(cand.atom.env, sr) else {
            return false;
        };
        let omega = std::f64::consts::TAU * cand.atom.f as f64 / sr as f64;
        let Some(quad) = fit::accumulate(&self.residual, &env.samples, cand.atom.t0, omega) else {
            return false;
        };
        let Some(proj) = quad.solve(cfg.rho_sq_max) else {
            return false;
        };

        let at = hrmp::Placement {
            t0: cand.atom.t0,
            f: cand.atom.f,
            fit_len: fit::fit_end(&env),
        };
        let v = hrmp::evaluate(&self.residual, &env, at, &quad, &proj, cfg);
        if !v.accepted() || v.energy <= 0.0 {
            return false;
        }
        // The phase is held at the ordinary fit's; only the amplitude is clamped.
        cand.atom.amp = v.amp;
        cand.hr_score = Some(v.energy);
        true
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
    use crate::fof::{AtomParams, EnvelopeParams};
    use crate::hrmp::HrmpConfig;
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

    /// Promoting more candidates must not change what gets selected until refinement exists.
    ///
    /// Scoring a seed recomputes exactly the quantity the frame table already holds, so the
    /// best-scoring candidate is always the strongest seed however many are promoted. Once
    /// refinement can move an atom off the grid that stops being true — and this test is what says
    /// so, by starting to fail.
    #[test]
    fn candidate_count_does_not_change_selection_without_refinement() {
        let d = tiny_dict();
        let sig = noise(600, 0x2222_3333_4444_5555);
        let base = MpConfig {
            max_atoms: 20,
            target_snr_db: f32::INFINITY,
            ..Default::default()
        };

        let (one, one_res) = run(&d, &sig, &base);
        assert!(one.len() >= 15, "only {} atoms selected", one.len());
        for k in [2, 8, 32] {
            let (many, many_res) = run(&d, &sig, &MpConfig { candidate_count: k, ..base });
            assert_eq!(many.selections, one.selections, "candidate_count = {k}");
            assert_eq!(many_res, one_res, "candidate_count = {k}: residuals differ");
        }
    }

    /// A single event lights up a run of frames; the candidate list must spend its budget on
    /// distinct events instead of on one ridge.
    #[test]
    fn promoted_seeds_are_distinct_events() {
        let d = tiny_dict();
        let b0 = &d.blocks[0];
        let span = b0.k_hi - b0.k_lo;
        let mut sig = Signal::silence(4_000, SR);
        for (t0, off) in [(200i64, span / 4), (1_500, span / 2), (3_000, 3 * span / 4)] {
            let a = on_grid(&d, 0, b0.k_lo + off, t0, 1.0, 0.3);
            crate::signal::add_at(&mut sig.samples, &a.render(SR).unwrap(), t0);
        }
        let mut planner = Planner::new();
        let mp = Mp::new(&d, &sig, &mut planner);

        let seeds = mp.top_candidates(6);
        assert!(seeds.len() >= 3, "only {} seeds", seeds.len());
        for (i, a) in seeds.iter().enumerate() {
            for b in &seeds[i + 1..] {
                assert!(
                    a.block != b.block
                        || a.onset.abs_diff(b.onset) * 2 >= d.blocks[a.block].support_len(),
                    "seeds {a:?} and {b:?} are the same event"
                );
            }
        }
    }

    /// The end-to-end case HRMP exists for: energy invented in a gap.
    ///
    /// The dictionary holds only the *long* shape, so ordinary MP has no choice but to explain two
    /// separated bursts with an atom that spans the silence between them — the adversarial case the
    /// spec describes. (Given the matching short shape, plain MP picks it and never bridges, so the
    /// temptation has to be constructed deliberately.) HRMP must refuse to fill the gap.
    #[test]
    fn hrmp_stops_a_long_atom_inventing_energy_in_a_gap() {
        let mut planner = Planner::new();
        let cfg = BlockConfig { f_min: 300.0, f_max: 4000.0, ..BlockConfig::default() };
        let d = Dictionary::from_grid(&[(80.0, 0.001)], SR, &mut planner, &cfg).unwrap();
        let long_support = d.blocks[0].support_len();

        let f = d.blocks[0].bin_hz((d.blocks[0].k_lo + d.blocks[0].k_hi) / 2);
        let short = EnvelopeParams::new(2147.0, 0.0003);
        let short_len = crate::fof::Envelope::render(short, SR).unwrap().support_len();
        let (t1, t2) = (400i64, 400 + (long_support / 5) as i64);

        let mut sig = Signal::silence(long_support + 8_000, SR);
        for t0 in [t1, t2] {
            let a = AtomParams { t0, f, env: short, phi: 0.4, amp: 1.0 };
            crate::signal::add_at(&mut sig.samples, &a.render(SR).unwrap(), t0);
        }

        // The silence between the bursts, with a margin so neither burst leaks in.
        let gap = (t1 as usize + short_len + 64)..(t2 as usize - 64);
        assert!(gap.end > gap.start, "fixture has no gap");
        assert!(
            crate::signal::energy_of(&sig.samples[gap.clone()]) < 1e-9,
            "the gap must actually be silent"
        );

        let base = MpConfig {
            max_atoms: 8,
            target_snr_db: f32::INFINITY,
            candidate_count: 4,
            ..Default::default()
        };
        let with_hr = MpConfig {
            hrmp: HrmpConfig { enabled: true, ..HrmpConfig::default() },
            ..base
        };

        let mut planner = Planner::new();
        let plain = Mp::new(&d, &sig, &mut planner).run(&base);
        let guarded = Mp::new(&d, &sig, &mut planner).run(&with_hr);
        assert!(!plain.is_empty(), "plain MP selected nothing");

        let gap_energy = |b: &Book| {
            b.resynthesize(sig.len())
                .map(|r| crate::signal::energy_of(&r.samples[gap.clone()]))
                .unwrap_or(f64::INFINITY)
        };
        let (plain_gap, guarded_gap) = (gap_energy(&plain), gap_energy(&guarded));

        assert!(
            plain_gap > 0.0,
            "the fixture is not adversarial: plain MP put no energy in the gap"
        );
        assert!(
            guarded_gap < 0.5 * plain_gap,
            "HRMP put {guarded_gap:.3e} into a silent gap against plain MP's {plain_gap:.3e}"
        );
        assert!(
            guarded.selections.iter().all(|s| s.hr_score.is_some()),
            "HRMP was enabled but recorded no verdicts"
        );

        // Whatever it selects, the pursuit must still be strictly decreasing.
        for w in guarded.selections.windows(2) {
            assert!(w[1].residual_energy < w[0].residual_energy, "residual rose under HRMP");
        }
    }

    /// A rejected candidate must send the loop elsewhere, not stop it and not spin it.
    #[test]
    fn hrmp_rejection_demotes_the_cell_instead_of_stalling() {
        let d = tiny_dict();
        let sig = noise(4_000, 0x5151_2323_9999_1111);
        // Tolerances tight enough that almost nothing survives, so the demotion path is what runs.
        let cfg = MpConfig {
            max_atoms: 10,
            target_snr_db: f32::INFINITY,
            candidate_count: 3,
            max_stalls: 4,
            hrmp: HrmpConfig {
                enabled: true,
                phase_tolerance_rad: 0.02,
                min_probe_energy: 1e-9,
                depth: 3,
                ..HrmpConfig::default()
            },
            ..Default::default()
        };
        let (book, _) = run(&d, &sig, &cfg);
        // The point is that it terminates and stays monotone, whatever it manages to select.
        for w in book.selections.windows(2) {
            assert!(w[1].residual_energy < w[0].residual_energy);
        }
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
