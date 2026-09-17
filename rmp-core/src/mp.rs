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
//! # The lazy update
//!
//! A stale frame is not recomputed; it is *bounded*. A frame's best energy is `||P r||^2` for `P`
//! the orthogonal projector onto `span(E sin, E cos)`, and subtracting `a` changes `r` by `a`.
//! Since `||P(r - a)|| <= ||P r|| + ||P a||`,
//!
//! ```text
//! E_new <= (sqrt(E_old) + ||P a||)^2        for every bin, hence for the best one
//! ```
//!
//! and `||P a||^2` is bounded two ways, O(1) each given prefix sums of `|a|` and `a^2` and a
//! range-max over the envelope: by `||a||^2`, and by `(sum |a| E)^2 / lambda_min(G)`, since
//! `|<a, E e^{-i omega t}>| <= sum |a| E` at every frequency. The table holds the smaller and the
//! frame is marked dirty. Nothing is transformed until a dirty frame's bound reaches the top of the
//! table, and then only that frame is recomputed — see [`Mp::top_candidates`].
//!
//! The point is that most stale frames never get there. A frame whose window catches only the
//! decayed tail of the subtracted atom moves by a rounding error; the eager update paid a full
//! transform to learn that. On a low-`alpha` dictionary those transforms are 331k points, the stale
//! set is every frame in the block, and they were 85% of the runtime.
//!
//! # Equivalence
//!
//! The selected atom is identical to the eager update's, by construction: a bound is never below
//! the true value, so once every frame at or above the weakest seed is exact, nothing dirty can
//! outrank or shadow a seed. [`MpConfig::full_update`] recomputes every frame of every block each
//! iteration and is the reference; the lazy path must produce a bit-identical book, and its bounds
//! must never be seen to undercut an exact value — both are tests.

use crate::book::{Book, Selection};
use crate::cand::{Candidate, FrameTable, Seed, top_seeds};
use crate::corr::Correlator;
use crate::dict::Dictionary;
use crate::fft::RealFftPlanner;
use crate::fit;
use crate::hrmp::{self, HrmpConfig};
use crate::refine::{EnvelopeCache, RefineConfig, refine};
use crate::signal::{Signal, overlap, snr_db, subtract_at_core};
use rayon::prelude::*;

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
    ///
    /// It has to be generous. A rejection is local — one frame whose residual does not support the
    /// atom the dictionary proposes there — and on dense polyphonic material a long run of them is
    /// ordinary, not a signal that the decomposition is finished. Each stall demotes at least one
    /// frame, so a run of stalls is progress through the table rather than a spin; setting this
    /// small turns a strict HRMP configuration into an early stop, which reads as "HRMP produces
    /// far too few atoms".
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
            max_stalls: 4096,
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

/// Per-block frame table.
///
/// **There is deliberately no max tree here.** One used to sit alongside `energy`, but its only
/// reader was [`Mp::global_argmax`], which is called from a `debug_assert!` and nowhere else —
/// seeds come off [`top_seeds`], which scans linearly anyway. In a release build it was therefore
/// pure cost: a `2 * next_power_of_two(frames)` array of `f64` is 16–32 bytes per frame against the
/// 13 the table itself needs, so it was over half of a structure that already scales with the
/// signal, plus an `O(log F)` write per stale frame per atom on the hot path.
struct BlockState {
    /// Best projected energy per frame — or, where `dirty`, an upper bound on it.
    energy: Vec<f64>,
    /// Bin achieving that energy. Stale wherever `dirty`.
    bin: Vec<u32>,
    /// Frames holding a bound rather than an exact value.
    dirty: Vec<bool>,
    /// Running maxima of the envelope from the left and from the right, and where it peaks, for
    /// an O(1) upper bound on the envelope over any index range.
    env_prefix_max: Vec<f32>,
    env_suffix_max: Vec<f32>,
    env_peak: usize,
    /// The smallest eigenvalue of any live bin's Gram, `(P/2)(1 - max rho)`. A correlation of
    /// magnitude `|d|` at any bin captures at most `|d|^2 / lambda_min` of energy.
    lambda_min: f64,
}

impl BlockState {
    /// An upper bound on the envelope over indices `[s, e)`.
    ///
    /// Exact when the range contains the peak; otherwise the range lies on one flank, and the
    /// running maximum from that flank's end is at least the range's own maximum.
    fn env_max(&self, s: usize, e: usize) -> f32 {
        if s <= self.env_peak && self.env_peak < e {
            self.env_prefix_max[self.env_peak]
        } else if s > self.env_peak {
            self.env_suffix_max[s]
        } else {
            self.env_prefix_max[e - 1]
        }
    }
}

pub struct Mp<'a> {
    dict: &'a Dictionary,
    corrs: Vec<Correlator>,
    /// Further correlators per block, forked from `corrs` when a refresh splits a heavy block's
    /// frames across threads, and kept for the refreshes after.
    spares: Vec<Vec<Correlator>>,
    states: Vec<BlockState>,
    residual: Vec<f32>,
    /// Envelopes rendered during refinement, reused across candidates and iterations.
    cache: EnvelopeCache,
    /// Frames bounded rather than recomputed, and frames recomputed on demand, over the run.
    marked: usize,
    resolved: usize,
    /// The same, per block, so the cost of resolution can be attributed to transform length.
    per_block: Vec<(usize, usize)>,
    energy: f64,
    initial_energy: f64,
    /// The leading range of the residual this pursuit is responsible for, and its energy now and at
    /// the start. Equal to the whole residual unless [`Mp::with_core`] said otherwise.
    ///
    /// The stopping rule reads these, not the whole-buffer figures: under [`run_windowed`] the tail
    /// past the core is a guard the *next* window decomposes, so counting its energy would make
    /// every window look permanently unfinished and spend its whole atom budget failing to finish.
    core_len: usize,
    core_energy: f64,
    initial_core_energy: f64,
    sample_rate: f32,
}

impl<'a> Mp<'a> {
    /// Correlate every frame of every block once, up front.
    pub fn new(dict: &'a Dictionary, signal: &Signal, planner: &mut dyn RealFftPlanner) -> Self {
        let len = signal.len();
        Self::with_core(dict, signal, planner, len)
    }

    /// [`Mp::new`], restricted to selecting atoms whose onset falls in `[0, core_len)`.
    ///
    /// The rest of the signal is still correlated, still refreshed, and still subtracted from — it
    /// is what an atom near the core's end extends into, and scoring against a truncated support
    /// would rank that atom by how much residual it was allowed to ignore rather than by what it
    /// removes. It is simply not *selected from*.
    pub fn with_core(
        dict: &'a Dictionary,
        signal: &Signal,
        planner: &mut dyn RealFftPlanner,
        core_len: usize,
    ) -> Self {
        let mut corrs: Vec<Correlator> = dict
            .blocks
            .iter()
            .map(|b| Correlator::new(b, planner))
            .collect();

        // Correlating every frame of every block is the one unavoidable up-front cost, and the
        // blocks are independent, so it goes wide.
        let samples = &signal.samples;
        let states: Vec<BlockState> = corrs
            .par_iter_mut()
            .zip(dict.blocks.par_iter())
            .map(|(corr, block)| {
                let frames = block.frame_count(signal.len());
                let mut energy = vec![f64::NEG_INFINITY; frames];
                let mut bin = vec![0u32; frames];
                for n in 0..frames {
                    let (k, p) = scan_frame(corr, block, samples, block.frame_onset(n));
                    energy[n] = p.energy;
                    bin[n] = k as u32;
                }
                let env = &block.env.samples;
                let mut env_prefix_max = env.to_vec();
                for i in 1..env_prefix_max.len() {
                    env_prefix_max[i] = env_prefix_max[i].max(env_prefix_max[i - 1]);
                }
                let mut env_suffix_max = env.to_vec();
                for i in (0..env_suffix_max.len().saturating_sub(1)).rev() {
                    env_suffix_max[i] = env_suffix_max[i].max(env_suffix_max[i + 1]);
                }
                let env_peak = env
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv { (i, v) } else { (bi, bv) }
                    })
                    .0;

                let max_rho = block
                    .live_bins()
                    .filter_map(|k| block.rho(k))
                    .fold(0.0f32, f32::max) as f64;
                let lambda_min = 0.5 * block.energy * (1.0 - max_rho);

                BlockState {
                    dirty: vec![false; frames],
                    energy,
                    bin,
                    env_prefix_max,
                    env_suffix_max,
                    env_peak,
                    lambda_min,
                }
            })
            .collect();

        let initial_energy = signal.energy();
        let core_len = core_len.min(signal.len());
        let initial_core_energy = if core_len == signal.len() {
            initial_energy
        } else {
            signal.samples[..core_len].iter().map(|&x| (x as f64) * (x as f64)).sum()
        };
        Self {
            dict,
            spares: corrs.iter().map(|_| Vec::new()).collect(),
            corrs,
            states,
            residual: signal.samples.clone(),
            cache: EnvelopeCache::new(),
            marked: 0,
            resolved: 0,
            per_block: vec![(0, 0); dict.blocks.len()],
            energy: initial_energy,
            initial_energy,
            core_len,
            core_energy: initial_core_energy,
            initial_core_energy,
            sample_rate: signal.sample_rate,
        }
    }

    /// Frames of block `bi` whose onset falls in the core, or `None` when that is all of them.
    ///
    /// Frame `n` starts at `n * hop`, so `n * hop < core_len` is `n < ceil(core_len / hop)`.
    fn core_frames(&self, bi: usize) -> Option<usize> {
        if self.core_len == self.residual.len() {
            return None;
        }
        Some(self.core_len.div_ceil(self.dict.blocks[bi].hop))
    }

    /// The leading range this pursuit selects from, and its energy now.
    pub fn core(&self) -> (usize, f64) {
        (self.core_len, self.core_energy)
    }

    pub fn residual(&self) -> &[f32] {
        &self.residual
    }

    pub fn residual_energy(&self) -> f64 {
        self.energy
    }

    pub fn run(&mut self, cfg: &MpConfig) -> Book {
        self.run_with(cfg, &|| false)
    }

    /// [`Mp::run`], interruptible.
    ///
    /// `cancel` is polled once per iteration, before any work is done for it, and a true answer
    /// ends the pursuit and returns the book built so far. It must be **sticky** — once true it has
    /// to stay true — because that is how the caller distinguishes a run that was cut short from
    /// one that met `target_snr_db`: the book itself looks the same either way.
    ///
    /// Polling is per *iteration*, not per unit of time. A single iteration over a low-alpha
    /// dictionary is hundreds of milliseconds, so this bounds the wait by one atom, not by
    /// anything finer.
    pub fn run_with(&mut self, cfg: &MpConfig, cancel: &dyn Fn() -> bool) -> Book {
        let mut book = Book::new(self.initial_energy, self.sample_rate);
        let mut stalls = 0usize;

        // `max_atoms` bounds *selected* atoms, not iterations: an iteration in which HRMP rejects
        // every candidate adds nothing to the book, so charging it to the atom budget would let a
        // strict HRMP setting exhaust the budget on atoms it refused. Termination does not depend
        // on this loop being counted, because each rejection demotes at least one frame to zero and
        // `max_stalls` bounds how many may pass without a selection.
        while book.len() < cfg.max_atoms {
            if cancel() {
                break;
            }
            if snr_db(self.initial_core_energy, self.core_energy) >= cfg.target_snr_db {
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
            if top.energy <= self.core_energy * cfg.min_gain_fraction {
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
            //
            // This happens whether or not some *other* seed was selected. A rejection is a fact
            // about the residual at that frame, so leaving the frame's energy standing because a
            // different candidate won means the next iteration promotes the same doomed seed again
            // and burns a candidate slot on it every iteration until an atom happens to overlap it.
            let demoted = !chosen.demote.is_empty();
            for &(bi, frame, energy) in &chosen.demote {
                self.states[bi].energy[frame] = energy;
                // A demotion is a deliberate value, not a bound: it must not be "resolved" back.
                self.states[bi].dirty[frame] = false;
            }

            let Some(best) = chosen.best else {
                if !demoted {
                    break;
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
            let Some((src_start, tau, atom_len)) =
                overlap(self.residual.len(), rendered.len(), best.atom.t0)
            else {
                break;
            };

            let before = self.energy;
            let (after, core_after) = subtract_at_core(
                &mut self.residual,
                &rendered,
                best.atom.t0,
                before,
                self.core_energy,
                self.core_len,
            );
            (self.energy, self.core_energy) = (after, core_after);
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
                self.mark_stale(tau, &rendered[src_start..src_start + atom_len]);
            }
        }

        book
    }

    /// The strongest seeds this iteration, best first — every one of them exact.
    ///
    /// The tables may hold bounds. A seed list read off them is trustworthy once every dirty frame
    /// at or above the weakest seed has been recomputed: a bound is never below the truth, so a
    /// dirty frame under that line can neither belong in the list nor be the inflated neighbour
    /// that hides a real local maximum from it. Resolving lowers values, which can lower the line
    /// and expose more dirty frames above it, so this iterates; each pass makes at least one frame
    /// exact, so it ends.
    fn top_candidates(&mut self, k: usize) -> Vec<Seed> {
        // Seeds are read off the *clean* frames only: a dirty frame is masked out of the scan. The
        // line is then set by certain values, and one pass resolves every dirty frame above it as
        // a single parallel batch. Setting the line from the unmasked table instead was measured
        // as a pathology: when the top entry is an inflated bound, each pass resolves exactly that
        // one frame, its value drops, the next bound becomes the top, and a single atom costs
        // hundreds of serial passes over a 66k-frame table.
        //
        // A masked seed is a true local maximum once nothing dirty remains above the line: its
        // clean neighbours are in the scan, and a dirty neighbour's bound — hence its value — is
        // below it. So the list returned is the same one the eager tables would give.
        //
        // The masking is a *predicate on read* rather than a masked copy of the tables. The copy
        // was one `f64` per frame of the whole dictionary, allocated and freed on every pass of
        // this loop — hundreds of megabytes per selected atom on a long clip, for a value that is
        // `NEG_INFINITY` wherever `dirty` and the stored energy everywhere else.
        loop {
            let seeds = {
                let tables: Vec<FrameTable<'_>> = self
                    .states
                    .iter()
                    .enumerate()
                    .map(|(bi, st)| FrameTable {
                        block: bi,
                        energy: &st.energy,
                        bin: &st.bin,
                        dirty: Some(&st.dirty),
                        core_frames: self.core_frames(bi),
                        hop: self.dict.blocks[bi].hop,
                        support_len: self.dict.blocks[bi].support_len(),
                    })
                    .collect();
                top_seeds(&tables, k)
            };
            // Fewer clean seeds than asked for means there is no line to reason about; resolve
            // everything dirty, which is the eager behaviour and equally correct.
            let threshold = if seeds.len() == k {
                seeds[k - 1].energy
            } else {
                f64::NEG_INFINITY
            };
            let pending = self.dirty_frames(|e| e >= threshold);
            if pending.iter().all(Vec::is_empty) {
                return seeds;
            }
            self.resolve(&pending);
        }
    }

    /// Each block's dirty frames whose stored bound satisfies `keep`, in frame order.
    ///
    /// A pass over every frame of every block, twice per selected atom, so on a long clip it is
    /// worth spreading across the pool; each block's list is its own, so the result is the same.
    fn dirty_frames(&self, keep: impl Fn(f64) -> bool + Sync) -> Vec<Vec<usize>> {
        let scan = |st: &BlockState| -> Vec<usize> {
            st.dirty
                .iter()
                .zip(&st.energy)
                .enumerate()
                .filter(|&(_, (&d, &e))| d && keep(e))
                .map(|(n, _)| n)
                .collect()
        };
        let frames: usize = self.states.iter().map(|s| s.energy.len()).sum();
        if frames >= PARALLEL_SCAN_FRAMES {
            self.states.par_iter().map(scan).collect()
        } else {
            self.states.iter().map(scan).collect()
        }
    }

    /// Recompute the listed frames of each block exactly.
    fn resolve(&mut self, per_block: &[Vec<usize>]) {
        for (bi, frames) in per_block.iter().enumerate() {
            self.resolved += frames.len();
            self.per_block[bi].1 += frames.len();
        }
        let exact = refresh_frames(
            &self.dict.blocks,
            &mut self.corrs,
            &mut self.spares,
            &self.residual,
            per_block,
        );
        for (bi, (frames, exact)) in per_block.iter().zip(exact).enumerate() {
            let state = &mut self.states[bi];
            for (&n, (bin, energy)) in frames.iter().zip(exact) {
                debug_assert!(
                    energy <= state.energy[n],
                    "block {bi} frame {n}: exact {energy} exceeds its bound {}",
                    state.energy[n]
                );
                (state.energy[n], state.bin[n], state.dirty[n]) = (energy, bin, false);
            }
        }
    }

    /// Make every frame exact. The tables then equal a full recompute's, which is what lets the
    /// table-for-table gate keep its teeth on the lazy path.
    pub fn resolve_all(&mut self) {
        let pending = self.dirty_frames(|_| true);
        if !pending.iter().all(Vec::is_empty) {
            self.resolve(&pending);
        }
    }

    /// Frames bounded rather than recomputed, and frames recomputed on demand, so far.
    pub fn lazy_stats(&self) -> (usize, usize) {
        (self.marked, self.resolved)
    }

    /// [`Mp::lazy_stats`] per block.
    pub fn lazy_stats_per_block(&self) -> &[(usize, usize)] {
        &self.per_block
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
    ///
    /// Only the `debug_assert!` in [`Mp::run`] calls this, so a linear scan is the right shape: it
    /// costs a debug build one pass over the tables per atom, and it saves a release build the max
    /// tree that used to answer it in `O(log F)` at 16–32 bytes per frame. Reads the raw table,
    /// bounds included, which is what the assertion compares against.
    fn global_argmax(&self) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, f64)> = None;
        for (bi, st) in self.states.iter().enumerate() {
            let core = self.core_frames(bi).unwrap_or(st.energy.len());
            for (n, &e) in st.energy.iter().enumerate().take(core) {
                if e.is_finite() && best.as_ref().is_none_or(|&(_, _, b)| e > b) {
                    best = Some((bi, n, e));
                }
            }
        }
        best.map(|(b, f, _)| (b, f))
    }

    /// Bound every frame whose read window overlaps the subtracted atom, without transforming any.
    ///
    /// `atom` is the subtracted samples as they landed in the residual, starting at `tau`. The range
    /// itself comes from [`stale_range_of`], so the arithmetic has exactly one implementation and
    /// the test that pins it against the overlap definition guards this path too.
    fn mark_stale(&mut self, tau: usize, atom: &[f32]) {
        // Prefix sums of |a| and a^2, so either norm of the atom under any window is a subtraction.
        let mut a1 = Vec::with_capacity(atom.len() + 1);
        let mut a2 = Vec::with_capacity(atom.len() + 1);
        a1.push(0.0f64);
        a2.push(0.0f64);
        for &x in atom {
            let x = x as f64;
            a1.push(a1.last().unwrap() + x.abs());
            a2.push(a2.last().unwrap() + x * x);
        }
        let atom_end = tau + atom.len();
        let blocks = &self.dict.blocks;

        // Each block bounds only its own frames, so the blocks go across the pool once there are
        // enough stale frames to pay for waking it. Returns how many frames the block bounded.
        let bound_block = |(bi, st): (usize, &mut BlockState)| -> usize {
            let block = &blocks[bi];
            let Some((n_lo, n_hi)) = stale_range_of(block, st.energy.len(), tau, atom.len()) else {
                return 0;
            };
            let support = block.support_len();
            let mut marked = 0;
            for n in n_lo..=n_hi {
                let onset = block.frame_onset(n);
                let (lo, hi) = (onset.max(tau), (onset + support).min(atom_end));
                if hi <= lo {
                    continue;
                }

                // The frame's energy is ||P R||^2 for P the orthogonal projector onto
                // span(E sin, E cos) — the projection of the *residual*, not of the windowed
                // signal the transform sees. So the change is the atom `a` itself, and the
                // envelope enters once, through the basis. (Writing the change as a*E and
                // bounding with E^2 double-counts it; that bound undercut by a hair on a frame
                // whose energy rose, and the gate caught it.)
                //
                // Two valid bounds on how much energy the change can add at any one bin; the
                // smaller is taken.
                //
                // The norm bound: ||P a||^2 <= ||a||^2 = sum a^2 over the overlap. No envelope at
                // all, so it is loose wherever the window catches only the atom's tail.
                //
                // The correlation bound: for every omega, |<a, E e^{-i omega t}>| <= sum |a| E,
                // and d'G^-1 d <= |d|^2 / lambda_min. Within a small constant of the matched
                // projection by Cauchy-Schwarz however long the window, and it decays with the
                // envelope — which is the case that matters.
                //
                // The overlap is chunked so the envelope's range-max tracks its decay rather than
                // pinning every chunk to the value at the overlap's start.
                let (mut d_max, mut w2) = (0.0f64, 0.0f64);
                let chunks = 8usize.min(hi - lo);
                for c in 0..chunks {
                    let cs = lo + (hi - lo) * c / chunks;
                    let ce = lo + (hi - lo) * (c + 1) / chunks;
                    if ce <= cs {
                        continue;
                    }
                    let e_max = st.env_max(cs - onset, ce - onset) as f64;
                    d_max += e_max * (a1[ce - tau] - a1[cs - tau]);
                    w2 += a2[ce - tau] - a2[cs - tau];
                }
                let delta = w2.min(d_max * d_max / st.lambda_min);

                // The stored value may itself be a bound; compounding bounds is still a bound. The
                // margin covers the f32 transform's rounding, which the algebra knows nothing of.
                let old = st.energy[n].max(0.0);
                let bound = (old.sqrt() + delta.sqrt()).powi(2) * (1.0 + 1e-4) + 1e-12;
                st.energy[n] = bound;
                st.dirty[n] = true;
                marked += 1;
            }
            marked
        };

        let stale: usize = self
            .states
            .iter()
            .zip(blocks)
            .filter_map(|(st, b)| stale_range_of(b, st.energy.len(), tau, atom.len()))
            .map(|(lo, hi)| hi - lo + 1)
            .sum();
        let marked: Vec<usize> = if stale >= PARALLEL_BOUND_FRAMES {
            self.states.par_iter_mut().enumerate().map(bound_block).collect()
        } else {
            self.states.iter_mut().enumerate().map(bound_block).collect()
        };
        for (bi, m) in marked.into_iter().enumerate() {
            self.marked += m;
            self.per_block[bi].0 += m;
        }
    }

    /// Recompute every frame — the reference behaviour.
    fn refresh_all(&mut self) {
        let every: Vec<Vec<usize>> = self.states.iter().map(|s| (0..s.energy.len()).collect()).collect();
        let exact = refresh_frames(
            &self.dict.blocks,
            &mut self.corrs,
            &mut self.spares,
            &self.residual,
            &every,
        );
        for (state, exact) in self.states.iter_mut().zip(exact) {
            for (n, (bin, energy)) in exact.into_iter().enumerate() {
                (state.energy[n], state.bin[n], state.dirty[n]) = (energy, bin, false);
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
        stale_range_of(
            &self.dict.blocks[bi],
            self.states[bi].energy.len(),
            tau,
            atom_len,
        )
    }
}

/// Recompute `frames[b]` of every block `b`, returning each frame's `(bin, energy)` in the order
/// listed.
///
/// The whole per-frame cost of the pursuit — one FFT and one bin scan — lives inside this. A frame's
/// result depends only on the residual and its block, never on which correlator or thread computed
/// it, so the parallel result is identical to computing the frames in sequence, which is what lets
/// the bit-identity gates keep their teeth.
///
/// **The work is divided by transform samples, not by block.** A block's frames used to run in
/// sequence on one thread, and a low-`alpha` block pending several 337,500-point frames was then a
/// chain every other thread waited on: on 10 s of piano the `alpha = 1` blocks averaged six pending
/// frames a pass and up to 25, and summing each pass's largest block gave 2.9x the floor that
/// dividing the work evenly allows. So each block's list is cut into chunks of about the pass's
/// work over the thread count — a heavy block into several, each on a correlator forked from the
/// block's own and kept in `spares` for the passes after, and a light block left whole, which keeps
/// the locality that made a pure per-frame split measure 30% slower.
///
/// Small jobs stay on one thread: rayon's per-task overhead is real next to a handful of short
/// frames, and the test fixtures are all that size.
fn refresh_frames(
    blocks: &[crate::dict::Block],
    corrs: &mut [Correlator],
    spares: &mut [Vec<Correlator>],
    residual: &[f32],
    frames: &[Vec<usize>],
) -> Vec<Vec<(u32, f64)>> {
    let scan = |bi: usize, corr: &mut Correlator, list: &[usize]| -> Vec<(u32, f64)> {
        let block = &blocks[bi];
        list.iter()
            .map(|&n| {
                let (k, p) = scan_frame(corr, block, residual, block.frame_onset(n));
                (k as u32, p.energy)
            })
            .collect()
    };

    let work: usize = frames.iter().zip(blocks).map(|(f, b)| f.len() * b.fft_len).sum();
    if work < PARALLEL_TRANSFORM_SAMPLES {
        return corrs.iter_mut().zip(frames).enumerate().map(|(bi, (c, f))| scan(bi, c, f)).collect();
    }

    let threads = rayon::current_num_threads().max(1);
    let target = work.div_ceil(threads);
    let mut tasks: Vec<(usize, &mut Correlator, &[usize])> = Vec::new();
    for (bi, ((corr, spare), list)) in corrs.iter_mut().zip(spares.iter_mut()).zip(frames).enumerate() {
        if list.is_empty() {
            continue;
        }
        let chunks = (list.len() * blocks[bi].fft_len).div_ceil(target).clamp(1, list.len().min(threads));
        while spare.len() + 1 < chunks {
            spare.push(corr.fork());
        }
        let size = list.len().div_ceil(chunks);
        for (c, part) in std::iter::once(corr).chain(spare.iter_mut()).zip(list.chunks(size)) {
            tasks.push((bi, c, part));
        }
    }

    let done: Vec<(usize, Vec<(u32, f64)>)> = tasks
        .into_par_iter()
        .with_max_len(1)
        .map(|(bi, c, part)| (bi, scan(bi, c, part)))
        .collect();
    let mut out = vec![Vec::new(); frames.len()];
    for (bi, part) in done {
        out[bi].extend(part);
    }
    out
}

/// Bytes of frame table one frame costs: `energy` (f64), `bin` (u32) and `dirty` (bool).
///
/// Used to size a window from a memory budget, so it has to track [`BlockState`]. It is the whole
/// per-frame cost — everything else in a `BlockState` is per *support sample* or a scalar.
const FRAME_TABLE_BYTES: usize = 8 + 4 + 1;

/// How a signal too large to analyse at once is cut into windows.
///
/// The pursuit's working set is `sum_b frames_b` frame tables, and `frames_b = n / hop_b` with
/// `hop_b ∝ 1/alpha_b`, so it grows linearly in the signal with a constant the *dictionary* sets —
/// over five frames per input sample at the default grid and tolerance. Nothing about that can be
/// streamed away: the tables are working state, not results. What can be bounded is `n`.
///
/// So the signal is analysed in windows and the tables sized to one window. This is the one change
/// in the crate that alters which atoms are selected: greedy order becomes per-window rather than
/// global, and `target_snr_db` / `min_gain` / `max_atoms` become per-window quantities. A signal
/// that fits the budget is a single window and takes the original path exactly, bit for bit — which
/// is both the compatibility guarantee and what keeps every existing oracle gate meaningful.
#[derive(Clone, Copy, Debug)]
pub struct WindowPlan {
    /// Samples each window is responsible for selecting atoms in.
    pub core_len: usize,
    /// Samples carried past the core so a core atom is scored on its whole support.
    pub guard_len: usize,
    pub count: usize,
    /// Frame-table bytes one input sample costs, from the dictionary's own hops.
    pub bytes_per_sample: f64,
    /// Whether `core_len` had to exceed what was asked for to stay above the guard floor.
    pub over_budget: bool,
}

impl WindowPlan {
    /// Size the windows for `signal_len` under a frame-table budget.
    ///
    /// `forced_core` overrides the budget, for a run that must be reproducible across machines.
    pub fn new(
        dict: &Dictionary,
        signal_len: usize,
        cfg: &MpConfig,
        budget_bytes: usize,
        forced_core: Option<usize>,
    ) -> Self {
        // The longest atom the run can produce. Refinement is bounded by `max_atom_samples` rather
        // than by any block, and it can lower `alpha` past the grid, so it sets the guard whenever
        // it is on. Anything shorter and an atom selected at the core's end would be scored against
        // a residual that stops inside its own support.
        let longest_block = dict.blocks.iter().map(|b| b.support_len()).max().unwrap_or(0);
        let guard_len = if cfg.refine.enabled {
            longest_block.max(cfg.refine.max_atom_samples)
        } else {
            longest_block
        }
        .min(signal_len);

        let bytes_per_sample: f64 = dict
            .blocks
            .iter()
            .map(|b| FRAME_TABLE_BYTES as f64 / b.hop as f64)
            .sum();

        let from_budget = || {
            let affordable = (budget_bytes as f64 / bytes_per_sample.max(f64::MIN_POSITIVE)) as usize;
            affordable.saturating_sub(guard_len)
        };
        let wanted = forced_core.unwrap_or_else(from_budget);

        // A core shorter than a few guards is not worth cutting: every window would re-correlate
        // more guard than core, and an atom deferred out of one window's tail would land in the
        // next window's tail again. Correctness wins over the budget here, and the caller is told.
        let floor = (4 * guard_len).max(1);
        let core_len = wanted.max(floor).min(signal_len.max(1));
        // Reported whenever the floor overrode what was asked for, whether that was the budget or
        // an explicit `window_seconds`: in both cases the run is not doing what the setting said.
        let over_budget = core_len < signal_len && core_len > wanted;

        Self {
            core_len,
            guard_len,
            count: signal_len.div_ceil(core_len.max(1)).max(1),
            bytes_per_sample,
            over_budget,
        }
    }

    /// Frame-table bytes one window costs.
    pub fn window_bytes(&self) -> f64 {
        (self.core_len + self.guard_len) as f64 * self.bytes_per_sample
    }
}

/// A windowed run's result: the book, what it could not explain, and the refresh counters.
pub struct WindowedRun {
    pub book: Book,
    pub residual: Vec<f32>,
    /// [`Mp::lazy_stats`], summed over the windows.
    pub marked: usize,
    pub resolved: usize,
    pub per_block: Vec<(usize, usize)>,
    /// Time spent correlating every frame up front, summed over the windows.
    ///
    /// A windowed run pays this once per window rather than once, which is the cost of the whole
    /// scheme; reporting it separately is what makes that cost visible instead of folded into the
    /// pursuit and invisible.
    pub init: std::time::Duration,
}

/// Decompose `signal` a window at a time, so the frame tables never size to the whole clip.
///
/// Each window owns `[offset, offset + core)` and carries a guard past it; the pursuit selects only
/// from the core, and an atom starting in the core is fully inside the window by construction. The
/// window's residual — guard included — is written back before the next window reads it, so every
/// atom is subtracted exactly once and an atom the argmax wanted in the guard is simply deferred to
/// the window that owns it.
///
/// `progress` is called with `(window index, windows, atoms so far)` after each window. `cancel` is
/// polled both between windows and inside each window's own pursuit — see [`Mp::run_with`] for what
/// it has to promise.
pub fn run_windowed(
    dict: &Dictionary,
    signal: &Signal,
    planner: &mut dyn RealFftPlanner,
    cfg: &MpConfig,
    plan: &WindowPlan,
    progress: &mut dyn FnMut(usize, usize, usize),
    cancel: &dyn Fn() -> bool,
) -> WindowedRun {
    if plan.count <= 1 {
        // The original path, untouched, so a signal that fits the budget is bit-identical.
        let t = std::time::Instant::now();
        let mut mp = Mp::new(dict, signal, planner);
        let init = t.elapsed();
        let book = mp.run_with(cfg, cancel);
        let (marked, resolved) = mp.lazy_stats();
        let per_block = mp.lazy_stats_per_block().to_vec();
        return WindowedRun { book, residual: mp.residual, marked, resolved, per_block, init };
    }

    let sr = signal.sample_rate;
    let total = signal.len();
    let mut residual = signal.samples.clone();
    let mut book = Book::new(signal.energy(), sr);
    // The book's `residual_energy` column is a global running total, not a per-window one: the SNR
    // curve `stats` and `rmpstat snr` read off it has to mean the same thing end to end.
    let mut running = signal.energy();
    let (mut marked, mut resolved) = (0usize, 0usize);
    let mut per_block = vec![(0usize, 0usize); dict.blocks.len()];

    let mut init = std::time::Duration::ZERO;
    let mut offset = 0usize;
    let mut w = 0usize;
    while offset < total {
        if cancel() {
            break;
        }
        let core = plan.core_len.min(total - offset);
        let end = (offset + core + plan.guard_len).min(total);
        let window = Signal::new(residual[offset..end].to_vec(), sr);

        // The atom budget is shared out by duration, so `max_atoms` still bounds the whole run.
        let share = (cfg.max_atoms as u128 * core as u128 / total as u128) as usize;
        let wcfg = MpConfig { max_atoms: share.max(1), ..*cfg };

        let t = std::time::Instant::now();
        let mut mp = Mp::with_core(dict, &window, planner, core);
        init += t.elapsed();
        let wbook = mp.run_with(&wcfg, cancel);

        residual[offset..end].copy_from_slice(&mp.residual);
        let (m, r) = mp.lazy_stats();
        marked += m;
        resolved += r;
        for (acc, &(bm, br)) in per_block.iter_mut().zip(mp.lazy_stats_per_block()) {
            acc.0 += bm;
            acc.1 += br;
        }

        for mut s in wbook.selections {
            s.atom.t0 += offset as i64;
            s.onset += offset;
            // `energy_removed` is measured over the atom's whole support inside its window, so it
            // is the atom's true removal and accumulates into a global residual energy directly.
            running = (running - s.energy_removed).max(0.0);
            s.residual_energy = running;
            book.selections.push(s);
        }

        offset += core;
        w += 1;
        progress(w, plan.count, book.len());
    }

    WindowedRun { book, residual, marked, resolved, per_block, init }
}

/// Frames below which collecting the dirty frames stays on one thread: about 6 ns a frame against
/// tens of microseconds to wake the pool, twice per selected atom. The same line as `cand`'s scan,
/// for the same reason.
const PARALLEL_SCAN_FRAMES: usize = 1 << 13;

/// Stale frames below which bounding them stays on one thread. A bound is about 170 ns — eight
/// range-maxima and prefix-sum lookups — so this line is far lower than the scan's.
const PARALLEL_BOUND_FRAMES: usize = 1 << 10;

/// Transform samples below which the blocks are refreshed on one thread — a few hundred
/// microseconds of work, against rayon's tens for waking the pool.
///
/// **Measured in transform samples, not frames.** This used to be the size of the whole frame
/// table, which says nothing about the work: a frame of an `alpha = 1` block is a 337,500-point
/// transform. A 2 s excerpt has a table under the old 4096-frame line, so every refresh of those
/// transforms ran on one thread: 100% CPU on a 24-thread machine, with the refresh 5–20x slower
/// than the same frames spread across the blocks.
const PARALLEL_TRANSFORM_SAMPLES: usize = 1 << 16;

/// Frames of `block` whose read window overlaps `[tau, tau + atom_len)`.
///
/// The single definition, used by the incremental update and by the test that pins it against the
/// overlap predicate directly.
fn stale_range_of(
    block: &crate::dict::Block,
    frames: usize,
    tau: usize,
    atom_len: usize,
) -> Option<(usize, usize)> {
    if frames == 0 {
        return None;
    }
    let n_lo = tau.saturating_sub(block.support_len() - 1).div_ceil(block.hop);
    let n_hi = ((tau + atom_len).saturating_sub(1) / block.hop).min(frames - 1);
    (n_lo <= n_hi).then_some((n_lo, n_hi))
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
    use crate::atom::Shape;
    use crate::dict::BlockConfig;
    use crate::fft::Planner;
    use crate::fof::{AtomParams, EnvelopeParams};
    use crate::gauss::GaussianParams;
    use crate::hrmp::HrmpConfig;
    use crate::refine::RefineConfig;
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

    /// One short FOF block and one short Gaussian block, so the gates that run on `tiny_dict` also
    /// run across kinds: cross-block ranking, the stale range and the lazy bound all have to hold
    /// when the blocks' envelopes are shaped nothing alike.
    fn mixed_dict() -> Dictionary {
        let mut planner = Planner::new();
        let cfg = BlockConfig { f_min: 500.0, f_max: 6000.0, ..BlockConfig::default() };
        let shapes: Vec<Shape> =
            vec![EnvelopeParams::new(2147.0, 0.0003).into(), GaussianParams::new(0.0004).into()];
        Dictionary::from_shapes(&shapes, SR, &mut planner, &cfg).unwrap()
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
        let mut mp = Mp::new(&d, &sig, &mut planner);

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

    /// Spec 17.7: identical input and configuration must give an identical book.
    ///
    /// Run with every stage engaged — multiple candidates, refinement and HRMP — because each adds
    /// a place where an unordered container or a non-total float comparison could leak in.
    #[test]
    fn the_decomposition_is_reproducible() {
        let d = tiny_dict();
        let sig = noise(3_000, 0xd00d_1234_5678_9abc);
        let cfg = MpConfig {
            max_atoms: 12,
            target_snr_db: f32::INFINITY,
            candidate_count: 4,
            refine: RefineConfig { enabled: true, ..RefineConfig::default() },
            hrmp: HrmpConfig { enabled: true, ..HrmpConfig::default() },
            ..Default::default()
        };

        let (first, first_res) = run(&d, &sig, &cfg);
        assert!(first.len() >= 5, "only {} atoms selected", first.len());
        for _ in 0..3 {
            let (again, again_res) = run(&d, &sig, &cfg);
            assert_eq!(again.selections, first.selections);
            assert_eq!(again_res, first_res);
        }
    }

    /// The lazy update rests on one inequality: a bound is never below the exact value. This
    /// checks it on every dirty frame after every atom, against a fresh exact scan, rather than
    /// trusting the algebra — the f32 transform is where it would quietly fail.
    #[test]
    fn lazy_bounds_never_undercut_the_exact_value() {
        for (name, d) in [("fof", tiny_dict()), ("mixed", mixed_dict())] {
            let sig = noise(600, 0x1eaf_1eaf_1eaf_1eaf);
            let mut planner = Planner::new();
            let mut mp = Mp::new(&d, &sig, &mut planner);
            let one = MpConfig { max_atoms: 1, target_snr_db: f32::INFINITY, ..Default::default() };

            let mut checked = 0usize;
            for step in 0..12 {
                let book = mp.run(&one);
                assert_eq!(book.len(), 1, "{name} step {step}: pursuit stopped early");
                for bi in 0..d.blocks.len() {
                    let block = &d.blocks[bi];
                    let mut corr = Correlator::new(block, &mut planner);
                    for n in 0..mp.states[bi].energy.len() {
                        if !mp.states[bi].dirty[n] {
                            continue;
                        }
                        let (_, p) =
                            scan_frame(&mut corr, block, &mp.residual, block.frame_onset(n));
                        assert!(
                            p.energy <= mp.states[bi].energy[n],
                            "{name} step {step} block {bi} frame {n}: exact {} above bound {}",
                            p.energy,
                            mp.states[bi].energy[n]
                        );
                        checked += 1;
                    }
                }
            }
            assert!(checked > 100, "{name}: only {checked} bounds were ever checked");
            let (marked, resolved) = mp.lazy_stats();
            assert!(
                resolved < marked,
                "{name}: lazy update resolved every frame it bounded: {resolved}/{marked}"
            );
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

    /// A rejection costs an iteration, never a slot in the atom budget.
    ///
    /// The failure this pins down is quiet: with `max_atoms` counting iterations, an HRMP setting
    /// that rejects most candidates returns a book far shorter than the budget and reports having
    /// stopped short of the SNR target, which reads as "HRMP finds no atoms" rather than as a
    /// budget spent on refusals.
    #[test]
    fn rejections_do_not_consume_the_atom_budget() {
        let d = tiny_dict();
        let sig = noise(4_000, 0x5151_2323_9999_1111);
        // Tight enough that most candidates are refused, loose enough that some still pass.
        let hrmp = HrmpConfig {
            enabled: true,
            phase_tolerance_rad: 0.15,
            min_probe_energy: 1e-9,
            depth: 3,
            ..HrmpConfig::default()
        };
        let cfg = MpConfig {
            max_atoms: 12,
            target_snr_db: f32::INFINITY,
            candidate_count: 1,
            max_stalls: 4096,
            hrmp,
            ..Default::default()
        };
        let (book, _) = run(&d, &sig, &cfg);
        assert_eq!(
            book.len(),
            12,
            "the budget is atoms, not iterations: got {} of 12",
            book.len()
        );
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
        for (name, d) in [("fof", tiny_dict()), ("mixed", mixed_dict())] {
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
                assert_eq!(a.selections, b.selections, "{name} step {step}: different atom");
                assert_eq!(a.len(), 1, "{name} step {step}: pursuit stopped early");

                // The lazy table holds bounds; only the resolved table is comparable to a recompute.
                fast.resolve_all();
                for bi in 0..d.blocks.len() {
                    let (fe, se) = (fast.frame_energy(bi), slow.frame_energy(bi));
                    assert_eq!(fe.len(), se.len());
                    for n in 0..fe.len() {
                        assert_eq!(
                            fe[n], se[n],
                            "{name} step {step} block {bi} frame {n}: stale energy {} vs fresh {}",
                            fe[n], se[n]
                        );
                    }
                    assert_eq!(
                        fast.frame_bin(bi),
                        slow.frame_bin(bi),
                        "{name} step {step} block {bi}"
                    );
                }
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
    /// Dividing a refresh by transform samples — heavy blocks cut into chunks, each on a correlator
    /// forked from the block's own — computes exactly what one correlator computes frame by frame.
    ///
    /// Run twice so the second pass reuses the forks with the first pass's buffers still in them.
    #[test]
    fn a_divided_refresh_matches_one_frame_at_a_time() {
        let mut planner = Planner::new();
        let cfg = BlockConfig { f_min: 100.0, f_max: 6000.0, ..BlockConfig::default() };
        let d = Dictionary::from_grid(&[(40.0, 0.001), (300.0, 0.002), (2147.0, 0.0003)], SR, &mut planner, &cfg)
            .unwrap();
        let sig = noise(60_000, 11);
        let frames: Vec<Vec<usize>> = d
            .blocks
            .iter()
            .enumerate()
            .map(|(bi, b)| (0..b.frame_count(sig.len())).filter(|n| n % (bi + 1) == 0).collect())
            .collect();
        let work: usize = frames.iter().zip(&d.blocks).map(|(f, b)| f.len() * b.fft_len).sum();
        assert!(work >= PARALLEL_TRANSFORM_SAMPLES, "the fixture must take the divided path");

        let expected: Vec<Vec<(u32, f64)>> = d
            .blocks
            .iter()
            .zip(&frames)
            .map(|(b, list)| {
                let mut c = Correlator::new(b, &mut planner);
                list.iter()
                    .map(|&n| {
                        let (k, p) = scan_frame(&mut c, b, &sig.samples, b.frame_onset(n));
                        (k as u32, p.energy)
                    })
                    .collect()
            })
            .collect();

        let mut corrs: Vec<Correlator> = d.blocks.iter().map(|b| Correlator::new(b, &mut planner)).collect();
        let mut spares: Vec<Vec<Correlator>> = d.blocks.iter().map(|_| Vec::new()).collect();
        for pass in 0..2 {
            let got = refresh_frames(&d.blocks, &mut corrs, &mut spares, &sig.samples, &frames);
            assert_eq!(got.len(), expected.len());
            for (bi, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(g.len(), e.len(), "pass {pass}, block {bi}");
                for (i, (a, b)) in g.iter().zip(e).enumerate() {
                    assert!(a.0 == b.0 && a.1.to_bits() == b.1.to_bits(), "pass {pass}, block {bi}, frame {i}");
                }
            }
        }
        if rayon::current_num_threads() > 1 {
            assert!(spares.iter().any(|s| !s.is_empty()), "no block was divided");
        }
    }

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

    /// One planted atom of each kind, each found by a block of its own kind at its own onset.
    ///
    /// The two supports are shaped nothing alike — a sharp-attack decay against a symmetric bump —
    /// so a pursuit that ranked them unfairly across blocks would explain one with the other.
    #[test]
    fn recovers_planted_atoms_of_both_kinds() {
        let d = mixed_dict();
        // On each block's own hop grid, or the "on-grid" atom is not.
        let (t_f, t_g) = (40 / d.blocks[0].hop * d.blocks[0].hop, 300 / d.blocks[1].hop * d.blocks[1].hop);
        let atoms = [on_grid(&d, 0, 12, t_f as i64, 1.0, 0.5), on_grid(&d, 1, 8, t_g as i64, 0.7, 1.9)];
        let sig = Signal::from_atoms(&atoms, 600, SR).unwrap();
        let (book, _) = run(&d, &sig, &MpConfig { max_atoms: 10, target_snr_db: 40.0, ..Default::default() });

        assert!(book.snr_db() > 30.0, "SNR only {} dB", book.snr_db());
        let first: Vec<(usize, usize)> =
            book.selections.iter().take(2).map(|s| (s.block, s.onset)).collect();
        assert!(first.contains(&(0, t_f)), "missed the FOF at {t_f}, got {first:?}");
        assert!(first.contains(&(1, t_g)), "missed the gaussian at {t_g}, got {first:?}");
        for s in book.selections.iter().take(2) {
            assert_eq!(s.atom.kind(), d.blocks[s.block].env.kind());
        }
    }

    /// The oracle gate across kinds: the fast path and the brute-force scan pick the same block, onset
    /// and bin, atom after atom, when the dictionary offers both a FOF and a Gaussian.
    #[test]
    fn matches_the_naive_oracle_on_a_mixed_dictionary() {
        let mut planner = Planner::new();
        let cfg = BlockConfig { f_min: 1000.0, f_max: 4000.0, ..BlockConfig::default() };
        let shapes: Vec<Shape> =
            vec![EnvelopeParams::new(2147.0, 0.0003).into(), GaussianParams::new(0.0004).into()];
        let mut d = Dictionary::from_shapes(&shapes, SR, &mut planner, &cfg).unwrap();
        for b in &mut d.blocks {
            b.hop = 1; // align the search spaces
        }

        let sig = noise(320, 0x7777_2222_3333_4444);
        let cfg = MpConfig { max_atoms: 6, target_snr_db: f32::INFINITY, ..Default::default() };
        let (fast, _) = run(&d, &sig, &MpConfig { refine: RefineConfig { enabled: false, ..cfg.refine }, ..cfg });
        let (slow, _) = NaiveMp::new(&d).run(
            &sig,
            &NaiveConfig { max_atoms: 6, target_residual_fraction: 0.0, min_gain_fraction: 0.0 },
        );

        assert_eq!(fast.len(), slow.len());
        let kinds: std::collections::HashSet<_> = fast.selections.iter().map(|s| s.atom.kind()).collect();
        assert_eq!(kinds.len(), 2, "the fixture should exercise both kinds");
        for (i, (a, b)) in fast.selections.iter().zip(&slow.selections).enumerate() {
            assert_eq!((a.block, a.onset, a.bin), (b.block, b.onset, b.bin), "atom {i}");
            let rel = (a.projected_energy - b.projected_energy).abs()
                / b.projected_energy.abs().max(1e-12);
            assert!(rel < 1e-3, "atom {i}: energy {} vs {}", a.projected_energy, b.projected_energy);
        }
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

    // ── the windowed pursuit ────────────────────────────────────────────────

    /// A dictionary whose longest support is short enough to window a test-sized signal.
    fn window_dict() -> Dictionary {
        tiny_dict()
    }

    fn plan_for(d: &Dictionary, len: usize, cfg: &MpConfig, core: usize) -> WindowPlan {
        WindowPlan::new(d, len, cfg, 0, Some(core))
    }

    fn windowed(d: &Dictionary, sig: &Signal, cfg: &MpConfig, plan: &WindowPlan) -> WindowedRun {
        let mut planner = Planner::new();
        run_windowed(d, sig, &mut planner, cfg, plan, &mut |_, _, _| {}, &|| false)
    }

    /// The compatibility guarantee: a signal that fits the budget is not windowed, and the book is
    /// the one the un-windowed pursuit always produced -- bit for bit, not merely close.
    #[test]
    fn a_signal_inside_the_budget_is_one_window_and_bit_identical() {
        let d = window_dict();
        let sig = noise(4000, 0x5eed_1234_9876_0001);
        let cfg = MpConfig { max_atoms: 20, target_snr_db: f32::INFINITY, ..Default::default() };

        // A budget far above what the tables need.
        let plan = WindowPlan::new(&d, sig.len(), &cfg, 1 << 30, None);
        assert_eq!(plan.count, 1, "a small signal should not be windowed");

        let (want, want_res) = run(&d, &sig, &cfg);
        let got = windowed(&d, &sig, &cfg, &plan);
        assert_eq!(got.book.selections, want.selections);
        assert_eq!(got.residual, want_res);
    }

    /// The budget is what decides, and it decides in the direction it says.
    #[test]
    fn a_tighter_budget_cuts_more_windows() {
        let d = window_dict();
        let cfg = MpConfig::default();
        let len = 2_000_000;
        let generous = WindowPlan::new(&d, len, &cfg, 1 << 30, None);
        let tight = WindowPlan::new(&d, len, &cfg, 1 << 20, None);
        assert!(
            tight.count > generous.count,
            "tight {} windows, generous {}",
            tight.count,
            generous.count
        );
        assert!(tight.window_bytes() < generous.window_bytes());
        // Whatever the budget, a window must carry a whole atom's support past its core.
        for p in [generous, tight] {
            assert!(p.guard_len >= d.blocks.iter().map(|b| b.support_len()).max().unwrap());
        }
    }

    /// Windowing changes the greedy order, so nothing here is bit-identity. What must hold is that
    /// the decomposition still works: the residual comes down, and by a comparable amount.
    #[test]
    fn windowing_still_decomposes_the_signal() {
        let d = window_dict();
        let sig = noise(20_000, 0x5eed_9999_1111_2222);
        let cfg = MpConfig { max_atoms: 60, target_snr_db: f32::INFINITY, ..Default::default() };

        let whole = windowed(&d, &sig, &cfg, &WindowPlan::new(&d, sig.len(), &cfg, 1 << 30, None));
        let cut = windowed(&d, &sig, &cfg, &plan_for(&d, sig.len(), &cfg, 4000));
        assert!(cut.book.len() > 1, "the windowed run selected {} atoms", cut.book.len());

        let energy = |r: &[f32]| r.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        let (e0, e_whole, e_cut) = (sig.energy(), energy(&whole.residual), energy(&cut.residual));
        assert!(e_cut < e0, "the windowed run removed no energy");
        // Within a factor of two of the un-windowed run's residual: the orders differ, the quality
        // does not collapse.
        assert!(
            e_cut < 2.0 * e_whole,
            "windowed residual {e_cut:.6e} against un-windowed {e_whole:.6e}"
        );
    }

    /// The book's own bookkeeping has to survive being assembled from several windows: onsets are
    /// global, and `residual_energy` is a global running total the SNR curve can be read off.
    #[test]
    fn a_windowed_book_reads_as_one_decomposition() {
        let d = window_dict();
        let sig = noise(20_000, 0x5eed_4444_5555_6666);
        let cfg = MpConfig { max_atoms: 60, target_snr_db: f32::INFINITY, ..Default::default() };
        let got = windowed(&d, &sig, &cfg, &plan_for(&d, sig.len(), &cfg, 4000));

        assert!(got.book.len() > 1);
        // Onsets are translated out of window coordinates, so they advance through the signal.
        let onsets: Vec<usize> = got.book.selections.iter().map(|s| s.onset).collect();
        assert!(
            onsets.iter().any(|&o| o >= 4000),
            "no atom past the first core: {onsets:?}"
        );
        // The recorded residual energy is monotone and ends at the residual actually left behind.
        let mut prev = f64::INFINITY;
        for s in &got.book.selections {
            assert!(s.residual_energy <= prev, "residual energy rose: {prev} -> {}", s.residual_energy);
            prev = s.residual_energy;
        }
        let measured: f64 = got.residual.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let recorded = got.book.selections.last().unwrap().residual_energy;
        assert!(
            (measured - recorded).abs() <= 1e-6 * measured.max(1e-30) + 1e-9,
            "book says {recorded:.6e}, the residual is {measured:.6e}"
        );
    }

    /// The guard is the whole point: an atom whose onset sits just inside a core must be selected
    /// once, with its full support, not split at the boundary or dropped.
    #[test]
    fn an_atom_astride_a_core_boundary_is_selected_whole() {
        let d = window_dict();
        let core = 4000usize;
        // Planted so that half its support lies on each side of the boundary. Support comes from
        // rendering, never a formula -- the rule the rest of the crate follows.
        let support = crate::fof::Envelope::render(d.blocks[0].env.params, SR)
            .unwrap()
            .support_len();
        let t0 = core as i64 - support as i64 / 2;
        let atom = on_grid(&d, 0, (d.blocks[0].k_lo + d.blocks[0].k_hi) / 2, t0, 1.0, 0.3);
        assert!(t0 as usize + support > core, "the fixture does not cross the boundary");

        let sig = Signal::from_atoms(&[atom], 20_000, SR).unwrap();
        let cfg = MpConfig { max_atoms: 4, target_snr_db: 40.0, ..Default::default() };
        let got = windowed(&d, &sig, &cfg, &plan_for(&d, sig.len(), &cfg, core));

        assert!(!got.book.is_empty(), "the planted atom was not found");
        let first = got.book.selections[0];
        assert!(
            (first.atom.t0 - t0).abs() <= d.blocks[first.block].hop as i64,
            "recovered t0 {} against planted {t0}",
            first.atom.t0
        );
        // Recovered whole: one atom accounts for nearly all the energy, rather than two half-atoms
        // meeting at the boundary.
        assert!(
            first.energy_removed > 0.95 * sig.energy(),
            "the first atom removed {:.3} of the signal's energy",
            first.energy_removed / sig.energy()
        );
    }

    /// The stopping rule reads the core, not the whole window. A window whose guard is loud but
    /// whose core is already explained must stop, or every window spends its whole budget on
    /// material the next window owns.
    #[test]
    fn the_stopping_rule_ignores_the_guard() {
        let d = window_dict();
        let core = 4000usize;
        // Silence in the core, a strong burst well inside the guard.
        let loud = on_grid(&d, 0, (d.blocks[0].k_lo + d.blocks[0].k_hi) / 2, core as i64 + 500, 1.0, 0.0);
        let sig = Signal::from_atoms(&[loud], 20_000, SR).unwrap();

        let cfg = MpConfig { max_atoms: 30, target_snr_db: 20.0, ..Default::default() };
        let mut planner = Planner::new();
        let mut mp = Mp::with_core(&d, &sig, &mut planner, core);
        let book = mp.run(&cfg);
        assert!(
            book.is_empty(),
            "selected {} atoms from a silent core",
            book.len()
        );
    }
}
