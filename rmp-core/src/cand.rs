//! Coarse candidate discovery: local time-frequency maxima, merged across blocks.
//!
//! Ordinary MP takes the single global argmax of the frame tables and commits to it. That is
//! optimal only when the dictionary contains the true atom, which stops being true the moment
//! refinement can move an atom off the grid: the best *seed* is then not necessarily the seed that
//! refines to the best atom. So the pursuit promotes several seeds and lets refinement decide.
//!
//! # Only local maxima are promoted
//!
//! A strong event lights up a run of consecutive frames, all reporting nearly the same energy.
//! Taking the top `k` raw frames would spend the whole candidate budget on one event. Keeping only
//! frames that are local maxima in frame index, then suppressing the survivors within half an
//! atom's duration, gives `k` seeds that are actually distinct events.
//!
//! # Tie-breaking
//!
//! `(energy desc, block asc, frame asc)`, resolving ties to the lowest index.
//! The plateau test is `e[n] > e[n-1] && e[n] >= e[n+1]`, which picks the *leftmost* frame of a
//! plateau — the lowest-index frame of the run. Both are needed for `candidate_count = 1`
//! to reproduce the plain global argmax exactly.

use crate::dict::Block;
use crate::fof::AtomParams;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One block's frame table, as the candidate search sees it.
pub struct FrameTable<'a> {
    pub block: usize,
    /// Best projected energy per frame.
    pub energy: &'a [f64],
    /// Bin achieving it.
    pub bin: &'a [u32],
    /// Frames holding a bound rather than an exact value, hidden from the scan. `None` where every
    /// frame is exact.
    ///
    /// The pursuit used to hide them by building a masked *copy* of every table on every pass —
    /// one `f64` per frame of the whole dictionary, allocated and freed per selected atom. Masking
    /// on read is the same predicate with no allocation at all.
    pub dirty: Option<&'a [bool]>,
    /// Frames the search may select from — the leading `core_frames`, under the windowed pursuit.
    ///
    /// Frames past the core exist and are kept current, because atoms starting inside the core
    /// reach into them, but selecting one would place an atom the next window is responsible for.
    /// `None` means the whole table, which is the single-window case.
    pub core_frames: Option<usize>,
    pub hop: usize,
    pub support_len: usize,
}

impl FrameTable<'_> {
    /// The energy the scan sees at frame `n`: `-inf` where the frame is dirty, outside the core, or
    /// non-finite.
    ///
    /// A non-finite stored value must not block its neighbour either — `x >= NAN` is false, so
    /// comparing against the raw value would silently drop the frame next to it.
    fn masked(&self, n: usize) -> f64 {
        if self.core_frames.is_some_and(|c| n >= c) {
            return f64::NEG_INFINITY;
        }
        if self.dirty.is_some_and(|d| d[n]) {
            return f64::NEG_INFINITY;
        }
        let e = self.energy[n];
        if e.is_finite() { e } else { f64::NEG_INFINITY }
    }
}

/// A promoted local maximum, before any refinement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seed {
    pub block: usize,
    pub frame: usize,
    pub bin: usize,
    /// Grid onset in samples, `frame * hop`.
    pub onset: usize,
    pub energy: f64,
}

/// A candidate carried through refinement to selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub seed: Seed,
    /// Replayable parameters. Seeded from the block, overwritten by refinement.
    pub atom: AtomParams,
    /// Ordinary-MP captured energy at `atom`, on the full support.
    pub mp_score: f64,
    /// HRMP score, when HRMP ran on this candidate.
    pub hr_score: Option<f64>,
    /// Whether refinement moved the parameters off the seed.
    pub refined: bool,
}

impl Candidate {
    /// Seed a candidate from its block's own envelope, bin and frame onset.
    pub fn from_seed(seed: Seed, block: &Block, amp: f32, phi: f32, mp_score: f64) -> Self {
        Self {
            seed,
            atom: AtomParams {
                t0: seed.onset as i64,
                f: block.bin_hz(seed.bin),
                env: block.env.params,
                phi,
                amp,
            },
            mp_score,
            hr_score: None,
            refined: false,
        }
    }

    /// The score selection compares: the HR score when HRMP ran, the ordinary one otherwise.
    pub fn score(&self) -> f64 {
        self.hr_score.unwrap_or(self.mp_score)
    }
}

/// Frames across all tables above which the blocks are scanned in parallel.
///
/// A scan costs about 6 ns a frame against tens of microseconds to wake the pool, and it runs twice
/// per selected atom. The line sits between 3 s of piano at the `chopin-nocturne-2.toml` grid
/// (4,399 frames, serial) and 10 s (14,632, parallel); every test fixture is below it.
const PARALLEL_SCAN_FRAMES: usize = 1 << 13;

/// The best `k` distinct seeds across every block.
///
/// Seeds are suppressed against already-accepted ones from the *same* block whose onsets are within
/// half a support length. There is deliberately no suppression *across* blocks: two blocks peaking
/// at the same instant are proposing different envelope shapes for it, which is exactly the choice
/// refinement exists to make.
///
/// **That is what lets the blocks be searched independently.** The seeds kept from one block depend
/// only on that block's own maxima in rank order, so each block's best `k` after suppression can be
/// found on its own, and the best `k` of all of those, by rank, are exactly the seeds one scan over
/// every table in rank order would keep. The blocks are scanned in parallel once the tables are
/// large enough to pay for it; the result is the same either way.
pub fn top_seeds(tables: &[FrameTable<'_>], k: usize) -> Vec<Seed> {
    if k == 0 {
        return Vec::new();
    }
    let frames: usize = tables.iter().map(|t| t.energy.len()).sum();
    let mut seeds: Vec<Seed> = if frames >= PARALLEL_SCAN_FRAMES {
        tables.par_iter().flat_map_iter(|t| block_seeds(t, k)).collect()
    } else {
        tables.iter().flat_map(|t| block_seeds(t, k)).collect()
    };
    seeds.sort_by(by_rank);
    seeds.truncate(k);
    seeds
}

/// One table's best `k` seeds after suppression, strongest first.
fn block_seeds(table: &FrameTable<'_>, k: usize) -> Vec<Seed> {
    // Suppression can reject an arbitrary number of maxima, so a capped scan can in principle run
    // out of candidates before it has `k`. It cannot then be trusted, and the cap is doubled and
    // the scan repeated. This is a safety net, not a path: `k` is `candidate_count`, 1 by default,
    // for which the strongest maximum is always kept, and suppression only rejects seeds within
    // half a support of one already kept.
    let mut cap = (k * 64).max(64);
    loop {
        let (kept, saturated) = top_seeds_capped(std::slice::from_ref(table), k, cap);
        if kept.len() == k || !saturated {
            return kept;
        }
        cap *= 2;
    }
}

/// [`top_seeds`] over the strongest `cap` local maxima, and whether that many were found.
fn top_seeds_capped(tables: &[FrameTable<'_>], k: usize, cap: usize) -> (Vec<Seed>, bool) {
    // Only the strongest few maxima can survive suppression, so only the strongest few are kept.
    //
    // Collecting *every* local maximum and sorting it was the obvious form and is unusable at
    // scale: on dense material the maxima are a sizeable fraction of the frames, so a long clip
    // built and sorted a multi-million-element vector per selected atom. `cap` is the retained
    // prefix; because `by_rank` is a strict total order, the top-`cap` set is uniquely determined
    // and identical to what sorting the whole thing and truncating would give.
    let mut heap: BinaryHeap<Weakest> = BinaryHeap::with_capacity(cap + 1);
    for t in tables {
        for n in 0..t.energy.len() {
            let e = t.masked(n);
            if !e.is_finite() {
                continue;
            }
            let rising = n == 0 || e > t.masked(n - 1);
            let falling = n + 1 == t.energy.len() || e >= t.masked(n + 1);
            if !(rising && falling) {
                continue;
            }
            let seed = Seed {
                block: t.block,
                frame: n,
                bin: t.bin[n] as usize,
                onset: n * t.hop,
                energy: e,
            };
            // The heap's root is the *weakest* retained seed, so a full heap admits a newcomer only
            // by displacing it.
            if heap.len() < cap {
                heap.push(Weakest(seed));
            } else if by_rank(&seed, &heap.peek().expect("cap > 0").0).is_lt() {
                heap.pop();
                heap.push(Weakest(seed));
            }
        }
    }
    let saturated = heap.len() == cap;

    let mut maxima: Vec<Seed> = heap.into_iter().map(|w| w.0).collect();
    maxima.sort_by(by_rank);

    let mut kept: Vec<Seed> = Vec::with_capacity(k);
    for s in maxima {
        if kept.len() == k {
            break;
        }
        let support = tables
            .iter()
            .find(|t| t.block == s.block)
            .map_or(0, |t| t.support_len);
        let crowded = kept.iter().any(|o| {
            o.block == s.block && o.onset.abs_diff(s.onset) * 2 < support
        });
        if !crowded {
            kept.push(s);
        }
    }
    (kept, saturated)
}

/// The seed ordering: strongest first, ties to the lowest block then the lowest frame.
///
/// A strict total order over distinct `(block, frame)`, which is what lets the capped scan retain
/// exactly the prefix an unbounded sort would.
fn by_rank(a: &Seed, b: &Seed) -> Ordering {
    b.energy
        .total_cmp(&a.energy)
        .then(a.block.cmp(&b.block))
        .then(a.frame.cmp(&b.frame))
}

/// A `Seed` ordered so that [`BinaryHeap`]'s maximum is the *weakest* under [`by_rank`].
struct Weakest(Seed);

impl Ord for Weakest {
    fn cmp(&self, other: &Self) -> Ordering {
        by_rank(&other.0, &self.0)
    }
}
impl PartialOrd for Weakest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Weakest {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Weakest {}

#[cfg(test)]
mod tests {
    use super::*;

    fn table<'a>(block: usize, energy: &'a [f64], bin: &'a [u32], hop: usize, support: usize)
    -> FrameTable<'a> {
        FrameTable { block, energy, bin, dirty: None, core_frames: None, hop, support_len: support }
    }

    /// The property that makes `candidate_count = 1` reproduce the old behaviour exactly.
    #[test]
    fn the_first_seed_is_the_global_argmax() {
        let a = [1.0, 5.0, 2.0, 9.0, 1.0];
        let b = [3.0, 8.0, 0.5];
        let (ba, bb) = ([0u32; 5], [1u32; 3]);
        let seeds = top_seeds(&[table(0, &a, &ba, 10, 1), table(1, &b, &bb, 10, 1)], 8);
        assert_eq!(seeds[0].block, 0);
        assert_eq!(seeds[0].frame, 3);
        assert_eq!(seeds[0].energy, 9.0);
    }

    #[test]
    fn only_local_maxima_are_promoted() {
        // A single broad ridge must yield one seed, not five.
        let e = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let bins = [0u32; 7];
        let seeds = top_seeds(&[table(0, &e, &bins, 10, 1)], 8);
        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].frame, 3);
    }

    #[test]
    fn plateaus_resolve_to_the_lowest_frame() {
        // Matches SegTree::argmax, so the two paths cannot disagree about which frame won.
        let e = [1.0, 7.0, 7.0, 7.0, 1.0];
        let bins = [0u32; 5];
        let seeds = top_seeds(&[table(0, &e, &bins, 10, 1)], 8);
        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].frame, 1);
    }

    #[test]
    fn edges_can_win() {
        let bins = [0u32; 3];
        let rising = [9.0, 2.0, 1.0];
        assert_eq!(top_seeds(&[table(0, &rising, &bins, 1, 1)], 4)[0].frame, 0);
        let falling = [1.0, 2.0, 9.0];
        assert_eq!(top_seeds(&[table(0, &falling, &bins, 1, 1)], 4)[0].frame, 2);
    }

    #[test]
    fn suppression_is_within_a_block_only() {
        // Two peaks a quarter of a support apart in the same block: the weaker is suppressed.
        let e = [0.0, 9.0, 0.0, 8.0, 0.0];
        let bins = [0u32; 5];
        let same = top_seeds(&[table(0, &e, &bins, 10, 100)], 8);
        assert_eq!(same.len(), 1, "{same:?}");

        // The same two peaks in different blocks are both kept: they propose different envelopes
        // for one event, which is the choice refinement is there to make.
        let hi = [0.0, 9.0, 0.0];
        let lo = [0.0, 8.0, 0.0];
        let b3 = [0u32; 3];
        let split = top_seeds(&[table(0, &hi, &b3, 10, 100), table(1, &lo, &b3, 10, 100)], 8);
        assert_eq!(split.len(), 2);
        assert_eq!((split[0].block, split[1].block), (0, 1));
    }

    #[test]
    fn ties_break_by_block_then_frame() {
        let e = [0.0, 5.0, 0.0, 5.0, 0.0];
        let bins = [0u32; 5];
        // support 1 disables suppression, so both peaks survive to be ordered.
        let seeds = top_seeds(&[table(1, &e, &bins, 10, 1), table(0, &e, &bins, 10, 1)], 8);
        assert_eq!(
            seeds.iter().map(|s| (s.block, s.frame)).collect::<Vec<_>>(),
            vec![(0, 1), (0, 3), (1, 1), (1, 3)]
        );
    }

    #[test]
    fn non_finite_frames_are_skipped() {
        let e = [f64::NEG_INFINITY, 1.0, f64::NAN];
        let bins = [0u32; 3];
        let seeds = top_seeds(&[table(0, &e, &bins, 10, 1)], 8);
        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].frame, 1);
    }

    /// Searching the blocks separately and merging by rank is exactly one scan over every table in
    /// rank order — the original form, which `top_seeds_capped` over all the tables still is.
    ///
    /// Big enough to take the parallel path and small enough to also check the serial one, with
    /// dirty frames, a core, non-finite values and plateaus. The last block is a dense comb of
    /// strong maxima packed far closer than half its support, so suppression rejects hundreds in a
    /// row and a block's capped scan runs out before it has `k` — the retry has to grow the cap.
    #[test]
    fn searching_blocks_separately_matches_one_scan_over_all_of_them() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rng = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for len in [300usize, 4_000] {
            let blocks: Vec<(Vec<f64>, Vec<u32>, Vec<bool>)> = (0..5)
                .map(|_| {
                    let energy = (0..len)
                        .map(|_| match rng() % 50 {
                            0 => f64::NAN,
                            1 => f64::NEG_INFINITY,
                            // Coarse values, so plateaus and exact ties across blocks occur.
                            _ => (rng() % 64) as f64,
                        })
                        .collect();
                    let bin = (0..len).map(|_| (rng() % 1000) as u32).collect();
                    let dirty = (0..len).map(|_| rng() % 5 == 0).collect();
                    (energy, bin, dirty)
                })
                .collect();
            // Peaks every third frame, strictly weakening, all above every random value.
            let comb: Vec<f64> =
                (0..len).map(|n| if n % 3 == 0 { 1_000.0 - n as f64 * 1e-3 } else { 0.0 }).collect();
            let comb_bin = vec![0u32; len];
            let mut tables: Vec<FrameTable<'_>> = blocks
                .iter()
                .enumerate()
                .map(|(b, (energy, bin, dirty))| FrameTable {
                    block: b,
                    energy,
                    bin,
                    dirty: (b % 2 == 0).then_some(dirty.as_slice()),
                    core_frames: (b == 3).then_some(len * 2 / 3),
                    hop: 3 + b,
                    support_len: [1, 40, 400, 4_000, 90][b],
                })
                .collect();
            tables.push(FrameTable {
                block: 5,
                energy: &comb,
                bin: &comb_bin,
                dirty: None,
                core_frames: None,
                hop: 1,
                support_len: len / 4,
            });
            let everything = 6 * len + 1;
            for k in [1, 2, 3, 8, 40, 200] {
                let one_scan = top_seeds_capped(&tables, k, everything).0;
                assert_eq!(top_seeds(&tables, k), one_scan, "len {len}, k {k}");
            }
        }
    }

    #[test]
    fn empty_inputs_and_zero_k() {
        let bins = [0u32; 0];
        assert!(top_seeds(&[table(0, &[], &bins, 10, 1)], 4).is_empty());
        assert!(top_seeds(&[], 4).is_empty());
        let e = [1.0];
        let b1 = [0u32; 1];
        assert!(top_seeds(&[table(0, &e, &b1, 10, 1)], 0).is_empty());
    }
}
