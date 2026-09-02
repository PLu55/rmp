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
//! `(energy desc, block asc, frame asc)`, matching [`crate::select::SegTree`]'s lowest-index rule.
//! The plateau test is `e[n] > e[n-1] && e[n] >= e[n+1]`, which picks the *leftmost* frame of a
//! plateau — again what `SegTree::argmax` would return. Both are needed for `candidate_count = 1`
//! to reproduce the plain global argmax exactly.

use crate::dict::Block;
use crate::fof::AtomParams;

/// One block's frame table, as the candidate search sees it.
pub struct FrameTable<'a> {
    pub block: usize,
    /// Best projected energy per frame.
    pub energy: &'a [f64],
    /// Bin achieving it.
    pub bin: &'a [u32],
    pub hop: usize,
    pub support_len: usize,
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

/// The best `k` distinct seeds across every block.
///
/// Seeds are suppressed against already-accepted ones from the *same* block whose onsets are within
/// half a support length. There is deliberately no suppression *across* blocks: two blocks peaking
/// at the same instant are proposing different envelope shapes for it, which is exactly the choice
/// refinement exists to make.
pub fn top_seeds(tables: &[FrameTable<'_>], k: usize) -> Vec<Seed> {
    if k == 0 {
        return Vec::new();
    }
    // A non-finite neighbour must not block a valid frame: `x >= NAN` is false, so comparing
    // against the raw value would silently drop the frame next to it.
    let at = |v: &[f64], i: usize| {
        let e = v[i];
        if e.is_finite() { e } else { f64::NEG_INFINITY }
    };

    let mut maxima = Vec::new();
    for t in tables {
        for n in 0..t.energy.len() {
            let e = t.energy[n];
            if !e.is_finite() {
                continue;
            }
            let rising = n == 0 || e > at(t.energy, n - 1);
            let falling = n + 1 == t.energy.len() || e >= at(t.energy, n + 1);
            if rising && falling {
                maxima.push(Seed {
                    block: t.block,
                    frame: n,
                    bin: t.bin[n] as usize,
                    onset: n * t.hop,
                    energy: e,
                });
            }
        }
    }

    maxima.sort_by(|a, b| {
        b.energy
            .total_cmp(&a.energy)
            .then(a.block.cmp(&b.block))
            .then(a.frame.cmp(&b.frame))
    });

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
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table<'a>(block: usize, energy: &'a [f64], bin: &'a [u32], hop: usize, support: usize)
    -> FrameTable<'a> {
        FrameTable { block, energy, bin, hop, support_len: support }
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
