//! Linking per-frame peaks into ridges (§11).
//!
//! Each open track predicts where it will be in the next frame by extrapolating its own recent
//! slope, and peaks are assigned to the track whose *prediction* they are nearest. Following the
//! slope rather than the last frequency is what keeps two crossing partials on their own tracks
//! (§46): at the crossing both peaks are near both tracks' last positions, but each is near only one
//! track's prediction. It is also what keeps a vibrato as one track, since a vibrato's frame-to-frame
//! step is mostly slope.
//!
//! Assignment is greedy over every (track, peak) pair within reach, cheapest first, ties broken by
//! track then peak index. That is a total order, so the result does not depend on anything but the
//! peaks. A track that finds no peak survives `max_gap_frames` frames before it is closed.
//!
//! **Reach does not grow across a gap.** A partial that reappears after a few missing frames
//! reappears where it was, so a resumed track may move no further than one frame's
//! `max_jump_cents_per_frame` from its prediction. Letting the reach grow with the frames missed
//! was the first design, and on a piano book it chained consecutive notes into glissandi: a note
//! ended, its track missed two frames, and the next note a semitone away was within 150 cents.
//!
//! **`max_drift_cents` bounds how far a track may wander from its own running mean.** The
//! per-frame limit cannot see a slow slide, and a slow slide is what two overlapping notes look
//! like on the map: while one decays and the next rises, their kernels merge into one peak whose
//! position moves with the balance of energy. For a fixed-pitch instrument no partial moves, so a
//! limit of half a semitone turns every such slide back into two notes. It is off by default,
//! because a vibrato or a glissando is exactly this motion on a voice or a violin.
//!
//! **A crossing is an occlusion, not a gap.** Where two partials cross, the map has one peak for
//! both of them for as long as they are within about a kernel width of each other — several frames
//! for a slow crossing, more than any sensible `max_gap_frames`. The track that loses the merged
//! peak is not missing: its prediction is sitting on a peak another track took. Such a frame counts
//! against a separate, longer allowance of [`MAX_OCCLUDED_FRAMES`], and the track resumes on its own
//! slope when the peaks separate. Without this a crossing never swapped identities, but it cut the
//! losing partial in two. An occluded frame is still unmatched, so it still lowers persistence: a
//! track that only ever shadows another cannot pass as a partial of its own.
//!
//! While occluded, a track's reach grows by its own speed, `|slope|` per frame occluded. Peaks
//! pull toward each other as they approach a crossing, which bends the slope the prediction is
//! extrapolated from, so a crossing track needs some slack to find itself again. A *steady* track
//! gets none. That is not a detail: letting every occluded track's reach grow, speed or no speed,
//! brought back 107 of the 170 piano glissandi, because a neighbouring note's peak within reach
//! is enough to mark a steady track occluded, and from there its growing reach caught the next
//! note.

use super::peaks::Peak;
use crate::config::PartialTrackingConfig;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RidgePoint {
    pub frame: usize,
    pub cents: f64,
    pub energy: f64,
}

/// One linked track: its matched peaks, in frame order. Missed frames are simply absent.
#[derive(Clone, Debug, PartialEq)]
pub struct Ridge {
    pub points: Vec<RidgePoint>,
}

impl Ridge {
    pub fn first_frame(&self) -> usize {
        self.points[0].frame
    }

    pub fn last_frame(&self) -> usize {
        self.points[self.points.len() - 1].frame
    }

    /// Frames from the first matched to the last, inclusive.
    pub fn lifetime_frames(&self) -> usize {
        self.last_frame() - self.first_frame() + 1
    }

    /// Matched frames over the lifetime (§12). In `(0, 1]`.
    pub fn persistence(&self) -> f64 {
        self.points.len() as f64 / self.lifetime_frames() as f64
    }

    /// Where the track expects to be at `frame`: its last position plus its recent slope.
    ///
    /// The slope is taken over up to the last three matched points, which averages out a frame of
    /// interpolation jitter without lagging far behind a vibrato.
    fn predict(&self, frame: usize) -> f64 {
        let last = self.points[self.points.len() - 1];
        last.cents + self.slope() * (frame - last.frame) as f64
    }

    /// Cents per frame over up to the last three matched points; zero for a single point.
    fn slope(&self) -> f64 {
        let n = self.points.len();
        if n < 2 {
            return 0.0;
        }
        let (last, back) = (self.points[n - 1], self.points[n - 1 - (n - 1).min(2)]);
        (last.cents - back.cents) / (last.frame - back.frame) as f64
    }
}

/// The most consecutive frames a track may spend occluded by another before it is closed: 100 ms
/// at the default hop, about the time two partials a semitone apart per second take to pass through
/// each other's kernels.
pub const MAX_OCCLUDED_FRAMES: usize = 10;

struct Open {
    ridge: Ridge,
    /// Sum of the matched positions, for the track's running mean.
    sum_cents: f64,
    /// Consecutive unmatched frames with no taken peak near the prediction.
    missed: usize,
    /// Consecutive unmatched frames spent under another track's peak.
    occluded: usize,
}

/// Link `peaks` (one list per frame) into ridges. Ridges come out in the order they close; the
/// caller sorts.
pub fn track(peaks: &[Vec<Peak>], cfg: &PartialTrackingConfig) -> Vec<Ridge> {
    let mut open: Vec<Open> = Vec::new();
    let mut closed = Vec::new();
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();

    for (frame, frame_peaks) in peaks.iter().enumerate() {
        pairs.clear();
        let mut predicted = Vec::with_capacity(open.len());
        for (t, o) in open.iter().enumerate() {
            // Only an occluded track's reach grows, and only as fast as the track itself moves:
            // see the module docs.
            let reach = cfg.max_jump_cents_per_frame + o.ridge.slope().abs() * o.occluded as f64;
            let at = o.ridge.predict(frame);
            let mean = o.sum_cents / o.ridge.points.len() as f64;
            predicted.push((at, reach));
            for (p, peak) in frame_peaks.iter().enumerate() {
                let d = (peak.cents - at).abs();
                let drift_ok =
                    cfg.max_drift_cents == 0.0 || (peak.cents - mean).abs() <= cfg.max_drift_cents;
                if d <= reach && drift_ok {
                    pairs.push((d, t, p));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        let mut track_taken = vec![false; open.len()];
        let mut peak_taken = vec![false; frame_peaks.len()];
        for &(_, t, p) in &pairs {
            if track_taken[t] || peak_taken[p] {
                continue;
            }
            track_taken[t] = true;
            peak_taken[p] = true;
            let pk = frame_peaks[p];
            let o = &mut open[t];
            o.ridge.points.push(RidgePoint { frame, cents: pk.cents, energy: pk.energy });
            o.sum_cents += pk.cents;
            o.missed = 0;
            o.occluded = 0;
        }

        // Unmatched tracks: occluded if a peak another track took lies within their reach of
        // where they expected to be, missing otherwise. Close what has run out of either.
        let mut kept = Vec::with_capacity(open.len());
        for (t, mut o) in open.drain(..).enumerate() {
            if !track_taken[t] {
                let (at, reach) = predicted[t];
                let under = frame_peaks
                    .iter()
                    .zip(&peak_taken)
                    .any(|(pk, &taken)| taken && (pk.cents - at).abs() <= reach);
                if under {
                    o.occluded += 1;
                } else {
                    o.missed += 1;
                }
            }
            if o.missed > cfg.max_gap_frames
                || o.missed + o.occluded > cfg.max_gap_frames.max(MAX_OCCLUDED_FRAMES)
            {
                closed.push(o.ridge);
            } else {
                kept.push(o);
            }
        }
        open = kept;

        for (p, pk) in frame_peaks.iter().enumerate() {
            if !peak_taken[p] {
                open.push(Open {
                    ridge: Ridge {
                        points: vec![RidgePoint { frame, cents: pk.cents, energy: pk.energy }],
                    },
                    sum_cents: pk.cents,
                    missed: 0,
                    occluded: 0,
                });
            }
        }
    }
    closed.extend(open.into_iter().map(|o| o.ridge));
    closed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PartialTrackingConfig {
        PartialTrackingConfig {
            max_jump_cents_per_frame: 50.0,
            max_gap_frames: 2,
            max_drift_cents: 0.0,
            min_duration_ms: 100.0,
            min_persistence: 0.5,
            min_relative_level_db: -60.0,
        }
    }

    fn frames(tracks: &[&dyn Fn(usize) -> Option<f64>], n: usize) -> Vec<Vec<Peak>> {
        (0..n)
            .map(|k| {
                let mut v: Vec<Peak> = tracks
                    .iter()
                    .filter_map(|f| f(k).map(|c| Peak { frame: k, cents: c, energy: 1.0 }))
                    .collect();
                v.sort_by(|a, b| a.cents.total_cmp(&b.cents));
                v
            })
            .collect()
    }

    #[test]
    fn a_steady_peak_is_one_ridge_and_a_short_gap_does_not_split_it() {
        let p = frames(&[&|k| (k != 10 && k != 11).then_some(0.0)], 40);
        let r = track(&p, &cfg());
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].lifetime_frames(), 40);
        assert!((r[0].persistence() - 38.0 / 40.0).abs() < 1e-12);
    }

    #[test]
    fn a_gap_longer_than_allowed_splits_the_ridge() {
        let p = frames(&[&|k| (!(10..13).contains(&k)).then_some(0.0)], 40);
        assert_eq!(track(&p, &cfg()).len(), 2);
    }

    #[test]
    fn a_jump_beyond_reach_starts_a_new_ridge() {
        let p = frames(&[&|k| Some(if k < 20 { 0.0 } else { 80.0 })], 40);
        assert_eq!(track(&p, &cfg()).len(), 2);
    }

    /// Two glides crossing in frequency, with the peaks merging into one at the crossing: each
    /// ridge keeps its own direction of travel through it.
    #[test]
    fn crossing_glides_keep_their_identities() {
        let up = |k: usize| Some(-300.0 + 20.0 * k as f64);
        let down = |k: usize| Some(300.0 - 20.0 * k as f64);
        // They meet at frame 15; near it the two are one peak.
        let p: Vec<Vec<Peak>> = frames(&[&up, &down], 31)
            .into_iter()
            .map(|mut v| {
                if v.len() == 2 && (v[1].cents - v[0].cents).abs() < 30.0 {
                    let c = 0.5 * (v[0].cents + v[1].cents);
                    v = vec![Peak { cents: c, ..v[0] }];
                }
                v
            })
            .collect();
        let r = track(&p, &cfg());
        let long: Vec<_> = r.iter().filter(|r| r.lifetime_frames() > 20).collect();
        assert_eq!(long.len(), 2, "{r:?}");
        for ridge in long {
            let at = |k| ridge.points.iter().find(|p| p.frame == k).unwrap().cents;
            // The direction before the crossing is the direction after it.
            let (before, after) = (at(5) - at(0), at(30) - at(25));
            assert!(before * after > 0.0, "a ridge turned back at the crossing: {ridge:?}");
        }
    }

    /// A track that has lost its peak for a few frames may only resume where it was. Its reach
    /// used to grow by `max_jump_cents_per_frame` per missed frame, so after two missed frames it
    /// could take a peak a semitone away: on a piano book that joined consecutive notes into 170
    /// glissandi, and simplification drew a straight line across each gap.
    #[test]
    fn a_track_does_not_resume_on_another_note_after_a_gap() {
        let p = frames(&[&|k| match k {
            0..20 => Some(0.0),
            20 | 21 => None,
            _ => Some(100.0),
        }], 40);
        let r = track(&p, &cfg());
        assert_eq!(r.len(), 2, "{r:?}");
    }

    /// A slow slide passes the per-frame limit every frame. The drift limit is what stops it.
    #[test]
    fn the_drift_limit_cuts_a_slow_slide_and_nothing_else() {
        let slide = |k: usize| Some(-2.0 * k as f64);
        let p = frames(&[&slide], 60);
        assert_eq!(track(&p, &cfg()).len(), 1, "no limit: one ridge");
        let limited = PartialTrackingConfig { max_drift_cents: 25.0, ..cfg() };
        let r = track(&p, &limited);
        assert!(r.len() >= 2, "{r:?}");
        for ridge in &r {
            let (lo, hi) = ridge.points.iter().fold((f64::MAX, f64::MIN), |(lo, hi), q| {
                (lo.min(q.cents), hi.max(q.cents))
            });
            assert!(hi - lo <= 2.0 * 25.0 + 1e-9, "ridge spans {} cents", hi - lo);
        }
        // A steady ridge with a few cents of jitter is untouched by the same limit.
        let steady = |k: usize| Some(if k.is_multiple_of(2) { 3.0 } else { -3.0 });
        assert_eq!(track(&frames(&[&steady], 60), &limited).len(), 1);
    }

    #[test]
    fn vibrato_is_one_ridge() {
        // ±80 cents at 5.5 Hz, sampled at 100 frames per second.
        let v = |k: usize| Some(80.0 * (std::f64::consts::TAU * 5.5 * k as f64 / 100.0).sin());
        let r = track(&frames(&[&v], 200), &cfg());
        assert_eq!(r.len(), 1);
    }
}
