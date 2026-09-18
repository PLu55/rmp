//! A slowly varying quantity as a few breakpoints rather than a value per frame.
//!
//! Shared by partials (frequency, level) and, later, stems (level, fundamental). Between two points
//! the value is linear in whatever unit the trajectory is stored in. A frequency trajectory is
//! stored in Hz but simplified in cents, so reading between its points is linear in Hz where the
//! simplification assumed linear in cents; for two points 100 cents apart the two differ by under
//! one cent at the midpoint, well inside the simplification tolerance.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    /// Absolute sample position on the source timeline.
    pub time_samples: u64,
    pub value: f64,
}

/// Breakpoints in strictly increasing time order.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    pub points: Vec<TrajectoryPoint>,
}

impl Trajectory {
    pub fn start(&self) -> Option<u64> {
        self.points.first().map(|p| p.time_samples)
    }

    pub fn end(&self) -> Option<u64> {
        self.points.last().map(|p| p.time_samples)
    }

    /// The value at `t`, interpolated linearly, or `None` outside the trajectory's span.
    pub fn value_at(&self, t: u64) -> Option<f64> {
        let pts = &self.points;
        let (first, last) = (pts.first()?, pts.last()?);
        if t < first.time_samples || t > last.time_samples {
            return None;
        }
        // The first point at or after `t`.
        let i = pts.partition_point(|p| p.time_samples < t);
        let b = pts[i];
        if b.time_samples == t || i == 0 {
            return Some(b.value);
        }
        let a = pts[i - 1];
        let x = (t - a.time_samples) as f64 / (b.time_samples - a.time_samples) as f64;
        Some(a.value + x * (b.value - a.value))
    }

    /// The value at `start + k * step` for `k in 0..n`, `None` outside the span.
    ///
    /// One pass over the breakpoints rather than a search per sample, which is what morphology will
    /// want when it puts every partial on one common grid.
    pub fn resample(&self, start: u64, step: u64, n: usize) -> Vec<Option<f64>> {
        let pts = &self.points;
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        for k in 0..n {
            let t = start + k as u64 * step.max(1);
            match (pts.first(), pts.last()) {
                (Some(f), Some(l)) if t >= f.time_samples && t <= l.time_samples => {
                    while pts[i].time_samples < t {
                        i += 1;
                    }
                    let b = pts[i];
                    if b.time_samples == t || i == 0 {
                        out.push(Some(b.value));
                    } else {
                        let a = pts[i - 1];
                        let x = (t - a.time_samples) as f64 / (b.time_samples - a.time_samples) as f64;
                        out.push(Some(a.value + x * (b.value - a.value)));
                    }
                }
                _ => out.push(None),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(pts: &[(u64, f64)]) -> Trajectory {
        Trajectory {
            points: pts.iter().map(|&(t, v)| TrajectoryPoint { time_samples: t, value: v }).collect(),
        }
    }

    #[test]
    fn values_are_linear_between_points_and_absent_outside() {
        let t = tr(&[(100, 0.0), (200, 10.0), (400, 10.0)]);
        assert_eq!(t.value_at(99), None);
        assert_eq!(t.value_at(100), Some(0.0));
        assert_eq!(t.value_at(150), Some(5.0));
        assert_eq!(t.value_at(300), Some(10.0));
        assert_eq!(t.value_at(400), Some(10.0));
        assert_eq!(t.value_at(401), None);
    }

    #[test]
    fn resampling_agrees_with_pointwise_reads() {
        let t = tr(&[(100, 0.0), (200, 10.0), (400, -3.0)]);
        let got = t.resample(40, 30, 16);
        for (k, v) in got.iter().enumerate() {
            assert_eq!(*v, t.value_at(40 + 30 * k as u64), "k={k}");
        }
    }
}
