//! What a ridge measures, and whether it is a partial (§12–§13).
//!
//! **Persistence** is matched frames over the lifetime. It is the main defence against atom-level
//! detail: a ridge that exists only now and then is not a persistent component, however loud.
//! **Significance** is `normalized_energy × persistence`, with both factors stored beside it so that
//! another combination can be tried later without re-running anything (§13).

use super::peaks::hz;
use super::tracking::Ridge;
use crate::config::PartialTrackingConfig;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RidgeMetrics {
    pub lifetime_frames: usize,
    pub duration_ms: f64,
    pub persistence: f64,
    pub mean_frequency_hz: f64,
    pub geometric_mean_frequency_hz: f64,
    pub frequency_std_cents: f64,
    /// Mean over matched frames of the square root of the frame energy.
    pub mean_amplitude: f64,
}

/// Measured over the matched frames only: an interpolated frame is not evidence.
pub fn measure(ridge: &Ridge, time_step_ms: f64) -> RidgeMetrics {
    let n = ridge.points.len() as f64;
    let lifetime = ridge.lifetime_frames();
    let mean_cents = ridge.points.iter().map(|p| p.cents).sum::<f64>() / n;
    let var = ridge.points.iter().map(|p| (p.cents - mean_cents).powi(2)).sum::<f64>() / n;
    RidgeMetrics {
        lifetime_frames: lifetime,
        duration_ms: lifetime as f64 * time_step_ms,
        persistence: ridge.persistence(),
        mean_frequency_hz: ridge.points.iter().map(|p| hz(p.cents)).sum::<f64>() / n,
        geometric_mean_frequency_hz: hz(mean_cents),
        frequency_std_cents: var.sqrt(),
        mean_amplitude: ridge.points.iter().map(|p| p.energy.sqrt()).sum::<f64>() / n,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Accept,
    /// Shorter than `min_duration_ms`.
    Short,
    /// Present in too few of its frames.
    Sparse,
}

pub fn judge(m: &RidgeMetrics, cfg: &PartialTrackingConfig) -> Verdict {
    if m.duration_ms < cfg.min_duration_ms {
        Verdict::Short
    } else if m.persistence < cfg.min_persistence {
        Verdict::Sparse
    } else {
        Verdict::Accept
    }
}

pub fn significance(normalized_energy: f64, persistence: f64) -> f64 {
    normalized_energy * persistence
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partial::tracking::RidgePoint;

    fn ridge(frames: &[usize]) -> Ridge {
        Ridge {
            points: frames.iter().map(|&frame| RidgePoint { frame, cents: 0.0, energy: 4.0 }).collect(),
        }
    }

    #[test]
    fn persistence_is_matched_frames_over_the_lifetime() {
        assert_eq!(ridge(&[0, 1, 2, 3]).persistence(), 1.0);
        assert_eq!(ridge(&[10, 12, 14, 16, 18]).persistence(), 5.0 / 9.0);
        assert_eq!(ridge(&[7]).persistence(), 1.0);
    }

    #[test]
    fn a_ridge_is_judged_short_before_sparse() {
        let cfg = PartialTrackingConfig {
            max_jump_cents_per_frame: 50.0,
            max_gap_frames: 2,
            max_drift_cents: 0.0,
            min_duration_ms: 100.0,
            min_persistence: 0.5,
            min_relative_level_db: -60.0,
        };
        let long: Vec<usize> = (0..20).collect();
        assert_eq!(judge(&measure(&ridge(&long), 10.0), &cfg), Verdict::Accept);
        assert_eq!(judge(&measure(&ridge(&[0, 1, 2]), 10.0), &cfg), Verdict::Short);
        let sparse: Vec<usize> = (0..10).map(|i| 3 * i).collect();
        assert_eq!(judge(&measure(&ridge(&sparse), 10.0), &cfg), Verdict::Sparse);
        let m = measure(&ridge(&long), 10.0);
        assert_eq!((m.duration_ms, m.mean_amplitude, m.frequency_std_cents), (200.0, 2.0, 0.0));
        assert!((m.mean_frequency_hz - 440.0).abs() < 1e-9);
    }
}
