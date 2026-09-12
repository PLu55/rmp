//! The ERB frequency scale, and where the bands sit on it.
//!
//! Glasberg & Moore's equivalent-rectangular-bandwidth formulae, in the form the spec asks for.
//! Everything here is `f64`: the band centres feed pole placement, and a centre that is a few
//! hundredths of a hertz out shifts a narrow low band's pole by more than its own bandwidth.

use crate::residual::error::ResidualAnalysisError;

/// Equivalent rectangular bandwidth at `f`, in Hz.
///
/// `24.7 * (1 + 4.37 * f / 1000)` — 24.7 Hz at DC, widening to about 3.4 kHz at 20 kHz.
pub fn erb_bandwidth_hz(f: f64) -> f64 {
    24.7 * (1.0 + 4.37 * f / 1000.0)
}

/// ERB-rate (the "number of ERBs below `f`"), the scale band centres are uniform on.
pub fn erb_rate(f: f64) -> f64 {
    21.4 * (1.0 + 0.00437 * f).log10()
}

/// Inverse of [`erb_rate`].
pub fn erb_rate_to_hz(e: f64) -> f64 {
    (10f64.powf(e / 21.4) - 1.0) / 0.00437
}

/// Band centres in Hz, uniformly spaced on ERB-rate from `min_hz` to `max_hz` inclusive.
///
/// The endpoints are placed exactly rather than at half-band insets: the range in the settings
/// document is the range the user asked for, and §32 of the spec is explicit that explicit user
/// values are not silently modified.
pub fn center_freqs(bands: usize, min_hz: f64, max_hz: f64) -> Result<Vec<f64>, ResidualAnalysisError> {
    if bands == 0 {
        return Err(ResidualAnalysisError::InvalidBandCount(bands));
    }
    if !(min_hz.is_finite() && max_hz.is_finite() && min_hz > 0.0 && max_hz > min_hz) {
        return Err(ResidualAnalysisError::InvalidFrequencyRange { min_hz, max_hz });
    }
    if bands == 1 {
        return Ok(vec![min_hz]);
    }

    let (e_lo, e_hi) = (erb_rate(min_hz), erb_rate(max_hz));
    let span = e_hi - e_lo;
    let last = bands - 1;
    Ok((0..bands)
        .map(|b| {
            // The endpoints are returned as given rather than round-tripped through the ERB
            // conversions, so `first == min_hz` and `last == max_hz` hold to the bit.
            match b {
                0 => min_hz,
                b if b == last => max_hz,
                _ => erb_rate_to_hz(e_lo + span * b as f64 / last as f64),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §29.1: the round trip is exact, and the two formulae agree on the reference values in the
    /// literature.
    #[test]
    fn erb_rate_and_its_inverse_agree() {
        for &f in &[20.0, 50.0, 440.0, 1000.0, 5000.0, 20_000.0] {
            let back = erb_rate_to_hz(erb_rate(f));
            assert!((back - f).abs() < 1e-9 * f, "{f} -> {back}");
        }
        // A 1 kHz ERB is about 132 Hz, and 1 kHz sits about 15.6 ERBs up the scale.
        assert!((erb_bandwidth_hz(1000.0) - 132.639).abs() < 1e-3);
        assert!((erb_rate(1000.0) - 15.622).abs() < 1e-3);
        assert!((erb_bandwidth_hz(0.0) - 24.7).abs() < 1e-12);
        assert_eq!(erb_rate(0.0), 0.0);
    }

    /// §29.1: strictly increasing, endpoints exact, uniform on ERB-rate.
    #[test]
    fn band_centres_are_uniform_on_erb_rate() {
        let f = center_freqs(48, 50.0, 20_000.0).unwrap();
        assert_eq!(f.len(), 48);
        assert_eq!(f[0], 50.0);
        assert_eq!(f[47], 20_000.0);
        assert!(f.windows(2).all(|w| w[1] > w[0]), "not strictly increasing");

        let step = (erb_rate(20_000.0) - erb_rate(50.0)) / 47.0;
        for (b, &fc) in f.iter().enumerate() {
            let want = erb_rate(50.0) + step * b as f64;
            assert!((erb_rate(fc) - want).abs() < 1e-9, "band {b}: {fc} Hz");
        }
    }

    #[test]
    fn degenerate_and_invalid_ranges() {
        assert_eq!(center_freqs(1, 440.0, 20_000.0).unwrap(), vec![440.0]);
        assert!(center_freqs(0, 50.0, 20_000.0).is_err());
        assert!(center_freqs(8, 0.0, 20_000.0).is_err());
        assert!(center_freqs(8, -1.0, 20_000.0).is_err());
        assert!(center_freqs(8, 20_000.0, 50.0).is_err());
        assert!(center_freqs(8, 50.0, f64::NAN).is_err());
    }
}

