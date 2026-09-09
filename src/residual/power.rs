//! The band-power detector.
//!
//! One pole per band: `P[n] = a*P[n-1] + (1-a)*y^2[n]`, with `a = exp(-1/(tau*fs))`. That is the
//! whole of the temporal model, deliberately. The residual carries rhythm and transients that the
//! atom book does not, and the point of the design is to keep them: no second smoothing stage is
//! applied anywhere, and none should be added without saying so in the settings document.
//!
//! `tau` is either the same everywhere or tied to each band's own bandwidth. Bandwidth-relative is
//! the default because a 24.7 Hz band cannot resolve a 0.5 ms event in the first place — its own
//! ringing is 40 ms long — while a 3 kHz band at the top of the range can, and a single `tau`
//! either over-smooths the top or leaves the bottom reading its own envelope ripple.

use serde::{Deserialize, Serialize};

use crate::residual::error::ResidualAnalysisError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualPowerTimeMode {
    /// Every band uses `tau_ms`.
    Fixed,
    /// `tau_b = clamp(tau_scale / bandwidth_b, tau_min, tau_max)`.
    #[default]
    BandwidthRelative,
}

impl ResidualPowerTimeMode {
    /// The name the settings document and the book both use, so a summary line can be pasted
    /// straight back into a `.toml`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::BandwidthRelative => "bandwidth_relative",
        }
    }
}

impl std::fmt::Display for ResidualPowerTimeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The resolved power-detector settings: seconds, never milliseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualPowerConfig {
    pub mode: ResidualPowerTimeMode,
    /// Used by [`ResidualPowerTimeMode::Fixed`].
    pub fixed_tau_seconds: f64,
    /// Used by [`ResidualPowerTimeMode::BandwidthRelative`]. Dimensionless.
    pub bandwidth_tau_scale: f64,
    pub tau_min_seconds: f64,
    pub tau_max_seconds: f64,
}

impl Default for ResidualPowerConfig {
    fn default() -> Self {
        Self {
            mode: ResidualPowerTimeMode::BandwidthRelative,
            fixed_tau_seconds: 2e-3,
            bandwidth_tau_scale: 1.0,
            tau_min_seconds: 0.5e-3,
            tau_max_seconds: 10e-3,
        }
    }
}

impl ResidualPowerConfig {
    /// Reject time constants that would produce no detector at all.
    ///
    /// Both modes' constants are checked whichever mode is selected: the unused pair still appears
    /// in the settings document, and a negative number sitting there is a typo whether or not this
    /// run reads it.
    pub fn validate(&self) -> Result<(), ResidualAnalysisError> {
        let positive = |name: &str, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(ResidualAnalysisError::InvalidPowerTimeConstant(format!(
                    "{name} must be a positive number of seconds, got {v}"
                )))
            }
        };
        positive("tau_ms", self.fixed_tau_seconds)?;
        positive("tau_min_ms", self.tau_min_seconds)?;
        positive("tau_max_ms", self.tau_max_seconds)?;
        positive("tau_scale", self.bandwidth_tau_scale)?;
        if self.tau_min_seconds > self.tau_max_seconds {
            return Err(ResidualAnalysisError::InvalidPowerTimeConstant(format!(
                "tau_min_ms ({}) must not exceed tau_max_ms ({})",
                self.tau_min_seconds * 1e3,
                self.tau_max_seconds * 1e3
            )));
        }
        Ok(())
    }

    /// The time constant each band ends up with, in seconds.
    pub fn taus(&self, bandwidths_hz: &[f64]) -> Vec<f64> {
        bandwidths_hz
            .iter()
            .map(|&b| match self.mode {
                ResidualPowerTimeMode::Fixed => self.fixed_tau_seconds,
                ResidualPowerTimeMode::BandwidthRelative => (self.bandwidth_tau_scale / b)
                    .clamp(self.tau_min_seconds, self.tau_max_seconds),
            })
            .collect()
    }
}

/// `a_b = exp(-1/(tau_b * fs))`, the one-pole coefficient each band's detector runs with.
///
/// Computed once, before any sample is touched: the spec is explicit that milliseconds are not to
/// be converted inside the loop, and the same goes for the exponential.
pub fn detector_coeffs(sample_rate: f64, taus_seconds: &[f64]) -> Vec<f64> {
    taus_seconds
        .iter()
        .map(|&tau| (-1.0 / (tau * sample_rate)).exp())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(a: f64, q: &[f64]) -> Vec<f64> {
        let mut p = 0.0;
        q.iter()
            .map(|&qi| {
                p = a * p + (1.0 - a) * qi;
                p
            })
            .collect()
    }

    /// §29.4: constant squared input converges to that value, and reaches 1 - 1/e of it after
    /// exactly `tau` seconds — which is what makes `tau` mean something to a user.
    #[test]
    fn constant_input_converges_to_it() {
        let (fs, tau) = (48_000.0, 2e-3);
        let a = detector_coeffs(fs, &[tau])[0];
        let n = (tau * fs) as usize;
        let p = run(a, &vec![4.0; 20 * n]);

        assert!((p[20 * n - 1] - 4.0).abs() < 1e-7, "{}", p[20 * n - 1]);
        let at_tau = p[n - 1] / 4.0;
        assert!((at_tau - (1.0 - (-1.0f64).exp())).abs() < 1e-3, "{at_tau}");
    }

    /// §29.4: after the input stops, the state decays as `exp(-t/tau)`.
    #[test]
    fn a_burst_decays_at_the_time_constant() {
        let (fs, tau) = (48_000.0, 1e-3);
        let a = detector_coeffs(fs, &[tau])[0];
        let n = (tau * fs) as usize;
        let mut q = vec![1.0; 50 * n];
        q.extend(std::iter::repeat_n(0.0, 5 * n));
        let p = run(a, &q);

        let peak = p[50 * n - 1];
        for k in 1..=4 {
            let got = p[50 * n - 1 + k * n] / peak;
            let want = (-(k as f64)).exp();
            assert!((got - want).abs() < 1e-3 * want.max(1e-3), "at {k} tau: {got} vs {want}");
        }
    }

    /// Zero in, exactly zero out — the detector adds nothing of its own (§29.6 in miniature).
    #[test]
    fn silence_stays_exactly_zero() {
        let a = detector_coeffs(48_000.0, &[2e-3])[0];
        assert!(run(a, &vec![0.0; 1000]).iter().all(|&p| p == 0.0));
    }

    #[test]
    fn bandwidth_relative_taus_hit_both_rails() {
        let cfg = ResidualPowerConfig::default();
        // 24.7 Hz -> 40 ms, clamped to 10; 3000 Hz -> 0.33 ms, clamped to 0.5; 200 Hz -> 5 ms.
        let taus = cfg.taus(&[24.7, 200.0, 3000.0]);
        assert_eq!(taus[0], cfg.tau_max_seconds);
        assert!((taus[1] - 5e-3).abs() < 1e-9, "{}", taus[1]);
        assert_eq!(taus[2], cfg.tau_min_seconds);

        let fixed = ResidualPowerConfig {
            mode: ResidualPowerTimeMode::Fixed,
            ..cfg
        };
        assert_eq!(fixed.taus(&[24.7, 200.0, 3000.0]), vec![2e-3; 3]);
    }

    /// The summary line and the settings document spell the mode the same way.
    #[test]
    fn the_mode_prints_as_the_document_spells_it() {
        for mode in [ResidualPowerTimeMode::Fixed, ResidualPowerTimeMode::BandwidthRelative] {
            let json = serde_json::to_string(&mode).unwrap();
            assert_eq!(json, format!("\"{mode}\""));
        }
    }

    #[test]
    fn validation_names_the_field() {
        let bad = ResidualPowerConfig {
            tau_min_seconds: 20e-3,
            ..Default::default()
        };
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("tau_min_ms") && err.contains("tau_max_ms"), "{err}");

        for cfg in [
            ResidualPowerConfig { fixed_tau_seconds: 0.0, ..Default::default() },
            ResidualPowerConfig { bandwidth_tau_scale: -1.0, ..Default::default() },
            ResidualPowerConfig { tau_max_seconds: f64::NAN, ..Default::default() },
        ] {
            assert!(cfg.validate().is_err());
        }
        assert!(ResidualPowerConfig::default().validate().is_ok());
    }
}
