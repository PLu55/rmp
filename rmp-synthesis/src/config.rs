//! Render settings, and the one place milliseconds become seconds.
//!
//! [`GainSmoothingConfig`] deliberately mirrors [`rmp_core::residual::power::ResidualPowerConfig`]:
//! same two modes, same `clamp(scale / bandwidth, min, max)` rule, same "seconds, never
//! milliseconds" discipline inside the struct. The two are the analysis and synthesis ends of one
//! idea — how fast a band's level is allowed to move — and there is no reason for them to be
//! spelled differently.

use crate::error::RenderError;

/// How the gain smoothing time constant is chosen per band (§14).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GainSmoothingMode {
    /// Every band uses `fixed_seconds`.
    #[default]
    Fixed,
    /// `tau_b = clamp(scale / bandwidth_b, min, max)`.
    BandwidthRelative,
}

impl GainSmoothingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::BandwidthRelative => "bandwidth_relative",
        }
    }
}

impl std::fmt::Display for GainSmoothingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The resolved gain-smoothing settings: seconds, never milliseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GainSmoothingConfig {
    pub mode: GainSmoothingMode,
    pub fixed_seconds: f64,
    /// Dimensionless, used by [`GainSmoothingMode::BandwidthRelative`].
    pub scale: f64,
    pub min_seconds: f64,
    pub max_seconds: f64,
}

impl Default for GainSmoothingConfig {
    /// §13.1's recommended 1.0 ms, fixed across bands.
    ///
    /// Short on purpose. The analysis detector has already smoothed the power with its own `tau`;
    /// this second pole exists to stop a step between book frames becoming a click, not to smooth
    /// the envelope again. Anything long enough to matter here would erase the transients the
    /// residual analysis went to some trouble to keep.
    fn default() -> Self {
        Self {
            mode: GainSmoothingMode::Fixed,
            fixed_seconds: 1e-3,
            scale: 1.0,
            min_seconds: 0.2e-3,
            max_seconds: 10e-3,
        }
    }
}

impl GainSmoothingConfig {
    /// Both modes' constants are checked whichever mode is selected — a negative number sitting in
    /// the unused pair is a typo whether or not this run reads it.
    pub fn validate(&self) -> Result<(), RenderError> {
        let positive = |name: &str, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(RenderError::InvalidConfig(format!(
                    "{name} must be a positive number of milliseconds, got {}",
                    v * 1e3
                )))
            }
        };
        positive("--gain-smoothing-ms", self.fixed_seconds)?;
        positive("gain smoothing min_ms", self.min_seconds)?;
        positive("gain smoothing max_ms", self.max_seconds)?;
        if !(self.scale.is_finite() && self.scale > 0.0) {
            return Err(RenderError::InvalidConfig(format!(
                "gain smoothing scale must be positive, got {}",
                self.scale
            )));
        }
        if self.min_seconds > self.max_seconds {
            return Err(RenderError::InvalidConfig(format!(
                "gain smoothing min_ms ({}) must not exceed max_ms ({})",
                self.min_seconds * 1e3,
                self.max_seconds * 1e3
            )));
        }
        Ok(())
    }

    /// The time constant each band ends up with, in seconds.
    pub fn taus(&self, bandwidths_hz: &[f64]) -> Vec<f64> {
        bandwidths_hz
            .iter()
            .map(|&b| match self.mode {
                GainSmoothingMode::Fixed => self.fixed_seconds,
                GainSmoothingMode::BandwidthRelative => {
                    (self.scale / b).clamp(self.min_seconds, self.max_seconds)
                }
            })
            .collect()
    }
}

/// Output sample format (§20).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputEncoding {
    #[default]
    Float32,
    Pcm24,
}

impl OutputEncoding {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Pcm24 => "pcm24",
        }
    }
}

impl std::fmt::Display for OutputEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What to do about samples past full scale (§22).
///
/// Never normalise. A residual reconstruction whose level was quietly adjusted cannot be compared
/// with the analysis that produced it, which is the main thing anyone would want to do with it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClippingPolicy {
    /// Count them, report them, write them as they are. The float default.
    #[default]
    Report,
    /// Refuse to write the file.
    Error,
    /// Hard-clip to `[-1, 1]`.
    Clip,
}

impl ClippingPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Report => "report",
            Self::Error => "error",
            Self::Clip => "clip",
        }
    }
}

impl std::fmt::Display for ClippingPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Everything that shapes a render (§23).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderConfig {
    pub seed: u64,
    pub gain_smoothing: GainSmoothingConfig,
    pub output_gain_db: f64,
    pub output_encoding: OutputEncoding,
    pub clipping: ClippingPolicy,
    /// Output sample 0 is source sample 0, so a book analysed from an offset renders leading
    /// silence (§16). False trims to the analysed excerpt.
    pub preserve_timeline: bool,
}

impl Default for RenderConfig {
    /// §47's suggested initial defaults.
    fn default() -> Self {
        Self {
            seed: 1,
            gain_smoothing: GainSmoothingConfig::default(),
            output_gain_db: 0.0,
            output_encoding: OutputEncoding::Float32,
            clipping: ClippingPolicy::Report,
            preserve_timeline: true,
        }
    }
}

impl RenderConfig {
    pub fn validate(&self) -> Result<(), RenderError> {
        self.gain_smoothing.validate()?;
        if !self.output_gain_db.is_finite() {
            return Err(RenderError::InvalidConfig(format!(
                "--gain-db must be a finite number of decibels, got {}",
                self.output_gain_db
            )));
        }
        Ok(())
    }

    /// The linear output gain, `10^(dB/20)` (§21).
    pub fn output_gain(&self) -> f64 {
        10f64.powf(self.output_gain_db / 20.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §47: the documented defaults are the ones the code actually starts from.
    #[test]
    fn the_defaults_are_the_documented_ones() {
        let c = RenderConfig::default();
        assert_eq!(c.seed, 1);
        assert_eq!(c.gain_smoothing.fixed_seconds, 1e-3);
        assert_eq!(c.gain_smoothing.mode, GainSmoothingMode::Fixed);
        assert_eq!(c.output_gain_db, 0.0);
        assert_eq!(c.output_encoding, OutputEncoding::Float32);
        assert_eq!(c.clipping, ClippingPolicy::Report);
        assert!(c.preserve_timeline);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn zero_db_is_unity_and_six_db_is_two() {
        assert!((RenderConfig::default().output_gain() - 1.0).abs() < 1e-15);
        let up = RenderConfig {
            output_gain_db: 20.0,
            ..Default::default()
        };
        assert!((up.output_gain() - 10.0).abs() < 1e-12);
    }

    /// The same clamp rule the analysis detector uses, so the two ends agree about what
    /// bandwidth-relative means.
    #[test]
    fn bandwidth_relative_taus_hit_both_rails() {
        let cfg = GainSmoothingConfig {
            mode: GainSmoothingMode::BandwidthRelative,
            ..Default::default()
        };
        let taus = cfg.taus(&[24.7, 200.0, 8000.0]);
        assert_eq!(taus[0], cfg.max_seconds);
        assert!((taus[1] - 5e-3).abs() < 1e-12, "{}", taus[1]);
        assert_eq!(taus[2], cfg.min_seconds);

        let fixed = GainSmoothingConfig::default();
        assert_eq!(fixed.taus(&[24.7, 200.0, 8000.0]), vec![1e-3; 3]);
    }

    #[test]
    fn validation_names_the_flag() {
        let bad = RenderConfig {
            gain_smoothing: GainSmoothingConfig {
                fixed_seconds: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("--gain-smoothing-ms"), "{err}");

        let bad = RenderConfig {
            output_gain_db: f64::NAN,
            ..Default::default()
        };
        assert!(bad.validate().unwrap_err().to_string().contains("--gain-db"));

        let inverted = GainSmoothingConfig {
            min_seconds: 20e-3,
            ..Default::default()
        };
        assert!(inverted.validate().is_err());
    }

    /// Diagnostics spell an enum the way a settings document would, as the book enums do.
    #[test]
    fn enums_print_as_a_document_spells_them() {
        assert_eq!(OutputEncoding::Float32.to_string(), "float32");
        assert_eq!(OutputEncoding::Pcm24.to_string(), "pcm24");
        assert_eq!(ClippingPolicy::Report.to_string(), "report");
        assert_eq!(GainSmoothingMode::BandwidthRelative.to_string(), "bandwidth_relative");
    }
}
