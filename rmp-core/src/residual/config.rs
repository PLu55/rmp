//! The resolved residual-analysis settings.
//!
//! The crate already separates the hand-edited document from the runtime value — `BlockSettings`
//! becomes `BlockConfig`, `RefineSettings` becomes `RefineConfig` — and this is the same split. The
//! document lives in [`crate::config`] and speaks milliseconds; this speaks samples and seconds,
//! and by the time one exists it has been validated against the sample rate it will run at.

use crate::residual::book::{
    ErbFilterKind, ErbNormalization, ErbSpacing, ResidualStorage,
};
use crate::residual::erb::center_freqs;
use crate::residual::error::ResidualAnalysisError;
use crate::residual::power::ResidualPowerConfig;
use crate::residual::{MAX_ORDER, MIN_BANDS, MIN_ORDER};

/// How close to Nyquist a band centre may sit.
///
/// A gammatone centred right at Nyquist has half its skirt folded back on top of itself, so the
/// band would report the fold rather than the signal. The guard is small because the limit is a
/// property of the design, not a taste: `max_freq_hz` above it is an error, never a silent clamp.
pub const NYQUIST_GUARD: f64 = 0.98;

#[derive(Clone, Debug, PartialEq)]
pub struct ErbBankConfig {
    pub bands: usize,
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    pub spacing: ErbSpacing,
    pub filter_kind: ErbFilterKind,
    pub filter_order: usize,
    pub normalization: ErbNormalization,
}

impl Default for ErbBankConfig {
    fn default() -> Self {
        Self {
            bands: 48,
            min_freq_hz: 50.0,
            max_freq_hz: 20_000.0,
            spacing: ErbSpacing::ErbRate,
            filter_kind: ErbFilterKind::Gammatone,
            filter_order: 4,
            normalization: ErbNormalization::UnitNoisePower,
        }
    }
}

impl ErbBankConfig {
    pub fn center_freqs(&self) -> Result<Vec<f64>, ResidualAnalysisError> {
        center_freqs(self.bands, self.min_freq_hz, self.max_freq_hz)
    }

    /// Checks that do not need a sample rate.
    pub fn validate(&self) -> Result<(), ResidualAnalysisError> {
        if self.bands < MIN_BANDS {
            return Err(ResidualAnalysisError::InvalidBandCount(self.bands));
        }
        if !(MIN_ORDER..=MAX_ORDER).contains(&self.filter_order) {
            return Err(ResidualAnalysisError::UnsupportedFilterOrder(self.filter_order));
        }
        // Also rejects non-finite bounds and an inverted range.
        self.center_freqs().map(|_| ())
    }
}

/// Residual analysis, fully resolved for one sample rate.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualAnalysisConfig {
    pub enabled: bool,
    /// The book's update period, in samples. Milliseconds are converted here and nowhere else.
    pub update_samples: usize,
    pub erb: ErbBankConfig,
    pub power: ResidualPowerConfig,
    pub storage: ResidualStorage,
}

impl Default for ResidualAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            update_samples: 48,
            erb: ErbBankConfig::default(),
            power: ResidualPowerConfig::default(),
            storage: ResidualStorage::F32LinearPower,
        }
    }
}

impl ResidualAnalysisConfig {
    /// Everything §16 asks for, checked once, before any audio is read.
    pub fn validate(&self, sample_rate: f64) -> Result<(), ResidualAnalysisError> {
        if !(sample_rate.is_finite() && sample_rate > 0.0) {
            return Err(ResidualAnalysisError::InvalidSampleRate(sample_rate));
        }
        if self.update_samples < 1 {
            return Err(ResidualAnalysisError::InvalidUpdateInterval {
                update_ms: 0.0,
                sample_rate,
            });
        }
        self.erb.validate()?;
        self.power.validate()?;

        let limit = NYQUIST_GUARD * sample_rate / 2.0;
        if self.erb.max_freq_hz >= limit {
            return Err(ResidualAnalysisError::FrequencyAboveNyquist {
                freq_hz: self.erb.max_freq_hz,
                nyquist_hz: limit,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_bank_is_valid_at_the_usual_rates() {
        let cfg = ResidualAnalysisConfig::default();
        for sr in [44_100.0, 48_000.0, 96_000.0] {
            assert!(cfg.validate(sr).is_ok(), "{sr}");
        }
        // At 32 kHz the default 20 kHz top is past the guarded limit, and that is an error rather
        // than a clamp: §32 is explicit that an explicit user value is not silently moved.
        let err = cfg.validate(32_000.0).unwrap_err().to_string();
        assert!(err.contains("max_freq_hz") && err.contains("15680"), "{err}");
    }

    #[test]
    fn validation_covers_every_field_the_spec_lists() {
        let base = ResidualAnalysisConfig::default();

        let too_few = ResidualAnalysisConfig {
            erb: ErbBankConfig { bands: 3, ..base.erb.clone() },
            ..base.clone()
        };
        assert!(too_few.validate(48_000.0).unwrap_err().to_string().contains("bands"));

        let bad_order = ResidualAnalysisConfig {
            erb: ErbBankConfig { filter_order: 0, ..base.erb.clone() },
            ..base.clone()
        };
        assert!(bad_order.validate(48_000.0).unwrap_err().to_string().contains("order"));

        let inverted = ResidualAnalysisConfig {
            erb: ErbBankConfig { min_freq_hz: 9000.0, max_freq_hz: 500.0, ..base.erb.clone() },
            ..base.clone()
        };
        assert!(inverted.validate(48_000.0).is_err());

        let no_update = ResidualAnalysisConfig { update_samples: 0, ..base.clone() };
        assert!(no_update.validate(48_000.0).is_err());

        assert!(base.validate(0.0).is_err());
    }

    #[test]
    fn centres_span_the_configured_range() {
        let cfg = ErbBankConfig::default();
        let f = cfg.center_freqs().unwrap();
        assert_eq!(f.len(), 48);
        assert_eq!((f[0], f[47]), (50.0, 20_000.0));
    }
}
