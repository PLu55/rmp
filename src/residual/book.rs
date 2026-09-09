//! The residual book: what the ERB analysis produces, and everything needed to read it back.
//!
//! A residual book is regularly sampled stochastic-control data, which is why it is a section of
//! its own rather than something interleaved with the atom list. Atoms are sparse deterministic
//! events with their own onsets; residual frames are a matrix on a fixed grid, and the two have
//! nothing useful to say about each other's ordering.
//!
//! The matrix is flat — `power[frame * band_count + band]` — because that is both the serialisation
//! order and the order a synthesis bank reads it in: one frame's worth of gains, contiguous.

use serde::{Deserialize, Serialize};

use crate::residual::power::ResidualPowerTimeMode;

/// Bumped when the meaning of an existing field changes. Readers reject anything higher.
pub const RESIDUAL_BOOK_VERSION: u32 = 1;

/// How band centres are laid out. One value today; the enum exists so a book never has to be
/// guessed at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErbSpacing {
    #[default]
    ErbRate,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErbFilterKind {
    #[default]
    Gammatone,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErbNormalization {
    /// Unit-variance white noise leaves each band with unit variance.
    #[default]
    UnitNoisePower,
}

/// `Display` for a book enum spells it exactly as the settings document does, so a diagnostic line
/// can be pasted back into a `.toml` without translation.
macro_rules! document_spelling {
    ($t:ty { $($variant:ident => $name:literal),* $(,)? }) => {
        impl $t {
            pub fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $name),* }
            }
        }
        impl std::fmt::Display for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

document_spelling!(ErbSpacing { ErbRate => "erb_rate" });
document_spelling!(ErbFilterKind { Gammatone => "gammatone" });
document_spelling!(ErbNormalization { UnitNoisePower => "unit_noise_power" });

/// How the stored numbers encode power. Linear `f32` today; the enum is where a later quantised
/// encoding announces itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualStorage {
    #[default]
    F32LinearPower,
}

document_spelling!(ResidualStorage { F32LinearPower => "f32_linear_power" });

/// The power detector's settings as they were actually resolved, per band.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualPowerDescriptor {
    pub mode: ResidualPowerTimeMode,
    /// One time constant per band, in seconds. Stored rather than recomputed: in
    /// bandwidth-relative mode it depends on the ERB formula and on two clamps, and a synthesiser
    /// should not have to reproduce all three to know how fast a band was tracked.
    pub tau_seconds: Vec<f64>,
}

/// Everything a synthesis bank needs to know what the power numbers mean.
///
/// Centres, bandwidths and gains are stored explicitly even though they are all recomputable. The
/// book has to survive a future correction to the ERB formula or a change of filter design without
/// silently changing what its own numbers mean.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualErbBankDescriptor {
    pub band_count: u32,
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    pub filter_order: u32,
    pub spacing: ErbSpacing,
    pub filter_kind: ErbFilterKind,
    pub normalization: ErbNormalization,
    pub center_freq_hz: Vec<f64>,
    pub bandwidth_hz: Vec<f64>,
    /// The measured unit-noise-power gain each band was run with.
    pub normalization_gain: Vec<f64>,
    pub power_detector: ResidualPowerDescriptor,
}

/// A fixed-rate map of residual power over ERB bands.
///
/// Frame `k` is the state of every band's causal power detector immediately **after** processing
/// the sample at `start_sample + k * update_samples`. No timestamps are stored — the grid is exact
/// by construction, and a per-frame timestamp would be a second definition of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualBook {
    pub version: u32,
    pub sample_rate: f64,
    /// Where the analysed residual began, in samples from the file's origin. Carries `--start`
    /// through, so a book knows which excerpt it describes.
    pub start_sample: u64,
    /// Length of the analysed residual in samples.
    pub source_samples: u64,
    pub update_samples: u32,
    pub band_count: u32,
    pub frame_count: u64,
    pub storage: ResidualStorage,
    /// `frame * band_count + band`.
    pub power: Vec<f32>,
    pub bank: ResidualErbBankDescriptor,
}

impl ResidualBook {
    /// One frame's bands, in band order.
    pub fn frame(&self, k: usize) -> &[f32] {
        let b = self.band_count as usize;
        &self.power[k * b..(k + 1) * b]
    }

    /// Sample offset of frame `k`, from the file's origin.
    pub fn frame_sample(&self, k: usize) -> u64 {
        self.start_sample + k as u64 * self.update_samples as u64
    }

    /// Summed power across all bands at each frame — the total the bank saw.
    ///
    /// Not the residual's energy: the bands overlap, so this over-counts, and the detector is a
    /// running average rather than a sum. It is a level trace, useful for checking that the
    /// analysis tracks the residual in time (§28), and it is not an energy partition.
    pub fn total_power(&self) -> Vec<f64> {
        (0..self.frame_count as usize)
            .map(|k| self.frame(k).iter().map(|&p| p as f64).sum())
            .collect()
    }

    /// Reject a book this build cannot interpret, and catch a truncated matrix.
    pub fn validate(&self) -> Result<(), String> {
        if self.version > RESIDUAL_BOOK_VERSION {
            return Err(format!(
                "residual book version {} is newer than this build understands ({})",
                self.version, RESIDUAL_BOOK_VERSION
            ));
        }
        let want = self.frame_count as usize * self.band_count as usize;
        if self.power.len() != want {
            return Err(format!(
                "residual book holds {} power values, expected {} ({} frames x {} bands)",
                self.power.len(),
                want,
                self.frame_count,
                self.band_count
            ));
        }
        let b = self.band_count as usize;
        for (name, len) in [
            ("center_freq_hz", self.bank.center_freq_hz.len()),
            ("bandwidth_hz", self.bank.bandwidth_hz.len()),
            ("normalization_gain", self.bank.normalization_gain.len()),
            ("power_detector.tau_seconds", self.bank.power_detector.tau_seconds.len()),
        ] {
            if len != b {
                return Err(format!(
                    "residual book bank.{name} has {len} entries, expected {b}"
                ));
            }
        }
        if self.bank.band_count as usize != b {
            return Err(format!(
                "residual book bank.band_count ({}) disagrees with band_count ({b})",
                self.bank.band_count
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn a_book() -> ResidualBook {
        ResidualBook {
            version: RESIDUAL_BOOK_VERSION,
            sample_rate: 48_000.0,
            start_sample: 120,
            source_samples: 96,
            update_samples: 48,
            band_count: 3,
            frame_count: 2,
            storage: ResidualStorage::F32LinearPower,
            power: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            bank: ResidualErbBankDescriptor {
                band_count: 3,
                min_freq_hz: 50.0,
                max_freq_hz: 16_000.0,
                filter_order: 4,
                spacing: ErbSpacing::ErbRate,
                filter_kind: ErbFilterKind::Gammatone,
                normalization: ErbNormalization::UnitNoisePower,
                center_freq_hz: vec![50.0, 1000.0, 16_000.0],
                bandwidth_hz: vec![29.1, 132.6, 1753.0],
                normalization_gain: vec![0.1, 0.2, 0.3],
                power_detector: ResidualPowerDescriptor {
                    mode: ResidualPowerTimeMode::BandwidthRelative,
                    tau_seconds: vec![10e-3, 7.5e-3, 0.5e-3],
                },
            },
        }
    }

    /// §29.10: both wire formats carry every field, the matrix included.
    #[test]
    fn round_trips_through_json_and_toml() {
        let book = a_book();
        let json: ResidualBook = serde_json::from_str(&serde_json::to_string(&book).unwrap()).unwrap();
        assert_eq!(json, book);
        let toml: ResidualBook = toml::from_str(&toml::to_string_pretty(&book).unwrap()).unwrap();
        assert_eq!(toml, book);
    }

    /// The enums serialise as the snake_case names the settings document uses, so a book and a
    /// config spell the same thing the same way.
    #[test]
    fn enums_use_the_settings_document_spelling() {
        let text = serde_json::to_string(&a_book()).unwrap();
        for want in [
            "\"erb_rate\"",
            "\"gammatone\"",
            "\"unit_noise_power\"",
            "\"f32_linear_power\"",
            "\"bandwidth_relative\"",
        ] {
            assert!(text.contains(want), "missing {want}");
        }

        // And a diagnostic line spells them the same way the wire format does.
        assert_eq!(serde_json::to_string(&ErbSpacing::ErbRate).unwrap(), "\"erb_rate\"");
        assert_eq!(ErbSpacing::ErbRate.to_string(), "erb_rate");
        assert_eq!(ErbFilterKind::Gammatone.to_string(), "gammatone");
        assert_eq!(ErbNormalization::UnitNoisePower.to_string(), "unit_noise_power");
        assert_eq!(ResidualStorage::F32LinearPower.to_string(), "f32_linear_power");
    }

    #[test]
    fn frames_index_and_locate_correctly() {
        let b = a_book();
        assert_eq!(b.frame(0), &[1.0, 2.0, 3.0]);
        assert_eq!(b.frame(1), &[4.0, 5.0, 6.0]);
        assert_eq!(b.frame_sample(0), 120);
        assert_eq!(b.frame_sample(1), 168);
        assert_eq!(b.total_power(), vec![6.0, 15.0]);
    }

    #[test]
    fn validation_catches_a_future_version_and_a_short_matrix() {
        assert!(a_book().validate().is_ok());

        let mut newer = a_book();
        newer.version = RESIDUAL_BOOK_VERSION + 1;
        assert!(newer.validate().unwrap_err().contains("newer than this build"));

        let mut short = a_book();
        short.power.pop();
        assert!(short.validate().is_err());

        let mut ragged = a_book();
        ragged.bank.bandwidth_hz.pop();
        assert!(ragged.validate().unwrap_err().contains("bandwidth_hz"));
    }
}
