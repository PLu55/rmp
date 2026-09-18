//! The structure settings document.
//!
//! A TOML file of its own, separate from the analysis settings: `[structure.partials]` and
//! `[structure.stems]`. It follows the conventions of `rmp_core::config` exactly — every field
//! optional with a default, unknown keys rejected so a typo fails loudly, and a `validate` that runs
//! before any work — but it is a separate document because it describes a different computation.
//! Changing how partials are tracked says nothing about how the book was decomposed, and keeping the
//! two apart means it cannot make a finished decomposition look out of date.
//!
//! The partial settings are flat in the file, as the specification writes them. [`FrequencyScale`]
//! and [`PartialTrackingConfig`] are the typed views the algorithm actually reads, derived here and
//! nowhere else.

use crate::error::{Result, StructureError};
use serde::{Deserialize, Serialize};

/// Both halves of structural analysis.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StructureAnalysisConfig {
    pub partials: PartialAnalysisConfig,
    pub stems: StemAnalysisConfig,
}

/// The file's top level: everything lives under `[structure]`, so a settings file says what it is
/// for and could one day sit beside other sections without renaming anything.
#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Document {
    structure: StructureAnalysisConfig,
}

impl StructureAnalysisConfig {
    pub fn from_toml(text: &str) -> std::result::Result<Self, String> {
        let doc: Document = toml::from_str(text).map_err(|e| e.to_string())?;
        Ok(doc.structure)
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(&Document { structure: self.clone() }).unwrap_or_default()
    }

    pub fn validate(&self) -> Result<()> {
        self.partials.validate()?;
        self.stems.validate()
    }
}

/// Which frequency axis the time-frequency accumulation is binned on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScaleKind {
    LogCents,
    LinearHz,
    Erb,
}

/// The accumulation's frequency axis, resolved with its resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrequencyScale {
    LinearHz { bin_width_hz: f64 },
    LogCents { cents_per_bin: f64, reference_hz: f64 },
    Erb { bands_per_erb: f64 },
}

/// Ridge linking and rejection, as §11 names them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartialTrackingConfig {
    pub max_jump_cents_per_frame: f64,
    pub max_gap_frames: usize,
    pub max_drift_cents: f64,
    pub min_duration_ms: f64,
    pub min_persistence: f64,
    pub min_relative_level_db: f64,
}

/// Persistent partial extraction: `[structure.partials]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PartialAnalysisConfig {
    /// Frame hop of the accumulation grid.
    pub time_step_ms: f64,
    /// Width of the analysis window each atom is blurred by in time.
    pub window_ms: f64,
    pub frequency_scale: ScaleKind,
    /// Resolution of `log-cents`.
    pub cents_per_bin: f64,
    /// The frequency `log-cents` counts from. Only moves where bin edges fall.
    pub reference_hz: f64,
    /// Resolution of `linear-hz`.
    pub bin_width_hz: f64,
    /// Resolution of `erb`.
    pub bands_per_erb: f64,
    /// Frequency range of the grid. `f_max` is also capped at Nyquist.
    pub f_min: f64,
    pub f_max: f64,

    pub max_jump_cents_per_frame: f64,
    pub max_gap_frames: usize,
    /// How far a partial may wander from its own mean frequency, in cents. `0` is no limit. Set it
    /// for fixed-pitch instruments — piano, harp, mallets — where a partial cannot glide, and any
    /// apparent glide is two notes being joined.
    pub max_drift_cents: f64,
    pub min_duration_ms: f64,
    pub min_persistence: f64,
    /// Peaks below the grid's strongest cell by more than this are not candidates.
    pub min_relative_level_db: f64,

    /// Moving-average width, in frames, applied before simplification. 1 disables it.
    pub smoothing_frames: usize,
    pub frequency_simplify_cents: f64,
    pub amplitude_simplify_db: f64,

    /// Return the accumulation grid and the raw peaks with the result. For inspection; it changes
    /// nothing about the partials and is not part of the document.
    #[serde(skip)]
    pub keep_intermediates: bool,
}

impl Default for PartialAnalysisConfig {
    fn default() -> Self {
        Self {
            time_step_ms: 10.0,
            window_ms: 40.0,
            frequency_scale: ScaleKind::LogCents,
            cents_per_bin: 20.0,
            reference_hz: 440.0,
            bin_width_hz: 10.0,
            bands_per_erb: 4.0,
            f_min: 20.0,
            f_max: 20_000.0,
            max_jump_cents_per_frame: 50.0,
            max_gap_frames: 2,
            max_drift_cents: 0.0,
            min_duration_ms: 100.0,
            min_persistence: 0.5,
            min_relative_level_db: -60.0,
            smoothing_frames: 3,
            frequency_simplify_cents: 5.0,
            amplitude_simplify_db: 0.5,
            keep_intermediates: false,
        }
    }
}

fn positive(name: &str, v: f64) -> Result<()> {
    if v > 0.0 && v.is_finite() {
        Ok(())
    } else {
        Err(StructureError::InvalidConfig(format!("{name} must be > 0 and finite, got {v}")))
    }
}

fn unit(name: &str, v: f64) -> Result<()> {
    if (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(StructureError::InvalidConfig(format!("{name} must be in [0, 1], got {v}")))
    }
}

impl PartialAnalysisConfig {
    pub fn scale(&self) -> FrequencyScale {
        match self.frequency_scale {
            ScaleKind::LogCents => FrequencyScale::LogCents {
                cents_per_bin: self.cents_per_bin,
                reference_hz: self.reference_hz,
            },
            ScaleKind::LinearHz => FrequencyScale::LinearHz { bin_width_hz: self.bin_width_hz },
            ScaleKind::Erb => FrequencyScale::Erb { bands_per_erb: self.bands_per_erb },
        }
    }

    pub fn tracking(&self) -> PartialTrackingConfig {
        PartialTrackingConfig {
            max_jump_cents_per_frame: self.max_jump_cents_per_frame,
            max_gap_frames: self.max_gap_frames,
            max_drift_cents: self.max_drift_cents,
            min_duration_ms: self.min_duration_ms,
            min_persistence: self.min_persistence,
            min_relative_level_db: self.min_relative_level_db,
        }
    }

    pub fn validate(&self) -> Result<()> {
        positive("time_step_ms", self.time_step_ms)?;
        positive("window_ms", self.window_ms)?;
        positive("cents_per_bin", self.cents_per_bin)?;
        positive("reference_hz", self.reference_hz)?;
        positive("bin_width_hz", self.bin_width_hz)?;
        positive("bands_per_erb", self.bands_per_erb)?;
        positive("f_min", self.f_min)?;
        positive("f_max", self.f_max)?;
        if self.f_max <= self.f_min {
            return Err(StructureError::InvalidConfig(format!(
                "f_max ({}) must be above f_min ({})",
                self.f_max, self.f_min
            )));
        }
        positive("max_jump_cents_per_frame", self.max_jump_cents_per_frame)?;
        if !(self.max_drift_cents >= 0.0 && self.max_drift_cents.is_finite()) {
            return Err(StructureError::InvalidConfig(
                "max_drift_cents must be >= 0 (0 disables the limit)".into(),
            ));
        }
        if !(self.min_duration_ms >= 0.0 && self.min_duration_ms.is_finite()) {
            return Err(StructureError::InvalidConfig("min_duration_ms must be >= 0".into()));
        }
        unit("min_persistence", self.min_persistence)?;
        if !(self.min_relative_level_db < 0.0 && self.min_relative_level_db.is_finite()) {
            return Err(StructureError::InvalidConfig(
                "min_relative_level_db must be negative: it is a level below the strongest cell"
                    .into(),
            ));
        }
        if self.smoothing_frames == 0 {
            return Err(StructureError::InvalidConfig(
                "smoothing_frames must be >= 1 (1 disables smoothing)".into(),
            ));
        }
        if !(self.frequency_simplify_cents >= 0.0 && self.amplitude_simplify_db >= 0.0) {
            return Err(StructureError::InvalidConfig(
                "simplification tolerances must be >= 0".into(),
            ));
        }
        Ok(())
    }
}

/// Stem extraction: `[structure.stems]`.
///
/// Declared in full now so a settings file written today holds every section it will need, but
/// nothing reads it yet: stem extraction is the next phase, after partials have been validated on
/// real books.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StemAnalysisConfig {
    /// Rate of the common grid every partial's morphology is resampled onto.
    pub morphology_rate_hz: f64,
    pub onset_tau_ms: f64,
    pub offset_tau_ms: f64,
    /// Largest timing offset tolerated when comparing two partials' morphology.
    pub max_morphology_lag_ms: f64,

    pub weight_onset: f64,
    pub weight_offset: f64,
    pub weight_amplitude: f64,
    pub weight_frequency: f64,
    pub weight_harmonic: f64,
    pub weight_spectral: f64,
    pub weight_spatial: f64,

    pub seed_affinity: f64,
    pub merge_affinity: f64,
    /// Overlap as a fraction of the shorter partial's lifetime below which no edge is made.
    pub min_partial_overlap: f64,
    pub min_stem_confidence: f64,
}

impl Default for StemAnalysisConfig {
    fn default() -> Self {
        Self {
            morphology_rate_hz: 100.0,
            onset_tau_ms: 20.0,
            offset_tau_ms: 30.0,
            max_morphology_lag_ms: 20.0,
            weight_onset: 1.0,
            weight_offset: 0.5,
            weight_amplitude: 2.0,
            weight_frequency: 2.0,
            weight_harmonic: 1.0,
            weight_spectral: 1.0,
            weight_spatial: 2.0,
            seed_affinity: 0.80,
            merge_affinity: 0.65,
            min_partial_overlap: 0.30,
            min_stem_confidence: 0.50,
        }
    }
}

impl StemAnalysisConfig {
    pub fn validate(&self) -> Result<()> {
        positive("morphology_rate_hz", self.morphology_rate_hz)?;
        positive("onset_tau_ms", self.onset_tau_ms)?;
        positive("offset_tau_ms", self.offset_tau_ms)?;
        if !(self.max_morphology_lag_ms >= 0.0 && self.max_morphology_lag_ms.is_finite()) {
            return Err(StructureError::InvalidConfig("max_morphology_lag_ms must be >= 0".into()));
        }
        let weights = [
            ("weight_onset", self.weight_onset),
            ("weight_offset", self.weight_offset),
            ("weight_amplitude", self.weight_amplitude),
            ("weight_frequency", self.weight_frequency),
            ("weight_harmonic", self.weight_harmonic),
            ("weight_spectral", self.weight_spectral),
            ("weight_spatial", self.weight_spatial),
        ];
        for (name, w) in weights {
            if !(w >= 0.0 && w.is_finite()) {
                return Err(StructureError::InvalidConfig(format!("{name} must be >= 0")));
            }
        }
        if weights.iter().all(|&(_, w)| w == 0.0) {
            return Err(StructureError::InvalidConfig("at least one weight must be > 0".into()));
        }
        unit("seed_affinity", self.seed_affinity)?;
        unit("merge_affinity", self.merge_affinity)?;
        unit("min_partial_overlap", self.min_partial_overlap)?;
        unit("min_stem_confidence", self.min_stem_confidence)?;
        Ok(())
    }
}

/// The commentary `rmpstruct --write-config` puts above the defaults.
pub const DEFAULT_CONFIG_HEADER: &str = "\
# rmp structural analysis settings (rmpstruct).
#
# Every section and field is optional; omitted values fall back to these defaults.
# Unknown keys are rejected rather than ignored, so a typo fails loudly.
# MANUAL.md, section \"Structural analysis\", documents each setting.
#
# [structure.partials]  atoms -> persistent partials
#   frequency_scale     \"log-cents\" (cents_per_bin, reference_hz), \"linear-hz\" (bin_width_hz)
#                       or \"erb\" (bands_per_erb)
#
# [structure.stems]     partials -> stems. Not implemented yet; read but unused.

";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_round_trip_and_validate() {
        let c = StructureAnalysisConfig::default();
        c.validate().unwrap();
        let text = c.to_toml();
        assert!(text.contains("[structure.partials]"), "{text}");
        assert!(text.contains("frequency_scale = \"log-cents\""), "{text}");
        assert_eq!(StructureAnalysisConfig::from_toml(&text).unwrap(), c);
        let with_header = format!("{DEFAULT_CONFIG_HEADER}{text}");
        assert_eq!(StructureAnalysisConfig::from_toml(&with_header).unwrap(), c);
    }

    #[test]
    fn the_example_in_the_specification_parses() {
        let text = r#"
            [structure.partials]
            time_step_ms = 10.0
            window_ms = 40.0
            frequency_scale = "log-cents"
            cents_per_bin = 20.0
            min_duration_ms = 100.0
            min_persistence = 0.50
            min_relative_level_db = -60.0
            max_jump_cents_per_frame = 50.0
            max_gap_frames = 2
            frequency_simplify_cents = 5.0
            amplitude_simplify_db = 0.5

            [structure.stems]
            morphology_rate_hz = 100.0
            onset_tau_ms = 20.0
            offset_tau_ms = 30.0
            max_morphology_lag_ms = 20.0
            weight_onset = 1.0
            weight_offset = 0.5
            weight_amplitude = 2.0
            weight_frequency = 2.0
            weight_harmonic = 1.0
            weight_spectral = 1.0
            weight_spatial = 2.0
            seed_affinity = 0.80
            merge_affinity = 0.65
            min_partial_overlap = 0.30
            min_stem_confidence = 0.50
        "#;
        let c = StructureAnalysisConfig::from_toml(text).unwrap();
        assert_eq!(c, StructureAnalysisConfig::default());
    }

    /// The shipped piano settings parse, validate, and set what their header says they set.
    #[test]
    fn the_piano_settings_parse() {
        let text = include_str!("../../data/config/structure-piano.toml");
        let c = StructureAnalysisConfig::from_toml(text).unwrap();
        c.validate().unwrap();
        assert_eq!(c.partials.max_drift_cents, 50.0);
    }

    #[test]
    fn a_typo_and_a_bad_value_are_both_refused() {
        assert!(StructureAnalysisConfig::from_toml("[structure.partials]\ntime_stp_ms = 5\n").is_err());
        assert!(StructureAnalysisConfig::from_toml("[partials]\n").is_err());
        let mut c = StructureAnalysisConfig::default();
        c.partials.min_persistence = 1.5;
        assert!(c.validate().is_err());
        let mut c = StructureAnalysisConfig::default();
        c.partials.min_relative_level_db = 10.0;
        assert!(c.validate().is_err());
        let mut c = StructureAnalysisConfig::default();
        c.partials.max_drift_cents = -1.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn every_scale_kind_is_spelled_as_the_specification_spells_it() {
        for (kind, name) in [
            (ScaleKind::LogCents, "log-cents"),
            (ScaleKind::LinearHz, "linear-hz"),
            (ScaleKind::Erb, "erb"),
        ] {
            let mut c = StructureAnalysisConfig::default();
            c.partials.frequency_scale = kind;
            assert!(c.to_toml().contains(&format!("frequency_scale = \"{name}\"")));
        }
    }
}
