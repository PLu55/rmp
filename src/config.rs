//! The settings document.
//!
//! TOML, so the file can carry comments explaining what each knob costs. Every section is optional
//! and every field defaults, so a minimal settings file is legal and an empty one reproduces the
//! built-in voice dictionary.
//!
//! Unknown fields are rejected rather than ignored: in a document whose whole purpose is to be
//! hand-edited, a silently-dropped typo would look exactly like a setting that had no effect.

use crate::dict::BlockConfig;
use crate::fof::ReleasePolicy;
use crate::mp::MpConfig;
use crate::hrmp::{HrmpConfig, MagnitudePolicy, ProbeMode};
use crate::refine::RefineConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub dictionary: DictionarySettings,
    pub envelope: EnvelopeSettings,
    pub blocks: BlockSettings,
    pub pursuit: PursuitSettings,
    pub refine: RefineSettings,
    pub hrmp: HrmpSettings,
}

/// The release policy, fixed for the whole analysis.
///
/// rfofs's release is a linear ramp to zero; only where it starts and how long it lasts are settings.
/// `fade_dur` is derived from `alpha` rather than being a constant — see [`ReleasePolicy`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct EnvelopeSettings {
    /// Amplitude relative to peak at which the release begins. 0.001 is -60 dB.
    pub fade_level: f32,
    /// Release duration is `fade_dur_scale / alpha`, clamped to the bounds below.
    pub fade_dur_scale: f32,
    /// Clamps on the release duration, in milliseconds.
    pub fade_dur_min_ms: f32,
    pub fade_dur_max_ms: f32,
}

impl Default for EnvelopeSettings {
    fn default() -> Self {
        let d = ReleasePolicy::default();
        Self {
            fade_level: d.fade_level,
            fade_dur_scale: d.fade_dur_scale,
            fade_dur_min_ms: d.fade_dur_min * 1000.0,
            fade_dur_max_ms: d.fade_dur_max * 1000.0,
        }
    }
}

impl From<&EnvelopeSettings> for ReleasePolicy {
    fn from(s: &EnvelopeSettings) -> Self {
        Self {
            fade_level: s.fade_level,
            fade_dur_scale: s.fade_dur_scale,
            fade_dur_min: s.fade_dur_min_ms / 1000.0,
            fade_dur_max: s.fade_dur_max_ms / 1000.0,
        }
    }
}

/// The `(alpha, beta)` grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DictionarySettings {
    /// Decay rates in s^-1. The -3 dB bandwidth is `alpha / pi` Hz.
    pub alphas: Vec<f32>,
    /// Attack (skirt) durations in milliseconds.
    pub betas_ms: Vec<f32>,
    /// Drop combinations above this. rfofs renders `alpha*beta > 10` as silence, and its `amax`
    /// normalisation is ill-conditioned well before that.
    pub alpha_beta_max: f32,
}

impl Default for DictionarySettings {
    fn default() -> Self {
        Self {
            alphas: vec![80.0, 128.0, 205.0, 328.0, 524.0, 839.0, 1342.0, 2147.0],
            betas_ms: vec![0.3, 1.0, 3.0],
            alpha_beta_max: 4.0,
        }
    }
}

impl DictionarySettings {
    /// Expand to `(alpha, beta_seconds)` pairs, dropping those past the cap.
    pub fn grid(&self) -> Vec<(f32, f32)> {
        self.alphas
            .iter()
            .flat_map(|&a| self.betas_ms.iter().map(move |&b| (a, b / 1000.0)))
            .filter(|&(a, b)| a * b <= self.alpha_beta_max)
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BlockSettings {
    /// Worst-case fraction of an atom's energy a frame must still capture when the true onset falls
    /// between hop positions. Lower means a coarser hop: faster, but selection degrades.
    pub capture_tolerance: f64,
    /// Frequency range represented, in Hz.
    pub f_min: f32,
    pub f_max: f32,
    /// Bins whose `rho^2` exceeds this are disabled as ill-conditioned.
    pub rho_sq_max: f32,
}

impl Default for BlockSettings {
    fn default() -> Self {
        let d = BlockConfig::default();
        Self {
            capture_tolerance: d.capture_tol,
            f_min: d.f_min,
            f_max: d.f_max,
            rho_sq_max: d.rho_sq_max,
        }
    }
}

/// Note this leaves [`BlockConfig::release`] at its default: the release policy lives in a different
/// section, so only [`Config::block_config`] can produce a fully-populated value. Prefer that.
impl From<&BlockSettings> for BlockConfig {
    fn from(s: &BlockSettings) -> Self {
        Self {
            capture_tol: s.capture_tolerance,
            f_min: s.f_min,
            f_max: s.f_max,
            rho_sq_max: s.rho_sq_max,
            release: ReleasePolicy::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PursuitSettings {
    /// Hard cap on atoms selected.
    pub max_atoms: usize,
    /// Stop once this reconstruction SNR is reached.
    pub target_snr_db: f32,
    /// Stop when the best atom would remove less than this fraction of the residual.
    pub min_gain: f64,
    /// Local maxima promoted to exact scoring each iteration.
    ///
    /// 1 is the plain global argmax. Promoting more only pays once refinement can move an atom off
    /// the grid, since otherwise every candidate is scored by the same table that ranked the seeds.
    pub candidate_count: usize,
    /// Give up after this many consecutive iterations in which every candidate was rejected.
    ///
    /// Only reachable under HRMP. On dense polyphonic material a run of rejections is normal, so
    /// this is what decides whether HRMP declines a few atoms or ends the pursuit outright.
    pub max_stalls: usize,
}

impl Default for PursuitSettings {
    fn default() -> Self {
        let d = MpConfig::default();
        Self {
            max_atoms: d.max_atoms,
            target_snr_db: d.target_snr_db,
            min_gain: d.min_gain_fraction,
            candidate_count: d.candidate_count,
            max_stalls: d.max_stalls,
        }
    }
}

impl From<&PursuitSettings> for MpConfig {
    fn from(s: &PursuitSettings) -> Self {
        Self {
            max_atoms: s.max_atoms,
            target_snr_db: s.target_snr_db,
            min_gain_fraction: s.min_gain,
            candidate_count: s.candidate_count,
            refine: RefineConfig::default(),
            hrmp: HrmpConfig::default(),
            max_stalls: s.max_stalls,
            full_update: false,
        }
    }
}

/// Local refinement of `(t0, f, alpha, beta)` after a candidate is selected.
///
/// The frequency range, the conditioning gate and the `alpha*beta` cap are deliberately absent:
/// they are shared with `[blocks]` and `[dictionary]`, and [`Config::mp_config`] copies them across
/// so refinement cannot wander into a region the coarse search treats as dead.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RefineSettings {
    pub enabled: bool,
    /// Passes of the `f -> alpha -> beta -> t0` cycle.
    pub rounds: usize,
    /// Stop early when a whole round improves the fit score by less than this fraction.
    pub score_tol: f64,
    /// Golden-section evaluations per parameter. The dominant cost knob.
    pub golden_iters: usize,
    /// Bounds on the refined envelope, wider than the dictionary grid at both ends.
    pub alpha_min: f32,
    pub alpha_max: f32,
    pub beta_min_ms: f32,
    pub beta_max_ms: f32,
    /// Reject any refined envelope longer than this, whatever the bounds imply.
    pub max_atom_samples: usize,
    /// Search radii around the seed: bins, then multiplicative factors, then samples.
    pub f_bracket_bins: f32,
    pub alpha_bracket: f32,
    pub beta_bracket: f32,
    /// 0 derives the onset radius from the block's own hop.
    pub t0_radius: usize,
}

impl Default for RefineSettings {
    fn default() -> Self {
        let d = RefineConfig::default();
        Self {
            enabled: d.enabled,
            rounds: d.rounds,
            score_tol: d.score_tol,
            golden_iters: d.golden_iters,
            alpha_min: d.alpha_min,
            alpha_max: d.alpha_max,
            // Written as milliseconds directly: `beta_min * 1000.0` in f32 renders as
            // 0.099999994 in the emitted document, which reads like a bug in a hand-edited file.
            beta_min_ms: 0.1,
            beta_max_ms: 10.0,
            max_atom_samples: d.max_atom_samples,
            f_bracket_bins: d.f_bracket_bins,
            alpha_bracket: d.alpha_bracket,
            beta_bracket: d.beta_bracket,
            t0_radius: d.t0_radius,
        }
    }
}

/// High-Resolution Matching Pursuit: reject or clamp an atom the residual does not support across
/// its whole extent.
///
/// Off by default. It is a *stricter* criterion than ordinary MP, so it trades reconstruction SNR
/// per atom for atoms that describe events actually present.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct HrmpSettings {
    pub enabled: bool,
    /// `localized_candidate` masks the refined atom's own envelope. `legacy_scaled_fof` uses
    /// smaller same-frequency FOFs, as in the historical implementation.
    pub mode: ProbeMode,
    /// Probe count, and in `legacy_scaled_fof` also probe scale: that mode needs a larger depth
    /// than `localized_candidate` to reach the same locality, because a mask can be short at any
    /// depth while a scaled FOF's support shrinks only as `2^-depth`.
    pub depth: u32,
    /// Reject when a probe's local phase disagrees with the global fit by more than this.
    pub phase_tolerance_deg: f32,
    /// Structural floor on a probe's share of the atom's energy before it gets a vote.
    pub minimum_probe_energy: f64,
    /// Bound on each local amplitude's relative standard error. A probe must see at least
    /// `1/noise_epsilon^2` times the local residual noise power in atom energy to be believed.
    pub noise_epsilon: f64,
    /// Skip HRMP entirely when a probe would span fewer carrier periods than this: its Gram cannot
    /// be conditioned, and an atom that short cannot bridge anything anyway.
    pub min_mask_periods: f32,
    /// `strict_min` is the original criterion. A robust quantile is deliberately not offered here,
    /// because it is a different algorithm and should not be mistaken for this one.
    pub magnitude_policy: MagnitudePolicy,
}

impl Default for HrmpSettings {
    fn default() -> Self {
        let d = HrmpConfig::default();
        Self {
            enabled: d.enabled,
            mode: d.mode,
            depth: d.depth,
            phase_tolerance_deg: d.phase_tolerance_rad.to_degrees(),
            minimum_probe_energy: d.min_probe_energy,
            noise_epsilon: d.noise_epsilon,
            min_mask_periods: d.min_mask_periods,
            magnitude_policy: d.magnitude_policy,
        }
    }
}

impl Config {
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Block settings with the release policy from `[envelope]` folded in.
    pub fn block_config(&self) -> BlockConfig {
        BlockConfig {
            release: (&self.envelope).into(),
            ..(&self.blocks).into()
        }
    }

    /// Pursuit settings with `[refine]` folded in.
    ///
    /// The frequency range, conditioning gate and `alpha*beta` cap are copied from the sections
    /// that already own them, so the coarse and refined paths cannot disagree about which
    /// parameters are representable.
    pub fn mp_config(&self) -> MpConfig {
        let r = &self.refine;
        MpConfig {
            refine: RefineConfig {
                enabled: r.enabled,
                rounds: r.rounds,
                score_tol: r.score_tol,
                golden_iters: r.golden_iters,
                f_min: self.blocks.f_min,
                f_max: self.blocks.f_max,
                alpha_min: r.alpha_min,
                alpha_max: r.alpha_max,
                beta_min: r.beta_min_ms / 1000.0,
                beta_max: r.beta_max_ms / 1000.0,
                alpha_beta_max: self.dictionary.alpha_beta_max,
                max_atom_samples: r.max_atom_samples,
                f_bracket_bins: r.f_bracket_bins,
                alpha_bracket: r.alpha_bracket,
                beta_bracket: r.beta_bracket,
                t0_radius: r.t0_radius,
                rho_sq_max: self.blocks.rho_sq_max as f64,
            },
            hrmp: HrmpConfig {
                enabled: self.hrmp.enabled,
                mode: self.hrmp.mode,
                depth: self.hrmp.depth,
                phase_tolerance_rad: self.hrmp.phase_tolerance_deg.to_radians(),
                min_probe_energy: self.hrmp.minimum_probe_energy,
                noise_epsilon: self.hrmp.noise_epsilon,
                min_mask_periods: self.hrmp.min_mask_periods,
                magnitude_policy: self.hrmp.magnitude_policy,
                rho_sq_max: self.blocks.rho_sq_max as f64,
            },
            ..(&self.pursuit).into()
        }
    }

    /// A fully-populated settings document, for `--write-config`.
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    /// Reject settings that would produce an unusable dictionary before any work starts.
    pub fn validate(&self) -> Result<(), String> {
        if self.dictionary.grid().is_empty() {
            return Err(
                "dictionary grid is empty: check alphas, betas_ms and alpha_beta_max".into(),
            );
        }
        // NaN must fail too, hence >= rather than a negated <.
        if self.blocks.f_min >= self.blocks.f_max || !self.blocks.f_min.is_finite() {
            return Err(format!(
                "f_min ({}) must be below f_max ({})",
                self.blocks.f_min, self.blocks.f_max
            ));
        }
        if !(0.0..1.0).contains(&self.blocks.capture_tolerance) || self.blocks.capture_tolerance == 0.0 {
            return Err("capture_tolerance must be in (0, 1)".into());
        }
        let release: ReleasePolicy = (&self.envelope).into();
        release
            .validate()
            .map_err(|e| format!("[envelope]: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_document_is_the_default_dictionary() {
        let cfg = Config::from_toml("").unwrap();
        // 8 alphas x 3 betas, less 1342*3ms = 4.026 and 2147*3ms = 6.441.
        assert_eq!(cfg.dictionary.grid().len(), 22);
        assert_eq!(cfg.pursuit.target_snr_db, MpConfig::default().target_snr_db);
    }

    #[test]
    fn partial_document_keeps_other_defaults() {
        let cfg = Config::from_toml("[pursuit]\nmax_atoms = 12\n").unwrap();
        assert_eq!(cfg.pursuit.max_atoms, 12);
        assert_eq!(cfg.blocks.f_max, BlockConfig::default().f_max);
        assert_eq!(cfg.dictionary.grid().len(), 22);
    }

    #[test]
    fn envelope_section_feeds_the_release_policy() {
        let cfg = Config::from_toml("").unwrap();
        assert_eq!(cfg.block_config().release, ReleasePolicy::default());

        let cfg = Config::from_toml(
            "[envelope]\nfade_level = 0.01\nfade_dur_scale = 4.0\nfade_dur_max_ms = 20.0\n",
        )
        .unwrap();
        let r = cfg.block_config().release;
        assert_eq!(r.fade_level, 0.01);
        assert_eq!(r.fade_dur_scale, 4.0);
        assert_eq!(r.fade_dur_max, 20e-3);
        assert_eq!(r.fade_dur_min, ReleasePolicy::default().fade_dur_min); // untouched
        assert!(cfg.validate().is_ok());

        // block_config must carry the other [blocks] settings through, not just the release.
        let cfg = Config::from_toml("[blocks]\nf_max = 8000.0\n").unwrap();
        assert_eq!(cfg.block_config().f_max, 8000.0);
    }

    #[test]
    fn unusable_release_policies_are_rejected() {
        for doc in [
            "[envelope]\nfade_level = 0.0\n",
            "[envelope]\nfade_level = 1.0\n",
            "[envelope]\nfade_dur_min_ms = 20.0\nfade_dur_max_ms = 1.0\n",
        ] {
            assert!(Config::from_toml(doc).unwrap().validate().is_err(), "{doc}");
        }
    }

    #[test]
    fn unknown_fields_are_rejected() {
        // A typo in a hand-edited settings file must fail loudly, not be silently ignored.
        let err = Config::from_toml("[pursuit]\nmax_atom = 12\n").unwrap_err();
        assert!(err.to_string().contains("max_atom"), "{err}");

        assert!(Config::from_toml("[pursiut]\nmax_atoms = 12\n").is_err());
    }

    #[test]
    fn grid_applies_the_alpha_beta_cap() {
        let cfg = Config::from_toml(
            "[dictionary]\nalphas = [100.0, 2000.0]\nbetas_ms = [1.0, 5.0]\nalpha_beta_max = 4.0\n",
        )
        .unwrap();
        // 100*0.001, 100*0.005, 2000*0.001 pass; 2000*0.005 = 10 does not.
        assert_eq!(cfg.dictionary.grid().len(), 3);
    }

    #[test]
    fn betas_are_milliseconds_in_the_document_and_seconds_in_the_grid() {
        let cfg =
            Config::from_toml("[dictionary]\nalphas = [100.0]\nbetas_ms = [2.5]\n").unwrap();
        assert_eq!(cfg.dictionary.grid(), vec![(100.0, 0.0025)]);
    }

    #[test]
    fn validation_catches_unusable_settings() {
        let empty = Config::from_toml("[dictionary]\nalphas = []\n").unwrap();
        assert!(empty.validate().is_err());

        let inverted = Config::from_toml("[blocks]\nf_min = 9000.0\nf_max = 100.0\n").unwrap();
        assert!(inverted.validate().is_err());

        let bad_tol = Config::from_toml("[blocks]\ncapture_tolerance = 1.5\n").unwrap();
        assert!(bad_tol.validate().is_err());

        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn round_trips_through_toml() {
        let cfg = Config::default();
        let restored = Config::from_toml(&cfg.to_toml()).unwrap();
        assert_eq!(cfg.dictionary.grid(), restored.dictionary.grid());
        assert_eq!(cfg.pursuit.max_atoms, restored.pursuit.max_atoms);
    }
}
