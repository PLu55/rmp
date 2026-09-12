//! The settings document.
//!
//! TOML, so the file can carry comments explaining what each knob costs. Every section is optional
//! and every field defaults, so a minimal settings file is legal and an empty one reproduces the
//! built-in voice dictionary.
//!
//! Unknown fields are rejected rather than ignored: in a document whose whole purpose is to be
//! hand-edited, a silently-dropped typo would look exactly like a setting that had no effect.

use crate::atom::Shape;
use crate::dict::BlockConfig;
use crate::fof::{EnvelopeParams, ReleasePolicy};
use crate::gauss::{DEFAULT_CUTOFF_LEVEL, GaussianParams};
use crate::mp::MpConfig;
use crate::hrmp::{HrmpConfig, MagnitudePolicy, ProbeMode};
use crate::refine::RefineConfig;
use crate::residual::book::{ErbFilterKind, ErbNormalization, ErbSpacing, ResidualStorage};
use crate::residual::config::{ErbBankConfig, ResidualAnalysisConfig};
use crate::residual::error::ResidualAnalysisError;
use crate::residual::power::{ResidualPowerConfig, ResidualPowerTimeMode};
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
    pub residual: ResidualSettings,
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

/// Which atoms the dictionary offers: one family per atom kind.
///
/// Blocks are built FOF family first, then Gaussian. That order is the block index a book records
/// and the tie-break selection uses, so it is what keeps a FOF-only document decomposing exactly as
/// it did before there was a second family.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DictionarySettings {
    pub fof: FofFamilySettings,
    pub gaussian: GaussianFamilySettings,
}

/// The FOF `(alpha, beta)` grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FofFamilySettings {
    /// Decay rates in s^-1. The -3 dB bandwidth is `alpha / pi` Hz. Empty disables the family.
    pub alphas: Vec<f32>,
    /// Attack (skirt) durations in milliseconds.
    pub betas_ms: Vec<f32>,
    /// Drop combinations above this. rfofs renders `alpha*beta > 10` as silence, and its `amax`
    /// normalisation is ill-conditioned well before that.
    pub alpha_beta_max: f32,
}

impl Default for FofFamilySettings {
    fn default() -> Self {
        Self {
            alphas: vec![80.0, 128.0, 205.0, 328.0, 524.0, 839.0, 1342.0, 2147.0],
            betas_ms: vec![0.3, 1.0, 3.0],
            alpha_beta_max: 4.0,
        }
    }
}

impl FofFamilySettings {
    /// Expand to `(alpha, beta_seconds)` pairs, dropping those past the cap.
    pub fn grid(&self) -> Vec<(f32, f32)> {
        self.alphas
            .iter()
            .flat_map(|&a| self.betas_ms.iter().map(move |&b| (a, b / 1000.0)))
            .filter(|&(a, b)| a * b <= self.alpha_beta_max)
            .collect()
    }
}

/// The Gaussian `sigma` ladder. Off by default.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct GaussianFamilySettings {
    /// Envelope standard deviations in milliseconds. The -3 dB bandwidth is `0.265 / sigma` and
    /// the support `±3.72 sigma` at the default cutoff. Empty disables the family.
    pub sigmas_ms: Vec<f32>,
    /// Amplitude relative to the peak at which the support is truncated. 0.001 is -60 dB.
    pub cutoff_level: f32,
}

impl Default for GaussianFamilySettings {
    fn default() -> Self {
        Self { sigmas_ms: Vec::new(), cutoff_level: DEFAULT_CUTOFF_LEVEL }
    }
}

impl GaussianFamilySettings {
    pub fn shapes(&self) -> Vec<GaussianParams> {
        self.sigmas_ms
            .iter()
            .map(|&s| GaussianParams { sigma: s / 1000.0, cutoff_level: self.cutoff_level })
            .collect()
    }
}

/// Keys that lived directly under `[dictionary]` before it held more than one family.
const MOVED_TO_FOF: [&str; 3] = ["alphas", "betas_ms", "alpha_beta_max"];

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
    /// Frame-table budget, in MiB, that decides how long a window the pursuit analyses at once.
    ///
    /// The tables scale with the *signal*, at a per-sample cost the dictionary sets, so a long clip
    /// is analysed a window at a time. A signal whose tables fit under this is one window and is
    /// decomposed exactly as it always was; past it, greedy selection becomes per-window and
    /// `target_snr_db`, `min_gain` and `max_atoms` become per-window quantities.
    ///
    /// Raise it to keep more of the clip under one greedy order; lower it to fit a smaller machine.
    pub max_memory_mb: usize,
    /// Analyse windows of exactly this many seconds, ignoring `max_memory_mb`.
    ///
    /// The budget picks a window from the machine's memory, which makes a book depend on the
    /// machine. Set this when a run has to reproduce elsewhere. 0 means "use the budget".
    pub window_seconds: f32,
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
            max_memory_mb: 1024,
            window_seconds: 0.0,
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
/// they are shared with `[blocks]` and `[dictionary.fof]`, and [`Config::mp_config`] copies them across
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
    /// Bounds on a refined Gaussian's `sigma`.
    pub sigma_min_ms: f32,
    pub sigma_max_ms: f32,
    /// Reject any refined envelope longer than this, whatever the bounds imply.
    pub max_atom_samples: usize,
    /// Search radii around the seed: bins, then multiplicative factors, then samples.
    pub f_bracket_bins: f32,
    pub alpha_bracket: f32,
    pub beta_bracket: f32,
    pub sigma_bracket: f32,
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
            sigma_min_ms: 0.5,
            sigma_max_ms: 200.0,
            max_atom_samples: d.max_atom_samples,
            f_bracket_bins: d.f_bracket_bins,
            alpha_bracket: d.alpha_bracket,
            beta_bracket: d.beta_bracket,
            sigma_bracket: d.sigma_bracket,
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

/// Stochastic analysis of what the pursuit could not explain.
///
/// Off by default, and a strict post-processing stage: enabling it cannot change which atoms were
/// selected. See [`crate::residual`] for the model.
///
/// Milliseconds here, samples and seconds in [`ResidualAnalysisConfig`] — the conversion happens
/// once, in [`Config::residual_config`], because it needs the sample rate the file turns out to
/// have.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ResidualSettings {
    pub enabled: bool,
    /// How often a power frame is recorded. Independent of the filter bank's own rate, which is
    /// always the audio rate.
    pub update_ms: f64,
    /// How the stored numbers encode power.
    pub storage: ResidualStorage,
    pub erb: ResidualErbSettings,
    pub power: ResidualPowerSettings,
}

impl Default for ResidualSettings {
    fn default() -> Self {
        let d = ResidualAnalysisConfig::default();
        Self {
            enabled: d.enabled,
            update_ms: 1.0,
            storage: d.storage,
            erb: ResidualErbSettings::default(),
            power: ResidualPowerSettings::default(),
        }
    }
}

/// The analysis filter bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ResidualErbSettings {
    pub bands: usize,
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    pub spacing: ErbSpacing,
    pub filter: ErbFilterKind,
    pub order: usize,
    pub normalization: ErbNormalization,
}

impl Default for ResidualErbSettings {
    fn default() -> Self {
        let d = ErbBankConfig::default();
        Self {
            bands: d.bands,
            min_freq_hz: d.min_freq_hz,
            max_freq_hz: d.max_freq_hz,
            spacing: d.spacing,
            filter: d.filter_kind,
            order: d.filter_order,
            normalization: d.normalization,
        }
    }
}

impl From<&ResidualErbSettings> for ErbBankConfig {
    fn from(s: &ResidualErbSettings) -> Self {
        Self {
            bands: s.bands,
            min_freq_hz: s.min_freq_hz,
            max_freq_hz: s.max_freq_hz,
            spacing: s.spacing,
            filter_kind: s.filter,
            filter_order: s.order,
            normalization: s.normalization,
        }
    }
}

/// The per-band power detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ResidualPowerSettings {
    pub mode: ResidualPowerTimeMode,
    /// Used by `mode = "fixed"`.
    pub tau_ms: f64,
    /// Used by `mode = "bandwidth_relative"`: `tau_b = clamp(tau_scale / bandwidth_b, min, max)`,
    /// with `tau_scale` dimensionless.
    pub tau_scale: f64,
    pub tau_min_ms: f64,
    pub tau_max_ms: f64,
}

impl Default for ResidualPowerSettings {
    fn default() -> Self {
        let d = ResidualPowerConfig::default();
        Self {
            mode: d.mode,
            tau_ms: d.fixed_tau_seconds * 1e3,
            tau_scale: d.bandwidth_tau_scale,
            tau_min_ms: d.tau_min_seconds * 1e3,
            tau_max_ms: d.tau_max_seconds * 1e3,
        }
    }
}

impl From<&ResidualPowerSettings> for ResidualPowerConfig {
    fn from(s: &ResidualPowerSettings) -> Self {
        Self {
            mode: s.mode,
            fixed_tau_seconds: s.tau_ms / 1e3,
            bandwidth_tau_scale: s.tau_scale,
            tau_min_seconds: s.tau_min_ms / 1e3,
            tau_max_seconds: s.tau_max_ms / 1e3,
        }
    }
}

impl Config {
    /// Read a settings document from a file, or the defaults when there is no path.
    ///
    /// `None` is the defaults rather than an error because every front end treats a missing
    /// settings file that way, and the errors name the path, which [`Config::from_toml`] cannot.
    pub fn load(path: Option<&std::path::Path>) -> Result<Self, String> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        Self::from_toml(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
    }

    /// Parse a settings document.
    ///
    /// A document written before `[dictionary]` held more than one family puts `alphas` directly
    /// under it. `deny_unknown_fields` would reject that anyway, but only as "unknown field", which
    /// reads like a typo in a file that was correct yesterday — so the move is named instead.
    pub fn from_toml(text: &str) -> Result<Self, String> {
        let table: toml::Table = toml::from_str(text).map_err(|e| e.to_string())?;
        if let Some(toml::Value::Table(dict)) = table.get("dictionary")
            && let Some(key) = MOVED_TO_FOF.iter().find(|k| dict.contains_key(**k))
        {
            return Err(format!(
                "[dictionary] {key} has moved to [dictionary.fof]: the dictionary now holds one \
                 family per atom kind ([dictionary.fof], [dictionary.gaussian]), so put a \
                 [dictionary.fof] header above alphas, betas_ms and alpha_beta_max"
            ));
        }
        toml::from_str(text).map_err(|e| e.to_string())
    }

    /// Every block shape the dictionary will hold, FOF family first.
    pub fn dictionary_shapes(&self) -> Vec<Shape> {
        let release: ReleasePolicy = (&self.envelope).into();
        let fof = self
            .dictionary
            .fof
            .grid()
            .into_iter()
            .map(|(a, b)| Shape::from(EnvelopeParams::with_policy(a, b, &release)));
        let gauss = self.dictionary.gaussian.shapes().into_iter().map(Shape::from);
        fof.chain(gauss).collect()
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
                alpha_beta_max: self.dictionary.fof.alpha_beta_max,
                sigma_min: r.sigma_min_ms / 1000.0,
                sigma_max: r.sigma_max_ms / 1000.0,
                max_atom_samples: r.max_atom_samples,
                f_bracket_bins: r.f_bracket_bins,
                alpha_bracket: r.alpha_bracket,
                beta_bracket: r.beta_bracket,
                sigma_bracket: r.sigma_bracket,
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

    /// Residual settings resolved against the sample rate the analysis will actually run at.
    ///
    /// This is the only place `update_ms` becomes samples: the spec is explicit that milliseconds
    /// are converted once and the sample count carried thereafter. Fails rather than clamps —
    /// `max_freq_hz` above the usable limit at this rate is a settings error, and an `update_ms`
    /// under a sample is not a request for a one-sample interval.
    pub fn residual_config(
        &self,
        sample_rate: f64,
    ) -> Result<ResidualAnalysisConfig, ResidualAnalysisError> {
        let r = &self.residual;
        if !(r.update_ms.is_finite() && r.update_ms > 0.0) {
            return Err(ResidualAnalysisError::InvalidUpdateInterval {
                update_ms: r.update_ms,
                sample_rate,
            });
        }
        let update = ((r.update_ms * 0.001) * sample_rate).round();
        if !(update >= 1.0 && update <= usize::MAX as f64) {
            return Err(ResidualAnalysisError::InvalidUpdateInterval {
                update_ms: r.update_ms,
                sample_rate,
            });
        }

        let cfg = ResidualAnalysisConfig {
            enabled: r.enabled,
            update_samples: update as usize,
            erb: (&r.erb).into(),
            power: (&r.power).into(),
            storage: r.storage,
        };
        cfg.validate(sample_rate)?;
        Ok(cfg)
    }

    /// A fully-populated settings document, for `--write-config`.
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    /// Reject settings that would produce an unusable dictionary before any work starts.
    pub fn validate(&self) -> Result<(), String> {
        if self.dictionary_shapes().is_empty() {
            return Err(
                "dictionary is empty: [dictionary.fof] (alphas, betas_ms, alpha_beta_max) and \
                 [dictionary.gaussian] (sigmas_ms) give no shapes between them"
                    .into(),
            );
        }
        let g = &self.dictionary.gaussian;
        if let Some(s) = g.sigmas_ms.iter().find(|s| !(s.is_finite() && **s > 0.0)) {
            return Err(format!("[dictionary.gaussian] sigmas_ms must be positive, got {s}"));
        }
        if !(g.cutoff_level > 0.0 && g.cutoff_level < 1.0) {
            return Err(format!(
                "[dictionary.gaussian] cutoff_level must be in (0, 1), got {}",
                g.cutoff_level
            ));
        }
        let r = &self.refine;
        if !(r.sigma_min_ms > 0.0 && r.sigma_min_ms <= r.sigma_max_ms && r.sigma_max_ms.is_finite())
        {
            return Err(format!(
                "[refine] needs 0 < sigma_min_ms <= sigma_max_ms, got {} and {}",
                r.sigma_min_ms, r.sigma_max_ms
            ));
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
        if self.pursuit.max_memory_mb == 0 {
            return Err("pursuit.max_memory_mb must be at least 1".into());
        }
        if !(self.pursuit.window_seconds.is_finite() && self.pursuit.window_seconds >= 0.0) {
            return Err(format!(
                "pursuit.window_seconds must be 0 (use max_memory_mb) or a positive number of \
                 seconds, got {}",
                self.pursuit.window_seconds
            ));
        }
        let release: ReleasePolicy = (&self.envelope).into();
        release
            .validate()
            .map_err(|e| format!("[envelope]: {e}"))?;

        // Only when the stage will run. The checks that need a sample rate live in
        // `residual_config`, which the analysis calls before the dictionary is built; these are the
        // ones a settings document can be judged on by itself.
        if self.residual.enabled {
            let erb: ErbBankConfig = (&self.residual.erb).into();
            erb.validate().map_err(|e| e.to_string())?;
            let power: ResidualPowerConfig = (&self.residual.power).into();
            power.validate().map_err(|e| e.to_string())?;
            if !(self.residual.update_ms.is_finite() && self.residual.update_ms > 0.0) {
                return Err(format!(
                    "residual.update_ms must be a positive number of milliseconds, got {}",
                    self.residual.update_ms
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_document_is_the_default_dictionary() {
        let cfg = Config::from_toml("").unwrap();
        // 8 alphas x 3 betas, less 1342*3ms = 4.026 and 2147*3ms = 6.441. No Gaussians.
        assert_eq!(cfg.dictionary.fof.grid().len(), 22);
        assert_eq!(cfg.dictionary_shapes().len(), 22);
        assert!(cfg.dictionary_shapes().iter().all(|s| s.as_fof().is_some()));
        assert_eq!(cfg.pursuit.target_snr_db, MpConfig::default().target_snr_db);
    }

    #[test]
    fn partial_document_keeps_other_defaults() {
        let cfg = Config::from_toml("[pursuit]\nmax_atoms = 12\n").unwrap();
        assert_eq!(cfg.pursuit.max_atoms, 12);
        assert_eq!(cfg.blocks.f_max, BlockConfig::default().f_max);
        assert_eq!(cfg.dictionary.fof.grid().len(), 22);
    }

    #[test]
    fn both_families_build_shapes_fof_first() {
        let cfg = Config::from_toml(
            "[dictionary.gaussian]\nsigmas_ms = [2.0, 10.0]\ncutoff_level = 0.01\n\n\
             [dictionary.fof]\nalphas = [100.0]\nbetas_ms = [1.0]\n\n\
             [envelope]\nfade_level = 0.01\n",
        )
        .unwrap();
        assert!(cfg.validate().is_ok());
        let shapes = cfg.dictionary_shapes();
        assert_eq!(shapes.len(), 3);
        // The release policy reaches the FOF family, and the cutoff the Gaussian one.
        assert_eq!(shapes[0].as_fof().unwrap().fade_level, 0.01);
        assert_eq!(shapes[1], GaussianParams { sigma: 0.002, cutoff_level: 0.01 }.into());
        assert_eq!(shapes[2].as_gaussian().unwrap().sigma, 0.01);
    }

    #[test]
    fn a_gaussian_only_dictionary_is_legal_and_an_empty_one_is_not() {
        let only = Config::from_toml(
            "[dictionary.fof]\nalphas = []\n\n[dictionary.gaussian]\nsigmas_ms = [5.0]\n",
        )
        .unwrap();
        assert!(only.validate().is_ok());
        assert_eq!(only.dictionary_shapes().len(), 1);

        let none = Config::from_toml("[dictionary.fof]\nalphas = []\n").unwrap();
        assert!(none.validate().is_err());
        for bad in [
            "[dictionary.gaussian]\nsigmas_ms = [0.0]\n",
            "[dictionary.gaussian]\nsigmas_ms = [5.0]\ncutoff_level = 1.0\n",
            "[refine]\nsigma_min_ms = 10.0\nsigma_max_ms = 1.0\n",
        ] {
            assert!(Config::from_toml(bad).unwrap().validate().is_err(), "{bad}");
        }
    }

    /// A document from before the families existed fails with a message naming the fix, not with a
    /// bare "unknown field".
    #[test]
    fn the_flat_dictionary_section_is_rejected_with_directions() {
        for doc in [
            "[dictionary]\nalphas = [100.0]\n",
            "[dictionary]\nbetas_ms = [1.0]\n",
            "[dictionary]\nalpha_beta_max = 4.0\n",
        ] {
            let err = Config::from_toml(doc).unwrap_err();
            assert!(err.contains("[dictionary.fof]"), "{doc}: {err}");
        }
    }

    /// The shipped settings documents parse, validate, and hold the families their names promise.
    #[test]
    fn the_shipped_configs_parse() {
        // (name, document, has FOFs, has Gaussians)
        for (name, text, fof, gauss) in [
            ("mp_1", include_str!("../../data/config/mp_1.toml"), true, false),
            ("lux-eterna-1", include_str!("../../data/config/lux-eterna-1.toml"), true, false),
            ("lux-eterna-1-mixed", include_str!("../../data/config/lux-eterna-1-mixed.toml"), true, true),
            ("lux-eterna-1-gaussian", include_str!("../../data/config/lux-eterna-1-gaussian.toml"), false, true),
            ("chopin-nocturne-2", include_str!("../../data/config/chopin-nocturne-2.toml"), true, false),
            ("zyklus-mp-1", include_str!("../../data/config/zyklus-mp-1.toml"), true, false),
        ] {
            let cfg = Config::from_toml(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(cfg.validate().is_ok(), "{name}");
            assert_eq!(!cfg.dictionary.fof.grid().is_empty(), fof, "{name}: FOF family");
            assert_eq!(!cfg.dictionary.gaussian.sigmas_ms.is_empty(), gauss, "{name}: Gaussian family");
        }
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
            "[dictionary.fof]\nalphas = [100.0, 2000.0]\nbetas_ms = [1.0, 5.0]\nalpha_beta_max = 4.0\n",
        )
        .unwrap();
        // 100*0.001, 100*0.005, 2000*0.001 pass; 2000*0.005 = 10 does not.
        assert_eq!(cfg.dictionary.fof.grid().len(), 3);
    }

    #[test]
    fn betas_are_milliseconds_in_the_document_and_seconds_in_the_grid() {
        let cfg =
            Config::from_toml("[dictionary.fof]\nalphas = [100.0]\nbetas_ms = [2.5]\n").unwrap();
        assert_eq!(cfg.dictionary.fof.grid(), vec![(100.0, 0.0025)]);
    }

    #[test]
    fn validation_catches_unusable_settings() {
        let empty = Config::from_toml("[dictionary.fof]\nalphas = []\n").unwrap();
        assert!(empty.validate().is_err());

        let inverted = Config::from_toml("[blocks]\nf_min = 9000.0\nf_max = 100.0\n").unwrap();
        assert!(inverted.validate().is_err());

        let bad_tol = Config::from_toml("[blocks]\ncapture_tolerance = 1.5\n").unwrap();
        assert!(bad_tol.validate().is_err());

        assert!(Config::default().validate().is_ok());
    }

    /// §29.5 from the document's side: 1 ms at 48 kHz is 48 samples, converted here and only here.
    #[test]
    fn residual_settings_resolve_to_samples_and_seconds() {
        let cfg = Config::from_toml(
            "[residual]\nenabled = true\nupdate_ms = 1.0\n\n\
             [residual.erb]\nbands = 32\nmax_freq_hz = 16000.0\n\n\
             [residual.power]\nmode = \"fixed\"\ntau_ms = 4.0\n",
        )
        .unwrap();
        assert!(cfg.validate().is_ok());

        let r = cfg.residual_config(48_000.0).unwrap();
        assert!(r.enabled);
        assert_eq!(r.update_samples, 48);
        assert_eq!(r.erb.bands, 32);
        assert_eq!(r.erb.max_freq_hz, 16_000.0);
        assert_eq!(r.erb.min_freq_hz, 50.0); // untouched
        assert_eq!(r.power.mode, ResidualPowerTimeMode::Fixed);
        assert_eq!(r.power.fixed_tau_seconds, 4e-3);
        assert_eq!(r.power.tau_max_seconds, 10e-3); // untouched

        // Same document, other rates: the sample count follows.
        assert_eq!(cfg.residual_config(96_000.0).unwrap().update_samples, 96);
        assert_eq!(cfg.residual_config(44_100.0).unwrap().update_samples, 44);
    }

    #[test]
    fn residual_is_off_and_silent_by_default() {
        let cfg = Config::from_toml("").unwrap();
        assert!(!cfg.residual.enabled);
        assert!(!cfg.residual_config(48_000.0).unwrap().enabled);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn residual_settings_are_validated_when_enabled() {
        let bad = |doc: &str| {
            let cfg = Config::from_toml(doc).unwrap();
            cfg.validate().is_err() || cfg.residual_config(48_000.0).is_err()
        };
        assert!(bad("[residual]\nenabled = true\nupdate_ms = 0.0\n"));
        // Under one sample at 48 kHz.
        assert!(bad("[residual]\nenabled = true\nupdate_ms = 0.001\n"));
        assert!(bad("[residual]\nenabled = true\n\n[residual.erb]\nbands = 2\n"));
        assert!(bad("[residual]\nenabled = true\n\n[residual.erb]\norder = 0\n"));
        assert!(bad("[residual]\nenabled = true\n\n[residual.erb]\nmin_freq_hz = 9000.0\nmax_freq_hz = 100.0\n"));
        assert!(bad("[residual]\nenabled = true\n\n[residual.erb]\nmax_freq_hz = 40000.0\n"));
        assert!(bad("[residual]\nenabled = true\n\n[residual.power]\ntau_min_ms = 20.0\n"));
        assert!(bad("[residual]\nenabled = true\n\n[residual.power]\ntau_scale = 0.0\n"));

        // Disabled, the same nonsense is nobody's business: the stage will not run.
        let off = Config::from_toml("[residual]\nenabled = false\nupdate_ms = 0.0\n").unwrap();
        assert!(off.validate().is_ok());
    }

    #[test]
    fn residual_enums_use_the_snake_case_the_rest_of_the_document_uses() {
        let doc = Config::default().to_toml();
        for want in [
            "spacing = \"erb_rate\"",
            "filter = \"gammatone\"",
            "normalization = \"unit_noise_power\"",
            "storage = \"f32_linear_power\"",
            "mode = \"bandwidth_relative\"",
        ] {
            assert!(doc.contains(want), "--write-config is missing {want}\n{doc}");
        }
        assert!(Config::from_toml(&doc).is_ok());
    }

    #[test]
    fn round_trips_through_toml() {
        let mut cfg = Config::default();
        cfg.dictionary.gaussian.sigmas_ms = vec![1.5, 6.0];
        let doc = cfg.to_toml();
        assert!(doc.contains("[dictionary.fof]") && doc.contains("[dictionary.gaussian]"), "{doc}");
        let restored = Config::from_toml(&doc).unwrap();
        assert_eq!(cfg.dictionary_shapes(), restored.dictionary_shapes());
        assert_eq!(cfg.refine.sigma_bracket, restored.refine.sigma_bracket);
        assert_eq!(cfg.pursuit.max_atoms, restored.pursuit.max_atoms);
    }
}
