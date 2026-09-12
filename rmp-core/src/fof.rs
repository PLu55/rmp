//! Bridge to `rfofs` — the FOF atom definition.
//!
//! Every atom rmp analyses is rendered by the same code that synthesizes it, so an analysis result
//! replays through rfofs exactly. Nothing here reimplements the FOF math; envelopes and support
//! lengths are obtained by *rendering a probe grain* and inspecting it.
//!
//! # Why rendering rather than a formula
//!
//! rfofs's death sample depends on `ceil(beta_samples)`, on
//! `max(attack_end, ceil(-ln(fade_level)/alpha_per_sample))`, and on the emit-then-increment order
//! inside `fill_block`. A reimplemented formula would agree today and drift silently later.
//!
//! # The probe
//!
//! Rendering with `f = 0, phi = PI/2` makes the carrier `sin(2*pi*0.25) = 1` at every sample, so the
//! render is the envelope alone. `phi = PI/2` is essential — at `phi = 0` the carrier is identically
//! zero and the probe renders silence.
//!
//! rfofs scales output by `amp / amax(alpha*beta)`, where [`rfofs::fof_amax`] approximates the peak
//! of `E`, so a probe at `amp = 1` comes back approximately peak-normalized.
//!
//! **The amplitude mapping does not use `amax`, and must not.** A coefficient fitted against this
//! basis maps *directly* to [`FofParams::amp`], because rendering at `amp = A` gives exactly
//! `A * basis` — the `1/amax` factor is already baked into the basis. Multiplying by `amax`
//! anywhere in the fitting path would double-normalize.
//!
//! `amax` is used here only to reject `alpha*beta > 10` (where it returns 0.0 and the grain renders
//! silent) *before* allocating and rendering a probe.
//!
//! Note `amax` is a polynomial fit rather than the exact peak, so the probe peak lands within about
//! 3.4% of 1.0 across the useful parameter range rather than on it. That is harmless — the fitted
//! coefficient absorbs it — but it means "peak-normalized" is approximate.
//!
//! # Accuracy note
//!
//! rfofs's carrier uses a degree-9 polynomial on SIMD lanes (error ~1.7e-5) and an LUT on the
//! scalar tail (~7.7e-4), so the probe carries a small per-sample ripple rather than being exactly
//! `E/amax`. This perturbs the selection criterion slightly; it does not affect correctness of the
//! decomposition, because residual energy is measured from the atom actually rendered. Build rfofs
//! with its `std-sin` feature to remove the ripple when diagnosing.

use rfofs::fof::{FofParams, FofPhase, FofState};

// The generic atom and envelope live in `atom`, which dispatches to this module for FOFs. Re-exported
// so the many `crate::fof::{AtomParams, Envelope}` paths written before there was a second kind
// keep meaning what they meant.
pub use crate::atom::{AtomParams, Envelope};

/// How the final release is chosen, fixed once when the analyzer is initialized.
///
/// rfofs's release is a linear amplitude ramp to zero, entered where the raw exponential decay
/// reaches `fade_level`. Only its *duration* is a free parameter here, and it is derived from
/// `alpha` rather than being a constant: a fixed duration would make the ramp several times longer
/// than the atom body at large `alpha`, inflating that block's FFT length for no representational
/// gain. Fixing the *policy* — this struct — is what the design requires; fixing the resulting
/// number is not.
///
/// The same policy must be used for analysis and for resynthesis, or the atom subtracted is not the
/// atom the book replays.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReleasePolicy {
    /// Amplitude relative to peak at which the natural fade-out begins.
    pub fade_level: f32,
    /// `fade_dur = fade_dur_scale / alpha`, before clamping.
    pub fade_dur_scale: f32,
    /// Lower clamp on the fade duration, seconds.
    pub fade_dur_min: f32,
    /// Upper clamp on the fade duration, seconds.
    pub fade_dur_max: f32,
}

impl Default for ReleasePolicy {
    fn default() -> Self {
        Self {
            fade_level: 0.001,
            fade_dur_scale: 2.0,
            fade_dur_min: 1e-3,
            fade_dur_max: 10e-3,
        }
    }
}

impl ReleasePolicy {
    pub fn validate(&self) -> Result<(), FofError> {
        if !(self.fade_level > 0.0 && self.fade_level < 1.0) {
            return Err(FofError::Invalid("fade_level must be in (0, 1)"));
        }
        if !(self.fade_dur_scale >= 0.0 && self.fade_dur_scale.is_finite()) {
            return Err(FofError::Invalid("fade_dur_scale must be >= 0 and finite"));
        }
        if !(self.fade_dur_min >= 0.0 && self.fade_dur_min.is_finite()) {
            return Err(FofError::Invalid("fade_dur_min must be >= 0 and finite"));
        }
        if !(self.fade_dur_max >= self.fade_dur_min && self.fade_dur_max.is_finite()) {
            return Err(FofError::Invalid("fade_dur_max must be >= fade_dur_min"));
        }
        Ok(())
    }
}

/// Envelope-shaping parameters.
///
/// `E(t)` depends only on these — **not** on `f` or `phi`. That independence is what lets a single
/// FFT of an envelope-windowed frame yield correlations against every frequency at once.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EnvelopeParams {
    /// Exponential decay coefficient, s^-1. The -3 dB bandwidth is `alpha / PI` Hz.
    pub alpha: f32,
    /// Attack (skirt) duration in seconds, half-cosine.
    pub beta: f32,
    /// Amplitude relative to peak at which the natural fade-out begins.
    pub fade_level: f32,
    /// Duration of the linear fade-out ramp, seconds.
    pub fade_dur: f32,
}

impl EnvelopeParams {
    /// Envelope parameters under the default [`ReleasePolicy`].
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self::with_policy(alpha, beta, &ReleasePolicy::default())
    }

    /// Envelope parameters under an explicit release policy.
    ///
    /// The clamp is written as `min`/`max` rather than `f32::clamp` so that inverted or NaN bounds
    /// cannot panic here — a bad policy is rejected by [`ReleasePolicy::validate`] and by
    /// [`EnvelopeParams::validate`], which is where the error belongs.
    pub fn with_policy(alpha: f32, beta: f32, policy: &ReleasePolicy) -> Self {
        let (lo, hi) = (
            policy.fade_dur_min.min(policy.fade_dur_max),
            policy.fade_dur_min.max(policy.fade_dur_max),
        );
        Self {
            alpha,
            beta,
            fade_level: policy.fade_level,
            fade_dur: (policy.fade_dur_scale / alpha).max(lo).min(hi),
        }
    }

    /// `alpha * beta`, which governs how sharply the attack cuts into the decay.
    pub fn alpha_beta(&self) -> f32 {
        self.alpha * self.beta
    }

    /// rfofs's envelope-peak normalization factor; 0.0 past the `alpha*beta > 10` cliff.
    ///
    /// Useful for screening a dictionary grid before building blocks. It is deliberately *not* part
    /// of the amplitude mapping — see the module docs.
    pub fn amax(&self) -> f32 {
        rfofs::fof_amax(self.alpha, self.beta)
    }

    pub(crate) fn validate(&self) -> Result<(), FofError> {
        if !(self.alpha > 0.0 && self.alpha.is_finite()) {
            return Err(FofError::UnboundedSupport("alpha must be > 0"));
        }
        if !(self.beta >= 0.0 && self.beta.is_finite()) {
            return Err(FofError::Invalid("beta must be >= 0 and finite"));
        }
        if !(self.fade_level > 0.0 && self.fade_level < 1.0) {
            return Err(FofError::UnboundedSupport("fade_level must be in (0, 1)"));
        }
        if !(self.fade_dur >= 0.0 && self.fade_dur.is_finite()) {
            return Err(FofError::Invalid("fade_dur must be >= 0 and finite"));
        }
        // Screen the amax cliff up front rather than discovering it from an all-zero probe.
        if self.amax() == 0.0 {
            return Err(FofError::SilentGrain {
                alpha_beta: self.alpha_beta(),
            });
        }
        Ok(())
    }
}

/// rfofs parameters for one FOF starting at absolute sample `start`.
///
/// Carrier phase follows rfofs's convention, `sin(phi + 2*pi*f*t/sr)` with `t` counted from the
/// onset. Amplitude maps directly: it is the fitted coefficient against the rendered basis.
pub fn fof_params(env: &EnvelopeParams, start: u64, f: f32, phi: f32, amp: f32) -> FofParams {
    FofParams {
        id: 0,
        start_sample: start,
        f,
        gliss: 0.0,
        phi,
        amp,
        alpha: env.alpha,
        beta: env.beta,
        fade_level: env.fade_level,
        fade_dur: env.fade_dur,
        azm: 0.0,
        elev: 0.0,
        distance: 0.0,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FofError {
    /// Parameters describe a grain that never terminates.
    UnboundedSupport(&'static str),
    /// `alpha * beta` past rfofs's `amax` cliff — the grain renders silent.
    SilentGrain { alpha_beta: f32 },
    Invalid(&'static str),
}

impl std::fmt::Display for FofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnboundedSupport(m) => write!(f, "unbounded FOF support: {m}"),
            Self::SilentGrain { alpha_beta } => {
                write!(f, "alpha*beta = {alpha_beta:.3} renders a silent grain")
            }
            Self::Invalid(m) => write!(f, "invalid FOF parameters: {m}"),
        }
    }
}

impl std::error::Error for FofError {}

/// Render the probe grain and return the envelope, truncated to its support.
///
/// The FOF arm of [`Envelope::render`].
pub(crate) fn render_probe(params: EnvelopeParams, sample_rate: f32) -> Result<Vec<f32>, FofError> {
    params.validate()?;
    if !(sample_rate > 0.0 && sample_rate.is_finite()) {
        return Err(FofError::Invalid("sample_rate must be > 0"));
    }

    // carrier == 1 at every sample
    let probe = fof_params(&params, 0, 0.0, std::f32::consts::FRAC_PI_2, 1.0);

    // Upper bound on the support, plus slack so we can prove the grain died inside the buffer.
    //
    // rfofs clamps `decay_end = decay_end_raw.max(attack_end)`, so a grain whose attack
    // outlasts its decay lives until the attack finishes. Bounding by the decay alone
    // underestimates the support for large `beta`.
    let attack = (params.beta * sample_rate).ceil() as usize;
    // Attack-plus-decay duration in samples, excluding the fade ramp.
    let natural = probe.natural_duration_samples(sample_rate).ceil() as usize;
    let fade = (params.fade_dur * sample_rate).ceil() as usize;
    let capacity = attack.max(natural) + fade + 64;

    let mut buf = vec![0.0f32; capacity];
    let mut state = FofState::spawn(probe, sample_rate);
    state.fill_block(sample_rate, 0, &mut buf);

    if state.phase != FofPhase::Dead {
        return Err(FofError::UnboundedSupport(
            "grain outlived its computed support bound",
        ));
    }

    let support_len = match buf.iter().rposition(|&s| s != 0.0) {
        Some(last) => last + 1,
        // amax returned 0.0 (alpha*beta past the cliff) so the whole grain is silent.
        None => return Err(FofError::SilentGrain { alpha_beta: params.alpha_beta() }),
    };
    buf.truncate(support_len);
    Ok(buf)
}

/// Render one FOF in isolation, starting at index 0 of `buf`.
///
/// `buf` is **overwritten** (rfofs's `fill_block` accumulates, so it is zeroed first). The whole
/// grain is rendered in a single `fill_block` call: `decay_acc` is a running product carried
/// across calls, so splitting the render would not be bit-identical.
pub(crate) fn render_atom_into(
    env: &EnvelopeParams,
    f: f32,
    phi: f32,
    amp: f32,
    sample_rate: f32,
    buf: &mut [f32],
) {
    buf.fill(0.0);
    let mut state = FofState::spawn(fof_params(env, 0, f, phi, amp), sample_rate);
    state.fill_block(sample_rate, 0, buf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const SR: f32 = 48_000.0;

    /// Independent analytic envelope, per the FOF definition — the oracle, mirroring the `NaiveRef`
    /// discipline in rfofs's own `src/fof.rs`. Deliberately does not consult rfofs.
    fn analytic_envelope(p: EnvelopeParams, sr: f32, n: usize) -> Vec<f64> {
        let a = (p.alpha / sr) as f64;
        let beta_s = (p.beta * sr) as f64;
        let attack_end = beta_s.ceil();
        let t_f = attack_end.max((-(p.fade_level as f64).ln() / a).ceil());
        let fade_s = (p.fade_dur * sr).ceil() as f64;
        (0..n)
            .map(|i| {
                let t = i as f64;
                if t < attack_end && beta_s > 0.0 {
                    0.5 * (1.0 - (PI * t / beta_s).cos()) * (-a * t).exp()
                } else if t < t_f {
                    (-a * t).exp()
                } else if t < t_f + fade_s {
                    (-a * t_f).exp() * (1.0 - (t - t_f) / fade_s).max(0.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn probe_peak_equals_true_peak_over_amax() {
        // rfofs scales by amp/amax, so the probe's peak must be exactly max(E)/amax. Checking
        // against the analytic envelope and the now-public fof_amax pins rfofs's normalization
        // precisely, rather than asserting a loose band around 1.0.
        for (alpha, beta) in [(80.0, 0.001), (251.0, 0.003), (1342.0, 0.0003), (328.0, 0.01)] {
            let p = EnvelopeParams::new(alpha, beta);
            let env = Envelope::render(p, SR).unwrap();
            let got = env.samples.iter().cloned().fold(0.0f32, f32::max) as f64;

            let want_peak = analytic_envelope(p, SR, env.support_len())
                .into_iter()
                .fold(0.0f64, f64::max);
            let expect = want_peak / p.amax() as f64;

            assert!(
                (got - expect).abs() / expect < 5e-3,
                "alpha={alpha} beta={beta}: probe peak {got:.5} != max(E)/amax {expect:.5}"
            );
        }
    }

    #[test]
    fn amax_is_a_fit_so_the_probe_is_only_approximately_normalized() {
        // Documents why "peak-normalized" is approximate: amax is exp of a cubic in ln(alpha*beta),
        // not the true peak. Measured deviation stays within ~3.4% over the useful range. The
        // fitted coefficient absorbs this, so it costs nothing — but the basis peak is not 1.0.
        let mut worst = 0.0f64;
        for (alpha, beta) in [
            (80.0, 0.001),
            (251.0, 0.003),
            (524.0, 0.001),
            (1342.0, 0.0003),
            (328.0, 0.01),
            (1000.0, 0.004),
        ] {
            let p = EnvelopeParams::new(alpha, beta);
            let env = Envelope::render(p, SR).unwrap();
            let peak = env.samples.iter().cloned().fold(0.0f32, f32::max) as f64;
            worst = worst.max((peak - 1.0).abs());
        }
        assert!(worst < 0.05, "amax fit error grew to {worst:.4}");
        assert!(worst > 0.005, "probe is exactly normalized — is amax still a fit?");
    }

    #[test]
    fn probe_envelope_matches_analytic_shape() {
        for (alpha, beta) in [(80.0, 0.001), (251.0, 0.003), (524.0, 0.001)] {
            let p = EnvelopeParams::new(alpha, beta);
            let env = Envelope::render(p, SR).unwrap();
            let want = analytic_envelope(p, SR, env.support_len());
            let want_peak = want.iter().cloned().fold(0.0f64, f64::max);
            let got_peak = env.samples.iter().cloned().fold(0.0f32, f32::max) as f64;
            assert!(want_peak > 0.0 && got_peak > 0.0);

            // Normalize each by its own peak, so this tests the envelope *shape* and not amax's
            // fit error. Tolerance covers rfofs's LUT carrier error (~7.7e-4) on the scalar tail
            // of each phase-uniform run.
            let err = env
                .samples
                .iter()
                .zip(&want)
                .map(|(&got, &w)| (got as f64 / got_peak - w / want_peak).abs())
                .fold(0.0f64, f64::max);
            assert!(
                err < 5e-3,
                "alpha={alpha} beta={beta}: max envelope shape deviation {err:.2e}"
            );
        }
    }

    #[test]
    fn support_bound_accounts_for_attack_outlasting_decay() {
        // rfofs clamps decay_end to at least attack_end, so a long beta with a fast alpha lives
        // far beyond -ln(fade_level)/alpha. Bounding the probe buffer by the decay alone would
        // truncate the grain and report it as unbounded.
        //
        // The attack outlasts the decay only when alpha*beta > ln(1/fade_level) = 6.908, and the
        // grain goes silent above 10, so this is only reachable in a narrow band. The production
        // grid caps at alpha*beta <= 4, but refinement is allowed up to 8, so the bound must hold.
        let p = EnvelopeParams { alpha: 1000.0, beta: 0.008, fade_level: 0.001, fade_dur: 0.005 };
        assert!(p.alpha_beta() <= 10.0, "must stay inside the amax cliff");
        let attack = (p.beta * SR).ceil() as usize;
        let decay = (-(p.fade_level.ln()) / p.alpha * SR).ceil() as usize;
        assert!(attack > decay, "test premise: attack {attack} should outlast decay {decay}");

        let env = Envelope::render(p, SR).unwrap();
        assert!(env.support_len() >= attack, "support {} truncated", env.support_len());
    }

    #[test]
    fn support_length_shrinks_as_alpha_grows() {
        let mut prev = usize::MAX;
        for alpha in [80.0f32, 251.0, 524.0, 1342.0, 2147.0] {
            let env = Envelope::render(EnvelopeParams::new(alpha, 0.0003), SR).unwrap();
            let len = env.support_len();
            assert!(len < prev, "alpha={alpha}: support {len} did not shrink below {prev}");
            // Support is set by the decay: roughly ln(1/fade_level)*sr/alpha = 6.9*sr/alpha.
            let expect = 6.908 * SR / alpha;
            let ratio = len as f32 / expect;
            assert!(
                (0.8..1.6).contains(&ratio),
                "alpha={alpha}: support {len} vs expected ~{expect:.0}"
            );
            prev = len;
        }
    }

    #[test]
    fn envelope_is_zero_outside_support_and_ends_nonzero() {
        let env = Envelope::render(EnvelopeParams::new(251.0, 0.002), SR).unwrap();
        assert_ne!(*env.samples.last().unwrap(), 0.0, "support_len is too long");
        assert!(env.energy > 0.0);
    }

    #[test]
    fn amplitude_round_trips_through_render() {
        // Rendering at amp = A must give exactly A times the peak-normalized basis, which is the
        // property that makes fof_amax unnecessary.
        let env_p = EnvelopeParams::new(251.0, 0.002);
        let base = AtomParams { t0: 0, f: 1000.0, env: env_p.into(), phi: 0.3, amp: 1.0 };
        let unit = base.render(SR).unwrap();

        for a in [0.25f32, 2.0, -1.5] {
            let scaled = AtomParams { amp: a, ..base }.render(SR).unwrap();
            let err = unit
                .iter()
                .zip(&scaled)
                .map(|(&u, &s)| (s - a * u).abs())
                .fold(0.0f32, f32::max);
            assert!(err < 1e-5, "amp={a}: render is not linear in amp, max err {err:.2e}");
        }
    }

    #[test]
    fn render_is_deterministic() {
        let p = AtomParams {
            t0: 0,
            f: 440.0,
            env: EnvelopeParams::new(251.0, 0.002).into(),
            phi: 1.1,
            amp: 0.7,
        };
        assert_eq!(p.render(SR).unwrap(), p.render(SR).unwrap());
    }

    #[test]
    fn render_into_overwrites_rather_than_accumulates() {
        // rfofs's fill_block uses `+=`, so a dirty buffer would silently double the atom.
        let p = AtomParams {
            t0: 0,
            f: 440.0,
            env: EnvelopeParams::new(251.0, 0.002).into(),
            phi: 0.0,
            amp: 1.0,
        };
        let clean = p.render(SR).unwrap();
        let mut dirty = vec![9.0f32; clean.len()];
        p.render_into(SR, &mut dirty);
        assert_eq!(clean, dirty);
    }

    #[test]
    fn silent_grain_past_the_amax_cliff_is_rejected() {
        // amax returns 0.0 for alpha*beta > 10, so the grain renders as silence.
        let p = EnvelopeParams { alpha: 2000.0, beta: 0.01, fade_level: 0.001, fade_dur: 0.005 };
        assert!(p.alpha_beta() > 10.0);
        assert!(matches!(
            Envelope::render(p, SR),
            Err(FofError::SilentGrain { .. })
        ));
    }

    #[test]
    fn unbounded_parameters_are_rejected() {
        let base = EnvelopeParams::new(251.0, 0.002);
        for bad in [
            EnvelopeParams { alpha: 0.0, ..base },
            EnvelopeParams { alpha: -1.0, ..base },
            EnvelopeParams { fade_level: 0.0, ..base },
            EnvelopeParams { fade_level: 1.0, ..base },
        ] {
            assert!(Envelope::render(bad, SR).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn fade_dur_scales_with_alpha() {
        // A fixed fade_dur would make the tail dwarf the body at high alpha.
        assert!(EnvelopeParams::new(80.0, 0.001).fade_dur > EnvelopeParams::new(2000.0, 0.001).fade_dur);
        assert_eq!(EnvelopeParams::new(80.0, 0.001).fade_dur, 10e-3); // clamped
        assert_eq!(EnvelopeParams::new(5000.0, 0.0001).fade_dur, 1e-3); // clamped
    }

    // ── the release policy ──────────────────────────────────────────────────────────────────────

    #[test]
    fn default_policy_reproduces_the_previously_hardcoded_constants() {
        // Lifting the constants into ReleasePolicy must not move a single envelope, or every
        // measured hop, support length and performance number changes underneath us.
        for (alpha, beta) in [(80.0, 0.001), (251.0, 0.003), (2147.0, 0.0003)] {
            let old = EnvelopeParams {
                alpha,
                beta,
                fade_level: 0.001,
                fade_dur: (2.0f32 / alpha).clamp(1e-3, 10e-3),
            };
            assert_eq!(EnvelopeParams::new(alpha, beta), old);
        }
    }

    #[test]
    fn policy_controls_the_release_and_survives_hostile_bounds() {
        let p = ReleasePolicy {
            fade_level: 0.01,
            fade_dur_scale: 4.0,
            fade_dur_min: 2e-3,
            fade_dur_max: 20e-3,
        };
        let e = EnvelopeParams::with_policy(500.0, 0.001, &p);
        assert_eq!(e.fade_level, 0.01);
        assert_eq!(e.fade_dur, 8e-3); // 4/500, inside the clamps

        // Inverted and non-finite bounds must not panic — they are rejected, not evaluated.
        let inverted = ReleasePolicy { fade_dur_min: 10e-3, fade_dur_max: 1e-3, ..p };
        let _ = EnvelopeParams::with_policy(500.0, 0.001, &inverted);
        assert!(inverted.validate().is_err());
        assert!(ReleasePolicy { fade_level: 0.0, ..p }.validate().is_err());
        assert!(ReleasePolicy { fade_dur_scale: f32::NAN, ..p }.validate().is_err());
        assert!(ReleasePolicy::default().validate().is_ok());
    }

    // ── spec 17.1: envelope shape, asserted on the render rather than on a formula ───────────────

    /// The attack rise factor at `t`, recovered from the render by dividing out `1/amax` and the
    /// exponential decay. It must be `0.5*(1 - cos(pi*t/beta_samples))` up to the attack end and
    /// exactly 1 afterwards.
    fn rise_factor(env: &Envelope, t: usize) -> f64 {
        let p = env.params.as_fof().unwrap();
        let a = (p.alpha / env.sample_rate) as f64;
        env.samples[t] as f64 * p.amax() as f64 * (a * t as f64).exp()
    }

    #[test]
    fn envelope_is_zero_at_onset_for_every_grid_shape() {
        // Not *exactly* zero: the attack rise is `0.5*(1 - cos(pi*t/beta))` evaluated through
        // rfofs's 4096-entry sine LUT, so `cos(0)` comes back about 1.1e-6 short of 1 and the first
        // sample lands a few times 1e-7 either side of zero. That is the LUT's accuracy, not a
        // shape error, and it is six orders below the peak.
        for alpha in [80.0, 205.0, 524.0, 1342.0, 2147.0] {
            for beta in [0.0003, 0.001, 0.003] {
                if alpha * beta > 4.0 {
                    continue;
                }
                let env = Envelope::render(EnvelopeParams::new(alpha, beta), SR).unwrap();
                let peak = env.samples.iter().cloned().fold(0.0f32, f32::max);
                assert!(
                    env.samples[0].abs() < 1e-5 * peak,
                    "alpha={alpha} beta={beta}: onset {} vs peak {peak}",
                    env.samples[0]
                );
            }
        }
    }

    #[test]
    fn attack_reaches_full_amplitude_at_beta_and_holds() {
        // rfofs's beta is an attack DURATION in seconds, so the rise completes at t = beta*sr —
        // the spec's "attack end at beta*tau = pi" under its rad/s reading of beta.
        for (alpha, beta) in [(80.0, 0.003), (251.0, 0.002), (524.0, 0.001)] {
            let env = Envelope::render(EnvelopeParams::new(alpha, beta), SR).unwrap();
            let attack_end = (beta * SR).ceil() as usize;

            assert!(rise_factor(&env, attack_end / 2) - 0.5 < 5e-3);
            for t in attack_end..(attack_end + 8).min(env.support_len()) {
                let r = rise_factor(&env, t);
                assert!(
                    (r - 1.0).abs() < 2e-3,
                    "alpha={alpha} beta={beta} t={t}: rise {r:.6} should be 1 past the attack"
                );
            }
        }
    }

    #[test]
    fn attack_decay_junction_is_continuous() {
        for (alpha, beta) in [(80.0, 0.003), (251.0, 0.002), (524.0, 0.001)] {
            let env = Envelope::render(EnvelopeParams::new(alpha, beta), SR).unwrap();
            let j = (beta * SR).ceil() as usize;
            let step = (env.samples[j] - env.samples[j - 1]).abs() as f64;
            // The one-sample step at the junction must be no larger than the neighbouring steps
            // inside the attack, i.e. the phase change introduces no discontinuity of its own.
            let inside = (env.samples[j - 1] - env.samples[j - 2]).abs() as f64;
            assert!(
                step <= inside.max(1e-6) * 1.5,
                "alpha={alpha} beta={beta}: junction step {step:.3e} vs {inside:.3e}"
            );
        }
    }

    #[test]
    fn decay_log_slope_recovers_alpha() {
        for (alpha, beta) in [(80.0, 0.001), (251.0, 0.003), (839.0, 0.0003)] {
            let p = EnvelopeParams::new(alpha, beta);
            let env = Envelope::render(p, SR).unwrap();
            let t1 = (beta * SR).ceil() as usize + 16;
            let t2 = fade_start(&p) - 16;
            let slope = ((env.samples[t2] as f64).ln() - (env.samples[t1] as f64).ln())
                / (t2 - t1) as f64;
            let got = -slope * SR as f64;
            assert!(
                (got - alpha as f64).abs() / (alpha as f64) < 2e-3,
                "alpha={alpha} beta={beta}: recovered {got:.3}"
            );
        }
    }

    /// Where rfofs enters the linear release: the raw decay reaches `fade_level`, clamped so the
    /// attack always completes first.
    fn fade_start(p: &EnvelopeParams) -> usize {
        let natural = (-(p.fade_level as f64).ln() / (p.alpha as f64 / SR as f64)).ceil() as usize;
        natural.max((p.beta * SR).ceil() as usize)
    }

    #[test]
    fn release_is_exactly_linear_to_zero() {
        for (alpha, beta) in [(80.0, 0.001), (251.0, 0.003), (2147.0, 0.0003)] {
            let p = EnvelopeParams::new(alpha, beta);
            let env = Envelope::render(p, SR).unwrap();
            let t_f = fade_start(&p);
            let n = env.support_len();
            assert!(t_f + 4 < n, "alpha={alpha}: release too short to test");

            // A linear ramp has a constant first difference; check it against the ramp's own slope.
            let step = (env.samples[t_f + 2] - env.samples[t_f + 1]) as f64;
            assert!(step < 0.0, "alpha={alpha}: release must descend");
            for t in t_f + 2..n {
                let d = (env.samples[t] - env.samples[t - 1]) as f64;
                assert!(
                    (d - step).abs() < 1e-6 + step.abs() * 1e-3,
                    "alpha={alpha} t={t}: release step {d:.3e} != {step:.3e}"
                );
            }
            // ...and it lands on zero: one more step past the last sample would cross it.
            assert!(
                env.samples[n - 1] as f64 + step <= 1e-9,
                "alpha={alpha}: release does not reach zero at the end of support"
            );
        }
    }

    #[test]
    fn analysis_and_resynthesis_share_one_envelope() {
        // The atom rmp subtracts is rendered by the same code that will replay the book. Rendering
        // an atom at f=0, phi=PI/2 must reproduce Envelope::render bit for bit.
        let p = EnvelopeParams::new(251.0, 0.002);
        let env = Envelope::render(p, SR).unwrap();
        let atom = AtomParams { t0: 0, f: 0.0, env: p.into(), phi: std::f32::consts::FRAC_PI_2, amp: 1.0 };
        assert_eq!(atom.render(SR).unwrap(), env.samples);
    }
}
