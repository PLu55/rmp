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
//! rfofs scales output by `amp / amax(alpha*beta)` where `amax` is the peak of `E`, so a probe at
//! `amp = 1` comes back with **peak 1.0**: the envelope, peak-normalized. Two consequences:
//!
//! - The private `fof_amax` is not needed. A coefficient fitted against this basis maps *directly*
//!   to [`FofParams::amp`], because rendering at `amp = A` gives exactly `A * basis`.
//! - The `alpha*beta > 10` cliff (where `amax` returns 0.0 and the grain is silent) is detected by
//!   the probe coming back all-zero — more robust than reimplementing the polynomial.
//!
//! # Accuracy note
//!
//! rfofs's carrier uses a degree-9 polynomial on SIMD lanes (error ~1.7e-5) and an LUT on the
//! scalar tail (~7.7e-4), so the probe carries a small per-sample ripple rather than being exactly
//! `E/amax`. This perturbs the selection criterion slightly; it does not affect correctness of the
//! decomposition, because residual energy is measured from the atom actually rendered. Build rfofs
//! with its `std-sin` feature to remove the ripple when diagnosing.

use rfofs::fof::{FofParams, FofPhase, FofState};

/// Envelope-shaping parameters.
///
/// `E(t)` depends only on these — **not** on `f` or `phi`. That independence is what lets a single
/// FFT of an envelope-windowed frame yield correlations against every frequency at once.
#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// rfofs's conventional fade settings, with `fade_dur` scaled to the decay rate.
    ///
    /// A fixed `fade_dur` would make the fade tail several times longer than the atom body for
    /// large `alpha`, inflating that block's FFT length for no representational gain.
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self {
            alpha,
            beta,
            fade_level: 0.001,
            fade_dur: (2.0 / alpha).clamp(1e-3, 10e-3),
        }
    }

    /// `alpha * beta`, which governs how sharply the attack cuts into the decay.
    pub fn alpha_beta(&self) -> f32 {
        self.alpha * self.beta
    }

    fn validate(&self) -> Result<(), FofError> {
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
        Ok(())
    }
}

/// A complete atom: envelope, carrier and placement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomParams {
    /// Onset in samples, relative to the analysed signal's origin. Signed so an atom may start
    /// before the excerpt.
    pub t0: i64,
    /// Carrier frequency, Hz.
    pub f: f32,
    pub env: EnvelopeParams,
    /// Carrier phase, radians. rfofs convention: `sin(phi + 2*pi*f*t/sr)`.
    pub phi: f32,
    /// Linear amplitude. Equals the fitted coefficient against the peak-normalized envelope.
    pub amp: f32,
}

impl AtomParams {
    /// Convert to rfofs parameters for rendering at absolute sample `origin + t0`.
    ///
    /// Panics if the resulting onset is negative — rfofs's `start_sample` is unsigned.
    pub fn to_fof_params(&self, origin: u64) -> FofParams {
        let start = origin as i64 + self.t0;
        assert!(start >= 0, "atom onset {start} precedes the sample clock origin");
        FofParams {
            id: 0,
            start_sample: start as u64,
            f: self.f,
            gliss: 0.0,
            phi: self.phi,
            amp: self.amp,
            alpha: self.env.alpha,
            beta: self.env.beta,
            fade_level: self.env.fade_level,
            fade_dur: self.env.fade_dur,
            azm: 0.0,
            elev: 0.0,
            distance: 0.0,
        }
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

/// A block's envelope, obtained by rendering.
#[derive(Clone, Debug)]
pub struct Envelope {
    pub params: EnvelopeParams,
    pub sample_rate: f32,
    /// The envelope, peak-normalized, truncated to its support. `samples.len() == support_len`.
    pub samples: Vec<f32>,
    /// `sum(E^2)`, accumulated in f64.
    pub energy: f64,
}

impl Envelope {
    /// Render the probe grain and extract the envelope.
    pub fn render(params: EnvelopeParams, sample_rate: f32) -> Result<Self, FofError> {
        params.validate()?;
        if !(sample_rate > 0.0 && sample_rate.is_finite()) {
            return Err(FofError::Invalid("sample_rate must be > 0"));
        }

        let probe = AtomParams {
            t0: 0,
            f: 0.0,
            env: params,
            phi: std::f32::consts::FRAC_PI_2, // carrier == 1 at every sample
            amp: 1.0,
        };

        // Upper bound on the support, plus slack so we can prove the grain died inside the buffer.
        //
        // rfofs clamps `decay_end = decay_end_raw.max(attack_end)`, so a grain whose attack
        // outlasts its decay lives until the attack finishes. Bounding by the decay alone
        // underestimates the support for large `beta`.
        let attack = (params.beta * sample_rate).ceil() as usize;
        let natural = probe.env_natural_samples(sample_rate);
        let fade = (params.fade_dur * sample_rate).ceil() as usize;
        let capacity = attack.max(natural) + fade + 64;

        let mut buf = vec![0.0f32; capacity];
        let mut state = FofState::spawn(probe.to_fof_params(0), sample_rate);
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

        let energy = buf.iter().map(|&s| (s as f64) * (s as f64)).sum();
        Ok(Self {
            params,
            sample_rate,
            samples: buf,
            energy,
        })
    }

    pub fn support_len(&self) -> usize {
        self.samples.len()
    }
}

impl AtomParams {
    /// Attack-plus-decay duration in samples, excluding the fade ramp.
    fn env_natural_samples(&self, sample_rate: f32) -> usize {
        let fp = self.to_fof_params(0);
        fp.natural_duration_samples(sample_rate).ceil() as usize
    }

    /// Render this atom in isolation, starting at index 0 of `buf`.
    ///
    /// `buf` is **overwritten** (rfofs's `fill_block` accumulates, so it is zeroed first). The whole
    /// grain is rendered in a single `fill_block` call: `decay_acc` is a running product carried
    /// across calls, so splitting the render would not be bit-identical.
    pub fn render_into(&self, sample_rate: f32, buf: &mut [f32]) {
        buf.fill(0.0);
        let mut placed = *self;
        placed.t0 = 0;
        let mut state = FofState::spawn(placed.to_fof_params(0), sample_rate);
        state.fill_block(sample_rate, 0, buf);
    }

    /// Render this atom into a fresh buffer sized to its support.
    pub fn render(&self, sample_rate: f32) -> Result<Vec<f32>, FofError> {
        let env = Envelope::render(self.env, sample_rate)?;
        let mut buf = vec![0.0f32; env.support_len()];
        self.render_into(sample_rate, &mut buf);
        Ok(buf)
    }
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
    fn probe_envelope_is_approximately_peak_normalized() {
        // rfofs scales by amp/amax, and amax approximates the peak of E — but it is a *polynomial
        // fit* (exp of a cubic in ln(alpha*beta)), not the exact peak, so the probe lands near 1.0
        // rather than on it. The exact value does not matter: rendering at amp = A gives exactly
        // A times this basis either way, which is what makes fof_amax unnecessary. This only
        // asserts the normalization is happening at all.
        for (alpha, beta) in [(80.0, 0.001), (251.0, 0.003), (1342.0, 0.0003)] {
            let env = Envelope::render(EnvelopeParams::new(alpha, beta), SR).unwrap();
            let peak = env.samples.iter().cloned().fold(0.0f32, f32::max);
            assert!(
                (0.95..1.05).contains(&peak),
                "alpha={alpha} beta={beta}: peak {peak} is not near 1.0"
            );
        }
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
        let base = AtomParams { t0: 0, f: 1000.0, env: env_p, phi: 0.3, amp: 1.0 };
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
            env: EnvelopeParams::new(251.0, 0.002),
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
            env: EnvelopeParams::new(251.0, 0.002),
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
}
