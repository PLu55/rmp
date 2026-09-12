//! The Gaussian (Gabor) atom.
//!
//! ```text
//! atom[n] = amp * g[n] * sin(phi + omega * n)      g[n] = exp(-(n - h)^2 / (2 s^2))
//! ```
//!
//! with `s = sigma * sr` and `n = 0 .. 2h`, so `t0` is the first sample of the support and the peak
//! sits at `t0 + h`. The phase is referenced to `t0`, exactly as a FOF's is to its onset, so
//! [`crate::fit`]'s local-atom-time rule needs no second case.
//!
//! # rmp owns this definition
//!
//! A FOF's envelope is obtained by rendering through rfofs, because rfofs is the synthesizer that will
//! replay it and a reimplemented formula would drift from its rounding. A Gaussian has no such
//! external definition: this module *is* the definition, for analysis and for resynthesis alike. So
//! the formula is evaluated directly, and the support is a closed form rather than a measurement.
//!
//! # Normalisation and support
//!
//! The peak is exactly 1, so a fitted coefficient maps directly onto `amp` as the peak amplitude —
//! there is no `amax` analogue to keep out of the amplitude path.
//!
//! The support is truncated where `g` falls below `cutoff_level`: `h = floor(s * sqrt(2 ln(1/c)))`,
//! so both end samples are at or above the cutoff and one more sample would be below it. At the
//! default −60 dB that is ±3.72 sigma. The truncation is a hard step of `cutoff_level` relative to
//! the peak, which puts a spectral leakage floor near that level; a FOF avoids the step with its
//! linear release, but for a symmetric window whose tail is already −60 dB the step costs less than
//! lengthening every block's transform would.
//!
//! # Bandwidth
//!
//! `|G(f)|^2 ∝ exp(-4 pi^2 sigma^2 f^2)`, so the full −3 dB width is `sqrt(ln 2) / (pi sigma)`,
//! about `0.265 / sigma` Hz: 5 ms is 53 Hz wide, 40 ms is 6.6 Hz.

use crate::fof::FofError;

/// The release level a Gaussian is truncated at when none is configured: −60 dB, matching
/// [`crate::fof::ReleasePolicy`]'s default `fade_level`.
pub const DEFAULT_CUTOFF_LEVEL: f32 = 0.001;

/// Envelope parameters of a Gaussian atom.
///
/// `deny_unknown_fields` is load-bearing: a book stores [`crate::atom::Shape`] untagged, and it is
/// what stops a FOF envelope from ever being read as a Gaussian one.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianParams {
    /// Standard deviation of the envelope, seconds.
    pub sigma: f32,
    /// Amplitude relative to the peak at which the support is truncated.
    pub cutoff_level: f32,
}

impl GaussianParams {
    /// A Gaussian at the default cutoff.
    pub fn new(sigma: f32) -> Self {
        Self { sigma, cutoff_level: DEFAULT_CUTOFF_LEVEL }
    }

    /// `sigma` in samples, in f64.
    pub fn sigma_samples(&self, sample_rate: f32) -> f64 {
        self.sigma as f64 * sample_rate as f64
    }

    /// Samples from the first sample to the peak. The support is `2 * half_len + 1`.
    pub fn half_len(&self, sample_rate: f32) -> usize {
        let s = self.sigma_samples(sample_rate);
        let reach = (2.0 * (1.0 / self.cutoff_level as f64).ln()).sqrt();
        let h = (s * reach).floor();
        if h.is_finite() && h > 0.0 { h as usize } else { 0 }
    }

    pub fn support_len(&self, sample_rate: f32) -> usize {
        2 * self.half_len(sample_rate) + 1
    }

    /// The full −3 dB bandwidth, Hz.
    pub fn bandwidth_hz(&self) -> f32 {
        (std::f64::consts::LN_2.sqrt() / (std::f64::consts::PI * self.sigma as f64)) as f32
    }

    pub fn validate(&self, sample_rate: f32) -> Result<(), FofError> {
        if !(self.sigma > 0.0 && self.sigma.is_finite()) {
            return Err(FofError::Invalid("gaussian sigma must be > 0 and finite"));
        }
        if !(self.cutoff_level > 0.0 && self.cutoff_level < 1.0) {
            return Err(FofError::Invalid("gaussian cutoff_level must be in (0, 1)"));
        }
        if !(sample_rate > 0.0 && sample_rate.is_finite()) {
            return Err(FofError::Invalid("sample_rate must be > 0"));
        }
        if self.half_len(sample_rate) == 0 {
            return Err(FofError::Invalid("gaussian sigma is too short for a 3-sample support"));
        }
        Ok(())
    }

    /// `g[n]`, in f64.
    #[inline]
    fn value(n: usize, h: usize, two_s2: f64) -> f64 {
        let d = n as f64 - h as f64;
        (-(d * d) / two_s2).exp()
    }

    /// The envelope over its whole support, peak exactly 1.
    pub(crate) fn render_envelope(&self, sample_rate: f32) -> Result<Vec<f32>, FofError> {
        self.validate(sample_rate)?;
        let h = self.half_len(sample_rate);
        let s = self.sigma_samples(sample_rate);
        let two_s2 = 2.0 * s * s;
        Ok((0..2 * h + 1).map(|n| Self::value(n, h, two_s2) as f32).collect())
    }

    /// Render `amp * g[n] * sin(phi + omega n)` into `buf`, overwriting it.
    ///
    /// Evaluated per sample in f64 with an exact `sin`: there is no synthesizer whose carrier this
    /// has to match. Samples past the support are zero; a buffer shorter than the support is
    /// filled as far as it reaches, exactly as a FOF render is.
    pub(crate) fn render_atom_into(&self, f: f32, phi: f32, amp: f32, sample_rate: f32, buf: &mut [f32]) {
        buf.fill(0.0);
        let h = self.half_len(sample_rate);
        let s = self.sigma_samples(sample_rate);
        let two_s2 = 2.0 * s * s;
        let omega = std::f64::consts::TAU * f as f64 / sample_rate as f64;
        let (amp, phi) = (amp as f64, phi as f64);
        let n = buf.len().min(2 * h + 1);
        for (i, y) in buf.iter_mut().enumerate().take(n) {
            *y = (amp * Self::value(i, h, two_s2) * (phi + omega * i as f64).sin()) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// The oracle: the formula written out independently, sample by sample.
    fn analytic(sigma: f64, cutoff: f64, sr: f64) -> Vec<f64> {
        let s = sigma * sr;
        let mut h = 0i64;
        while (-((h + 1) as f64).powi(2) / (2.0 * s * s)).exp() >= cutoff {
            h += 1;
        }
        (-h..=h).map(|d| (-(d as f64).powi(2) / (2.0 * s * s)).exp()).collect()
    }

    #[test]
    fn the_envelope_is_the_formula() {
        for sigma in [0.0005f32, 0.002, 0.0137, 0.05] {
            let p = GaussianParams::new(sigma);
            let got = p.render_envelope(SR).unwrap();
            let want = analytic(sigma as f64, p.cutoff_level as f64, SR as f64);
            assert_eq!(got.len(), want.len(), "sigma={sigma}");
            let err = got.iter().zip(&want).map(|(&g, &w)| (g as f64 - w).abs()).fold(0.0, f64::max);
            assert!(err < 1e-7, "sigma={sigma}: max error {err:.2e}");
        }
    }

    #[test]
    fn the_peak_is_exactly_one_at_the_centre_and_the_envelope_is_symmetric() {
        let p = GaussianParams::new(0.004);
        let g = p.render_envelope(SR).unwrap();
        let h = p.half_len(SR);
        assert_eq!(g[h], 1.0);
        assert_eq!(g.iter().cloned().fold(0.0f32, f32::max), 1.0);
        for d in 1..=h {
            assert_eq!(g[h - d], g[h + d], "d={d}");
        }
    }

    #[test]
    fn the_support_ends_at_the_cutoff() {
        for cutoff in [0.01f32, 0.001, 1e-5] {
            let p = GaussianParams { sigma: 0.003, cutoff_level: cutoff };
            let g = p.render_envelope(SR).unwrap();
            let h = p.half_len(SR);
            let s = p.sigma_samples(SR);
            assert!(g[0] as f64 >= cutoff as f64 * (1.0 - 1e-6), "end sample below the cutoff");
            let beyond = (-((h + 1) as f64).powi(2) / (2.0 * s * s)).exp();
            assert!(beyond < cutoff as f64, "one more sample would still be above the cutoff");
        }
    }

    #[test]
    fn the_atom_is_linear_in_amp_and_overwrites_its_buffer() {
        let p = GaussianParams::new(0.002);
        let n = p.support_len(SR);
        let mut unit = vec![0.0f32; n];
        p.render_atom_into(700.0, 0.3, 1.0, SR, &mut unit);
        for a in [0.25f32, -1.5] {
            let mut scaled = vec![9.0f32; n + 10];
            p.render_atom_into(700.0, 0.3, a, SR, &mut scaled);
            assert!(scaled[n..].iter().all(|&v| v == 0.0), "stale samples past the support");
            let err = unit.iter().zip(&scaled).map(|(&u, &s)| (s - a * u).abs()).fold(0.0, f32::max);
            assert!(err < 1e-6, "amp={a}: {err:.2e}");
        }
    }

    #[test]
    fn the_bandwidth_is_the_half_power_width_of_the_spectrum() {
        // |G(f)|^2 ∝ exp(-4 pi^2 sigma^2 f^2): evaluate it at half the claimed width.
        let p = GaussianParams::new(0.005);
        let half = p.bandwidth_hz() as f64 / 2.0;
        let sigma = p.sigma as f64;
        let power = (-4.0 * std::f64::consts::PI.powi(2) * sigma * sigma * half * half).exp();
        assert!((power - 0.5).abs() < 1e-6, "power at the edge {power}");
    }

    #[test]
    fn unusable_parameters_are_rejected() {
        assert!(GaussianParams::new(0.0).validate(SR).is_err());
        assert!(GaussianParams::new(f32::NAN).validate(SR).is_err());
        assert!(GaussianParams { sigma: 0.01, cutoff_level: 1.0 }.validate(SR).is_err());
        assert!(GaussianParams { sigma: 0.01, cutoff_level: 0.0 }.validate(SR).is_err());
        // Well under a sample: no 3-sample support exists.
        assert!(GaussianParams::new(1e-6).validate(SR).is_err());
        assert!(GaussianParams::new(0.001).validate(SR).is_ok());
    }
}
