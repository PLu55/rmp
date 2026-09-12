//! The ERB analysis filters.
//!
//! One band is a cascade of `N` identical complex one-poles — the sampled gammatone, whose impulse
//! response is `C(n+N-1, N-1) p^n` against the continuous-time `t^(N-1) e^(-2 pi b t) e^(j 2 pi f t)`.
//! The band signal is `2*Re(.)` of the cascade output, which is real and carries the
//! negative-frequency image explicitly rather than pretending it is not there.
//!
//! Writing it as a complex cascade rather than a fixed recipe of real biquads is what makes the
//! order a genuine setting: one code path serves every order in [`MIN_ORDER`]..=[`MAX_ORDER`].
//!
//! **The normalisation gain is measured, not derived.** `g_b` comes from rendering the band's own
//! impulse response and summing its squares, so by Parseval `(1/2pi) * integral |H_b|^2 dw` is
//! exactly 1 — unit-variance white noise leaves a normalised band with unit variance. A closed-form
//! pole expression would have to model the `2*Re(.)` image and its cross term, and would be a second
//! definition of the filter to keep in step with the one that actually runs. This follows the rest
//! of the crate, where support lengths come from rendering a probe grain and never from a formula.

use crate::residual::erb::erb_bandwidth_hz;
use crate::residual::error::ResidualAnalysisError;
use crate::residual::{MAX_ORDER, MIN_ORDER};

/// Gammatone bandwidth correction: the `b` in `t^(N-1) e^(-2 pi b ERB(f) t)` that makes the
/// fourth-order gammatone's equivalent rectangular bandwidth equal `ERB(f)`. Patterson's value.
pub const GAMMATONE_B: f64 = 1.019;

/// One ERB analysis band.
///
/// Kept behind this trait so a different bank design — a warped FIR bank, a polyphase design — can
/// replace the gammatone without the analyser noticing.
pub trait AnalysisBand {
    fn reset(&mut self);
    fn process_block(&mut self, input: &[f32], output: &mut [f32]);
}

/// A complex-pole gammatone band with its unit-noise-power gain folded in.
#[derive(Clone, Debug)]
pub struct GammatoneBand {
    /// Centre frequency in Hz, as designed.
    pub center_hz: f64,
    /// `ERB(center_hz)` — the bandwidth the descriptor reports and the power detector reads.
    pub bandwidth_hz: f64,
    /// The measured unit-noise-power gain.
    pub gain: f64,
    order: usize,
    /// The repeated pole, as `(re, im)`.
    pole: (f64, f64),
    /// One complex accumulator per cascade section.
    state: Vec<(f64, f64)>,
}

impl GammatoneBand {
    /// Design a band centred at `center_hz`.
    ///
    /// Fails rather than clamps: a centre above Nyquist is a configuration error, not something to
    /// quietly move.
    pub fn design(
        center_hz: f64,
        sample_rate: f64,
        order: usize,
    ) -> Result<Self, ResidualAnalysisError> {
        if !(sample_rate.is_finite() && sample_rate > 0.0) {
            return Err(ResidualAnalysisError::InvalidSampleRate(sample_rate));
        }
        if !(MIN_ORDER..=MAX_ORDER).contains(&order) {
            return Err(ResidualAnalysisError::UnsupportedFilterOrder(order));
        }
        let nyquist = sample_rate / 2.0;
        if !(center_hz.is_finite() && center_hz > 0.0) || center_hz >= nyquist {
            return Err(ResidualAnalysisError::FrequencyAboveNyquist {
                freq_hz: center_hz,
                nyquist_hz: nyquist,
            });
        }

        let bandwidth_hz = erb_bandwidth_hz(center_hz);
        // Pole radius from the decay rate, angle from the centre frequency.
        let decay = 2.0 * std::f64::consts::PI * GAMMATONE_B * bandwidth_hz / sample_rate;
        let radius = (-decay).exp();
        let theta = 2.0 * std::f64::consts::PI * center_hz / sample_rate;
        if !(radius < 1.0 && radius > 0.0) {
            return Err(ResidualAnalysisError::FilterDesignFailed(format!(
                "pole radius {radius} at {center_hz} Hz is not inside the unit circle"
            )));
        }

        let mut band = Self {
            center_hz,
            bandwidth_hz,
            gain: 1.0,
            order,
            pole: (radius * theta.cos(), radius * theta.sin()),
            state: vec![(0.0, 0.0); order],
        };
        band.gain = band.unit_noise_power_gain(decay)?;
        Ok(band)
    }

    /// `1 / sqrt(sum_n h[n]^2)` for the unnormalised band, by rendering the impulse response.
    ///
    /// The only thing taken from a formula is where to stop. The envelope of `|h[n]|` peaks at
    /// `(N-1)/decay` and then falls as `n^(N-1) r^n`, so running `64/decay` samples past the peak
    /// leaves a tail below 1e-23 of it — far under the f64 resolution of the sum it would join.
    fn unit_noise_power_gain(&mut self, decay: f64) -> Result<f64, ResidualAnalysisError> {
        let n_max = (((self.order - 1) as f64 + 64.0) / decay).ceil() + 16.0;
        if !(n_max.is_finite() && n_max < (1u64 << 24) as f64) {
            return Err(ResidualAnalysisError::FilterDesignFailed(format!(
                "impulse response at {} Hz does not decay within 2^24 samples",
                self.center_hz
            )));
        }

        self.reset();
        let mut sum = 0.0f64;
        for n in 0..n_max as usize {
            let h = self.step(if n == 0 { 1.0 } else { 0.0 });
            sum += h * h;
        }
        self.reset();

        if !(sum.is_finite() && sum > 0.0) {
            return Err(ResidualAnalysisError::FilterDesignFailed(format!(
                "impulse response at {} Hz summed to {sum}",
                self.center_hz
            )));
        }
        Ok(1.0 / sum.sqrt())
    }

    /// One sample through the cascade, before the gain. `2*Re(.)` of the last section.
    #[inline]
    fn step(&mut self, x: f64) -> f64 {
        let (pr, pi) = self.pole;
        let mut vr = x;
        let mut vi = 0.0;
        for s in &mut self.state {
            // s = v + p*s
            let (sr, si) = *s;
            let nr = vr + pr * sr - pi * si;
            let ni = vi + pr * si + pi * sr;
            *s = (nr, ni);
            vr = nr;
            vi = ni;
        }
        2.0 * vr
    }

    /// One sample of the normalised band signal.
    #[inline]
    pub fn process_sample(&mut self, x: f64) -> f64 {
        self.gain * self.step(x)
    }

    /// `|H(e^jw)|` for the normalised band, in closed form.
    ///
    /// `2*Re(h[n])` is `h[n] + conj(h[n])`, and conjugating the impulse response conjugates the
    /// pole, so the real band's transfer function is the sum of the cascade at `p` and at `p*`.
    /// Used by the tests and the diagnostics table only — never in the analysis path.
    pub fn magnitude_response(&self, omega: f64) -> f64 {
        let (zr, zi) = (omega.cos(), -omega.sin()); // z^-1
        let branch = |p: (f64, f64)| {
            // (1 - p z^-1)^-N
            let dr = 1.0 - (p.0 * zr - p.1 * zi);
            let di = -(p.0 * zi + p.1 * zr);
            let (mut ar, mut ai) = (1.0, 0.0);
            for _ in 0..self.order {
                let (nr, ni) = (ar * dr - ai * di, ar * di + ai * dr);
                ar = nr;
                ai = ni;
            }
            // 1 / (ar + j ai)
            let d = ar * ar + ai * ai;
            (ar / d, -ai / d)
        };
        let a = branch(self.pole);
        let b = branch((self.pole.0, -self.pole.1));
        let (re, im) = (a.0 + b.0, a.1 + b.1);
        self.gain * (re * re + im * im).sqrt()
    }
}

impl AnalysisBand for GammatoneBand {
    fn reset(&mut self) {
        self.state.iter_mut().for_each(|s| *s = (0.0, 0.0));
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        for (&x, y) in input.iter().zip(output.iter_mut()) {
            *y = self.process_sample(x as f64) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::erb::center_freqs;
    use crate::residual::pseudo_noise;

    /// §29.3: unit-noise-power really is unit noise power. By Parseval the impulse response's
    /// squared sum *is* the normalised integral, so this is an identity check on the gain — done
    /// against a fresh render rather than the one `design` used.
    #[test]
    fn every_band_integrates_to_unit_power() {
        for &sr in &[44_100.0, 48_000.0, 96_000.0] {
            for &fc in center_freqs(48, 50.0, 18_000.0).unwrap().iter() {
                let mut band = GammatoneBand::design(fc, sr, 4).unwrap();
                let n = 1 << 17;
                let mut sum = 0.0f64;
                for i in 0..n {
                    let h = band.process_sample(if i == 0 { 1.0 } else { 0.0 });
                    sum += h * h;
                }
                assert!((sum - 1.0).abs() < 1e-9, "sr {sr}, fc {fc}: sum h^2 = {sum}");
            }
        }
    }

    /// The same figure again, from the frequency side: the mean of `|H|^2` over the circle.
    #[test]
    fn the_frequency_integral_agrees_with_parseval() {
        let band = GammatoneBand::design(1000.0, 48_000.0, 4).unwrap();
        let n = 1 << 16;
        let mean: f64 = (0..n)
            .map(|k| {
                let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                band.magnitude_response(w).powi(2)
            })
            .sum::<f64>()
            / n as f64;
        assert!((mean - 1.0).abs() < 1e-6, "mean |H|^2 = {mean}");
    }

    /// §29.2: nothing in the configured range produces a non-finite sample, whatever it is fed.
    #[test]
    fn every_band_is_stable_on_every_fixture() {
        let noise = pseudo_noise(4096);
        let fixtures: [Vec<f32>; 4] = [
            std::iter::once(1.0).chain(std::iter::repeat_n(0.0, 4095)).collect(),
            vec![0.0; 4096],
            vec![1.0; 4096],
            noise,
        ];
        for &order in &[1usize, 2, 4, 8] {
            for &fc in center_freqs(24, 50.0, 20_000.0).unwrap().iter() {
                let mut band = GammatoneBand::design(fc, 48_000.0, order).unwrap();
                for fixture in &fixtures {
                    band.reset();
                    let mut out = vec![0.0f32; fixture.len()];
                    band.process_block(fixture, &mut out);
                    assert!(
                        out.iter().all(|v| v.is_finite()),
                        "order {order}, fc {fc}: non-finite output"
                    );
                }
            }
        }
    }

    /// The peak of the response sits at the design frequency.
    #[test]
    fn the_response_peaks_at_the_centre_frequency() {
        let sr = 48_000.0;
        for &fc in &[100.0, 440.0, 1000.0, 5000.0, 15_000.0] {
            let band = GammatoneBand::design(fc, sr, 4).unwrap();
            let n = 200_000;
            let best = (1..n)
                .max_by(|&a, &b| {
                    let m = |k: usize| {
                        band.magnitude_response(std::f64::consts::PI * k as f64 / n as f64)
                    };
                    m(a).partial_cmp(&m(b)).unwrap()
                })
                .unwrap();
            let peak_hz = (sr / 2.0) * best as f64 / n as f64;
            // Within one grid step of the sweep, plus a hundredth of the band's own width.
            let tol = (sr / 2.0) / n as f64 + 0.01 * band.bandwidth_hz;
            assert!((peak_hz - fc).abs() < tol, "fc {fc}: peak at {peak_hz}");
        }
    }

    #[test]
    fn design_rejects_what_it_cannot_build() {
        assert!(GammatoneBand::design(1000.0, 48_000.0, 0).is_err());
        assert!(GammatoneBand::design(1000.0, 48_000.0, MAX_ORDER + 1).is_err());
        assert!(GammatoneBand::design(24_000.0, 48_000.0, 4).is_err());
        assert!(GammatoneBand::design(30_000.0, 48_000.0, 4).is_err());
        assert!(GammatoneBand::design(0.0, 48_000.0, 4).is_err());
        assert!(GammatoneBand::design(1000.0, 0.0, 4).is_err());
    }

    /// `reset` really does return the filter to its designed state.
    #[test]
    fn reset_makes_the_band_repeat_itself() {
        let mut band = GammatoneBand::design(700.0, 48_000.0, 4).unwrap();
        let input = pseudo_noise(1000);
        let mut a = vec![0.0f32; input.len()];
        let mut b = vec![0.0f32; input.len()];
        band.process_block(&input, &mut a);
        band.reset();
        band.process_block(&input, &mut b);
        assert_eq!(a, b);
    }
}
