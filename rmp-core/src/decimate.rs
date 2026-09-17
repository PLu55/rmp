//! A low-passed, decimated copy of the residual, for correlating the longest FOF blocks at a
//! fraction of the rate — `[blocks] decimate`.
//!
//! A block's frame is correlated by one FFT of `r[onset..] * E`, `fft_len` points long, but only the
//! bins up to `f_max` are ever read. For an `alpha = 1` block that is 337,500 points to read about a
//! twelfth of the spectrum. So the residual is low-passed once, below `sample_rate / (2D)`, and kept
//! every `D`th sample; a decimated block reads `r_D[onset/D + m] * D * E[mD]` and transforms
//! `fft_len / D` points. Bin `k` is still frequency `k * sample_rate / fft_len`, so the block's bin
//! range and Gram tables are unchanged.
//!
//! Filtering each frame instead would cost more than the transform it shortens — the filter is
//! tens of taps per output sample — so the filter runs once over the signal and then only over the
//! stretch each subtracted atom changed, recomputed rather than updated, so nothing drifts.
//!
//! # How close it is
//!
//! The result is an approximation, and it changes which atoms are selected. What makes it close is
//! that a long block's envelope is narrow in frequency: `(h * r) E` and `h * (r E)` differ at an
//! in-band bin only through residual content in the stopband, weighted by the envelope's spectrum
//! thousands of hertz from its centre. Measured on every frame of the twelve decimated blocks over
//! 10 s of piano at `D = 6`: the best bin agreed in all 406 frames, and the energy there was within
//! 6.8e-4 (p99 2.3e-4). On white noise, the worst case for a low-pass filter, 8e-4, except a frame
//! with 344 samples of signal left at 6e-3. The error floor near 1e-4 does not move with the
//! stopband — 60, 90 and 110 dB give the same p99 — so it is the envelope's short attack aliasing
//! when sampled every `D`th sample, not the filter; 90 dB is kept for a smaller median.

use std::f64::consts::PI;

/// Stopband attenuation, dB.
const ATTENUATION_DB: f64 = 90.0;

/// Clearance, Hz, above `f_max` before the passband ends, and below the first alias of `f_max`
/// before the stopband begins.
const GUARD_HZ: f64 = 150.0;

/// The narrowest transition band a factor may leave, Hz, which bounds the filter's length.
const MIN_TRANSITION_HZ: f64 = 800.0;

/// A zero-phase low-pass FIR and the factor it decimates by.
#[derive(Clone, Debug)]
pub struct Decimator {
    pub factor: usize,
    /// `h[j - half]` for `j = 0 ..= 2 * half`, summing to one.
    pub taps: Vec<f32>,
    pub half: usize,
    /// Passband edge and stopband edge, Hz, as designed.
    pub f_pass: f64,
    pub f_stop: f64,
    pub attenuation_db: f64,
}

impl Decimator {
    /// The largest factor that keeps every alias of `[0, f_max]` in the stopband with a usable
    /// transition band, or `None` if even 2 does not.
    pub fn factor_for(sample_rate: f32, f_max: f32) -> Option<usize> {
        Self::edges(sample_rate as f64, f_max as f64).map(|(d, _, _)| d)
    }

    fn edges(fs: f64, f_max: f64) -> Option<(usize, f64, f64)> {
        let f_pass = f_max + GUARD_HZ;
        // 5-smooth only: a block's transform length is 5-smooth and has to be divisible by it.
        [8usize, 6, 5, 4, 3, 2].into_iter().find_map(|d| {
            let f_stop = fs / d as f64 - f_max - GUARD_HZ;
            (f_stop - f_pass >= MIN_TRANSITION_HZ).then_some((d, f_pass, f_stop))
        })
    }

    /// A Kaiser-windowed sinc with its cutoff midway between the edges.
    pub fn design(sample_rate: f32, f_max: f32) -> Option<Self> {
        let fs = sample_rate as f64;
        let (factor, f_pass, f_stop) = Self::edges(fs, f_max as f64)?;
        let a = ATTENUATION_DB;
        let beta = if a > 50.0 {
            0.1102 * (a - 8.7)
        } else if a >= 21.0 {
            0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
        } else {
            0.0
        };
        let width = 2.0 * PI * (f_stop - f_pass) / fs;
        let half = (((a - 8.0) / (2.285 * width)).ceil() as usize).div_ceil(2);
        let cutoff = (f_pass + f_stop) / 2.0 / fs;

        let raw: Vec<f64> = (0..=2 * half)
            .map(|j| {
                let n = j as f64 - half as f64;
                let sinc = if n == 0.0 {
                    2.0 * cutoff
                } else {
                    (2.0 * PI * cutoff * n).sin() / (PI * n)
                };
                let r = n / half as f64;
                sinc * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / bessel_i0(beta)
            })
            .collect();
        let sum: f64 = raw.iter().sum();
        let taps = raw.iter().map(|&h| (h / sum) as f32).collect();
        Some(Self { factor, taps, half, f_pass, f_stop, attenuation_db: a })
    }

    /// Length of the decimated signal for a residual of `len` samples.
    pub fn output_len(&self, len: usize) -> usize {
        len.div_ceil(self.factor)
    }

    /// The whole decimated residual.
    pub fn apply(&self, residual: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.output_len(residual.len())];
        let n = out.len();
        self.update(residual, &mut out, 0, n);
        out
    }

    /// The decimated samples that read any of residual samples `[start, end)`.
    pub fn affected(&self, start: usize, end: usize, out_len: usize) -> (usize, usize) {
        let lo = start.saturating_sub(self.half).div_ceil(self.factor);
        let hi = ((end + self.half).saturating_sub(1) / self.factor + 1).min(out_len);
        (lo.min(hi), hi)
    }

    /// Recompute `out[m]` for `m` in `[lo, hi)` from `residual`, read as zero outside it.
    pub fn update(&self, residual: &[f32], out: &mut [f32], lo: usize, hi: usize) {
        let len = residual.len();
        for (m, o) in out.iter_mut().enumerate().take(hi).skip(lo) {
            let centre = m * self.factor;
            *o = if centre >= self.half && centre + self.half < len {
                dot(&self.taps, &residual[centre - self.half..=centre + self.half])
            } else {
                let mut acc = 0.0f32;
                for (j, &h) in self.taps.iter().enumerate() {
                    if let Some(i) = (centre + j).checked_sub(self.half)
                        && i < len
                    {
                        acc += h * residual[i];
                    }
                }
                acc
            };
        }
    }

    /// The filter's (real, zero-phase) response at `f` Hz.
    pub fn response(&self, f: f64, sample_rate: f32) -> f64 {
        let w = 2.0 * PI * f / sample_rate as f64;
        self.taps
            .iter()
            .enumerate()
            .map(|(j, &h)| h as f64 * (w * (j as f64 - self.half as f64)).cos())
            .sum()
    }
}

/// A dot product in eight lanes, so it vectorizes.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let whole = a.len() / 8 * 8;
    for (ca, cb) in a[..whole].chunks_exact(8).zip(b[..whole].chunks_exact(8)) {
        for k in 0..8 {
            lanes[k] += ca[k] * cb[k];
        }
    }
    let mut acc: f32 = lanes.iter().sum();
    for (x, y) in a[whole..].iter().zip(&b[whole..]) {
        acc += x * y;
    }
    acc
}

/// The zeroth-order modified Bessel function of the first kind, by its power series.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, half) = (1.0, 1.0, x / 2.0);
    for k in 1..64 {
        term *= half / k as f64;
        let t = term * term;
        sum += t;
        if t < sum * 1e-17 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    #[test]
    fn the_factor_leaves_every_alias_in_the_stopband() {
        assert_eq!(Decimator::factor_for(SR, 3000.0), Some(6));
        assert_eq!(Decimator::factor_for(SR, 5000.0), Some(4));
        assert_eq!(Decimator::factor_for(SR, 10_000.0), Some(2));
        assert_eq!(Decimator::factor_for(SR, 20_000.0), None);
        for f_max in [1000.0f32, 3000.0, 5000.0, 10_000.0] {
            let d = Decimator::factor_for(SR, f_max).unwrap();
            let f_stop = SR as f64 / d as f64 - f_max as f64 - GUARD_HZ;
            assert!(f_stop - (f_max as f64 + GUARD_HZ) >= MIN_TRANSITION_HZ);
        }
    }

    #[test]
    fn the_filter_meets_its_specification() {
        for f_max in [3000.0f32, 5000.0] {
            let dec = Decimator::design(SR, f_max).unwrap();
            let ripple = 10f64.powf(-dec.attenuation_db / 20.0) * 4.0;
            let mut f = 0.0;
            while f <= dec.f_pass {
                let h = dec.response(f, SR);
                assert!((h - 1.0).abs() < ripple, "f_max {f_max}: passband {f} Hz gain {h}");
                f += 10.0;
            }
            let floor = 10f64.powf(-(dec.attenuation_db - 3.0) / 20.0);
            let mut f = dec.f_stop;
            while f <= SR as f64 / 2.0 {
                let h = dec.response(f, SR).abs();
                assert!(h < floor, "f_max {f_max}: stopband {f} Hz gain {h:.2e}");
                f += 10.0;
            }
        }
    }

    #[test]
    fn a_partial_update_equals_filtering_everything_again() {
        let dec = Decimator::design(SR, 3000.0).unwrap();
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut residual: Vec<f32> = (0..20_000)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / 8_388_608.0 - 1.0
            })
            .collect();
        let mut low = dec.apply(&residual);
        for (start, end) in [(0usize, 50usize), (7_001, 9_333), (19_900, 20_000)] {
            for x in &mut residual[start..end] {
                *x *= -0.5;
            }
            let (lo, hi) = dec.affected(start, end, low.len());
            dec.update(&residual, &mut low, lo, hi);
            let fresh = dec.apply(&residual);
            assert_eq!(low.len(), fresh.len());
            for (m, (a, b)) in low.iter().zip(&fresh).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "after [{start}, {end}), m = {m}");
            }
        }
    }
}
