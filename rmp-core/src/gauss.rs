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
//!
//! # Rendering is vectorised and still exactly the formula
//!
//! A Gaussian analysis renders an envelope for every `sigma` refinement probes, over supports up to
//! 214,000 samples, and one glibc `exp` per sample was a quarter of its time. Two things take that
//! away without moving a bit of what is rendered:
//!
//! - **Half the support.** `g[h - d]` and `g[h + d]` are the same expression of the same `d^2`, so
//!   each is evaluated once.
//! - **A vectorised `exp`, kept only where it cannot differ.** [`exp_fast`] lands within 4.3e-16
//!   of glibc's — two ULPs, worst over 250,000 exponents. Every rendered value is an f32, so a fast result is
//!   kept only when all of `y * (1 ± 1e-13)` rounds to one f32 — an interval holding the exact
//!   value, so that f32 is the one the exact value rounds to too. Anything nearer a rounding
//!   boundary than that, a few per million samples, is evaluated the slow way.
//!
//! `the_vectorised_envelope_is_the_formula_to_the_bit` and its atom-render twin hold both paths to
//! the per-sample definition across sigmas, cutoffs, carriers and phases.

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

    /// `g[n]`, in f64 — the definition.
    #[inline]
    fn value(n: usize, h: usize, two_s2: f64) -> f64 {
        let d = n as f64 - h as f64;
        (-(d * d) / two_s2).exp()
    }

    /// `g` at distance `d = 0 ..= h` from the peak, approximately: [`exp_fast`] of exactly the
    /// exponent [`Self::value`] takes. Only ever read through [`settled_f32`].
    fn half_envelope_fast(h: usize, two_s2: f64) -> Vec<f64> {
        let exponents: Vec<f64> = (0..=h)
            .map(|d| {
                let d = d as f64;
                -(d * d) / two_s2
            })
            .collect();
        let mut out = vec![0.0; exponents.len()];
        exp_fast(&exponents, &mut out);
        out
    }

    /// The envelope over its whole support, peak exactly 1.
    pub(crate) fn render_envelope(&self, sample_rate: f32) -> Result<Vec<f32>, FofError> {
        self.validate(sample_rate)?;
        let h = self.half_len(sample_rate);
        let s = self.sigma_samples(sample_rate);
        let mut out = vec![0.0f32; 2 * h + 1];
        envelope_f32(h, 2.0 * s * s, &mut out);
        Ok(out)
    }

    /// Render `amp * g[n] * sin(phi + omega n)` into `buf`, overwriting it.
    ///
    /// Defined per sample in f64 with an exact `sin`: there is no synthesizer whose carrier this
    /// has to match. The envelope comes from [`exp_fast`] under the same guard as
    /// [`Self::render_envelope`]'s, applied to the whole product. Samples past the support are
    /// zero; a buffer shorter than the support is filled as far as it reaches, exactly as a FOF
    /// render is.
    pub(crate) fn render_atom_into(&self, f: f32, phi: f32, amp: f32, sample_rate: f32, buf: &mut [f32]) {
        buf.fill(0.0);
        let h = self.half_len(sample_rate);
        let s = self.sigma_samples(sample_rate);
        let two_s2 = 2.0 * s * s;
        let omega = std::f64::consts::TAU * f as f64 / sample_rate as f64;
        let (amp, phi) = (amp as f64, phi as f64);
        let n = buf.len().min(2 * h + 1);
        let half = Self::half_envelope_fast(h, two_s2);
        for (i, y) in buf.iter_mut().enumerate().take(n) {
            let carrier = (phi + omega * i as f64).sin();
            let fast = amp * half[i.abs_diff(h)] * carrier;
            *y = settled_f32(fast)
                .unwrap_or_else(|| (amp * Self::value(i, h, two_s2) * carrier) as f32);
        }
    }
}

/// How far, relative, a fast result may sit from the exact one and still be trusted to round to the
/// same f32. [`exp_fast`] was measured within 4.3e-16 of glibc's `exp`, so this is a margin of about
/// 230; `the_fast_exp_is_within_a_hundredth_of_the_guard` fails long before the margin is gone.
const SETTLE: f64 = 1e-13;

/// `y as f32`, if every value within a relative [`SETTLE`] of `y` rounds to the same f32.
///
/// Rounding to f32 is monotonic, so it is enough that both ends of that interval agree. The exact
/// value lies inside it, so it rounds to the same f32: the result is the exact computation's to the
/// bit. `None` near a rounding boundary, where the caller computes exactly.
#[inline]
fn settled_f32(y: f64) -> Option<f32> {
    let (lo, hi) = ((y * (1.0 - SETTLE)) as f32, (y * (1.0 + SETTLE)) as f32);
    (lo.to_bits() == hi.to_bits()).then_some(lo)
}

/// The whole envelope, `out[h ± d] = exp(-(d * d) / two_s2) as f32`, exactly —
/// [`GaussianParams::value`] at distance `d` from the peak. `out` holds `2h + 1` samples.
///
/// Where AVX2 is available the exponent, the `exp`, the rounding guard and both stores — the
/// distances after the peak and, reversed, the same distances before it — are one loop four
/// distances wide. Only a lane the guard does not settle is computed again the slow way.
fn envelope_f32(h: usize, two_s2: f64, out: &mut [f32]) {
    assert_eq!(out.len(), 2 * h + 1);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected on this machine just now.
        unsafe { envelope_f32_avx2(h, two_s2, out) };
        return;
    }
    for d in 0..=h {
        let g = exact_half_envelope(d, two_s2);
        out[h - d] = g;
        out[h + d] = g;
    }
}

/// The definition [`envelope_f32`] reproduces, one sample.
#[inline]
fn exact_half_envelope(d: usize, two_s2: f64) -> f32 {
    let d = d as f64;
    (-(d * d) / two_s2).exp() as f32
}

/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn envelope_f32_avx2(h: usize, two_s2: f64, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let kernel = ExpKernel::new();
    let two_s2_v = _mm256_set1_pd(two_s2);
    let negate = _mm256_set1_pd(-0.0);
    let four = _mm256_set1_pd(4.0);
    // Distances as exact f64 integers, stepped four at a time.
    let mut d = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);

    // Distances `base ..= base + 3` go to `h + base ..` and, reversed, to `h - base - 3 ..`. The
    // first set writes the peak twice with the same value.
    let whole = (h + 1) / 4 * 4;
    for base in (0..whole).step_by(4) {
        // `-(d * d) / two_s2`, with the negation a sign flip exactly as the scalar one is.
        let x = _mm256_div_pd(_mm256_xor_pd(_mm256_mul_pd(d, d), negate), two_s2_v);
        let (y, inside) = kernel.apply(x);
        let (rounded, settled) = settled_f32_avx2(y);
        let after = &mut out[h + base..h + base + 4];
        // SAFETY: `after` is exactly four f32 long.
        unsafe { _mm_storeu_ps(after.as_mut_ptr(), rounded) };
        let redo = !(settled & inside) & 0xf;
        if redo != 0 {
            for lane in (0..4).filter(|l| redo & (1 << l) != 0) {
                after[lane] = exact_half_envelope(base + lane, two_s2);
            }
        }
        let mut lanes = [0.0f32; 4];
        lanes.copy_from_slice(after);
        lanes.reverse();
        out[h - base - 3..=h - base].copy_from_slice(&lanes);
        d = _mm256_add_pd(d, four);
    }
    for dist in whole..=h {
        let g = exact_half_envelope(dist, two_s2);
        out[h - dist] = g;
        out[h + dist] = g;
    }
}

/// [`settled_f32`] four lanes at a time: the rounded lanes, and a 4-bit mask of those it settles.
/// An unsettled lane's value is meaningless.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn settled_f32_avx2(y: std::arch::x86_64::__m256d) -> (std::arch::x86_64::__m128, i32) {
    use std::arch::x86_64::*;
    let lo = _mm256_cvtpd_ps(_mm256_mul_pd(y, _mm256_set1_pd(1.0 - SETTLE)));
    let hi = _mm256_cvtpd_ps(_mm256_mul_pd(y, _mm256_set1_pd(1.0 + SETTLE)));
    let same = _mm_cmpeq_epi32(_mm_castps_si128(lo), _mm_castps_si128(hi));
    (lo, _mm_movemask_ps(_mm_castsi128_ps(same)))
}

/// Exponents inside which [`exp_fast`] takes its vector path: `2^k` stays a normal double and the
/// reduction `x - k ln 2` is exact.
const EXP_FAST_RANGE: std::ops::RangeInclusive<f64> = -700.0..=700.0;

/// `out[i] ≈ exp(x[i])`, within a ULP of glibc's, four at a time where AVX2 is available.
///
/// `exp(x) = 2^k exp(r)` with `k = round(x / ln 2)`, `|r| <= ln 2 / 2`, and `exp(r)` a degree-13
/// Taylor polynomial, whose truncation is 4e-18 at the ends of that interval, evaluated by Estrin's
/// scheme — 20% faster than Horner's rule here, which chains thirteen multiply-adds per lane. `ln 2` is split so
/// that `k ln2_hi` is exact. Exponents outside [`EXP_FAST_RANGE`], non-finite ones, and the
/// remainder of a short slice go to glibc.
fn exp_fast(x: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected on this machine just now.
        unsafe { exp_fast_avx2(x, out) };
        return;
    }
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v.exp();
    }
}

/// `1/k!` for `k = 0 ..= 13`.
const EXP_TAYLOR: [f64; 14] = [
    1.0,
    1.0,
    0.5,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5_040.0,
    1.0 / 40_320.0,
    1.0 / 362_880.0,
    1.0 / 3_628_800.0,
    1.0 / 39_916_800.0,
    1.0 / 479_001_600.0,
    1.0 / 6_227_020_800.0,
];

/// `ln 2` in two parts, fdlibm's: `LN2_HI` has its low 21 bits clear, so `k * LN2_HI` is exact for
/// any `k` [`EXP_FAST_RANGE`] produces, and `x - k * LN2_HI` is exact by Sterbenz's lemma.
const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);

/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn exp_fast_avx2(x: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::*;
    let kernel = ExpKernel::new();
    let whole = x.len() / 4 * 4;
    for (xs, os) in x[..whole].chunks_exact(4).zip(out[..whole].chunks_exact_mut(4)) {
        // SAFETY: both chunks are exactly four f64 long.
        let (y, inside) = kernel.apply(unsafe { _mm256_loadu_pd(xs.as_ptr()) });
        // SAFETY: as above.
        unsafe { _mm256_storeu_pd(os.as_mut_ptr(), y) };
        let outside = !inside & 0xf;
        if outside != 0 {
            for lane in (0..4).filter(|l| outside & (1 << l) != 0) {
                os[lane] = xs[lane].exp();
            }
        }
    }
    for (o, &v) in out[whole..].iter_mut().zip(&x[whole..]) {
        *o = v.exp();
    }
}

/// The vector `exp`'s constants, set up once per loop.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct ExpKernel {
    lo: std::arch::x86_64::__m256d,
    hi: std::arch::x86_64::__m256d,
    inv_ln2: std::arch::x86_64::__m256d,
    ln2_hi: std::arch::x86_64::__m256d,
    ln2_lo: std::arch::x86_64::__m256d,
    taylor: [std::arch::x86_64::__m256d; 14],
    bias: std::arch::x86_64::__m128i,
}

#[cfg(target_arch = "x86_64")]
impl ExpKernel {
    #[target_feature(enable = "avx2")]
    fn new() -> Self {
        use std::arch::x86_64::*;
        Self {
            lo: _mm256_set1_pd(*EXP_FAST_RANGE.start()),
            hi: _mm256_set1_pd(*EXP_FAST_RANGE.end()),
            inv_ln2: _mm256_set1_pd(std::f64::consts::LN_2.recip()),
            ln2_hi: _mm256_set1_pd(LN2_HI),
            ln2_lo: _mm256_set1_pd(LN2_LO),
            taylor: EXP_TAYLOR.map(|c| _mm256_set1_pd(c)),
            bias: _mm_set1_epi32(1023),
        }
    }

    /// `exp` of each lane, and a 4-bit mask of the lanes inside [`EXP_FAST_RANGE`]. A lane outside
    /// it — or NaN — holds a meaningless finite value the caller must replace.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn apply(&self, v: std::arch::x86_64::__m256d) -> (std::arch::x86_64::__m256d, i32) {
        use std::arch::x86_64::*;
        let inside = _mm256_and_pd(_mm256_cmp_pd(v, self.lo, _CMP_GE_OQ), _mm256_cmp_pd(v, self.hi, _CMP_LE_OQ));
        // Out-of-range lanes are computed at zero, so nothing overflows.
        let v = _mm256_and_pd(v, inside);
        let k = _mm256_round_pd(_mm256_mul_pd(v, self.inv_ln2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let r = _mm256_sub_pd(_mm256_sub_pd(v, _mm256_mul_pd(k, self.ln2_hi)), _mm256_mul_pd(k, self.ln2_lo));
        // Estrin's scheme: the same polynomial as Horner's rule, in a tree four products deep
        // rather than a chain thirteen long. `c[k]` is `1/k!`.
        let c = &self.taylor;
        let pair = |k: usize| _mm256_add_pd(c[k], _mm256_mul_pd(c[k + 1], r));
        let r2 = _mm256_mul_pd(r, r);
        let r4 = _mm256_mul_pd(r2, r2);
        let r8 = _mm256_mul_pd(r4, r4);
        let low = _mm256_add_pd(
            _mm256_add_pd(pair(0), _mm256_mul_pd(pair(2), r2)),
            _mm256_mul_pd(_mm256_add_pd(pair(4), _mm256_mul_pd(pair(6), r2)), r4),
        );
        let high = _mm256_add_pd(
            _mm256_add_pd(pair(8), _mm256_mul_pd(pair(10), r2)),
            _mm256_mul_pd(pair(12), r4),
        );
        let p = _mm256_add_pd(low, _mm256_mul_pd(high, r8));
        // 2^k assembled from its exponent bits, which the range keeps normal.
        let exponent = _mm_add_epi32(_mm256_cvtpd_epi32(k), self.bias);
        let two_k = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_cvtepi32_epi64(exponent), 52));
        (_mm256_mul_pd(p, two_k), _mm256_movemask_pd(inside))
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

    /// The per-sample definition, written out: what both renders must equal to the bit.
    fn envelope_by_definition(p: &GaussianParams) -> Vec<f32> {
        let h = p.half_len(SR);
        let s = p.sigma_samples(SR);
        (0..2 * h + 1).map(|n| GaussianParams::value(n, h, 2.0 * s * s) as f32).collect()
    }

    #[test]
    fn the_vectorised_envelope_is_the_formula_to_the_bit() {
        for sigma in [0.0005f32, 0.00123, 0.006, 0.0137, 0.04, 0.1, 0.25, 0.6] {
            for cutoff in [0.1f32, 0.001, 1e-7] {
                let p = GaussianParams { sigma, cutoff_level: cutoff };
                let got = p.render_envelope(SR).unwrap();
                let want = envelope_by_definition(&p);
                assert_eq!(got.len(), want.len());
                for (n, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g.to_bits(), w.to_bits(), "sigma {sigma}, cutoff {cutoff}, n {n}");
                }
            }
        }
    }

    #[test]
    fn the_vectorised_atom_render_is_the_formula_to_the_bit() {
        for (sigma, f, phi, amp) in [
            (0.0021f32, 440.0f32, 0.0f32, 1.0f32),
            (0.015, 1234.5, -2.9, 0.031),
            (0.1, 90.25, 1.3, -7.5),
            (0.6, 3999.0, 3.1, 1e-4),
        ] {
            let p = GaussianParams::new(sigma);
            let h = p.half_len(SR);
            let s = p.sigma_samples(SR);
            let two_s2 = 2.0 * s * s;
            let omega = std::f64::consts::TAU * f as f64 / SR as f64;
            // Longer than the support, and shorter than it.
            for len in [p.support_len(SR) + 17, h + 3] {
                let mut got = vec![5.0f32; len];
                p.render_atom_into(f, phi, amp, SR, &mut got);
                for (n, &g) in got.iter().enumerate() {
                    let want = if n < 2 * h + 1 {
                        let carrier = (phi as f64 + omega * n as f64).sin();
                        (amp as f64 * GaussianParams::value(n, h, two_s2) * carrier) as f32
                    } else {
                        0.0
                    };
                    assert_eq!(g.to_bits(), want.to_bits(), "sigma {sigma}, len {len}, n {n}");
                }
            }
        }
    }

    /// The guard's margin is only sound if the fast `exp` is far inside it.
    #[test]
    fn the_fast_exp_is_within_a_hundredth_of_the_guard() {
        let mut s = 0x0123_4567_89ab_cdefu64;
        let mut unit = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut x: Vec<f64> = (0..200_000).map(|_| -80.0 * unit()).collect();
        x.extend((0..50_000).map(|_| 1400.0 * unit() - 700.0));
        x.extend([0.0, -0.0, 700.0, -700.0, 700.5, -745.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
        let mut fast = vec![0.0; x.len()];
        exp_fast(&x, &mut fast);
        let mut worst = 0.0f64;
        for (&v, &y) in x.iter().zip(&fast) {
            let exact = v.exp();
            if !exact.is_finite() || exact == 0.0 || !EXP_FAST_RANGE.contains(&v) {
                assert_eq!(y.to_bits(), exact.to_bits(), "outside the fast range, x = {v}");
                continue;
            }
            let rel = ((y - exact) / exact).abs();
            worst = worst.max(rel);
            assert!(rel < SETTLE * 1e-2, "x = {v}: relative error {rel:.2e}");
        }
        println!("worst relative error against glibc: {worst:.2e}");
    }

    #[test]
    fn a_value_near_a_rounding_boundary_is_not_settled() {
        // Midway between two adjacent f32 values, rounding depends on the last bits.
        let a = 0.7f32;
        let midpoint = (a as f64 + f32::from_bits(a.to_bits() + 1) as f64) / 2.0;
        assert_eq!(settled_f32(midpoint), None);
        assert_eq!(settled_f32(-midpoint), None);
        assert_eq!(settled_f32(midpoint * (1.0 + 1e-14)), None);
        // Well inside an f32's rounding interval, the rounding is certain.
        assert_eq!(settled_f32(a as f64), Some(a));
        assert_eq!(settled_f32(-(a as f64)), Some(-a));
        assert_eq!(settled_f32(0.0), Some(0.0));
    }

    /// The vector guard is a second implementation of [`settled_f32`], so it is held to it lane by
    /// lane — on exact rounding midpoints, where settling wrongly is the whole risk.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_vector_guard_settles_exactly_what_the_scalar_one_does() {
        use std::arch::x86_64::*;
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut values = Vec::new();
        for a in [0.7f32, 1.0, 0.001_234, 3.5e-30, -0.25, 12_345.678] {
            let next = f32::from_bits(a.to_bits() + 1) as f64;
            let midpoint = (a as f64 + next) / 2.0;
            values.extend([a as f64, midpoint, midpoint * (1.0 + 1e-14), midpoint * (1.0 + 1e-12), next]);
        }
        values.extend([0.0, -0.0, 1e-300]);
        for lanes in values.chunks(4).filter(|c| c.len() == 4) {
            // SAFETY: AVX2 was detected above, and `lanes` holds exactly four f64.
            let (rounded, mask) = unsafe { settled_f32_avx2(_mm256_loadu_pd(lanes.as_ptr())) };
            let mut got = [0.0f32; 4];
            // SAFETY: `got` holds exactly four f32.
            unsafe { _mm_storeu_ps(got.as_mut_ptr(), rounded) };
            for (lane, &y) in lanes.iter().enumerate() {
                let vector = (mask & (1 << lane) != 0).then_some(got[lane]);
                assert_eq!(vector.map(f32::to_bits), settled_f32(y).map(f32::to_bits), "y = {y:e}");
            }
        }
    }

    /// The guard, proven applied in the vector loop, on the two samples where skipping it shows.
    ///
    /// - **The fast `exp` rounds to the other f32**: about once in a billion samples, found by
    ///   searching a few billion. It rounds the other way under both Horner and Estrin evaluation.
    /// - **The value the loop would store unguarded is wrong**: the loop stores `y * (1 - SETTLE)`
    ///   rounded, which differs from the exact f32 about once in a million samples.
    ///
    /// No ordinary envelope reliably contains either. Each fixture asserts its own premise first,
    /// so a change to the kernel that moves it fails loudly rather than leaving a test of nothing;
    /// a new one then has to be searched for.
    #[test]
    fn the_guard_holds_on_the_samples_where_it_matters() {
        let avx2 = cfg!(target_arch = "x86_64") && std::arch::is_x86_feature_detected!("avx2");
        for (bits, hit, premise) in [
            (0x41c0_f256_cd0b_8b9eu64, 36_076usize, "the fast exp rounds the other way"),
            (0x4138_ddf1_28aa_593a, 1_928, "the unguarded store rounds the other way"),
        ] {
            let two_s2 = f64::from_bits(bits);
            let x = {
                let d = hit as f64;
                -(d * d) / two_s2
            };
            // Four lanes, so the vector path takes it rather than the scalar remainder.
            let mut fast = [0.0; 4];
            exp_fast(&[x; 4], &mut fast);
            let exact = (x.exp() as f32).to_bits();
            if avx2 {
                let wrong = match hit {
                    36_076 => (fast[0] as f32).to_bits(),
                    _ => ((fast[0] * (1.0 - SETTLE)) as f32).to_bits(),
                };
                assert_ne!(wrong, exact, "fixture no longer holds: {premise}");
            }
            let h = hit + 6;
            let mut out = vec![0.0f32; 2 * h + 1];
            envelope_f32(h, two_s2, &mut out);
            for d in 0..=h {
                let want = exact_half_envelope(d, two_s2).to_bits();
                assert_eq!(out[h + d].to_bits(), want, "{premise}: after the peak, d = {d}");
                assert_eq!(out[h - d].to_bits(), want, "{premise}: before the peak, d = {d}");
            }
        }
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
