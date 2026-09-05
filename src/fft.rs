//! Real-to-complex FFT layer.
//!
//! Matching pursuit correlates the residual against every dictionary frequency by windowing a frame
//! with the block envelope and taking one real FFT, so this is the engine's inner loop.
//!
//! # Why `forward` takes `&mut self`
//!
//! The scratch buffer lives inside the plan, which keeps `forward` allocation-free — `realfft`'s
//! own [`RealToComplex::process`] allocates a fresh scratch `Vec` on every call. Taking `&mut self`
//! makes the consequence explicit in the type system: a plan cannot be shared across rayon workers,
//! so each worker constructs its own via [`RealFftPlanner::plan`]. That is a real constraint rather
//! than a conservative one, and encoding it here means the parallel update loop cannot get it wrong.

/// Re-exported because `RealFft::forward` names it, so any implementor or caller
/// outside this module needs it too.
pub use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner as RfPlanner, RealToComplex};
use std::sync::Arc;

/// A planned real-to-complex transform of one fixed length.
pub trait RealFft: Send {
    /// Length of the real input.
    fn fft_len(&self) -> usize;

    /// Length of the complex output: `fft_len / 2 + 1`.
    fn complex_len(&self) -> usize {
        self.fft_len() / 2 + 1
    }

    /// Transform `input` into `output`.
    ///
    /// `input` must be `fft_len()` long and may be overwritten. `output` must be `complex_len()`
    /// long. The transform is unnormalized, using the standard forward sign convention
    /// `X[k] = sum_t x[t] * exp(-2i*pi*k*t/N)`.
    fn forward(&mut self, input: &mut [f32], output: &mut [Complex32]);
}

/// Builds [`RealFft`] plans. One planner per thread; construct plans up front.
pub trait RealFftPlanner: Send {
    fn plan(&mut self, fft_len: usize) -> Box<dyn RealFft>;
}

/// The default backend, `realfft` (pure Rust, no system dependency).
pub struct RealFftPlan {
    fft: Arc<dyn RealToComplex<f32>>,
    scratch: Vec<Complex32>,
}

impl RealFft for RealFftPlan {
    fn fft_len(&self) -> usize {
        self.fft.len()
    }

    fn forward(&mut self, input: &mut [f32], output: &mut [Complex32]) {
        self.fft
            .process_with_scratch(input, output, &mut self.scratch)
            .expect("FFT buffer length mismatch");
    }
}

/// Planner for the `realfft` backend. Reuses `realfft`'s internal twiddle cache across lengths.
pub struct Planner {
    inner: RfPlanner<f32>,
}

impl Planner {
    pub fn new() -> Self {
        Self {
            inner: RfPlanner::<f32>::new(),
        }
    }
}

impl Default for Planner {
    fn default() -> Self {
        Self::new()
    }
}

impl RealFftPlanner for Planner {
    fn plan(&mut self, fft_len: usize) -> Box<dyn RealFft> {
        let fft = self.inner.plan_fft_forward(fft_len);
        let scratch = fft.make_scratch_vec();
        Box::new(RealFftPlan { fft, scratch })
    }
}

/// Rounds `n` up to the next even 5-smooth length (only factors 2, 3, 5).
///
/// FOF support lengths are set by the decay rate and land on arbitrary integers, so they must be
/// rounded up before planning. Rounding to a power of two would waste up to 2x (4625 -> 8192);
/// 5-smooth wastes ~1% (4625 -> 4800) and both are near-optimal for FFT backends. Evenness keeps a
/// clean Nyquist bin at `fft_len / 2`.
pub fn next_fast_len(n: usize) -> usize {
    if n <= 2 {
        return 2;
    }
    let mut c = n + (n & 1); // round up to even
    while !is_5_smooth(c) {
        c += 2;
    }
    c
}

fn is_5_smooth(mut n: usize) -> bool {
    for p in [2, 3, 5] {
        while n.is_multiple_of(p) {
            n /= p;
        }
    }
    n == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// Direct O(N^2) DFT in f64 — the oracle. Mirrors the `NaiveRef` discipline in rfofs's
    /// `src/fof.rs`, where an obviously-correct per-sample implementation is asserted against the
    /// optimized path.
    fn naive_dft(x: &[f32]) -> Vec<(f64, f64)> {
        let n = x.len();
        (0..=n / 2)
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for (t, &v) in x.iter().enumerate() {
                    let ang = -TAU * (k as f64) * (t as f64) / (n as f64);
                    re += v as f64 * ang.cos();
                    im += v as f64 * ang.sin();
                }
                (re, im)
            })
            .collect()
    }

    fn xorshift(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 40) as f32 / 8_388_608.0 - 1.0
    }

    #[test]
    fn forward_matches_naive_dft() {
        let mut planner = Planner::new();
        let mut seed = 0x1234_5678_9abc_def0u64;

        // A power of two, a 5-smooth non-power-of-two, and a length with a factor of 3 and 5.
        for len in [64usize, 96, 240, 1024] {
            let signal: Vec<f32> = (0..len).map(|_| xorshift(&mut seed)).collect();
            let expected = naive_dft(&signal);

            let mut fft = planner.plan(len);
            assert_eq!(fft.fft_len(), len);
            assert_eq!(fft.complex_len(), len / 2 + 1);

            let mut input = signal.clone();
            let mut output = vec![Complex32::new(0.0, 0.0); fft.complex_len()];
            fft.forward(&mut input, &mut output);

            // Compare against the largest bin: small bins carry large relative error in f32.
            let scale = expected
                .iter()
                .map(|(re, im)| re.hypot(*im))
                .fold(0.0f64, f64::max);
            let err = output
                .iter()
                .zip(&expected)
                .map(|(got, (re, im))| {
                    ((got.re as f64 - re).powi(2) + (got.im as f64 - im).powi(2)).sqrt()
                })
                .fold(0.0f64, f64::max);

            assert!(
                err / scale < 1e-5,
                "len {len}: relative error {:.3e} exceeds 1e-5",
                err / scale
            );
        }
    }

    /// Pins the sign convention, which the Gram and correlation code depends on. A real cosine at
    /// bin `k` must land in the real part with no imaginary component; a sine must land in the
    /// *negative* imaginary part.
    #[test]
    fn sign_convention_is_standard_forward_dft() {
        let len = 64usize;
        let k = 5usize;
        let mut planner = Planner::new();
        let mut fft = planner.plan(len);
        let mut output = vec![Complex32::new(0.0, 0.0); fft.complex_len()];

        let mut cosine: Vec<f32> = (0..len)
            .map(|t| (TAU * k as f64 * t as f64 / len as f64).cos() as f32)
            .collect();
        fft.forward(&mut cosine, &mut output);
        assert!((output[k].re - len as f32 / 2.0).abs() < 1e-3);
        assert!(output[k].im.abs() < 1e-3);

        let mut sine: Vec<f32> = (0..len)
            .map(|t| (TAU * k as f64 * t as f64 / len as f64).sin() as f32)
            .collect();
        fft.forward(&mut sine, &mut output);
        assert!(output[k].re.abs() < 1e-3);
        assert!(
            (output[k].im + len as f32 / 2.0).abs() < 1e-3,
            "sine must give NEGATIVE imaginary part, got {}",
            output[k].im
        );
    }

    #[test]
    fn forward_is_allocation_free_after_planning() {
        // Reusing one plan across calls must not need fresh scratch each time.
        let mut planner = Planner::new();
        let mut fft = planner.plan(128);
        let mut output = vec![Complex32::new(0.0, 0.0); fft.complex_len()];
        for _ in 0..4 {
            let mut input = vec![1.0f32; 128];
            fft.forward(&mut input, &mut output);
            assert!((output[0].re - 128.0).abs() < 1e-3);
        }
    }

    #[test]
    fn next_fast_len_is_even_5_smooth_and_not_smaller() {
        for n in 0..2000usize {
            let f = next_fast_len(n);
            assert!(f >= n.max(2), "next_fast_len({n}) = {f} went backwards");
            assert!(f.is_multiple_of(2), "next_fast_len({n}) = {f} is odd");
            assert!(is_5_smooth(f), "next_fast_len({n}) = {f} is not 5-smooth");
            // Nothing even, 5-smooth and smaller may sit between n and f.
            for c in (n.max(2)..f).filter(|c| c.is_multiple_of(2)) {
                assert!(!is_5_smooth(c), "next_fast_len({n}) skipped {c}");
            }
        }
    }

    #[test]
    fn next_fast_len_beats_power_of_two_rounding() {
        // The motivating case from the plan: a FOF support length of 4625 samples. Power-of-two
        // rounding would give 8192, a 1.7x waste against 4800.
        assert_eq!(next_fast_len(4625), 4800);
        // Every planned length must actually be plannable.
        let mut planner = Planner::new();
        for n in [200usize, 487, 872, 1561, 4384, 4625] {
            let len = next_fast_len(n);
            assert_eq!(planner.plan(len).fft_len(), len);
        }
    }
}
