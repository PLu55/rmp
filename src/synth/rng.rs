//! Deterministic per-band excitation.
//!
//! Written out rather than pulled from a crate, for the same reason `residual::pseudo_noise` is:
//! the render has to be reproducible across builds, and a fixture that can change with a dependency
//! bump is not. `rand` is not a dependency of this crate and should not become one for thirty lines
//! of shift-and-xor.
//!
//! **The stream's variance is exactly 1, and the bank calibration depends on it.** A synthesis band
//! is scaled so that `sum_b |H_b|^2 = 1` against *unit-variance* white noise; feed it uniform noise
//! on `[-1, 1)` — variance 1/3 — and the whole reconstruction comes out 4.8 dB low. So the output is
//! uniform on `[-sqrt(3), sqrt(3))`, and [`the_stream_has_unit_variance`] pins it.
//!
//! Seeds come from the master seed, the band index and the channel index, never from band creation
//! order (§12): re-seeding band 7 gives band 7's stream whatever else was built first, so a future
//! parallel or partial render cannot quietly change the output.

/// Uniform white excitation with unit variance.
pub trait NoiseSource {
    fn next_f32(&mut self) -> f32;

    /// Fill a block. The only form the renderer uses — see §30 on band-major blocks.
    fn fill(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = self.next_f32();
        }
    }
}

/// Vigna's splitmix64, used only to expand one seed word into four uncorrelated ones.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++ — 32 bytes of state, no allocation, no dynamic dispatch in the inner loop.
#[derive(Clone, Debug)]
pub struct Xoshiro256pp {
    s: [u64; 4],
}

/// `sqrt(3)`, the scale that takes uniform `[-1, 1)` to unit variance.
const UNIT_VARIANCE_SCALE: f32 = 1.732_050_8;

impl Xoshiro256pp {
    /// The stream for one `(band, channel)` of one master seed.
    ///
    /// The band and channel are multiplied by odd constants before they meet the master seed, and
    /// the four state words then come out of splitmix64. Adjacent band indices therefore do not get
    /// adjacent states — xoshiro recovers from a poor seed quickly, but "quickly" is still several
    /// outputs into a stream that starts at sample zero.
    pub fn seeded(master: u64, band: u32, channel: u32) -> Self {
        let mut state = master
            ^ (band as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (channel as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        let mut s = [0u64; 4];
        for w in &mut s {
            *w = splitmix64(&mut state);
        }
        // An all-zero state is xoshiro's one fixed point; splitmix64 will not produce it, but the
        // generator is only correct if it cannot happen at all.
        if s == [0; 4] {
            s[0] = 0x2545_F491_4F6C_DD1D;
        }
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

impl NoiseSource for Xoshiro256pp {
    /// Uniform on `[-sqrt(3), sqrt(3))`, mean 0, variance 1.
    ///
    /// The top 24 bits give an exact f32 in `[0, 1)` — the mantissa is 24 bits, so every value is
    /// representable and the distribution has no gaps or repeats.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32; // [0, 1)
        (2.0 * u - 1.0) * UNIT_VARIANCE_SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(n: usize, rng: &mut Xoshiro256pp) -> (f64, f64) {
        let (mut sum, mut sq) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let v = rng.next_f32() as f64;
            sum += v;
            sq += v * v;
        }
        (sum / n as f64, sq / n as f64)
    }

    /// §11: zero mean and *known* variance. The variance is the load-bearing one — the bank
    /// calibration assumes exactly 1, and a stream at 1/3 would put the whole render 4.8 dB low.
    #[test]
    fn the_stream_has_unit_variance() {
        let mut rng = Xoshiro256pp::seeded(1, 0, 0);
        let (mean, var) = moments(1 << 20, &mut rng);
        assert!(mean.abs() < 3e-3, "mean {mean}");
        assert!((var - 1.0).abs() < 5e-3, "variance {var}");
    }

    /// The range is exactly the one the variance was computed for.
    #[test]
    fn the_stream_stays_inside_its_range() {
        let mut rng = Xoshiro256pp::seeded(7, 3, 0);
        for _ in 0..100_000 {
            let v = rng.next_f32();
            assert!((-UNIT_VARIANCE_SCALE..UNIT_VARIANCE_SCALE).contains(&v), "{v}");
        }
    }

    /// §12: the same seed gives the same stream, and `fill` is the same stream as `next_f32`.
    #[test]
    fn a_seed_reproduces_its_stream() {
        let mut a = Xoshiro256pp::seeded(0xDEAD_BEEF, 11, 0);
        let mut b = Xoshiro256pp::seeded(0xDEAD_BEEF, 11, 0);
        let want: Vec<f32> = (0..1000).map(|_| a.next_f32()).collect();
        let mut got = vec![0.0f32; 1000];
        b.fill(&mut got);
        assert_eq!(want, got);
    }

    /// §11: an independent stream per band. Adjacent band indices are the case that would fail if
    /// the seeds were merely `master + band`.
    #[test]
    fn each_band_gets_a_different_stream() {
        let n = 1 << 16;
        let draw = |band: u32, ch: u32| {
            let mut r = Xoshiro256pp::seeded(1, band, ch);
            (0..n).map(|_| r.next_f32() as f64).collect::<Vec<_>>()
        };
        let a = draw(0, 0);
        for (band, ch) in [(1u32, 0u32), (2, 0), (47, 0), (0, 1)] {
            let b = draw(band, ch);
            assert_ne!(a, b, "band {band} channel {ch} repeats band 0");
            // Unit-variance streams, so the correlation is the mean product.
            let r: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>() / n as f64;
            assert!(r.abs() < 0.02, "band {band} channel {ch} correlates at {r}");
        }
    }
}
