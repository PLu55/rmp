//! The per-band gain trajectory.
//!
//! One pole per band: `g[n] = a*g[n-1] + (1-a)*g_t`, with `a = exp(-1/(tau*fs))` — the same form
//! the analysis power detector uses, one level up. It converts the book's staircase of frames into
//! something without a step at every update, and nothing more; §13.1's 1 ms default is short
//! against the analysis detector's own 0.5–10 ms, so this pole shapes the click and not the
//! envelope.
//!
//! **The square root happens here and only here**, at control rate (§38). The book stores power;
//! the trajectory needs amplitude; converting per sample would put a `sqrt` in the inner loop for a
//! value that changes once per `update_samples`.

/// One band's smoothed gain (§31).
#[derive(Clone, Copy, Debug, Default)]
pub struct BandGainState {
    /// The smoothed value, and the state of the pole.
    pub current: f32,
    /// Where it is heading — `sqrt(P_b[k])` of the frame currently in force.
    pub target: f32,
    /// `exp(-1/(tau*fs))`, designed in f64 and stored f32.
    pub coeff: f32,
    /// `1 - coeff`, kept so the inner loop is a multiply-add pair and not a subtract as well.
    one_minus_coeff: f32,
}

impl BandGainState {
    /// Start silent (§32): no pre-roll, and the residual is exactly zero until a frame loads.
    pub fn new(tau_seconds: f64, sample_rate: f64) -> Self {
        let coeff = (-1.0 / (tau_seconds * sample_rate)).exp();
        Self {
            current: 0.0,
            target: 0.0,
            coeff: coeff as f32,
            one_minus_coeff: (1.0 - coeff) as f32,
        }
    }

    /// A book frame becomes active: power becomes amplitude, once.
    #[inline]
    pub fn load_power(&mut self, power: f32) {
        self.target = power.max(0.0).sqrt();
    }

    /// One sample of the trajectory.
    #[inline]
    pub fn step(&mut self) -> f32 {
        self.current = self.coeff * self.current + self.one_minus_coeff * self.target;
        self.current
    }

    pub fn reset(&mut self) {
        self.current = 0.0;
        self.target = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same one-pole reference `residual::power`'s tests use, written out independently.
    fn run(a: f64, targets: &[f64]) -> Vec<f64> {
        let mut g = 0.0;
        targets
            .iter()
            .map(|&t| {
                g = a * g + (1.0 - a) * t;
                g
            })
            .collect()
    }

    /// §13.1: the pole is the one the formula says it is, and `tau` means what a user thinks.
    #[test]
    fn the_trajectory_matches_a_reference_one_pole() {
        let (fs, tau) = (48_000.0, 1e-3);
        let mut g = BandGainState::new(tau, fs);
        g.load_power(0.25); // -> target 0.5
        let n = (tau * fs) as usize;

        let want = run((-1.0 / (tau * fs)).exp(), &vec![0.5; 10 * n]);
        let got: Vec<f64> = (0..10 * n).map(|_| g.step() as f64).collect();
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
        // 1 - 1/e of the way there after exactly tau.
        let at_tau = got[n - 1] / 0.5;
        assert!((at_tau - (1.0 - (-1.0f64).exp())).abs() < 1e-3, "{at_tau}");
    }

    /// §38: the book stores power, the trajectory carries amplitude, and the conversion is the
    /// only square root in the chain.
    #[test]
    fn power_becomes_amplitude_at_load_time() {
        let mut g = BandGainState::new(1e-3, 48_000.0);
        g.load_power(4.0);
        assert_eq!(g.target, 2.0);
        g.load_power(0.0);
        assert_eq!(g.target, 0.0);
    }

    /// A book that somehow carries a negative power must not produce a NaN gain that then spreads
    /// through the whole output buffer.
    #[test]
    fn a_negative_power_clamps_rather_than_producing_nan() {
        let mut g = BandGainState::new(1e-3, 48_000.0);
        g.load_power(-1.0);
        assert_eq!(g.target, 0.0);
        assert!(g.step().is_finite());
    }

    /// §32: silent until something is loaded, and exactly silent — not merely small.
    #[test]
    fn it_starts_and_stays_at_exactly_zero() {
        let mut g = BandGainState::new(2e-3, 48_000.0);
        assert_eq!(g.current, 0.0);
        for _ in 0..1000 {
            assert_eq!(g.step(), 0.0);
        }
    }

    /// A longer tau is a slower approach, at every point of the trajectory.
    #[test]
    fn a_longer_time_constant_is_slower() {
        let mut fast = BandGainState::new(0.5e-3, 48_000.0);
        let mut slow = BandGainState::new(8e-3, 48_000.0);
        fast.load_power(1.0);
        slow.load_power(1.0);
        for _ in 0..200 {
            assert!(fast.step() > slow.step());
        }
    }
}
