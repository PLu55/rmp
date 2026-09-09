//! The power-complementary ERB synthesis bank.
//!
//! The bands are the analysis bands — `residual::filter::GammatoneBand`, rebuilt from the centre
//! frequencies the book stores (§8), never regenerated from `min/max/band_count`. Nothing here is a
//! second definition of a gammatone; the only thing this module adds is a scale factor per band.
//!
//! # Why there is a scale factor at all
//!
//! An analysis band is *unit noise power*: `(1/2pi) * integral |H_b|^2 dw = 1`. That makes `P_b` a
//! weighted **average** of the residual's power spectrum over the band — a density estimate — and
//! not a share of its energy. `residual/mod.rs` says the same thing from the other side, and
//! CLAUDE.md records the consequence measured on real material: the summed band powers sit 27 dB
//! above the residual's actual variance.
//!
//! So the reconstruction cannot simply run each band at `sqrt(P_b)` with the analysis filters and
//! add them up. With independent excitation per band the output spectrum is
//!
//! ```text
//! S(w) = sum_b g_b^2 |H_b(w)|^2
//! ```
//!
//! and for that to be the residual's own spectrum we need `g_b = sqrt(P_b)` **together with**
//! `sum_b |H_b(w)|^2 = 1` (§7). A bank of unit-noise-power gammatones is nowhere near that: each
//! band's `|H_b|^2` peaks at roughly `fs / (2 * ERB(f_b))`, so the sum runs from ~1150 at 50 Hz to
//! ~13 at 20 kHz — a 19 dB tilt, not a constant. Calibration is what removes it.
//!
//! Spec §6's `g = sqrt(P_b / C_b)` is the same equation read the other way, and applying both would
//! cancel: dividing by the calibrated band's own noise power `C_b` undoes exactly the scaling that
//! made the bank complementary. §7 is the one to follow; `C_b` is not divided out anywhere.

use crate::residual::book::{ErbFilterKind, ErbNormalization, ErbSpacing, ResidualErbBankDescriptor};
use crate::residual::erb::erb_bandwidth_hz;
use crate::residual::filter::{AnalysisBand, GammatoneBand};
use crate::synth::error::RenderError;

/// One synthesis band (§9).
///
/// A separate trait from [`AnalysisBand`] even though `GammatoneBand` satisfies both: this is the
/// seam a different bank design would replace, and it is the synthesis side's name for it.
pub trait SynthesisBand {
    fn reset(&mut self);
    fn process_sample(&mut self, x: f32) -> f32;
    fn process_block(&mut self, input: &[f32], output: &mut [f32]);
}

impl SynthesisBand for GammatoneBand {
    fn reset(&mut self) {
        AnalysisBand::reset(self);
    }

    fn process_sample(&mut self, x: f32) -> f32 {
        GammatoneBand::process_sample(self, x as f64) as f32
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        AnalysisBand::process_block(self, input, output);
    }
}

/// Frequency points the calibration is fitted and measured on.
///
/// 1.46 Hz apart at 48 kHz, so the narrowest band in the default bank — 24.7 Hz at 50 Hz — is
/// covered by seventeen of them. Deterministic by construction: §10 rules out Monte Carlo, and a
/// fixed grid also means the reported diagnostics are the same numbers every run.
const GRID: usize = 16_384;

/// Sweeps of the multiplicative fit. It converges monotonically and quickly; thirty-two is well
/// past the point where the reported diagnostics stop moving.
const CALIBRATION_ITERS: usize = 32;

/// Relative agreement demanded between the rebuilt bank and what the book says was analysed.
const DESCRIPTOR_TOLERANCE: f64 = 1e-6;

/// How flat `sum_b |H_b|^2` came out (§10).
///
/// Measured over the interior of the bank only. The outermost band has no neighbour beyond it, so
/// the sum necessarily falls away there — that is the bank's edge, not a calibration failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BankCalibration {
    pub min: f64,
    pub max: f64,
    /// RMS of `W(w) - 1` over the measured range.
    pub rms_deviation: f64,
    /// The range the three figures above were measured over, in Hz.
    pub range_hz: (f64, f64),
}

impl BankCalibration {
    /// The worst deviation from unity, in dB — the one number worth printing.
    pub fn worst_db(&self) -> f64 {
        let lo = 10.0 * self.min.max(1e-300).log10();
        let hi = 10.0 * self.max.max(1e-300).log10();
        if lo.abs() > hi.abs() { lo } else { hi }
    }
}

/// The rebuilt, calibrated bank.
#[derive(Clone, Debug)]
pub struct SynthesisBank {
    bands: Vec<GammatoneBand>,
    /// `c_b`, the complementarity correction alone — the unit-noise-power gain is not in here.
    /// Diagnostics only; it is already folded into each band's own gain.
    scale: Vec<f64>,
    calibration: BankCalibration,
}

impl SynthesisBank {
    /// Rebuild the bank a residual book describes, and calibrate it.
    pub fn design(
        descriptor: &ResidualErbBankDescriptor,
        sample_rate: f64,
    ) -> Result<Self, RenderError> {
        // The enums have one variant each today. Matching exhaustively rather than ignoring them is
        // what makes a future second variant a compile error here instead of a silent wrong render.
        match descriptor.filter_kind {
            ErbFilterKind::Gammatone => {}
        }
        match descriptor.spacing {
            ErbSpacing::ErbRate => {}
        }
        match descriptor.normalization {
            ErbNormalization::UnitNoisePower => {}
        }

        let n = descriptor.center_freq_hz.len();
        if n < 2 {
            return Err(RenderError::BankMismatch(format!(
                "a bank of {n} bands cannot be made complementary"
            )));
        }

        let order = descriptor.filter_order as usize;
        let mut bands = Vec::with_capacity(n);
        for (b, &fc) in descriptor.center_freq_hz.iter().enumerate() {
            let band = GammatoneBand::design(fc, sample_rate, order)?;
            check_agrees(
                "bandwidth_hz",
                b,
                fc,
                band.bandwidth_hz,
                descriptor.bandwidth_hz[b],
            )?;
            // The stored gain is the analysis filter's measured unit-noise-power gain. If a
            // rebuild produces a different one, the filter design has moved since the book was
            // written and `P_b` no longer means what this render would assume it means.
            check_agrees(
                "normalization_gain",
                b,
                fc,
                band.gain,
                descriptor.normalization_gain[b],
            )?;
            bands.push(band);
        }

        // A cheap sanity check on the ERB formula itself, independent of the filter: the descriptor
        // says the centres are ERB-spaced, so their own bandwidths must follow the ERB curve.
        for (b, (&fc, &bw)) in descriptor
            .center_freq_hz
            .iter()
            .zip(&descriptor.bandwidth_hz)
            .enumerate()
        {
            check_agrees("erb", b, fc, erb_bandwidth_hz(fc), bw)?;
        }

        let (scale, calibration) = calibrate(&bands, sample_rate);
        for (band, &c) in bands.iter_mut().zip(&scale) {
            band.gain *= c;
        }

        Ok(Self {
            bands,
            scale,
            calibration,
        })
    }

    pub fn len(&self) -> usize {
        self.bands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bands.is_empty()
    }

    pub fn calibration(&self) -> BankCalibration {
        self.calibration
    }

    /// The complementarity correction per band, for the diagnostic table.
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    pub fn band_mut(&mut self, b: usize) -> &mut GammatoneBand {
        &mut self.bands[b]
    }

    pub fn reset(&mut self) {
        for band in &mut self.bands {
            SynthesisBand::reset(band);
        }
    }

    /// `sum_b |H_b(w)|^2` at one frequency, in closed form. Diagnostics and tests only.
    pub fn power_sum(&self, omega: f64) -> f64 {
        self.bands
            .iter()
            .map(|b| b.magnitude_response(omega).powi(2))
            .sum()
    }
}

/// Fail on a rebuilt quantity that does not match what the book recorded.
fn check_agrees(what: &str, band: usize, fc: f64, got: f64, want: f64) -> Result<(), RenderError> {
    let scale = want.abs().max(got.abs()).max(f64::MIN_POSITIVE);
    if (got - want).abs() > DESCRIPTOR_TOLERANCE * scale {
        return Err(RenderError::BankMismatch(format!(
            "band {band} at {fc} Hz: {what} rebuilds as {got}, the book records {want}"
        )));
    }
    Ok(())
}

/// Fit per-band scales so `sum_b c_b^2 |H_b|^2` is as near 1 as a bank of this shape gets (§10).
///
/// This is a non-negative least squares fit of `W(w) = sum_b c_b^2 |H_b(w)|^2` to the constant 1,
/// solved by the standard ISRA multiplicative update
///
/// ```text
/// c_b^2 <- c_b^2 * (sum_i m_b[i]) / (sum_i m_b[i] W[i]),      m_b[i] = |H_b(w_i)|^2
/// ```
///
/// which keeps every scale positive, converges monotonically, and updates every band from the same
/// `W` so the result does not depend on band order. Deterministic throughout: a fixed grid, a fixed
/// iteration count, no Monte Carlo (§10).
///
/// The obvious cheaper rule — divide each band by `W` at its own centre — was tried first and is
/// worse in the way that matters. It drives `W` to exactly 1 *at the centres* and leaves the dips
/// between them untouched: on the default 48-band bank it scallops to -0.55 dB, against -0.30 dB
/// for the fit below (RMS 0.072 against 0.044). Weighting by the band's own response is what lets a
/// band see the gap it is supposed to be filling.
///
/// How flat it comes out is a property of the *bank*, not of the fit. Band count is what decides
/// it: at 48 bands over 50 Hz .. 20 kHz the sum holds to 0.3 dB, at 64 to 0.08 dB — but at 24 bands
/// the centres are 1.7 ERB apart and no choice of scales can fill between them, and the fit
/// scallops by 5 dB. That is what the reported diagnostics are for.
///
/// The fit runs only between the outermost band centres. Beyond them the sum has to roll off —
/// there is no further band to overlap with — and asking the fit to flatten that region would only
/// push the edge bands up trying to reach past the end of the bank.
fn calibrate(bands: &[GammatoneBand], sample_rate: f64) -> (Vec<f64>, BankCalibration) {
    let n = bands.len();
    let omega = |i: usize| std::f64::consts::PI * i as f64 / (GRID - 1) as f64;

    // |H_b|^2 on the grid, once. This is the whole cost of the stage.
    let response: Vec<Vec<f64>> = bands
        .iter()
        .map(|b| (0..GRID).map(|i| b.magnitude_response(omega(i)).powi(2)).collect())
        .collect();

    // The grid point nearest each band's centre, which bounds the fitted range.
    let nyquist = sample_rate / 2.0;
    let centre_index = |b: &GammatoneBand| {
        ((b.center_hz / nyquist) * (GRID - 1) as f64)
            .round()
            .clamp(0.0, (GRID - 1) as f64) as usize
    };
    let first = centre_index(&bands[0]);
    let last = centre_index(&bands[n - 1]);
    let (fit_lo, fit_hi) = (first.min(last), first.max(last));
    let fit = fit_lo..=fit_hi;

    // The ISRA numerator does not depend on the scales, so it is computed once.
    let numerator: Vec<f64> = response.iter().map(|r| r[fit.clone()].iter().sum()).collect();

    let mut c2 = vec![1.0f64; n];
    let mut w = vec![0.0f64; GRID];
    for _ in 0..CALIBRATION_ITERS {
        power_sum(&response, &c2, &mut w);
        for (b, c) in c2.iter_mut().enumerate() {
            let denominator: f64 = response[b][fit.clone()]
                .iter()
                .zip(&w[fit.clone()])
                .map(|(m, wi)| m * wi)
                .sum();
            if denominator > 0.0 && denominator.is_finite() {
                *c *= numerator[b] / denominator;
            }
        }
    }
    power_sum(&response, &c2, &mut w);

    // Measure over the interior: the first and last band have nothing beyond them to overlap with,
    // so the sum rolls off there whatever the scales are.
    let (lo, hi) = if n >= 4 {
        (centre_index(&bands[1]), centre_index(&bands[n - 2]))
    } else {
        (fit_lo, fit_hi)
    };
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    let measured = &w[lo..=hi];
    let min = measured.iter().copied().fold(f64::INFINITY, f64::min);
    let max = measured.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let rms = (measured.iter().map(|v| (v - 1.0).powi(2)).sum::<f64>() / measured.len() as f64).sqrt();

    (
        c2.iter().map(|v| v.sqrt()).collect(),
        BankCalibration {
            min,
            max,
            rms_deviation: rms,
            range_hz: (
                omega(lo) / std::f64::consts::PI * nyquist,
                omega(hi) / std::f64::consts::PI * nyquist,
            ),
        },
    )
}

fn power_sum(response: &[Vec<f64>], c2: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for (r, &c) in response.iter().zip(c2) {
        for (o, &m) in out.iter_mut().zip(r) {
            *o += c * m;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::analyze::analyze_residual;
    use crate::residual::config::ResidualAnalysisConfig;
    use crate::residual::pseudo_noise;

    fn a_descriptor(bands: usize, sample_rate: f64) -> ResidualErbBankDescriptor {
        let mut cfg = ResidualAnalysisConfig::default();
        cfg.erb.bands = bands;
        // One frame of silence is enough: only the descriptor is wanted, and it is built from the
        // config rather than from the signal.
        analyze_residual(&vec![0.0f32; 64], sample_rate, 0, &cfg).unwrap().bank
    }

    /// §10: the calibrated bank really is power-complementary across its interior.
    ///
    /// Without the calibration this sum runs from ~1150 to ~13 across the bank — the 19 dB tilt
    /// that would otherwise land straight in the output.
    #[test]
    fn the_squared_magnitudes_sum_to_one() {
        for &(bands, sr) in &[(48usize, 48_000.0f64), (64, 44_100.0), (96, 96_000.0)] {
            let bank = SynthesisBank::design(&a_descriptor(bands, sr), sr).unwrap();
            let cal = bank.calibration();
            assert!(
                cal.worst_db().abs() < 0.5,
                "{bands} bands at {sr}: W in {} .. {} ({:.2} dB)",
                cal.min,
                cal.max,
                cal.worst_db()
            );
            assert!(cal.rms_deviation < 0.06, "{bands} bands: rms {}", cal.rms_deviation);
        }
    }

    /// And when it cannot, the diagnostics say so rather than the render quietly scalloping.
    ///
    /// 24 bands over 50 Hz .. 20 kHz puts the centres 1.7 ERB apart. No choice of per-band scales
    /// fills the gaps between responses that barely overlap, so `W` dips ~5 dB — a real comb in the
    /// reconstruction, and the reason `--verbose` prints this figure.
    #[test]
    fn a_sparse_bank_cannot_be_made_complementary_and_says_so() {
        let sr = 48_000.0;
        let cal = SynthesisBank::design(&a_descriptor(24, sr), sr).unwrap().calibration();
        assert!(cal.worst_db().abs() > 3.0, "24 bands reported only {:.2} dB", cal.worst_db());
        assert!(cal.rms_deviation > 0.2, "rms {}", cal.rms_deviation);
    }

    /// The uncalibrated bank is not flat, so the previous test is testing something.
    #[test]
    fn an_uncalibrated_bank_is_steeply_tilted() {
        let sr = 48_000.0;
        let d = a_descriptor(48, sr);
        let raw: Vec<GammatoneBand> = d
            .center_freq_hz
            .iter()
            .map(|&fc| GammatoneBand::design(fc, sr, 4).unwrap())
            .collect();
        let at = |hz: f64| {
            raw.iter()
                .map(|b| b.magnitude_response(2.0 * std::f64::consts::PI * hz / sr).powi(2))
                .sum::<f64>()
        };
        let (low, high) = (at(100.0), at(10_000.0));
        assert!(low > 100.0, "sum at 100 Hz is only {low}");
        assert!(low / high > 20.0, "tilt is only {}", low / high);
    }

    /// §8: the stored centres are canonical, not a starting point for regeneration.
    #[test]
    fn the_bank_uses_the_stored_centre_frequencies() {
        let sr = 48_000.0;
        let mut d = a_descriptor(8, sr);
        // Move one centre well away from where the ERB grid would put it, and fix up the two
        // quantities the descriptor cross-check compares against.
        d.center_freq_hz[3] = 3333.0;
        d.bandwidth_hz[3] = erb_bandwidth_hz(3333.0);
        d.normalization_gain[3] = GammatoneBand::design(3333.0, sr, 4).unwrap().gain;

        let bank = SynthesisBank::design(&d, sr).unwrap();
        assert_eq!(bank.bands[3].center_hz, 3333.0);
    }

    /// A descriptor whose stored bandwidth no longer matches the ERB formula this build uses is a
    /// book written by a different definition, and rendering it would silently mean something else.
    #[test]
    fn a_descriptor_that_does_not_match_this_build_is_refused() {
        let sr = 48_000.0;
        let mut d = a_descriptor(8, sr);
        d.bandwidth_hz[2] *= 1.05;
        let err = SynthesisBank::design(&d, sr).unwrap_err().to_string();
        assert!(err.contains("bandwidth_hz"), "{err}");

        let mut d = a_descriptor(8, sr);
        d.normalization_gain[5] *= 2.0;
        let err = SynthesisBank::design(&d, sr).unwrap_err().to_string();
        assert!(err.contains("normalization_gain"), "{err}");
    }

    /// The synthesis trait is the analysis filter, unchanged apart from its gain.
    #[test]
    fn the_band_is_the_analysis_filter() {
        let sr = 48_000.0;
        let mut bank = SynthesisBank::design(&a_descriptor(16, sr), sr).unwrap();
        let input = pseudo_noise(512);

        let mut want = vec![0.0f32; input.len()];
        let mut reference = GammatoneBand::design(bank.bands[4].center_hz, sr, 4).unwrap();
        reference.gain *= bank.scale()[4];
        AnalysisBand::process_block(&mut reference, &input, &mut want);

        let mut got = vec![0.0f32; input.len()];
        SynthesisBand::process_block(bank.band_mut(4), &input, &mut got);
        assert_eq!(got, want);
    }

    /// Resetting really does return the whole bank to its designed state.
    #[test]
    fn reset_makes_the_bank_repeat_itself() {
        let sr = 48_000.0;
        let mut bank = SynthesisBank::design(&a_descriptor(8, sr), sr).unwrap();
        let input = pseudo_noise(300);
        let run = |bank: &mut SynthesisBank| {
            let mut out = vec![0.0f32; input.len()];
            SynthesisBand::process_block(bank.band_mut(2), &input, &mut out);
            out
        };
        let a = run(&mut bank);
        bank.reset();
        assert_eq!(a, run(&mut bank));
    }

    /// The calibration is a property of the bank, not of the run: same descriptor, same numbers.
    #[test]
    fn calibration_is_reproducible() {
        let sr = 48_000.0;
        let d = a_descriptor(48, sr);
        let a = SynthesisBank::design(&d, sr).unwrap();
        let b = SynthesisBank::design(&d, sr).unwrap();
        assert_eq!(a.calibration(), b.calibration());
        assert_eq!(a.scale(), b.scale());
    }
}

