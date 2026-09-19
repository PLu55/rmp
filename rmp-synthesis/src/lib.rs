//! All synthesis: a book's atoms, and the stochastic residual an ERB analysis measured.
//!
//! `rmp` only analyses; everything that turns a book back into audio lives here and is driven by
//! `rmpsynth`. The atoms render through [`atoms`] — rfofs for a FOF, [`rmp_core::gauss`] for a
//! Gaussian, the same renders the pursuit subtracted — and the residual through the inverse of
//! [`rmp_core::residual`]:
//!
//! ```text
//! Book atoms   -> AtomParams::render (rfofs / gauss) ----------------------------+
//! ResidualBook -> ERB power trajectories -> smoothed band gains                  |
//!              -> independent white-noise sources -> power-complementary ERB bank |
//!              -> stochastic residual  -------------------------------------------+-> mix -> WAV
//! ```
//!
//! Four things shape the residual half of the design:
//!
//! **The book describes its own bank, and this rebuilds it rather than guessing.** Centres,
//! bandwidths, filter order and the measured normalisation gains all come out of the descriptor,
//! and a rebuild that disagrees with any of them is refused. That is the reason `residual::book`
//! stores recomputable quantities in the first place.
//!
//! **`P_b` is a spectral density, not a share of the energy** — and the level policy follows from
//! that one fact. See [`bank`] for the derivation; the short form is that the bands are scaled so
//! `sum_b |H_b|^2 = 1` and driven at `sqrt(P_b)`, and that spec §6's `sqrt(P_b / C_b)` would undo
//! exactly that scaling.
//!
//! **The render is a pure function of the book, the config and the seed.** Bands are summed in a
//! fixed order, the calibration is a deterministic frequency-grid fit rather than a Monte Carlo
//! one, and each band's noise stream is derived from `(seed, band, channel)` rather than from the
//! order the bands were built in. Nothing is parallel, because nothing needs to be.
//!
//! **It is mono, because a book is.** §19's per-channel model waits on a book format that can
//! express more than one channel.

pub mod atoms;
pub mod bank;
pub mod config;
pub mod error;
pub mod gain;
pub mod render;
pub mod rng;

pub use bank::{BankCalibration, SynthesisBand, SynthesisBank};
pub use config::{
    ClippingPolicy, GainSmoothingConfig, GainSmoothingMode, OutputEncoding, RenderConfig,
};
pub use error::RenderError;
pub use gain::BandGainState;
pub use render::{
    load_book, render_full_book, render_residual_book, render_to_file, BandReport, BookInput,
    RenderBookType, RenderReport, RenderRequest, StochasticRenderer,
};
pub use rng::{NoiseSource, Xoshiro256pp};

/// Fixtures shared by the tests in this module tree.
#[cfg(test)]
pub(crate) mod testing {
    use rmp_core::residual::analyze::analyze_residual;
    use rmp_core::residual::book::ResidualBook;
    use rmp_core::residual::config::ResidualAnalysisConfig;

    /// A valid book of the given shape, with every power set by `f(frame, band)`.
    ///
    /// The descriptor comes from a real `analyze_residual` run rather than being written out by
    /// hand: the synthesis bank cross-checks it against the filters it rebuilds, and a hand-written
    /// descriptor would be a second definition of the bank that could drift from the analysis one.
    pub(crate) fn book_with_power(
        sample_rate: f64,
        bands: usize,
        source_samples: usize,
        f: impl Fn(usize, usize) -> f32,
    ) -> ResidualBook {
        let mut cfg = ResidualAnalysisConfig::default();
        cfg.erb.bands = bands;
        let mut book =
            analyze_residual(&vec![0.0f32; source_samples], sample_rate, 0, &cfg).unwrap();
        for k in 0..book.frame_count as usize {
            for b in 0..bands {
                book.power[k * bands + b] = f(k, b);
            }
        }
        book
    }

    pub(crate) fn a_book(sample_rate: f64, bands: usize, source_samples: usize) -> ResidualBook {
        book_with_power(sample_rate, bands, source_samples, |_, _| 0.0)
    }

    /// A full book holding exactly these atoms, and no residual section.
    pub(crate) fn atom_book(sample_rate: f32, atoms: &[rmp_core::fof::AtomParams]) -> rmp_core::book::Book {
        let mut book = rmp_core::book::Book::new(1.0, sample_rate);
        book.selections = atoms
            .iter()
            .enumerate()
            .map(|(id, &atom)| rmp_core::book::Selection {
                id: id as u64,
                atom,
                block: 0,
                onset: 0,
                bin: 0,
                projected_energy: 1.0,
                energy_removed: 1.0,
                residual_energy: 0.0,
                hr_score: None,
                refined: false,
            })
            .collect();
        book
    }
}

/// §45: analysis and synthesis meet in the middle.
///
/// These are the tests that justify the level policy. Everything else checks that the renderer does
/// what it was told; these check that what it was told means what the analysis thought it meant.
/// A sample-by-sample comparison would be meaningless — the waveform is *supposed* to differ — so
/// what is compared is the band powers the analysis would measure again.
#[cfg(test)]
mod integration {
    use rmp_core::residual::analyze::analyze_residual;
    use rmp_core::residual::book::ResidualBook;
    use rmp_core::residual::config::ResidualAnalysisConfig;
    use rmp_core::residual::pseudo_noise;
    use rmp_core::signal::Signal;
    use crate::render::render_residual_book;
    use crate::RenderConfig;

    const SR: f64 = 48_000.0;
    const BANDS: usize = 48;

    fn analyse(x: &[f32]) -> ResidualBook {
        let mut cfg = ResidualAnalysisConfig::default();
        cfg.erb.bands = BANDS;
        analyze_residual(x, SR, 0, &cfg).unwrap()
    }

    /// Mean power per band over the settled part of a book.
    fn mean_powers(book: &ResidualBook, skip_frames: usize) -> Vec<f64> {
        let n = book.frame_count as usize;
        let mut sum = vec![0.0f64; BANDS];
        for k in skip_frames..n {
            for (b, s) in sum.iter_mut().enumerate() {
                *s += book.frame(k)[b] as f64;
            }
        }
        let count = (n - skip_frames) as f64;
        sum.iter().map(|s| s / count).collect()
    }

    fn round_trip(x: &[f32]) -> (ResidualBook, Signal, ResidualBook) {
        let analysed = analyse(x);
        let y = render_residual_book(&analysed, &RenderConfig::default()).unwrap();
        let reanalysed = analyse(&y.samples);
        (analysed, y, reanalysed)
    }

    fn variance(x: &[f32]) -> f64 {
        x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64
    }

    /// §45.A: white noise in, white noise of the same level out.
    ///
    /// **This is the test that fails if the level policy is wrong.** Running the analysis bands
    /// unscaled at `sqrt(P_b)` — spec §6 read with `C_b = 1` — puts the output 20-plus dB high,
    /// because the bands overlap and each reports a density. See [`super::bank`].
    #[test]
    fn white_noise_survives_the_round_trip_at_its_own_level() {
        let x = pseudo_noise(SR as usize);
        let (analysed, y, reanalysed) = round_trip(&x);

        // The bank covers 50 Hz .. 20 kHz of a 24 kHz band, so a little is legitimately lost.
        let ratio = variance(&y.samples) / variance(&x);
        assert!(
            (0.5..1.3).contains(&ratio),
            "output variance is {ratio:.3}x the input's ({:.1} dB)",
            10.0 * ratio.log10()
        );

        // Band for band, off the ends where the bank rolls off.
        let (want, got) = (mean_powers(&analysed, 100), mean_powers(&reanalysed, 100));
        for b in 3..BANDS - 3 {
            let db = 10.0 * (got[b] / want[b]).log10();
            assert!(db.abs() < 1.5, "band {b}: {db:.2} dB ({} vs {})", got[b], want[b]);
        }
    }

    /// §45.B: a colored source keeps its colour, and does not keep its waveform.
    #[test]
    fn colored_noise_keeps_its_spectral_shape_and_loses_its_waveform() {
        // White through a one-pole lowpass: a smooth, strongly tilted spectrum.
        let white = pseudo_noise(SR as usize);
        let mut s = 0.0f32;
        let x: Vec<f32> = white
            .iter()
            .map(|&v| {
                s = 0.9 * s + 0.1 * v;
                s * 3.0
            })
            .collect();

        let (analysed, y, reanalysed) = round_trip(&x);
        let (want, got) = (mean_powers(&analysed, 100), mean_powers(&reanalysed, 100));

        // The source really is tilted — otherwise this tests nothing the white case did not.
        assert!(want[3] / want[BANDS - 4] > 100.0, "source is not colored: {want:?}");

        for b in 3..BANDS - 3 {
            let db = 10.0 * (got[b] / want[b]).log10();
            assert!(db.abs() < 2.0, "band {b}: {db:.2} dB");
        }

        // Same spectrum, different signal. Unit-scaled correlation, so this is a real comparison.
        let dot: f64 = x
            .iter()
            .zip(&y.samples)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum::<f64>()
            / x.len() as f64;
        let r = dot / (variance(&x) * variance(&y.samples)).sqrt();
        assert!(r.abs() < 0.1, "waveform correlates at {r}");
    }

    /// §45.C: a burst comes back as a burst, at the same time and for the same length.
    #[test]
    fn a_noise_burst_keeps_its_time_and_duration() {
        let n = SR as usize / 2;
        let (lo, hi) = (n / 4, n / 4 + 2400); // 50 ms of noise in the middle of silence
        let white = pseudo_noise(n);
        let x: Vec<f32> = white
            .iter()
            .enumerate()
            .map(|(i, &v)| if (lo..hi).contains(&i) { v } else { 0.0 })
            .collect();

        let (_, y, _) = round_trip(&x);

        // Short-time energy in 5 ms windows, and where it lives.
        let w = 240;
        let energy: Vec<f64> = y.samples.chunks(w).map(variance).collect();
        let peak = energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            (lo / w..=hi / w).contains(&peak),
            "burst peaked in window {peak}, expected {}..{}",
            lo / w,
            hi / w
        );

        // Inside the burst against well outside it: the transient is not smeared across the file.
        let inside = variance(&y.samples[lo..hi]);
        let before = variance(&y.samples[..lo - 2400]);
        let after = variance(&y.samples[hi + 4800..]);
        assert!(inside > 1000.0 * before.max(1e-30), "before {before}, inside {inside}");
        assert!(inside > 1000.0 * after.max(1e-30), "after {after}, inside {inside}");
    }

    /// §44.6, from the other end: a different seed is a different noise realisation of the same
    /// spectrum, not a different spectrum.
    ///
    /// The per-band tolerance is loose on purpose. Two independent noise realisations of a band of
    /// width `B` measured over `T` seconds disagree by roughly `sqrt(2/(B*T))` in power — for the
    /// 43 Hz band at 173 Hz over one second that is 1.2 dB of honest sampling error, and a tighter
    /// bound here would be testing the fixture's luck. The wideband figure has no such problem, so
    /// that is where the test is strict.
    #[test]
    fn a_different_seed_keeps_the_spectral_envelope() {
        let x = pseudo_noise(SR as usize);
        let analysed = analyse(&x);
        let render = |seed: u64| {
            let cfg = RenderConfig {
                seed,
                ..Default::default()
            };
            render_residual_book(&analysed, &cfg).unwrap()
        };
        let (a, b) = (render(1), render(2));
        assert_ne!(a.samples, b.samples);

        let db = 10.0 * (variance(&a.samples) / variance(&b.samples)).log10();
        assert!(db.abs() < 0.3, "overall level differs by {db:.3} dB");

        let (pa, pb) = (mean_powers(&analyse(&a.samples), 100), mean_powers(&analyse(&b.samples), 100));
        for i in 3..BANDS - 3 {
            let db = 10.0 * (pa[i] / pb[i]).log10();
            assert!(db.abs() < 3.0, "band {i}: seeds differ by {db:.2} dB");
        }
    }
}
