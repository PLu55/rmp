//! Turning a residual into a residual book.
//!
//! Band-major, one pass per band (spec §22): the filter state, the detector state and the
//! coefficient all sit in registers while a band walks the whole residual, and the only writes are
//! into that band's own column. Nothing is shared, so the bands go out to rayon as they are.
//!
//! **The result is bit-identical at any thread count.** Each band's arithmetic is a fixed sequence
//! that no other band can interleave with, and the columns are gathered in band order by an indexed
//! `collect`. That is the same standard `mp::refresh_frames` and `tfmap` are held to, and
//! `the_parallel_bank_matches_a_serial_reference` is what keeps it honest.

use rayon::prelude::*;

use crate::residual::book::{
    ResidualBook, ResidualErbBankDescriptor, ResidualPowerDescriptor, RESIDUAL_BOOK_VERSION,
};
use crate::residual::config::ResidualAnalysisConfig;
use crate::residual::erb::erb_bandwidth_hz;
use crate::residual::error::ResidualAnalysisError;
use crate::residual::filter::GammatoneBand;
use crate::residual::power::detector_coeffs;

/// Analyse one channel of residual into a fixed-rate ERB power book.
///
/// `residual` is a single channel — `rmp` downmixes on read, so that is what the pursuit leaves
/// behind. Nothing here downmixes anything: a caller with more than one channel decides what it
/// wants and calls this once per channel.
///
/// `start_sample` is the residual's offset from the file's origin, so a book analysed under
/// `--start` still says which excerpt it describes.
pub fn analyze_residual(
    residual: &[f32],
    sample_rate: f64,
    start_sample: u64,
    cfg: &ResidualAnalysisConfig,
) -> Result<ResidualBook, ResidualAnalysisError> {
    cfg.validate(sample_rate)?;

    let bands = cfg.erb.bands;
    let order = cfg.erb.filter_order;
    let nu = cfg.update_samples;
    let centers = cfg.erb.center_freqs()?;

    // Bandwidth follows from the centre alone, so the detector coefficients are known before a
    // single filter is designed — and are computed once, never inside a loop.
    let bandwidths: Vec<f64> = centers.iter().map(|&f| erb_bandwidth_hz(f)).collect();
    let taus = cfg.power.taus(&bandwidths);
    let coeffs = detector_coeffs(sample_rate, &taus);

    let frame_count = residual.len().div_ceil(nu.max(1));

    // One design-and-run pass per band. The column is allocated once per band and never grows.
    let columns: Result<Vec<(f64, Vec<f32>)>, ResidualAnalysisError> = (0..bands)
        .into_par_iter()
        .map(|b| {
            let mut band = GammatoneBand::design(centers[b], sample_rate, order)?;
            let a = coeffs[b];
            let one_minus_a = 1.0 - a;
            let gain = band.gain;

            let mut column = vec![0.0f32; frame_count];
            let mut p = 0.0f64;

            // Chunking by the update period is what makes the frame convention structural: the
            // first sample of chunk k *is* sample k*Nu, and the frame is written immediately after
            // it has been through the detector.
            for (k, chunk) in residual.chunks(nu).enumerate() {
                let mut it = chunk.iter();
                if let Some(&x) = it.next() {
                    let y = band.process_sample(x as f64);
                    p = a * p + one_minus_a * y * y;
                    column[k] = p as f32;
                }
                for &x in it {
                    let y = band.process_sample(x as f64);
                    p = a * p + one_minus_a * y * y;
                }
            }
            Ok((gain, column))
        })
        .collect();
    let columns = columns?;

    // Each column is dropped as it is transposed rather than after the loop, so the band-major
    // columns and the frame-major matrix are never both fully resident — the peak is one matrix
    // plus the columns still to be read, not two matrices.
    let mut power = vec![0.0f32; frame_count * bands];
    let gains: Vec<f64> = columns.iter().map(|(g, _)| *g).collect();
    for (b, (_, column)) in columns.into_iter().enumerate() {
        for (k, v) in column.into_iter().enumerate() {
            power[k * bands + b] = v;
        }
    }

    let book = ResidualBook {
        version: RESIDUAL_BOOK_VERSION,
        sample_rate,
        start_sample,
        source_samples: residual.len() as u64,
        update_samples: nu as u32,
        band_count: bands as u32,
        frame_count: frame_count as u64,
        storage: cfg.storage,
        power,
        bank: ResidualErbBankDescriptor {
            band_count: bands as u32,
            min_freq_hz: cfg.erb.min_freq_hz,
            max_freq_hz: cfg.erb.max_freq_hz,
            filter_order: order as u32,
            spacing: cfg.erb.spacing,
            filter_kind: cfg.erb.filter_kind,
            normalization: cfg.erb.normalization,
            center_freq_hz: centers,
            bandwidth_hz: bandwidths,
            normalization_gain: gains,
            power_detector: ResidualPowerDescriptor {
                mode: cfg.power.mode,
                tau_seconds: taus,
            },
        },
    };
    debug_assert!(book.validate().is_ok(), "{:?}", book.validate());
    Ok(book)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual::config::ErbBankConfig;
    use crate::residual::power::{ResidualPowerConfig, ResidualPowerTimeMode};
    use crate::residual::pseudo_noise;

    const SR: f64 = 48_000.0;

    fn cfg(bands: usize, min_hz: f64, max_hz: f64, power: ResidualPowerConfig) -> ResidualAnalysisConfig {
        ResidualAnalysisConfig {
            enabled: true,
            update_samples: 48,
            erb: ErbBankConfig {
                bands,
                min_freq_hz: min_hz,
                max_freq_hz: max_hz,
                ..Default::default()
            },
            power,
            ..Default::default()
        }
    }

    fn fixed_tau(ms: f64) -> ResidualPowerConfig {
        ResidualPowerConfig {
            mode: ResidualPowerTimeMode::Fixed,
            fixed_tau_seconds: ms * 1e-3,
            ..Default::default()
        }
    }

    /// Mean power per band over the last `frac` of the book — past the detector's startup rise.
    fn tail_means(book: &ResidualBook, frac: f64) -> Vec<f64> {
        let n = book.frame_count as usize;
        let from = ((1.0 - frac) * n as f64) as usize;
        let b = book.band_count as usize;
        (0..b)
            .map(|i| {
                (from..n).map(|k| book.frame(k)[i] as f64).sum::<f64>() / (n - from) as f64
            })
            .collect()
    }

    /// §29.5: milliseconds become samples exactly once, and the frames land where §24 says.
    #[test]
    fn frames_sit_on_the_update_grid() {
        // The conversion itself, as the spec writes it.
        let update_samples = ((1.0f64 * 0.001) * SR).round() as usize;
        assert_eq!(update_samples, 48);

        let mut r = vec![0.0f32; 200];
        r[0] = 1.0;
        r[48] = 1.0;
        r[96] = 1.0;
        let book = analyze_residual(&r, SR, 1000, &cfg(8, 200.0, 8000.0, fixed_tau(2.0))).unwrap();

        assert_eq!(book.frame_count, 200_usize.div_ceil(48) as u64); // 5
        assert_eq!(book.update_samples, 48);
        assert_eq!(book.source_samples, 200);
        for k in 0..book.frame_count as usize {
            assert_eq!(book.frame_sample(k), 1000 + 48 * k as u64);
        }
        // A frame is written *after* its own sample, so the impulse at 0 is already in frame 0.
        assert!(book.frame(0).iter().all(|&p| p > 0.0), "frame 0 missed sample 0");
    }

    /// §29.6: zero in, exactly zero out. No dither, no floor, no epsilon.
    #[test]
    fn silence_analyses_to_exact_zeros() {
        let book = analyze_residual(&vec![0.0f32; 4800], SR, 0, &cfg(16, 50.0, 16_000.0, fixed_tau(2.0)))
            .unwrap();
        assert_eq!(book.frame_count, 100);
        assert!(book.power.iter().all(|&p| p == 0.0), "silence produced power");
    }

    /// §29.7: a tone lands in the band nearest its frequency, falls off monotonically either side,
    /// and leaves the far ends of the bank 40 dB down.
    #[test]
    fn a_tone_lands_in_the_nearest_band() {
        let f0 = 1000.0;
        let n = (SR as usize) / 2;
        let r: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * f0 * i as f64 / SR).sin() as f32)
            .collect();
        let book = analyze_residual(&r, SR, 0, &cfg(32, 50.0, 16_000.0, fixed_tau(20.0))).unwrap();

        let mean = tail_means(&book, 0.5);
        let peak = mean
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        // The winning band is the one whose centre is nearest 1 kHz.
        let nearest = book
            .bank
            .center_freq_hz
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - f0).abs().partial_cmp(&(b.1 - f0).abs()).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, nearest, "peak band {peak}, nearest band {nearest}");

        // Falling away from the peak through the overlapping neighbourhood. Only locally: far up
        // the bank the bands widen faster than the tone's skirt falls — an ERB is 2.2 kHz at
        // 20 kHz against 130 Hz at 1 kHz — so the last few bands turn back up at 80 dB down. That
        // is the bank's shape, not a leak, which is why the far ends are checked by level instead.
        let lo = peak.saturating_sub(4);
        assert!(mean[lo..=peak].windows(2).all(|w| w[1] > w[0]), "{mean:?}");
        assert!(mean[peak..peak + 5].windows(2).all(|w| w[1] < w[0]), "{mean:?}");

        let db = |p: f64| 10.0 * (p / mean[peak]).log10();
        assert!(db(mean[0]) < -40.0, "bottom band only {:.1} dB down", db(mean[0]));
        assert!(db(mean[31]) < -40.0, "top band only {:.1} dB down", db(mean[31]));
    }

    /// §29.8: white noise leaves every normalised band at the input's own variance. This is the
    /// unit-noise-power contract measured end to end rather than at the filter.
    #[test]
    fn broadband_noise_gives_flat_normalised_bands() {
        // Uniform on [-1, 1), so the variance is exactly 1/3.
        let r = pseudo_noise(2 * SR as usize);
        let book = analyze_residual(&r, SR, 0, &cfg(12, 200.0, 8000.0, fixed_tau(20.0))).unwrap();

        let mean = tail_means(&book, 0.75);
        for (b, &p) in mean.iter().enumerate() {
            let ratio = p / (1.0 / 3.0);
            assert!(
                (0.7..1.3).contains(&ratio),
                "band {b} ({:.0} Hz): power {p:.4}, {ratio:.3} of the input variance",
                book.bank.center_freq_hz[b]
            );
        }
    }

    /// §29.9: a short burst produces a short event. This is the test the whole temporal design
    /// exists for — an analyser that smoothed broadly would still pass everything above it.
    #[test]
    fn a_transient_stays_a_transient() {
        let ms = |t: f64| (t * 1e-3 * SR) as usize;
        let (onset, burst) = (ms(50.0), ms(2.0));
        let noise = pseudo_noise(burst);
        let mut r = vec![0.0f32; ms(300.0)];
        r[onset..onset + burst].copy_from_slice(&noise);

        // The default detector: bandwidth-relative, 0.5 to 10 ms.
        let book = analyze_residual(&r, SR, 0, &cfg(16, 200.0, 16_000.0, ResidualPowerConfig::default()))
            .unwrap();
        let total = book.total_power();
        let frame_of = |samples: usize| samples / book.update_samples as usize;

        // Nothing before the burst — the filters are causal and the detector starts at zero.
        assert!(total[..frame_of(onset)].iter().all(|&p| p == 0.0));

        let peak_frame = total
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let peak = total[peak_frame];
        let peak_ms = peak_frame as f64 * book.update_samples as f64 / SR * 1e3;
        assert!(
            (50.0..55.0).contains(&peak_ms),
            "peak at {peak_ms:.1} ms, burst at 50.0-52.0 ms"
        );

        // And gone again quickly: 20 dB down within 50 ms of the burst ending.
        let after = frame_of(onset + burst + ms(50.0));
        let down = 10.0 * (total[after] / peak).log10();
        assert!(down < -20.0, "only {down:.1} dB down 50 ms after the burst");
        assert!(
            total[frame_of(onset + burst + ms(200.0))] / peak < 1e-4,
            "still ringing 200 ms later"
        );
    }

    /// The parallel bank against a plainly-written serial one, bit for bit.
    ///
    /// The oracle is deliberately sample-major with a `%` test for the frame boundary — the
    /// reference implementation from spec §21 — so it shares no structure with the chunked
    /// band-major loop it is checking.
    #[test]
    fn the_parallel_bank_matches_a_serial_reference() {
        let r = pseudo_noise(20_000);
        let c = cfg(24, 50.0, 16_000.0, ResidualPowerConfig::default());
        let book = analyze_residual(&r, SR, 0, &c).unwrap();

        let centers = c.erb.center_freqs().unwrap();
        let bandwidths: Vec<f64> = centers.iter().map(|&f| erb_bandwidth_hz(f)).collect();
        let coeffs = detector_coeffs(SR, &c.power.taus(&bandwidths));
        let mut filters: Vec<GammatoneBand> = centers
            .iter()
            .map(|&f| GammatoneBand::design(f, SR, c.erb.filter_order).unwrap())
            .collect();

        let nb = c.erb.bands;
        let nu = c.update_samples;
        let mut p = vec![0.0f64; nb];
        let mut want = Vec::with_capacity(r.len().div_ceil(nu) * nb);
        for (n, &x) in r.iter().enumerate() {
            for b in 0..nb {
                let y = filters[b].process_sample(x as f64);
                p[b] = coeffs[b] * p[b] + (1.0 - coeffs[b]) * y * y;
            }
            if n % nu == 0 {
                want.extend(p.iter().map(|&v| v as f32));
            }
        }
        assert_eq!(book.power, want);
    }

    /// The descriptor says what the run actually used, not what was asked for.
    #[test]
    fn the_descriptor_records_the_resolved_bank() {
        let c = cfg(16, 50.0, 16_000.0, ResidualPowerConfig::default());
        let book = analyze_residual(&pseudo_noise(4800), SR, 0, &c).unwrap();
        let bank = &book.bank;

        assert_eq!(bank.band_count, 16);
        assert_eq!(bank.filter_order, 4);
        assert_eq!(bank.center_freq_hz.len(), 16);
        assert_eq!((bank.center_freq_hz[0], bank.center_freq_hz[15]), (50.0, 16_000.0));
        for (b, (&fc, &bw)) in bank.center_freq_hz.iter().zip(&bank.bandwidth_hz).enumerate() {
            assert!((bw - erb_bandwidth_hz(fc)).abs() < 1e-12, "band {b}");
            let designed = GammatoneBand::design(fc, SR, 4).unwrap();
            assert_eq!(bank.normalization_gain[b], designed.gain, "band {b}");
        }
        // Bandwidth-relative taus: 1/ERB where that lands inside the clamps, and the 29 Hz bottom
        // band asking for 34 ms is held at the 10 ms ceiling.
        let taus = &bank.power_detector.tau_seconds;
        assert_eq!(taus[0], c.power.tau_max_seconds);
        for (b, &tau) in taus.iter().enumerate() {
            let want = (1.0 / bank.bandwidth_hz[b])
                .clamp(c.power.tau_min_seconds, c.power.tau_max_seconds);
            assert_eq!(tau, want, "band {b}");
        }
        assert!(book.validate().is_ok());
    }

    #[test]
    fn an_empty_residual_gives_an_empty_book() {
        let book = analyze_residual(&[], SR, 0, &cfg(8, 200.0, 8000.0, fixed_tau(2.0))).unwrap();
        assert_eq!(book.frame_count, 0);
        assert!(book.power.is_empty());
        assert!(book.validate().is_ok());
    }

    /// A configuration error is refused here too, not only at load: `analyze_residual` is a public
    /// entry point and cannot assume the config came through `Config::validate`.
    #[test]
    fn an_invalid_config_is_refused() {
        let mut c = cfg(8, 200.0, 8000.0, fixed_tau(2.0));
        c.erb.max_freq_hz = 40_000.0;
        assert!(analyze_residual(&[0.0; 100], SR, 0, &c).is_err());

        c.erb.max_freq_hz = 8000.0;
        c.update_samples = 0;
        assert!(analyze_residual(&[0.0; 100], SR, 0, &c).is_err());
    }
}
