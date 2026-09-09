//! The renderer: book in, sound file out.
//!
//! ```text
//! ResidualBook -> band gains sqrt(P_b[k]) -> one-pole smoothing -> independent white noise
//!              -> power-complementary ERB bank -> stochastic residual [+ FOF audio] -> WAV
//! ```
//!
//! Three things fix the shape of the loop:
//!
//! **Block-structured, band-major** (§29, §30). The DSP core walks fixed 256-sample blocks and, for
//! each, every band in turn. That is the cache-friendly order, and it is the order an online engine
//! would need. The block size is *not* allowed to matter: each block is split at the book's exact
//! frame boundaries, and every band's noise stream and filter state run continuously across the
//! splits, so `the_block_size_does_not_change_the_output` holds bit for bit.
//!
//! **Bands are summed in a fixed order into an f64 accumulator** (§36, §43). Nothing here is
//! parallel. 48 bands of a fourth-order complex cascade is about 1500 flops per output sample —
//! comfortably faster than realtime serially — and parallelism would buy nothing while costing the
//! bit-identical determinism the seed is supposed to guarantee.
//!
//! **The only square root is at frame load** (§38). The book stores power; the trajectory carries
//! amplitude; `BandGainState::load_power` is where the two meet, once per band per frame.

use std::path::{Path, PathBuf};

use crate::audio;
use crate::book::Book;
use crate::residual::book::{ResidualBook, RESIDUAL_BOOK_VERSION};
use crate::signal::Signal;
use crate::synth::bank::{BankCalibration, SynthesisBand, SynthesisBank};
use crate::synth::config::{ClippingPolicy, OutputEncoding, RenderConfig};
use crate::synth::error::RenderError;
use crate::synth::gain::BandGainState;
use crate::synth::rng::{NoiseSource, Xoshiro256pp};

/// The DSP block length (§29). An implementation detail — it cannot change the output.
const BLOCK: usize = 256;

/// The only channel a residual book can describe today. Kept as a named constant so the places
/// that would have to change for §19's multichannel book are findable.
const CHANNEL: u32 = 0;

/// Which kind of document the render came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderBookType {
    Residual,
    Full,
}

impl std::fmt::Display for RenderBookType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Residual => "residual book",
            Self::Full => "full rmp book",
        })
    }
}

/// A book as loaded from disk, whichever of the two it turned out to be (§27).
#[derive(Debug)]
pub enum BookInput {
    Residual(ResidualBook),
    Full(Book),
}

impl BookInput {
    pub fn book_type(&self) -> RenderBookType {
        match self {
            Self::Residual(_) => RenderBookType::Residual,
            Self::Full(_) => RenderBookType::Full,
        }
    }

    /// The residual section, whichever kind of book this is (§27).
    pub fn residual(&self) -> Result<&ResidualBook, RenderError> {
        match self {
            Self::Residual(r) => Ok(r),
            Self::Full(b) => b.residual.as_ref().ok_or(RenderError::NoResidualBook),
        }
    }
}

/// A whole render, as a file-oriented request (§5).
pub struct RenderRequest {
    pub book: BookInput,
    /// Pre-rendered FOF synthesis to mix the residual into (§17). Never resampled, never
    /// time-stretched.
    pub fof_audio: Option<PathBuf>,
    pub output: PathBuf,
    pub config: RenderConfig,
}

/// One band, as the render resolved it — the `RMP_RESIDUAL_DETAIL` table.
#[derive(Clone, Copy, Debug)]
pub struct BandReport {
    pub center_hz: f64,
    pub bandwidth_hz: f64,
    /// Gain smoothing time constant, in seconds.
    pub tau_seconds: f64,
    /// The complementarity correction `c_b` applied to this band.
    pub scale: f64,
}

/// What a render did (§24).
#[derive(Clone, Debug)]
pub struct RenderReport {
    pub sample_rate: f64,
    pub channels: u16,
    pub samples_written: u64,
    /// Peak of the stochastic residual alone, before mixing and before the output gain.
    pub residual_peak: f32,
    /// Peak of what was actually written.
    pub mixed_peak: f32,
    pub clipped_samples: u64,
    pub seed: u64,
    pub book_type: RenderBookType,
    pub calibration: BankCalibration,
    pub bands: Vec<BandReport>,
    /// Length of the FOF audio in samples, when one was mixed in.
    pub fof_samples: Option<u64>,
    /// Channel count of the FOF file on disk, before the downmix.
    pub fof_channels: Option<usize>,
}

/// Read a book of either kind, deciding from the document itself (§27).
///
/// No flag distinguishes them: a full book has `selections`, a residual book has `power` and a
/// `bank`, and neither parses as the other. Both formats and the `.gz` suffix come free from
/// [`crate::book::read_doc`], which owns the extension rules.
pub fn load_book(path: &Path) -> Result<BookInput, RenderError> {
    match crate::book::read_doc::<Book>(path) {
        Ok(b) => {
            if let Some(r) = &b.residual {
                r.validate()
                    .map_err(|e| RenderError::Book(format!("{}: {e}", path.display())))?;
            }
            Ok(BookInput::Full(b))
        }
        Err(full_err) => match crate::book::read_doc::<ResidualBook>(path) {
            Ok(r) => {
                r.validate()
                    .map_err(|e| RenderError::Book(format!("{}: {e}", path.display())))?;
                Ok(BookInput::Residual(r))
            }
            // A document that is not valid TOML or JSON at all fails both attempts the same way,
            // and saying so twice helps nobody. Only a document that parses but fits neither shape
            // needs both halves of the story.
            Err(residual_err) if residual_err == full_err => Err(RenderError::Book(full_err)),
            Err(residual_err) => Err(RenderError::Book(format!(
                "{} is neither an rmp book ({full_err}) nor a residual book ({residual_err})",
                path.display()
            ))),
        },
    }
}

/// Render the stochastic residual a residual book describes.
pub fn render_residual_book(
    book: &ResidualBook,
    config: &RenderConfig,
) -> Result<Signal, RenderError> {
    let mut r = StochasticRenderer::new(book, config)?;
    r.render(book, config, BLOCK)
}

/// The same, from a full book's residual section (§27, §44.7).
///
/// Delegates rather than duplicating, so the two paths cannot produce different audio.
pub fn render_full_book(book: &Book, config: &RenderConfig) -> Result<Signal, RenderError> {
    let residual = book.residual.as_ref().ok_or(RenderError::NoResidualBook)?;
    render_residual_book(residual, config)
}

/// The whole file-oriented pipeline: render, optionally mix, gain, measure, write (§39).
pub fn render_to_file(request: &RenderRequest) -> Result<RenderReport, RenderError> {
    let cfg = &request.config;
    cfg.validate()?;

    let book = request.book.residual()?;
    let mut renderer = StochasticRenderer::new(book, cfg)?;
    let residual = renderer.render(book, cfg, BLOCK)?;
    let residual_peak = residual.peak();

    // The FOF file is taken as already sitting at the right place on the timeline (§17).
    let fof = match &request.fof_audio {
        Some(path) => {
            let input = audio::read(path)?;
            let want = book.sample_rate as f32;
            if input.signal.sample_rate != want {
                return Err(RenderError::SampleRateMismatch {
                    book: book.sample_rate,
                    fof_audio: input.signal.sample_rate,
                });
            }
            Some(input)
        }
        None => None,
    };

    // §18: a short FOF file is padded with zeros, a long one extends the output.
    let n_fof = fof.as_ref().map_or(0, |f| f.signal.len());
    let n_out = residual.len().max(n_fof);

    let gain = cfg.output_gain() as f32;
    let mut out = vec![0.0f32; n_out];
    for (n, y) in out.iter_mut().enumerate() {
        let r = residual.samples.get(n).copied().unwrap_or(0.0);
        let x = fof
            .as_ref()
            .and_then(|f| f.signal.samples.get(n))
            .copied()
            .unwrap_or(0.0);
        *y = (x + r) * gain;
    }

    let mut clipped = 0u64;
    let mut peak = 0.0f32;
    for y in &mut out {
        let a = y.abs();
        if a > peak {
            peak = a;
        }
        if a > 1.0 {
            clipped += 1;
            if cfg.clipping == ClippingPolicy::Clip {
                *y = y.clamp(-1.0, 1.0);
            }
        }
    }
    if clipped > 0 && cfg.clipping == ClippingPolicy::Error {
        return Err(RenderError::ClippingDetected { count: clipped, peak });
    }

    let signal = Signal::new(out, book.sample_rate as f32);
    match cfg.output_encoding {
        OutputEncoding::Float32 => audio::write(&request.output, &signal)?,
        OutputEncoding::Pcm24 => audio::write_pcm24(&request.output, &signal)?,
    }

    Ok(RenderReport {
        sample_rate: book.sample_rate,
        channels: 1,
        samples_written: signal.len() as u64,
        residual_peak,
        mixed_peak: peak,
        clipped_samples: clipped,
        seed: cfg.seed,
        book_type: request.book.book_type(),
        calibration: renderer.bank().calibration(),
        bands: renderer.band_reports(book),
        fof_samples: fof.as_ref().map(|f| f.signal.len() as u64),
        fof_channels: fof.as_ref().map(|f| f.channels),
    })
}

/// The DSP core. Everything it needs is allocated here; the render loop allocates nothing (§37).
pub struct StochasticRenderer {
    bank: SynthesisBank,
    rng: Vec<Xoshiro256pp>,
    gain: Vec<BandGainState>,
    /// Resolved smoothing time constant per band, for the diagnostic table.
    taus: Vec<f64>,
    noise: Vec<f32>,
    filtered: Vec<f32>,
    /// `(offset in block, length, frame in force)`. Reused, so it allocates once.
    segments: Vec<(usize, usize, Option<usize>)>,
}

impl StochasticRenderer {
    pub fn new(book: &ResidualBook, config: &RenderConfig) -> Result<Self, RenderError> {
        config.validate()?;
        if book.version > RESIDUAL_BOOK_VERSION {
            return Err(RenderError::UnsupportedResidualBookVersion(book.version));
        }
        book.validate().map_err(RenderError::Book)?;
        if !(book.sample_rate.is_finite() && book.sample_rate > 0.0) {
            return Err(RenderError::InvalidConfig(format!(
                "residual book sample rate {} is not a usable rate",
                book.sample_rate
            )));
        }
        if book.update_samples == 0 {
            return Err(RenderError::Book(
                "residual book update_samples is 0: the frame grid has no spacing".into(),
            ));
        }

        let bank = SynthesisBank::design(&book.bank, book.sample_rate)?;
        let n = bank.len();
        let taus = config.gain_smoothing.taus(&book.bank.bandwidth_hz);

        Ok(Self {
            bank,
            // §12: seeded from the master seed and the band index, never from creation order.
            rng: (0..n)
                .map(|b| Xoshiro256pp::seeded(config.seed, b as u32, CHANNEL))
                .collect(),
            gain: taus
                .iter()
                .map(|&t| BandGainState::new(t, book.sample_rate))
                .collect(),
            taus,
            noise: vec![0.0; BLOCK],
            filtered: vec![0.0; BLOCK],
            segments: Vec::with_capacity(BLOCK / 2 + 2),
        })
    }

    pub fn bank(&self) -> &SynthesisBank {
        &self.bank
    }

    /// The smoothing time constant each band resolved to, in seconds.
    pub fn taus(&self) -> &[f64] {
        &self.taus
    }

    /// Per-band diagnostics, in band order.
    pub fn band_reports(&self, book: &ResidualBook) -> Vec<BandReport> {
        (0..self.bank.len())
            .map(|b| BandReport {
                center_hz: book.bank.center_freq_hz[b],
                bandwidth_hz: book.bank.bandwidth_hz[b],
                tau_seconds: self.taus[b],
                scale: self.bank.scale()[b],
            })
            .collect()
    }

    /// Render the whole book.
    ///
    /// `block` is the DSP block length and is an implementation detail: the frame grid, the noise
    /// streams and the filter states all run across block boundaries, so the output does not depend
    /// on it. Exposed only so the tests can prove that.
    pub fn render(
        &mut self,
        book: &ResidualBook,
        config: &RenderConfig,
        block: usize,
    ) -> Result<Signal, RenderError> {
        let block = block.max(1).min(self.noise.len());
        let bands = self.bank.len();
        let sr = book.sample_rate as f32;

        // §16: output sample 0 is source sample 0 unless the timeline is explicitly trimmed away.
        // §33: rendering stops at source_samples — no filter tail.
        let start = if config.preserve_timeline {
            book.start_sample as usize
        } else {
            0
        };
        let n_out = start + book.source_samples as usize;
        let mut acc = vec![0.0f64; n_out];

        for band in &mut self.gain {
            band.reset();
        }
        self.bank.reset();

        let mut n0 = 0usize;
        while n0 < n_out {
            let len = block.min(n_out - n0);
            split_at_frames(n0, len, start, book, &mut self.segments);

            for b in 0..bands {
                let gain = &mut self.gain[b];
                let rng = &mut self.rng[b];
                let filter = self.bank.band_mut(b);
                for &(off, seg_len, frame) in &self.segments {
                    if let Some(k) = frame {
                        gain.load_power(book.power[k * bands + b]);
                    }
                    let noise = &mut self.noise[..seg_len];
                    let filtered = &mut self.filtered[..seg_len];
                    rng.fill(noise);
                    SynthesisBand::process_block(filter, noise, filtered);
                    for (i, &y) in filtered.iter().enumerate() {
                        acc[n0 + off + i] += (gain.step() * y) as f64;
                    }
                }
            }
            n0 += len;
        }

        Ok(Signal::new(
            acc.into_iter().map(|v| v as f32).collect(),
            sr,
        ))
    }
}

/// Cut one block at the book's exact frame boundaries (§15, §29).
///
/// A frame becomes active at `start + k * update_samples` and stays in force until the next one.
/// Before the first frame there is none, and the band gain stays at exactly zero (§32). After the
/// last, the last frame holds — `frame_count` is `source_samples / update_samples` rounded up, so
/// the final frame covers a short tail.
fn split_at_frames(
    n0: usize,
    len: usize,
    start: usize,
    book: &ResidualBook,
    out: &mut Vec<(usize, usize, Option<usize>)>,
) {
    out.clear();
    let step = book.update_samples as usize;
    let last = book.frame_count.saturating_sub(1) as usize;

    let mut off = 0usize;
    while off < len {
        let n = n0 + off;
        // The frame in force at `n`, and the sample where the next one takes over.
        let (frame, next_boundary) = if n < start {
            // Silence before the analysed excerpt; the next event is the first frame.
            (None, start)
        } else if book.frame_count == 0 {
            (None, usize::MAX)
        } else {
            let k = (n - start) / step;
            if k >= last {
                (Some(last), usize::MAX)
            } else {
                (Some(k), start + (k + 1) * step)
            }
        };
        // Reloading the frame already in force is a no-op — `load_power` writes the same target —
        // so a segment can simply always load, and block boundaries need no special case.
        let seg_end = next_boundary.saturating_sub(n0).clamp(off + 1, len);
        out.push((off, seg_end - off, frame));
        off = seg_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::testing::{a_book, book_with_power};

    /// §44.1: an all-zero book renders exact zeros, not merely quiet ones.
    #[test]
    fn a_zero_book_renders_exact_silence() {
        let book = book_with_power(48_000.0, 24, 4800, |_, _| 0.0);
        let out = render_residual_book(&book, &RenderConfig::default()).unwrap();
        assert!(out.samples.iter().all(|&s| s == 0.0));
    }

    /// §29: the block length is an implementation detail and must not reach the output.
    #[test]
    fn the_block_size_does_not_change_the_output() {
        let book = book_with_power(48_000.0, 16, 4800, |_, b| 0.01 * (b as f32 + 1.0));
        let cfg = RenderConfig::default();
        let want = {
            let mut r = StochasticRenderer::new(&book, &cfg).unwrap();
            r.render(&book, &cfg, 256).unwrap()
        };
        for block in [1usize, 7, 48, 64, 100, 256] {
            let mut r = StochasticRenderer::new(&book, &cfg).unwrap();
            let got = r.render(&book, &cfg, block).unwrap();
            assert_eq!(got.samples, want.samples, "block {block}");
        }
    }

    /// §44.6: the seed is the whole of the nondeterminism, and it is honoured.
    #[test]
    fn the_same_seed_renders_the_same_samples() {
        let book = book_with_power(48_000.0, 24, 9600, |_, _| 0.001);
        let cfg = RenderConfig::default();
        let a = render_residual_book(&book, &cfg).unwrap();
        let b = render_residual_book(&book, &cfg).unwrap();
        assert_eq!(a.samples, b.samples);

        let other = RenderConfig { seed: 999, ..cfg };
        let c = render_residual_book(&book, &other).unwrap();
        assert_ne!(a.samples, c.samples);
        // Different waveform, same spectral envelope: the levels agree closely.
        let (ra, rc) = (a.rms(), c.rms());
        assert!((ra / rc).log10().abs() < 0.05, "rms {ra} vs {rc}");
    }

    /// §16, §32: a book analysed from an offset renders leading silence, exactly.
    #[test]
    fn the_timeline_is_preserved_by_default() {
        let mut book = book_with_power(48_000.0, 16, 4800, |_, _| 0.01);
        book.start_sample = 1000;
        let cfg = RenderConfig::default();

        let out = render_residual_book(&book, &cfg).unwrap();
        assert_eq!(out.len(), 1000 + 4800);
        assert!(out.samples[..1000].iter().all(|&s| s == 0.0));
        assert!(out.samples[1000..].iter().any(|&s| s != 0.0));

        let trimmed = RenderConfig {
            preserve_timeline: false,
            ..cfg
        };
        let out = render_residual_book(&book, &trimmed).unwrap();
        assert_eq!(out.len(), 4800);
    }

    /// §33: rendering stops at `source_samples`, with no filter tail appended.
    #[test]
    fn the_output_is_exactly_the_source_length() {
        for &n in &[48usize, 100, 4800, 5000] {
            let book = book_with_power(48_000.0, 8, n, |_, _| 0.01);
            let out = render_residual_book(&book, &RenderConfig::default()).unwrap();
            assert_eq!(out.len(), n);
        }
    }

    /// §44.4: a burst arrives where the book puts it, to within the smoothing time constant, and
    /// leaves again. The one-sample-accurate claim of §15 is what is being checked.
    #[test]
    fn a_burst_lands_on_the_books_own_sample_positions() {
        // 1 ms frames at 48 kHz; the burst runs over frames 40..60, so samples 1920..2880.
        let book = book_with_power(48_000.0, 24, 4800, |k, _| if (40..60).contains(&k) { 1.0 } else { 0.0 });
        let out = render_residual_book(&book, &RenderConfig::default()).unwrap();

        let energy = |lo: usize, hi: usize| -> f64 {
            out.samples[lo..hi].iter().map(|&s| (s as f64).powi(2)).sum()
        };
        // Nothing before the burst at all, and the gain state is exactly zero there.
        assert!(out.samples[..1920].iter().all(|&s| s == 0.0));
        // The burst carries essentially all the energy; a couple of time constants of tail is fine.
        let inside = energy(1920, 2880 + 480);
        let after = energy(2880 + 480, 4800);
        assert!(inside > 200.0 * after, "inside {inside}, after {after}");
    }

    /// §44.3: one active band puts its noise where that band is, and nowhere else.
    #[test]
    fn a_single_active_band_is_localised_at_its_centre() {
        use crate::residual::filter::{AnalysisBand, GammatoneBand};

        let sr = 48_000.0;
        let book = book_with_power(sr, 24, 48_000, |_, b| if b == 12 { 1.0 } else { 0.0 });
        let out = render_residual_book(&book, &RenderConfig::default()).unwrap();

        // Re-analyse through the same filters and see which band the energy landed in.
        let power: Vec<f64> = book
            .bank
            .center_freq_hz
            .iter()
            .map(|&fc| {
                let mut band = GammatoneBand::design(fc, sr, 4).unwrap();
                let mut y = vec![0.0f32; out.len()];
                AnalysisBand::process_block(&mut band, &out.samples, &mut y);
                y.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / y.len() as f64
            })
            .collect();

        let peak = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 12, "energy peaked in band {peak}: {power:?}");
        // Four bands away is already far down.
        assert!(power[12] > 50.0 * power[12 - 4], "{:?}", power);
        assert!(power[12] > 50.0 * power[12 + 4], "{:?}", power);
    }

    /// §44.2: constant power gives stationary noise — no periodic artifact at the frame rate.
    #[test]
    fn constant_band_power_is_stationary() {
        let book = book_with_power(48_000.0, 32, 48_000, |_, _| 0.001);
        let out = render_residual_book(&book, &RenderConfig::default()).unwrap();

        // Skip the first 20 ms while the gains settle, then compare 10 ms windows.
        let windows: Vec<f64> = out.samples[960..]
            .chunks_exact(480)
            .map(|w| w.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / 480.0)
            .collect();
        let mean = windows.iter().sum::<f64>() / windows.len() as f64;
        let spread = windows.iter().map(|w| (w / mean - 1.0).abs()).fold(0.0, f64::max);
        assert!(spread < 0.35, "window powers vary by {spread} around {mean}");
    }

    /// §44.5: many high bands for a few milliseconds stays a broadband transient at the right time.
    #[test]
    fn a_multi_band_transient_keeps_its_place() {
        let book = book_with_power(48_000.0, 32, 24_000, |k, b| {
            if (100..105).contains(&k) && b >= 20 { 1.0 } else { 0.0 }
        });
        let out = render_residual_book(&book, &RenderConfig::default()).unwrap();
        let peak_at = out
            .samples
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        // Frames 100..105 are samples 4800..5040; the smoothing and the filter ringing widen that.
        assert!((4700..5600).contains(&peak_at), "peak at sample {peak_at}");
    }

    /// §44.7: the same audio whichever kind of book carried the residual section.
    #[test]
    fn a_full_book_renders_its_embedded_residual_identically() {
        let residual = book_with_power(48_000.0, 16, 4800, |_, b| 0.001 * b as f32);
        let full = Book {
            selections: vec![],
            initial_energy: 1.0,
            sample_rate: 48_000.0,
            residual: Some(residual.clone()),
        };
        let cfg = RenderConfig::default();
        assert_eq!(
            render_full_book(&full, &cfg).unwrap().samples,
            render_residual_book(&residual, &cfg).unwrap().samples
        );
    }

    /// §44.8.
    #[test]
    fn a_full_book_without_a_residual_section_is_refused() {
        let full = Book {
            selections: vec![],
            initial_energy: 1.0,
            sample_rate: 48_000.0,
            residual: None,
        };
        let err = render_full_book(&full, &RenderConfig::default()).unwrap_err();
        assert!(matches!(err, RenderError::NoResidualBook));
        assert!(err.to_string().contains("--residual-analysis"));
    }

    /// A book this build cannot interpret is refused rather than rendered as something else.
    #[test]
    fn a_newer_book_version_is_refused() {
        let mut book = a_book(48_000.0, 8, 480);
        book.version = RESIDUAL_BOOK_VERSION + 1;
        assert!(matches!(
            render_residual_book(&book, &RenderConfig::default()),
            Err(RenderError::UnsupportedResidualBookVersion(_))
        ));
    }

    /// An empty book is a legitimate analysis result (`an_empty_residual_gives_an_empty_book`), so
    /// it must render rather than panic.
    #[test]
    fn an_empty_book_renders_nothing() {
        let book = book_with_power(48_000.0, 8, 0, |_, _| 0.0);
        let out = render_residual_book(&book, &RenderConfig::default()).unwrap();
        assert!(out.is_empty());
    }

    /// The gain smoothing is a real setting: a long time constant visibly slows the attack.
    #[test]
    fn a_longer_smoothing_time_softens_the_attack() {
        use crate::synth::config::GainSmoothingConfig;
        let book = book_with_power(48_000.0, 16, 4800, |k, _| if k >= 10 { 1.0 } else { 0.0 });
        let onset = 480..480 + 96; // the first 2 ms after the step

        let energy = |ms: f64| {
            let cfg = RenderConfig {
                gain_smoothing: GainSmoothingConfig {
                    fixed_seconds: ms * 1e-3,
                    ..Default::default()
                },
                ..Default::default()
            };
            let out = render_residual_book(&book, &cfg).unwrap();
            out.samples[onset.clone()].iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
        };
        assert!(energy(0.2) > 3.0 * energy(5.0), "{} vs {}", energy(0.2), energy(5.0));
    }

    /// The file-level half of §44: everything that needs a real book, a real soundfile and a real
    /// output path. Temp files are named by process id and cleaned up, as `book`'s tests are.
    mod files {
        use super::*;
        use crate::residual::book::ResidualBook;

        fn tmp(name: &str) -> PathBuf {
            // The pid goes before the name, so the extension stays the last thing on the path.
            std::env::temp_dir().join(format!("rmp_synth_{}_{name}", std::process::id()))
        }

        fn request(book: BookInput, out: &Path) -> RenderRequest {
            RenderRequest {
                book,
                fof_audio: None,
                output: out.to_path_buf(),
                config: RenderConfig::default(),
            }
        }

        fn write_wav(path: &Path, samples: Vec<f32>, sample_rate: f32) {
            audio::write(path, &Signal::new(samples, sample_rate)).unwrap();
        }

        /// §27: both kinds of book are recognised from the document, in both wire formats, with and
        /// without compression — no flag says which is which.
        #[test]
        fn a_book_of_either_kind_is_recognised_from_the_document() {
            let residual = book_with_power(48_000.0, 8, 480, |_, b| b as f32);
            let full = Book {
                selections: vec![],
                initial_energy: 1.0,
                sample_rate: 48_000.0,
                residual: Some(residual.clone()),
            };

            for ext in ["toml", "json", "json.gz"] {
                let p = tmp(&format!("detect.{ext}"));
                crate::book::write_doc(&p, &residual).unwrap();
                match load_book(&p).unwrap() {
                    BookInput::Residual(r) => assert_eq!(r, residual),
                    BookInput::Full(_) => panic!("{ext}: read a residual book as a full one"),
                }

                crate::book::write_doc(&p, &full).unwrap();
                match load_book(&p).unwrap() {
                    BookInput::Full(b) => assert_eq!(b.residual.unwrap(), residual),
                    BookInput::Residual(_) => panic!("{ext}: read a full book as a residual one"),
                }
                let _ = std::fs::remove_file(&p);
            }
        }

        /// A file that is neither says so, naming both attempts.
        #[test]
        fn something_that_is_neither_book_is_refused() {
            let p = tmp("nonsense.json");
            std::fs::write(&p, "{\"hello\": 1}").unwrap();
            let err = load_book(&p).unwrap_err().to_string();
            assert!(err.contains("neither an rmp book") && err.contains("residual book"), "{err}");
            let _ = std::fs::remove_file(&p);
        }

        /// §44.9: with a zero residual the output is the FOF file, sample for sample.
        #[test]
        fn a_zero_residual_mixes_to_exactly_the_fof_audio() {
            let fof = tmp("mix_fof.wav");
            let out = tmp("mix_out.wav");
            let want: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.05).sin() * 0.4).collect();
            write_wav(&fof, want.clone(), 48_000.0);

            let book = book_with_power(48_000.0, 8, 2000, |_, _| 0.0);
            let mut req = request(BookInput::Residual(book), &out);
            req.fof_audio = Some(fof.clone());
            let report = render_to_file(&req).unwrap();

            assert_eq!(report.residual_peak, 0.0);
            assert_eq!(report.fof_samples, Some(2000));
            assert_eq!(report.fof_channels, Some(1));
            let got = audio::read(&out).unwrap().signal;
            assert_eq!(got.samples, want);
            for p in [&fof, &out] {
                let _ = std::fs::remove_file(p);
            }
        }

        /// §44.10, §18: no silent resampling.
        #[test]
        fn a_sample_rate_mismatch_is_refused() {
            let fof = tmp("rate_fof.wav");
            let out = tmp("rate_out.wav");
            write_wav(&fof, vec![0.1; 1000], 44_100.0);

            let book = book_with_power(48_000.0, 8, 1000, |_, _| 0.0);
            let mut req = request(BookInput::Residual(book), &out);
            req.fof_audio = Some(fof.clone());

            match render_to_file(&req) {
                Err(RenderError::SampleRateMismatch { book, fof_audio }) => {
                    assert_eq!((book, fof_audio), (48_000.0, 44_100.0));
                }
                other => panic!("expected a rate mismatch, got {:?}", other.map(|r| r.samples_written)),
            }
            let _ = std::fs::remove_file(&fof);
        }

        /// §44.11, §18: the output is as long as the longer of the two, and a short FOF file is
        /// padded rather than truncating the residual.
        #[test]
        fn the_output_is_as_long_as_the_longer_input() {
            let book = book_with_power(48_000.0, 8, 4800, |_, _| 0.001);
            for (fof_len, want) in [(1000usize, 4800usize), (4800, 4800), (9600, 9600)] {
                let fof = tmp(&format!("len_fof_{fof_len}.wav"));
                let out = tmp(&format!("len_out_{fof_len}.wav"));
                write_wav(&fof, vec![0.01; fof_len], 48_000.0);

                let mut req = request(BookInput::Residual(book.clone()), &out);
                req.fof_audio = Some(fof.clone());
                let report = render_to_file(&req).unwrap();
                assert_eq!(report.samples_written, want as u64, "fof of {fof_len}");
                assert_eq!(audio::read(&out).unwrap().signal.len(), want);
                for p in [&fof, &out] {
                    let _ = std::fs::remove_file(p);
                }
            }
        }

        /// §44.12, §21, §22: gain is applied, overs are counted, and each policy does its own thing.
        /// Nothing is ever normalised.
        #[test]
        fn overs_are_counted_and_the_policy_is_honoured() {
            let fof = tmp("clip_fof.wav");
            let book = book_with_power(48_000.0, 8, 1000, |_, _| 0.0);
            // Half the samples are past full scale once the gain is applied.
            let x: Vec<f32> = (0..1000).map(|i| if i % 2 == 0 { 0.9 } else { 0.1 }).collect();
            write_wav(&fof, x, 48_000.0);

            let render = |policy: ClippingPolicy, name: &str| {
                let out = tmp(name);
                let mut req = request(BookInput::Residual(book.clone()), &out);
                req.fof_audio = Some(fof.clone());
                req.config.output_gain_db = 6.0206; // exactly 2x
                req.config.clipping = policy;
                (render_to_file(&req), out)
            };

            // Report: written through untouched, and counted.
            let (report, out) = render(ClippingPolicy::Report, "clip_report.wav");
            let report = report.unwrap();
            assert_eq!(report.clipped_samples, 500);
            assert!((report.mixed_peak - 1.8).abs() < 1e-5, "peak {}", report.mixed_peak);
            assert!((audio::read(&out).unwrap().signal.peak() - 1.8).abs() < 1e-5);
            let _ = std::fs::remove_file(&out);

            // Clip: bounded, still counted, and the file is written.
            let (report, out) = render(ClippingPolicy::Clip, "clip_clip.wav");
            assert_eq!(report.unwrap().clipped_samples, 500);
            assert!(audio::read(&out).unwrap().signal.peak() <= 1.0);
            let _ = std::fs::remove_file(&out);

            // Error: refused, and nothing is written.
            let (report, out) = render(ClippingPolicy::Error, "clip_error.wav");
            match report {
                Err(RenderError::ClippingDetected { count, .. }) => assert_eq!(count, 500),
                other => panic!("expected a clipping error, got {}", other.is_ok()),
            }
            assert!(!out.exists());
            let _ = std::fs::remove_file(&fof);
        }

        /// §20: both encodings write a readable file at the book's own rate.
        #[test]
        fn both_encodings_round_trip() {
            let book = book_with_power(48_000.0, 8, 1000, |_, _| 0.01);
            for (enc, name) in [
                (OutputEncoding::Float32, "enc_f32.wav"),
                (OutputEncoding::Pcm24, "enc_pcm24.wav"),
            ] {
                let out = tmp(name);
                let mut req = request(BookInput::Residual(book.clone()), &out);
                req.config.output_encoding = enc;
                let report = render_to_file(&req).unwrap();
                assert_eq!(report.samples_written, 1000);

                let got = audio::read(&out).unwrap().signal;
                assert_eq!(got.sample_rate, 48_000.0);
                assert_eq!(got.len(), 1000);
                assert!(got.peak() > 0.0, "{name} is silent");
                let _ = std::fs::remove_file(&out);
            }
        }

        /// §21: the output gain is exactly `10^(dB/20)`, applied to the residual too.
        #[test]
        fn the_output_gain_scales_the_whole_mix() {
            let book = book_with_power(48_000.0, 16, 4800, |_, _| 0.001);
            let render = |db: f64, name: &str| {
                let out = tmp(name);
                let mut req = request(BookInput::Residual(book.clone()), &out);
                req.config.output_gain_db = db;
                let report = render_to_file(&req).unwrap();
                let peak = report.mixed_peak;
                let _ = std::fs::remove_file(&out);
                peak
            };
            let (unity, up) = (render(0.0, "gain_0.wav"), render(20.0, "gain_20.wav"));
            assert!((up / unity - 10.0).abs() < 1e-3, "{up} / {unity}");
        }

        /// The report says which kind of book it rendered, and the audio does not depend on it.
        #[test]
        fn the_report_names_the_source_kind() {
            let residual: ResidualBook = book_with_power(48_000.0, 8, 480, |_, b| 0.01 * b as f32);
            let full = Book {
                selections: vec![],
                initial_energy: 1.0,
                sample_rate: 48_000.0,
                residual: Some(residual.clone()),
            };

            let a = tmp("kind_r.wav");
            let b = tmp("kind_f.wav");
            let ra = render_to_file(&request(BookInput::Residual(residual), &a)).unwrap();
            let rb = render_to_file(&request(BookInput::Full(full), &b)).unwrap();
            assert_eq!(ra.book_type, RenderBookType::Residual);
            assert_eq!(rb.book_type, RenderBookType::Full);
            assert_eq!(
                audio::read(&a).unwrap().signal.samples,
                audio::read(&b).unwrap().signal.samples
            );
            for p in [&a, &b] {
                let _ = std::fs::remove_file(p);
            }
        }
    }
}
