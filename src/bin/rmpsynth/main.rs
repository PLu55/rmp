//! `rmpsynth` — reconstruct the stochastic residual an RMP analysis measured.
//!
//! ```text
//! rmpsynth -b book.json.gz -o residual.wav
//! rmpsynth -b book.json.gz --fof-audio fof.wav -o mixed.wav
//! ```
//!
//! A front end and nothing else: every decision about the audio lives in [`rmp::synth`], so `rmp`
//! itself can call the same renderer without going through a process (§28).
//!
//! `--book` takes either kind of book and works out which it is from the document (§27). A full
//! book's residual section is used; its atoms are not synthesised here — render those with `rmp -b
//! book -o fof.wav` and pass the result as `--fof-audio`.

use clap::{Parser, ValueEnum};
use rmp::synth::{
    load_book, render_to_file, ClippingPolicy, GainSmoothingConfig, GainSmoothingMode,
    OutputEncoding, RenderConfig, RenderRequest,
};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(
    name = "rmpsynth",
    about = "Stochastic resynthesis of an RMP residual book",
    version
)]
struct Args {
    /// The book to render: a standalone residual book, or a full rmp book carrying one. Format
    /// follows the extension — .toml or .json, either optionally with a trailing .gz.
    #[arg(short, long)]
    book: PathBuf,

    /// Output soundfile.
    #[arg(short, long)]
    output: PathBuf,

    /// Pre-rendered FOF synthesis to mix the residual into. Must already sit at the right place on
    /// the timeline and be at the book's sample rate; nothing is resampled or stretched.
    #[arg(long, value_name = "PATH")]
    fof_audio: Option<PathBuf>,

    /// Seed for the per-band noise. The same seed renders the same samples, every time.
    #[arg(long, default_value_t = RenderConfig::default().seed)]
    seed: u64,

    /// Gain smoothing time constant. Short: it exists to stop a step between book frames becoming
    /// a click, not to smooth the envelope a second time.
    #[arg(long, value_name = "MS", default_value_t = 1.0)]
    gain_smoothing_ms: f64,

    /// Tie the smoothing time constant to each band's own bandwidth instead of fixing it. In that
    /// mode --gain-smoothing-ms sets the constant for a 24.7 Hz band and every wider band scales
    /// down from it, clamped to 0.2 .. 10 ms — the same rule the analysis detector follows.
    #[arg(long, value_name = "MODE", value_enum, default_value_t = SmoothingMode::Fixed)]
    gain_smoothing_mode: SmoothingMode,

    /// Output gain, applied after mixing. Nothing is normalised, ever.
    #[arg(long, value_name = "DB", default_value_t = 0.0)]
    gain_db: f64,

    /// Output sample format.
    #[arg(long, value_enum, default_value_t = Encoding::Float32)]
    encoding: Encoding,

    /// Hard-clip samples past full scale instead of writing them through.
    #[arg(long)]
    clip: bool,

    /// Refuse to write a file that has samples past full scale.
    #[arg(long, conflicts_with = "clip")]
    error_on_clip: bool,

    /// Trim the output to the analysed excerpt instead of preserving the source timeline.
    #[arg(long)]
    trim_to_residual: bool,

    /// Print the per-band table. The RMP_RESIDUAL_DETAIL environment variable does the same.
    #[arg(short, long)]
    verbose: bool,

    /// Suppress the report.
    #[arg(short, long)]
    quiet: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Encoding {
    Float32,
    Pcm24,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SmoothingMode {
    Fixed,
    BandwidthRelative,
}

impl Args {
    fn config(&self) -> RenderConfig {
        RenderConfig {
            seed: self.seed,
            gain_smoothing: GainSmoothingConfig {
                mode: match self.gain_smoothing_mode {
                    SmoothingMode::Fixed => GainSmoothingMode::Fixed,
                    SmoothingMode::BandwidthRelative => GainSmoothingMode::BandwidthRelative,
                },
                // The one place milliseconds become seconds.
                fixed_seconds: self.gain_smoothing_ms * 1e-3,
                scale: self.gain_smoothing_ms * 1e-3 * 24.7,
                ..GainSmoothingConfig::default()
            },
            output_gain_db: self.gain_db,
            output_encoding: match self.encoding {
                Encoding::Float32 => OutputEncoding::Float32,
                Encoding::Pcm24 => OutputEncoding::Pcm24,
            },
            clipping: match (self.clip, self.error_on_clip) {
                (true, _) => ClippingPolicy::Clip,
                (_, true) => ClippingPolicy::Error,
                _ => ClippingPolicy::Report,
            },
            preserve_timeline: !self.trim_to_residual,
        }
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rmpsynth: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), String> {
    // Reports go to stderr, as `rmp`'s do, so a caller can redirect the audio without losing them.
    let say = |line: &str| {
        if !args.quiet {
            eprintln!("{line}");
        }
    };

    let book = load_book(&args.book).map_err(|e| e.to_string())?;
    let book_type = book.book_type();
    let residual = book.residual().map_err(|e| e.to_string())?;
    let cfg = args.config();

    say("Residual synthesis:");
    say(&format!("  book:                 {}", args.book.display()));
    say(&format!("  source type:          {book_type}"));
    say(&format!("  sample rate:          {} Hz", residual.sample_rate));
    say(&format!(
        "  ERB bands:            {} over {:.1} .. {:.1} Hz, order {} {}",
        residual.band_count,
        residual.bank.min_freq_hz,
        residual.bank.max_freq_hz,
        residual.bank.filter_order,
        residual.bank.filter_kind
    ));
    say(&format!("  residual frames:      {}", residual.frame_count));
    say(&format!(
        "  update interval:      {} samples / {:.3} ms",
        residual.update_samples,
        residual.update_samples as f64 / residual.sample_rate * 1e3
    ));
    say(&format!("  seed:                 {}", cfg.seed));
    say(&format!(
        "  gain smoothing:       {:.2} ms, {}",
        args.gain_smoothing_ms, cfg.gain_smoothing.mode
    ));

    // §17 takes the FOF file as already sitting at the right place on the timeline, and nothing
    // here can check that it does. The one case where it is predictably wrong is worth saying out
    // loud: `rmp -s 2.0 -o fof.wav` writes the excerpt starting at sample 0, while the book it
    // wrote alongside records where the excerpt came from.
    if args.fof_audio.is_some() && residual.start_sample > 0 && !args.trim_to_residual {
        say(&format!(
            "  note:                 the book was analysed from sample {} ({:.3} s), so the \
             residual is placed there. If --fof-audio was rendered from the excerpt alone, pass \
             --trim-to-residual to line the two up.",
            residual.start_sample,
            residual.start_sample as f64 / residual.sample_rate
        ));
    }

    let request = RenderRequest {
        book,
        fof_audio: args.fof_audio.clone(),
        output: args.output.clone(),
        config: cfg,
    };
    let report = render_to_file(&request).map_err(|e| e.to_string())?;

    let cal = report.calibration;
    say(&format!(
        "  bank complementarity: {:+.2} dB worst, rms {:.3}, over {:.0} .. {:.0} Hz",
        cal.worst_db(),
        cal.rms_deviation,
        cal.range_hz.0,
        cal.range_hz.1
    ));
    if cal.worst_db().abs() > 1.0 {
        say(&format!(
            "  warning:              the bank cannot be made power-complementary to better than \
             {:.1} dB — the reconstruction will comb. More residual.erb.bands is the fix.",
            cal.worst_db().abs()
        ));
    }

    if args.verbose || std::env::var_os("RMP_RESIDUAL_DETAIL").is_some() {
        say("    band   center_hz  bandwidth_hz  tau_ms       scale");
        for (b, r) in report.bands.iter().enumerate() {
            say(&format!(
                "    {b:>4}  {:>10.2}  {:>12.2}  {:>6.2}  {:>10.3e}",
                r.center_hz,
                r.bandwidth_hz,
                r.tau_seconds * 1e3,
                r.scale
            ));
        }
    }

    if let (Some(n), Some(ch)) = (report.fof_samples, report.fof_channels) {
        say("");
        say("FOF audio:");
        say(&format!(
            "  input:                {}",
            args.fof_audio.as_ref().map_or_else(String::new, |p| p.display().to_string())
        ));
        say(&format!("  channels:             {ch}{}", if ch > 1 { " (downmixed to mono)" } else { "" }));
        say(&format!("  samples:              {n}"));
    }

    say("");
    say("Output:");
    say(&format!("  file:                 {}", args.output.display()));
    say(&format!("  encoding:             {}", request.config.output_encoding));
    say(&format!("  samples written:      {}", report.samples_written));
    say(&format!("  residual peak:        {:.4}", report.residual_peak));
    say(&format!("  peak:                 {:.4}", report.mixed_peak));
    say(&format!("  samples > 1.0:        {}", report.clipped_samples));
    Ok(())
}
