//! `rmpsynth` — render an rmp book to a soundfile: its atoms, its stochastic residual, or both.
//!
//! ```text
//! rmpsynth -b book.json.gz -o resynth.wav                        # atoms + embedded residual
//! rmpsynth -b book.json.gz --no-residual -o atoms.wav            # the atoms alone
//! rmpsynth -b book.json --residual-book bank.json.gz -o mix.wav  # atoms + a standalone residual
//! rmpsynth -b bank.json.gz -o stochastic.wav                     # a residual book: noise only
//! ```
//!
//! All synthesis lives here; `rmp` only analyses. A front end and nothing else: every decision about
//! the audio lives in [`rmp::synth`].
//!
//! `--book` takes either kind of book and works out which it is from the document (§27). FOF atoms
//! render through rfofs and Gaussian atoms through rmp's own definition, each exactly as the pursuit
//! subtracted it.

use clap::{Parser, ValueEnum};
use rmp::synth::{
    load_book, render_to_file, BookInput, ClippingPolicy, GainSmoothingConfig, GainSmoothingMode,
    OutputEncoding, RenderConfig, RenderRequest,
};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(
    name = "rmpsynth",
    about = "Render an rmp book: its atoms, its stochastic residual, or both",
    version
)]
struct Args {
    /// The book to render: a full rmp book, or a standalone residual book. Format follows the
    /// extension — .toml or .json, either optionally with a trailing .gz.
    #[arg(short, long)]
    book: PathBuf,

    /// Output soundfile.
    #[arg(short, long)]
    output: PathBuf,

    /// A standalone residual book, as written by `rmp --residual-book`, to render with the book's
    /// atoms. Replaces the book's own residual section if it has one.
    #[arg(long, value_name = "PATH")]
    residual_book: Option<PathBuf>,

    /// Leave the atoms out.
    #[arg(long)]
    no_atoms: bool,

    /// Leave the stochastic residual out.
    #[arg(long)]
    no_residual: bool,

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

    /// Start the output at the analysed excerpt instead of preserving the source timeline.
    #[arg(long, alias = "trim-to-residual")]
    trim_to_excerpt: bool,

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
            preserve_timeline: !self.trim_to_excerpt,
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
    let residual_book = match &args.residual_book {
        None => None,
        Some(path) => match load_book(path).map_err(|e| e.to_string())? {
            BookInput::Residual(r) => Some(r),
            BookInput::Full(b) => Some(b.residual.ok_or_else(|| {
                format!("--residual-book {} carries no residual section", path.display())
            })?),
        },
    };
    let embedded = matches!(&book, BookInput::Full(b) if b.residual.is_some());

    let request = RenderRequest {
        book,
        residual_book,
        atoms: !args.no_atoms,
        residual: !args.no_residual,
        output: args.output.clone(),
        config: args.config(),
    };
    let cfg = &request.config;

    say("Book:");
    say(&format!("  file:                 {}", args.book.display()));
    say(&format!("  source type:          {}", request.book.book_type()));

    if let Some(b) = request.atom_source() {
        say("");
        say("Atoms:");
        let kinds = rmp::synth::atoms::count_by_kind(b);
        say(&format!(
            "  count:                {}",
            kinds.iter().map(|(k, n)| format!("{n} {k}")).collect::<Vec<_>>().join(", ")
        ));
        say(&format!("  sample rate:          {} Hz", b.sample_rate));
        if b.start_sample > 0 {
            say(&format!(
                "  excerpt starts at:    source sample {} ({:.3} s)",
                b.start_sample,
                b.start_sample as f64 / b.sample_rate as f64
            ));
        }
    }

    if let Some(residual) = request.residual_source() {
        say("");
        say("Residual synthesis:");
        if request.residual_book.is_some() && embedded {
            say("  note:                 --residual-book replaces the book's own residual section");
        }
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
    }

    let report = render_to_file(&request).map_err(|e| e.to_string())?;

    if let Some(cal) = report.calibration {
        say(&format!(
            "  bank complementarity: {:+.2} dB worst, rms {:.3}, over {:.0} .. {:.0} Hz",
            cal.worst_db(),
            cal.rms_deviation,
            cal.range_hz.0,
            cal.range_hz.1
        ));
        if cal.worst_db().abs() > 1.0 {
            say(&format!(
                "  warning:              the bank cannot be made power-complementary to better \
                 than {:.1} dB — the reconstruction will comb. More residual.erb.bands is the fix.",
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
    }

    say("");
    say("Output:");
    say(&format!("  file:                 {}", args.output.display()));
    say(&format!("  encoding:             {}", cfg.output_encoding));
    say(&format!("  samples written:      {}", report.samples_written));
    if report.timeline_origin > 0 {
        say(&format!(
            "  timeline:             {}",
            if cfg.preserve_timeline {
                format!(
                    "source sample 0; the excerpt starts at {} (--trim-to-excerpt drops the lead-in)",
                    report.timeline_origin
                )
            } else {
                "trimmed to the excerpt".to_string()
            }
        ));
    }
    if report.atom_samples.is_some() {
        say(&format!("  atom peak:            {:.4}", report.atom_peak));
    }
    if report.residual_samples.is_some() {
        say(&format!("  residual peak:        {:.4}", report.residual_peak));
    }
    say(&format!("  peak:                 {:.4}", report.mixed_peak));
    say(&format!("  samples > 1.0:        {}", report.clipped_samples));
    Ok(())
}
