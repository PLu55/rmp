//! `rmp` — decompose a soundfile into FOF and Gaussian atoms.
//!
//! ```text
//! rmp input.wav -b book.json.gz [-c settings.toml] [-r residual.wav]
//!     [-s start_seconds] [-d duration_seconds]
//! rmp --write-config > settings.toml
//! ```
//!
//! Analysis only. Turning a book back into audio is `rmpsynth`'s job: `rmpsynth -b book -o out.wav`.

mod report;

use clap::Parser;
use rmp_core::audio;
use rmp_core::book;
use rmp_core::config::{Config, DEFAULT_CONFIG_HEADER};
use rmp_core::fft::Planner;
use rmp_core::pipeline;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(
    name = "rmp",
    about = "Matching-pursuit decomposition of audio into FOF and Gaussian atoms",
    version
)]
struct Args {
    /// Input soundfile (WAV, AIFF, FLAC). Multi-channel input is downmixed to mono.
    input: Option<PathBuf>,

    /// Removed: synthesis moved to rmpsynth. Kept only to say so.
    #[arg(short, long, hide = true)]
    out: Option<PathBuf>,

    /// Settings document (TOML). Defaults are used if omitted.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Also write the residual — what the decomposition failed to explain.
    #[arg(short, long)]
    residual: Option<PathBuf>,

    /// Write the book of atoms here. Format follows the extension: .toml or .json, either of which
    /// may carry a trailing .gz to be compressed. Render it with `rmpsynth -b`.
    #[arg(short, long)]
    book: Option<PathBuf>,

    /// Analyse from this offset into the file, in seconds. Defaults to the start.
    #[arg(short, long, value_name = "SECONDS")]
    start: Option<f32>,

    /// Analyse only this many seconds. Defaults to the rest of the file.
    #[arg(short, long, value_name = "SECONDS")]
    duration: Option<f32>,

    /// Analyse the final residue into an ERB power book, overriding [residual] enabled.
    #[arg(long)]
    residual_analysis: bool,

    /// Skip the residual analysis even if the settings document enables it.
    #[arg(long, conflicts_with = "residual_analysis")]
    no_residual_analysis: bool,

    /// Residual-book update interval, overriding [residual] update_ms.
    #[arg(long, value_name = "MS")]
    residual_update_ms: Option<f64>,

    /// Write the residual analysis to its own file instead of embedding it in --book. Same format
    /// rules as --book. The power matrix is far larger than the atom list, so this is how to keep
    /// the two apart.
    #[arg(long, value_name = "PATH")]
    residual_book: Option<PathBuf>,

    /// Print a fully-commented default settings document and exit.
    #[arg(long)]
    write_config: bool,

    /// Suppress progress reporting.
    #[arg(short, long)]
    quiet: bool,
}

impl Args {
    /// The residual toggle as an override: `None` when neither flag was given, so the settings
    /// document keeps its say. Precedence is defaults < settings file < CLI, and a flag that was
    /// not typed overrides nothing.
    fn residual_enabled(&self) -> Option<bool> {
        match (self.residual_analysis, self.no_residual_analysis) {
            (true, _) => Some(true),
            (_, true) => Some(false),
            _ => None,
        }
    }
}

fn main() -> ExitCode {
    rmp_core::threads::configure_pool();
    rmp_core::threads::prefer_fast_cores();
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rmp: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), String> {
    if args.write_config {
        print!("{DEFAULT_CONFIG_HEADER}{}", Config::default().to_toml());
        return Ok(());
    }

    // Synthesis used to live here, with `--book` read rather than written when no soundfile was
    // given. It is `rmpsynth`'s job now, and a stale command line should be told where it went
    // rather than fail as an unknown flag.
    if let Some(out) = &args.out {
        let book = args
            .book
            .as_deref()
            .map_or_else(|| "book.json".into(), |b| b.display().to_string());
        return Err(format!(
            "rmp no longer synthesises; render the book with `rmpsynth -b {book} -o {}`",
            out.display()
        ));
    }
    match args.input.as_deref() {
        Some(input) => analyse(args, input),
        None => Err(
            "give an input soundfile to analyse; to render a book, use `rmpsynth -b book -o out.wav`"
                .into(),
        ),
    }
}

/// Decompose a soundfile, and write whichever of the three outputs were asked for.
///
/// The decomposition itself is [`rmp_core::pipeline::analyse`]; what is left here is the command
/// line's own business — merging the flags into the settings document, reading the file, deciding
/// where each result goes, and printing progress through [`report::Cli`].
fn analyse(args: &Args, input: &Path) -> Result<(), String> {
    if args.book.is_none() && args.residual.is_none() && args.residual_book.is_none() {
        return Err(
            "nothing to write: give --book for the atoms, --residual for what is left over, or \
             --residual-book for its ERB analysis"
                .into(),
        );
    }

    // Defaults < settings file < CLI, and only for the fields actually given on the command line.
    // There is one configuration path, not two: the flags edit the document and everything
    // downstream reads the merged result.
    let mut config = Config::load(args.config.as_deref())?;
    if let Some(enabled) = args.residual_enabled() {
        config.residual.enabled = enabled;
    }
    if let Some(ms) = args.residual_update_ms {
        config.residual.update_ms = ms;
    }
    // `--residual-book` is a request for the analysis, not merely somewhere to put it.
    if args.residual_book.is_some() && args.residual_enabled().is_none() {
        config.residual.enabled = true;
    }
    config.validate()?;

    // ── read ────────────────────────────────────────────────────────────────
    let read = audio::read(input).map_err(|e| e.to_string())?;
    let downmixed = read.was_downmixed();
    let channels = read.channels;
    let whole = read.signal;
    let sr = whole.sample_rate;
    let whole_len = whole.len();
    let (offset, signal) = pipeline::excerpt(whole, args.start, args.duration)?;
    let duration = signal.len() as f32 / sr;

    let mut cli = report::Cli {
        quiet: args.quiet,
        sr,
        mp_cfg: config.mp_config(),
        max_memory_mb: config.pursuit.max_memory_mb,
    };
    cli.input(input, whole_len, channels, downmixed, offset, signal.len());
    if signal.energy() <= 0.0 {
        return Err(if signal.len() == whole_len {
            "input is silent".into()
        } else {
            "the selected excerpt is silent".to_string()
        });
    }

    // Resolved once, against the rate the file turns out to have, and before any work starts: an
    // ERB range that does not fit under this Nyquist is a settings error, and finding it out after
    // a seven-second analysis would be a waste of everyone's time.
    let residual_cfg = config
        .residual_config(sr as f64)
        .map_err(|e| e.to_string())?;
    if residual_cfg.enabled && args.book.is_none() && args.residual_book.is_none() {
        cli.say(
            "  warning: residual analysis is enabled but neither --book nor --residual-book was \
             given, so there is nowhere to put it; skipping",
        );
    }
    let run_residual =
        residual_cfg.enabled && (args.book.is_some() || args.residual_book.is_some());

    // ── analyse ─────────────────────────────────────────────────────────────
    let mut planner = Planner::new();
    let mut result = pipeline::analyse(
        pipeline::AnalysisRequest {
            signal: &signal,
            offset,
            config: &config,
            residual: run_residual.then_some(&residual_cfg),
        },
        &mut planner,
        &mut cli,
    )?;

    cli.analysis(&result, duration);
    cli.residual_levels(&result.residual, &signal);
    if let Some(rb) = &result.residual_book {
        cli.residual_analysis(rb, result.timing.residual, sr);
    }

    // ── write ───────────────────────────────────────────────────────────────
    if let Some(path) = &args.residual {
        // Taken from the pursuit rather than by subtracting the resynthesis, so it is exactly what
        // the algorithm could not explain.
        audio::write_samples(path, &result.residual, sr).map_err(|e| e.to_string())?;
        cli.say(&format!("wrote {}", path.display()));
    }

    // Written on its own when asked for, and only then embedded — the power matrix is far larger
    // than the atom list, and keeping both copies would double a file for nothing.
    if let Some(rb) = result.residual_book.take() {
        match &args.residual_book {
            Some(path) => {
                book::write_doc(path, &rb)?;
                cli.say(&format!("wrote {}", path.display()));
            }
            None => result.book.residual = Some(rb),
        }
    }

    if let Some(path) = &args.book {
        book::write(path, &result.book)?;
        cli.say(&format!("wrote {}", path.display()));
    }

    Ok(())
}
