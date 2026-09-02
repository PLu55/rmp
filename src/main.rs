//! `rmp` — decompose a soundfile into FOF atoms and resynthesise it.
//!
//! ```text
//! rmp input.wav -o resynth.wav [-c settings.toml] [-r residual.wav] [-b book.toml]
//! rmp --write-config > settings.toml
//! ```

use clap::Parser;
use rmp::audio;
use rmp::book::Book;
use rmp::config::Config;
use rmp::dict::Dictionary;
use rmp::fft::Planner;
use rmp::mp::{Mp, MpConfig};
use rmp::signal::Signal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "rmp",
    about = "Matching-pursuit decomposition of audio into FOF atoms",
    version
)]
struct Args {
    /// Input soundfile (WAV, AIFF, FLAC). Multi-channel input is downmixed to mono.
    #[arg(required_unless_present = "write_config")]
    input: Option<PathBuf>,

    /// Resynthesised output, as 32-bit float WAV.
    #[arg(short, long, required_unless_present = "write_config")]
    out: Option<PathBuf>,

    /// Settings document (TOML). Defaults are used if omitted.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Also write the residual — what the decomposition failed to explain.
    #[arg(short, long)]
    residual: Option<PathBuf>,

    /// Also write the book of recovered atoms. Format follows the extension: .toml or .json.
    #[arg(short, long)]
    book: Option<PathBuf>,

    /// Print a fully-commented default settings document and exit.
    #[arg(long)]
    write_config: bool,

    /// Suppress progress reporting.
    #[arg(short, long)]
    quiet: bool,
}

fn main() -> ExitCode {
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

    let input = args.input.as_ref().expect("required by clap");
    let out = args.out.as_ref().expect("required by clap");

    let config = load_config(args.config.as_deref())?;
    config.validate()?;

    // ── read ────────────────────────────────────────────────────────────────
    let read = audio::read(input).map_err(|e| e.to_string())?;
    let downmixed = read.was_downmixed();
    let channels = read.channels;
    let signal = read.signal;
    let sr = signal.sample_rate;
    let duration = signal.len() as f32 / sr;

    let say = |m: &str| {
        if !args.quiet {
            eprintln!("{m}");
        }
    };
    say(&format!(
        "{}: {:.2} s, {} Hz, {} channel(s)",
        input.display(),
        duration,
        sr as u32,
        channels
    ));
    if downmixed {
        say("  downmixed to mono; out-of-phase content between channels partially cancels");
    }
    if signal.energy() <= 0.0 {
        return Err("input is silent".into());
    }

    // ── dictionary ──────────────────────────────────────────────────────────
    // Built at the file's own sample rate: hop and bin spacing both depend on it.
    let mut planner = Planner::new();
    let t = Instant::now();
    let dict = Dictionary::from_grid(
        &config.dictionary.grid(),
        sr,
        &mut planner,
        &config.block_config(),
    )
    .map_err(|e| format!("building dictionary: {e}"))?;
    say(&format!(
        "dictionary: {} blocks in {:.2?}",
        dict.blocks.len(),
        t.elapsed()
    ));

    // ── analyse ─────────────────────────────────────────────────────────────
    let t = Instant::now();
    let mut mp = Mp::new(&dict, &signal, &mut planner);
    let init = t.elapsed();

    let mp_cfg: MpConfig = (&config.pursuit).into();
    let t = Instant::now();
    let book = mp.run(&mp_cfg);
    let pursuit = t.elapsed();

    say(&format!(
        "analysis: {} atoms, {:.1} dB in {:.2?} (init {:.2?}, {:.1}x realtime)",
        book.len(),
        book.snr_db(),
        pursuit,
        init,
        (init + pursuit).as_secs_f32() / duration.max(1e-9)
    ));
    if book.is_empty() {
        say("  warning: no atoms selected — check the dictionary covers the signal's content");
    } else if book.snr_db() < mp_cfg.target_snr_db {
        say(&format!(
            "  stopped short of the {:.1} dB target (max_atoms = {})",
            mp_cfg.target_snr_db, mp_cfg.max_atoms
        ));
    }

    // ── write ───────────────────────────────────────────────────────────────
    let resynth = book
        .resynthesize(signal.len())
        .map_err(|e| format!("resynthesis: {e}"))?;
    audio::write(out, &resynth).map_err(|e| e.to_string())?;
    say(&format!("wrote {}", out.display()));

    if let Some(path) = &args.residual {
        // Taken from the pursuit rather than by subtracting the resynthesis, so it is exactly what
        // the algorithm could not explain.
        let residual = Signal::new(mp.residual().to_vec(), sr);
        audio::write(path, &residual).map_err(|e| e.to_string())?;
        say(&format!("wrote {}", path.display()));
    }

    if let Some(path) = &args.book {
        write_book(path, &book)?;
        say(&format!("wrote {}", path.display()));
    }

    Ok(())
}

fn load_config(path: Option<&Path>) -> Result<Config, String> {
    let Some(path) = path else {
        return Ok(Config::default());
    };
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("reading {}: {e}", path.display()))?;
    Config::from_toml(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
}

/// Serialise the book, picking the format from the file extension.
fn write_book(path: &Path, book: &Book) -> Result<(), String> {
    let text = match path.extension().and_then(|e| e.to_str()) {
        Some("json") => serde_json::to_string_pretty(book)
            .map_err(|e| format!("serialising book: {e}"))?,
        Some("toml") | None => {
            toml::to_string_pretty(book).map_err(|e| format!("serialising book: {e}"))?
        }
        Some(other) => {
            return Err(format!(
                "unknown book format '.{other}' — use .toml or .json"
            ));
        }
    };
    std::fs::write(path, text).map_err(|e| format!("writing {}: {e}", path.display()))
}

const DEFAULT_CONFIG_HEADER: &str = "\
# rmp analysis settings.
#
# Every section and field is optional; omitted values fall back to these defaults.
# Unknown keys are rejected rather than ignored, so a typo fails loudly.
#
# [dictionary]
#   alphas          decay rates in s^-1. The -3 dB bandwidth is alpha/pi Hz, so
#                   80 -> 25 Hz and 2147 -> 683 Hz. A ratio of about 1.6 between
#                   neighbours costs roughly 5% of an atom's energy at the worst
#                   point between two rungs.
#   betas_ms        attack (skirt) durations. Shapes only the first few ms.
#   alpha_beta_max  combinations above this are dropped. rfofs renders
#                   alpha*beta > 10 as silence and its amplitude normalisation is
#                   ill-conditioned well before that.
#
# [envelope]
#   The release policy, fixed for the whole analysis and shared with resynthesis.
#   rfofs's release is a linear ramp to zero, entered where the raw exponential
#   decay reaches fade_level.
#
#   fade_level                    -60 dB by default.
#   fade_dur_scale                the release lasts fade_dur_scale/alpha seconds,
#                                 not a fixed time: a constant duration would run
#                                 several times longer than the atom body at large
#                                 alpha, inflating that block's FFT length for no
#                                 representational gain.
#   fade_dur_min_ms, _max_ms      clamps on that duration.
#
# [blocks]
#   capture_tolerance  worst-case fraction of an atom's energy a frame must still
#                      capture when the true onset falls between hop positions.
#                      This sets the hop, and so most of the analysis cost:
#                      onset capture falls off as exp(-2*alpha*|delta|).
#   f_min, f_max       frequency range represented, in Hz.
#   rho_sq_max         disables bins where the sine and cosine basis vectors are
#                      nearly parallel and the projection is ill-conditioned.
#
# [pursuit]
#   max_atoms      hard cap on atoms selected.
#   target_snr_db  stop once reconstruction reaches this SNR.
#   min_gain       stop when the best atom would remove less than this fraction
#                  of the remaining residual.

";
