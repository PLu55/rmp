//! `rmp` — decompose a soundfile into FOF atoms and resynthesise it.
//!
//! ```text
//! rmp input.wav -o resynth.wav [-c settings.toml] [-r residual.wav] [-b book.toml]
//!     [-s start_seconds] [-d duration_seconds]
//! rmp input.wav -b book.toml            # analyse only, no resynthesis
//! rmp -b book.toml -o resynth.wav       # synthesise a book, no analysis
//! rmp --write-config > settings.toml
//! ```
//!
//! The two roles of `--book` are told apart by whether an input soundfile is given: with one it is
//! written, without one it is read.

use clap::Parser;
use rmp::audio;
use rmp::book;
use rmp::config::Config;
use rmp::dict::Dictionary;
use rmp::fft::Planner;
use rmp::mp::{Mp, MpConfig};
use rmp::signal::{db_fs, peak_of, rms_of, Signal};
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
    /// Input soundfile (WAV, AIFF, FLAC). Multi-channel input is downmixed to mono. Omit it to
    /// synthesise the book given by --book instead of analysing anything.
    input: Option<PathBuf>,

    /// Resynthesised output, as 32-bit float WAV. Optional when analysing — omitting it and
    /// giving --book analyses without rendering the result.
    #[arg(short, long)]
    out: Option<PathBuf>,

    /// Settings document (TOML). Defaults are used if omitted.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Also write the residual — what the decomposition failed to explain.
    #[arg(short, long)]
    residual: Option<PathBuf>,

    /// The book of atoms: written when a soundfile is analysed, read and synthesised when no
    /// input soundfile is given. Format follows the extension: .toml or .json, either of which
    /// may carry a trailing .gz to be compressed.
    #[arg(short, long)]
    book: Option<PathBuf>,

    /// Analyse from this offset into the file, in seconds. Defaults to the start.
    #[arg(short, long, value_name = "SECONDS")]
    start: Option<f32>,

    /// Analyse only this many seconds. Defaults to the rest of the file.
    #[arg(short, long, value_name = "SECONDS")]
    duration: Option<f32>,

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

    // `--book` is an output when there is something to analyse and an input when there is not.
    // Nothing else distinguishes the two modes: a book carries its own sample rate and its atoms
    // are the whole of what synthesis needs.
    match (args.input.as_deref(), args.book.as_deref()) {
        (Some(input), _) => analyse(args, input),
        (None, Some(book)) => synthesise(args, book),
        (None, None) => Err(
            "give an input soundfile to analyse, or a book to synthesise with --book".into(),
        ),
    }
}

/// Render a book back to audio, with no analysis and no dictionary.
///
/// The book is replayed in the frame it was analysed in: sample 0 of the output is the origin the
/// atom onsets are relative to, and an atom that began before it is clipped there. So a book from
/// `--start 2.5` synthesises the excerpt, not the file.
fn synthesise(args: &Args, book_path: &Path) -> Result<(), String> {
    let Some(out) = args.out.as_deref() else {
        return Err(format!(
            "synthesising {} needs somewhere to write it: give --out",
            book_path.display()
        ));
    };
    for (flag, unused) in [
        ("--config", args.config.is_some()),
        ("--residual", args.residual.is_some()),
        ("--start", args.start.is_some()),
        ("--duration", args.duration.is_some()),
    ] {
        if unused {
            return Err(format!("{flag} applies to analysis; synthesising a book ignores it"));
        }
    }

    let say = |m: &str| {
        if !args.quiet {
            eprintln!("{m}");
        }
    };

    let book = book::read(book_path)?;
    if book.is_empty() {
        return Err(format!("{} contains no atoms", book_path.display()));
    }
    if !(book.sample_rate > 0.0 && book.sample_rate.is_finite()) {
        return Err(format!(
            "{} has an unusable sample rate ({})",
            book_path.display(),
            book.sample_rate
        ));
    }

    let t = Instant::now();
    let len = book.natural_len().map_err(|e| format!("sizing the book: {e}"))?;
    let signal = book
        .resynthesize(len)
        .map_err(|e| format!("resynthesis: {e}"))?;
    say(&format!(
        "{}: {} atoms, {} Hz, {:.2} s reconstructed at {:.1} dB in {:.2?}",
        book_path.display(),
        book.len(),
        book.sample_rate as u32,
        len as f32 / book.sample_rate,
        book.snr_db(),
        t.elapsed()
    ));

    audio::write(out, &signal).map_err(|e| e.to_string())?;
    say(&format!("wrote {}", out.display()));
    Ok(())
}

/// Decompose a soundfile, and write whichever of the three outputs were asked for.
fn analyse(args: &Args, input: &Path) -> Result<(), String> {
    if args.out.is_none() && args.book.is_none() && args.residual.is_none() {
        return Err(
            "nothing to write: give --out for the resynthesis, --book for the atoms, or \
             --residual for what is left over"
                .into(),
        );
    }

    let config = load_config(args.config.as_deref())?;
    config.validate()?;

    // ── read ────────────────────────────────────────────────────────────────
    let read = audio::read(input).map_err(|e| e.to_string())?;
    let downmixed = read.was_downmixed();
    let channels = read.channels;
    let whole = read.signal;
    let sr = whole.sample_rate;
    let whole_len = whole.len();
    let (offset, signal) = excerpt(whole, args.start, args.duration)?;
    let duration = signal.len() as f32 / sr;

    let say = |m: &str| {
        if !args.quiet {
            eprintln!("{m}");
        }
    };
    say(&format!(
        "{}: {:.2} s, {} Hz, {} channel(s)",
        input.display(),
        whole_len as f32 / sr,
        sr as u32,
        channels
    ));
    if signal.len() != whole_len {
        // Everything downstream — atom onsets in the book, the resynthesis, the residual — is
        // relative to this excerpt, not to the file.
        say(&format!(
            "  analysing {:.3}-{:.3} s ({} samples from {})",
            offset as f32 / sr,
            (offset + signal.len()) as f32 / sr,
            signal.len(),
            offset
        ));
    }
    if downmixed {
        say("  downmixed to mono; out-of-phase content between channels partially cancels");
    }
    if signal.energy() <= 0.0 {
        return Err(if signal.len() == whole_len {
            "input is silent".into()
        } else {
            "the selected excerpt is silent".to_string()
        });
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

    let mp_cfg: MpConfig = config.mp_config();
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

    // The residual is the pursuit's own working buffer, so this is the level of what the
    // decomposition could not explain.
    //
    // Reported relative to the input first, because that is the figure you act on — an absolute
    // dBFS residual means nothing without knowing how loud the input was. The rms ratio is the
    // negated SNR by construction; it is restated here so the two absolute levels beside it do not
    // have to be subtracted by eye. The peak ratio is the one that carries new information: it is
    // where the decomposition is worst rather than where it is on average.
    let residual = mp.residual();
    let (r_rms, r_peak) = (rms_of(residual), peak_of(residual) as f64);
    let (s_rms, s_peak) = (signal.rms(), signal.peak() as f64);
    say(&format!(
        "residual: {:+.1} dB rms, {:+.1} dB peak relative to input",
        db_fs(r_rms) - db_fs(s_rms),
        db_fs(r_peak) - db_fs(s_peak),
    ));
    say(&format!(
        "  absolute: {:.1} dBFS rms, {:.1} dBFS peak (input {:.1} dBFS rms, {:.1} dBFS peak)",
        db_fs(r_rms),
        db_fs(r_peak),
        db_fs(s_rms),
        db_fs(s_peak),
    ));

    // ── write ───────────────────────────────────────────────────────────────
    // Resynthesis is skipped outright when no --out was given: it renders every atom a second
    // time, which is not free on a large book, and an analysis-only run has no use for it.
    if let Some(out) = &args.out {
        let resynth = book
            .resynthesize(signal.len())
            .map_err(|e| format!("resynthesis: {e}"))?;
        audio::write(out, &resynth).map_err(|e| e.to_string())?;
        say(&format!("wrote {}", out.display()));
    }

    if let Some(path) = &args.residual {
        // Taken from the pursuit rather than by subtracting the resynthesis, so it is exactly what
        // the algorithm could not explain.
        let residual = Signal::new(mp.residual().to_vec(), sr);
        audio::write(path, &residual).map_err(|e| e.to_string())?;
        say(&format!("wrote {}", path.display()));
    }

    if let Some(path) = &args.book {
        book::write(path, &book)?;
        say(&format!("wrote {}", path.display()));
    }

    Ok(())
}

/// Cut `[start, start + duration)` out of the signal, in seconds.
///
/// Returns the sample offset the excerpt begins at along with the excerpt itself. `None` for
/// either bound means "the whole file" in that direction; the end is clamped to the file, so
/// asking for more seconds than remain is not an error, but starting past the end is.
fn excerpt(
    signal: Signal,
    start: Option<f32>,
    duration: Option<f32>,
) -> Result<(usize, Signal), String> {
    if start.is_none() && duration.is_none() {
        return Ok((0, signal));
    }
    let sr = signal.sample_rate;

    let start = start.unwrap_or(0.0);
    if !(start.is_finite() && start >= 0.0) {
        return Err(format!("--start must be a non-negative number of seconds, got {start}"));
    }
    let begin = (start as f64 * sr as f64).round() as usize;
    if begin >= signal.len() {
        return Err(format!(
            "--start {start} s is at or past the end of the {:.3} s input",
            signal.len() as f32 / sr
        ));
    }

    let end = match duration {
        None => signal.len(),
        Some(d) => {
            if !(d.is_finite() && d > 0.0) {
                return Err(format!("--duration must be a positive number of seconds, got {d}"));
            }
            let n = (d as f64 * sr as f64).round() as usize;
            if n == 0 {
                return Err(format!(
                    "--duration {d} s is under one sample at {} Hz",
                    sr as u32
                ));
            }
            signal.len().min(begin + n)
        }
    };

    Ok((begin, Signal::new(signal.samples[begin..end].to_vec(), sr)))
}

fn load_config(path: Option<&Path>) -> Result<Config, String> {
    let Some(path) = path else {
        return Ok(Config::default());
    };
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("reading {}: {e}", path.display()))?;
    Config::from_toml(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
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
#   candidate_count  local time-frequency maxima promoted to exact scoring each
#                    iteration. 1 is the plain global argmax. Measured, raising
#                    it changes nothing: the strongest seed is also the seed
#                    that refines best. It matters only when a candidate can be
#                    rejected outright rather than merely outscored.
#   max_stalls     give up after this many consecutive iterations in which every
#                  candidate was rejected. Reachable only under HRMP, and it
#                  wants to be generous: a rejection is one frame's residual
#                  declining one proposed atom, and on dense material a long run
#                  of them is ordinary. A small value turns a strict HRMP setting
#                  into an early stop that looks like 'HRMP finds no atoms'.
#
#   max_atoms counts atoms actually selected; a rejected iteration adds nothing
#   to the book and is not charged to the budget.
#
# [refine]
#   Moves a selected atom off the grid before it is subtracted, by maximising
#   captured energy over t0, f, alpha and beta one parameter at a time.
#   Amplitude and phase are never searched -- they come out of the projection in
#   closed form. This is the single largest win available: off-grid input needs
#   12x more atoms than on-grid without it and 3x with it, for about 2% of an
#   iteration's cost.
#
#   enabled                 turn refinement off to get the plain grid pursuit.
#   rounds, golden_iters    work per candidate, and the dominant cost knobs.
#   score_tol               stop early when a whole round gains less than this.
#   alpha_*, beta_*_ms      bounds on the refined envelope, deliberately wider
#                           than the dictionary grid at both ends.
#   max_atom_samples        hard cap on the refined support, whatever the bounds
#                           imply -- a small alpha is a very long atom.
#   *_bracket, t0_radius    how far from the seed to search. The defaults reach
#                           the neighbouring grid rung in each direction, so no
#                           true value is out of reach. t0_radius = 0 derives
#                           the onset radius from the block's own hop.
#
# [hrmp]
#   High-Resolution Matching Pursuit. Ordinary MP scores an atom by its global
#   correlation, so a long atom can win by summing evidence from two separated
#   events and claiming the silence between them -- pre-echo, and energy
#   invented in gaps. HRMP asks whether the residual supports the atom
#   *everywhere* it claims to be, and clamps or rejects it if not.
#
#   Off by default: it is a stricter criterion than ordinary MP, so it trades
#   reconstruction SNR per atom for atoms that describe events actually present.
#
#   enabled
#   mode                    localized_candidate masks the refined atom's own
#                           envelope; legacy_scaled_fof uses smaller
#                           same-frequency FOFs, as in the historical code.
#   depth                   2^depth probes, placed at equal-energy quantiles
#                           rather than equal time: a FOF decays 60 dB across
#                           its support, so uniform probes would leave the late
#                           ones reading noise, and a strict minimum over
#                           unequal variances measures the noisiest probe
#                           instead of the least supported region.
#                           It is also the strictness knob. Rejection is 'any
#                           probe disagrees', so the rejection rate climbs with
#                           the probe count: on piano at 48 kHz, depth 2 rejects
#                           roughly as many candidates as it accepts where depth
#                           1 rejects none and still clamps 85% of them.
#   phase_tolerance_deg     reject when a probe's local phase disagrees with the
#                           global fit by more than this. Capped in effect at 90:
#                           the sign rule rejects anything beyond a quarter turn
#                           by itself, so larger values are no-ops. Smaller ones
#                           bite hard -- 45 rejects the large majority of
#                           candidates on polyphonic material, because a local
#                           residual carrying other events routinely sits a
#                           quarter turn from the global fit.
#   minimum_probe_energy    structural floor on a probe's share of the atom's
#                           energy before it gets a vote.
#   noise_epsilon           bound on each local amplitude's relative standard
#                           error. A probe must see at least 1/noise_epsilon^2
#                           times the local residual noise power in atom energy
#                           to be believed. This is the gate that does the work.
#   min_mask_periods        skip HRMP when a probe would span fewer carrier
#                           periods than this: its Gram cannot be conditioned,
#                           and an atom that short cannot bridge anything.
#   magnitude_policy        strict_min is the original criterion.

";

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Signal {
        Signal::new((0..n).map(|i| i as f32).collect(), 1000.0)
    }

    #[test]
    fn no_bounds_returns_the_whole_signal() {
        let (off, got) = excerpt(ramp(500), None, None).unwrap();
        assert_eq!((off, got.len()), (0, 500));
    }

    #[test]
    fn start_and_duration_cut_a_window() {
        let (off, got) = excerpt(ramp(500), Some(0.1), Some(0.2)).unwrap();
        assert_eq!((off, got.len()), (100, 200));
        assert_eq!(got.samples[0], 100.0);
    }

    #[test]
    fn duration_is_clamped_to_the_end_of_the_file() {
        let (off, got) = excerpt(ramp(500), Some(0.4), Some(9.0)).unwrap();
        assert_eq!((off, got.len()), (400, 100));
    }

    #[test]
    fn start_alone_runs_to_the_end() {
        let (off, got) = excerpt(ramp(500), Some(0.25), None).unwrap();
        assert_eq!((off, got.len()), (250, 250));
    }

    #[test]
    fn out_of_range_or_negative_bounds_are_errors() {
        assert!(excerpt(ramp(500), Some(0.5), None).is_err());
        assert!(excerpt(ramp(500), Some(-0.1), None).is_err());
        assert!(excerpt(ramp(500), None, Some(0.0)).is_err());
        // Rounds to zero samples at 1 kHz.
        assert!(excerpt(ramp(500), None, Some(1e-4)).is_err());
    }
}
