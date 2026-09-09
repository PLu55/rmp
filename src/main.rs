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
        ("--residual-analysis", args.residual_analysis),
        ("--no-residual-analysis", args.no_residual_analysis),
        ("--residual-update-ms", args.residual_update_ms.is_some()),
        ("--residual-book", args.residual_book.is_some()),
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
    if args.out.is_none()
        && args.book.is_none()
        && args.residual.is_none()
        && args.residual_book.is_none()
    {
        return Err(
            "nothing to write: give --out for the resynthesis, --book for the atoms, \
             --residual for what is left over, or --residual-book for its ERB analysis"
                .into(),
        );
    }

    // Defaults < settings file < CLI, and only for the fields actually given on the command line.
    // There is one configuration path, not two: the flags edit the document and everything
    // downstream reads the merged result.
    let mut config = load_config(args.config.as_deref())?;
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

    // Resolved once, against the rate the file turns out to have, and before any work starts: an
    // ERB range that does not fit under this Nyquist is a settings error, and finding it out after
    // a seven-second analysis would be a waste of everyone's time.
    let residual_cfg = config
        .residual_config(sr as f64)
        .map_err(|e| e.to_string())?;
    if residual_cfg.enabled && args.book.is_none() && args.residual_book.is_none() {
        say(
            "  warning: residual analysis is enabled but neither --book nor --residual-book was \
             given, so there is nowhere to put it; skipping",
        );
    }
    let run_residual =
        residual_cfg.enabled && (args.book.is_some() || args.residual_book.is_some());

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

    let mp_cfg: MpConfig = config.mp_config();

    // A block whose support exceeds `max_atom_samples` can never be refined: `refine` asks the
    // envelope cache for the seed's own shape first, the cache refuses it as out of bounds, and
    // refinement declines. The atom is still selected — it is simply pinned to the grid, at grid
    // frequency, grid onset and grid envelope.
    //
    // Reported as a fact rather than a fault, because it cuts both ways. Leaving it unnoticed cost
    // 13 dB of residual peak on a low-alpha dictionary; but capping *deliberately* below the
    // longest block is also the best setting measured, since it stops refinement chasing
    // seven-second atoms it cannot converge on in the rounds available. The `refined:` line below
    // says what actually happened.
    if mp_cfg.refine.enabled {
        let stuck: Vec<&_> = dict
            .blocks
            .iter()
            .filter(|b| b.support_len() > mp_cfg.refine.max_atom_samples)
            .collect();
        if !stuck.is_empty() {
            let longest = stuck.iter().map(|b| b.support_len()).max().unwrap_or(0);
            say(&format!(
                "  note: {} of {} blocks are longer than refine.max_atom_samples ({}), so their \
                 atoms stay on the grid unrefined; longest support {} samples (alpha {:.3})",
                stuck.len(),
                dict.blocks.len(),
                mp_cfg.refine.max_atom_samples,
                longest,
                stuck
                    .iter()
                    .map(|b| b.env.params.alpha)
                    .fold(f32::INFINITY, f32::min),
            ));
        }
    }

    // ── analyse ─────────────────────────────────────────────────────────────
    let t = Instant::now();
    let mut mp = Mp::new(&dict, &signal, &mut planner);
    let init = t.elapsed();

    let t = Instant::now();
    let mut book = mp.run(&mp_cfg);
    let pursuit = t.elapsed();

    say(&format!(
        "analysis: {} atoms, {:.1} dB in {:.2?} (init {:.2?}, {:.1}x realtime)",
        book.len(),
        book.snr_db(),
        pursuit,
        init,
        (init + pursuit).as_secs_f32() / duration.max(1e-9)
    ));
    if !book.is_empty() && mp_cfg.refine.enabled {
        let refined = book.selections.iter().filter(|s| s.refined).count();
        say(&format!(
            "  refined: {refined}/{} atoms moved off the grid ({:.0}%)",
            book.len(),
            100.0 * refined as f64 / book.len() as f64
        ));
    }
    if book.is_empty() {
        say("  warning: no atoms selected — check the dictionary covers the signal's content");
    } else if book.snr_db() < mp_cfg.target_snr_db {
        say(&format!(
            "  stopped short of the {:.1} dB target (max_atoms = {})",
            mp_cfg.target_snr_db, mp_cfg.max_atoms
        ));
    }

    let (marked, resolved) = mp.lazy_stats();
    if marked > 0 {
        say(&format!(
            "  refresh: {marked} frames bounded, {resolved} recomputed ({:.1}%)",
            100.0 * resolved as f64 / marked as f64
        ));
        if std::env::var_os("RMP_REFRESH_DETAIL").is_some() {
            for (bi, &(m, r)) in mp.lazy_stats_per_block().iter().enumerate() {
                let b = &dict.blocks[bi];
                say(&format!(
                    "    block {bi:>2} alpha {:>6.1} fft {:>7}: {m:>8} bounded {r:>8} recomputed ({:>5.1}%)  ~{:.0} Msamples",
                    b.env.params.alpha, b.fft_len, 100.0 * r as f64 / m.max(1) as f64,
                    (r * b.fft_len) as f64 / 1e6
                ));
            }
        }
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

    // ── residual analysis ───────────────────────────────────────────────────
    // Strictly after the pursuit, on the pursuit's own residual buffer. It cannot change which
    // atoms were selected, and the book above is already final by the time this runs.
    if run_residual {
        let t = Instant::now();
        let rb = rmp::residual::analyze_residual(residual, sr as f64, offset as u64, &residual_cfg)
            .map_err(|e| format!("residual analysis: {e}"))?;
        let elapsed = t.elapsed();

        let taus = &rb.bank.power_detector.tau_seconds;
        let (tau_lo, tau_hi) = (
            taus.iter().cloned().fold(f64::INFINITY, f64::min) * 1e3,
            taus.iter().cloned().fold(0.0, f64::max) * 1e3,
        );
        say(&format!(
            "residual analysis: {} ERB bands, {:.1} .. {:.1} Hz, order {} gammatone, in {:.2?}",
            rb.band_count, rb.bank.min_freq_hz, rb.bank.max_freq_hz, rb.bank.filter_order, elapsed
        ));
        say(&format!(
            "  update: {} samples / {:.3} ms -> {} frames ({} values)",
            rb.update_samples,
            rb.update_samples as f64 * 1e3 / sr as f64,
            rb.frame_count,
            rb.power.len()
        ));
        say(&format!(
            "  power:  {}, tau {tau_lo:.2} .. {tau_hi:.2} ms",
            rb.bank.power_detector.mode
        ));
        if std::env::var_os("RMP_RESIDUAL_DETAIL").is_some() {
            say("    band   center_hz  bandwidth_hz  tau_ms   norm_gain");
            for (b, &tau) in taus.iter().enumerate() {
                say(&format!(
                    "    {b:>4}  {:>10.2}  {:>12.2}  {:>6.2}  {:>10.3e}",
                    rb.bank.center_freq_hz[b],
                    rb.bank.bandwidth_hz[b],
                    tau * 1e3,
                    rb.bank.normalization_gain[b],
                ));
            }
        }

        // Written on its own when asked for, and only then embedded — the power matrix is far
        // larger than the atom list, and keeping both copies would double a file for nothing.
        if let Some(path) = &args.residual_book {
            book::write_doc(path, &rb)?;
            say(&format!("wrote {}", path.display()));
        } else {
            book.residual = Some(rb);
        }
    }

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
#                      onset capture falls off as exp(-2*alpha*|delta|), and the
#                      frame count scales as 1/ln(1/tolerance). With refinement
#                      on, 0.5 is measured 5-6x faster than 0.95 on real material
#                      for about 8% more atoms to the same SNR; 0.3 is slower
#                      again. Without refinement leave it at 0.95.
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
#
# [residual]
#   Stochastic analysis of what the atoms could not explain. The pursuit leaves
#   x = sum(FOF) + r; this turns r into a fixed-rate map of power over ERB
#   bands, which a later noise bank can excite with g_b = sqrt(P_b). It is a
#   post-processing stage: it runs once, after the pursuit has stopped, and
#   cannot change which atoms were selected.
#
#   Off by default. --residual-analysis turns it on from the command line, and
#   --residual-book writes it to its own file.
#
#   enabled
#   update_ms   how often a power frame is recorded. Independent of the bank,
#               which always runs at the audio rate. Converted to an exact
#               number of samples once, and that count is what the book stores.
#               This is the size knob: 48 bands at 1 ms is 48000 numbers per
#               second of audio, which dwarfs the atom list. Give --book a .gz
#               suffix, or keep the two apart with --residual-book.
#   storage     f32 linear power. Not quantised.
#
# [residual.erb]
#   bands             band centres are uniform on the ERB-rate scale.
#   min_freq_hz       the first and last centres sit exactly on these, so the
#   max_freq_hz       range is the range asked for. max_freq_hz too close to
#                     Nyquist is an error rather than a silent clamp.
#   spacing, filter   erb_rate and gammatone. One value each today; they are
#   normalization     in the document so a book never has to be guessed at.
#   order             length of the complex one-pole cascade, 1 to 8. 4 is the
#                     classical gammatone.
#   normalization     unit_noise_power: each band's gain is measured from its
#                     own rendered impulse response, so unit-variance white
#                     noise leaves the band with unit variance and a band power
#                     can be read as a fraction of the residual's variance.
#
# [residual.power]
#   The only temporal smoothing in the chain, deliberately. A residual carries
#   rhythm and transients that the atom book does not, and blurring them would
#   throw away the part worth keeping.
#
#   mode          fixed uses tau_ms everywhere. bandwidth_relative sets
#                 tau_b = clamp(tau_scale / ERB(f_b), tau_min_ms, tau_max_ms):
#                 a 25 Hz band cannot resolve a half-millisecond event in the
#                 first place -- its own ringing is 40 ms long -- while a 3 kHz
#                 band can, and one constant either over-smooths the top of the
#                 bank or leaves the bottom reading its own envelope ripple.
#   tau_ms        used by mode = \"fixed\".
#   tau_scale     dimensionless; used by mode = \"bandwidth_relative\".
#   tau_min_ms    clamps on the bandwidth-relative time constant. The floor is
#   tau_max_ms    what bounds how long a transient can smear.

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
