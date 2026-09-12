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
use rmp_core::config::Config;
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
const DEFAULT_CONFIG_HEADER: &str = "\
# rmp analysis settings.
#
# Every section and field is optional; omitted values fall back to these defaults.
# Unknown keys are rejected rather than ignored, so a typo fails loudly.
#
# [dictionary.fof]
#   FOF atoms (formant wave functions), rendered through rfofs.
#   alphas          decay rates in s^-1. The -3 dB bandwidth is alpha/pi Hz, so
#                   80 -> 25 Hz and 2147 -> 683 Hz. A ratio of about 1.6 between
#                   neighbours costs roughly 5% of an atom's energy at the worst
#                   point between two rungs. Empty disables the family.
#   betas_ms        attack (skirt) durations. Shapes only the first few ms.
#   alpha_beta_max  combinations above this are dropped. rfofs renders
#                   alpha*beta > 10 as silence and its amplitude normalisation is
#                   ill-conditioned well before that.
#
# [dictionary.gaussian]
#   Symmetric Gaussian (Gabor) atoms, rendered by rmp itself. Off by default.
#   Blocks are built FOF family first, so adding this leaves the FOF blocks'
#   indices where they were.
#   sigmas_ms       envelope standard deviations. The -3 dB bandwidth is
#                   0.265/sigma Hz (5 ms -> 53 Hz) and the atom is 7.4 sigma
#                   long at the default cutoff. A ratio of about 2.5 between
#                   rungs matches refine.sigma_bracket; [1, 2.5, 6, 15, 40] is a
#                   reasonable first ladder. Empty disables the family.
#   cutoff_level    amplitude relative to the peak where the support is cut.
#                   0.001 is -60 dB; halving it lengthens every atom by about 5%.
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
#   max_memory_mb  frame-table budget, and so how long a stretch is analysed at
#                  once. The tables scale with the signal -- about 18 bytes per
#                  input sample on this default grid -- so a long clip is cut
#                  into windows. A signal that fits is one window and is
#                  decomposed exactly as it always was; past it, selection is
#                  greedy within a window rather than across the clip, and
#                  target_snr_db, min_gain and max_atoms become per-window.
#   window_seconds  analyse windows of exactly this length, ignoring
#                   max_memory_mb. The budget makes a book depend on the machine
#                   that produced it; set this when a run must reproduce
#                   elsewhere. 0 means use the budget.
#
#   max_atoms counts atoms actually selected; a rejected iteration adds nothing
#   to the book and is not charged to the budget.
#
# [refine]
#   Moves a selected atom off the grid before it is subtracted, by maximising
#   captured energy over t0, f and the shape one parameter at a time: alpha and
#   beta for a FOF, sigma for a gaussian (searched about the atom's centre).
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
#   sigma_*_ms              bounds on a refined gaussian's sigma.
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
