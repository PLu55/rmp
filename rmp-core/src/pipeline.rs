//! One analysis, from a signal to a book, driven the same way by every front end.
//!
//! [`analyse`] is what `rmp` the binary does between reading the soundfile and writing the outputs:
//! build the dictionary, plan the windows, run the pursuit, and analyse what is left over. It
//! exists as a library function rather than as the body of `main` so that a graphical front end and
//! the command line cannot drift into two different decompositions of the same input.
//!
//! **It computes and reports facts; it never formats a line of text.** That is the same rule
//! [`crate::stats`] follows, and it is what lets the CLI keep printing exactly the progress it
//! always did while a GUI renders the same facts as widgets. Two things follow from it:
//!
//! - The [`Event`] stream carries only what happens *while work is in progress* — the three points
//!   where the pipeline pauses long enough to have something to say. Everything a front end
//!   reports at the end (atom counts, SNR, how many atoms refined, the residual's level, the
//!   refresh counters) is a field of [`Analysis`] or is derived from the book, so it needs no
//!   event and arrives in whatever order the caller wants to present it.
//! - Nothing here opens or writes a file. Reading the soundfile, cutting the excerpt out of it and
//!   deciding where the results go are the caller's, which is also what keeps a GUI free to
//!   analyse a buffer it never read from disk.
//!
//! The residual analysis is resolved by the *caller* — [`AnalysisRequest::residual`] is already a
//! [`ResidualAnalysisConfig`] rather than a flag — because an ERB range that does not fit under the
//! input's Nyquist is a settings error, and finding it out after a seven-second analysis would be a
//! waste of everyone's time. Resolving it before the request is built is what makes that failure
//! immediate.

use crate::book::Book;
use crate::config::Config;
use crate::dict::Dictionary;
use crate::fft::Planner;
use crate::mp::{self, MpConfig, WindowPlan};
use crate::residual::book::ResidualBook;
use crate::residual::config::ResidualAnalysisConfig;
use crate::signal::Signal;
use std::time::{Duration, Instant};

/// What to decompose, and how.
pub struct AnalysisRequest<'a> {
    /// The excerpt itself. Already cut — see [`excerpt`].
    pub signal: &'a Signal,
    /// Where that excerpt began in its source, in samples. Recorded on the book as
    /// `start_sample`, which is what puts a rendered book back where it came from.
    pub offset: usize,
    pub config: &'a Config,
    /// Resolved residual settings, or `None` to skip the stage. Get one from
    /// [`Config::residual_config`], which needs the sample rate the file turns out to have.
    pub residual: Option<&'a ResidualAnalysisConfig>,
}

/// Something the pipeline has to say while it is still working.
///
/// Every variant borrows rather than owns, because the caller that wants to *print* it needs
/// nothing else, and the caller that wants to keep it — a GUI forwarding to another thread — knows
/// better than this module which parts are worth copying.
pub enum Event<'a> {
    /// The dictionary is built. Its blocks are worth inspecting here: which of them exceed
    /// `refine.max_atom_samples`, and so will never be refined, is a fact a front end should
    /// surface, and it is derivable from this and the config alone.
    Dictionary { dict: &'a Dictionary, elapsed: Duration },
    /// The signal exceeded the frame-table budget and will be decomposed a window at a time.
    /// Not emitted for a single window, which is the ordinary path.
    Windows { plan: &'a WindowPlan },
    /// One window is finished. `atoms` is the running total over the whole clip.
    Window { index: usize, of: usize, atoms: usize },
}

/// How a front end watches a run, and how it stops one.
///
/// Both methods have defaults, so `()` is a complete implementation for a caller that wants
/// neither — which is what [`analyse`]'s own tests and any batch use should pass.
pub trait Reporter {
    fn event(&mut self, _: Event<'_>) {}

    /// Polled once per atom and once per window. Must be **sticky**: once it answers true it has to
    /// keep answering true, because that is how [`Analysis::cancelled`] is decided after the fact.
    fn cancelled(&self) -> bool {
        false
    }
}

impl Reporter for () {}

/// Where the time went. Reported rather than derived because only the pipeline can see the split.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    pub dictionary: Duration,
    /// Building the correlator, frame tables and trees, summed over windows.
    pub init: Duration,
    /// The pursuit proper, with `init` already taken out.
    pub pursuit: Duration,
    pub residual: Duration,
}

impl Timing {
    /// Wall clock per second of audio: below 1 is faster than realtime.
    ///
    /// `init + pursuit`, and deliberately *not* the dictionary — that is built once per settings
    /// rather than once per second of audio, so folding it in would make the rate depend on how
    /// short the excerpt was. Nor the residual analysis, which is 0.5 ms for 0.5 s of audio and
    /// would not move the figure it appeared in.
    ///
    /// Here rather than in a front end because both of them report it, and a rate that meant one
    /// thing in the terminal and another in the window would be worse than no rate at all.
    pub fn realtime_factor(&self, seconds: f32) -> f32 {
        (self.init + self.pursuit).as_secs_f32() / seconds.max(1e-9)
    }
}

/// Lazy-refresh counters: frames given a bound, and frames that then had to be recomputed.
#[derive(Debug, Clone, Default)]
pub struct Refresh {
    pub marked: usize,
    pub resolved: usize,
    pub per_block: Vec<(usize, usize)>,
    /// Decimated frames recomputed above their bound; see `[blocks] decimate`.
    pub undercuts: usize,
}

/// Everything one analysis produced.
pub struct Analysis {
    pub book: Book,
    /// What the decomposition could not explain — the pursuit's own working buffer, so this is the
    /// residue itself rather than the input minus a resynthesis.
    pub residual: Vec<f32>,
    /// The ERB analysis of that residue, when one was asked for. Beside the book rather than inside
    /// it: whether the two share a file is the caller's decision.
    pub residual_book: Option<ResidualBook>,
    /// The dictionary the pursuit ran against, returned because a front end wants it afterwards to
    /// interpret a selection's `block` and to describe the shapes on offer.
    pub dict: Dictionary,
    pub timing: Timing,
    pub refresh: Refresh,
    /// True when [`Reporter::cancelled`] ended the run. The book is whatever had been selected by
    /// then, which is a valid decomposition of a prefix — but not one that met `target_snr_db`,
    /// and nothing about the book itself says so.
    pub cancelled: bool,
}

/// Decompose one excerpt.
///
/// The residual analysis is strictly after the pursuit, on the pursuit's own residual buffer: it
/// cannot change which atoms were selected, and the book is already final by the time it runs. It
/// is returned beside the book rather than embedded in it, because whether the two share a file is
/// the caller's decision — the power matrix is far larger than the atom list.
pub fn analyse(
    req: AnalysisRequest<'_>,
    planner: &mut Planner,
    report: &mut dyn Reporter,
) -> Result<Analysis, String> {
    let AnalysisRequest { signal, offset, config, residual: residual_cfg } = req;
    let sr = signal.sample_rate;
    let mut timing = Timing::default();

    // Built at the file's own sample rate: hop and bin spacing both depend on it.
    let t = Instant::now();
    let dict = Dictionary::from_shapes(
        &config.dictionary_shapes(),
        sr,
        planner,
        &config.block_config(),
    )
    .map_err(|e| format!("building dictionary: {e}"))?;
    timing.dictionary = t.elapsed();
    report.event(Event::Dictionary { dict: &dict, elapsed: timing.dictionary });

    let mp_cfg: MpConfig = config.mp_config();

    // The frame tables scale with the signal at a per-sample cost the dictionary sets, so a clip
    // long enough to exceed the budget is decomposed a window at a time. Under the budget this
    // plans a single window and takes the original path, bit for bit.
    let forced_core = (config.pursuit.window_seconds > 0.0)
        .then_some((config.pursuit.window_seconds * sr) as usize);
    let plan = WindowPlan::new(
        &dict,
        signal.len(),
        &mp_cfg,
        config.pursuit.max_memory_mb << 20,
        forced_core,
    );
    if plan.count > 1 {
        report.event(Event::Windows { plan: &plan });
    }

    // The pursuit wants a `&mut` progress sink and a `&` cancel predicate at the same time, and
    // both are the one reporter. A `RefCell` is what reconciles them; the two are never called
    // nested, so it cannot conflict at runtime.
    let t = Instant::now();
    let report = std::cell::RefCell::new(report);
    let run = {
        let mut progress = |w: usize, of: usize, atoms: usize| {
            report.borrow_mut().event(Event::Window { index: w, of, atoms });
        };
        let cancel = || report.borrow().cancelled();
        mp::run_windowed(&dict, signal, planner, &mp_cfg, &plan, &mut progress, &cancel)
    };
    let report = report.into_inner();
    timing.init = run.init;
    timing.pursuit = t.elapsed().saturating_sub(run.init);

    let mut book = run.book;
    // Onsets are relative to the excerpt; this is what puts a rendered book back where it came from.
    book.start_sample = offset as u64;

    // Strictly after the pursuit, on the pursuit's own residual buffer.
    let mut residual_book = None;
    if let Some(cfg) = residual_cfg {
        let t = Instant::now();
        residual_book = Some(
            crate::residual::analyze_residual(&run.residual, sr as f64, offset as u64, cfg)
                .map_err(|e| format!("residual analysis: {e}"))?,
        );
        timing.residual = t.elapsed();
    }

    Ok(Analysis {
        book,
        residual: run.residual,
        residual_book,
        dict,
        timing,
        refresh: Refresh {
            marked: run.marked,
            resolved: run.resolved,
            per_block: run.per_block,
            undercuts: run.undercuts,
        },
        cancelled: report.cancelled(),
    })
}

/// Cut `[start, start + duration)` out of the signal, in seconds.
///
/// Returns the sample offset the excerpt begins at along with the excerpt itself. `None` for
/// either bound means "the whole file" in that direction; the end is clamped to the file, so
/// asking for more seconds than remain is not an error, but starting past the end is.
pub fn excerpt(
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

/// Blocks whose support exceeds `refine.max_atom_samples`, and so can never be refined.
///
/// `refine` asks the envelope cache for the seed's own shape first, the cache refuses it as out of
/// bounds, and refinement declines. The atom is still selected — it is simply pinned to the grid,
/// at grid frequency, grid onset and grid envelope.
///
/// A fact rather than a fault, because it cuts both ways. Leaving it unnoticed cost 13 dB of
/// residual peak on a low-alpha dictionary; but capping *deliberately* below the longest block is
/// also the best setting measured, since it stops refinement chasing seven-second atoms it cannot
/// converge on in the rounds available.
///
/// Here rather than in a front end because both of them want to say it, and the rule for what
/// counts is `refine`'s, not a presentation choice.
pub fn unrefinable_blocks<'a>(dict: &'a Dictionary, cfg: &MpConfig) -> Vec<&'a crate::dict::Block> {
    if !cfg.refine.enabled {
        return Vec::new();
    }
    dict.blocks.iter().filter(|b| b.support_len() > cfg.refine.max_atom_samples).collect()
}

#[cfg(test)]
mod tests {

    /// The realtime factor is `init + pursuit` over the excerpt, and nothing else.
    ///
    /// The dictionary is deliberately out: it is built once per settings rather than once per
    /// second of audio, so counting it would make a short excerpt look slow and a long one fast for
    /// no reason but its length. CLAUDE.md measures that build at 64 ms against 2.4 ms across two
    /// configs of one piece of material — enough to dominate the figure on a one-second excerpt.
    #[test]
    fn the_realtime_factor_counts_the_pursuit_and_not_the_dictionary() {
        let t = Timing {
            dictionary: Duration::from_millis(500),
            init: Duration::from_millis(100),
            pursuit: Duration::from_millis(900),
            residual: Duration::from_millis(50),
        };
        // 1.0 s of work over 2.0 s of audio.
        assert!((t.realtime_factor(2.0) - 0.5).abs() < 1e-6, "got {}", t.realtime_factor(2.0));

        // Neither of the excluded phases moves it.
        let mut u = t;
        u.dictionary = Duration::from_secs(60);
        u.residual = Duration::from_secs(60);
        assert_eq!(u.realtime_factor(2.0), t.realtime_factor(2.0));
    }

    /// A zero-length excerpt must not divide by zero and report infinity.
    #[test]
    fn the_realtime_factor_survives_an_empty_excerpt() {
        let t = Timing { pursuit: Duration::from_millis(1), ..Timing::default() };
        assert!(t.realtime_factor(0.0).is_finite());
    }

    use super::*;

    /// A signal with something in it, and a dictionary small enough to analyse it in milliseconds.
    fn noisy(n: usize, sr: f32) -> Signal {
        let mut x = crate::residual::pseudo_noise(n);
        for (i, v) in x.iter_mut().enumerate() {
            *v += 0.5 * (i as f32 * 0.05).sin();
        }
        Signal::new(x, sr)
    }

    fn small_config() -> Config {
        let mut c = Config::default();
        c.dictionary.fof.alphas = vec![256.0];
        c.dictionary.fof.betas_ms = vec![1.0];
        c.dictionary.gaussian.sigmas_ms.clear();
        c.blocks.f_min = 200.0;
        c.blocks.f_max = 2000.0;
        c.pursuit.max_atoms = 40;
        c.pursuit.target_snr_db = 60.0;
        c.refine.enabled = false;
        c
    }

    /// A reporter that says "stop" from the outset.
    #[derive(Default)]
    struct StopAtOnce {
        events: usize,
    }

    impl Reporter for StopAtOnce {
        fn event(&mut self, _: Event<'_>) {
            self.events += 1;
        }
        fn cancelled(&self) -> bool {
            true
        }
    }

    /// The gate behind "closing a tab stops the work" and behind `rmpsynth`'s never seeing a
    /// half-finished book without knowing it: a cancelled run selects nothing and *says* it was
    /// cancelled, because the book alone cannot be told from a completed one.
    #[test]
    fn a_cancelled_run_selects_nothing_and_reports_that_it_was_cancelled() {
        let sig = noisy(8_000, 48_000.0);
        let cfg = small_config();
        let mut planner = Planner::new();
        let mut report = StopAtOnce::default();

        let got = analyse(
            AnalysisRequest { signal: &sig, offset: 0, config: &cfg, residual: None },
            &mut planner,
            &mut report,
        )
        .unwrap();

        assert!(got.cancelled, "the run was cancelled and must say so");
        assert!(got.book.is_empty(), "cancelled before the first atom, got {}", got.book.len());
        // The dictionary is built before the pursuit starts, so that event still fires: cancelling
        // stops the *search*, it does not abandon the request.
        assert!(report.events >= 1, "the dictionary event should still have been reported");
    }

    /// The same run, uninterrupted, to show the fixture is not simply barren — otherwise the
    /// assertion above would pass for the wrong reason.
    #[test]
    fn the_same_fixture_uncancelled_does_select_atoms() {
        let sig = noisy(8_000, 48_000.0);
        let cfg = small_config();
        let mut planner = Planner::new();

        let got = analyse(
            AnalysisRequest { signal: &sig, offset: 0, config: &cfg, residual: None },
            &mut planner,
            &mut (),
        )
        .unwrap();

        assert!(!got.cancelled);
        assert!(!got.book.is_empty(), "the fixture has nothing to decompose");
    }

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
