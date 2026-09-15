//! Running an analysis without freezing the window.
//!
//! This is the piece the pipeline extraction was for, and the only part of the scaffold that is
//! finished rather than stubbed: a decomposition is seconds to minutes of work, so it cannot happen
//! on the UI thread, and everything the rest of the GUI will do hangs off being able to start one,
//! watch it, and stop it.
//!
//! The shape is the usual one — a worker thread, a channel back, an [`AtomicBool`] forward — with
//! two details that come from what [`rmp_core::pipeline`] actually promises:
//!
//! **[`Event`] borrows, so nothing can be forwarded as-is.** A `&Dictionary` cannot outlive the
//! call it arrived in. [`Progress`] is the owned form: what a front end needs to *display*, copied
//! out at the moment the event fires. Keeping it small is deliberate — this crosses a channel on
//! every window, and an accidental `Dictionary` clone is megabytes.
//!
//! **Cancellation is sticky and coarse.** The flag is polled once per selected atom, and one
//! iteration over a low-alpha dictionary is hundreds of milliseconds, so "stop" means "within about
//! an atom", not "now". The worker still returns its book: an interrupted run is a valid
//! decomposition of a prefix, and [`rmp_core::pipeline::Analysis::cancelled`] is how you know that
//! is what it is.

use rmp_core::audio;
use rmp_core::config::Config;
use rmp_core::fft::Planner;
use rmp_core::pipeline::{self, Analysis, AnalysisRequest, Event, Reporter};
use rmp_core::signal::Signal;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

/// What to analyse. The GUI's form, before any file has been touched.
#[derive(Clone, Debug)]
pub struct Job {
    pub input: PathBuf,
    pub config: Config,
    pub start: Option<f32>,
    pub duration: Option<f32>,
    /// Keep the pursuit's leftover buffer. It is produced either way — this decides whether to hold
    /// on to it, which is one excerpt of f32 per tab and the difference between being able to hear
    /// the measured residue afterwards and not.
    /// Where to write the book. Chosen before the run rather than after, as `rmp -b` is: a
    /// decomposition that took minutes and then had nowhere to go would be the worst outcome here.
    pub book_output: PathBuf,
    pub keep_residual: bool,
    /// Measure the residue into ERB band powers. Overrides `[residual] enabled` in the settings
    /// document the way the CLI's `--residual-analysis` flag does: defaults < document < panel.
    pub residual_analysis: bool,
}

/// An [`Event`], flattened to something that can be owned and sent.
#[derive(Clone, Debug)]
pub enum Progress {
    /// What was read, before any analysis: file duration, rate, channels, and whether a downmix
    /// happened.
    Input { whole_seconds: f32, sample_rate: f32, channels: usize, downmixed: bool },
    /// The excerpt actually being decomposed, in samples from the start of the file.
    Excerpt { offset: usize, len: usize },
    Dictionary { blocks: usize, kinds: Vec<(String, usize)>, unrefinable: usize, elapsed: Duration },
    /// Something the worker did that has no stage of its own — writing the book, or failing to.
    Note(String),
    Windows { count: usize, core_seconds: f32, guard_seconds: f32, over_budget: bool },
    Window { index: usize, of: usize, atoms: usize },
}

/// What the worker sends back. Exactly one [`Update::Done`] or [`Update::Failed`] ever arrives, and
/// it is the last message on the channel.
pub enum Update {
    Progress(Progress),
    Done(Box<Outcome>),
    Failed(String),
}

/// A finished run, with the excerpt kept alongside it.
///
/// The signal is worth holding: every level the GUI will want to show — the residual relative to
/// the input, a waveform under the atoms — is a comparison against it, and re-reading the file to
/// get it back would be both slow and a chance to read a *different* file.
pub struct Outcome {
    pub analysis: Analysis,
    pub signal: Signal,
    pub offset: usize,
}

/// A run in flight.
pub struct Running {
    updates: mpsc::Receiver<Update>,
    cancel: Arc<AtomicBool>,
    finished: bool,
}

impl Running {
    /// Ask the pursuit to stop. Idempotent, and takes effect within about one atom.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn cancel_requested(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// True once a `Done` or `Failed` has been drained.
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Everything the worker has said since the last call. Never blocks, so it is safe to call once
    /// per frame; the caller is expected to repaint while a run is live.
    pub fn drain(&mut self) -> Vec<Update> {
        let mut out = Vec::new();
        while let Ok(u) = self.updates.try_recv() {
            if matches!(u, Update::Done(_) | Update::Failed(_)) {
                self.finished = true;
            }
            out.push(u);
        }
        out
    }
}

/// Start `job` on a worker thread.
///
/// The thread is detached: dropping [`Running`] abandons it rather than joining, which would block
/// the UI for as long as the pursuit takes. Dropping also *cancels* — see the `Drop` impl.
pub fn spawn(job: Job) -> Running {
    let (tx, updates) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();

    std::thread::Builder::new()
        .name("rmp-analysis".into())
        .spawn(move || {
            let tx2 = tx.clone();
            match work(job, &flag, &tx) {
                Ok(outcome) => tx2.send(Update::Done(Box::new(outcome))).ok(),
                Err(e) => tx2.send(Update::Failed(e)).ok(),
            };
        })
        .expect("spawning the analysis thread");

    Running { updates, cancel, finished: false }
}

/// Dropping a run cancels it.
///
/// Closing a tab drops its `Running`, and without this the worker would keep a core busy to the end
/// of a decomposition nobody is going to look at. A closed channel is *not* enough on its own: the
/// worker only discovers that on its next send, and a single-window run sends nothing between
/// starting and finishing.
impl Drop for Running {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// The worker body: the same sequence `rmp`'s `analyse` performs, reported to a channel instead of
/// to stderr.
fn work(job: Job, cancel: &AtomicBool, tx: &mpsc::Sender<Update>) -> Result<Outcome, String> {
    job.config.validate()?;

    let read = audio::read(&job.input).map_err(|e| e.to_string())?;
    let sample_rate = read.signal.sample_rate;
    let whole_len = read.signal.len();
    let downmixed = read.was_downmixed();
    let channels = read.channels;
    let (offset, signal) = pipeline::excerpt(read.signal, job.start, job.duration)?;

    let _ = tx.send(Update::Progress(Progress::Input {
        whole_seconds: whole_len as f32 / sample_rate,
        sample_rate,
        channels,
        downmixed,
    }));
    let _ = tx.send(Update::Progress(Progress::Excerpt { offset, len: signal.len() }));

    if signal.energy() <= 0.0 {
        return Err("the selected excerpt is silent".into());
    }

    // Resolved here rather than inside the pursuit so that an ERB range that does not fit under
    // this file's Nyquist fails now, not after a seven-second analysis.
    let mut residual_cfg =
        job.config.residual_config(sample_rate as f64).map_err(|e| e.to_string())?;
    // The panel decides, overriding the document — the same precedence the CLI's
    // `--residual-analysis` has over `[residual] enabled`, and for the same reason: the control you
    // just touched should be the one that wins.
    residual_cfg.enabled = job.residual_analysis;
    let want_residual = residual_cfg.enabled;

    let mut reporter =
        Forward { tx: tx.clone(), cancel, mp_cfg: job.config.mp_config(), sr: sample_rate };
    let mut planner = Planner::new();
    let analysis = pipeline::analyse(
        AnalysisRequest {
            signal: &signal,
            offset,
            config: &job.config,
            residual: want_residual.then_some(&residual_cfg),
        },
        &mut planner,
        &mut reporter,
    )?;

    let mut analysis = analysis;

    // Embedded rather than left beside the book, which is what `rmp` does when `--residual-book`
    // names no separate file. Two things follow: the file written carries everything the run
    // produced, and the book held in memory has the same shape as one read back from disk, so
    // nothing downstream has to ask which of the two places the residual is in.
    analysis.book.residual = analysis.residual_book.take();

    // Written here, on the worker, and reported rather than returned: a write that fails must not
    // throw away a decomposition that has already been paid for. The line says so and the results
    // stay.
    let note = match rmp_core::book::write(&job.book_output, &analysis.book) {
        Ok(()) => format!("wrote {}", job.book_output.display()),
        Err(e) => format!("could not write the book: {e}"),
    };
    let _ = tx.send(Update::Progress(Progress::Note(note)));

    if !job.keep_residual {
        // Freed rather than never made: the pursuit's residue *is* its working buffer, so there is
        // nothing to skip computing — only something to stop holding.
        analysis.residual = Vec::new();
    }
    Ok(Outcome { analysis, signal, offset })
}

/// Turns borrowed [`Event`]s into owned [`Progress`] on the channel.
struct Forward<'a> {
    tx: mpsc::Sender<Update>,
    cancel: &'a AtomicBool,
    mp_cfg: rmp_core::mp::MpConfig,
    /// The window plan is in samples; seconds are what a reader wants.
    sr: f32,
}

impl Reporter for Forward<'_> {
    fn event(&mut self, e: Event<'_>) {
        let p = match e {
            Event::Dictionary { dict, elapsed } => Progress::Dictionary {
                blocks: dict.blocks.len(),
                kinds: rmp_core::atom::AtomKind::ALL
                    .iter()
                    .map(|&k| {
                        (k.to_string(), dict.blocks.iter().filter(|b| b.env.kind() == k).count())
                    })
                    .filter(|&(_, n)| n > 0)
                    .collect(),
                unrefinable: pipeline::unrefinable_blocks(dict, &self.mp_cfg).len(),
                elapsed,
            },
            Event::Windows { plan } => Progress::Windows {
                count: plan.count,
                core_seconds: plan.core_len as f32 / self.sr,
                guard_seconds: plan.guard_len as f32 / self.sr,
                over_budget: plan.over_budget,
            },
            Event::Window { index, of, atoms } => Progress::Window { index, of, atoms },
        };
        // A closed channel means the GUI dropped this run. Nothing to do about it here; the
        // cancel flag is the orderly way out, and the send failing is the disorderly one.
        let _ = self.tx.send(Update::Progress(p));
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// What a render should contain: the two halves of a book, independently.
///
/// The same two `rmpsynth` exposes, and they mean the same thing — atoms through rfofs, and the
/// stochastic reconstruction of the residue from its ERB band powers. "Residual" here is always
/// the *synthesised* one; the measured residue is a buffer the analysis already holds and is a
/// thing to listen to rather than to synthesise (see [`crate::playback`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RenderParts {
    pub atoms: bool,
    pub residual: bool,
}

impl Default for RenderParts {
    /// Atoms alone. `[residual] enabled` is off by default, so a residual is the exception.
    fn default() -> Self {
        Self { atoms: true, residual: false }
    }
}

impl RenderParts {
    /// The word that goes in the output file name, so renders of different parts do not overwrite
    /// each other.
    pub fn suffix(self) -> &'static str {
        match (self.atoms, self.residual) {
            (true, true) => "mixed",
            (true, false) => "atoms",
            (false, true) => "residual-synth",
            (false, false) => "empty",
        }
    }

    /// Whether the finished run can produce this. Checked rather than attempted, so an impossible
    /// request is a disabled button that says why instead of a render that fails after the save
    /// dialog.
    ///
    /// Judged against [`crate::playback::Available`] — what the run *made* — and not against the
    /// `Book` alone. `pipeline::analyse` returns the residual book beside the atom book rather than
    /// inside it, so a book straight from a run has `Book::residual` empty however the analysis was
    /// configured; asking it directly refused every residual render, the mixed one included.
    pub fn available(self, av: crate::playback::Available) -> Result<(), &'static str> {
        if !self.atoms && !self.residual {
            return Err("nothing selected to render");
        }
        if self.atoms && !av.atoms {
            return Err("no atoms were selected by the analysis");
        }
        if self.residual && !av.residual_synthesised {
            return Err("this run measured no residual — tick residual analysis and analyse again");
        }
        Ok(())
    }
}

/// Rendering a finished book to a soundfile.
///
/// A far smaller thing than [`spawn`]: synthesis has no stages worth reporting and no meaningful
/// cancellation — 8 s of audio renders in about 0.22 s — so there is one message and it is the
/// last. It still runs off the UI thread, because "usually fast" is not "always fast": a long book
/// is thousands of atoms and a second or two, and a window that freezes for a second reads as a
/// window that has crashed.
pub struct SynthJob {
    pub book: rmp_core::book::Book,
    /// The stochastic model, when the run produced one. Passed separately because that is where
    /// `pipeline::analyse` leaves it, and `RenderRequest::residual_book` exists for exactly this —
    /// the standalone residual book `rmp --residual-book` writes beside an atom book.
    pub residual_book: Option<rmp_core::residual::ResidualBook>,
    pub parts: RenderParts,
    pub output: PathBuf,
}

pub enum SynthUpdate {
    /// What was written, already said. Nothing holds on to the path: Play sounds the analysis
    /// itself rather than replaying a file, so a render is an output and not a step towards one.
    Done(String),
    Failed(String),
}

/// A render in flight.
pub struct Synthesising {
    updates: mpsc::Receiver<SynthUpdate>,
    finished: bool,
}

impl Synthesising {
    pub fn finished(&self) -> bool {
        self.finished
    }

    pub fn drain(&mut self) -> Vec<SynthUpdate> {
        let mut out = Vec::new();
        while let Ok(u) = self.updates.try_recv() {
            self.finished = true;
            out.push(u);
        }
        out
    }
}

pub fn spawn_synthesis(job: SynthJob) -> Synthesising {
    let (tx, updates) = mpsc::channel();
    std::thread::Builder::new()
        .name("rmp-synthesis".into())
        .spawn(move || {
            let where_to = job.output.display().to_string();
            let request = rmp_synthesis::RenderRequest {
                book: rmp_synthesis::BookInput::Full(job.book),
                residual_book: job.residual_book,
                atoms: job.parts.atoms,
                residual: job.parts.residual,
                output: job.output,
                config: rmp_synthesis::RenderConfig::default(),
            };
            let msg = match rmp_synthesis::render_to_file(&request) {
                Ok(r) => SynthUpdate::Done(format!("{} → {where_to}", describe_render(&r))),
                Err(e) => SynthUpdate::Failed(e.to_string()),
            };
            tx.send(msg).ok();
        })
        .expect("spawning the synthesis thread");

    Synthesising { updates, finished: false }
}

/// What a finished render did. The peaks are the part worth reading: the residual's is where a
/// stochastic model is weakest, and `clipped` is the only thing here that is a fault.
fn describe_render(r: &rmp_synthesis::RenderReport) -> String {
    let kinds: Vec<String> = r.atoms.iter().map(|(k, n)| format!("{n} {k}")).collect();
    let what = if kinds.is_empty() { "residual".to_string() } else { kinds.join(", ") };
    let mut line = format!(
        "wrote {} — {:.2} s, peak {:.1} dBFS",
        what,
        r.samples_written as f64 / r.sample_rate,
        rmp_core::signal::db_fs(r.mixed_peak as f64),
    );
    if r.residual_samples.is_some() {
        line.push_str(&format!(
            "; residual peak {:.1} dBFS",
            rmp_core::signal::db_fs(r.residual_peak as f64)
        ));
    }
    if r.clipped_samples > 0 {
        line.push_str(&format!(" — {} samples clipped", r.clipped_samples));
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of asking for a file up front: a run must actually leave one behind, and it
    /// must be a book that reads back.
    ///
    /// Against a real soundfile, because the thing under test is the chain from audio through the
    /// pursuit to `book::write` — a synthetic buffer would exercise the same code but would not
    /// catch a file the analysis could not open. Skipped where `data/` is not checked out, since it
    /// is gitignored.
    #[test]
    fn a_run_writes_a_book_that_reads_back_with_its_residual_inside() {
        let input = std::path::Path::new("../data/audio/chopin-nocturne-2.wav");
        if !input.is_file() {
            eprintln!("no {} here; skipping", input.display());
            return;
        }
        let out = std::env::temp_dir()
            .join(format!("rmp-gui-book-test-{}.json.gz", std::process::id()));

        let mut cfg = rmp_core::config::Config::default();
        cfg.dictionary.fof.alphas = vec![256.0];
        cfg.dictionary.fof.betas_ms = vec![1.0];
        cfg.blocks.f_min = 200.0;
        cfg.blocks.f_max = 2000.0;
        cfg.pursuit.max_atoms = 20;
        cfg.refine.enabled = false;

        let mut run = spawn(Job {
            input: input.to_path_buf(),
            config: cfg,
            // Past the file's silent lead-in, and short.
            start: Some(2.0),
            duration: Some(0.3),
            book_output: out.clone(),
            keep_residual: true,
            residual_analysis: true,
        });

        let mut done = None;
        for _ in 0..600 {
            for u in run.drain() {
                if let Update::Done(o) = u {
                    done = Some(o);
                }
            }
            if run.finished() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let outcome = done.expect("the run produced no result");

        // In memory: the residual ends up *inside* the book, so nothing downstream has to ask
        // which of the two places it is in.
        assert!(outcome.analysis.residual_book.is_none(), "it was moved, not copied");
        assert!(outcome.analysis.book.residual.is_some(), "and moved into the book");

        // On disk: the same, and readable by the same reader `rmpstat` and `rmpsynth` use.
        assert!(out.is_file(), "no book was written to {}", out.display());
        let back = rmp_core::book::read(&out).expect("the book must read back");
        assert_eq!(back.len(), outcome.analysis.book.len(), "a different book came back");
        assert!(back.residual.is_some(), "the residual did not survive the write");

        std::fs::remove_file(&out).ok();
    }

    /// A smoke test over the whole worker plumbing: spawn, run, report, terminate.
    ///
    /// A path that does not exist is the cheapest job that still goes all the way through
    /// `spawn` -> thread -> `work` -> channel, and it needs no audio fixture, which matters because
    /// `data/` is not checked in. What it pins is that exactly one terminal message arrives and
    /// that `finished` latches on it — the two things the UI's `pump` relies on to stop polling.
    #[test]
    fn a_job_that_cannot_start_reports_one_failure_and_finishes() {
        let mut run = spawn(Job {
            input: PathBuf::from("/nonexistent/definitely-not-here.wav"),
            config: Default::default(),
            start: None,
            duration: None,
            book_output: std::env::temp_dir().join("rmp-gui-never-written.json"),
            keep_residual: true,
            residual_analysis: false,
        });

        // The worker is a real thread; give it a moment rather than spinning forever.
        let mut updates = Vec::new();
        for _ in 0..200 {
            updates.extend(run.drain());
            if run.finished() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert!(run.finished(), "the worker never reported a terminal message");
        let terminal: Vec<_> = updates
            .iter()
            .filter(|u| matches!(u, Update::Done(_) | Update::Failed(_)))
            .collect();
        assert_eq!(terminal.len(), 1, "exactly one terminal message");
        assert!(matches!(terminal[0], Update::Failed(_)), "a missing file must fail, not succeed");
    }
}
