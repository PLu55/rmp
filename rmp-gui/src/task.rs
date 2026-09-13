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
    // `[residual] enabled` in the settings document is the whole of the decision. There used to be
    // a GUI checkbox ANDed with it, which meant two controls for one thing and no way to tell from
    // the panel which of them was saying no.
    let residual_cfg = job.config.residual_config(sample_rate as f64).map_err(|e| e.to_string())?;
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

/// What a render should contain.
///
/// A decomposition has three separable parts and this is which of them to hear. They are *not*
/// analysis settings — nothing here changes a book — which is why they live in the window rather
/// than in the settings document.
///
/// The distinction that matters is between the two residuals. **Measured** is the pursuit's own
/// leftover buffer: literally what the atoms failed to explain, sample for sample. **Synthesised**
/// is the stochastic model of it rebuilt from the residual book's ERB band powers. They sound
/// different on purpose — CLAUDE.md records the model's reconstruction peaking ~10 dB below the
/// residue at matched rms, because the impulsive part of a residue is exactly what a noise model
/// does not carry — and hearing them side by side is how you judge whether that matters for a
/// given piece of material.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum RenderMode {
    /// The book's atoms alone. Always available once anything was selected.
    #[default]
    Atoms,
    /// The analysis residue itself, as `rmp -r` writes it.
    ResidualMeasured,
    /// The residue's stochastic reconstruction. Needs a residual book, so it needs
    /// `[residual] enabled` at analysis time.
    ResidualSynthesised,
    /// Atoms and the synthesised residual on one timeline — what `rmpsynth` produces by default.
    Mixed,
}

impl RenderMode {
    pub const ALL: [RenderMode; 4] = [
        RenderMode::Atoms,
        RenderMode::ResidualMeasured,
        RenderMode::ResidualSynthesised,
        RenderMode::Mixed,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RenderMode::Atoms => "Atoms",
            RenderMode::ResidualMeasured => "Residual (measured)",
            RenderMode::ResidualSynthesised => "Residual (synthesised)",
            RenderMode::Mixed => "Atoms + residual",
        }
    }

    /// The word that goes in the output file name, so the four do not overwrite each other.
    pub fn suffix(self) -> &'static str {
        match self {
            RenderMode::Atoms => "atoms",
            RenderMode::ResidualMeasured => "residual",
            RenderMode::ResidualSynthesised => "residual-synth",
            RenderMode::Mixed => "mixed",
        }
    }

    /// Whether this book can produce this. Checked rather than attempted, so an unavailable mode is
    /// a greyed-out entry that says why instead of a render that fails after the save dialog.
    pub fn available(self, book: &rmp_core::book::Book) -> bool {
        match self {
            RenderMode::Atoms => !book.is_empty(),
            RenderMode::ResidualMeasured => true,
            RenderMode::ResidualSynthesised => book.residual.is_some(),
            RenderMode::Mixed => !book.is_empty() && book.residual.is_some(),
        }
    }

    /// Why it is unavailable, for the tooltip.
    pub fn why_not(self) -> &'static str {
        match self {
            RenderMode::Atoms => "no atoms were selected",
            RenderMode::ResidualMeasured => "",
            RenderMode::ResidualSynthesised | RenderMode::Mixed => {
                "this book has no residual analysis — set [residual] enabled and analyse again"
            }
        }
    }
}

/// Rendering a finished book back to a soundfile.
///
/// A far smaller thing than [`spawn`]: synthesis has no stages worth reporting and no meaningful
/// cancellation — 8 s of audio renders in about 0.22 s — so there is one message and it is the
/// last. It still runs off the UI thread, because "usually fast" is not "always fast": a long book
/// is thousands of atoms and a second or two, and a window that freezes for a second reads as a
/// window that has crashed.
pub struct SynthJob {
    pub book: rmp_core::book::Book,
    /// The pursuit's own residue, for [`RenderMode::ResidualMeasured`]. Carried even when the mode
    /// does not need it, because a job is built once and the cost is one excerpt of f32.
    pub residual: Vec<f32>,
    pub sample_rate: f32,
    pub mode: RenderMode,
    pub output: PathBuf,
}

pub enum SynthUpdate {
    /// The path is carried back rather than remembered by the caller: a render that failed then
    /// cannot leave anything pointing at a file that was never written.
    Done { path: PathBuf, summary: String },
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
            let path = job.output.clone();
            let msg = match render(job) {
                Ok(summary) => SynthUpdate::Done { path, summary },
                Err(e) => SynthUpdate::Failed(e),
            };
            tx.send(msg).ok();
        })
        .expect("spawning the synthesis thread");

    Synthesising { updates, finished: false }
}

/// The render itself, and the one line it gets to say about what it did.
fn render(job: SynthJob) -> Result<String, String> {
    // The measured residue is not a synthesis at all: it is a buffer the analysis already has, and
    // writing it is what `rmp -r` does. Routing it through `render_to_file` would mean inventing a
    // book to carry samples that are not atoms and not a band-power model.
    if job.mode == RenderMode::ResidualMeasured {
        rmp_core::audio::write_samples(&job.output, &job.residual, job.sample_rate)
            .map_err(|e| e.to_string())?;
        let rms = rmp_core::signal::db_fs(rmp_core::signal::rms_of(&job.residual));
        let peak = rmp_core::signal::db_fs(rmp_core::signal::peak_of(&job.residual) as f64);
        return Ok(format!(
            "wrote the measured residual: {:.2} s, {rms:.1} dBFS rms, {peak:.1} dBFS peak",
            job.residual.len() as f32 / job.sample_rate,
        ));
    }

    let atoms = matches!(job.mode, RenderMode::Atoms | RenderMode::Mixed);
    let residual = matches!(job.mode, RenderMode::ResidualSynthesised | RenderMode::Mixed);
    let request = rmp_synthesis::RenderRequest {
        book: rmp_synthesis::BookInput::Full(job.book),
        residual_book: None,
        atoms,
        residual,
        output: job.output,
        config: rmp_synthesis::RenderConfig::default(),
    };
    let r = rmp_synthesis::render_to_file(&request).map_err(|e| e.to_string())?;
    Ok(describe_render(&r, job.mode))
}

/// What a finished render did. The peaks are the part worth reading: the residual's is where a
/// stochastic model is weakest, and `clipped` is the only thing here that is a fault.
fn describe_render(r: &rmp_synthesis::RenderReport, mode: RenderMode) -> String {
    let kinds: Vec<String> = r.atoms.iter().map(|(k, n)| format!("{n} {k}")).collect();
    let what = if kinds.is_empty() { mode.label().to_string() } else { kinds.join(", ") };
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
