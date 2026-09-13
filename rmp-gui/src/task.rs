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
