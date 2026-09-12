//! The window: a scaffold, not an implementation.
//!
//! Every panel below is a placeholder naming the `rmp-core` or `rmp-synthesis` call that will fill
//! it. What is real here is the shape — a settings side panel, a log that a live run writes into,
//! and result tabs that only exist once there is a book — and the wiring to [`crate::task`], which
//! is what makes a long analysis survivable in a UI.
//!
//! The reason to keep it this thin is that none of the interesting decisions are the GUI's. The
//! atom kinds, the settings and their validation, the statistics and the time-frequency map all
//! already exist and are already tested; a panel that recomputed any of them would be a second
//! definition. So each stub says which existing function it is a view of.

use crate::task::{self, Outcome, Progress, Running, Update};
use rmp_core::config::Config;
use rmp_core::signal::{db_fs, rms_of};
use std::path::PathBuf;

/// What the results area is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    /// `rmp_core::stats::BookSummary` — the same figures `rmpstat summary` prints.
    Summary,
    /// `rmp_core::stats::Histogram` over a `Quantity`, as `rmpstat hist`.
    Distribution,
    /// `rmp_core::tfmap::TfMap` — the atom-based pseudo-Wigner map, as `rmpstat wv`.
    TimeFrequency,
    /// `rmp_synthesis::render` — resynthesis, and writing it out.
    Synthesis,
}

impl Tab {
    const ALL: [Tab; 4] = [Tab::Summary, Tab::Distribution, Tab::TimeFrequency, Tab::Synthesis];

    fn label(self) -> &'static str {
        match self {
            Tab::Summary => "Summary",
            Tab::Distribution => "Distributions",
            Tab::TimeFrequency => "Time-frequency",
            Tab::Synthesis => "Synthesis",
        }
    }
}

pub struct RmpApp {
    input: Option<PathBuf>,
    start: String,
    duration: String,
    residual_analysis: bool,
    /// The settings document itself, edited in place. Held as the real `Config` rather than as
    /// widget state so that `Config::validate` is the only definition of what is legal.
    config: Config,
    running: Option<Running>,
    /// The last finished run, if any.
    outcome: Option<Box<Outcome>>,
    log: Vec<String>,
    tab: Tab,
}

impl Default for RmpApp {
    fn default() -> Self {
        Self {
            input: None,
            start: String::new(),
            duration: String::new(),
            residual_analysis: false,
            config: Config::default(),
            running: None,
            outcome: None,
            log: Vec::new(),
            tab: Tab::Summary,
        }
    }
}

impl eframe::App for RmpApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.pump();
        // A run reports only once per window, which can be minutes apart, so the repaint has to be
        // asked for rather than waited on: without this the window sleeps and the log arrives late.
        if self.running.is_some() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::Panel::top("input").show(ui, |ui| self.input_bar(ui));
        egui::Panel::left("settings").default_size(320.0).show(ui, |ui| self.settings(ui));
        egui::Panel::bottom("log")
            .resizable(true)
            .default_size(140.0)
            .show(ui, |ui| self.log_panel(ui));
        egui::CentralPanel::default().show(ui, |ui| self.results(ui));
    }
}

impl RmpApp {
    /// Drain the worker and fold its messages into the window's state.
    fn pump(&mut self) {
        let Some(run) = &mut self.running else { return };
        for update in run.drain() {
            match update {
                Update::Progress(p) => self.log.push(describe(&p)),
                Update::Done(outcome) => {
                    let a = &outcome.analysis;
                    self.log.push(format!(
                        "{} atoms, {:.1} dB{}",
                        a.book.len(),
                        a.book.snr_db(),
                        if a.cancelled { " (interrupted)" } else { "" }
                    ));
                    self.outcome = Some(outcome);
                }
                Update::Failed(e) => self.log.push(format!("failed: {e}")),
            }
        }
        if run.finished() {
            self.running = None;
        }
    }

    fn input_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Open…").clicked()
                && let Some(p) = rfd::FileDialog::new()
                    .add_filter("audio", &["wav", "aiff", "aif", "flac"])
                    .pick_file()
            {
                self.input = Some(p);
            }
            ui.label(
                self.input
                    .as_deref()
                    .map_or_else(|| "no input".to_string(), |p| p.display().to_string()),
            );

            ui.separator();
            ui.label("start");
            ui.add(egui::TextEdit::singleline(&mut self.start).desired_width(56.0));
            ui.label("duration");
            ui.add(egui::TextEdit::singleline(&mut self.duration).desired_width(56.0));
            ui.label("s");

            ui.separator();
            match &self.running {
                Some(run) => {
                    let stopping = run.cancel_requested();
                    if ui
                        .add_enabled(!stopping, egui::Button::new("Stop"))
                        .on_hover_text("takes effect within about one atom")
                        .clicked()
                    {
                        run.cancel();
                    }
                    ui.spinner();
                    ui.label(if stopping { "stopping…" } else { "analysing…" });
                }
                None => {
                    if ui.add_enabled(self.input.is_some(), egui::Button::new("Analyse")).clicked() {
                        self.start_run();
                    }
                }
            }
        });
    }

    fn start_run(&mut self) {
        let Some(input) = self.input.clone() else { return };
        self.log.clear();
        self.outcome = None;
        self.running = Some(task::spawn(task::Job {
            input,
            config: self.config.clone(),
            start: self.start.trim().parse().ok(),
            duration: self.duration.trim().parse().ok(),
            residual_analysis: self.residual_analysis,
        }));
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        ui.checkbox(&mut self.residual_analysis, "Analyse the residue into ERB band power");
        ui.separator();
        // TODO: edit `self.config` in place, section by section, and show what
        // `Config::validate` says. `Config::to_toml` and `Config::from_toml` are the
        // import/export pair, and `data/config/*.toml` are the worked examples.
        ui.label("Dictionary, envelope, blocks, pursuit, refine, HRMP and residual sections go here,");
        ui.label("editing rmp_core::Config directly so validation has one definition.");
    }

    fn log_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Progress");
        egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
            for line in &self.log {
                ui.monospace(line);
            }
        });
    }

    fn results(&mut self, ui: &mut egui::Ui) {
        let Some(outcome) = &self.outcome else {
            ui.centered_and_justified(|ui| ui.label("Open a soundfile and analyse it."));
            return;
        };

        ui.horizontal(|ui| {
            for t in Tab::ALL {
                ui.selectable_value(&mut self.tab, t, t.label());
            }
        });
        ui.separator();

        let book = &outcome.analysis.book;
        match self.tab {
            // TODO: rmp_core::stats::BookSummary::of(book) — the figures `rmpstat summary` prints.
            // The three lines below are the shape of it, and the reason `Outcome` keeps the
            // excerpt: the residual is only meaningful against the input it came from.
            Tab::Summary => {
                ui.monospace(format!("{} atoms, {:.1} dB", book.len(), book.snr_db()));
                ui.monospace(format!(
                    "excerpt: {} samples from {}",
                    outcome.signal.len(),
                    outcome.offset
                ));
                ui.monospace(format!(
                    "residual: {:+.1} dB rms relative to input",
                    db_fs(rms_of(&outcome.analysis.residual)) - db_fs(outcome.signal.rms())
                ));
                ui.label("rmp_core::stats::BookSummary goes here.");
            }
            // TODO: rmp_core::stats::Histogram over a chosen Quantity and Weight, drawn with
            // egui_plot rather than through plotters — the charts are rmpstat's, the data is not.
            Tab::Distribution => {
                ui.label("rmp_core::stats::Histogram over alpha, bandwidth, f, sigma … goes here.");
            }
            // TODO: rmp_core::tfmap::TfMap::build, blitted as one egui::ColorImage. Never as
            // per-cell rectangles: a useful grid is ~10^6 cells.
            Tab::TimeFrequency => {
                ui.label("rmp_core::tfmap::TfMap goes here.");
            }
            // TODO: rmp_synthesis::render_full_book / render_to_file, on a worker thread of its
            // own — a long book takes seconds to render.
            Tab::Synthesis => {
                ui.label("rmp_synthesis::render goes here: atoms, residual, or both.");
            }
        }
    }
}

/// One line of log for a progress message.
fn describe(p: &Progress) -> String {
    match p {
        Progress::Input { whole_seconds, sample_rate, channels, downmixed } => format!(
            "{whole_seconds:.2} s, {} Hz, {channels} channel(s){}",
            *sample_rate as u32,
            if *downmixed { ", downmixed to mono" } else { "" }
        ),
        Progress::Excerpt { offset, len } => {
            format!("analysing {len} samples from {offset}")
        }
        Progress::Dictionary { blocks, kinds, unrefinable, elapsed } => {
            let k: Vec<_> = kinds.iter().map(|(k, n)| format!("{n} {k}")).collect();
            let note = if *unrefinable > 0 {
                format!(", {unrefinable} too long to refine")
            } else {
                String::new()
            };
            format!("dictionary: {blocks} blocks ({}){note} in {elapsed:.2?}", k.join(", "))
        }
        Progress::Windows { count, core_seconds, guard_seconds, over_budget } => format!(
            "windows: {count} x {core_seconds:.2} s core + {guard_seconds:.2} s guard{}",
            if *over_budget { " (raised to four guards)" } else { "" }
        ),
        Progress::Window { index, of, atoms } => {
            format!("  window {index}/{of}: {atoms} atoms so far")
        }
    }
}
