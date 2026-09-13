//! The window: a tab strip, and one analysis per tab.
//!
//! Each tab is a [`Session`] — an input file, a settings document, a run, its log and its results —
//! and switching tabs switches all of them together. That is the shape this tool wants: the tuning
//! recorded in `CLAUDE.md` is all *comparison between runs*, and it was done by writing books to
//! disk and reading them back for want of somewhere to put two at once.
//!
//! Inside a tab the panels are still a scaffold. Each names the `rmp-core` or `rmp-synthesis` call
//! that will fill it, because none of the interesting decisions are the GUI's: the atom kinds, the
//! settings and their validation, the statistics and the time-frequency map all exist and are
//! tested, and a panel that recomputed any of them would be a second definition.

use crate::task::{self, Outcome, Progress, Running, Update};
use rmp_core::config::Config;
use rmp_core::signal::{db_fs, rms_of};
use std::path::PathBuf;

/// What a tab's results area is showing.
///
/// Not called `Tab`: a tab is a document now, and two things by that name in one file is how the
/// results strip and the document strip get confused for each other.
#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    /// `rmp_core::stats::BookSummary` — the same figures `rmpstat summary` prints.
    Summary,
    /// `rmp_core::stats::Histogram` over a `Quantity`, as `rmpstat hist`.
    Distribution,
    /// `rmp_core::tfmap::TfMap` — the atom-based pseudo-Wigner map, as `rmpstat wv`.
    TimeFrequency,
    /// `rmp_synthesis::render` — resynthesis, and writing it out.
    Synthesis,
}

impl View {
    const ALL: [View; 4] = [View::Summary, View::Distribution, View::TimeFrequency, View::Synthesis];

    fn label(self) -> &'static str {
        match self {
            View::Summary => "Summary",
            View::Distribution => "Distributions",
            View::TimeFrequency => "Time-frequency",
            View::Synthesis => "Synthesis",
        }
    }
}

/// One tab: an independent analysis, from the file it reads to the results it keeps.
struct Session {
    /// Unique among the open tabs, and the first half of the title. Not an index — closing a tab
    /// does not renumber the others.
    number: u32,
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
    view: View,
}

impl Session {
    fn new(number: u32) -> Self {
        Self {
            number,
            input: None,
            start: String::new(),
            duration: String::new(),
            residual_analysis: false,
            config: Config::default(),
            running: None,
            outcome: None,
            log: Vec::new(),
            view: View::Summary,
        }
    }

    /// The same inputs and settings under a new number, with none of the results.
    ///
    /// The A/B case: one file, one setting changed. Copying the log or the outcome would attach a
    /// book to settings that did not produce it, which is the one thing a comparison view must not
    /// do.
    fn duplicate(&self, number: u32) -> Self {
        Self {
            number,
            input: self.input.clone(),
            start: self.start.clone(),
            duration: self.duration.clone(),
            residual_analysis: self.residual_analysis,
            config: self.config.clone(),
            running: None,
            outcome: None,
            log: Vec::new(),
            view: self.view,
        }
    }

    /// `NN filename.wav`, derived every frame rather than cached, so choosing a file in a tab that had
    /// none renames it at once and keeps its number.
    fn title(&self) -> String {
        let name = self
            .input
            .as_ref()
            .and_then(|p| p.file_name())
            .map_or_else(|| "(no input)".to_string(), |n| n.to_string_lossy().into_owned());
        format!("{:02} {name}", self.number)
    }

    /// Drain the worker and fold its messages into this tab's state.
    ///
    /// Called for every session each frame, not only the visible one: a tab analysing in the
    /// background still has to drain its channel, or its log and its result appear all at once at
    /// the moment you switch to it and the tab looks frozen until then.
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
        // `auto_shrink` off in both directions, or the scroll area collapses to its content and
        // takes the panel down with it — an empty log would leave a panel one heading tall, which
        // is not what `default_size` asked for.
        egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(
            ui,
            |ui| {
                for line in &self.log {
                    ui.monospace(line);
                }
            },
        );
    }

    fn results(&mut self, ui: &mut egui::Ui) {
        let Some(outcome) = &self.outcome else {
            ui.centered_and_justified(|ui| ui.label("Open a soundfile and analyse it."));
            return;
        };

        ui.horizontal(|ui| {
            for v in View::ALL {
                ui.selectable_value(&mut self.view, v, v.label());
            }
        });
        ui.separator();

        let book = &outcome.analysis.book;
        match self.view {
            // TODO: rmp_core::stats::BookSummary::of(book) — the figures `rmpstat summary` prints.
            // The three lines below are the shape of it, and the reason `Outcome` keeps the
            // excerpt: the residual is only meaningful against the input it came from.
            View::Summary => {
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
            View::Distribution => {
                ui.label("rmp_core::stats::Histogram over alpha, bandwidth, f, sigma … goes here.");
            }
            // TODO: rmp_core::tfmap::TfMap::build, blitted as one egui::ColorImage. Never as
            // per-cell rectangles: a useful grid is ~10^6 cells.
            View::TimeFrequency => {
                ui.label("rmp_core::tfmap::TfMap goes here.");
            }
            // TODO: rmp_synthesis::render_full_book / render_to_file, on a worker thread of its
            // own — a long book takes seconds to render.
            View::Synthesis => {
                ui.label("rmp_synthesis::render goes here: atoms, residual, or both.");
            }
        }
    }
}

/// What a click on the tab strip asked for, applied after the strip has been drawn.
///
/// The strip borrows `sessions` to draw itself, so it cannot add to or remove from that same list
/// while iterating. One deferred action per frame is enough: nothing here can be clicked twice.
enum Action {
    Select(usize),
    Close(usize),
    New,
    Duplicate,
}

pub struct RmpApp {
    sessions: Vec<Session>,
    active: usize,
}

impl Default for RmpApp {
    fn default() -> Self {
        Self { sessions: vec![Session::new(1)], active: 0 }
    }
}

impl eframe::App for RmpApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Every tab, not just the visible one — see `Session::pump`.
        for s in &mut self.sessions {
            s.pump();
        }
        // A run reports only once per window, which can be minutes apart, so the repaint has to be
        // asked for rather than waited on: without this the window sleeps and the log arrives late.
        // Asked for if *any* tab is running, since a hidden tab still drives its strip spinner.
        if self.sessions.iter().any(|s| s.running.is_some()) {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::Panel::top("tabs").show(ui, |ui| self.tab_bar(ui));

        // Panel ids are shared across tabs on purpose: a panel's width and height are window
        // chrome, and they should not jump when you switch. Widget state inside them is not — see
        // the `push_id` below.
        let salt = self.sessions[self.active].number;
        let session = &mut self.sessions[self.active];
        egui::Panel::top("input").show(ui, |ui| {
            ui.push_id(salt, |ui| session.input_bar(ui));
        });
        egui::Panel::left("settings").default_size(320.0).show(ui, |ui| {
            ui.push_id(salt, |ui| session.settings(ui));
        });
        egui::Panel::bottom("log").resizable(true).default_size(140.0).show(ui, |ui| {
            ui.push_id(salt, |ui| session.log_panel(ui));
        });
        egui::CentralPanel::default().show(ui, |ui| {
            ui.push_id(salt, |ui| session.results(ui));
        });
    }
}

impl RmpApp {
    fn tab_bar(&mut self, ui: &mut egui::Ui) {
        let mut action = None;
        egui::ScrollArea::horizontal().auto_shrink([false, true]).show(ui, |ui| {
            ui.horizontal(|ui| {
                for (i, s) in self.sessions.iter().enumerate() {
                    // Grouped so the label, the spinner and the close button read as one tab
                    // rather than as a row of loose widgets.
                    ui.group(|ui| {
                        if ui.selectable_label(i == self.active, s.title()).clicked() {
                            action = Some(Action::Select(i));
                        }
                        // A background run is visible without switching to it.
                        if s.running.is_some() {
                            ui.spinner();
                        }
                        if ui
                            .small_button("×")
                            .on_hover_text("close this tab, stopping any analysis in it")
                            .clicked()
                        {
                            action = Some(Action::Close(i));
                        }
                    });
                }

                ui.separator();
                if ui.button("+").on_hover_text("a new tab, at the default settings").clicked() {
                    action = Some(Action::New);
                }
                if ui
                    .button("Duplicate")
                    .on_hover_text("this tab's file and settings, without its results")
                    .clicked()
                {
                    action = Some(Action::Duplicate);
                }
            });
        });

        match action {
            Some(Action::Select(i)) => self.active = i,
            Some(Action::Close(i)) => self.close(i),
            Some(Action::New) => self.push(Session::new(self.free_number())),
            Some(Action::Duplicate) => {
                self.push(self.sessions[self.active].duplicate(self.free_number()))
            }
            None => {}
        }
    }

    /// The lowest number no open tab is using.
    ///
    /// Reuse is what keeps `{:02}` honest: a monotonic counter would print three digits after the
    /// hundredth tab and there would be nothing sensible for the format to do about it.
    fn free_number(&self) -> u32 {
        (1..).find(|n| self.sessions.iter().all(|s| s.number != *n)).expect("u32 exhausted")
    }

    fn push(&mut self, s: Session) {
        self.sessions.push(s);
        self.active = self.sessions.len() - 1;
    }

    /// Close a tab, cancelling whatever it was running.
    ///
    /// The cancel is `Running`'s `Drop`, not anything written here, so it cannot be forgotten at
    /// some future call site. Closing the last tab leaves a fresh one rather than an empty window:
    /// an empty state would exist for this one case alone.
    ///
    /// **Clamping `active` is not enough on its own.** Removing a tab *below* the active one shifts
    /// the rest down, so the same index now names a different tab — you close tab 1 of four with
    /// tab 2 selected and find yourself looking at what was tab 3. The index has to follow the
    /// shift first, and only then be clamped for the case where the active tab was itself the last.
    fn close(&mut self, i: usize) {
        self.sessions.remove(i);
        if self.sessions.is_empty() {
            self.sessions.push(Session::new(1));
            self.active = 0;
            return;
        }
        if i < self.active {
            self.active -= 1;
        }
        self.active = self.active.min(self.sessions.len() - 1);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn with(paths: &[Option<&str>]) -> RmpApp {
        let mut app = RmpApp { sessions: Vec::new(), active: 0 };
        for p in paths {
            let n = app.free_number();
            let mut s = Session::new(n);
            s.input = p.map(PathBuf::from);
            app.sessions.push(s);
        }
        app
    }

    #[test]
    fn a_title_is_a_two_digit_number_then_the_file_name() {
        let app = with(&[Some("/a/b/chopin-nocturne-2.wav"), None]);
        assert_eq!(app.sessions[0].title(), "01 chopin-nocturne-2.wav");
        assert_eq!(app.sessions[1].title(), "02 (no input)");
    }

    /// Choosing a file renames the tab but must not renumber it — the number is the tab's identity
    /// for as long as it is open.
    #[test]
    fn picking_a_file_renames_a_tab_without_renumbering_it() {
        let mut app = with(&[None, None]);
        assert_eq!(app.sessions[1].title(), "02 (no input)");
        app.sessions[1].input = Some(PathBuf::from("zyklus.wav"));
        assert_eq!(app.sessions[1].title(), "02 zyklus.wav");
    }

    /// Reuse is what keeps the number two digits: closing the middle tab frees 02, and the next
    /// new tab takes it rather than counting on to 04.
    #[test]
    fn a_closed_number_is_reused_by_the_next_tab() {
        let mut app = with(&[Some("a.wav"), Some("b.wav"), Some("c.wav")]);
        assert_eq!(app.free_number(), 4);
        app.close(1);
        assert_eq!(app.free_number(), 2);
        app.push(Session::new(app.free_number()));
        let mut got: Vec<u32> = app.sessions.iter().map(|s| s.number).collect();
        got.sort_unstable();
        assert_eq!(got, [1, 2, 3]);
    }

    #[test]
    fn numbers_stay_unique_across_a_run_of_opens_and_closes() {
        let mut app = with(&[None]);
        for _ in 0..20 {
            app.push(Session::new(app.free_number()));
            app.push(Session::new(app.free_number()));
            app.close(0);
            let mut ns: Vec<u32> = app.sessions.iter().map(|s| s.number).collect();
            let before = ns.len();
            ns.sort_unstable();
            ns.dedup();
            assert_eq!(ns.len(), before, "two tabs share a number");
        }
    }

    /// The tab you were looking at is the tab you are still looking at.
    ///
    /// Asserted by *number* rather than by index, because the index is exactly what the bug moves:
    /// a clamp alone keeps `active` in bounds while pointing it at a different tab, and an
    /// index-based assertion agrees with the bug.
    #[test]
    fn closing_another_tab_leaves_you_on_the_same_one() {
        let mut app = with(&[None, None, None, None]); // 01 02 03 04
        app.active = 1;
        app.close(0); // a tab below the active one
        assert_eq!(app.sessions[app.active].number, 2, "the selection followed the shift");

        app.active = 2; // 04
        app.close(1); // 03, again below
        assert_eq!(app.sessions[app.active].number, 4);
    }

    /// Closing the active tab selects its neighbour, and never runs off the end.
    #[test]
    fn closing_the_active_tab_keeps_the_index_in_bounds() {
        let mut app = with(&[None, None, None]);
        app.active = 2;
        app.close(2); // the last one
        assert_eq!(app.active, 1);
        assert_eq!(app.sessions[app.active].number, 2);

        app.active = 0;
        app.close(0);
        assert_eq!(app.active, 0);
        assert_eq!(app.sessions[app.active].number, 2);
    }

    #[test]
    fn closing_the_last_tab_leaves_a_fresh_one_rather_than_an_empty_window() {
        let mut app = with(&[Some("a.wav")]);
        app.close(0);
        assert_eq!(app.sessions.len(), 1);
        assert_eq!(app.sessions[0].title(), "01 (no input)");
        assert_eq!(app.active, 0);
    }

    /// A duplicate carries the inputs and the settings and nothing that came out of them: a book
    /// shown beside settings that did not produce it is the one thing a comparison view must not do.
    #[test]
    fn a_duplicate_carries_the_settings_but_not_the_results() {
        let mut app = with(&[Some("a.wav")]);
        app.sessions[0].start = "2.5".into();
        app.sessions[0].residual_analysis = true;
        app.sessions[0].config.pursuit.max_atoms = 4321;
        app.sessions[0].log.push("something happened".into());

        let copy = app.sessions[0].duplicate(7);
        assert_eq!(copy.number, 7);
        assert_eq!(copy.input, app.sessions[0].input);
        assert_eq!(copy.start, "2.5");
        assert!(copy.residual_analysis);
        assert_eq!(copy.config.pursuit.max_atoms, 4321);
        assert!(copy.log.is_empty());
        assert!(copy.outcome.is_none());
        assert!(copy.running.is_none());
    }
}
