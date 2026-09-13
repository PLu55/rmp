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

use crate::help::Help;
use crate::settings::SettingsDoc;
use crate::task::{self, Outcome, Progress, Running, Update};
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
    /// The file this tab is the analysis of. Not an `Option` and never written after construction:
    /// a tab exists because a file was opened, and its results, log and title all describe that one
    /// file. Swapping it would leave a book describing a file the tab no longer names.
    input: PathBuf,
    start: String,
    duration: String,
    /// The settings this tab analyses with: a document, loaded and saved as a file.
    settings: SettingsDoc,
    /// The *effective* settings the displayed results came from — see `SettingsDoc::effective`.
    /// `None` until a run starts. What makes a result knowable as stale once the document moves on.
    ran_with: Option<String>,
    running: Option<Running>,
    /// The last finished run, if any.
    outcome: Option<Box<Outcome>>,
    log: Vec<String>,
    view: View,
}

impl Session {
    fn new(number: u32, input: PathBuf) -> Self {
        Self {
            number,
            input,
            start: String::new(),
            duration: String::new(),
            settings: SettingsDoc::default(),
            ran_with: None,
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
            settings: self.settings.clone(),
            ran_with: None,
            running: None,
            outcome: None,
            log: Vec::new(),
            view: self.view,
        }
    }

    /// `NN filename.wav`.
    ///
    /// The fallback is the whole path, for the paths that have no final component at all. A file
    /// chosen from a dialog always has one, so it is a formality rather than a case to plan around.
    fn title(&self) -> String {
        let name = self
            .input
            .file_name()
            .map_or_else(|| self.input.display().to_string(), |n| n.to_string_lossy().into_owned());
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
            // Reported, not chosen: a tab's file is what the tab is.
            ui.label(self.input.display().to_string())
                .on_hover_text("a tab's file cannot be changed; Open… puts another file in its own tab");

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
                    if ui.button("Analyse").clicked() {
                        self.start_run();
                    }
                }
            }
        });
    }

    fn start_run(&mut self) {
        let Ok(config) = self.settings.status() else { return };
        let config = config.clone();
        // Recorded now rather than on completion, so an interrupted run is still attributed to the
        // settings it ran under.
        self.ran_with = self.settings.effective().map(str::to_owned);
        self.log.clear();
        self.outcome = None;
        self.running = Some(task::spawn(task::Job {
            input: self.input.clone(),
            config,
            start: self.start.trim().parse().ok(),
            duration: self.duration.trim().parse().ok(),
        }));
    }

    /// Whether the settings panel has moved on from what the displayed results came from.
    ///
    /// Compared through `effective`, so reflowing the document or annotating a line does not read
    /// as a change. A document that currently does not parse counts as stale: it cannot be shown to
    /// agree with anything.
    fn results_are_stale(&self) -> bool {
        match (&self.ran_with, self.settings.effective()) {
            (Some(ran), Some(now)) => ran != now,
            (Some(_), None) => true,
            (None, _) => false,
        }
    }

    /// The settings document: load, edit, save, save as.
    ///
    /// The residue's ERB analysis is `[residual] enabled` in the document like everything else.
    /// It used to have a checkbox of its own here, ANDed with the setting — two controls for one
    /// thing, and no way to tell from the panel which of them was the one saying no.
    fn settings(&mut self, ui: &mut egui::Ui, out: &mut SettingsOut) {
        let err = &mut out.error;
        ui.horizontal(|ui| {
            ui.heading("Settings");
            if ui
                .button("?")
                .on_hover_text("what every setting does, from MANUAL.md")
                .clicked()
            {
                out.open_help = true;
            }
            if ui.button("Load…").clicked()
                && let Some(p) = pick_settings()
            {
                match SettingsDoc::load(&p) {
                    // Deliberately not reset: the results stay, and `results_are_stale` starts
                    // reporting them against the document that is now on screen.
                    Ok(d) => self.settings = d,
                    Err(e) => *err = Some(e),
                }
            }
            // Nothing to write, or nowhere to write it.
            let can_save = self.settings.path().is_some() && self.settings.modified();
            if ui
                .add_enabled(can_save, egui::Button::new("Save"))
                .on_disabled_hover_text(if self.settings.path().is_some() {
                    "no changes to save"
                } else {
                    "this document has no file yet — use Save as"
                })
                .clicked()
                && let Err(e) = self.settings.save()
            {
                *err = Some(e);
            }
            if ui.button("Save as…").clicked()
                && let Some(p) = pick_settings_save(self.settings.path())
                && let Err(e) = self.settings.save_as(&p)
            {
                *err = Some(e);
            }
        });

        ui.horizontal(|ui| {
            match self.settings.path() {
                Some(p) => ui.label(p.display().to_string()),
                None => ui.weak("unsaved — the built-in defaults"),
            };
            if self.settings.modified() {
                ui.strong("*");
            }
        });
        ui.separator();

        // The verdict, above the editor rather than below it, so it does not move as the document
        // grows.
        match self.settings.status() {
            Ok(c) => {
                let blocks = c.dictionary_shapes().len();
                ui.label(format!("valid — {blocks} block shapes"));
            }
            Err(e) => {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
        }
        ui.add_space(4.0);

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let resp = ui.add_sized(
                ui.available_size(),
                egui::TextEdit::multiline(self.settings.text_mut()).code_editor(),
            );
            if resp.changed() {
                self.settings.reparse();
            }
        });
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
            ui.centered_and_justified(|ui| ui.label("Press Analyse."));
            return;
        };

        let stale = self.results_are_stale();
        ui.horizontal(|ui| {
            for v in View::ALL {
                ui.selectable_value(&mut self.view, v, v.label());
            }
        });
        // The results are a fact about the settings that produced them, and the panel beside them
        // no longer shows those. Said here rather than by hiding or discarding the book: a
        // decomposition can take minutes, and it is still the result you got.
        if stale {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "settings have changed since this run — analyse again to match them",
            );
        }
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

/// What the settings panel asks of the window.
///
/// The panel is a `Session` method, so it can reach neither the help window nor its own log while
/// drawing. Both are carried out and applied once the borrow has ended.
#[derive(Default)]
struct SettingsOut {
    error: Option<String>,
    open_help: bool,
}

/// What a click on the tab strip asked for, applied after the strip has been drawn.
///
/// The strip borrows `sessions` to draw itself, so it cannot add to or remove from that same list
/// while iterating. One deferred action per frame is enough: nothing here can be clicked twice.
enum Action {
    Select(usize),
    Close(usize),
    Duplicate,
    Open(PathBuf),
}

/// The window.
///
/// Starts with no tabs, and the default is exactly that: a tab is the analysis of a file, so there
/// is nothing to show one of until a file has been opened, and nothing to seed the list with.
#[derive(Default)]
pub struct RmpApp {
    sessions: Vec<Session>,
    /// One help window for the whole app, not one per tab: it is the manual, and it is the same
    /// manual whichever tab you asked from.
    help: Help,
    /// Index into `sessions`. Meaningless while that is empty, which is the one time nothing
    /// indexes it.
    active: usize,
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

        // No tabs: the window a fresh start looks like, and the one it returns to when the last
        // tab is closed. The per-tab panels are not drawn at all rather than drawn empty, because
        // there is no session for them to be about.
        if self.sessions.is_empty() {
            let mut opened = None;
            egui::CentralPanel::default().show(ui, |ui| {
                // Not `centered_and_justified`: that justifies the *widget* too, and a button
                // stretched over the whole panel reads as a broken background rather than a button.
                ui.add_space(ui.available_height() * 0.4);
                ui.vertical_centered(|ui| {
                    if ui.button("Open a soundfile…").clicked() {
                        opened = pick_file();
                    }
                    ui.add_space(8.0);
                    ui.weak("each file gets a tab of its own");
                });
            });
            if let Some(p) = opened {
                self.open(p);
            }
            return;
        }

        // Panel ids are shared across tabs on purpose: a panel's width and height are window
        // chrome, and they should not jump when you switch. Widget state inside them is not — see
        // the `push_id` below.
        let salt = self.sessions[self.active].number;
        // A failed load or save belongs in the tab's own log, but the panel drawing it holds the
        // session borrow; it is carried out and pushed afterwards.
        let mut out = SettingsOut::default();
        let session = &mut self.sessions[self.active];
        egui::Panel::top("input").show(ui, |ui| {
            ui.push_id(salt, |ui| session.input_bar(ui));
        });
        egui::Panel::left("settings").default_size(440.0).show(ui, |ui| {
            ui.push_id(salt, |ui| session.settings(ui, &mut out));
        });
        egui::Panel::bottom("log").resizable(true).default_size(140.0).show(ui, |ui| {
            ui.push_id(salt, |ui| session.log_panel(ui));
        });
        egui::CentralPanel::default().show(ui, |ui| {
            ui.push_id(salt, |ui| session.results(ui));
        });
        if let Some(e) = out.error {
            session.log.push(format!("settings: {e}"));
        }
        if out.open_help {
            self.help.open_or_raise();
        }
        // Outside every panel: it is a window of its own, not part of this one's layout.
        self.help.show(ui);
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
                        // A background run is visible without switching to it, and so are unsaved
                        // settings — both are reasons to come back to a tab you are not looking at.
                        if s.running.is_some() {
                            ui.spinner();
                        }
                        if s.settings.modified() {
                            ui.strong("*").on_hover_text("unsaved settings changes");
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

                if !self.sessions.is_empty() {
                    ui.separator();
                }
                if ui
                    .button("Open…")
                    .on_hover_text("a soundfile, in a tab of its own")
                    .clicked()
                    && let Some(p) = pick_file()
                {
                    action = Some(Action::Open(p));
                }
                if ui
                    .add_enabled(!self.sessions.is_empty(), egui::Button::new("Duplicate"))
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
            Some(Action::Duplicate) => {
                self.push(self.sessions[self.active].duplicate(self.free_number()))
            }
            Some(Action::Open(p)) => self.open(p),
            None => {}
        }
    }

    /// Open a file, in a tab of its own. The only way a tab comes into existence.
    ///
    /// A tab is never empty and its file never changes, and both of those are facts about
    /// `Session` rather than rules the UI has to keep remembering: `input` is a `PathBuf` set at
    /// construction, so there is no state for an empty tab to be in and nothing to write a second
    /// file into.
    ///
    /// The new tab starts from the defaults. Carrying settings across to a different file is
    /// `Duplicate`'s job, and one button cannot be both without becoming unpredictable.
    fn open(&mut self, path: PathBuf) {
        self.push(Session::new(self.free_number(), path));
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
    /// some future call site. Closing the last tab leaves *no* tabs, and the window falls back to
    /// the same empty state it starts in — a tab is the analysis of a file, so there is nothing to
    /// show one of once the last file is closed.
    ///
    /// **Clamping `active` is not enough on its own.** Removing a tab *below* the active one shifts
    /// the rest down, so the same index now names a different tab — you close tab 1 of four with
    /// tab 2 selected and find yourself looking at what was tab 3. The index has to follow the
    /// shift first, and only then be clamped for the case where the active tab was itself the last.
    fn close(&mut self, i: usize) {
        self.sessions.remove(i);
        if i < self.active {
            self.active -= 1;
        }
        // `saturating_sub` for the empty case, where `active` is not an index into anything.
        self.active = self.active.min(self.sessions.len().saturating_sub(1));
    }
}

/// The one audio file dialog, shared by the tab strip and the empty window.
fn pick_file() -> Option<PathBuf> {
    rfd::FileDialog::new().add_filter("audio", &["wav", "aiff", "aif", "flac"]).pick_file()
}

fn pick_settings() -> Option<PathBuf> {
    rfd::FileDialog::new().add_filter("settings", &["toml"]).pick_file()
}

/// Save-as, starting wherever the document currently lives.
fn pick_settings_save(current: Option<&std::path::Path>) -> Option<PathBuf> {
    let mut d = rfd::FileDialog::new().add_filter("settings", &["toml"]);
    if let Some(p) = current {
        if let Some(dir) = p.parent() {
            d = d.set_directory(dir);
        }
        if let Some(name) = p.file_name() {
            d = d.set_file_name(name.to_string_lossy());
        }
    }
    d.save_file()
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

    /// Change a setting, the way typing over its value in the editor would.
    ///
    /// By replacing the line in place rather than appending: the document ends inside
    /// `[residual.power]`, so an appended `max_atoms = …` is an unknown field there and *breaks*
    /// the document instead of changing it — which is subtle enough that the first version of
    /// these tests did exactly that, and the staleness one passed for the wrong reason.
    fn set(s: &mut Session, key: &str, value: &str) {
        let text = s.settings.text_mut();
        let prefix = format!("{key} = ");
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with(&prefix))
            .unwrap_or_else(|| panic!("{key} is not in the default document"))
            .to_string();
        *text = text.replace(&line, &format!("{key} = {value}"));
        s.settings.reparse();
        assert!(s.settings.status().is_ok(), "the fixture broke the document: {:?}", s.settings.status().err());
    }

    /// A change that reaches the text but not the analysis. Safe to append: a comment is legal
    /// anywhere, including at the end of the last section.
    fn comment(s: &mut Session, note: &str) {
        s.settings.text_mut().push_str(&format!("\n# {note}\n"));
        s.settings.reparse();
        assert!(s.settings.status().is_ok());
    }

    fn break_doc(s: &mut Session) {
        s.settings.text_mut().push_str("\nthis is not toml =\n");
        s.settings.reparse();
        assert!(s.settings.status().is_err(), "the fixture was supposed to break it");
    }

    /// An app holding one tab per path, built the way the UI builds them.
    fn with(paths: &[&str]) -> RmpApp {
        let mut app = RmpApp::default();
        for p in paths {
            app.open(PathBuf::from(p));
        }
        app
    }

    /// A tab exists because a file was opened, so there is nothing to show one of before that.
    #[test]
    fn the_window_starts_with_no_tabs() {
        let app = RmpApp::default();
        assert!(app.sessions.is_empty());
    }

    #[test]
    fn a_title_is_a_two_digit_number_then_the_file_name() {
        let app = with(&["/a/b/chopin-nocturne-2.wav", "zyklus.wav"]);
        assert_eq!(app.sessions[0].title(), "01 chopin-nocturne-2.wav");
        assert_eq!(app.sessions[1].title(), "02 zyklus.wav");
    }

    /// Open is the only way a tab appears, and it always makes a new one — there is no empty tab
    /// for it to land in, and it never touches the file of a tab that exists.
    #[test]
    fn open_always_makes_a_new_tab_and_selects_it() {
        let mut app = RmpApp::default();

        app.open(PathBuf::from("/a/piano.wav"));
        assert_eq!(app.sessions.len(), 1);
        assert_eq!(app.active, 0);

        app.open(PathBuf::from("/a/zyklus.wav"));
        assert_eq!(app.sessions.len(), 2, "a second file is a second tab");
        assert_eq!(app.active, 1, "the new tab is selected");
        assert_eq!(app.sessions[0].title(), "01 piano.wav", "the first tab is untouched");
    }

    /// The rule behind it: a tab's results and log describe the file it was opened with, and
    /// nothing can point them at another one.
    #[test]
    fn opening_more_files_never_disturbs_an_existing_tab() {
        let mut app = with(&["/a/piano.wav"]);
        app.sessions[0].log.push("analysed piano".into());

        app.open(PathBuf::from("/a/zyklus.wav"));
        app.active = 0;
        app.open(PathBuf::from("/a/drums.wav"));

        assert_eq!(app.sessions[0].title(), "01 piano.wav");
        assert_eq!(app.sessions[0].log, ["analysed piano"], "its log still describes its own file");
        let titles: Vec<String> = app.sessions.iter().map(|s| s.title()).collect();
        assert_eq!(titles, ["01 piano.wav", "02 zyklus.wav", "03 drums.wav"]);
    }

    /// A tab Open creates is a default one, not a copy of whatever was selected. Carrying settings
    /// to a different file is `Duplicate`'s job.
    #[test]
    fn a_tab_that_open_creates_starts_from_the_defaults() {
        let mut app = with(&["/a/piano.wav"]);
        set(&mut app.sessions[0], "max_atoms", "4321");
        app.sessions[0].start = "2.5".into();

        app.open(PathBuf::from("/a/zyklus.wav"));
        let new = &app.sessions[1];
        assert_eq!(new.settings.effective(), SettingsDoc::default().effective());
        assert!(!new.settings.modified());
        assert!(new.start.is_empty());
    }

    /// The results describe the settings that produced them, so the panel moving on has to be
    /// visible. Compared through `effective`, so a comment is not a change.
    #[test]
    fn results_go_stale_when_the_settings_move_on_but_not_when_a_comment_does() {
        let mut app = with(&["/a/piano.wav"]);
        let s = &mut app.sessions[0];

        assert!(!s.results_are_stale(), "nothing has run yet");

        // Stand in for a finished run: this is what `start_run` records.
        s.ran_with = s.settings.effective().map(str::to_owned);
        assert!(!s.results_are_stale());

        comment(s, "just a note");
        assert!(!s.results_are_stale(), "a comment does not change the analysis");

        set(s, "max_atoms", "4321");
        assert!(s.results_are_stale(), "a real change does");
    }

    /// A document that does not parse cannot be shown to agree with anything.
    #[test]
    fn a_broken_settings_document_reads_as_stale() {
        let mut app = with(&["/a/piano.wav"]);
        let s = &mut app.sessions[0];
        s.ran_with = s.settings.effective().map(str::to_owned);

        break_doc(s);
        assert!(s.results_are_stale());
    }

    /// Reuse is what keeps the number two digits: closing the middle tab frees 02, and the next
    /// new tab takes it rather than counting on to 04.
    #[test]
    fn a_closed_number_is_reused_by_the_next_tab() {
        let mut app = with(&["a.wav", "b.wav", "c.wav"]);
        assert_eq!(app.free_number(), 4);
        app.close(1);
        assert_eq!(app.free_number(), 2);
        app.open(PathBuf::from("d.wav"));
        let mut got: Vec<u32> = app.sessions.iter().map(|s| s.number).collect();
        got.sort_unstable();
        assert_eq!(got, [1, 2, 3]);
    }

    #[test]
    fn numbers_stay_unique_across_a_run_of_opens_and_closes() {
        let mut app = with(&["a.wav"]);
        for _ in 0..20 {
            app.open(PathBuf::from("b.wav"));
            app.open(PathBuf::from("c.wav"));
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
        let mut app = with(&["a.wav", "b.wav", "c.wav", "d.wav"]); // 01 02 03 04
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
        let mut app = with(&["a.wav", "b.wav", "c.wav"]);
        app.active = 2;
        app.close(2); // the last one
        assert_eq!(app.active, 1);
        assert_eq!(app.sessions[app.active].number, 2);

        app.active = 0;
        app.close(0);
        assert_eq!(app.active, 0);
        assert_eq!(app.sessions[app.active].number, 2);
    }

    /// Back to the window a fresh start shows, rather than to a blank tab.
    #[test]
    fn closing_the_last_tab_leaves_no_tabs() {
        let mut app = with(&["a.wav"]);
        app.close(0);
        assert!(app.sessions.is_empty());
        // `active` is not an index into anything now, but it must not be left out of range for the
        // next open either.
        app.open(PathBuf::from("b.wav"));
        assert_eq!(app.active, 0);
        assert_eq!(app.sessions[app.active].title(), "01 b.wav");
    }

    /// A duplicate carries the inputs and the settings and nothing that came out of them: a book
    /// shown beside settings that did not produce it is the one thing a comparison view must not do.
    #[test]
    fn a_duplicate_carries_the_settings_but_not_the_results() {
        let mut app = with(&["a.wav"]);
        app.sessions[0].start = "2.5".into();
        set(&mut app.sessions[0], "max_atoms", "4321");
        app.sessions[0].log.push("something happened".into());
        app.sessions[0].ran_with = Some("whatever ran".into());

        let copy = app.sessions[0].duplicate(7);
        assert_eq!(copy.number, 7);
        assert_eq!(copy.input, app.sessions[0].input);
        assert_eq!(copy.start, "2.5");
        assert_eq!(copy.settings.status().unwrap().pursuit.max_atoms, 4321);
        assert!(copy.log.is_empty());
        assert!(copy.outcome.is_none());
        assert!(copy.running.is_none());
        assert!(copy.ran_with.is_none(), "a copy has not run, so nothing of its own is stale");
        assert!(!copy.results_are_stale());
    }
}
