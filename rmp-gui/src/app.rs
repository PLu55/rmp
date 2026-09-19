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

use crate::audio::Audio;
use crate::help::Help;
use crate::playback::{self, Available, Sources, Which};
use crate::project::{self, ProjectDoc, TabDoc};
use crate::results::Results;
use crate::view::distribution::DistributionView;
use crate::view::function::FunctionView;
use crate::view::summary::SummaryView;
use crate::view::structure::StructureView;
use crate::view::timefreq::TimeFreqView;
use crate::settings::SettingsDoc;
use crate::task::{self, Progress, Running, Update};
use std::path::{Path, PathBuf};

/// What a tab's results area is showing.
///
/// Not called `Tab`: a tab is a document now, and two things by that name in one file is how the
/// results strip and the document strip get confused for each other.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum View {
    /// `rmp_core::stats::BookSummary` — the same figures `rmpstat summary` prints.
    Summary,
    /// `rmp_core::stats::Histogram` over a `Quantity`, as `rmpstat hist`.
    Distribution,
    /// Curves over the book, as a function of atom index — `Book::snr_trace` and its relatives.
    Function,
    /// `rmp_core::tfmap::TfMap` — the atom-based pseudo-Wigner map, as `rmpstat wv`.
    TimeFrequency,
    /// `rmp_structure::analyze_partials` — persistent partials, as `rmpstruct partials`.
    Structure,
}

impl View {
    const ALL: [View; 5] = [
        View::Summary,
        View::Distribution,
        View::Function,
        View::TimeFrequency,
        View::Structure,
    ];

    fn label(self) -> &'static str {
        match self {
            View::Summary => "Summary",
            View::Distribution => "Distributions",
            View::Function => "Functions",
            View::TimeFrequency => "Time-frequency",
            View::Structure => "Structure",
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
    /// A render in flight. Independent of `running`: a tab can be analysing the next excerpt while
    /// the previous book is still being written out.
    synthesising: Option<task::Synthesising>,
    /// The Analyse panel's own switches. Not analysis *settings* — they decide which by-products
    /// to keep, not how the pursuit runs — so they are not in the settings document. What they do
    /// decide is which sources Play can offer afterwards.
    keep_residual: bool,
    run_residual_analysis: bool,
    /// The Synthesize panel: which halves of the book to write.
    parts: task::RenderParts,
    /// The Play panel: which sources to hear together.
    sources: Sources,
    /// Where the last run's book went, so re-analysing proposes the same file rather than making a
    /// new one every time a setting is nudged.
    book_path: Option<PathBuf>,
    /// What the *finished* run kept, as against what is ticked now. A switch changed after a run
    /// must not make a source look available that the run did not produce.
    available: Available,
    /// The last finished run, or a book read back from disk after a project restored this tab.
    results: Option<Results>,
    log: Vec<String>,
    view: View,
    /// The result tabs' own state: what each has computed, and under which options. Invalidated
    /// together when a run finishes, since all three are views of one book.
    summary_view: SummaryView,
    distribution_view: DistributionView,
    function_view: FunctionView,
    timefreq_view: TimeFreqView,
    structure_view: StructureView,
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
            synthesising: None,
            keep_residual: true,
            run_residual_analysis: false,
            parts: task::RenderParts::default(),
            sources: Sources::default(),
            book_path: None,
            available: Available::default(),
            results: None,
            log: Vec::new(),
            view: View::Summary,
            summary_view: SummaryView::default(),
            distribution_view: DistributionView::default(),
            function_view: FunctionView::default(),
            timefreq_view: TimeFreqView::default(),
            structure_view: StructureView::default(),
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
            synthesising: None,
            keep_residual: self.keep_residual,
            run_residual_analysis: self.run_residual_analysis,
            parts: self.parts,
            sources: self.sources,
            // Not the original's file: two tabs writing one book is the collision the tab number
            // exists to prevent, and `default_book_name` already numbers them apart.
            book_path: None,
            available: Available::default(),
            results: None,
            log: Vec::new(),
            view: self.view,
            summary_view: SummaryView::default(),
            distribution_view: DistributionView::default(),
            function_view: FunctionView::default(),
            timefreq_view: TimeFreqView::default(),
            structure_view: StructureView::default(),
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

    /// Whether anything is happening in this tab that the window has to keep repainting for.
    fn busy(&self) -> bool {
        self.running.is_some() || self.synthesising.is_some()
    }

    /// Drain the worker and fold its messages into this tab's state.
    ///
    /// Called for every session each frame, not only the visible one: a tab analysing in the
    /// background still has to drain its channel, or its log and its result appear all at once at
    /// the moment you switch to it and the tab looks frozen until then.
    fn pump(&mut self) {
        if let Some(synth) = &mut self.synthesising {
            for update in synth.drain() {
                self.log.push(match update {
                    task::SynthUpdate::Done(summary) => summary,
                    task::SynthUpdate::Failed(e) => format!("synthesis failed: {e}"),
                });
            }
            if synth.finished() {
                self.synthesising = None;
            }
        }

        let Some(run) = &mut self.running else { return };
        for update in run.drain() {
            match update {
                Update::Progress(p) => self.log.push(describe(&p)),
                Update::Done(outcome) => {
                    let a = &outcome.analysis;
                    // Recorded from the *finished* run, so a switch flipped afterwards cannot make
                    // Play offer a source this decomposition never produced.
                    self.available = Available::of(a, self.keep_residual);
                    // Every result tab is a view of the book that just changed.
                    self.summary_view.invalidate();
                    self.distribution_view.invalidate();
                    self.function_view.invalidate();
                    self.timefreq_view.invalidate();
                    self.structure_view.invalidate();
                    // And anything already ticked that the run did not make is dropped, rather than
                    // left ticked and silently ignored.
                    for w in Which::ALL {
                        if !self.available.has(w) {
                            w.set(&mut self.sources, false);
                        }
                    }
                    self.log.push(format!(
                        "{} atoms, {:.1} dB{}",
                        a.book.len(),
                        a.book.snr_db(),
                        if a.cancelled { " (interrupted)" } else { "" }
                    ));
                    self.results = Some(Results::Run(outcome));
                }
                Update::Failed(e) => self.log.push(format!("failed: {e}")),
            }
        }
        if run.finished() {
            self.running = None;
        }
    }

    /// The three things a tab does, each with the switches that belong to it.
    ///
    /// Grouped rather than laid out as one row of buttons because the switches only make sense
    /// beside their verb: "residual" means a different thing to each of the three, and a flat row
    /// of six checkboxes would leave that ambiguous.
    fn controls(&mut self, ui: &mut egui::Ui, out: &mut SettingsOut, playing: bool, project_dir: Option<&Path>) {
        ui.label(self.input.display().to_string())
            .on_hover_text("a tab's file cannot be changed; Open… puts another file in its own tab");

        ui.horizontal_top(|ui| {
            self.analyse_panel(ui, project_dir);
            self.synthesise_panel(ui, project_dir);
            self.play_panel(ui, out, playing);
        });
    }

    fn analyse_panel(&mut self, ui: &mut egui::Ui, project_dir: Option<&Path>) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
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
                            if ui
                                .add_enabled(
                                    self.settings.status().is_ok(),
                                    egui::Button::new("Analyse"),
                                )
                                .on_disabled_hover_text("the settings document does not parse")
                                .clicked()
                            {
                                self.start_run(project_dir);
                            }
                        }
                    }
                    ui.label("start");
                    ui.add(egui::TextEdit::singleline(&mut self.start).desired_width(48.0));
                    ui.label("duration");
                    ui.add(egui::TextEdit::singleline(&mut self.duration).desired_width(48.0));
                    ui.label("s");
                });
                ui.checkbox(&mut self.keep_residual, "residual")
                    .on_hover_text("keep what the atoms could not explain, to play or compare");
                ui.checkbox(&mut self.run_residual_analysis, "residual analysis")
                    .on_hover_text(
                        "measure the residue into ERB band powers, which is what a synthesised \
                         residual is rebuilt from",
                    );
            });
        });
    }

    fn synthesise_panel(&mut self, ui: &mut egui::Ui, project_dir: Option<&Path>) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                match &self.synthesising {
                    Some(_) => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("rendering…");
                        });
                    }
                    None => {
                        // Checked against what is actually ticked, not merely "is there a book":
                        // finding out after the save dialog would be worse than a disabled button.
                        let can = if self.results.is_some() {
                            self.parts.available(self.available)
                        } else {
                            Err("analyse something first")
                        };
                        if ui
                            .add_enabled(can.is_ok(), egui::Button::new("Synthesize"))
                            .on_disabled_hover_text(can.err().unwrap_or_default())
                            .on_hover_text("write this book back out as a soundfile")
                            .clicked()
                        {
                            self.start_synthesis(project_dir);
                        }
                    }
                }
                ui.checkbox(&mut self.parts.atoms, "atoms");
                ui.checkbox(&mut self.parts.residual, "residual (synthesised)")
                    .on_hover_text("the stochastic model, not the measured residue");
            });
        });
    }

    fn play_panel(&mut self, ui: &mut egui::Ui, out: &mut SettingsOut, playing: bool) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                if playing {
                    if ui.button("Stop").clicked() {
                        out.stop = true;
                    }
                } else {
                    let ready = self.results.is_some() && self.sources.any();
                    if ui
                        .add_enabled(ready, egui::Button::new("Play"))
                        .on_disabled_hover_text(if self.results.is_none() {
                            "analyse something first"
                        } else {
                            "nothing selected to play"
                        })
                        .on_hover_text("sound the ticked sources together, at their own levels")
                        .clicked()
                    {
                        out.play = true;
                    }
                }
                // Ticked against what the *finished run* produced, not what is ticked in Analyse
                // now: changing a switch after a run must not offer a source that run never made.
                for w in Which::ALL {
                    let has = self.results.is_some() && self.available.has(w);
                    let mut on = w.get(&self.sources) && has;
                    ui.add_enabled_ui(has, |ui| {
                        if ui.checkbox(&mut on, w.label()).on_disabled_hover_text(w.why_not()).changed() {
                            w.set(&mut self.sources, on);
                        }
                    });
                }
            });
        });
    }

    fn start_run(&mut self, project_dir: Option<&Path>) {
        let Ok(config) = self.settings.status() else { return };
        let config = config.clone();
        // Asked before the run, as `rmp -b` requires it: a decomposition that took minutes and then
        // had nowhere to go would be the worst outcome here. Cancelling the dialog cancels the run.
        let Some(book_output) =
            pick_book_save(self.book_path.as_deref(), project_dir, &self.default_book_name())
        else {
            return;
        };
        self.book_path = Some(book_output.clone());
        // Recorded now rather than on completion, so an interrupted run is still attributed to the
        // settings it ran under.
        self.ran_with = self.settings.effective().map(str::to_owned);
        self.log.clear();
        self.results = None;
        self.running = Some(task::spawn(task::Job {
            input: self.input.clone(),
            config,
            start: self.start.trim().parse().ok(),
            duration: self.duration.trim().parse().ok(),
            book_output,
            keep_residual: self.keep_residual,
            residual_analysis: self.run_residual_analysis,
        }));
    }

    /// The file name Save as proposes: `<audio file, less its extension>-<tab number>.toml`.
    ///
    /// Derived every time rather than only for a document that has no file yet, because the tab
    /// number in it is the point. Duplicating a tab copies the settings *and the file they came
    /// from*, so two tabs comparing one setting on one soundfile would both propose the original's
    /// name and the second would silently offer to overwrite the first. The number is what tells
    /// them apart, and it only helps if it is the name actually offered.
    ///
    /// The directory still comes from wherever the document was last saved — see
    /// `pick_settings_save` — so this changes what is proposed, not where.
    fn start_synthesis(&mut self, project_dir: Option<&Path>) {
        let Some(results) = &self.results else { return };
        let Some(output) = pick_audio_save(&self.default_render_name(), project_dir) else {
            return;
        };
        // Cloned rather than borrowed: the render outlives this frame on a thread of its own, and
        // the tab stays live meanwhile — you can edit its settings, or start the next analysis.
        self.synthesising = Some(task::spawn_synthesis(task::SynthJob {
            book: results.book().clone(),
            residual_book: results.residual_book().cloned(),
            parts: self.parts,
            output,
        }));
    }

    fn default_settings_name(&self) -> String {
        let stem = self
            .input
            .file_stem()
            .map_or_else(|| "settings".to_string(), |s| s.to_string_lossy().into_owned());
        format!("{stem}-{:02}.toml", self.number)
    }

    /// The file name Analyse proposes for the book:
    /// `<audio file, less its extension>-<tab number>-book[-residual].json.gz`.
    ///
    /// Gzipped by default because a JSON book is 1.19x the size of the f32 WAV it decomposes and
    /// the suffix is worth about 7x; `book::write` reads the format from the extension *beneath*
    /// the `.gz`, so this is a JSON book either way.
    ///
    /// The `-residual` is there when the run will measure one, and it is worth saying in the name:
    /// the residual book is roughly 12x the atom list on a short excerpt, so two files of the same
    /// stem can differ by an order of magnitude in size and in what they can be rendered into.
    fn default_book_name(&self) -> String {
        let stem = self
            .input
            .file_stem()
            .map_or_else(|| "book".to_string(), |s| s.to_string_lossy().into_owned());
        let residual = if self.run_residual_analysis { "-residual" } else { "" };
        format!("{stem}-{:02}-book{residual}.json.gz", self.number)
    }

    /// The file name Synthesize proposes:
    /// `<audio file, less its extension>-<tab number>-<what it holds>.wav`.
    ///
    /// The tab number is there for the same reason the settings document has one: two tabs on one
    /// soundfile must not propose one output file. The mode is there so the four kinds of render do
    /// not overwrite each other either — the whole point of having them is to hear them together.
    fn default_render_name(&self) -> String {
        let stem = self
            .input
            .file_stem()
            .map_or_else(|| "render".to_string(), |s| s.to_string_lossy().into_owned());
        format!("{stem}-{:02}-{}.wav", self.number, self.parts.suffix())
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
    fn settings(&mut self, ui: &mut egui::Ui, out: &mut SettingsOut, project_dir: Option<&Path>) {
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
            if ui
                .add_enabled(project_dir.is_some(), egui::Button::new("Import…"))
                .on_disabled_hover_text("no project open — File > New or Open first")
                .on_hover_text("a settings file, copied into the project directory and loaded here")
                .clicked()
                && let Some(dir) = project_dir
                && let Some(p) = pick_settings()
                && let Err(e) = self.import_settings(&p, dir)
            {
                *err = Some(e);
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
                && let Some(p) = pick_settings_save(
                    self.settings.path(),
                    project_dir,
                    &self.default_settings_name(),
                )
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
        let Some(results) = &self.results else {
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

        // Each tab is a view of a call in `rmp-core`; none of them recomputes anything, which is
        // what keeps the window and `rmpstat` reporting one set of numbers.
        match self.view {
            View::Summary => self.summary_view.ui(ui, results),
            View::Distribution => self.distribution_view.ui(ui, results.book()),
            View::Function => self.function_view.ui(ui, results.book()),
            View::TimeFrequency => self.timefreq_view.ui(ui, results.book()),
            View::Structure => self.structure_view.ui(ui, results.book()),
        }
    }

    /// This tab, as a project document remembers it.
    fn to_tab_doc(&self, project_dir: &Path) -> TabDoc {
        TabDoc {
            number: self.number,
            input: project::store_path(project_dir, &self.input),
            start: self.start.clone(),
            duration: self.duration.clone(),
            settings: self.settings.path().map(|p| project::store_path(project_dir, p)),
            keep_residual: self.keep_residual,
            run_residual_analysis: self.run_residual_analysis,
            parts: self.parts,
            sources: self.sources,
            book: self.book_path.as_deref().map(|p| project::store_path(project_dir, p)),
            view: self.view,
        }
    }

    /// Rebuild a tab from a project document: its settings and, when a book is on record and reads
    /// back, its results — without re-running the pursuit. See [`load_results`].
    ///
    /// A tab whose book or input can no longer be read still opens, with the failure logged rather
    /// than refusing the whole project: settings and switches are still worth having back, and a
    /// tab is never empty regardless — there is always an `input`, even one that can no longer be
    /// found, which is exactly today's "Press Analyse." state.
    fn restore(project_dir: &Path, doc: &TabDoc) -> Self {
        let mut s = Self::new(doc.number, project::resolve_path(project_dir, &doc.input));
        s.start = doc.start.clone();
        s.duration = doc.duration.clone();
        s.keep_residual = doc.keep_residual;
        s.run_residual_analysis = doc.run_residual_analysis;
        s.parts = doc.parts;
        s.sources = doc.sources;
        s.view = doc.view;

        if let Some(p) = &doc.settings {
            let p = project::resolve_path(project_dir, p);
            match SettingsDoc::load(&p) {
                Ok(d) => s.settings = d,
                Err(e) => s.log.push(format!("settings: {e}")),
            }
        }

        if let Some(p) = &doc.book {
            let book_path = project::resolve_path(project_dir, p);
            s.book_path = Some(book_path.clone());
            match load_results(&s.input, &s.start, &s.duration, &book_path) {
                Ok(results) => {
                    s.available = Available::of_loaded(results.book());
                    // The settings on disk are what the loaded book is presumed to agree with;
                    // editing them from here starts `results_are_stale` reporting against a
                    // document the loaded book has moved on from, same as any other tab.
                    s.ran_with = s.settings.effective().map(str::to_owned);
                    s.results = Some(results);
                }
                Err(e) => s.log.push(format!("could not reload results: {e}")),
            }
        }
        s
    }

    /// Give this tab's settings a file, as part of Save Project: one that has never been saved gets
    /// the project's own default name, one that already has a path and unsaved edits is written
    /// back to it. A restored project can only read a settings document that actually exists.
    fn ensure_settings_saved(&mut self, project_dir: &Path) {
        if self.settings.path().is_none() {
            let name = self.default_settings_name();
            if let Err(e) = self.settings.save_as(&project_dir.join(name)) {
                self.log.push(format!("settings: {e}"));
            }
        } else if self.settings.modified()
            && let Err(e) = self.settings.save()
        {
            self.log.push(format!("settings: {e}"));
        }
    }

    /// Move this tab's settings file into `new_dir`, for Save Project As — but only when it is
    /// either unsaved or was living inside `old_dir`, the common case once a project defaults saves
    /// into its own directory. A settings document saved somewhere else on purpose is left alone:
    /// this re-homes what the project owns, not every file a tab happens to reference.
    fn rehome_settings(&mut self, old_dir: Option<&Path>, new_dir: &Path) {
        let inside_old =
            self.settings.path().is_some_and(|p| old_dir.is_some_and(|d| p.starts_with(d)));
        if self.settings.path().is_none() || inside_old {
            let name = self.default_settings_name();
            if let Err(e) = self.settings.save_as(&new_dir.join(name)) {
                self.log.push(format!("settings: {e}"));
            }
        }
    }

    /// The settings panel's Import: copy a settings document into the project directory, then load
    /// the copy — the same `copy_into_project` the tab strip's `import_audio` uses, so a project's
    /// settings, like its audio, need not depend on wherever the original file lives.
    ///
    /// Deliberately not reset otherwise: the results stay, and `results_are_stale` starts reporting
    /// them against the document that is now on screen, same as Load.
    fn import_settings(&mut self, src: &Path, project_dir: &Path) -> Result<(), String> {
        let dest = copy_into_project(src, project_dir)?;
        self.settings = SettingsDoc::load(&dest)?;
        Ok(())
    }
}

/// Re-read the excerpt a book was analysed from, and the book itself — the same two reads
/// `task::work` does before a run, minus the pursuit. Blocking, on the caller's thread: comparable
/// cost to `SettingsDoc::load`, which already blocks the same way, and there is no pursuit here to
/// make worth a worker thread.
fn load_results(input: &Path, start: &str, duration: &str, book_path: &Path) -> Result<Results, String> {
    let book = rmp_core::book::read(book_path)?;
    let read = rmp_core::audio::read(input).map_err(|e| e.to_string())?;
    let (offset, signal) = rmp_core::pipeline::excerpt(
        read.signal,
        start.trim().parse().ok(),
        duration.trim().parse().ok(),
    )?;
    Ok(Results::Loaded(Box::new(crate::results::Loaded { book, signal, offset })))
}

/// What the settings panel asks of the window.
///
/// The panel is a `Session` method, so it can reach neither the help window nor its own log while
/// drawing. Both are carried out and applied once the borrow has ended.
#[derive(Default)]
struct SettingsOut {
    error: Option<String>,
    open_help: bool,
    /// Sound the active tab's chosen sources. The device lives on `RmpApp` — one output for the
    /// whole window, since one pair of speakers is what the machine has.
    play: bool,
    stop: bool,
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
    /// A soundfile chosen for Import — copied into the project directory before it becomes a tab.
    Import(PathBuf),
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
    /// The output device, opened on the first Play and kept for the life of the window. One for the
    /// app rather than one per tab, because one pair of speakers is what the machine has — and
    /// playing two tabs at once would be a mix, not a comparison.
    audio: Option<Audio>,
    /// Which tab's sound is going, by its number. Only that tab offers Stop; the others keep
    /// offering Play, since stopping from a tab that is not sounding would be a surprise.
    playing_tab: Option<u32>,
    /// Index into `sessions`. Meaningless while that is empty, which is the one time nothing
    /// indexes it.
    active: usize,
    /// The open project, if any. `None` is free-standing mode, unchanged from before this existed:
    /// every tab behaves exactly as it always did, and nothing here is required to use the window.
    project: Option<Project>,
    /// New/Open Project asked for while the current tabs (or project) held unsaved changes; the
    /// confirm dialog is drawn from this rather than acting immediately.
    pending_replace: Option<PendingReplace>,
    /// A project-level error — creating a directory, reading `project.toml` — from before any tab
    /// exists to log it into.
    last_error: Option<String>,
}

/// The project a window has open: where it lives, and the document as it was last written, so a
/// freshly built one can be compared against it — the same trick `SettingsDoc` plays with
/// `text`/`saved`, one level up.
struct Project {
    dir: PathBuf,
    saved_doc: ProjectDoc,
}

#[derive(Clone)]
enum PendingReplace {
    New(PathBuf),
    Open(PathBuf),
}

impl eframe::App for RmpApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Every tab, not just the visible one — see `Session::pump`.
        for s in &mut self.sessions {
            s.pump();
        }
        // A run reports only once per window, which can be minutes apart, so the repaint has to be
        // asked for rather than waited on: without this the window sleeps and the log arrives late.
        // Asked for if *any* tab is busy, since a hidden tab still drives its strip spinner — and
        // a render that finished would otherwise sit unreported until the mouse moved.
        // Playing counts too: the Stop button has to turn back into Play when the sound ends, and
        // nothing else would wake the window to notice.
        if self.sessions.iter().any(Session::busy)
            || self.audio.as_ref().is_some_and(Audio::playing)
        {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui));
        egui::Panel::top("tabs").show(ui, |ui| self.tab_bar(ui));

        // No tabs: the window a fresh start looks like, and the one it returns to when the last
        // tab is closed. The per-tab panels are not drawn at all rather than drawn empty, because
        // there is no session for them to be about.
        if self.sessions.is_empty() {
            egui::CentralPanel::default().show(ui, |ui| {
                // A hint, not a second Open button: there is already one in the strip above, and
                // two ways to do the same thing in one empty window is one too many.
                ui.centered_and_justified(|ui| {
                    ui.weak("Open a soundfile — each one gets a tab of its own.");
                });
            });
            return;
        }

        // Panel ids are shared across tabs on purpose: a panel's width and height are window
        // chrome, and they should not jump when you switch. Widget state inside them is not — see
        // the `push_id` below.
        let salt = self.sessions[self.active].number;
        // A failed load or save belongs in the tab's own log, but the panel drawing it holds the
        // session borrow; it is carried out and pushed afterwards.
        let mut out = SettingsOut::default();
        // Read before the session borrow begins; the device is the window's, not the tab's. It is
        // *this tab* sounding that decides whether to offer Stop, not merely that something is —
        // otherwise every tab would offer to stop another tab's playback.
        let sounding = self.audio.as_ref().is_some_and(Audio::playing);
        if !sounding {
            self.playing_tab = None;
        }
        let project_dir = self.project.as_ref().map(|p| p.dir.as_path());
        let session = &mut self.sessions[self.active];
        let playing = sounding && self.playing_tab == Some(session.number);
        egui::Panel::top("input").show(ui, |ui| {
            ui.push_id(salt, |ui| session.controls(ui, &mut out, playing, project_dir));
        });
        egui::Panel::left("settings").default_size(440.0).show(ui, |ui| {
            ui.push_id(salt, |ui| session.settings(ui, &mut out, project_dir));
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
        if out.stop && let Some(a) = &mut self.audio {
            a.stop();
        }
        if out.play {
            // Opened on first use, so a machine with no sound card fails here rather than at
            // launch, over a feature it may never be asked for.
            if self.audio.is_none() {
                match Audio::open() {
                    Ok(a) => self.audio = Some(a),
                    Err(e) => session.log.push(e),
                }
            }
            if let Some(a) = &mut self.audio
                && let Some(r) = &session.results
            {
                // Mixed here rather than on a thread: rendering the atoms of a large book is the
                // one slow part, and it is the same render the pursuit already did per atom. If it
                // ever bites, this is the call to move, not the playback.
                match playback::mix(r.book(), r.residual(), r.residual_book(), r.signal(), session.sources) {
                    Ok(mix) => {
                        session.log.push(format!(
                            "playing {} — {:.2} s, peak {:.1} dBFS",
                            describe_sources(session.sources),
                            mix.len() as f32 / mix.sample_rate,
                            rmp_core::signal::db_fs(mix.peak() as f64),
                        ));
                        a.play_samples(&mix);
                        self.playing_tab = Some(session.number);
                    }
                    Err(e) => session.log.push(e),
                }
            }
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
                        // Background work is visible without switching to it, and so are unsaved
                        // settings — both are reasons to come back to a tab you are not looking at.
                        if s.busy() {
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
                    .add_enabled(self.project.is_some(), egui::Button::new("Import…"))
                    .on_disabled_hover_text("no project open — File > New or Open first")
                    .on_hover_text("a soundfile, copied into the project directory and opened in a tab of its own")
                    .clicked()
                    && let Some(p) = pick_file()
                {
                    action = Some(Action::Import(p));
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
            Some(Action::Import(p)) => self.import_audio(p),
            None => {}
        }
    }

    /// Open a file, in a tab of its own. Together with `Session::restore` (used when a project is
    /// opened), the only way a tab comes into existence — Import funnels through this too, once its
    /// copy into the project directory has finished.
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

    /// Import a soundfile: copy it into the open project's directory, then open it exactly as
    /// `open` would. Disabled with no project active — there is nowhere to copy into — so this is
    /// never reached without one; `import_audio` still checks, since a click and the panel that
    /// gated it are two different frames apart.
    fn import_audio(&mut self, src: PathBuf) {
        let Some(dir) = self.project.as_ref().map(|p| p.dir.clone()) else { return };
        match copy_into_project(&src, &dir) {
            Ok(dest) => self.open(dest),
            Err(e) => self.last_error = Some(e),
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

    /// What a project document would look like right now, if it were saved to `dir`. The single
    /// definition both `project_modified` and `save_project` use, so "what counts as the project"
    /// cannot drift between deciding there is something to save and actually saving it.
    fn build_project_doc(&self, dir: &Path) -> ProjectDoc {
        ProjectDoc::new(self.sessions.iter().map(|s| s.to_tab_doc(dir)).collect(), self.active)
    }

    /// Whether the open project has anything Save Project would change on disk — the same
    /// `text != saved` trick `SettingsDoc` plays, one level up: a freshly built document compared
    /// against the one last written.
    fn project_modified(&self) -> bool {
        self.project.as_ref().is_some_and(|p| p.saved_doc != self.build_project_doc(&p.dir))
    }

    /// Whether starting or opening a different project would discard something: any tab's unsaved
    /// settings, or the open project itself having moved on since it was last saved.
    fn is_dirty(&self) -> bool {
        self.sessions.iter().any(|s| s.settings.modified()) || self.project_modified()
    }

    fn new_project(&mut self) {
        let Some(dir) = pick_new_project_dir() else { return };
        if self.is_dirty() {
            self.pending_replace = Some(PendingReplace::New(dir));
        } else {
            self.do_new_project(dir);
        }
    }

    /// Creates the directory (a save-style dialog is what named it — see `pick_new_project_dir` —
    /// so it does not exist yet in the ordinary case) and writes an empty `project.toml`
    /// immediately, so "New" concretely leaves a project on disk rather than just an empty window
    /// waiting for a first Save.
    ///
    /// Refuses a directory that already holds a `project.toml` rather than overwriting it: "New" is
    /// for a project that does not exist yet, and a name that collides with one that does is a
    /// mistake to report, not silently fold into.
    fn do_new_project(&mut self, dir: PathBuf) {
        if dir.join(project::PROJECT_FILE).is_file() {
            self.last_error = Some(format!(
                "{} already has a project — Open it instead, or choose a different name",
                dir.display()
            ));
            return;
        }
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.last_error = Some(format!("creating {}: {e}", dir.display()));
            return;
        }
        self.sessions.clear();
        self.active = 0;
        let doc = self.build_project_doc(&dir);
        match rmp_core::book::write_doc(&dir.join(project::PROJECT_FILE), &doc) {
            Ok(()) => self.last_error = None,
            Err(e) => self.last_error = Some(e),
        }
        self.project = Some(Project { dir, saved_doc: doc });
    }

    fn open_project(&mut self) {
        let Some(dir) = rfd::FileDialog::new().pick_folder() else { return };
        if self.is_dirty() {
            self.pending_replace = Some(PendingReplace::Open(dir));
        } else {
            self.do_open_project(dir);
        }
    }

    /// The command-line entry point: `rmp-gui PROJECT` opens `dir` the moment the window has
    /// something to draw into. No dialog and no dirty check — there is nothing open yet to lose —
    /// so this goes straight to `do_open_project`, which already reports a missing directory or an
    /// unreadable `project.toml` through `last_error` exactly as a menu-driven Open Project does.
    pub(crate) fn open_project_at(&mut self, dir: PathBuf) {
        self.do_open_project(dir);
    }

    fn do_open_project(&mut self, dir: PathBuf) {
        match rmp_core::book::read_doc::<ProjectDoc>(&dir.join(project::PROJECT_FILE)) {
            Ok(doc) => {
                self.sessions = doc.tabs.iter().map(|t| Session::restore(&dir, t)).collect();
                self.active = doc.active.min(self.sessions.len().saturating_sub(1));
                self.project = Some(Project { dir, saved_doc: doc });
                self.last_error = None;
            }
            Err(e) => self.last_error = Some(format!("opening project: {e}")),
        }
    }

    /// Gives every tab's settings a file first (a restored project can only read a settings
    /// document that actually exists), then writes `project.toml` capturing the tabs as they are
    /// now.
    fn save_project(&mut self) {
        let Some(dir) = self.project.as_ref().map(|p| p.dir.clone()) else { return };
        for s in &mut self.sessions {
            s.ensure_settings_saved(&dir);
        }
        let doc = self.build_project_doc(&dir);
        match rmp_core::book::write_doc(&dir.join(project::PROJECT_FILE), &doc) {
            Ok(()) => {
                if let Some(p) = &mut self.project {
                    p.saved_doc = doc;
                }
                self.last_error = None;
            }
            Err(e) => self.last_error = Some(e),
        }
    }

    /// Re-homes each tab's *settings* document into the new directory and saves there — but does
    /// not move or copy any book or rendered soundfile already written to disk. Remembering what a
    /// tab produced is this feature's job; moving the files it produced is a distinct one, and
    /// building it silently into Save As would risk quietly duplicating or losing large books.
    fn save_project_as(&mut self) {
        let Some(new_dir) = rfd::FileDialog::new().pick_folder() else { return };
        if let Err(e) = std::fs::create_dir_all(&new_dir) {
            self.last_error = Some(format!("creating {}: {e}", new_dir.display()));
            return;
        }
        let old_dir = self.project.as_ref().map(|p| p.dir.clone());
        for s in &mut self.sessions {
            s.rehome_settings(old_dir.as_deref(), &new_dir);
        }
        self.project = Some(Project { dir: new_dir, saved_doc: ProjectDoc::default() });
        self.save_project();
    }

    fn apply_pending(&mut self, action: PendingReplace) {
        match action {
            PendingReplace::New(dir) => self.do_new_project(dir),
            PendingReplace::Open(dir) => self.do_open_project(dir),
        }
    }

    /// `File`, and a label for whichever project is open.
    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New Project…").clicked() {
                    self.new_project();
                    ui.close();
                }
                if ui.button("Open Project…").clicked() {
                    self.open_project();
                    ui.close();
                }
                if ui
                    .add_enabled(
                        self.project.is_some() && self.project_modified(),
                        egui::Button::new("Save Project"),
                    )
                    .clicked()
                {
                    self.save_project();
                    ui.close();
                }
                if ui
                    .add_enabled(self.project.is_some(), egui::Button::new("Save Project As…"))
                    .clicked()
                {
                    self.save_project_as();
                    ui.close();
                }
            });
            match &self.project {
                Some(p) => {
                    ui.separator();
                    ui.label(p.dir.display().to_string());
                    if self.project_modified() {
                        ui.strong("*");
                    }
                }
                None => {
                    ui.weak("no project — files open free-standing");
                }
            }
            if let Some(e) = &self.last_error {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
        });
        self.replace_confirm_modal(ui);
    }

    /// New/Open Project asked for while something was unsaved; drawn as its own small window rather
    /// than acting immediately.
    fn replace_confirm_modal(&mut self, ui: &mut egui::Ui) {
        let Some(action) = self.pending_replace.clone() else { return };
        enum Choice {
            SaveThenContinue,
            Discard,
            Cancel,
        }
        let mut choice = None;
        egui::Window::new("Unsaved changes").collapsible(false).resizable(false).show(
            ui.ctx(),
            |ui| {
                ui.label(
                    "The current project has unsaved changes — settings edited but not saved, \
                     or the project document itself has moved on.",
                );
                ui.horizontal(|ui| {
                    if ui.button("Save then continue").clicked() {
                        choice = Some(Choice::SaveThenContinue);
                    }
                    if ui.button("Discard").clicked() {
                        choice = Some(Choice::Discard);
                    }
                    if ui.button("Cancel").clicked() {
                        choice = Some(Choice::Cancel);
                    }
                });
            },
        );
        match choice {
            Some(Choice::SaveThenContinue) => {
                if self.project.is_some() {
                    self.save_project();
                }
                self.pending_replace = None;
                self.apply_pending(action);
            }
            Some(Choice::Discard) => {
                self.pending_replace = None;
                self.apply_pending(action);
            }
            Some(Choice::Cancel) => self.pending_replace = None,
            None => {}
        }
    }
}

/// The one audio file dialog, shared by the tab strip and the empty window.
fn pick_file() -> Option<PathBuf> {
    rfd::FileDialog::new().add_filter("audio", &["wav", "aiff", "aif", "flac"]).pick_file()
}

/// Where to create a new project: a save-style dialog rather than a folder picker, so New Project
/// names a folder that does not exist yet instead of pointing at one that might already hold
/// something. `do_new_project` still refuses if the name collides with an existing project.
fn pick_new_project_dir() -> Option<PathBuf> {
    rfd::FileDialog::new().set_file_name("project").save_file()
}

/// Copy `src` into `dir`, keeping its file name unless that collides with something already there
/// — appending `-2`, `-3`, … until it does not, the same idea `free_number` uses for tab numbers.
/// What Import copies in, for a soundfile (the tab strip's `import_audio`) and a settings document
/// (`Session::settings`'s own `Import…`) alike — one definition of "bring a file into the project."
///
/// Re-importing a file already inside the project (its name already resolves to itself) is not a
/// collision to rename around: that path is returned as-is, both because there is nothing to copy
/// and because copying a file onto itself is not something `std::fs::copy` promises to do safely.
fn copy_into_project(src: &Path, dir: &Path) -> Result<PathBuf, String> {
    let name = src.file_name().ok_or_else(|| format!("{} has no file name", src.display()))?;
    let stem = src.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let ext = src.extension().map(|e| e.to_string_lossy().into_owned());

    let mut dest = dir.join(name);
    let mut n = 2;
    while dest.exists() {
        if same_file(src, &dest) {
            return Ok(dest);
        }
        dest = dir.join(match &ext {
            Some(e) => format!("{stem}-{n}.{e}"),
            None => format!("{stem}-{n}"),
        });
        n += 1;
    }

    std::fs::copy(src, &dest)
        .map_err(|e| format!("copying {} to {}: {e}", src.display(), dest.display()))?;
    Ok(dest)
}

/// Whether two paths name the same file on disk, resolving symlinks and relative components —
/// what `copy_into_project` needs to tell "already imported" apart from "a different file that
/// happens to share a name."
fn same_file(a: &Path, b: &Path) -> bool {
    matches!((a.canonicalize(), b.canonicalize()), (Ok(a), Ok(b)) if a == b)
}

/// Where to write a render. Unlike a settings document, which belongs beside the others, a render
/// is an output with no file of its own to start from — so once a project is active, its directory
/// is the starting guess; otherwise the file dialog's own last-used place is as good a one.
fn pick_audio_save(name: &str, project_dir: Option<&Path>) -> Option<PathBuf> {
    let mut d = rfd::FileDialog::new().add_filter("soundfile", &["wav", "aiff", "flac"]).set_file_name(name);
    if let Some(dir) = project_dir {
        d = d.set_directory(dir);
    }
    d.save_file()
}

/// Where to write a book: wherever the last one went, or the project directory on a tab's first
/// save, or the dialog's own last-used place with no project active.
fn pick_book_save(current: Option<&Path>, project_dir: Option<&Path>, name: &str) -> Option<PathBuf> {
    let mut d = rfd::FileDialog::new()
        .add_filter("book", &["gz", "json", "toml"])
        .set_file_name(name);
    if let Some(dir) = current.and_then(Path::parent).or(project_dir) {
        d = d.set_directory(dir);
    }
    d.save_file()
}

fn pick_settings() -> Option<PathBuf> {
    rfd::FileDialog::new().add_filter("settings", &["toml"]).pick_file()
}

/// Save-as, proposing `name` in whatever directory the document was last saved to, the project
/// directory on a first save, or the dialog's own last-used place with no project active.
///
/// The name and the directory come from different places on purpose: the name identifies the *tab*
/// (see `Session::default_settings_name`), while the directory is wherever this person — or this
/// project — keeps its settings.
fn pick_settings_save(current: Option<&Path>, project_dir: Option<&Path>, name: &str) -> Option<PathBuf> {
    let mut d = rfd::FileDialog::new().add_filter("settings", &["toml"]).set_file_name(name);
    if let Some(dir) = current.and_then(Path::parent).or(project_dir) {
        d = d.set_directory(dir);
    }
    d.save_file()
}

/// The ticked sources, as a name for the log and for the transport.
fn describe_sources(s: Sources) -> String {
    let on: Vec<&str> = Which::ALL.iter().filter(|w| w.get(&s)).map(|w| w.label()).collect();
    if on.is_empty() { "nothing".into() } else { on.join(" + ") }
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
        Progress::Note(line) => line.clone(),
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

    #[test]
    fn save_as_proposes_the_audio_file_and_the_tab_number() {
        let app = with(&["/a/b/chopin-nocturne-2.wav", "/a/zyklus.wav"]);
        assert_eq!(app.sessions[0].default_settings_name(), "chopin-nocturne-2-01.toml");
        assert_eq!(app.sessions[1].default_settings_name(), "zyklus-02.toml");
    }

    /// The case the number exists for: two tabs on one soundfile, comparing one setting. Without
    /// it both would propose the same name and the second would offer to overwrite the first.
    #[test]
    fn two_tabs_on_one_file_propose_different_names() {
        let mut app = with(&["/a/piano.wav"]);
        app.push(app.sessions[0].duplicate(app.free_number()));

        let names: Vec<String> =
            app.sessions.iter().map(|s| s.default_settings_name()).collect();
        assert_eq!(names, ["piano-01.toml", "piano-02.toml"]);
    }

    /// The number is padded like the tab title, and widens past 99 the same way rather than
    /// truncating into a collision.
    #[test]
    fn the_number_is_padded_and_matches_the_tab_title() {
        let mut app = with(&["/a/piano.wav"]);
        let s = &mut app.sessions[0];
        assert!(s.title().starts_with("01 ") && s.default_settings_name().contains("-01."));

        s.number = 7;
        assert_eq!(s.default_settings_name(), "piano-07.toml");
        s.number = 128;
        assert_eq!(s.default_settings_name(), "piano-128.toml");
    }

    /// Only the final extension goes, so a name with dots of its own survives; a file without one
    /// keeps its whole name.
    #[test]
    fn only_the_extension_is_dropped_from_the_name() {
        let mut app = with(&["/a/take.2.mix.wav", "/a/rawpcm"]);
        assert_eq!(app.sessions[0].default_settings_name(), "take.2.mix-01.toml");
        assert_eq!(app.sessions[1].default_settings_name(), "rawpcm-02.toml");

        // Nothing to derive from at all still produces a usable name rather than ".toml".
        app.sessions[0].input = PathBuf::from("/");
        assert_eq!(app.sessions[0].default_settings_name(), "settings-01.toml");
    }

    /// The book's name says what is in it. `-residual` is not decoration: the residual book is
    /// roughly 12x the atom list on a short excerpt, so two files of the same stem differ by an
    /// order of magnitude in size and in what they can be rendered into.
    #[test]
    fn the_book_name_says_whether_a_residual_was_measured() {
        let mut app = with(&["/a/b/chopin-nocturne-2.wav"]);
        let s = &mut app.sessions[0];

        assert_eq!(s.default_book_name(), "chopin-nocturne-2-01-book.json.gz");
        s.run_residual_analysis = true;
        assert_eq!(s.default_book_name(), "chopin-nocturne-2-01-book-residual.json.gz");
    }

    /// Gzipped, and JSON *beneath* the gzip — `book::write` reads the format from the extension
    /// under the suffix, so the proposed name has to carry both.
    #[test]
    fn the_proposed_book_is_gzipped_json() {
        let app = with(&["/a/piano.wav"]);
        let name = app.sessions[0].default_book_name();
        assert!(name.ends_with(".json.gz"), "got {name}");
    }

    /// Two tabs on one soundfile must not write one book, and a duplicate must not inherit the
    /// original's path — which would make its first Analyse silently overwrite the original's book.
    #[test]
    fn two_tabs_never_propose_or_inherit_one_book_file() {
        let mut app = with(&["/a/piano.wav"]);
        app.sessions[0].book_path = Some(PathBuf::from("/out/piano-01-book.json.gz"));

        app.push(app.sessions[0].duplicate(app.free_number()));
        assert!(app.sessions[1].book_path.is_none(), "a copy has written no book of its own");
        assert_ne!(
            app.sessions[0].default_book_name(),
            app.sessions[1].default_book_name(),
            "two tabs propose one book"
        );
    }

    /// Renders and settings share the `<file>-<tab number>` stem, so everything a tab produces
    /// sorts together in a directory. A render adds what it holds; see
    /// `each_render_mode_proposes_its_own_file_name`.
    #[test]
    fn a_render_and_a_settings_file_share_the_tabs_stem() {
        let app = with(&["/a/b/chopin-nocturne-2.wav", "/a/zyklus.wav"]);
        assert_eq!(app.sessions[0].default_settings_name(), "chopin-nocturne-2-01.toml");
        assert!(app.sessions[0].default_render_name().starts_with("chopin-nocturne-2-01-"));
        assert!(app.sessions[1].default_render_name().starts_with("zyklus-02-"));
    }

    /// A tab is "busy" for the repaint and the strip spinner whether it is analysing or rendering.
    /// Missing the render half would leave a finished one unreported until the mouse moved.
    #[test]
    fn a_tab_is_busy_while_either_kind_of_work_is_in_flight() {
        let mut app = with(&["a.wav"]);
        let s = &mut app.sessions[0];
        assert!(!s.busy());

        s.synthesising = Some(task::spawn_synthesis(task::SynthJob {
            book: rmp_core::book::Book::new(1.0, 48_000.0),
            residual_book: None,
            parts: task::RenderParts::default(),
            output: std::env::temp_dir().join("rmp-gui-busy-test.wav"),
        }));
        assert!(s.busy(), "a render in flight counts");

        // Let it finish and be drained, as `pump` does each frame.
        for _ in 0..200 {
            s.pump();
            if s.synthesising.is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(!s.busy(), "and stops counting once it is done");
        assert_eq!(s.log.len(), 1, "exactly one line, whatever the outcome: {:?}", s.log);
        std::fs::remove_file(std::env::temp_dir().join("rmp-gui-busy-test.wav")).ok();
    }

    /// A duplicate carries the *switches*, which are preferences, but nothing a run produced —
    /// and crucially not `available`, which is a fact about a decomposition this copy has not made.
    #[test]
    fn a_duplicate_keeps_the_switches_but_not_what_a_run_produced() {
        let mut app = with(&["a.wav"]);
        let s = &mut app.sessions[0];
        s.parts = task::RenderParts { atoms: false, residual: true };
        s.sources = Sources { atoms: true, ..Sources::default() };
        s.keep_residual = false;
        s.available = Available { atoms: true, ..Available::default() };

        let copy = s.duplicate(2);
        assert_eq!(copy.parts, task::RenderParts { atoms: false, residual: true });
        assert_eq!(copy.sources, Sources { atoms: true, ..Sources::default() });
        assert!(!copy.keep_residual);
        assert_eq!(copy.available, Available::default(), "a copy has run nothing");
        assert!(copy.results.is_none());
    }

    /// The combinations of atoms and residual must not overwrite each other's files: hearing them
    /// together is the point of being able to choose.
    #[test]
    fn each_combination_of_parts_proposes_its_own_file_name() {
        let mut app = with(&["/a/piano.wav"]);
        let mut names = Vec::new();
        for (atoms, residual) in [(true, false), (false, true), (true, true)] {
            app.sessions[0].parts = task::RenderParts { atoms, residual };
            names.push(app.sessions[0].default_render_name());
        }
        assert_eq!(
            names,
            ["piano-01-atoms.wav", "piano-01-residual-synth.wav", "piano-01-mixed.wav"]
        );
    }

    /// What Synthesize will and will not offer, judged against what the run produced.
    ///
    /// The residual-alone case is the one this is really about. `pipeline::analyse` hands the
    /// residual book back *beside* the atom book, so `Book::residual` is empty on a fresh run
    /// however it was configured — checking the book directly refused every residual render,
    /// including the mix, and the panel's residual tickbox did nothing at all.
    #[test]
    fn the_residual_can_be_rendered_alone_once_a_run_has_measured_one() {
        use task::RenderParts as P;

        let measured = Available {
            origin: true,
            atoms: true,
            residual_measured: true,
            residual_synthesised: true,
        };
        assert!(P { atoms: false, residual: true }.available(measured).is_ok(), "residual alone");
        assert!(P { atoms: true, residual: true }.available(measured).is_ok(), "and mixed");
        assert!(P { atoms: true, residual: false }.available(measured).is_ok(), "and atoms alone");
    }

    /// And refuses, with a reason, whatever the run did not make.
    #[test]
    fn synthesis_refuses_what_the_run_did_not_produce_and_says_why() {
        use task::RenderParts as P;

        let atoms_only = Available { origin: true, atoms: true, ..Available::default() };
        let nothing = Available { origin: true, ..Available::default() };

        for (parts, av, what) in [
            (P { atoms: false, residual: false }, atoms_only, "nothing ticked"),
            (P { atoms: false, residual: true }, atoms_only, "no residual was measured"),
            (P { atoms: true, residual: true }, atoms_only, "the mix needs one too"),
            (P { atoms: true, residual: false }, nothing, "no atoms were selected"),
        ] {
            let refusal = parts.available(av).expect_err(what);
            assert!(!refusal.is_empty(), "{parts:?} refuses silently ({what})");
        }
    }

    /// Play offers only what the *finished* run produced. Ticking "residual analysis" after the
    /// fact must not make a synthesised residual appear that no run ever measured.
    #[test]
    fn play_offers_what_the_run_made_not_what_is_ticked_now() {
        let mut app = with(&["a.wav"]);
        let s = &mut app.sessions[0];

        // As `pump` records it: a run that kept nothing.
        s.available = Available { origin: true, ..Available::default() };
        assert!(s.available.has(Which::Origin));
        assert!(!s.available.has(Which::ResidualMeasured));
        assert!(!s.available.has(Which::ResidualSynthesised));

        // Flipping the Analyse switches now changes nothing about that run.
        s.keep_residual = true;
        s.run_residual_analysis = true;
        assert!(!s.available.has(Which::ResidualMeasured), "the run did not keep it");
        assert!(!s.available.has(Which::ResidualSynthesised), "the run did not measure it");
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
        assert!(copy.results.is_none());
        assert!(copy.running.is_none());
        assert!(copy.ran_with.is_none(), "a copy has not run, so nothing of its own is stale");
        assert!(!copy.results_are_stale());
    }

    fn tmp_project_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("rmp-gui-project-test-{}-{name}", std::process::id()));
        p
    }

    /// A round trip through `to_tab_doc`/`Session::restore`: settings, switches and the view
    /// selection all have to come back, with no results — a book was never on record for this tab.
    #[test]
    fn restoring_a_tab_brings_back_its_settings_and_switches() {
        let dir = tmp_project_dir("restore-basic");
        std::fs::create_dir_all(&dir).unwrap();

        let mut app = with(&["/a/piano.wav"]);
        let s = &mut app.sessions[0];
        s.start = "2.5".into();
        s.duration = "0.5".into();
        set(s, "max_atoms", "4321");
        s.keep_residual = false;
        s.run_residual_analysis = true;
        s.parts = task::RenderParts { atoms: false, residual: true };
        s.sources = Sources { atoms: true, ..Sources::default() };
        s.view = View::Distribution;
        // `Save Project` would give the settings a file via `ensure_settings_saved`; done directly
        // here, since a document with no file has nothing for `to_tab_doc` to point a restore at.
        s.settings.save_as(&dir.join("piano-01.toml")).unwrap();

        let doc = s.to_tab_doc(&dir);
        let restored = Session::restore(&dir, &doc);

        assert_eq!(restored.number, s.number);
        assert_eq!(restored.input, s.input);
        assert_eq!(restored.start, "2.5");
        assert_eq!(restored.duration, "0.5");
        assert_eq!(restored.settings.status().unwrap().pursuit.max_atoms, 4321);
        assert!(!restored.keep_residual);
        assert!(restored.run_residual_analysis);
        assert_eq!(restored.parts, task::RenderParts { atoms: false, residual: true });
        assert_eq!(restored.sources, Sources { atoms: true, ..Sources::default() });
        assert_eq!(restored.view, View::Distribution);
        assert!(restored.results.is_none(), "no book was ever on record for this tab");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A tab still opens when its book can no longer be read — a moved or deleted file — with the
    /// failure logged rather than the whole project refusing to restore.
    #[test]
    fn a_tab_whose_book_cannot_be_read_still_opens_with_the_failure_logged() {
        let dir = tmp_project_dir("restore-missing-book");
        let doc = TabDoc {
            number: 3,
            input: PathBuf::from("/a/piano.wav"),
            start: String::new(),
            duration: String::new(),
            settings: None,
            keep_residual: true,
            run_residual_analysis: false,
            parts: task::RenderParts::default(),
            sources: Sources::default(),
            book: Some(PathBuf::from("missing-book.json.gz")),
            view: View::Summary,
        };

        let restored = Session::restore(&dir, &doc);
        assert_eq!(restored.number, 3);
        assert!(restored.results.is_none());
        assert!(
            restored.log.iter().any(|l| l.contains("could not reload results")),
            "no explanation logged: {:?}",
            restored.log
        );
    }

    /// The point of the whole feature: a tab whose book is on record reloads its results from disk
    /// on restore, with no pursuit re-run — `Available` and the mixable sources have to agree with
    /// what a live run of the same book would report, minus the measured residual a file never
    /// carries.
    ///
    /// Against a real soundfile, since the excerpt has to actually re-read the same samples the
    /// book was analysed from; skipped where `data/` is not checked out, as `task`'s own fixture
    /// test is.
    #[test]
    fn a_restored_tab_reloads_its_results_from_the_book_without_rerunning() {
        let input = std::path::Path::new("../data/audio/chopin-nocturne-2.wav");
        if !input.is_file() {
            eprintln!("no {} here; skipping", input.display());
            return;
        }
        // Absolute, like every real tab's `input` is — a relative path here would resolve against
        // the *project* directory, same as a stored settings or book path would.
        let input = input.canonicalize().expect("resolving the fixture path");
        let input = input.as_path();
        let dir = tmp_project_dir("restore-results");
        std::fs::create_dir_all(&dir).unwrap();
        let book_path = dir.join("piano-01-book.json.gz");

        let mut cfg = rmp_core::config::Config::default();
        cfg.dictionary.fof.alphas = vec![256.0];
        cfg.dictionary.fof.betas_ms = vec![1.0];
        cfg.blocks.f_min = 200.0;
        cfg.blocks.f_max = 2000.0;
        cfg.pursuit.max_atoms = 20;
        cfg.refine.enabled = false;

        let read = rmp_core::audio::read(input).expect("reading the fixture");
        let (offset, signal) = rmp_core::pipeline::excerpt(read.signal, Some(2.0), Some(0.3))
            .expect("cutting the excerpt");
        let mut planner = rmp_core::fft::Planner::new();
        let analysis = rmp_core::pipeline::analyse(
            rmp_core::pipeline::AnalysisRequest {
                signal: &signal,
                offset,
                config: &cfg,
                residual: None,
            },
            &mut planner,
            &mut (),
        )
        .expect("the fixture decomposes");
        rmp_core::book::write(&book_path, &analysis.book).expect("writing the book");

        let doc = TabDoc {
            number: 1,
            input: input.to_path_buf(),
            start: "2.0".into(),
            duration: "0.3".into(),
            settings: None,
            keep_residual: true,
            run_residual_analysis: false,
            parts: task::RenderParts::default(),
            sources: Sources::default(),
            book: Some(project::store_path(&dir, &book_path)),
            view: View::Summary,
        };

        let restored = Session::restore(&dir, &doc);
        assert!(restored.log.is_empty(), "nothing should have failed: {:?}", restored.log);
        let results = restored.results.as_ref().expect("the book reads back");
        assert_eq!(results.book().len(), analysis.book.len());
        assert!(!results.book().is_empty());
        assert_eq!(results.residual(), &[] as &[f32], "no measured residual in a loaded book");
        assert!(restored.available.atoms);
        assert!(!restored.available.residual_measured);
        assert!(!restored.results_are_stale(), "ran_with was set from the settings on restore");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `Save Project` is what clears the project-modified flag, and it gives every tab's settings a
    /// file along the way — a restored project can only read a settings document that exists.
    #[test]
    fn saving_a_project_clears_its_modified_flag_and_saves_unsaved_settings() {
        let dir = tmp_project_dir("dirty-flag");
        let mut app = RmpApp::default();

        app.do_new_project(dir.clone());
        assert!(app.project.is_some());
        assert!(!app.project_modified(), "freshly created, nothing to save yet");

        app.open(PathBuf::from("/a/piano.wav"));
        assert!(app.project_modified(), "a tab was added since the last save");

        app.save_project();
        assert!(!app.project_modified(), "save_project caught up");
        assert!(
            app.sessions[0].settings.path().is_some(),
            "save_project must give an unsaved tab's settings a file"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// "New" is for a project that does not exist yet — a name that collides with one that does
    /// must be reported, not silently overwritten, or picking a name close to an existing project
    /// could quietly discard it.
    #[test]
    fn new_project_refuses_a_directory_that_already_holds_one() {
        let dir = tmp_project_dir("new-refuses-existing");
        let mut first = RmpApp::default();
        first.do_new_project(dir.clone());
        first.open(PathBuf::from("/a/piano.wav"));
        first.save_project();
        assert!(dir.join(project::PROJECT_FILE).is_file());

        let mut second = RmpApp::default();
        second.do_new_project(dir.clone());
        assert!(second.project.is_none(), "must not adopt the existing directory as a fresh project");
        assert!(second.last_error.is_some(), "and must say why");

        // The original project is untouched.
        let doc: ProjectDoc = rmp_core::book::read_doc(&dir.join(project::PROJECT_FILE)).unwrap();
        assert_eq!(doc.tabs.len(), 1, "the first project's tab must survive the refused New");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn copy_into_project_places_the_file_under_its_own_name() {
        let dir = tmp_project_dir("import-basic");
        std::fs::create_dir_all(&dir).unwrap();
        let src_dir = tmp_project_dir("import-basic-src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src = src_dir.join("piano.wav");
        std::fs::write(&src, b"not really audio").unwrap();

        let dest = copy_into_project(&src, &dir).expect("copying must succeed");
        assert_eq!(dest, dir.join("piano.wav"));
        assert_eq!(std::fs::read(&dest).unwrap(), b"not really audio");

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&src_dir).ok();
    }

    /// Two different files that happen to share a name must not collide: the second import gets a
    /// new name rather than overwriting the first.
    #[test]
    fn copy_into_project_renames_around_a_different_files_name() {
        let dir = tmp_project_dir("import-collision");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("piano.wav"), b"already here").unwrap();

        let src_dir = tmp_project_dir("import-collision-src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src = src_dir.join("piano.wav");
        std::fs::write(&src, b"a different file").unwrap();

        let dest = copy_into_project(&src, &dir).expect("copying must succeed");
        assert_eq!(dest, dir.join("piano-2.wav"), "must not overwrite the unrelated file already there");
        assert_eq!(std::fs::read(&dest).unwrap(), b"a different file");
        assert_eq!(std::fs::read(dir.join("piano.wav")).unwrap(), b"already here", "untouched");

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&src_dir).ok();
    }

    /// Re-importing a file already inside the project is a no-op, not a self-copy — `std::fs::copy`
    /// makes no promises about a source and destination that are the same file.
    #[test]
    fn copy_into_project_is_a_no_op_when_the_file_is_already_there() {
        let dir = tmp_project_dir("import-reimport");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("piano.wav");
        std::fs::write(&src, b"already inside the project").unwrap();

        let dest = copy_into_project(&src, &dir).expect("must not error on a self-copy");
        assert_eq!(dest, src);
        assert_eq!(std::fs::read(&dest).unwrap(), b"already inside the project", "left untouched");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Import does nothing with no project active — there is nowhere to copy into — rather than
    /// silently falling back to a plain Open.
    #[test]
    fn import_does_nothing_without_an_open_project() {
        let mut app = RmpApp::default();
        app.import_audio(PathBuf::from("/a/piano.wav"));
        assert!(app.sessions.is_empty());
    }

    /// The point of the feature: an imported file ends up inside the project, and the tab it opens
    /// points at the copy, not at the original.
    #[test]
    fn importing_a_file_opens_a_tab_pointing_at_the_copy_inside_the_project() {
        let dir = tmp_project_dir("import-e2e");
        let src_dir = tmp_project_dir("import-e2e-src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src = src_dir.join("piano.wav");
        std::fs::write(&src, b"not really audio").unwrap();

        let mut app = RmpApp::default();
        app.do_new_project(dir.clone());
        app.import_audio(src.clone());

        assert_eq!(app.sessions.len(), 1);
        assert_eq!(app.sessions[0].input, dir.join("piano.wav"));
        assert_ne!(app.sessions[0].input, src, "the tab must point at the copy, not the original");

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&src_dir).ok();
    }

    /// `rmp-gui PROJECT` — a saved project opens straight into its tabs, no dialog involved.
    #[test]
    fn open_project_at_restores_a_saved_project() {
        let dir = tmp_project_dir("cli-open-existing");
        let mut first = RmpApp::default();
        first.do_new_project(dir.clone());
        first.open(PathBuf::from("/a/piano.wav"));
        first.save_project();

        let mut second = RmpApp::default();
        second.open_project_at(dir.clone());
        assert!(second.last_error.is_none());
        assert_eq!(second.sessions.len(), 1);
        assert!(second.project.is_some());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A path that is not a project — missing outright, or a directory with no `project.toml` —
    /// is reported through `last_error` rather than panicking or being silently ignored.
    #[test]
    fn open_project_at_reports_a_path_that_is_not_a_project() {
        let mut app = RmpApp::default();
        app.open_project_at(tmp_project_dir("cli-open-missing"));
        assert!(app.last_error.is_some());
        assert!(app.sessions.is_empty());
        assert!(app.project.is_none());
    }

    /// The settings panel's Import: the document ends up inside the project and loaded, not merely
    /// pointed at wherever it started.
    #[test]
    fn importing_a_settings_document_copies_it_into_the_project_and_loads_it() {
        let dir = tmp_project_dir("import-settings");
        std::fs::create_dir_all(&dir).unwrap();
        let src_dir = tmp_project_dir("import-settings-src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src = src_dir.join("piano.toml");
        let mut text = rmp_core::config::Config::default().to_toml();
        text = text.replace("max_atoms = 1000", "max_atoms = 4321");
        std::fs::write(&src, &text).unwrap();

        let mut app = with(&["/a/piano.wav"]);
        let s = &mut app.sessions[0];
        s.import_settings(&src, &dir).expect("importing must succeed");

        assert_eq!(s.settings.path(), Some(dir.join("piano.toml")).as_deref());
        assert_eq!(s.settings.status().unwrap().pursuit.max_atoms, 4321);
        assert!(!s.settings.modified(), "a freshly loaded document is unmodified");

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&src_dir).ok();
    }

    /// A collision with a different, unrelated settings file already in the project must not
    /// silently overwrite it — the same rule `copy_into_project` already enforces for audio.
    #[test]
    fn importing_a_settings_document_renames_around_a_collision() {
        let dir = tmp_project_dir("import-settings-collision");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("piano.toml"), "already here").unwrap();

        let src_dir = tmp_project_dir("import-settings-collision-src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src = src_dir.join("piano.toml");
        std::fs::write(&src, rmp_core::config::Config::default().to_toml()).unwrap();

        let mut app = with(&["/a/piano.wav"]);
        let s = &mut app.sessions[0];
        s.import_settings(&src, &dir).expect("importing must succeed");

        assert_eq!(s.settings.path(), Some(dir.join("piano-2.toml")).as_deref());
        assert_eq!(std::fs::read_to_string(dir.join("piano.toml")).unwrap(), "already here");

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&src_dir).ok();
    }
}
