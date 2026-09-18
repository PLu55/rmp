//! Structural analysis of the tab's book: a stub over `rmp_structure::analyze_partials`.
//!
//! What it shows is the partial level only, because that is all `rmp-structure` computes so far:
//! the diagnostics counts, and every partial's simplified frequency trajectory over the atom cloud
//! it was extracted from. Stems will be drawn here once they exist, as colours on the same lines.
//!
//! It keeps the result tabs' rule: nothing here is arithmetic of its own. The trajectories are the
//! partial book's breakpoints and the cloud is the library's own observations, drawn as they are.
//! It runs with the default structure settings; editing them from the window is not built yet.
//!
//! The frequency axis is `log2(Hz)` with Hz on the ticks. egui_plot has no log axis, and a linear
//! one would squeeze every partial below 1 kHz into the bottom fifth of the plot.

use crate::task;
use egui_plot::{Line, Plot, PlotPoints, Points};
use rmp_core::book::Book;
use rmp_structure::PartialAnalysis;

#[derive(Default)]
pub struct StructureView {
    result: Option<Box<task::StructureDone>>,
    running: Option<task::Structuring>,
    error: Option<String>,
    show_atoms: bool,
}

/// A colour per partial that neighbours rarely share: the golden-angle hue walk.
fn colour(id: u32) -> egui::Color32 {
    let h = (id as f32 * 0.618_034).fract();
    egui::ecolor::Hsva::new(h, 0.75, 0.9, 1.0).into()
}

impl StructureView {
    pub fn invalidate(&mut self) {
        *self = Self { show_atoms: self.show_atoms, ..Self::default() };
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, book: &Book) {
        self.pump();
        if self.result.is_none() && self.running.is_none() && self.error.is_none() {
            self.running = Some(task::spawn_structure(book.clone()));
        }
        if let Some(e) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, e);
            return;
        }
        let Some(done) = &self.result else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("extracting partials…");
            });
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            return;
        };
        let a = &done.analysis;

        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.show_atoms, "atom cloud")
                .on_hover_text("every atom's energy centroid and frequency, behind the partials");
            ui.separator();
            ui.weak("default structure settings");
            ui.separator();
            ui.weak("stems: not implemented yet");
        });
        egui::CollapsingHeader::new(format!(
            "{} partials from {} atoms in {:.3} s",
            a.diagnostics.accepted_partials, a.diagnostics.input_atoms, done.seconds
        ))
        .id_salt("structure-diagnostics")
        .show(ui, |ui| diagnostics(ui, a));
        ui.separator();
        self.plot(ui, a);
    }

    fn plot(&self, ui: &mut egui::Ui, a: &PartialAnalysis) {
        let sr = a.book.metadata.sample_rate;
        let secs = |t: u64| t as f64 / sr;
        let max_sig = a.book.partials.iter().map(|p| p.significance).fold(0.0, f64::max);

        Plot::new(egui::Id::new("structure-partials"))
            .x_axis_label("time (s)")
            .y_axis_label("frequency (Hz)")
            .y_axis_formatter(|m, _| format!("{:.0}", m.value.exp2()))
            .label_formatter(|pos| {
                let (name, p) = match pos {
                    egui_plot::HoverPosition::NearDataPoint { plot_name, position, .. } => {
                        (*plot_name, position)
                    }
                    egui_plot::HoverPosition::Elsewhere { position } => ("", position),
                };
                let head = if name.is_empty() { String::new() } else { format!("{name}\n") };
                Some(format!("{head}{:.3} s   {:.1} Hz", p.x, p.y.exp2()))
            })
            .show(ui, |plot| {
                if self.show_atoms
                    && let Some(i) = &a.intermediates
                {
                    let pts: Vec<[f64; 2]> = i
                        .observations
                        .iter()
                        .map(|o| [secs(o.time_center_samples), o.frequency_hz.log2()])
                        .collect();
                    plot.points(
                        Points::new("atoms", PlotPoints::from(pts))
                            .radius(1.0)
                            .color(egui::Color32::from_gray(128).gamma_multiply(0.5))
                            .allow_hover(false),
                    );
                }
                for p in &a.book.partials {
                    let pts: Vec<[f64; 2]> = p
                        .frequency
                        .points
                        .iter()
                        .map(|q| [secs(q.time_samples), q.value.log2()])
                        .collect();
                    // Line width follows significance, so the partials that carry the sound stand
                    // out from the long quiet ones.
                    let w = if max_sig > 0.0 { (p.significance / max_sig).sqrt() } else { 0.0 };
                    let name = format!(
                        "partial {}: {:.1} Hz, persistence {:.2}, {} atoms",
                        p.id.0,
                        p.geometric_mean_frequency_hz,
                        p.persistence,
                        p.supporting_atoms.len()
                    );
                    plot.line(
                        Line::new(name, PlotPoints::from(pts))
                            .width(1.0 + 3.0 * w as f32)
                            .color(colour(p.id.0)),
                    );
                }
            });
    }

    fn pump(&mut self) {
        let Some(run) = &mut self.running else { return };
        for update in run.drain() {
            match update {
                task::StructureUpdate::Done(done) => self.result = Some(done),
                task::StructureUpdate::Failed(e) => self.error = Some(e),
            }
        }
        if run.finished() {
            self.running = None;
        }
    }
}

fn diagnostics(ui: &mut egui::Ui, a: &PartialAnalysis) {
    let d = &a.diagnostics;
    egui::Grid::new("structure-diagnostics-grid").striped(true).show(ui, |ui| {
        let mut row = |k: &str, v: String| {
            ui.label(k);
            ui.monospace(v);
            ui.end_row();
        };
        row("input atoms", d.input_atoms.to_string());
        row("observations", d.observations.to_string());
        row("skipped", d.skipped.total().to_string());
        row("outside the grid", d.outside_grid.to_string());
        row("grid", format!("{} frames x {} bins", d.frames, d.bins));
        row("peaks", d.peaks.to_string());
        row("candidate ridges", d.candidate_ridges.to_string());
        row("rejected short", d.rejected_short.to_string());
        row("rejected sparse", d.rejected_sparse.to_string());
        row("partials", d.accepted_partials.to_string());
        row("supporting atoms", d.supporting_atoms.to_string());
        row("unsupported atoms", d.unsupported_atoms.to_string());
    });
}
