//! What the decomposition came to, and what it cost.
//!
//! The figures are `stats::summarize`'s — the same `BookSummary` behind `rmpstat summary` — and the
//! timings are `Analysis::timing`, which the worker has always carried back and nothing has read
//! until now.
//!
//! **The four phases are reported separately rather than as one total.** The dictionary build is
//! the part a low-alpha config blows up — CLAUDE.md measures 64 ms against 2.4 ms across two
//! configs of the same material — and folding it into the analysis time is exactly what hid that.

use crate::results::Results;
use rmp_core::pipeline::Timing;
use rmp_core::signal::{Signal, db_fs, peak_of, rms_of};
use rmp_core::stats::{self, BookSummary, Evaluator};
use std::time::Duration;

/// The summary, computed once per book.
#[derive(Default)]
pub struct SummaryView {
    cached: Option<Result<BookSummary, String>>,
}

impl SummaryView {
    /// Drop what was computed for a previous run.
    pub fn invalidate(&mut self) {
        self.cached = None;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, results: &Results) {
        let (book, origin) = (results.book(), results.signal());
        if self.cached.is_none() {
            let mut ev = Evaluator::new(book);
            self.cached = Some(stats::summarize(book, &mut ev).map_err(|e| e.to_string()));
        }

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            match self.cached.as_ref().expect("just computed") {
                Ok(s) => rows(ui, s, book.sample_rate),
                Err(e) => {
                    ui.colored_label(ui.visuals().error_fg_color, format!("summary: {e}"));
                }
            }

            // Where in the source file this book describes. Every onset in it is relative to the
            // excerpt, not to the file, so the origin is what puts a render back where it came from.
            row(
                ui,
                "excerpt",
                &format!(
                    "{} samples from {}  ({:.3} s at {:.3} s)",
                    origin.len(),
                    results.offset(),
                    origin.len() as f32 / origin.sample_rate.max(1.0),
                    results.offset() as f32 / origin.sample_rate.max(1.0),
                ),
            );

            ui.add_space(8.0);
            ui.heading("Residual");
            let residual = results.residual();
            if residual.is_empty() {
                ui.weak("not kept — tick `residual` in Analyse before running (or analyse again: a loaded book never carries one)");
            } else {
                // Relative first, because that is the figure you act on: an absolute dBFS residual
                // means nothing without knowing how loud the input was.
                let (r_rms, r_peak) = (rms_of(residual), peak_of(residual) as f64);
                let (s_rms, s_peak) = (origin.rms(), origin.peak() as f64);
                row(ui, "rms, relative to input", &format!("{:+.1} dB", db_fs(r_rms) - db_fs(s_rms)));
                row(
                    ui,
                    "peak, relative to input",
                    &format!("{:+.1} dB", db_fs(r_peak) - db_fs(s_peak)),
                );
                row(ui, "absolute", &format!("{:.1} dBFS rms, {:.1} dBFS peak", db_fs(r_rms), db_fs(r_peak)));
            }

            ui.add_space(8.0);
            ui.heading("Time");
            match results.timing() {
                Some(t) => timings(ui, t, results.cancelled(), origin),
                None => {
                    ui.weak(
                        "not available — this book was loaded from a saved project; analyse \
                         again to see timings",
                    );
                }
            }
        });
    }
}

fn rows(ui: &mut egui::Ui, s: &BookSummary, sr: f32) {
    ui.heading("Book");
    row(ui, "atoms", &s.atoms.to_string());
    row(ui, "sample rate", &format!("{} Hz", sr as u32));
    row(
        ui,
        "span",
        &format!(
            "{:.3} .. {:.3} s  ({:.3} s)",
            s.span.0 as f32 / sr,
            s.span.1 as f32 / sr,
            (s.span.1 - s.span.0) as f32 / sr
        ),
    );

    ui.add_space(8.0);
    ui.heading("Convergence");
    row(ui, "final SNR", &format!("{:.1} dB", s.snr_db));
    for &(target, at) in &s.atoms_to_reach {
        let v = at.map_or_else(|| "not reached".to_string(), |n| format!("{n} atoms"));
        row(ui, &format!("{target:.0} dB reached at"), &v);
    }
    row(ui, "refined off the grid", &format!("{:.0}%", 100.0 * s.refined_frac));

    ui.add_space(8.0);
    ui.heading("Energy");
    // Below 1 is ordinary and quantifies the overlap: greedy MP subtracts from a running residual,
    // so the parts sum to the whole only when no atom re-removed what an earlier one had taken.
    row(ui, "deposited / removed", &format!("{:.3}", s.deposited_frac));
    row(ui, "energy removed, median", &format!("{:.3e}", s.energy_removed.median));
    // This, not `hr_score`, is what an HRMP clamp looks like from the book.
    row(
        ui,
        "removed / projected",
        &format!(
            "median {:.3}, p5 {:.3}, min {:.3}",
            s.removed_over_projected.median, s.removed_over_projected.p5, s.removed_over_projected.min
        ),
    );
    row(ui, "atoms 1% short of projection", &s.shortfall_atoms.to_string());

    if s.hrmp_atoms > 0 {
        ui.add_space(8.0);
        ui.heading("HRMP");
        row(ui, "atoms with a verdict", &s.hrmp_atoms.to_string());
        // A consistency check on the book, not a clamp severity — the two fields are post-clamp
        // measures of the same energy.
        row(ui, "hr_score consistency", &format!("{:.1e}", s.hr_consistency));
    }
}

fn timings(ui: &mut egui::Ui, t: &Timing, cancelled: bool, origin: &Signal) {
    let duration = origin.len() as f32 / origin.sample_rate.max(1.0);

    row(ui, "dictionary", &dur(t.dictionary));
    row(ui, "init", &dur(t.init));
    row(ui, "pursuit", &dur(t.pursuit));
    if !t.residual.is_zero() {
        row(ui, "residual analysis", &dur(t.residual));
    }

    // The same figure `rmp` prints, through the same definition — see `Timing::realtime_factor`.
    let realtime = t.realtime_factor(duration);
    row(ui, "analysis", &format!("{} for {duration:.2} s — {realtime:.1}x realtime", dur(t.init + t.pursuit)));
    if cancelled {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            "interrupted — the book holds only what had been selected by then",
        );
    }
}

/// `{:.2?}` on a `Duration`, which is what the CLI prints: `5.07s`, `64.01ms`.
fn dur(d: Duration) -> String {
    format!("{d:.2?}")
}

fn row(ui: &mut egui::Ui, name: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.add_sized([210.0, 16.0], egui::Label::new(egui::RichText::new(name).weak()).halign(egui::Align::LEFT));
        ui.monospace(value);
    });
}
