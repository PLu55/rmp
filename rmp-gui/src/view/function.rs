//! Curves over a book: quantities as a function of atom index.
//!
//! Where Distributions asks *how the atoms are spread*, this asks *how the decomposition
//! progressed* — and the one that matters most is how the energy falls as atoms are spent, which is
//! the same trace `rmpstat snr` plots.
//!
//! Everything here is derived from fields the book already carries — `residual_energy`,
//! `energy_removed`, `projected_energy` — through `rmp_core`'s own dB conventions
//! (`signal::snr_db`), so the numbers are the CLI's numbers and not a second opinion about them.
//!
//! Two of the curves are **the same fact twice**, deliberately: residual energy falls, SNR rises,
//! and each is the other negated by construction. They are both here because the question is asked
//! both ways — CLAUDE.md, `MANUAL.md` and `atoms_to_reach` all speak in SNR, while "how much is
//! left" is the thing you watch. Reading them as independent evidence would be a mistake, which is
//! why the tooltip says so.

use egui_plot::{HLine, Line, Plot, PlotPoints};
use rmp_core::book::Book;
use rmp_core::signal::snr_db;

/// The dB targets the pursuit is usually asked for, drawn as guides so "where did it reach 30 dB"
/// is answerable off the picture.
const TARGETS: [f32; 4] = [10.0, 20.0, 30.0, 40.0];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Curve {
    /// `residual_energy` relative to the book's initial energy. Falls — the shape the question
    /// "how far has this got" is usually asked about.
    ResidualEnergy,
    /// The same trace negated, which is what `Book::snr_trace` and `rmpstat snr` report.
    Snr,
    /// What each atom took on its own. Decays, and how fast is the honest measure of how much a
    /// dictionary has left to say about the signal.
    RemovedPerAtom,
    /// What would be left if the atoms' own removals simply added up: `1 - sum(energy_removed) /
    /// initial`, in the same dB as the residual curve so the two can be read together.
    ///
    /// Plotted this way round on purpose. The running sum expressed as dB *of initial* rises from
    /// far below zero while the residual falls toward it — opposite directions, nothing to compare.
    /// Turned into an implied residual it falls alongside the real one, and the **gap between them
    /// is the overlap**: greedy MP subtracts from a running residue, so the parts sum to the whole
    /// only when no atom re-removed what an earlier one had taken. `BookSummary::deposited_frac`
    /// is the same quantity as a single number.
    SummedRemovals,
    /// `energy_removed / projected_energy`. What an HRMP clamp looks like from the book — and on a
    /// book where HRMP did not run, a shortfall here is a parameter-mapping error instead.
    RemovedOverProjected,
}

impl Curve {
    pub const ALL: [Curve; 5] = [
        Curve::ResidualEnergy,
        Curve::Snr,
        Curve::RemovedPerAtom,
        Curve::SummedRemovals,
        Curve::RemovedOverProjected,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Curve::ResidualEnergy => "residual energy (dB re. initial)",
            Curve::Snr => "SNR (dB)",
            Curve::RemovedPerAtom => "energy removed per atom (dB re. initial)",
            Curve::SummedRemovals => "residual implied by summed removals (dB re. initial)",
            Curve::RemovedOverProjected => "removed / projected",
        }
    }

    fn hover(self) -> &'static str {
        match self {
            Curve::ResidualEnergy => "how much of the signal is still unexplained",
            Curve::Snr => "the same trace negated — not independent evidence",
            Curve::RemovedPerAtom => "what each atom took on its own",
            Curve::SummedRemovals => {
                "the gap to the residual curve is the overlap — energy an atom took back from an \
                 earlier one"
            }
            Curve::RemovedOverProjected => {
                "what an HRMP clamp looks like; a shortfall without HRMP is a mapping error"
            }
        }
    }

    fn y_label(self) -> &'static str {
        match self {
            Curve::RemovedOverProjected => "ratio",
            _ => "dB",
        }
    }

    /// Whether the dB targets are worth drawing over this curve.
    fn has_targets(self) -> bool {
        matches!(self, Curve::ResidualEnergy | Curve::Snr)
    }

    /// The curve itself, one point per atom.
    ///
    /// `x` is the atom *count* — 1 for the first — so reading a value off the axis answers "after
    /// how many atoms", which is the question `atoms_to_reach` also answers.
    fn points(self, book: &Book) -> Vec<[f64; 2]> {
        let initial = book.initial_energy;
        let mut running = 0.0f64;
        book.selections
            .iter()
            .enumerate()
            .map(|(i, s)| {
                running += s.energy_removed;
                let y = match self {
                    // Through `snr_db` rather than an open-coded log: the dB convention is
                    // rmp-core's, and a second one here is how the window and the terminal would
                    // come to disagree about the same book.
                    Curve::Snr => snr_db(initial, s.residual_energy) as f64,
                    Curve::ResidualEnergy => -(snr_db(initial, s.residual_energy) as f64),
                    Curve::RemovedPerAtom => -(snr_db(initial, s.energy_removed) as f64),
                    // `1 - removed/initial` is what the parts claim is left. It sits at or above
                    // the real residual, never below, because the parts sum to at most the whole.
                    Curve::SummedRemovals => {
                        10.0 * (1.0 - running / initial.max(f64::MIN_POSITIVE)).log10()
                    }
                    Curve::RemovedOverProjected => {
                        s.energy_removed / s.projected_energy.max(f64::MIN_POSITIVE)
                    }
                };
                [i as f64 + 1.0, y]
            })
            // An atom that removed nothing gives -inf; dropping it keeps the axis finite rather
            // than collapsing the whole plot to one point.
            .filter(|p| p[1].is_finite())
            .collect()
    }
}

pub struct FunctionView {
    selected: Vec<Curve>,
    cache: std::collections::HashMap<Curve, Vec<[f64; 2]>>,
}

impl Default for FunctionView {
    fn default() -> Self {
        // The one that was asked for: how the energy falls.
        Self { selected: vec![Curve::ResidualEnergy], cache: Default::default() }
    }
}

impl FunctionView {
    pub fn invalidate(&mut self) {
        self.cache.clear();
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, book: &Book) {
        ui.horizontal_wrapped(|ui| {
            for c in Curve::ALL {
                let mut on = self.selected.contains(&c);
                if ui.checkbox(&mut on, c.label()).on_hover_text(c.hover()).changed() {
                    self.selected.retain(|&s| s != c);
                    if on {
                        self.selected.push(c);
                    }
                }
            }
        });
        ui.separator();

        if book.is_empty() {
            ui.centered_and_justified(|ui| ui.weak("The book has no atoms."));
            return;
        }
        if self.selected.is_empty() {
            ui.centered_and_justified(|ui| ui.weak("Tick a curve above."));
            return;
        }

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            for c in self.selected.clone() {
                let pts = self.cache.entry(c).or_insert_with(|| c.points(book)).clone();
                ui.label(c.label());
                // Zoom and drag left on, unlike the histograms: the interesting part of a
                // convergence curve is where it flattens, and that is a few hundred atoms wide in
                // a trace thousands long.
                Plot::new(egui::Id::new(("curve", c.label())))
                    .height(220.0)
                    .y_axis_label(c.y_label())
                    .x_axis_label("atoms")
                    .show(ui, |p| {
                        if c.has_targets() {
                            for t in TARGETS {
                                let y = if c == Curve::ResidualEnergy { -(t as f64) } else { t as f64 };
                                p.hline(HLine::new(format!("{t:.0} dB"), y).width(0.5));
                            }
                        }
                        p.line(Line::new(c.label(), PlotPoints::from(pts)));
                    });
                ui.add_space(12.0);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmp_core::config::Config;
    use rmp_core::fft::Planner;
    use rmp_core::pipeline::{self, AnalysisRequest};
    use rmp_core::signal::Signal;

    fn a_book() -> Book {
        let mut cfg = Config::default();
        cfg.dictionary.fof.alphas = vec![256.0];
        cfg.dictionary.fof.betas_ms = vec![1.0];
        cfg.blocks.f_min = 200.0;
        cfg.blocks.f_max = 2000.0;
        cfg.pursuit.max_atoms = 32;
        cfg.refine.enabled = false;

        let sig = Signal::new(rmp_core::residual::pseudo_noise(8_000), 48_000.0);
        let mut planner = Planner::new();
        pipeline::analyse(
            AnalysisRequest { signal: &sig, offset: 0, config: &cfg, residual: None },
            &mut planner,
            &mut (),
        )
        .expect("the fixture decomposes")
        .book
    }

    /// The SNR curve is `Book::snr_trace` — the same trace `rmpstat snr` plots — and not a second
    /// derivation of it that could drift.
    #[test]
    fn the_snr_curve_is_the_books_own_trace() {
        let book = a_book();
        let trace = book.snr_trace();
        let pts = Curve::Snr.points(&book);

        assert_eq!(pts.len(), trace.len(), "one point per atom");
        for (i, p) in pts.iter().enumerate() {
            assert_eq!(p[0], i as f64 + 1.0, "x is the atom count, first atom at 1");
            assert!((p[1] - trace[i] as f64).abs() < 1e-9, "y is not the book's own SNR");
        }
    }

    /// The requested curve falls as atoms are spent, and is the SNR curve negated — which is the
    /// whole reason both are offered rather than treated as two facts.
    #[test]
    fn residual_energy_falls_and_is_the_snr_curve_negated() {
        let book = a_book();
        let snr = Curve::Snr.points(&book);
        let res = Curve::ResidualEnergy.points(&book);

        assert_eq!(snr.len(), res.len());
        for (a, b) in snr.iter().zip(&res) {
            assert!((a[1] + b[1]).abs() < 1e-9, "not the negation");
        }
        assert!(res.first().unwrap()[1] > res.last().unwrap()[1], "energy must fall");
    }

    /// The summed removals never claim *more* progress than the residual actually shows — the
    /// parts sum to at most the whole, and the shortfall is the overlap. One-sided, which is what
    /// makes the gap between the two curves readable as a quantity rather than as noise.
    #[test]
    fn summed_removals_never_claim_more_progress_than_the_residual() {
        let book = a_book();
        let summed = Curve::SummedRemovals.points(&book);
        let residual = Curve::ResidualEnergy.points(&book);
        assert_eq!(summed.len(), residual.len());

        // Both are dB below the initial energy, so more progress is the *more negative* one.
        for (s, r) in summed.iter().zip(&residual) {
            assert!(
                s[1] >= r[1] - 1e-6,
                "summed removals claim {:.3} dB against the residual's {:.3} dB",
                s[1],
                r[1]
            );
        }
        // And both fall: they are two readings of the same progress.
        assert!(summed.first().unwrap()[1] > summed.last().unwrap()[1]);
    }

    /// A point that cannot be plotted is dropped rather than collapsing the axis: an atom that
    /// removed nothing is -inf dB.
    #[test]
    fn non_finite_points_are_dropped() {
        let mut book = a_book();
        book.selections[0].energy_removed = 0.0;
        let pts = Curve::RemovedPerAtom.points(&book);
        assert_eq!(pts.len(), book.len() - 1, "the -inf point survived");
        assert!(pts.iter().all(|p| p[1].is_finite()));
    }
}
