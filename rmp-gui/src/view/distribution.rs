//! Histograms over a book, with every control `rmpstat hist` has.
//!
//! **Plotted separately, never overlaid.** The quantities carry different units and ranges — Hz
//! against dBFS against a bare ratio — so one shared axis would say nothing. Each selected quantity
//! gets its own plot, stacked.
//!
//! **A quantity the book cannot describe is disabled, not hidden**, with the reason on hover, which
//! is the rule Play's sources and Synthesize's parts already follow: `alpha`, `beta`, `alpha*beta`,
//! `fade_dur` and `rho` describe FOF atoms only, and `sigma` Gaussian atoms only. `Histogram`
//! measures that itself as `inapplicable`, so availability is asked of the data rather than
//! hard-coded per quantity here — a third atom kind would need no change.

use egui_plot::{Bar, BarChart, Plot};
use rmp_core::book::Book;
use rmp_core::stats::{self, Evaluator, Histogram, Quantity, Weight};
use std::collections::HashMap;

/// Quantities per row in the checkbox grid. Four fits the longest label
/// ("energy (dB re. initial)") at the panel widths this window is used at.
const COLUMNS: usize = 4;

/// How the bins are spaced. `Auto` is what the CLI means by neither `--log` nor `--linear`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Scale {
    #[default]
    Auto,
    Linear,
    Log,
}

impl Scale {
    fn as_option(self) -> Option<bool> {
        match self {
            Scale::Auto => None,
            Scale::Linear => Some(false),
            Scale::Log => Some(true),
        }
    }
}

/// Everything `rmpstat hist` exposes.
#[derive(Clone, PartialEq, Debug)]
pub struct Options {
    pub bins: usize,
    pub scale: Scale,
    /// `--range LO,HI`. Both or neither; a half-open range is not something the call accepts.
    pub range: Option<(f64, f64)>,
    pub weight: Weight,
    /// Estimate support from `alpha` instead of rendering. Only `support` and `periods` care.
    pub fast_support: bool,
}

impl Default for Options {
    fn default() -> Self {
        // The CLI's defaults, so a histogram here and one from `rmpstat hist` are the same picture.
        Self { bins: 24, scale: Scale::Auto, range: None, weight: Weight::Count, fast_support: false }
    }
}

pub struct DistributionView {
    /// Which quantities to plot. The CLI's default list.
    selected: Vec<Quantity>,
    options: Options,
    /// Text for the range fields, kept separate so a half-typed number does not become `0`.
    range_lo: String,
    range_hi: String,
    /// Computed histograms, and the options they were computed under. Recomputed only when one of
    /// those moves — `Evaluator::column` renders envelopes for `support` and `periods`, and doing
    /// that per frame would make this tab unusable on a refined book.
    cache: HashMap<Quantity, Result<Histogram, String>>,
    cached_under: Option<Options>,
}

impl Default for DistributionView {
    fn default() -> Self {
        Self {
            selected: vec![Quantity::Alpha, Quantity::Beta, Quantity::Freq, Quantity::AmpDb],
            options: Options::default(),
            range_lo: String::new(),
            range_hi: String::new(),
            cache: HashMap::new(),
            cached_under: None,
        }
    }
}

impl DistributionView {
    pub fn invalidate(&mut self) {
        self.cache.clear();
        self.cached_under = None;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, book: &Book) {
        if self.cached_under.as_ref() != Some(&self.options) {
            self.cache.clear();
            self.cached_under = Some(self.options.clone());
        }
        self.controls(ui, book);
        ui.separator();

        if self.selected.is_empty() {
            ui.centered_and_justified(|ui| ui.weak("Tick a quantity above."));
            return;
        }

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            for q in self.selected.clone() {
                self.ensure(book, q);
                match self.cache.get(&q) {
                    Some(Ok(h)) => plot(ui, h),
                    Some(Err(e)) => {
                        ui.colored_label(ui.visuals().error_fg_color, format!("{}: {e}", q.label()));
                    }
                    None => {}
                }
                ui.add_space(12.0);
            }
        });
    }

    /// Compute one histogram if it is not already cached under the current options.
    fn ensure(&mut self, book: &Book, q: Quantity) {
        if self.cache.contains_key(&q) {
            return;
        }
        let mut ev = Evaluator::new(book);
        ev.fast_support = self.options.fast_support;
        let h = stats::histogram(
            book,
            &mut ev,
            q,
            self.options.bins.max(1),
            self.options.scale.as_option(),
            self.options.range,
            self.options.weight,
        )
        .map_err(|e| e.to_string());
        self.cache.insert(q, h);
    }

    fn controls(&mut self, ui: &mut egui::Ui, book: &Book) {
        // A fixed grid rather than `horizontal_wrapped`: the labels carry their units, so fifteen
        // of them do not fit on one line, and wrapping a *row of widgets* by available width made
        // egui wrap the label text instead — one character per line, tall enough to push the plots
        // off the panel entirely.
        egui::Grid::new("quantities").num_columns(COLUMNS).spacing([12.0, 2.0]).show(ui, |ui| {
            for (n, q) in Quantity::ALL.into_iter().enumerate() {
                // Asked of the data: a quantity is offered when at least one atom has a value for
                // it. `histogram` reports the rest as `inapplicable`, so this needs no per-quantity
                // rule of its own.
                let can = self.applicable(book, q);
                let mut on = self.selected.contains(&q) && can;
                ui.add_enabled_ui(can, |ui| {
                    let r = ui
                        .checkbox(&mut on, q.label())
                        .on_disabled_hover_text("no atom in this book has one");
                    if r.changed() {
                        self.selected.retain(|&s| s != q);
                        if on {
                            self.selected.push(q);
                        }
                    }
                });
                if (n + 1) % COLUMNS == 0 {
                    ui.end_row();
                }
            }
            ui.end_row();
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("bins");
            ui.add(egui::DragValue::new(&mut self.options.bins).range(1..=512).speed(1));

            ui.separator();
            ui.label("bins scale");
            for (s, name) in [(Scale::Auto, "auto"), (Scale::Linear, "linear"), (Scale::Log, "log")] {
                ui.selectable_value(&mut self.options.scale, s, name);
            }

            ui.separator();
            ui.label("weight");
            // The two "differ sharply": MP spends many weak atoms on the residual's tail, so
            // counting atoms and counting energy say different things about the same book.
            ui.selectable_value(&mut self.options.weight, Weight::Count, "count");
            ui.selectable_value(&mut self.options.weight, Weight::Energy, "energy");

            ui.separator();
            ui.checkbox(&mut self.options.fast_support, "fast support")
                .on_hover_text("estimate support from alpha instead of rendering; support and periods only");
        });

        ui.horizontal(|ui| {
            ui.label("range");
            let lo = ui.add(egui::TextEdit::singleline(&mut self.range_lo).desired_width(70.0).hint_text("lo"));
            let hi = ui.add(egui::TextEdit::singleline(&mut self.range_hi).desired_width(70.0).hint_text("hi"));
            if lo.changed() || hi.changed() {
                // Both or neither: a half-typed bound must not silently become a real one.
                self.options.range = match (self.range_lo.trim().parse(), self.range_hi.trim().parse()) {
                    (Ok(a), Ok(b)) if b > a => Some((a, b)),
                    _ => None,
                };
            }
            if self.options.range.is_none() && !(self.range_lo.is_empty() && self.range_hi.is_empty()) {
                ui.weak("(ignored until both are numbers and hi > lo)");
            }
        });
    }

    /// Whether any atom in this book has a value for `q`.
    ///
    /// Answered by computing the histogram once and asking it, which is why the cache is consulted
    /// first — the alternative is a table of which quantity applies to which atom kind, kept here,
    /// drifting from `stats`.
    fn applicable(&mut self, book: &Book, q: Quantity) -> bool {
        self.ensure(book, q);
        matches!(self.cache.get(&q), Some(Ok(h)) if h.total > 0.0)
    }
}

fn plot(ui: &mut egui::Ui, h: &Histogram) {
    let n = h.counts.len();
    ui.label(format!(
        "{}{}   n={}",
        h.quantity.label(),
        if h.log { "  (log bins)" } else { "" },
        h.stats.n
    ));

    // Bars sit on bin *index*, with the axis labelled by edge value: a geometric axis drawn on a
    // linear scale would crush every low bin into the left margin.
    let bars: Vec<Bar> = (0..n)
        .map(|i| Bar::new(i as f64 + 0.5, h.counts[i]).width(1.0))
        .collect();
    let edges = h.edges.clone();
    Plot::new(egui::Id::new(("hist", h.quantity.label())))
        .height(160.0)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .show_grid(false)
        .x_axis_formatter(move |m, _| {
            let i = (m.value.round().max(0.0) as usize).min(edges.len().saturating_sub(1));
            format!("{:.4}", edges[i])
        })
        .show(ui, |p| p.bar_chart(BarChart::new("bars", bars)));

    let s = &h.stats;
    ui.weak(format!(
        "min {:.4}   p5 {:.4}   median {:.4}   p95 {:.4}   max {:.4}   mean {:.4}",
        s.min, s.p5, s.median, s.p95, s.max, s.mean
    ));

    // Mass the bars do not show. Reported rather than dropped: `counts` never lies about the range
    // it covers, so the remainder has to be said somewhere.
    let mut notes = Vec::new();
    for (what, v) in [
        ("below", h.below),
        ("above", h.above),
        ("no finite value", h.skipped),
        ("not applicable", h.inapplicable),
    ] {
        if v > 0.0 {
            notes.push(format!("{what} {v:.0}"));
        }
    }
    if !notes.is_empty() {
        ui.weak(notes.join("   "));
    }
}
