//! The pseudo-Wigner map, with every control `rmpstat wv` has.
//!
//! Every step is `rmp-core`'s and in the CLI's order: `MapGrid::covering` (or `linear`/`log_freq`
//! when the time window is narrowed), `tfmap::compute`, `TfMap::to_db`, `tfmap::heat`. The ramp and
//! the dB scaling are shared with `rmpstat` deliberately — a diagnostic that coloured differently
//! in a chart than in a window would be worth less than one that did not exist.
//!
//! Three things this has to get right:
//!
//! **One texture, never per-cell rectangles.** A useful grid is ~10^6 cells. It is uploaded once
//! and `ui.image` scales it, which is also why the grid size is a *setting* rather than the panel's
//! pixel size: deriving it from the panel would recompute the whole map on every drag of the window
//! edge.
//!
//! **Computed on a worker.** CLAUDE.md measures 0.13 s for 5000 atoms at 1200x800, and the size is
//! user-settable, so this grows without bound.
//!
//! **The caption comes from the same `TfMap` the pixels did**, so the two cannot end up describing
//! different maps — the deposited fraction is the number that says whether you are looking at the
//! whole decomposition or a corner of it.

use crate::task;
use rmp_core::book::Book;
use rmp_core::tfmap::{Reference, Weight};

/// Everything `rmpstat wv` exposes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Options {
    pub n_t: usize,
    pub n_f: usize,
    pub log_freq: bool,
    pub floor_db: f32,
    pub reference: Reference,
    pub weight: Weight,
    /// Time window in seconds, as `rmp -s` / `-d` mean it. `None` is the whole book.
    pub start: Option<f64>,
    pub duration: Option<f64>,
    pub overlay: bool,
}

impl Default for Options {
    fn default() -> Self {
        // The CLI's defaults, so the same book gives the same map in both tools.
        Self {
            n_t: 1200,
            n_f: 800,
            log_freq: false,
            floor_db: 60.0,
            reference: Reference::Max,
            weight: Weight::EnergyRemoved,
            start: None,
            duration: None,
            overlay: false,
        }
    }
}

#[derive(Default)]
pub struct TimeFreqView {
    options: Options,
    start: String,
    duration: String,
    /// The uploaded image, and the options it was built under.
    texture: Option<egui::TextureHandle>,
    caption: String,
    /// Atom `(t0 seconds, f Hz)` for the overlay, from the map that was drawn.
    dots: Vec<(f64, f32)>,
    /// The drawn map's axes: time bounds in seconds, and the frequency bin edges. Kept because the
    /// overlay has to land on the *same* axes the pixels did, and the frequency one may be
    /// geometric.
    axes: Option<(f64, f64, Vec<f32>)>,
    built_under: Option<Options>,
    running: Option<task::Mapping>,
    error: Option<String>,
}

impl TimeFreqView {
    pub fn invalidate(&mut self) {
        self.texture = None;
        self.built_under = None;
        self.running = None;
        self.error = None;
        self.dots.clear();
        self.axes = None;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, book: &Book) {
        self.controls(ui);
        ui.separator();
        self.pump(ui);

        // Start a map whenever the options have moved away from what is on screen.
        if self.running.is_none() && self.built_under != Some(self.options) {
            self.running = Some(task::spawn_tfmap(task::MapJob {
                book: book.clone(),
                options: self.options,
            }));
            self.error = None;
        }

        if let Some(e) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, e);
            return;
        }
        let Some(tex) = &self.texture else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("computing the map…");
            });
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            return;
        };

        ui.weak(&self.caption);
        // Fitted to the panel: the grid is a setting, and the picture is scaled to whatever room
        // there is rather than recomputed to fit it.
        let available = ui.available_size();
        let response = ui.add(
            egui::Image::new((tex.id(), tex.size_vec2()))
                .fit_to_exact_size(available)
                .maintain_aspect_ratio(true),
        );

        if self.options.overlay {
            // "If the heat is not under the dots, something is wrong" — the atoms' own `(t0, f)`,
            // over the map their energy was deposited into.
            let r = response.rect;
            let painter = ui.painter_at(r);
            let colour = egui::Color32::from_rgba_unmultiplied(120, 220, 255, 150);
            for &(t, f) in &self.dots {
                if let Some(p) = self.place(t, f, r) {
                    painter.circle_filled(p, 1.5, colour);
                }
            }
        }
    }

    /// Where an atom's `(t, f)` lands on the drawn rect, or `None` when it is off the grid.
    ///
    /// Frequency goes through the *edges* rather than a linear interpolation of the range, because
    /// the axis may be geometric — mapping it linearly would put every dot in the wrong place on a
    /// log map, which is the one case the overlay is most useful for.
    fn place(&self, t: f64, f: f32, rect: egui::Rect) -> Option<egui::Pos2> {
        let (t0, t1, edges) = self.axes.as_ref()?;
        if t < *t0 || t > *t1 || edges.len() < 2 {
            return None;
        }
        let x = (t - t0) / (t1 - t0);
        let i = match edges.binary_search_by(|e| e.total_cmp(&f)) {
            Ok(i) => i as f64,
            Err(0) => return None,
            Err(i) if i >= edges.len() => return None,
            Err(i) => {
                let (a, b) = (edges[i - 1], edges[i]);
                (i - 1) as f64 + ((f - a) / (b - a)) as f64
            }
        };
        let y = 1.0 - i / (edges.len() - 1) as f64;
        Some(egui::pos2(
            rect.left() + rect.width() * x as f32,
            rect.top() + rect.height() * y as f32,
        ))
    }

    fn pump(&mut self, ui: &mut egui::Ui) {
        let Some(run) = &mut self.running else { return };
        for update in run.drain() {
            match update {
                task::MapUpdate::Done(done) => {
                    let image = egui::ColorImage::from_rgb([done.width, done.height], &done.rgb);
                    self.texture = Some(ui.ctx().load_texture(
                        "tfmap",
                        image,
                        egui::TextureOptions::LINEAR,
                    ));
                    self.caption = done.caption;
                    self.dots = done.dots;
                    self.axes = Some((done.t0, done.t1, done.f_edges));
                    self.built_under = Some(self.options);
                }
                task::MapUpdate::Failed(e) => {
                    self.error = Some(e);
                    self.built_under = Some(self.options);
                }
            }
        }
        if run.finished() {
            self.running = None;
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("grid");
            ui.add(egui::DragValue::new(&mut self.options.n_t).range(16..=8192).speed(8));
            ui.label("x");
            ui.add(egui::DragValue::new(&mut self.options.n_f).range(16..=8192).speed(8));
            ui.label("cells");

            ui.separator();
            ui.checkbox(&mut self.options.log_freq, "log frequency");

            ui.separator();
            ui.label("floor");
            ui.add(egui::DragValue::new(&mut self.options.floor_db).range(6.0..=140.0).speed(1.0).suffix(" dB"))
                .on_hover_text("display range below the reference; a map at the wrong floor is unreadable");
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("reference");
            ui.selectable_value(&mut self.options.reference, Reference::Max, "max")
                .on_hover_text("this map's loudest cell — self-scaling, so two maps are not comparable");
            ui.selectable_value(&mut self.options.reference, Reference::Initial, "initial")
                .on_hover_text("the book's initial energy — makes two maps directly comparable");

            ui.separator();
            ui.label("weight");
            ui.selectable_value(&mut self.options.weight, Weight::EnergyRemoved, "energy");
            ui.selectable_value(&mut self.options.weight, Weight::HrScore, "hr");
            ui.selectable_value(&mut self.options.weight, Weight::AtomEnergy, "atom");

            ui.separator();
            ui.checkbox(&mut self.options.overlay, "overlay atoms")
                .on_hover_text("mark each atom's (t0, f) — if the heat is not under the dots, something is wrong");
        });

        ui.horizontal(|ui| {
            ui.label("window");
            let a = ui.add(egui::TextEdit::singleline(&mut self.start).desired_width(56.0).hint_text("start"));
            let b = ui.add(egui::TextEdit::singleline(&mut self.duration).desired_width(56.0).hint_text("dur"));
            ui.label("s");
            if a.changed() || b.changed() {
                self.options.start = self.start.trim().parse().ok();
                self.options.duration = self.duration.trim().parse().ok().filter(|d: &f64| *d > 0.0);
            }
        });
    }
}
