//! The settings manual, in a window.
//!
//! The content is `MANUAL.md` itself, embedded with `include_str!` — not a summary of it written
//! for the GUI. `CLAUDE.md` already names that file as the settings reference, "every knob, what it
//! does to the result and to the clock, with the measured numbers", and a second explanation would
//! be a second manual to keep true. Embedding rather than reading from disk means the help is
//! always the manual this binary was built from, and works wherever the binary is run from.
//!
//! The manual turns out to be shaped exactly like the help this needs. Its `##` sections are the
//! TOML sections (`[blocks]`, `[refine]`, `[pursuit]`…) and its `###` subsections are named for the
//! settings themselves — "`### capture_tolerance` — default `0.95`" — so the table of contents is
//! a list of settings, and it is *derived* rather than written out here. A setting added to the
//! manual appears in this window with no code change.

use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use std::ops::Range;
use std::time::{Duration, Instant};

const MANUAL: &str = include_str!("../../MANUAL.md");

/// How long to let [`egui::ViewportCommand::Focus`] work before giving up on it and remapping the
/// window instead. Long enough for a compositor round trip, short enough not to read as a hang.
const FOCUS_GRACE: Duration = Duration::from_millis(150);

fn help_viewport() -> egui::ViewportId {
    egui::ViewportId::from_hash_of("rmp-help")
}

/// One `##` or `###` heading, and the text under it.
struct Entry {
    /// 2 for a `##` section, 3 for a `###` setting.
    level: u8,
    /// The heading, with its backticks stripped so the list reads as prose.
    label: String,
    /// Where this entry's markdown lives in [`MANUAL`], heading included. A `##` runs to the next
    /// `##` — subsections and all — so selecting a section shows it whole.
    body: Range<usize>,
}

pub struct Help {
    pub open: bool,
    /// Set by [`Help::open_or_raise`], consumed by the next [`Help::show`]. A flag rather than a
    /// direct call because the commands have to be sent *to the help viewport*, which only exists
    /// inside `show`.
    raise: bool,
    /// Set while waiting to see whether `Focus` actually raised the window. See [`Help::show`].
    focus_deadline: Option<Instant>,
    /// Skip drawing the viewport for one frame, which destroys the window so the next frame maps a
    /// fresh one.
    remap: bool,
    /// Whether the window was drawn last frame. A raise means nothing before there is a window.
    exists: bool,
    /// Mirrors the child viewport's own report of itself, read inside the callback because that is
    /// the only place its `ViewportInfo` is reachable.
    focused: bool,
    /// Outer position and inner size, carried across a remap so the window comes back where it was
    /// rather than wherever the compositor would put a new one.
    geometry: Option<(egui::Pos2, egui::Vec2)>,
    entries: Vec<Entry>,
    /// Index into `entries`, or `None` for the manual's preamble.
    selected: Option<usize>,
    filter: String,
    cache: CommonMarkCache,
}

impl Default for Help {
    fn default() -> Self {
        Self {
            open: false,
            raise: false,
            focus_deadline: None,
            remap: false,
            exists: false,
            focused: false,
            geometry: None,
            entries: parse(MANUAL),
            selected: None,
            filter: String::new(),
            cache: CommonMarkCache::default(),
        }
    }
}

impl Help {
    /// Show the manual, and bring it forward whether or not it was already up.
    ///
    /// Both halves matter: a window that is open but buried behind the main one is, from where the
    /// user is sitting, not open. Pressing `?` has to do something either way.
    pub fn open_or_raise(&mut self) {
        self.open = true;
        self.raise = true;
    }

    /// Open the manual in a window of its own, raising it if it is buried.
    ///
    /// A real OS window rather than an `egui::Window`, because the point of it is to be read
    /// *beside* the settings it explains — an in-app window is trapped inside the main one,
    /// covering the very panel you opened it to understand. As its own window it can be moved
    /// aside, put on a second screen, and alt-tabbed to.
    ///
    /// Immediate rather than deferred: a deferred viewport's callback must be `Send + Sync +
    /// 'static`, so it cannot borrow the filter, the selection or the markdown cache that live
    /// here. An immediate one is `FnMut` and runs inside this frame, which is what lets the window
    /// simply read the state it is about.
    ///
    /// # Raising it, which is harder than it should be
    ///
    /// On Wayland, winit 0.30 implements none of the obvious levers: `focus_window` is an empty
    /// function, `set_visible` says "Not possible on Wayland", `set_window_level` is empty, and
    /// `request_user_attention` builds its xdg-activation token with `set_surface` alone — no seat
    /// and serial proving recent user input — so KWin's focus-stealing prevention demotes a
    /// genuine `activate` to "demands attention", a highlight in the task manager. That is the
    /// behaviour this works around: pressing `?` marked the window and left it buried.
    ///
    /// So the fallback is to **remap** it: skip drawing the viewport for one frame, which destroys
    /// the window, then map a fresh one at the same position and size. A newly mapped window from
    /// the app that already has focus is one a compositor will normally focus, which is exactly
    /// the situation — the click that asked for it landed in the main window.
    ///
    /// It is a fallback rather than the first move, and the order matters: `Focus` is tried first
    /// and the remap happens only if the window is still unfocused [`FOCUS_GRACE`] later. Where
    /// `Focus` works — X11, macOS, Windows — nothing is destroyed and there is no blink; only
    /// where the platform refuses does the window flicker. No platform detection is involved, so
    /// a winit that grows Wayland focus support will quietly stop triggering the fallback.
    pub fn show(&mut self, ui: &mut egui::Ui) {
        if !self.open {
            self.exists = false;
            return;
        }
        let ctx = ui.ctx().clone();

        // Ask politely first, and only if there is a window and it is not already in front.
        if std::mem::take(&mut self.raise) && self.exists && !self.focused {
            ctx.send_viewport_cmd_to(help_viewport(), egui::ViewportCommand::Focus);
            ctx.send_viewport_cmd_to(
                help_viewport(),
                egui::ViewportCommand::RequestUserAttention(egui::UserAttentionType::Informational),
            );
            self.focus_deadline = Some(Instant::now() + FOCUS_GRACE);
        }

        // Did it work? Keep the frames coming while waiting, or the answer never arrives.
        if let Some(deadline) = self.focus_deadline {
            if self.focused {
                self.focus_deadline = None;
            } else if Instant::now() >= deadline {
                self.focus_deadline = None;
                self.remap = true;
            } else {
                ctx.request_repaint();
            }
        }

        if std::mem::take(&mut self.remap) {
            // Not drawing it *is* destroying it: an immediate viewport exists only while it is
            // being shown. The next frame maps a new one from `geometry`.
            self.exists = false;
            ctx.request_repaint();
            return;
        }

        let mut builder = egui::ViewportBuilder::default()
            .with_title("rmp settings — the manual")
            .with_min_inner_size([560.0, 360.0]);
        builder = match self.geometry {
            // A remap: put it back exactly where it was.
            Some((pos, size)) => builder.with_position(pos).with_inner_size(size),
            // First open: let the window manager place it.
            None => builder.with_inner_size([980.0, 720.0]),
        };

        let mut close = false;
        ctx.show_viewport_immediate(help_viewport(), builder, |ui, _class| {
            let ictx = ui.ctx().clone();
            ictx.input(|i| {
                let vp = i.viewport();
                self.focused = vp.focused.unwrap_or(false);
                if let Some(outer) = vp.outer_rect {
                    let size = vp.inner_rect.map_or(outer.size(), |r| r.size());
                    self.geometry = Some((outer.min, size));
                }
            });

            egui::CentralPanel::default().show(ui, |ui| self.contents(ui));

            // The OS close button. Without this the window shuts and the `?` cannot reopen it,
            // because `open` would still say it is up.
            if ictx.input(|i| i.viewport().close_requested()) {
                close = true;
            }
        });
        self.exists = true;

        if close {
            self.open = false;
            self.exists = false;
            self.focus_deadline = None;
        }
    }

    fn contents(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(300.0);
                ui.horizontal(|ui| {
                    ui.label("Find");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.filter)
                            .hint_text("a setting, or a word")
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.separator();
                self.toc(ui);
            });

            ui.separator();

            ui.vertical(|ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .id_salt(self.selected)
                    .show(ui, |ui| {
                        let text = match self.selected {
                            Some(i) => &MANUAL[self.entries[i].body.clone()],
                            // Before anything is chosen: the manual's own opening, down to the
                            // first section.
                            None => &MANUAL[..self.entries.first().map_or(MANUAL.len(), |e| e.body.start)],
                        };
                        CommonMarkViewer::new().show(ui, &mut self.cache, text);
                    });
            });
        });
    }

    /// The table of contents, filtered.
    ///
    /// A `##` section survives the filter if it matches *or any of its settings do*, so searching
    /// for a setting shows it under the section it belongs to rather than stripped of context.
    fn toc(&mut self, ui: &mut egui::Ui) {
        let needle = self.filter.trim().to_lowercase();
        let keep: Vec<bool> = self.matches(&needle);

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let mut any = false;
            for (i, (e, keep)) in self.entries.iter().zip(&keep).enumerate() {
                if !keep {
                    continue;
                }
                any = true;
                let selected = self.selected == Some(i);
                let text = if e.level == 2 {
                    egui::RichText::new(&e.label).strong()
                } else {
                    egui::RichText::new(format!("    {}", e.label))
                };
                if ui.selectable_label(selected, text).clicked() {
                    self.selected = Some(i);
                }
            }
            if !any {
                ui.weak("nothing matches");
            }
        });
    }

    /// Which entries survive the filter. Empty filter keeps everything.
    fn matches(&self, needle: &str) -> Vec<bool> {
        if needle.is_empty() {
            return vec![true; self.entries.len()];
        }
        let mut keep: Vec<bool> = self
            .entries
            .iter()
            .map(|e| {
                e.label.to_lowercase().contains(needle)
                    || MANUAL[e.body.clone()].to_lowercase().contains(needle)
            })
            .collect();
        // Pull each surviving `###` up with its `##`, so a hit is shown in context.
        let mut last_section = None;
        for i in 0..self.entries.len() {
            match self.entries[i].level {
                2 => last_section = Some(i),
                _ if keep[i] => {
                    if let Some(s) = last_section {
                        keep[s] = true;
                    }
                }
                _ => {}
            }
        }
        keep
    }
}

/// Split the manual at its `##` and `###` headings.
///
/// Only those two levels: `#` is the manual's own title and there is nothing below `###`. Fenced
/// code blocks are skipped, because `MANUAL.md` contains shell examples with `#` comments — a
/// `# analyse: needs at least one output` line inside a fence would otherwise become a section.
fn parse(md: &str) -> Vec<Entry> {
    let mut starts: Vec<(usize, u8)> = Vec::new();
    let mut in_fence = false;
    let mut at = 0usize;
    for line in md.split_inclusive('\n') {
        let t = line.trim_start();
        if t.starts_with("```") {
            in_fence = !in_fence;
        } else if !in_fence {
            if t.starts_with("### ") {
                starts.push((at, 3));
            } else if t.starts_with("## ") {
                starts.push((at, 2));
            }
        }
        at += line.len();
    }

    starts
        .iter()
        .enumerate()
        .map(|(n, &(start, level))| {
            // A `###` ends at the next heading of any level; a `##` runs past its own `###`
            // children to the next `##`, so selecting a section shows it whole.
            let end = starts[n + 1..]
                .iter()
                .find(|&&(_, l)| if level == 2 { l == 2 } else { true })
                .map_or(md.len(), |&(s, _)| s);
            Entry { level, label: label_of(&md[start..]), body: start..end }
        })
        .collect()
}

/// The heading text, without its `#`s or backticks.
fn label_of(from_heading: &str) -> String {
    from_heading
        .lines()
        .next()
        .unwrap_or_default()
        .trim_start_matches('#')
        .trim()
        .replace('`', "")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table of contents is derived from the manual, so this is really a check that the manual
    /// still has the shape the window assumes.
    #[test]
    fn the_manual_splits_into_sections_and_settings() {
        let e = parse(MANUAL);
        assert!(e.iter().filter(|e| e.level == 2).count() >= 10, "the numbered sections");
        assert!(e.iter().filter(|e| e.level == 3).count() >= 30, "one per setting, give or take");
        assert!(e.iter().all(|e| !e.label.is_empty()), "every entry needs a label");
        assert!(e.iter().all(|e| !e.label.contains('`')), "backticks are stripped for the list");
    }

    /// Shell examples in `MANUAL.md` use `#` comments, and a fenced block full of them would
    /// otherwise fill the contents with entries like "analyse: needs at least one output".
    #[test]
    fn hashes_inside_code_fences_are_not_headings() {
        let md = "## Real\n\ntext\n\n```bash\n## not a heading\n### nor this\n```\n\n## Also real\n";
        let e = parse(md);
        let labels: Vec<&str> = e.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, ["Real", "Also real"]);
    }

    /// The manual really does contain such a fence, so the guard is load-bearing here and not
    /// merely in principle.
    #[test]
    fn the_real_manual_has_no_entry_from_inside_a_fence() {
        let e = parse(MANUAL);
        assert!(
            MANUAL.contains("\n# analyse: needs at least one output"),
            "the fixture line this is about has moved; check the fence handling still matters"
        );
        assert!(
            !e.iter().any(|e| e.label.contains("needs at least one output")),
            "a shell comment became a section"
        );
    }

    /// A `##` carries its settings with it, so selecting a section reads as one piece.
    #[test]
    fn a_section_contains_its_settings_and_a_setting_contains_only_itself() {
        let md = "## Section\n\nintro\n\n### one\n\nA\n\n### two\n\nB\n\n## Next\n\nC\n";
        let e = parse(md);
        let body = |label: &str| {
            let x = e.iter().find(|e| e.label == label).unwrap();
            &md[x.body.clone()]
        };
        assert!(body("Section").contains("intro") && body("Section").contains("### two"));
        assert!(!body("Section").contains("## Next"), "a section stops at the next section");
        assert!(body("one").contains('A') && !body("one").contains('B'));
    }

    #[test]
    fn the_filter_finds_a_setting_and_keeps_it_under_its_section() {
        let h = Help::default();
        let keep = h.matches("capture_tolerance");
        let shown: Vec<&Entry> =
            h.entries.iter().zip(&keep).filter(|(_, k)| **k).map(|(e, _)| e).collect();

        assert!(
            shown.iter().any(|e| e.level == 3 && e.label.starts_with("capture_tolerance")),
            "the setting itself"
        );
        assert!(shown.iter().any(|e| e.level == 2 && e.label.contains("[blocks]")), "its section");
        assert!(shown.len() < h.entries.len(), "and not simply everything");
    }

    /// Pressing `?` on an already-open window must still ask for a raise. The tempting
    /// "optimisation" — only act when it is closed — is exactly the bug this is about: the window
    /// is open, buried behind the main one, and the button appears to do nothing.
    #[test]
    fn asking_for_help_again_still_asks_for_a_raise() {
        let mut h = Help::default();
        h.open_or_raise();
        assert!(h.open && h.raise);

        // As `show` does once it has sent the commands.
        h.raise = false;

        h.open_or_raise();
        assert!(h.raise, "a second press must raise the window that is already open");
    }

    /// The decision `show` makes each frame, lifted out so it can be tested without a compositor.
    ///
    /// Mirrors the first three blocks of `show` exactly. It is a duplicate of that logic rather
    /// than the logic itself, which is the weakness of this test — but the alternative is no
    /// coverage at all of a state machine whose failure mode is an invisible window.
    fn step(h: &mut Help, now: Instant) -> &'static str {
        if std::mem::take(&mut h.raise) && h.exists && !h.focused {
            h.focus_deadline = Some(now + FOCUS_GRACE);
            return "asked for focus";
        }
        if let Some(deadline) = h.focus_deadline {
            if h.focused {
                h.focus_deadline = None;
                return "focus worked";
            } else if now >= deadline {
                h.focus_deadline = None;
                h.remap = true;
            } else {
                return "waiting";
            }
        }
        if std::mem::take(&mut h.remap) {
            h.exists = false;
            return "remapped";
        }
        h.exists = true;
        "drawn"
    }

    /// The first open must not remap: there is no window yet, so there is nothing buried, and a
    /// blink before the window has even appeared would be pure noise.
    #[test]
    fn opening_for_the_first_time_just_draws() {
        let mut h = Help::default();
        h.open_or_raise();
        let t = Instant::now();
        assert_eq!(step(&mut h, t), "drawn");
        assert!(h.exists);
        assert!(h.focus_deadline.is_none(), "nothing to wait for on a first open");
    }

    /// Where `Focus` works — X11, macOS, Windows — the window is never destroyed, so there is no
    /// blink. This is what stops the Wayland workaround costing every other platform something.
    #[test]
    fn when_focus_works_the_window_is_never_remapped() {
        let mut h = Help::default();
        h.open_or_raise();
        step(&mut h, Instant::now()); // drawn

        h.open_or_raise();
        let t = Instant::now();
        assert_eq!(step(&mut h, t), "asked for focus");

        // The compositor honours it before the grace runs out.
        h.focused = true;
        assert_eq!(step(&mut h, t + Duration::from_millis(10)), "focus worked");
        assert_eq!(step(&mut h, t + Duration::from_millis(20)), "drawn");
        assert!(h.exists, "the window survived");
    }

    /// Where it does not, the window is remapped once the grace expires — and only then.
    #[test]
    fn when_focus_is_ignored_the_window_is_remapped_after_the_grace() {
        let mut h = Help::default();
        h.open_or_raise();
        step(&mut h, Instant::now());

        h.open_or_raise();
        let t = Instant::now();
        assert_eq!(step(&mut h, t), "asked for focus");
        assert_eq!(step(&mut h, t + Duration::from_millis(10)), "waiting", "not yet");
        assert_eq!(step(&mut h, t + FOCUS_GRACE), "remapped");
        assert!(!h.exists, "the viewport is gone for one frame");

        // And comes straight back.
        assert_eq!(step(&mut h, t + FOCUS_GRACE + Duration::from_millis(16)), "drawn");
        assert!(h.exists);
    }

    /// A raise while the window is already in front does nothing at all — no commands, no grace,
    /// and above all no blink.
    #[test]
    fn raising_a_window_that_is_already_in_front_is_a_no_op() {
        let mut h = Help::default();
        h.open_or_raise();
        step(&mut h, Instant::now());
        h.focused = true;

        h.open_or_raise();
        let t = Instant::now();
        assert_eq!(step(&mut h, t), "drawn");
        assert!(h.focus_deadline.is_none());
        assert!(h.exists);
    }

    /// The geometry is what makes the remap tolerable: the window comes back where it was, not
    /// wherever a new window would land.
    #[test]
    fn a_remap_keeps_the_position_and_size() {
        let mut h = Help::default();
        h.open_or_raise();
        step(&mut h, Instant::now());
        let placed = (egui::pos2(1200.0, 300.0), egui::vec2(700.0, 500.0));
        h.geometry = Some(placed);

        h.open_or_raise();
        let t = Instant::now();
        step(&mut h, t);
        step(&mut h, t + FOCUS_GRACE);
        assert_eq!(h.geometry, Some(placed), "a remap must not forget where the window was");
    }

    #[test]
    fn an_empty_filter_keeps_the_whole_contents() {
        let h = Help::default();
        assert_eq!(h.matches("").iter().filter(|k| **k).count(), h.entries.len());
    }

    #[test]
    fn a_filter_that_matches_nothing_keeps_nothing() {
        let h = Help::default();
        assert_eq!(h.matches("zzzznotasetting").iter().filter(|k| **k).count(), 0);
    }

    /// Every settings section the CLI's own default document names should be findable here, which
    /// is the property that makes this a *settings* manual rather than a document that happens to
    /// be in the window.
    #[test]
    fn every_toml_section_of_the_settings_document_is_in_the_manual() {
        let h = Help::default();
        for section in [
            "[dictionary.fof]",
            "[dictionary.gaussian]",
            "[envelope]",
            "[blocks]",
            "[pursuit]",
            "[refine]",
            "[hrmp]",
            "[residual]",
        ] {
            assert!(
                h.matches(section).iter().any(|k| *k),
                "the manual says nothing about {section}"
            );
        }
    }
}
