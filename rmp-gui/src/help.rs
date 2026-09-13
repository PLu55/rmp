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

const MANUAL: &str = include_str!("../../MANUAL.md");

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
            entries: parse(MANUAL),
            selected: None,
            filter: String::new(),
            cache: CommonMarkCache::default(),
        }
    }
}

impl Help {
    /// Open the manual in a window of its own.
    ///
    /// A real OS window rather than an `egui::Window`, because the point of it is to be read
    /// *beside* the settings it explains — and an in-app window is trapped inside the main one,
    /// covering the very panel you opened it to understand. As its own window it can be moved
    /// aside, put on a second screen, and alt-tabbed to.
    ///
    /// Immediate rather than deferred: a deferred viewport's callback must be `Send + Sync +
    /// 'static`, so it cannot borrow the filter, the selection or the markdown cache that live
    /// here. An immediate one is `FnMut` and runs inside this frame, which is what lets the window
    /// simply read the state it is about.
    ///
    /// Where the backend cannot open a second window, egui says so through
    /// [`egui::ViewportClass::EmbeddedWindow`] and falls back to an in-app one on its own. Nothing
    /// here has to handle that case differently; it is just less good.
    pub fn show(&mut self, ui: &mut egui::Ui) {
        if !self.open {
            return;
        }
        let ctx = ui.ctx().clone();
        let mut close = false;
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("rmp-help"),
            egui::ViewportBuilder::default()
                .with_title("rmp settings — the manual")
                .with_inner_size([980.0, 720.0])
                .with_min_inner_size([560.0, 360.0]),
            |ui, _class| {
                egui::CentralPanel::default().show(ui, |ui| self.contents(ui));
                // The OS close button. Without this the window shuts and the `?` cannot reopen it,
                // because `open` would still say it is up.
                if ui.ctx().input(|i| i.viewport().close_requested()) {
                    close = true;
                }
            },
        );
        if close {
            self.open = false;
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
