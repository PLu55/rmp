//! A tab's settings, as a document that can be loaded, edited and saved.
//!
//! The thing being edited is the TOML *text*, not a tree of widgets over [`Config`]'s fields. Three
//! reasons, and they are the same reasons the rest of this crate is thin:
//!
//! **There is one definition of the settings and it is [`Config`].** A widget per field would
//! enumerate every setting a second time, and would silently go stale the day a field is added —
//! the new knob would simply be unreachable, with nothing failing to say so. Parsing the text
//! through [`Config::from_toml`] means the editor gains a field exactly when `Config` does.
//!
//! **What you save is what you edited.** The text is written out verbatim, so comments, ordering
//! and the parts of a hand-written document that `toml::to_string_pretty` would discard all
//! survive a load-edit-save round trip. Re-serialising the parsed `Config` would quietly rewrite
//! someone's annotated settings file.
//!
//! **It is the same document the CLI works in.** `rmp --write-config`, `data/config/*.toml` and
//! `MANUAL.md` are all about this text, so a file written by one can be opened by the other.
//!
//! What the parse buys is that the document is checked continuously rather than at the moment you
//! press Analyse: [`SettingsDoc::status`] is recomputed on every edit, through the same
//! `from_toml` + `validate` pair the CLI runs.

use rmp_core::config::{Config, DEFAULT_CONFIG_HEADER};
use std::path::{Path, PathBuf};

/// The settings of one tab.
pub struct SettingsDoc {
    /// What is being edited. The only thing ever written to disk.
    text: String,
    /// What is on disk — or the starting document, when there is no file. `text != saved` is the
    /// whole of what "modified" means, so undoing an edit by hand clears the marker, which a plain
    /// dirty flag would not.
    saved: String,
    /// Where `saved` came from, and where Save writes. `None` until loaded or saved-as.
    path: Option<PathBuf>,
    /// `text` parsed and validated. Recomputed by [`SettingsDoc::reparse`] on every edit.
    parsed: Result<Config, String>,
    /// `parsed`'s normalised form, for comparing *effective* settings rather than text — see
    /// [`SettingsDoc::effective`].
    effective: Option<String>,
}

impl Default for SettingsDoc {
    /// The commented default document, the same text `rmp --write-config` prints.
    ///
    /// The comments are the point: in a text editor they are the manual, sitting beside the knob
    /// they describe.
    fn default() -> Self {
        Self::from_text(format!("{DEFAULT_CONFIG_HEADER}{}", Config::default().to_toml()), None)
    }
}

impl Clone for SettingsDoc {
    /// A duplicate carries the document *and* where it came from, so Save on the copy still writes
    /// to the file the original was loaded from. That is what makes "duplicate, change one knob,
    /// save" work; saving to a different file is Save as.
    fn clone(&self) -> Self {
        let mut c = Self::from_text(self.text.clone(), self.path.clone());
        c.saved = self.saved.clone();
        c
    }
}

impl SettingsDoc {
    fn from_text(text: String, path: Option<PathBuf>) -> Self {
        let mut d = Self {
            saved: text.clone(),
            text,
            path,
            parsed: Err(String::new()),
            effective: None,
        };
        d.reparse();
        d
    }

    /// Read a settings document. The text becomes both what is edited and the saved baseline, so a
    /// freshly loaded document is unmodified.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        Ok(Self::from_text(text, Some(path.to_path_buf())))
    }

    /// Write back to the file this was loaded from or last saved to.
    ///
    /// An invalid document saves like any other: a settings file is a thing you work on, and
    /// refusing to save a half-finished edit would lose it.
    pub fn save(&mut self) -> Result<(), String> {
        let path = self.path.clone().ok_or("no file to save to; use Save as")?;
        self.save_as(&path)
    }

    pub fn save_as(&mut self, path: &Path) -> Result<(), String> {
        std::fs::write(path, &self.text)
            .map_err(|e| format!("writing {}: {e}", path.display()))?;
        self.saved = self.text.clone();
        self.path = Some(path.to_path_buf());
        Ok(())
    }

    /// The text, for a `TextEdit`. **Call [`SettingsDoc::reparse`] after any edit** — nothing else
    /// updates [`SettingsDoc::status`], and a missed call leaves a stale verdict on screen.
    pub fn text_mut(&mut self) -> &mut String {
        &mut self.text
    }

    /// Re-check the document, through the same pair the CLI runs: parse, then validate.
    pub fn reparse(&mut self) {
        self.parsed = Config::from_toml(&self.text).and_then(|c| c.validate().map(|()| c));
        self.effective = self.parsed.as_ref().ok().map(|c| c.to_toml());
    }

    /// The settings to analyse with, or why the document is not usable.
    pub fn status(&self) -> Result<&Config, &str> {
        self.parsed.as_ref().map_err(|e| e.as_str())
    }

    /// A normalised form of the parsed settings, for asking whether two documents *mean* the same
    /// thing. Comments and formatting are gone from it, so reflowing a document or annotating a
    /// line does not read as a change to the analysis. `None` while the document does not parse.
    pub fn effective(&self) -> Option<&str> {
        self.effective.as_deref()
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn modified(&self) -> bool {
        self.text != self.saved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("rmp-gui-settings-test-{}-{name}.toml", std::process::id()));
        p
    }

    #[test]
    fn the_default_document_is_what_write_config_prints_and_it_parses() {
        let d = SettingsDoc::default();
        assert!(d.text.starts_with(DEFAULT_CONFIG_HEADER), "the commented preamble is the manual");
        assert!(d.status().is_ok(), "the default document must be usable: {:?}", d.status().err());
        assert!(!d.modified());
        assert!(d.path().is_none());
    }

    #[test]
    fn an_edit_marks_it_modified_and_undoing_the_edit_clears_that() {
        let mut d = SettingsDoc::default();
        let original = d.text.clone();

        d.text_mut().push_str("\n# a note\n");
        d.reparse();
        assert!(d.modified());
        assert!(d.status().is_ok(), "a comment does not break the document");

        *d.text_mut() = original;
        d.reparse();
        assert!(!d.modified(), "back to the saved text is not modified");
    }

    #[test]
    fn a_broken_document_says_why_and_recovers() {
        let mut d = SettingsDoc::default();
        d.text_mut().push_str("\nthis is not toml =\n");
        d.reparse();
        let err = d.status().expect_err("that is not a settings document");
        assert!(!err.is_empty(), "an error has to say something");
        assert!(d.effective().is_none(), "there are no effective settings while it does not parse");

        *d.text_mut() = SettingsDoc::default().text;
        d.reparse();
        assert!(d.status().is_ok(), "fixing the text fixes the verdict");
    }

    /// `validate` runs, not just `from_toml`: a document that parses but describes no dictionary is
    /// rejected here rather than at the moment Analyse is pressed.
    #[test]
    fn a_document_that_parses_but_cannot_work_is_rejected() {
        let mut d = SettingsDoc::default();
        *d.text_mut() = "[dictionary.fof]\nalphas = []\nbetas_ms = []\n".into();
        d.reparse();
        let err = d.status().expect_err("an empty dictionary cannot be analysed with");
        assert!(err.contains("dictionary"), "the message should name the problem, got {err:?}");
    }

    #[test]
    fn save_then_load_round_trips_the_text_verbatim() {
        let path = tmp("roundtrip");
        let mut d = SettingsDoc::default();
        // The kind of thing re-serialising a Config would silently discard.
        d.text_mut().push_str("\n# hand-written note, kept\n");
        d.reparse();

        d.save_as(&path).unwrap();
        assert!(!d.modified(), "saving clears the marker");
        assert_eq!(d.path(), Some(path.as_path()));

        let back = SettingsDoc::load(&path).unwrap();
        assert_eq!(back.text, d.text, "the file is the text, byte for byte");
        assert!(back.text.contains("hand-written note, kept"));
        assert!(!back.modified());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn save_needs_somewhere_to_save_to_until_save_as_gives_it_one() {
        let path = tmp("saveas");
        let mut d = SettingsDoc::default();
        assert!(d.save().is_err(), "a document with no file cannot Save");

        d.save_as(&path).unwrap();
        d.text_mut().push_str("\n# more\n");
        d.reparse();
        assert!(d.save().is_ok(), "Save as gave it a file, so Save works now");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), d.text);
        std::fs::remove_file(&path).ok();
    }

    /// What `effective` is for: it ignores the things that do not reach the analysis, so annotating
    /// a document does not make a finished run look stale.
    #[test]
    fn effective_settings_ignore_comments_and_formatting() {
        let plain = SettingsDoc::default();
        let mut annotated = SettingsDoc::default();
        annotated.text_mut().insert_str(0, "# my notes about this run\n\n\n");
        annotated.reparse();

        assert_ne!(annotated.text, plain.text);
        assert_eq!(annotated.effective(), plain.effective(), "same settings, different prose");

        // Setting a value, not commenting one out: every field defaults, so deleting a line that
        // already held its default changes nothing — which is how the first version of this test
        // managed to pass for the wrong reason.
        let mut changed = SettingsDoc::default();
        *changed.text_mut() = changed.text.replace("max_atoms = 1000", "max_atoms = 4321");
        changed.reparse();
        assert_ne!(changed.text, plain.text, "the fixture must actually differ");
        assert_ne!(changed.effective(), plain.effective(), "a real change does show");
        assert_eq!(changed.status().unwrap().pursuit.max_atoms, 4321);
    }

    /// A duplicate keeps the file, so Save on the copy writes where the original did.
    #[test]
    fn a_clone_carries_the_path_and_the_modified_state() {
        let mut d =
            SettingsDoc { path: Some(PathBuf::from("/a/settings.toml")), ..Default::default() };
        d.text_mut().push_str("\n# edited\n");
        d.reparse();

        let c = d.clone();
        assert_eq!(c.path(), d.path());
        assert!(c.modified(), "an unsaved edit is still unsaved in the copy");
        assert_eq!(c.effective(), d.effective());
    }
}
