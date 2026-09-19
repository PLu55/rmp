//! The document `Save Project` writes and `Open Project` reads.
//!
//! A project is a directory: `project.toml` beside whatever books, rendered soundfiles and settings
//! documents its tabs have written. The document is its own struct rather than `Session` serialised
//! directly — a file format is a promise about what its fields mean, and `Session` is free to grow
//! fields the format has no business remembering (view caches, a run in flight). Written through
//! `rmp_core::book::write_doc`/`read_doc`, the same extension-driven (de)serialisation a book uses,
//! so this format needs no I/O code of its own — see `CLAUDE.md`'s "one definition of how this
//! codebase writes a serialisable document."
//!
//! TOML, and un-gzipped: a project document is a handful of tab entries, small enough and plain
//! enough that a person might reasonably open it by hand — to repoint a moved input file, say — the
//! same reason a settings document is TOML.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const PROJECT_FILE: &str = "project.toml";

/// Bumped when a field's *meaning* changes, not merely when one appears — an appearing field is
/// `#[serde(default)]` and needs no bump, an incompatible one does. Nothing reads this yet; it is
/// insurance for the day something does.
const PROJECT_FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, PartialEq, Default)]
pub struct ProjectDoc {
    #[serde(default)]
    pub version: u32,
    pub tabs: Vec<TabDoc>,
    pub active: usize,
}

impl ProjectDoc {
    pub fn new(tabs: Vec<TabDoc>, active: usize) -> Self {
        Self { version: PROJECT_FORMAT_VERSION, tabs, active }
    }
}

/// One tab, as the project document remembers it. Everything a fresh run would ask for again
/// (settings, switches) plus where its last book and settings documents were written, so a restore
/// can read the book back rather than requiring a re-analysis.
#[derive(Serialize, Deserialize, PartialEq)]
pub struct TabDoc {
    pub number: u32,
    pub input: PathBuf,
    pub start: String,
    pub duration: String,
    pub settings: Option<PathBuf>,
    pub keep_residual: bool,
    pub run_residual_analysis: bool,
    pub parts: crate::task::RenderParts,
    pub sources: crate::playback::Sources,
    pub book: Option<PathBuf>,
    pub view: crate::app::View,
}

/// A path as the document stores it: relative to the project directory when it lives inside that
/// directory, absolute otherwise. Every path a tab's own dialogs propose by default lives inside
/// it once a project is active, so the common case round-trips as a relative path and the whole
/// directory stays portable without hand-editing; a path outside it — an input file living
/// elsewhere, or an output a save dialog was pointed away from deliberately — just stores absolute.
pub fn store_path(project_dir: &Path, p: &Path) -> PathBuf {
    p.strip_prefix(project_dir).map(Path::to_path_buf).unwrap_or_else(|_| p.to_path_buf())
}

/// The inverse of [`store_path`]: an absolute path is already resolved, a relative one is read
/// against the project directory.
pub fn resolve_path(project_dir: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() { p.to_path_buf() } else { project_dir.join(p) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_resolve_round_trip_a_path_inside_the_project_directory() {
        let dir = Path::new("/home/me/proj");
        let p = Path::new("/home/me/proj/piano-01-book.json.gz");
        let stored = store_path(dir, p);
        assert_eq!(stored, Path::new("piano-01-book.json.gz"));
        assert_eq!(resolve_path(dir, &stored), p);
    }

    #[test]
    fn a_path_outside_the_project_directory_stores_absolute() {
        let dir = Path::new("/home/me/proj");
        let p = Path::new("/home/me/audio/piano.wav");
        let stored = store_path(dir, p);
        assert_eq!(stored, p, "unchanged — it is not under the project directory");
        assert_eq!(resolve_path(dir, &stored), p);
    }

    #[test]
    fn a_project_document_round_trips_through_write_doc_and_read_doc() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rmp-gui-project-test-{}.toml", std::process::id()));

        let doc = ProjectDoc::new(
            vec![TabDoc {
                number: 1,
                input: PathBuf::from("/audio/piano.wav"),
                start: "2.0".into(),
                duration: "0.5".into(),
                settings: Some(PathBuf::from("piano-01.toml")),
                keep_residual: true,
                run_residual_analysis: false,
                parts: crate::task::RenderParts::default(),
                sources: crate::playback::Sources::default(),
                book: Some(PathBuf::from("piano-01-book.json.gz")),
                view: crate::app::View::Summary,
            }],
            0,
        );

        rmp_core::book::write_doc(&path, &doc).expect("writing the project document");
        let back: ProjectDoc = rmp_core::book::read_doc(&path).expect("reading it back");
        assert!(back == doc, "the document must round-trip exactly");

        std::fs::remove_file(&path).ok();
    }
}
