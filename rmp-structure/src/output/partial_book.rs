//! The partial book: the serialisable result of partial extraction (§17).
//!
//! Written and read through `rmp_core::book::{write_doc, read_doc}`, so its file-name rules are the
//! MP book's own — `.json`, `.toml`, either with `.gz` — and cannot drift from them. It never
//! replaces the book it was derived from; it records where that book was, and enough of the
//! configuration and versions to regenerate it.

use crate::config::PartialAnalysisConfig;
use crate::error::{Result, StructureError};
use crate::partial::Partial;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Bumped when the format changes in a way an older reader would misread.
pub const PARTIAL_BOOK_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartialBookMetadata {
    pub format_version: u32,
    pub sample_rate: f64,
    /// The MP book analysed, when it came from a file. The library cannot know this; the caller
    /// that read the book fills it in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_book: Option<PathBuf>,
    /// Where the analysed excerpt began in its source file, copied from the book. Every time in
    /// this document is absolute on the source timeline already; this is for reference.
    pub start_sample: u64,
    pub config: PartialAnalysisConfig,
    /// `rmp-structure`'s version.
    pub version: String,
    /// `rmp-core`'s version.
    pub rmp_version: String,
}

impl PartialBookMetadata {
    pub fn new(sample_rate: f64, start_sample: u64, config: PartialAnalysisConfig) -> Self {
        Self {
            format_version: PARTIAL_BOOK_VERSION,
            sample_rate,
            source_book: None,
            start_sample,
            config,
            version: crate::VERSION.to_string(),
            rmp_version: rmp_core::VERSION.to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartialBook {
    pub metadata: PartialBookMetadata,
    pub partials: Vec<crate::partial::Partial>,
}

impl PartialBook {
    pub fn write(&self, path: &Path) -> Result<()> {
        rmp_core::book::write_doc(path, self).map_err(StructureError::Io)
    }

    pub fn read(path: &Path) -> Result<Self> {
        let book: Self = rmp_core::book::read_doc(path).map_err(StructureError::Io)?;
        book.validate().map_err(|e| StructureError::Io(format!("{}: {e}", path.display())))?;
        Ok(book)
    }

    /// Reject a document this build cannot interpret, or one whose ids are not what analysis
    /// assigns.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.metadata.format_version > PARTIAL_BOOK_VERSION {
            return Err(format!(
                "partial book version {} is newer than this build understands ({})",
                self.metadata.format_version, PARTIAL_BOOK_VERSION
            ));
        }
        for (i, p) in self.partials.iter().enumerate() {
            if p.id.0 as usize != i {
                return Err(format!("partial {i} has id {}: ids must be 0, 1, 2, …", p.id.0));
            }
        }
        Ok(())
    }

    /// Partials by descending significance, ties by id.
    pub fn by_significance(&self) -> Vec<&Partial> {
        let mut v: Vec<&Partial> = self.partials.iter().collect();
        v.sort_by(|a, b| b.significance.total_cmp(&a.significance).then(a.id.cmp(&b.id)));
        v
    }
}
