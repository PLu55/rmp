//! Structural analysis of an rmp book.
//!
//! Runs *after* the pursuit and never touches it. The MP book stays the authoritative, detailed
//! description of the sound. What this crate derives from it are interpretations that deliberately
//! throw detail away, and they can be regenerated with other settings:
//!
//! ```text
//! MP atom   → a local acoustic event                     (the book, from rmp-core)
//! partial   → a persistent spectral component             (implemented: `analyze_partials`)
//! stem      → a group of partials sharing their behaviour  (planned; the format is in `output`)
//! ```
//!
//! `rmp_structure_spec.md` is the specification. The stages are kept separate on purpose:
//! [`observation`] normalises FOF and Gaussian atoms into one representation, [`partial`] extracts
//! partials from those observations, and stem extraction will read only partial books. Library code
//! prints nothing; every count a front end might report is in a returned diagnostics struct.

pub mod config;
pub mod error;
pub mod ids;
pub mod observation;
pub mod output;
pub mod partial;
pub mod trajectory;

pub use config::{
    FrequencyScale, PartialAnalysisConfig, PartialTrackingConfig, ScaleKind, StemAnalysisConfig,
    StructureAnalysisConfig,
};
pub use error::{Result, StructureError};
pub use ids::{AtomId, PartialId, StemId};
pub use observation::{AtomObservation, observe};
pub use output::partial_book::{PartialBook, PartialBookMetadata};
pub use output::stem_book::{
    Fundamental, Stem, StemBook, StemBookMetadata, StemMorphology, StemPartialMembership,
};
pub use partial::{Partial, PartialAnalysis, PartialDiagnostics, analyze_partials};
pub use rmp_core::AtomKind;
pub use trajectory::{Trajectory, TrajectoryPoint};

/// This crate's version, recorded in every document it writes.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
