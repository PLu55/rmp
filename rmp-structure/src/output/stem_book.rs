//! The stem book's format (§32–§35), declared ahead of the algorithm that fills it.
//!
//! Nothing produces a `StemBook` yet: stem extraction is the next phase, deliberately started only
//! once partial extraction has been checked on real books (§47). The types exist now so that the
//! format is settled while it is cheap to change, and so that membership is modelled as a list from
//! the start — a partial may belong to several stems with graded membership (§33), even though the
//! first clustering will give each at most one.

use crate::config::StemAnalysisConfig;
use crate::ids::{PartialId, StemId};
use crate::trajectory::Trajectory;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A stem's fundamental, when it has one. Inharmonic and noise-like stems are `None` (§24).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Fundamental {
    #[default]
    None,
    Static(f64),
    Trajectory(Trajectory),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StemPartialMembership {
    pub partial_id: PartialId,
    /// In `[0, 1]`.
    pub membership: f32,
}

/// Why a stem was grouped as it was: each cue summarised over its members, for inspection (§34).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StemMorphology {
    pub mean_onset_samples: u64,
    pub onset_spread_samples: f64,
    pub mean_offset_samples: u64,
    pub offset_spread_samples: f64,
    pub harmonicity: f32,
    pub amplitude_coherence: f32,
    pub frequency_coherence: f32,
    pub spectral_coherence: f32,
    pub spatial_coherence: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stem {
    pub id: StemId,
    pub start_samples: u64,
    pub end_samples: u64,
    pub partials: Vec<StemPartialMembership>,
    pub fundamental: Fundamental,
    pub amplitude: Trajectory,
    pub confidence: f32,
    pub morphology: StemMorphology,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StemBookMetadata {
    pub format_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_book: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_partial_book: Option<PathBuf>,
    pub config: StemAnalysisConfig,
    pub version: String,
    pub rmp_version: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StemBook {
    pub metadata: StemBookMetadata,
    pub stems: Vec<Stem>,
    pub unassigned_partials: Vec<PartialId>,
}
