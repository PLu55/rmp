//! Identifiers, one newtype per level so an atom index can never be passed where a partial is meant.
//!
//! Serialised transparently: a partial book lists `supporting_atoms = [3, 17, …]`, not a table per
//! id.

use serde::{Deserialize, Serialize};

/// An atom's index in its book's `selections`. The trace back from a partial to the book it came
/// from, so it is the book's own order and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AtomId(pub u32);

/// A partial within one partial book. Assigned after a deterministic sort (see
/// [`crate::partial`]), so it is stable for a given book and configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PartialId(pub u32);

/// A stem within one stem book.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StemId(pub u32);
