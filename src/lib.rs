//! Matching pursuit decomposition of audio into FOF atoms.
//!
//! See `CLAUDE.md` for the project overview and the design plan it references.

pub mod audio;
pub mod book;
pub mod cand;
pub mod config;
pub mod corr;
pub mod dict;
pub mod fft;
pub mod fit;
pub mod fof;
pub mod hrmp;
pub mod mp;
pub mod naive;
pub mod refine;
pub mod select;
pub mod signal;

pub use book::{Book, Selection};
pub use cand::{Candidate, Seed, top_seeds};
pub use config::Config;
pub use corr::{Correlator, Projection, project};
pub use dict::{Block, BlockConfig, Dictionary};
pub use fft::{Planner, RealFft, RealFftPlanner, next_fast_len};
pub use fit::{Gram, Quad, fit_end};
pub use fof::{AtomParams, Envelope, EnvelopeParams, FofError, ReleasePolicy};
pub use hrmp::{HrmpConfig, MagnitudePolicy, Outcome, Placement, ProbeMode, Verdict};
pub use mp::{Mp, MpConfig};
pub use naive::{NaiveConfig, NaiveMp};
pub use refine::{EnvelopeCache, RefineConfig, refine};
pub use select::SegTree;
pub use signal::Signal;
