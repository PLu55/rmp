//! Matching pursuit decomposition of audio into FOF atoms.
//!
//! See `CLAUDE.md` for the project overview and the design plan it references.

pub mod audio;
pub mod book;
pub mod config;
pub mod corr;
pub mod dict;
pub mod fft;
pub mod fof;
pub mod mp;
pub mod naive;
pub mod select;
pub mod signal;

pub use book::{Book, Selection};
pub use config::Config;
pub use corr::{Correlator, Projection, project};
pub use dict::{Block, BlockConfig, Dictionary};
pub use fft::{Planner, RealFft, RealFftPlanner, next_fast_len};
pub use fof::{AtomParams, Envelope, EnvelopeParams, FofError, ReleasePolicy};
pub use mp::{Mp, MpConfig};
pub use naive::{NaiveConfig, NaiveMp};
pub use select::SegTree;
pub use signal::Signal;
