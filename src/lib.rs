//! Matching pursuit decomposition of audio into FOF atoms.
//!
//! See `CLAUDE.md` for the project overview and the design plan it references.

pub mod fft;
pub mod fof;

pub use fft::{Planner, RealFft, RealFftPlanner, next_fast_len};
pub use fof::{AtomParams, Envelope, EnvelopeParams, FofError};
