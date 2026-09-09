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
pub mod residual;
pub mod select;
pub mod signal;
pub mod stats;
pub mod synth;
pub mod tfmap;

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
pub use residual::{
    ResidualAnalysisConfig, ResidualAnalysisError, ResidualBook, ResidualPowerConfig,
    ResidualPowerTimeMode, analyze_residual,
};
pub use select::SegTree;
pub use signal::Signal;
pub use stats::{BookSummary, Diagnostics, Evaluator, Histogram, Quantity, Summary, Weight};
pub use synth::{
    load_book, render_full_book, render_residual_book, render_to_file, BankCalibration, BookInput,
    ClippingPolicy, GainSmoothingConfig, GainSmoothingMode, OutputEncoding, RenderConfig,
    RenderError, RenderReport, RenderRequest, SynthesisBank,
};
pub use tfmap::{MapGrid, MapOptions, TfMap};
