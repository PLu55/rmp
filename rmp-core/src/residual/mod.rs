//! Stochastic analysis of the final MP residue.
//!
//! The pursuit leaves `x[n] = sum_k FOF_k[n] + r[n]`. The atoms are the deterministic part; `r[n]`
//! is the breath, the bow noise and the hammer thump — stochastic at waveform level, but strongly
//! nonstationary, so it still carries rhythm and transients. This module converts it into a
//! compact, fixed-rate map of power over ERB bands that a later real-time noise bank can excite
//! with `g_b = sqrt(P_b)`.
//!
//! ```text
//! r[n] -> ERB analysis bank -> y_b[n] -> one-pole power detectors -> sample every Nu -> ResidualBook
//! ```
//!
//! Three things shape everything here:
//!
//! **It is a post-processing stage and nothing else.** It runs once, after the pursuit has stopped,
//! on the pursuit's own residual buffer. It cannot change which atoms were selected, and it is off
//! by default.
//!
//! **Temporal smoothing is minimal and explicit.** The one-pole detector is the only smoothing in
//! the chain, and its time constant is a documented setting. Anything that blurred a transient
//! would destroy the part of the residual worth keeping.
//!
//! **The bank is described in the book, not assumed.** Centres, bandwidths, the measured
//! normalisation gains and the resolved per-band time constants are all serialised, so a
//! synthesiser reads what the analysis actually did rather than recomputing it from formulae that
//! may since have moved.

pub mod analyze;
pub mod book;
pub mod config;
pub mod erb;
pub mod error;
pub mod filter;
pub mod power;

pub use analyze::analyze_residual;
pub use book::{
    ErbFilterKind, ErbNormalization, ErbSpacing, ResidualBook, ResidualErbBankDescriptor,
    ResidualPowerDescriptor, ResidualStorage, RESIDUAL_BOOK_VERSION,
};
pub use config::{ErbBankConfig, ResidualAnalysisConfig, NYQUIST_GUARD};
pub use error::ResidualAnalysisError;
pub use filter::{AnalysisBand, GammatoneBand};
pub use power::{ResidualPowerConfig, ResidualPowerTimeMode};

/// Fewer bands than this cannot cover the audible range on an ERB scale at any useful resolution.
/// The spec's recommended practical floor.
pub const MIN_BANDS: usize = 4;

/// Orders the gammatone cascade implements. 4 is the classical value; the range exists because the
/// cascade is generic in its length, not because a recipe was written out four times.
pub const MIN_ORDER: usize = 1;
pub const MAX_ORDER: usize = 8;

/// A fixed pseudo-noise fixture, shared by the tests of this crate *and* of `rmp-synthesis`.
///
/// Public, and not `#[cfg(test)]`, because the residual synthesis tests in the other crate compare
/// against this exact stream. Two copies of a fixture whose values matter is precisely the drift
/// this crate avoids everywhere else, so there is one definition and it crosses the crate boundary.
///
/// Uniform on `[-1, 1)`, so its variance is exactly 1/3 — which is what
/// `broadband_noise_gives_flat_normalised_bands` compares the normalised band powers against. A
/// plain LCG rather than anything from a crate: §29.3 requires the tests be deterministic, and a
/// fixture that can change with a dependency bump is not.
pub fn pseudo_noise(n: usize) -> Vec<f32> {
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}
