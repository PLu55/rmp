//! What can go wrong designing or running the residual analyser.
//!
//! Every variant names the field it came from and the value that was rejected: a settings document
//! is hand-edited, so "bands must be at least 4" without the 2 that was written is half an error
//! message. The crate's boundaries speak `Result<_, String>` (see [`crate::config::Config::validate`]
//! and [`crate::book::read`]), so `main` converts with `.to_string()`, exactly as it does `FofError`.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum ResidualAnalysisError {
    /// Fewer bands than the bank can meaningfully cover.
    InvalidBandCount(usize),
    /// `min_freq_hz` is not below `max_freq_hz`, or one of them is not positive and finite.
    InvalidFrequencyRange { min_hz: f64, max_hz: f64 },
    /// A band centre at or above Nyquist. The pole would land on or outside the unit circle's
    /// useful arc and the band would alias rather than analyse.
    FrequencyAboveNyquist { freq_hz: f64, nyquist_hz: f64 },
    /// `update_ms` rounds to zero samples, or is not positive.
    InvalidUpdateInterval { update_ms: f64, sample_rate: f64 },
    /// A power-detector time constant is not positive, or `tau_min > tau_max`.
    InvalidPowerTimeConstant(String),
    /// The filter cascade does not implement this order.
    UnsupportedFilterOrder(usize),
    /// The designed filter came out unstable or its normalisation integral did not converge.
    FilterDesignFailed(String),
    /// The sample rate is not a usable positive number.
    InvalidSampleRate(f64),
}

impl fmt::Display for ResidualAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ResidualAnalysisError::*;
        match self {
            InvalidBandCount(n) => write!(
                f,
                "residual.erb.bands = {n}: at least {} bands are needed for an ERB bank",
                crate::residual::MIN_BANDS
            ),
            InvalidFrequencyRange { min_hz, max_hz } => write!(
                f,
                "residual.erb: min_freq_hz ({min_hz}) must be positive and below max_freq_hz ({max_hz})"
            ),
            FrequencyAboveNyquist {
                freq_hz,
                nyquist_hz,
            } => write!(
                f,
                "residual.erb.max_freq_hz = {freq_hz}: at this sample rate the usable limit is \
                 {nyquist_hz:.1} Hz"
            ),
            InvalidUpdateInterval {
                update_ms,
                sample_rate,
            } => write!(
                f,
                "residual.update_ms = {update_ms}: rounds to under one sample at {sample_rate} Hz"
            ),
            InvalidPowerTimeConstant(why) => write!(f, "residual.power: {why}"),
            UnsupportedFilterOrder(n) => write!(
                f,
                "residual.erb.order = {n}: the gammatone cascade implements orders {}..={}",
                crate::residual::MIN_ORDER,
                crate::residual::MAX_ORDER
            ),
            FilterDesignFailed(why) => write!(f, "residual filter design failed: {why}"),
            InvalidSampleRate(sr) => write!(f, "residual analysis needs a positive sample rate, got {sr}"),
        }
    }
}

impl std::error::Error for ResidualAnalysisError {}
