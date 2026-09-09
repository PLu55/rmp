//! What can go wrong rendering a residual book.
//!
//! The messages name the CLI flag the user typed, the way [`crate::residual::error`] names the
//! settings-document path: a renderer that says "sample rate mismatch" is telling the user a fact
//! about their files, and it should say which two files and which flag brought the second one in.

use crate::audio::AudioError;
use crate::residual::book::ErbFilterKind;
use crate::residual::error::ResidualAnalysisError;

#[derive(Debug)]
pub enum RenderError {
    /// Reading the FOF audio or writing the output.
    Audio(AudioError),
    /// Reading or parsing the book. Carries the message `book::read_doc` produced.
    Book(String),
    /// A full book was given, but it carries no residual section.
    NoResidualBook,
    UnsupportedResidualBookVersion(u32),
    /// A book describing a bank this build does not know how to rebuild.
    UnsupportedFilterKind(ErbFilterKind),
    /// The bank could not be designed from the book's own descriptor.
    BankDesign(ResidualAnalysisError),
    /// The rebuilt bank disagrees with what the book says was analysed — a filter design or an ERB
    /// formula has moved since the book was written.
    BankMismatch(String),
    SampleRateMismatch { book: f64, fof_audio: f32 },
    /// Reserved for the multi-channel residual book of §19, which the format cannot express yet.
    ChannelMismatch { channels: usize },
    InvalidConfig(String),
    ClippingDetected { count: u64, peak: f32 },
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Audio(e) => write!(f, "{e}"),
            Self::Book(m) => write!(f, "{m}"),
            Self::NoResidualBook => write!(
                f,
                "the book carries no residual section — analyse with --residual-analysis, \
                 or pass the standalone book written by --residual-book"
            ),
            Self::UnsupportedResidualBookVersion(v) => write!(
                f,
                "residual book version {v} is newer than this build understands"
            ),
            Self::UnsupportedFilterKind(k) => write!(
                f,
                "the book's bank uses filter kind '{k}', which this build cannot rebuild"
            ),
            Self::BankDesign(e) => write!(f, "rebuilding the synthesis bank: {e}"),
            Self::BankMismatch(m) => write!(
                f,
                "the rebuilt bank does not match the one the book was analysed with: {m}"
            ),
            Self::SampleRateMismatch { book, fof_audio } => write!(
                f,
                "--fof-audio is at {fof_audio} Hz but the book was analysed at {book} Hz; \
                 resample it yourself rather than have this do it silently"
            ),
            Self::ChannelMismatch { channels } => {
                write!(f, "cannot render {channels} channels: the book is mono")
            }
            Self::InvalidConfig(m) => write!(f, "{m}"),
            Self::ClippingDetected { count, peak } => write!(
                f,
                "{count} samples exceed full scale (peak {peak:.4}); \
                 lower --gain-db or pass --clip to hard-clip them"
            ),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<AudioError> for RenderError {
    fn from(e: AudioError) -> Self {
        Self::Audio(e)
    }
}

impl From<ResidualAnalysisError> for RenderError {
    fn from(e: ResidualAnalysisError) -> Self {
        Self::BankDesign(e)
    }
}
