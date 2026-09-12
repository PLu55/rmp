//! What can go wrong rendering a book.
//!
//! The messages name the CLI flag the user typed, the way [`crate::residual::error`] names the
//! settings-document path: a renderer that says "sample rate mismatch" is telling the user a fact
//! about their files, and it should say which two files and which flag brought the second one in.

use crate::audio::AudioError;
use crate::fof::FofError;
use crate::residual::book::ErbFilterKind;
use crate::residual::error::ResidualAnalysisError;

#[derive(Debug)]
pub enum RenderError {
    /// Writing the output.
    Audio(AudioError),
    /// Reading or parsing the book. Carries the message `book::read_doc` produced.
    Book(String),
    /// Only the residual was left to render, and there is none.
    NoResidualBook,
    /// The flags and the book together leave nothing at all to render.
    NothingToRender(&'static str),
    /// An atom the book records cannot be rendered.
    Atoms(FofError),
    UnsupportedResidualBookVersion(u32),
    /// A book describing a bank this build does not know how to rebuild.
    UnsupportedFilterKind(ErbFilterKind),
    /// The bank could not be designed from the book's own descriptor.
    BankDesign(ResidualAnalysisError),
    /// The rebuilt bank disagrees with what the book says was analysed — a filter design or an ERB
    /// formula has moved since the book was written.
    BankMismatch(String),
    /// The atoms and the residual were analysed at different rates.
    SampleRateMismatch { book: f32, residual_book: f64 },
    /// The atoms and the residual describe excerpts starting at different source samples.
    TimelineMismatch { book: u64, residual_book: u64 },
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
            Self::NothingToRender(why) => write!(f, "nothing to render: {why}"),
            Self::Atoms(e) => write!(f, "rendering the atoms: {e}"),
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
            Self::SampleRateMismatch { book, residual_book } => write!(
                f,
                "the book's atoms were analysed at {book} Hz but its residual at {residual_book} \
                 Hz; they come from different analyses and cannot share a timeline"
            ),
            Self::TimelineMismatch { book, residual_book } => write!(
                f,
                "the book's excerpt starts at source sample {book} but the residual's at \
                 {residual_book}; they come from different analyses"
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

impl From<FofError> for RenderError {
    fn from(e: FofError) -> Self {
        Self::Atoms(e)
    }
}

impl From<ResidualAnalysisError> for RenderError {
    fn from(e: ResidualAnalysisError) -> Self {
        Self::BankDesign(e)
    }
}
