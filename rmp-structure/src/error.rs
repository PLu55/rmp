//! What structural analysis can refuse.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum StructureError {
    /// A setting is out of range. The message names it.
    InvalidConfig(String),
    /// The book itself cannot be analysed — a sample rate that is not a positive number, say.
    InvalidBook(String),
    /// A derived document could not be read or written.
    Io(String),
}

impl fmt::Display for StructureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(m) => write!(f, "invalid structure settings: {m}"),
            Self::InvalidBook(m) => write!(f, "cannot analyse this book: {m}"),
            Self::Io(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for StructureError {}

pub type Result<T> = std::result::Result<T, StructureError>;
