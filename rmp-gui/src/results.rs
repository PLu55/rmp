//! What a tab has to show: either a run just finished, or a book read back from disk.
//!
//! A project restores a tab without re-running the pursuit — see `project::TabDoc` — so the views
//! that used to be handed a live [`Outcome`] need a second source that looks enough like one to
//! drive them, without fabricating the parts a file cannot carry.

use crate::task::Outcome;
use rmp_core::book::Book;
use rmp_core::pipeline::Timing;
use rmp_core::residual::ResidualBook;
use rmp_core::signal::Signal;

/// A tab's current results.
pub enum Results {
    /// A run just finished, in this session.
    Run(Box<Outcome>),
    /// Read back from a book a previous run wrote, after a project restored this tab. What a file
    /// cannot carry — the dictionary, the timings, the pursuit's own leftover buffer — is simply
    /// not here; see the accessors below for what stands in for each.
    Loaded(Box<Loaded>),
}

/// The pieces of a run a book on disk actually carries.
pub struct Loaded {
    pub book: Book,
    pub signal: Signal,
    pub offset: usize,
}

impl Results {
    pub fn book(&self) -> &Book {
        match self {
            Results::Run(o) => &o.analysis.book,
            Results::Loaded(l) => &l.book,
        }
    }

    pub fn signal(&self) -> &Signal {
        match self {
            Results::Run(o) => &o.signal,
            Results::Loaded(l) => &l.signal,
        }
    }

    pub fn offset(&self) -> usize {
        match self {
            Results::Run(o) => o.offset,
            Results::Loaded(l) => l.offset,
        }
    }

    /// The pursuit's own leftover buffer. Empty for a loaded book: it was never written to disk,
    /// only ever a run's working buffer, so "residual (measured)" is correctly unavailable after a
    /// restore until the tab is analysed again.
    pub fn residual(&self) -> &[f32] {
        match self {
            Results::Run(o) => &o.analysis.residual,
            Results::Loaded(_) => &[],
        }
    }

    pub fn residual_book(&self) -> Option<&ResidualBook> {
        match self {
            Results::Run(o) => crate::playback::residual_book(&o.analysis),
            Results::Loaded(l) => l.book.residual.as_ref(),
        }
    }

    /// `None` for a loaded book: how long a run took is a fact about a run, not about a file.
    pub fn timing(&self) -> Option<&Timing> {
        match self {
            Results::Run(o) => Some(&o.analysis.timing),
            Results::Loaded(_) => None,
        }
    }

    pub fn cancelled(&self) -> bool {
        match self {
            Results::Run(o) => o.analysis.cancelled,
            Results::Loaded(_) => false,
        }
    }
}
