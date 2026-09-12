//! Rendering a book's atoms, whatever their kind.
//!
//! Every atom goes through [`AtomParams::render`] — rfofs's `FofState` for a FOF, [`crate::gauss`]
//! for a Gaussian — which is exactly the render the pursuit subtracted. So a book that reached N dB
//! against its input reproduces that input to N dB here, and over the analysed excerpt the result is
//! bit for bit the signal the analysis explained.
//!
//! # Not `rfofs::OfflineRenderer`
//!
//! rfofs has an offline renderer, and it is the wrong tool for this job for four reasons, each
//! sufficient on its own:
//!
//! - it renders in engine blocks, and a FOF's `decay_acc` is a running product carried across
//!   `fill_block` calls, so a block-split grain is not bit-identical to the single-call render the
//!   analysis subtracted;
//! - it writes a float WAV directly, so nothing can be mixed into it — not a Gaussian, not the
//!   stochastic residual, not an output gain or a clipping policy;
//! - it requires weakly monotonic, non-negative onsets, and a book may start an atom before its
//!   excerpt;
//! - it stops at a 30-second safety limit on the tail.
//!
//! # A rendered book is longer than its excerpt
//!
//! [`natural_len`] sizes the output by each atom's rendered death, where analysis sized the residual
//! by the input. The atom tails the analysis truncated at the excerpt end are audible again — 2.9%
//! of the excerpt's energy on a 0.15 s piano fixture. Over the excerpt itself the renders agree to
//! the last bit, so this is a longer file, not a different one.

use crate::atom::AtomKind;
use crate::book::Book;
use crate::fof::{AtomParams, Envelope, FofError};
use crate::signal::Signal;

/// Samples a book occupies, from the analysis origin to the death of the last atom.
///
/// Atoms with a negative `t0` began before the analysed excerpt and are clipped at the origin here
/// exactly as [`render_atoms`] clips them, so a book replays in the same frame it was analysed in.
/// Support lengths come from rendering, never from a formula — the same rule the rest of the crate
/// follows.
pub fn natural_len(book: &Book) -> Result<usize, FofError> {
    let mut len = 0i64;
    for s in &book.selections {
        let env = Envelope::render(s.atom.env, book.sample_rate)?;
        len = len.max(s.atom.t0 + env.support_len() as i64);
    }
    Ok(len.max(0) as usize)
}

/// Render every atom into `len` samples, in the book's own frame: sample 0 is the analysis origin,
/// not the source file's.
pub fn render_atoms(book: &Book, len: usize) -> Result<Signal, FofError> {
    let atoms: Vec<AtomParams> = book.selections.iter().map(|s| s.atom).collect();
    Signal::from_atoms(&atoms, len, book.sample_rate)
}

/// How many atoms of each kind the book holds, in [`AtomKind::ALL`] order, omitting absent kinds.
pub fn count_by_kind(book: &Book) -> Vec<(AtomKind, usize)> {
    AtomKind::ALL
        .iter()
        .map(|&k| (k, book.selections.iter().filter(|s| s.atom.kind() == k).count()))
        .filter(|&(_, n)| n > 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::Selection;
    use crate::fof::EnvelopeParams;
    use crate::gauss::GaussianParams;

    const SR: f32 = 48_000.0;

    fn sel(atom: AtomParams) -> Selection {
        Selection {
            atom,
            block: 0,
            onset: 0,
            bin: 10,
            projected_energy: 1.0,
            energy_removed: 1.0,
            residual_energy: 0.5,
            hr_score: None,
            refined: false,
        }
    }

    fn fof_at(t0: i64) -> AtomParams {
        AtomParams { t0, f: 1000.0, env: EnvelopeParams::new(251.0, 0.001).into(), phi: 0.0, amp: 1.0 }
    }

    fn gauss_at(t0: i64) -> AtomParams {
        AtomParams { t0, f: 640.0, env: GaussianParams::new(0.004).into(), phi: 1.2, amp: 0.5 }
    }

    /// `natural_len` has to cover every atom's whole life and no more: rendering into a longer
    /// buffer must add nothing past it, and the samples just inside it must still be live.
    ///
    /// Also the reason a negative `t0` cannot extend it — `render_atoms` clips such an atom at the
    /// origin, so counting its pre-origin part would pad the output with silence the book does not
    /// contain.
    #[test]
    fn natural_len_is_exactly_where_the_last_atom_dies() {
        let mut b = Book::new(1.0, SR);
        assert_eq!(natural_len(&b).unwrap(), 0, "an empty book occupies nothing");

        let support = Envelope::render(fof_at(0).env, SR).unwrap().support_len();
        b.selections.push(sel(fof_at(-(support as i64) - 10)));
        assert_eq!(natural_len(&b).unwrap(), 0, "an atom entirely before the origin is clipped");

        b.selections.push(sel(fof_at(-200)));
        assert_eq!(natural_len(&b).unwrap(), support - 200);

        b.selections.push(sel(fof_at(5_000)));
        let len = natural_len(&b).unwrap();
        assert_eq!(len, support + 5_000);

        // A Gaussian ending later extends it by its own closed-form support.
        let g_support = GaussianParams::new(0.004).support_len(SR);
        b.selections.push(sel(gauss_at(9_000)));
        let len = natural_len(&b).unwrap();
        assert_eq!(len, (9_000 + g_support).max(support + 5_000));

        // Rendered with room to spare, the book is silent past its own length and audible inside.
        let s = render_atoms(&b, len + 4_096).unwrap();
        assert!(s.samples[len..].iter().all(|&v| v == 0.0), "energy past natural_len");
        assert!(s.samples[len - 64..len].iter().any(|&v| v != 0.0), "dead before natural_len");
    }

    /// The render is the per-atom render the pursuit subtracts, summed — for every kind, and with
    /// atoms hanging off either end.
    #[test]
    fn a_mixed_book_renders_as_the_sum_of_its_atoms() {
        let atoms = [fof_at(-300), gauss_at(700), fof_at(2_000), gauss_at(-50), gauss_at(11_900)];
        let mut b = Book::new(1.0, SR);
        for a in atoms {
            b.selections.push(sel(a));
        }
        let len = 12_000;

        let mut want = vec![0.0f32; len];
        for a in atoms {
            let n = Envelope::render(a.env, SR).unwrap().support_len();
            let mut one = vec![0.0f32; n];
            a.render_into(SR, &mut one);
            crate::signal::add_at(&mut want, &one, a.t0);
        }
        assert_eq!(render_atoms(&b, len).unwrap().samples, want);
        assert_eq!(count_by_kind(&b), vec![(AtomKind::Fof, 2), (AtomKind::Gaussian, 3)]);
    }
}
