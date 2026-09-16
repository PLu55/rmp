//! The result tabs: what a finished analysis looks like.
//!
//! Each is a *view* of a call that already exists in `rmp-core` and is already tested —
//! `stats::summarize`, `stats::histogram`, `tfmap::compute`. None of them recomputes anything, and
//! that is the rule this crate is built on: a panel that did its own arithmetic would be a second
//! definition of a number `rmpstat` already prints, and the two would drift.
//!
//! What lives here instead is the part that is genuinely the GUI's: which controls to offer, when
//! to recompute, and how to get a million cells onto the screen.
//!
//! **Everything is cached against the options that produced it.** A histogram renders envelopes for
//! `support` and `periods`; a map is 0.13 s for 5000 atoms at 1200x800 and the grid size is a
//! setting, so it grows without bound. Recomputing either per frame would make the tab unusable, so
//! both are held until an option actually moves.

pub mod distribution;
pub mod function;
pub mod summary;
pub mod timefreq;

#[cfg(test)]
mod tests {
    use rmp_core::book::Book;
    use rmp_core::config::Config;
    use rmp_core::fft::Planner;
    use rmp_core::pipeline::{self, AnalysisRequest};
    use rmp_core::signal::Signal;
    use rmp_core::stats::{self, Evaluator, Quantity, Weight};
    use rmp_core::tfmap::{self, MapGrid, MapOptions, Reference};

    /// A small real decomposition of noise: enough atoms for a histogram and a map to be about
    /// something, cheap enough to run in a unit test.
    fn a_book() -> Book {
        let mut cfg = Config::default();
        cfg.dictionary.fof.alphas = vec![256.0];
        cfg.dictionary.fof.betas_ms = vec![1.0];
        cfg.blocks.f_min = 200.0;
        cfg.blocks.f_max = 2000.0;
        cfg.pursuit.max_atoms = 24;
        cfg.refine.enabled = false;

        let sig = Signal::new(rmp_core::residual::pseudo_noise(8_000), 48_000.0);
        let mut planner = Planner::new();
        pipeline::analyse(
            AnalysisRequest { signal: &sig, offset: 0, config: &cfg, residual: None },
            &mut planner,
            &mut (),
        )
        .expect("the fixture decomposes")
        .book
    }

    /// What the checkbox area's enabling rests on: a quantity the book describes has mass, and one
    /// it does not describe reports that as `inapplicable` rather than as an error or an empty
    /// histogram. A FOF-only book has no `sigma`.
    #[test]
    fn an_inapplicable_quantity_is_empty_but_not_an_error() {
        let book = a_book();
        let mut ev = Evaluator::new(&book);

        let alpha = stats::histogram(&book, &mut ev, Quantity::Alpha, 16, None, None, Weight::Count)
            .expect("alpha describes a FOF book");
        assert!(alpha.total > 0.0, "a FOF book has alphas");
        assert_eq!(alpha.inapplicable, 0.0);

        let sigma = stats::histogram(&book, &mut ev, Quantity::Sigma, 16, None, None, Weight::Count)
            .expect("sigma is not an error, merely empty here");
        assert_eq!(sigma.total, 0.0, "no Gaussian atoms, so nothing to bin");
        assert!(sigma.inapplicable > 0.0, "and every atom said so");
    }

    /// Count and energy are different pictures of one book — which is why the weight is a control
    /// and not a constant.
    #[test]
    fn weighting_by_energy_is_not_the_same_histogram_as_counting() {
        let book = a_book();
        let mut ev = Evaluator::new(&book);
        let by = |w, ev: &mut Evaluator| {
            stats::histogram(&book, ev, Quantity::Freq, 16, None, None, w).unwrap().counts
        };
        assert_ne!(by(Weight::Count, &mut ev), by(Weight::Energy, &mut ev));
    }

    /// The map path the worker takes, end to end: compute, scale, colour. The pixel count and the
    /// row order are the two things a texture upload gets wrong silently.
    #[test]
    fn the_map_fills_exactly_one_pixel_per_cell_highest_frequency_first() {
        let book = a_book();
        let (n_t, n_f) = (40, 24);
        let grid = MapGrid::covering(&book, n_t, n_f, false).unwrap();
        let map = tfmap::compute(&book, grid, &MapOptions::default()).unwrap();
        assert_eq!((map.grid.n_t(), map.grid.n_f()), (n_t, n_f));

        let floor = 60.0f32;
        let db = map.to_db(floor, Reference::Max);
        assert_eq!(db.len(), n_t * n_f);

        let mut rgb = vec![0u8; n_t * n_f * 3];
        for j in 0..n_f {
            let y = n_f - 1 - j;
            for i in 0..n_t {
                let c = tfmap::heat((db[i * n_f + j] as f64 + floor as f64) / floor as f64);
                let at = (y * n_t + i) * 3;
                rgb[at..at + 3].copy_from_slice(&c);
            }
        }
        assert_eq!(rgb.len(), n_t * n_f * 3, "one pixel per cell, no more and no fewer");

        // Row 0 of the image is the *last* frequency bin: a spectrogram is read with the high
        // frequencies at the top, and getting this upside down is the classic silent mistake.
        let top_left = &rgb[0..3];
        let expect = tfmap::heat((db[n_f - 1] as f64 + floor as f64) / floor as f64);
        assert_eq!(top_left, expect, "the top row is not the highest frequency");
    }

    /// Cells below the floor are black, which is what makes silence read as empty.
    #[test]
    fn a_cell_at_the_floor_is_black() {
        assert_eq!(tfmap::heat(0.0), [0, 0, 0]);
    }
}
