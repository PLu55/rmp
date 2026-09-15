//! What Play sounds: a mix of the four things an analysis leaves behind.
//!
//! Built in memory from the finished [`Analysis`] rather than by replaying whatever Synthesize last
//! wrote. That is the whole point of having four switches: the comparisons worth making are between
//! *sources*, and going through a file would mean rendering, naming and saving one before you could
//! hear anything.
//!
//! The four, and why each is worth its switch:
//!
//! - **Origin** — the excerpt as it was read. The reference everything else is judged against.
//! - **Atoms** — what the pursuit selected, through the same per-atom render it subtracted.
//! - **Residual (measured)** — the pursuit's own leftover buffer: literally what the atoms failed
//!   to explain.
//! - **Residual (synthesised)** — the ERB band-power model of that residue, rebuilt.
//!
//! Two combinations are worth naming. *Atoms + residual (measured)* reconstructs the origin
//! exactly, by construction, and hearing it not do so means something is wrong. *Atoms + residual
//! (synthesised)* is the resynthesis — what `rmpsynth` produces, and the thing the whole decomposition
//! is for.
//!
//! Summed at unit gain and not normalised: the levels *are* the result. Scaling the mix to fit
//! would hide exactly the thing being judged — how much energy the atoms took and what is left.

use rmp_core::pipeline::Analysis;
use rmp_core::signal::Signal;

/// Which sources to hear together.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Sources {
    pub origin: bool,
    pub atoms: bool,
    pub residual_measured: bool,
    pub residual_synthesised: bool,
}

impl Default for Sources {
    /// The origin alone: what the file sounds like, before any claim about it.
    fn default() -> Self {
        Self {
            origin: true,
            atoms: false,
            residual_measured: false,
            residual_synthesised: false,
        }
    }
}

impl Sources {
    pub fn any(self) -> bool {
        self.origin || self.atoms || self.residual_measured || self.residual_synthesised
    }
}

/// What a finished analysis can offer, which is not the same as what was ticked.
///
/// The measured residue is kept only if the Analyse panel asked for it, and the synthesised one
/// needs a residual book, so both can be absent from a perfectly good decomposition.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Available {
    pub origin: bool,
    pub atoms: bool,
    pub residual_measured: bool,
    pub residual_synthesised: bool,
}

impl Available {
    pub fn of(analysis: &Analysis, kept_residual: bool) -> Self {
        Self {
            origin: true,
            atoms: !analysis.book.is_empty(),
            residual_measured: kept_residual && !analysis.residual.is_empty(),
            residual_synthesised: residual_book(analysis).is_some(),
        }
    }

    pub fn has(self, which: Which) -> bool {
        match which {
            Which::Origin => self.origin,
            Which::Atoms => self.atoms,
            Which::ResidualMeasured => self.residual_measured,
            Which::ResidualSynthesised => self.residual_synthesised,
        }
    }
}

/// One source, for the checkbox row to iterate over.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Which {
    Origin,
    Atoms,
    ResidualMeasured,
    ResidualSynthesised,
}

impl Which {
    pub const ALL: [Which; 4] =
        [Which::Origin, Which::Atoms, Which::ResidualMeasured, Which::ResidualSynthesised];

    pub fn label(self) -> &'static str {
        match self {
            Which::Origin => "origin",
            Which::Atoms => "atoms",
            Which::ResidualMeasured => "residual (measured)",
            Which::ResidualSynthesised => "residual (synthesised)",
        }
    }

    pub fn why_not(self) -> &'static str {
        match self {
            Which::Origin => "",
            Which::Atoms => "the analysis selected no atoms",
            Which::ResidualMeasured => "the residual was not kept — tick it before analysing",
            Which::ResidualSynthesised => {
                "no residual analysis — tick it before analysing"
            }
        }
    }

    pub fn get(self, s: &Sources) -> bool {
        match self {
            Which::Origin => s.origin,
            Which::Atoms => s.atoms,
            Which::ResidualMeasured => s.residual_measured,
            Which::ResidualSynthesised => s.residual_synthesised,
        }
    }

    pub fn set(self, s: &mut Sources, v: bool) {
        match self {
            Which::Origin => s.origin = v,
            Which::Atoms => s.atoms = v,
            Which::ResidualMeasured => s.residual_measured = v,
            Which::ResidualSynthesised => s.residual_synthesised = v,
        }
    }
}

/// Where an analysis keeps its stochastic model, whichever of the two places that is.
///
/// `pipeline::analyse` returns the residual book *beside* the atom book rather than inside it —
/// whether the two share a file is the caller's decision, not the pipeline's — but a book read back
/// from disk carries its own in `Book::residual`. Anything asking "is there a residual to work
/// with" has to look in both, and this is the single place that does.
///
/// Getting this wrong is not hypothetical: Synthesize checked only `book.residual` and so refused
/// every residual render, including the mixed one, however the analysis had been run.
pub fn residual_book(analysis: &Analysis) -> Option<&rmp_core::residual::ResidualBook> {
    analysis.residual_book.as_ref().or(analysis.book.residual.as_ref())
}

/// Build the mix.
///
/// Everything is aligned to the *excerpt*, sample 0 being the excerpt's first sample, so the four
/// sources line up without consulting `start_sample`. The atom render can run past the excerpt's
/// end — an atom truncated by the analysis is audible again here, about 3% of the energy on a short
/// piano fixture — so the mix is as long as the longest source rather than as long as the origin.
pub fn mix(
    analysis: &Analysis,
    origin: &Signal,
    want: Sources,
) -> Result<Signal, String> {
    let sr = origin.sample_rate;
    let mut parts: Vec<Vec<f32>> = Vec::new();

    if want.origin {
        parts.push(origin.samples.clone());
    }
    if want.atoms {
        let rendered = rmp_synthesis::atoms::render_atoms(&analysis.book, origin.len())
            .map_err(|e| format!("rendering the atoms: {e}"))?;
        parts.push(rendered.samples);
    }
    if want.residual_measured {
        parts.push(analysis.residual.clone());
    }
    if want.residual_synthesised {
        let book = residual_book(analysis).ok_or("this analysis has no residual book")?;
        let rendered =
            rmp_synthesis::render_residual_book(book, &rmp_synthesis::RenderConfig::default())
                .map_err(|e| format!("rendering the residual: {e}"))?;
        parts.push(rendered.samples);
    }

    if parts.is_empty() {
        return Err("nothing selected to play".into());
    }

    let len = parts.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = vec![0.0f32; len];
    for part in &parts {
        for (o, p) in out.iter_mut().zip(part) {
            *o += *p;
        }
    }
    Ok(Signal::new(out, sr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_origin_alone() {
        let s = Sources::default();
        assert!(s.origin && !s.atoms && !s.residual_measured && !s.residual_synthesised);
        assert!(s.any());
    }

    #[test]
    fn nothing_ticked_is_nothing_to_play() {
        let s = Sources {
            origin: false,
            atoms: false,
            residual_measured: false,
            residual_synthesised: false,
        };
        assert!(!s.any());
    }

    /// Every source that can be unavailable says why, or the checkbox is disabled with no
    /// explanation and looks broken.
    #[test]
    fn every_source_that_can_be_missing_explains_itself() {
        for w in Which::ALL {
            if w != Which::Origin {
                assert!(!w.why_not().is_empty(), "{w:?} says nothing about being unavailable");
            }
        }
    }

    #[test]
    fn get_and_set_cover_every_source() {
        let mut s = Sources { origin: false, ..Sources::default() };
        for w in Which::ALL {
            assert!(!w.get(&s), "{w:?} starts off");
            w.set(&mut s, true);
            assert!(w.get(&s), "{w:?} did not take");
        }
        assert!(s.origin && s.atoms && s.residual_measured && s.residual_synthesised);
    }

    /// The trap itself, against a real run: `pipeline::analyse` leaves the residual book *beside*
    /// the atom book, so `Book::residual` is empty even when a residual analysis ran. Anything that
    /// asks the book alone concludes there is no residual — which is exactly what made Synthesize
    /// refuse every residual render.
    #[test]
    fn a_fresh_run_keeps_its_residual_book_outside_the_atom_book() {
        use rmp_core::config::Config;
        use rmp_core::fft::Planner;
        use rmp_core::pipeline::{self, AnalysisRequest};

        let mut cfg = Config::default();
        cfg.dictionary.fof.alphas = vec![256.0];
        cfg.dictionary.fof.betas_ms = vec![1.0];
        cfg.blocks.f_min = 200.0;
        cfg.blocks.f_max = 2000.0;
        cfg.pursuit.max_atoms = 8;
        cfg.refine.enabled = false;
        cfg.residual.enabled = true;

        let samples: Vec<f32> = rmp_core::residual::pseudo_noise(8_000);
        let sig = Signal::new(samples, 48_000.0);
        let residual_cfg = cfg.residual_config(48_000.0).expect("a usable ERB range");
        let mut planner = Planner::new();

        let analysis = pipeline::analyse(
            AnalysisRequest {
                signal: &sig,
                offset: 0,
                config: &cfg,
                residual: Some(&residual_cfg),
            },
            &mut planner,
            &mut (),
        )
        .expect("the fixture decomposes");

        assert!(analysis.residual_book.is_some(), "the run measured a residual");
        assert!(
            analysis.book.residual.is_none(),
            "and left it beside the book — this is the trap, not an accident"
        );
        assert!(residual_book(&analysis).is_some(), "so only looking in both finds it");
        assert!(Available::of(&analysis, true).residual_synthesised);
    }

    /// The mix is the sum at unit gain, and as long as its longest part. Checked with plain
    /// buffers, since what is under test is the arithmetic and not the renderers.
    #[test]
    fn the_mix_sums_at_unit_gain_and_keeps_the_longest_part() {
        let out = sum(&[vec![1.0, 1.0, 1.0], vec![0.5, 0.5]]);
        assert_eq!(out, vec![1.5, 1.5, 1.0], "summed, not averaged, and not truncated");
    }

    /// Lifted from `mix` so the arithmetic can be tested without a book to render.
    fn sum(parts: &[Vec<f32>]) -> Vec<f32> {
        let len = parts.iter().map(Vec::len).max().unwrap_or(0);
        let mut out = vec![0.0f32; len];
        for part in parts {
            for (o, p) in out.iter_mut().zip(part) {
                *o += *p;
            }
        }
        out
    }
}
