//! Every atom, whatever its kind, as the same handful of physically comparable numbers.
//!
//! Everything after this module reads [`AtomObservation`]s and never asks which family an atom came
//! from. So the definitions below have to mean the same thing for a FOF and a Gaussian, and they do
//! because each is a property of the envelope's *energy*, `E²(t)`, not of how the envelope is
//! parameterised:
//!
//! - **energy** — `Selection::energy_removed`, the energy the pursuit measured leaving the residual.
//!   Not `projected_energy`: the book's own rule is that energy is measured, never assumed.
//! - **centre time** — the centroid of `E²(t)`. For a Gaussian that is the peak, `t0 + h`. For a
//!   FOF it is well after the onset, and further the slower the decay.
//! - **effective duration** — twice the RMS width of `E²(t)` about that centroid. For a Gaussian of
//!   standard deviation `s`, `E²` has standard deviation `s/√2`, so the duration is `√2·s`.
//! - **effective bandwidth** — the full −3 dB width, `Shape::bandwidth_hz`: `alpha/π` for a FOF,
//!   `√ln2/(π sigma)` for a Gaussian. The −3 dB width rather than an RMS bandwidth, because a FOF's
//!   spectrum has the Lorentzian skirt of an exponential and its RMS bandwidth is infinite; the −3 dB
//!   width is the one definition the rest of the codebase already uses for both kinds.
//! - **start / end** — the support on the source timeline: `start_sample + t0`, plus
//!   `Shape::approx_support_len`. Clamped at zero, since an atom may begin before its excerpt.
//!
//! # Why a FOF's moments are analytic, not rendered
//!
//! The codebase's rule is to derive a FOF's *support* by rendering, because rfofs's rounding decides
//! where a grain dies and a formula drifts from it. The moments here are a different kind of
//! quantity: a statistical descriptor for a deliberately coarse map, where a sample of rounding is
//! invisible. And rendering would be ruinous — refinement moves every atom's shape continuously, so
//! nothing memoises, and a low-`alpha` support is 332,000 samples. So the moments are integrated
//! from the envelope's definition: a half-cosine attack of `beta` seconds under `exp(−alpha t)`, the
//! decay down to `fade_level`, then a linear fade of `fade_dur`. The decay and fade segments are
//! closed forms and the attack a short numerical integral. The test
//! `fof_moments_match_a_rendered_envelope` holds them to rfofs's own render.

use crate::ids::AtomId;
use rayon::prelude::*;
use rmp_core::book::Book;
use rmp_core::fof::EnvelopeParams;
use rmp_core::gauss::GaussianParams;
use rmp_core::{AtomKind, Shape};

/// One atom, normalised.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomObservation {
    pub atom_id: AtomId,
    /// Which channel the atom was found in. `None` for the mono books rmp writes today; the type
    /// carries it so a multichannel book has somewhere to put it.
    pub channel: Option<u32>,
    /// Energy centroid, absolute samples.
    pub time_center_samples: u64,
    pub start_samples: u64,
    pub end_samples: u64,
    pub frequency_hz: f64,
    pub energy: f64,
    pub effective_bandwidth_hz: f64,
    pub effective_duration_samples: f64,
    pub phase: Option<f64>,
    pub atom_kind: AtomKind,
}

/// The atoms that could not be normalised, and why. A book written by rmp has none; a hand-edited
/// or damaged one might.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Skipped {
    /// `energy_removed` zero, negative or not finite.
    pub no_energy: usize,
    /// Carrier at or below 0 Hz, at or above Nyquist, or not finite.
    pub bad_frequency: usize,
    /// Envelope parameters that do not describe a finite envelope.
    pub bad_shape: usize,
}

impl Skipped {
    pub fn total(&self) -> usize {
        self.no_energy + self.bad_frequency + self.bad_shape
    }
}

/// Centroid and RMS width of `E²`, in seconds from the atom's first sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments {
    pub centroid_s: f64,
    pub rms_width_s: f64,
}

/// A Gaussian's moments, in closed form. The truncation at `cutoff_level` removes energy below
/// −60 dB at the default and is ignored.
pub fn gaussian_moments(g: &GaussianParams, sample_rate: f32) -> Moments {
    let sr = sample_rate as f64;
    Moments {
        centroid_s: g.half_len(sample_rate) as f64 / sr,
        rms_width_s: g.sigma as f64 / std::f64::consts::SQRT_2,
    }
}

/// Points in the attack's numerical integral. The attack is one smooth half-cosine bump, so a
/// midpoint rule at this density is exact to well under a percent of the attack's own moments.
const ATTACK_POINTS: usize = 64;

/// A FOF's moments, from the envelope's definition. See the module docs for why not by rendering.
pub fn fof_moments(p: &EnvelopeParams) -> Option<Moments> {
    let a = p.alpha as f64;
    let beta = (p.beta as f64).max(0.0);
    let fade_level = p.fade_level as f64;
    let fade = (p.fade_dur as f64).max(0.0);
    if !(a > 0.0 && a.is_finite() && fade_level > 0.0 && fade_level < 1.0 && beta.is_finite()) {
        return None;
    }
    // Raw moments of E², sum t^k E(t)^2 dt for k = 0, 1, 2.
    let mut m = [0.0f64; 3];
    let c = 2.0 * a;

    // Attack: E = 0.5 (1 − cos(pi t / beta)) exp(−a t), for 0 <= t < beta.
    if beta > 0.0 {
        let dt = beta / ATTACK_POINTS as f64;
        for i in 0..ATTACK_POINTS {
            let t = (i as f64 + 0.5) * dt;
            let rise = 0.5 * (1.0 - (std::f64::consts::PI * t / beta).cos());
            let e2 = rise * rise * (-c * t).exp() * dt;
            m[0] += e2;
            m[1] += t * e2;
            m[2] += t * t * e2;
        }
    }

    // Decay: E = exp(−a t) from the end of the attack to where it reaches fade_level. rfofs never
    // lets the fade begin before the attack has ended, and neither does this.
    let decay_end = (-fade_level.ln() / a).max(beta);
    if decay_end > beta {
        // Antiderivatives of t^k e^{−ct}.
        let f = |t: f64| {
            let e = (-c * t).exp();
            [
                -e / c,
                -e * (t / c + 1.0 / (c * c)),
                -e * (t * t / c + 2.0 * t / (c * c) + 2.0 / (c * c * c)),
            ]
        };
        let (hi, lo) = (f(decay_end), f(beta));
        for k in 0..3 {
            m[k] += hi[k] - lo[k];
        }
    }

    // Fade: E falls linearly from its value at decay_end to 0 over `fade`.
    if fade > 0.0 {
        let e0 = (-a * decay_end).exp();
        let (d, w) = (decay_end, e0 * e0);
        m[0] += w * fade / 3.0;
        m[1] += w * (d * fade / 3.0 + fade * fade / 12.0);
        m[2] += w * (d * d * fade / 3.0 + d * fade * fade / 6.0 + fade.powi(3) / 30.0);
    }

    // Also rejects NaN.
    if m[0].is_nan() || m[0] <= 0.0 {
        return None;
    }
    let centroid = m[1] / m[0];
    let var = (m[2] / m[0] - centroid * centroid).max(0.0);
    Some(Moments { centroid_s: centroid, rms_width_s: var.sqrt() })
}

/// The moments of either kind.
pub fn moments(shape: &Shape, sample_rate: f32) -> Option<Moments> {
    match shape {
        Shape::Fof(p) => fof_moments(p),
        Shape::Gaussian(g) => {
            if g.sigma > 0.0 && g.sigma.is_finite() && g.cutoff_level > 0.0 && g.cutoff_level < 1.0
            {
                Some(gaussian_moments(g, sample_rate))
            } else {
                None
            }
        }
    }
}

enum Outcome {
    Kept(AtomObservation),
    NoEnergy,
    BadFrequency,
    BadShape,
}

/// Normalise every atom of `book`, in book order.
///
/// Computed in parallel with an indexed collect, so the output order — and everything downstream of
/// it — is the book's order whatever the thread count.
pub fn observe(book: &Book) -> (Vec<AtomObservation>, Skipped) {
    let sr = book.sample_rate;
    let nyquist = sr as f64 / 2.0;
    let origin = book.start_sample as i64;

    let outcomes: Vec<Outcome> = book
        .selections
        .par_iter()
        .enumerate()
        .map(|(i, s)| {
            let energy = s.energy_removed;
            if !(energy > 0.0 && energy.is_finite()) {
                return Outcome::NoEnergy;
            }
            let f = s.atom.f as f64;
            if !(f > 0.0 && f < nyquist) {
                return Outcome::BadFrequency;
            }
            let Some(m) = moments(&s.atom.env, sr) else {
                return Outcome::BadShape;
            };
            let bandwidth = s.atom.env.bandwidth_hz() as f64;
            if !(bandwidth > 0.0 && bandwidth.is_finite()) {
                return Outcome::BadShape;
            }
            let t0 = origin + s.atom.t0;
            let centre = t0 as f64 + m.centroid_s * sr as f64;
            let len = s.atom.env.approx_support_len(sr) as i64;
            Outcome::Kept(AtomObservation {
                atom_id: AtomId(i as u32),
                channel: None,
                time_center_samples: centre.round().max(0.0) as u64,
                start_samples: t0.max(0) as u64,
                end_samples: (t0 + len).max(0) as u64,
                frequency_hz: f,
                energy,
                effective_bandwidth_hz: bandwidth,
                effective_duration_samples: 2.0 * m.rms_width_s * sr as f64,
                phase: Some(s.atom.phi as f64),
                atom_kind: s.atom.kind(),
            })
        })
        .collect();

    let mut skipped = Skipped::default();
    let mut kept = Vec::with_capacity(outcomes.len());
    for o in outcomes {
        match o {
            Outcome::Kept(obs) => kept.push(obs),
            Outcome::NoEnergy => skipped.no_energy += 1,
            Outcome::BadFrequency => skipped.bad_frequency += 1,
            Outcome::BadShape => skipped.bad_shape += 1,
        }
    }
    (kept, skipped)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use rmp_core::book::Selection;
    use rmp_core::fof::{AtomParams, Envelope};

    const SR: f32 = 48_000.0;

    /// Centroid and RMS width of a rendered envelope's energy, in seconds.
    fn rendered_moments(shape: Shape) -> Moments {
        let env = Envelope::render(shape, SR).unwrap();
        let (mut m0, mut m1, mut m2) = (0.0, 0.0, 0.0);
        for (n, &e) in env.samples.iter().enumerate() {
            let (t, e2) = (n as f64 / SR as f64, (e as f64) * (e as f64));
            m0 += e2;
            m1 += t * e2;
            m2 += t * t * e2;
        }
        let c = m1 / m0;
        Moments { centroid_s: c, rms_width_s: (m2 / m0 - c * c).sqrt() }
    }

    /// A book atom with enough filled in for the observation to be about it.
    pub(crate) fn sel(t0: i64, f: f32, env: Shape, energy: f64) -> Selection {
        Selection {
            atom: AtomParams { t0, f, env, phi: 0.25, amp: energy.sqrt() as f32 },
            block: 0,
            onset: t0.max(0) as usize,
            bin: 0,
            projected_energy: energy,
            energy_removed: energy,
            residual_energy: 0.0,
            hr_score: None,
            refined: true,
        }
    }

    /// The oracle for the analytic FOF moments: rfofs's own render, over the range of shapes a
    /// dictionary and refinement produce.
    #[test]
    fn fof_moments_match_a_rendered_envelope() {
        for alpha in [1.0f32, 5.0, 20.0, 80.0, 256.0, 1000.0, 4000.0] {
            for beta_ms in [0.5f32, 1.0, 3.0, 10.0, 50.0] {
                if alpha * beta_ms * 1e-3 > 4.0 {
                    continue;
                }
                let p = EnvelopeParams::new(alpha, beta_ms * 1e-3);
                let want = rendered_moments(p.into());
                let got = fof_moments(&p).unwrap();
                let one_sample = 1.0 / SR as f64;
                let tol_c = 0.02 * want.rms_width_s + 2.0 * one_sample;
                assert!(
                    (got.centroid_s - want.centroid_s).abs() <= tol_c,
                    "alpha {alpha} beta {beta_ms} ms: centroid {} vs rendered {}",
                    got.centroid_s,
                    want.centroid_s
                );
                let tol_w = 0.03 * want.rms_width_s + 2.0 * one_sample;
                assert!(
                    (got.rms_width_s - want.rms_width_s).abs() <= tol_w,
                    "alpha {alpha} beta {beta_ms} ms: rms width {} vs rendered {}",
                    got.rms_width_s,
                    want.rms_width_s
                );
            }
        }
    }

    #[test]
    fn gaussian_moments_match_a_rendered_envelope() {
        for sigma_ms in [0.5f32, 2.0, 7.1, 40.0] {
            let g = GaussianParams::new(sigma_ms * 1e-3);
            let want = rendered_moments(g.into());
            let got = gaussian_moments(&g, SR);
            assert!((got.centroid_s - want.centroid_s).abs() < 1e-9, "sigma {sigma_ms}");
            assert!(
                (got.rms_width_s - want.rms_width_s).abs() < 1e-3 * want.rms_width_s,
                "sigma {sigma_ms}: {} vs {}",
                got.rms_width_s,
                want.rms_width_s
            );
        }
    }

    /// A FOF and a Gaussian chosen to have the same centre, duration and bandwidth come out as the
    /// same observation apart from their kind: later stages cannot tell them apart.
    #[test]
    fn matched_atoms_of_either_kind_give_the_same_observation() {
        // Pick a FOF, then the Gaussian of the same effective duration.
        let fof = EnvelopeParams::new(60.0, 0.004);
        let mf = fof_moments(&fof).unwrap();
        let sigma = (mf.rms_width_s * std::f64::consts::SQRT_2) as f32;
        let gauss = GaussianParams::new(sigma);
        let mg = gaussian_moments(&gauss, SR);
        // Place each so the centroids coincide.
        let t_f = 10_000i64;
        let t_g = t_f + ((mf.centroid_s - mg.centroid_s) * SR as f64).round() as i64;

        let mut book = Book::new(1.0, SR);
        book.selections.push(sel(t_f, 700.0, fof.into(), 0.25));
        book.selections.push(sel(t_g, 700.0, gauss.into(), 0.25));
        let (obs, skipped) = observe(&book);
        assert_eq!(skipped.total(), 0);
        let (a, b) = (&obs[0], &obs[1]);
        assert_eq!((a.atom_kind, b.atom_kind), (AtomKind::Fof, AtomKind::Gaussian));
        assert!(a.time_center_samples.abs_diff(b.time_center_samples) <= 1);
        assert!((a.effective_duration_samples / b.effective_duration_samples - 1.0).abs() < 1e-3);
        assert_eq!((a.frequency_hz, a.energy), (b.frequency_hz, b.energy));
        // The bandwidths are each kind's own -3 dB width and are not matched here; both are finite
        // and in the same few-tens-of-Hz range for a ~10 ms atom.
        assert!(a.effective_bandwidth_hz > 1.0 && b.effective_bandwidth_hz > 1.0);
    }

    #[test]
    fn ids_trace_back_to_the_book_and_bad_atoms_are_counted_not_kept() {
        let g: Shape = GaussianParams::new(0.005).into();
        let mut book = Book::new(1.0, SR);
        book.start_sample = 1000;
        book.selections.push(sel(0, 440.0, g, 0.5));
        book.selections.push(sel(100, 440.0, g, 0.0)); // no energy
        book.selections.push(sel(200, 30_000.0, g, 0.5)); // above Nyquist
        book.selections.push(sel(-5000, 880.0, g, 0.5)); // starts before the file
        let (obs, skipped) = observe(&book);
        assert_eq!(obs.iter().map(|o| o.atom_id.0).collect::<Vec<_>>(), vec![0, 3]);
        assert_eq!((skipped.no_energy, skipped.bad_frequency, skipped.bad_shape), (1, 1, 0));
        assert_eq!(obs[0].start_samples, 1000, "onsets are on the source timeline");
        assert_eq!(obs[1].start_samples, 0, "clamped, not wrapped");
        assert_eq!(obs[0].phase, Some(0.25));
        assert_eq!(obs[0].channel, None);
    }
}
