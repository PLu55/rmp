//! The atom: an envelope shape, a carrier, and a placement.
//!
//! A dictionary holds several *kinds* of atom. Every kind is `amp * E(n) * sin(phi + omega n)` with
//! an envelope `E` that depends on the shape parameters alone — never on `f` or `phi` — which is the
//! one property the whole engine needs: one envelope-windowed FFT per frame still serves every
//! frequency, the Gram still comes from the spectrum of `E^2`, and the hop is still measured from
//! `E`'s own autocorrelation. So the pursuit, the correlator, the fit and the lazy update are
//! written against a rendered [`Envelope`] and do not know which kind they are looking at.
//!
//! What does differ is confined to [`Shape`]:
//!
//! - **FOF** ([`crate::fof`]): rendered through rfofs, which is also the synthesizer that replays
//!   it, so its envelope and support are obtained by rendering a probe grain.
//! - **Gaussian** ([`crate::gauss`]): defined by rmp itself, so its envelope is the formula.
//!
//! # A book stores the shape untagged
//!
//! `Shape` serialises as its inner struct, with no tag, so a book that holds only FOFs is byte for
//! byte what it was before there was more than one kind, and every book ever written still reads.
//! The variants are told apart by their fields — `alpha`/`beta`/`fade_*` against
//! `sigma`/`cutoff_level` — and [`GaussianParams`]'s `deny_unknown_fields` keeps that unambiguous.

use crate::fof::{EnvelopeParams, FofError};
use crate::gauss::GaussianParams;
use rfofs::fof::FofParams;

/// Which family an atom belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AtomKind {
    Fof,
    Gaussian,
}

impl AtomKind {
    pub const ALL: [AtomKind; 2] = [AtomKind::Fof, AtomKind::Gaussian];

    pub fn name(self) -> &'static str {
        match self {
            Self::Fof => "fof",
            Self::Gaussian => "gaussian",
        }
    }
}

impl std::fmt::Display for AtomKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// An envelope shape of any kind.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Shape {
    Fof(EnvelopeParams),
    Gaussian(GaussianParams),
}

impl From<EnvelopeParams> for Shape {
    fn from(p: EnvelopeParams) -> Self {
        Self::Fof(p)
    }
}

impl From<GaussianParams> for Shape {
    fn from(p: GaussianParams) -> Self {
        Self::Gaussian(p)
    }
}

impl Shape {
    pub fn kind(&self) -> AtomKind {
        match self {
            Self::Fof(_) => AtomKind::Fof,
            Self::Gaussian(_) => AtomKind::Gaussian,
        }
    }

    pub fn as_fof(&self) -> Option<EnvelopeParams> {
        match *self {
            Self::Fof(p) => Some(p),
            Self::Gaussian(_) => None,
        }
    }

    pub fn as_gaussian(&self) -> Option<GaussianParams> {
        match *self {
            Self::Gaussian(p) => Some(p),
            Self::Fof(_) => None,
        }
    }

    /// Exact-bits memo key for a rendered envelope.
    ///
    /// For a FOF this is `(alpha, beta)` and nothing else — the key every memo in the crate used
    /// before there was a second kind, kept exactly so the refinement cache's hit pattern, and with
    /// it every book, is unchanged. The release is fixed for a whole analysis, so it does not need to
    /// be in the key.
    pub fn cache_key(&self) -> (u8, u32, u32) {
        match self {
            Self::Fof(p) => (0, p.alpha.to_bits(), p.beta.to_bits()),
            Self::Gaussian(g) => (1, g.sigma.to_bits(), g.cutoff_level.to_bits()),
        }
    }

    /// The full −3 dB bandwidth in Hz: `alpha / pi` for a FOF, `sqrt(ln 2) / (pi sigma)` for a
    /// Gaussian.
    pub fn bandwidth_hz(&self) -> f32 {
        match self {
            Self::Fof(p) => p.alpha / std::f32::consts::PI,
            Self::Gaussian(g) => g.bandwidth_hz(),
        }
    }

    /// Support length without rendering: exact for a Gaussian, an estimate for a FOF.
    ///
    /// The FOF formula is `max(ln(1/fade_level) * sr / alpha, beta * sr)`, which ignores the release
    /// and rfofs's rounding; it agrees with a render within a factor of 0.8..1.6. Use
    /// [`Envelope::render`] where the length has to be right.
    pub fn approx_support_len(&self, sample_rate: f32) -> usize {
        match self {
            Self::Fof(p) => {
                let n = -(p.fade_level as f64).ln() * sample_rate as f64 / p.alpha as f64;
                let attack = (p.beta * sample_rate).ceil() as f64;
                n.max(attack).ceil().max(1.0) as usize
            }
            Self::Gaussian(g) => g.support_len(sample_rate),
        }
    }

    /// A short human-readable description, for reports.
    pub fn describe(&self) -> String {
        match self {
            Self::Fof(p) => format!("fof alpha {:.3} beta {:.2} ms", p.alpha, p.beta * 1e3),
            Self::Gaussian(g) => format!("gaussian sigma {:.2} ms", g.sigma * 1e3),
        }
    }
}

/// A complete atom: envelope, carrier and placement.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AtomParams {
    /// First sample of the support, relative to the analysed signal's origin. Signed so an atom may
    /// start before the excerpt.
    pub t0: i64,
    /// Carrier frequency, Hz.
    pub f: f32,
    pub env: Shape,
    /// Carrier phase, radians, referenced to `t0`: `sin(phi + 2*pi*f*(t - t0)/sr)`.
    pub phi: f32,
    /// Linear amplitude. Equals the fitted coefficient against the rendered envelope.
    pub amp: f32,
}

impl AtomParams {
    pub fn kind(&self) -> AtomKind {
        self.env.kind()
    }

    /// Convert to rfofs parameters for rendering at absolute sample `origin + t0`, or `None` for an
    /// atom rfofs cannot render.
    ///
    /// Panics if the resulting onset is negative — rfofs's `start_sample` is unsigned.
    pub fn to_fof_params(&self, origin: u64) -> Option<FofParams> {
        let env = self.env.as_fof()?;
        let start = origin as i64 + self.t0;
        assert!(start >= 0, "atom onset {start} precedes the sample clock origin");
        Some(crate::fof::fof_params(&env, start as u64, self.f, self.phi, self.amp))
    }

    /// Render this atom in isolation, starting at index 0 of `buf`. `buf` is **overwritten**.
    ///
    /// A FOF renders through rfofs in a single `fill_block` call — see [`crate::fof`] for why that
    /// matters. This is the render the pursuit subtracts, so it is also the one resynthesis must
    /// use.
    pub fn render_into(&self, sample_rate: f32, buf: &mut [f32]) {
        match &self.env {
            Shape::Fof(p) => {
                crate::fof::render_atom_into(p, self.f, self.phi, self.amp, sample_rate, buf)
            }
            Shape::Gaussian(g) => g.render_atom_into(self.f, self.phi, self.amp, sample_rate, buf),
        }
    }

    /// Render this atom into a fresh buffer sized to its support.
    pub fn render(&self, sample_rate: f32) -> Result<Vec<f32>, FofError> {
        let len = match &self.env {
            Shape::Fof(p) => crate::fof::render_probe(*p, sample_rate)?.len(),
            Shape::Gaussian(g) => {
                g.validate(sample_rate)?;
                g.support_len(sample_rate)
            }
        };
        let mut buf = vec![0.0f32; len];
        self.render_into(sample_rate, &mut buf);
        Ok(buf)
    }
}

/// A block's envelope, obtained by rendering.
#[derive(Clone, Debug)]
pub struct Envelope {
    pub params: Shape,
    pub sample_rate: f32,
    /// The envelope, truncated to its support. `samples.len() == support_len`.
    pub samples: Vec<f32>,
    /// `sum(E^2)`, accumulated in f64.
    pub energy: f64,
}

impl Envelope {
    /// Render the envelope of any shape.
    pub fn render(params: impl Into<Shape>, sample_rate: f32) -> Result<Self, FofError> {
        let params = params.into();
        let samples = match &params {
            Shape::Fof(p) => crate::fof::render_probe(*p, sample_rate)?,
            Shape::Gaussian(g) => g.render_envelope(sample_rate)?,
        };
        let energy = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
        Ok(Self {
            params,
            sample_rate,
            samples,
            energy,
        })
    }

    pub fn support_len(&self) -> usize {
        self.samples.len()
    }

    pub fn kind(&self) -> AtomKind {
        self.params.kind()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    #[test]
    fn a_fof_shape_serialises_exactly_as_its_envelope_did() {
        let p = EnvelopeParams::new(251.0, 0.002);
        let shape: Shape = p.into();
        assert_eq!(serde_json::to_string(&shape).unwrap(), serde_json::to_string(&p).unwrap());
        assert_eq!(toml::to_string(&shape).unwrap(), toml::to_string(&p).unwrap());
    }

    #[test]
    fn both_kinds_round_trip_and_never_read_as_each_other() {
        for shape in [
            Shape::from(EnvelopeParams::new(80.0, 0.001)),
            Shape::from(GaussianParams { sigma: 0.0123, cutoff_level: 0.01 }),
        ] {
            let json = serde_json::to_string(&shape).unwrap();
            assert_eq!(serde_json::from_str::<Shape>(&json).unwrap(), shape, "{json}");
            let doc = toml::to_string(&shape).unwrap();
            assert_eq!(toml::from_str::<Shape>(&doc).unwrap(), shape, "{doc}");
        }
        // A document with fields from both is neither.
        let mixed = r#"{"sigma":0.01,"cutoff_level":0.001,"alpha":80.0}"#;
        assert!(serde_json::from_str::<Shape>(mixed).is_err());
    }

    #[test]
    fn a_gaussian_atom_is_its_envelope_times_the_carrier() {
        let g = GaussianParams::new(0.003);
        let env = Envelope::render(g, SR).unwrap();
        let atom = AtomParams { t0: 0, f: 912.5, env: g.into(), phi: -0.7, amp: 0.4 };
        let got = atom.render(SR).unwrap();
        assert_eq!(got.len(), env.support_len());
        let omega = std::f64::consts::TAU * 912.5 / SR as f64;
        for (n, (&y, &e)) in got.iter().zip(&env.samples).enumerate() {
            let want = 0.4 * e as f64 * (-0.7 + omega * n as f64).sin();
            assert!((y as f64 - want).abs() < 1e-6, "n={n}: {y} vs {want}");
        }
        assert!(atom.to_fof_params(0).is_none(), "rfofs cannot render a gaussian");
    }

    #[test]
    fn a_fof_atom_still_converts_to_rfofs_parameters() {
        let atom = AtomParams {
            t0: 120,
            f: 440.0,
            env: EnvelopeParams::new(251.0, 0.002).into(),
            phi: 0.5,
            amp: 0.25,
        };
        let p = atom.to_fof_params(1000).unwrap();
        assert_eq!(p.start_sample, 1120);
        assert_eq!((p.alpha, p.beta, p.amp), (251.0, 0.002, 0.25));
    }

    #[test]
    fn the_approximate_support_is_exact_for_a_gaussian() {
        let g = Shape::from(GaussianParams::new(0.0071));
        assert_eq!(g.approx_support_len(SR), Envelope::render(g, SR).unwrap().support_len());
    }
}
