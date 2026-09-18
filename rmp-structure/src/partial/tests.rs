//! Partial extraction on synthetic books with known answers (§45, §46, §48).
//!
//! Books are built directly from `Selection`s — no pursuit runs — so each fixture says exactly
//! which atoms exist, and the test is about structure rather than about what MP would have chosen.

use super::*;
use crate::observation::tests::sel;
use crate::observation::fof_moments;
use rmp_core::Shape;
use rmp_core::fof::EnvelopeParams;
use rmp_core::gauss::GaussianParams;

const SR: f32 = 48_000.0;

fn gaussian(sigma_ms: f32) -> Shape {
    GaussianParams::new(sigma_ms * 1e-3).into()
}

/// Gaussian atoms every `spacing` samples from `t0` to `t1` with each atom's centre at `t` carrying
/// frequency `f(t)` and energy `e(t)`. `t0` of each atom is chosen so its *centre* lands at `t`.
fn tone(
    book: &mut Book,
    t0: i64,
    t1: i64,
    spacing: i64,
    shape: Shape,
    f: impl Fn(f64) -> f64,
    e: impl Fn(f64) -> f64,
) {
    let lead = match shape {
        Shape::Gaussian(g) => g.half_len(SR) as i64,
        Shape::Fof(p) => (fof_moments(&p).unwrap().centroid_s * SR as f64).round() as i64,
    };
    let mut t = t0;
    while t <= t1 {
        let secs = t as f64 / SR as f64;
        book.selections.push(sel(t - lead, f(secs) as f32, shape, e(secs)));
        t += spacing;
    }
}

fn run(book: &Book) -> PartialAnalysis {
    analyze_partials(book, &PartialAnalysisConfig::default()).unwrap()
}

fn cents_between(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2().abs()
}

/// Acceptance 4, 5 and 8: a steady sinusoid spread over many atoms becomes one partial, with a
/// trajectory reduced to a few points and every atom traced back to it.
#[test]
fn a_steady_sinusoid_is_one_partial_with_its_atoms_behind_it() {
    let mut book = Book::new(60.0, SR);
    tone(&mut book, 4800, 52_800, 800, gaussian(5.0), |_| 440.0, |_| 1.0);
    assert_eq!(book.selections.len(), 61);
    let a = run(&book);
    assert_eq!(a.book.partials.len(), 1, "{:?}", a.diagnostics);
    let p = &a.book.partials[0];
    assert!(cents_between(p.mean_frequency_hz, 440.0) < 2.0, "{}", p.mean_frequency_hz);
    assert!(p.frequency.points.len() <= 3, "{:?}", p.frequency);
    assert!(p.amplitude.points.len() <= 12, "{:?}", p.amplitude);
    assert!(p.supporting_atoms.len() >= 55, "{}", p.supporting_atoms.len());
    assert!(p.supporting_atoms.windows(2).all(|w| w[0] < w[1]));
    assert!((p.persistence - 1.0).abs() < 1e-12);
    // About one second long, give or take the kernel's reach at each end.
    let secs = p.duration_samples() as f64 / SR as f64;
    assert!((0.95..1.15).contains(&secs), "{secs}");
    assert!((p.significance - p.normalized_energy * p.persistence).abs() < 1e-15);
}

/// Acceptance 6: a partial 40 dB below its neighbour, none of whose atoms would pass a threshold
/// set anywhere near the loud one, survives because it persists.
#[test]
fn a_weak_persistent_partial_survives() {
    let mut book = Book::new(1.0, SR);
    tone(&mut book, 4800, 52_800, 800, gaussian(5.0), |_| 440.0, |_| 1.0);
    tone(&mut book, 4800, 52_800, 800, gaussian(5.0), |_| 1500.0, |_| 1e-4);
    let a = run(&book);
    let f: Vec<f64> = a.book.partials.iter().map(|p| p.mean_frequency_hz).collect();
    assert_eq!(f.len(), 2, "{f:?}");
    assert!(f.iter().any(|&f| cents_between(f, 1500.0) < 5.0), "{f:?}");
}

/// Acceptance 7: one loud, short atom is a transient, not a partial.
#[test]
fn a_strong_isolated_atom_is_not_a_partial() {
    let mut book = Book::new(1.0, SR);
    book.selections.push(sel(24_000, 2000.0, gaussian(3.0), 100.0));
    let a = run(&book);
    assert!(a.book.partials.is_empty(), "{:?}", a.book.partials);
    assert!(a.diagnostics.rejected_short >= 1, "{:?}", a.diagnostics);
    assert_eq!(a.diagnostics.unsupported_atoms, 1);
}

/// The same, beside a steady tone: the transient neither becomes a partial nor disturbs the tone.
#[test]
fn a_loud_transient_beside_a_tone_leaves_one_partial() {
    let mut book = Book::new(1.0, SR);
    tone(&mut book, 4800, 52_800, 800, gaussian(5.0), |_| 440.0, |_| 1.0);
    book.selections.push(sel(24_000, 2000.0, gaussian(3.0), 30.0));
    let a = run(&book);
    assert_eq!(a.book.partials.len(), 1, "{:?}", a.diagnostics);
    assert!(cents_between(a.book.partials[0].mean_frequency_hz, 440.0) < 2.0);
}

/// §46: a strong vibrato stays one partial rather than fragmenting into neighbouring tracks, and
/// its simplified trajectory still carries the vibrato.
#[test]
fn a_vibrato_is_one_partial() {
    let mut book = Book::new(1.0, SR);
    let vib = |t: f64| 440.0 * (80.0 * (std::f64::consts::TAU * 5.5 * t).sin() / 1200.0).exp2();
    tone(&mut book, 4800, 100_800, 240, gaussian(5.0), vib, |_| 1.0);
    let a = run(&book);
    assert_eq!(a.book.partials.len(), 1, "{:?}", a.diagnostics);
    let p = &a.book.partials[0];
    let (lo, hi) = p.frequency.points.iter().fold((f64::MAX, 0.0f64), |(lo, hi), q| {
        (lo.min(q.value), hi.max(q.value))
    });
    // The analysis blurs the vibrato in time, so the swing is somewhat less than ±80 cents.
    assert!(cents_between(hi, lo) > 80.0, "the vibrato was smoothed away: {lo}..{hi}");
    assert!(p.frequency.points.len() > 10, "{}", p.frequency.points.len());
}

/// §46: short holes in the MP representation do not split a continuous partial.
#[test]
fn short_missing_sections_do_not_split_a_partial() {
    let mut book = Book::new(1.0, SR);
    tone(&mut book, 4800, 52_800, 480, gaussian(3.0), |_| 660.0, |_| 1.0);
    // Two holes of 40 ms.
    let holes = [(20_000i64, 21_920i64), (36_000, 37_920)];
    book.selections.retain(|s| {
        let c = s.atom.t0 + 400;
        !holes.iter().any(|&(a, b)| (a..b).contains(&c))
    });
    let a = run(&book);
    assert_eq!(a.book.partials.len(), 1, "{:?}", a.diagnostics);
}

/// §46: two glides crossing in frequency keep their own directions through the crossing.
#[test]
fn crossing_partials_keep_their_identities() {
    let mut book = Book::new(1.0, SR);
    let up = |t: f64| 400.0 * ((t - 0.1) / 2.0).exp2();
    let down = |t: f64| 800.0 * (-(t - 0.1) / 2.0).exp2();
    tone(&mut book, 4800, 100_800, 480, gaussian(20.0), up, |_| 1.0);
    tone(&mut book, 4800, 100_800, 480, gaussian(20.0), down, |_| 1.0);
    let a = run(&book);
    let long: Vec<&Partial> = a
        .book
        .partials
        .iter()
        .filter(|p| p.duration_samples() as f64 > 0.8 * 96_000.0)
        .collect();
    assert_eq!(long.len(), 2, "{:#?}", a.book.partials);
    for p in long {
        let (first, last) = (p.frequency.points[0].value, p.frequency.points.last().unwrap().value);
        assert!(
            cents_between(first, last) > 900.0,
            "partial {:?} went {first:.0} → {last:.0} Hz: it swapped at the crossing",
            p.id
        );
    }
}

/// Legato on a fixed-pitch instrument: A5 decaying while A♭5 rises under it. 3 ms atoms are wide
/// enough that the two notes' kernels merge into one peak that slides from one note to the other.
/// With the drift limit at half a semitone, what comes out is notes, not a glissando.
#[test]
fn overlapping_notes_are_not_joined_into_a_glide_when_drift_is_limited() {
    let mut book = Book::new(1.0, SR);
    tone(&mut book, 4800, 48_000, 240, gaussian(3.0), |_| 880.0, |t| (-4.0 * t).exp());
    tone(&mut book, 28_800, 72_000, 240, gaussian(3.0), |_| 830.6, |t| {
        (1.0 - (-8.0 * (t - 0.6)).exp()).max(0.0)
    });
    let widest = |cfg: &PartialAnalysisConfig| {
        let a = analyze_partials(&book, cfg).unwrap();
        a.book.partials.iter().map(|p| p.frequency_std_cents).fold(0.0, f64::max)
    };
    let free = widest(&PartialAnalysisConfig::default());
    assert!(free > 20.0, "the premise: without a limit the notes slide into each other ({free})");
    let piano = PartialAnalysisConfig { max_drift_cents: 50.0, ..Default::default() };
    let limited = widest(&piano);
    assert!(limited < 15.0, "a partial still glides by {limited} cents");
}

/// §45 atom families: the same tone written in FOF atoms and in Gaussian atoms gives the same
/// partial, near enough that later stages could not tell which family it came from.
#[test]
fn fof_and_gaussian_renderings_of_one_tone_give_equivalent_partials() {
    let fof = EnvelopeParams::new(60.0, 0.003);
    let m = fof_moments(&fof).unwrap();
    let sigma_ms = (m.rms_width_s * std::f64::consts::SQRT_2 * 1e3) as f32;

    let mut a = Book::new(1.0, SR);
    tone(&mut a, 4800, 52_800, 960, fof.into(), |_| 523.25, |_| 1.0);
    let mut b = Book::new(1.0, SR);
    tone(&mut b, 4800, 52_800, 960, gaussian(sigma_ms), |_| 523.25, |_| 1.0);

    let (pa, pb) = (run(&a), run(&b));
    assert_eq!((pa.book.partials.len(), pb.book.partials.len()), (1, 1));
    let (x, y) = (&pa.book.partials[0], &pb.book.partials[0]);
    assert!(cents_between(x.mean_frequency_hz, y.mean_frequency_hz) < 5.0);
    let frame = 480;
    assert!(x.start_samples.abs_diff(y.start_samples) <= 2 * frame);
    assert!(x.end_samples.abs_diff(y.end_samples) <= 2 * frame);
    assert_eq!(x.supporting_atoms.len(), y.supporting_atoms.len());
}

/// Three decaying harmonic tones entering one after another. The atoms are 20 ms Gaussians, 13 Hz
/// wide, which is what a pursuit picks for steady harmonics; much shorter ones are too wide to
/// resolve 990 Hz from 1046 Hz at all.
fn polyphony() -> Book {
    let mut book = Book::new(1.0, SR);
    for (i, f0) in [220.0, 330.0, 523.0].into_iter().enumerate() {
        for h in 1..=4 {
            let decay = 1.5 + i as f64;
            tone(
                &mut book,
                4800 + 9600 * i as i64,
                96_000,
                960,
                gaussian(20.0),
                move |_| f0 * h as f64,
                move |t| (-decay * t).exp() / h as f64,
            );
        }
    }
    book
}

/// §41: the same book and settings give the same result, whatever the thread count.
#[test]
fn analysis_is_deterministic_across_thread_counts() {
    let book = polyphony();
    let cfg = PartialAnalysisConfig { keep_intermediates: true, ..Default::default() };
    let at = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(|| analyze_partials(&book, &cfg).unwrap())
    };
    let one = at(1);
    assert!(one.book.partials.len() >= 10, "{:?}", one.diagnostics);
    assert_eq!(one, at(5));
    assert_eq!(one, at(16));
}

/// Every partial's ids and support obey the rules the book's readers rely on.
fn check_invariants(a: &PartialAnalysis) {
    let partials = &a.book.partials;
    a.book.validate().unwrap();
    let mut seen = std::collections::HashSet::new();
    for w in partials.windows(2) {
        assert!(w[0].start_samples <= w[1].start_samples, "ids not in start order");
    }
    for p in partials {
        assert!(p.persistence > 0.0 && p.persistence <= 1.0, "{}", p.persistence);
        assert!(p.end_samples >= p.start_samples);
        assert!(!p.frequency.points.is_empty() && !p.amplitude.points.is_empty());
        assert_eq!(p.frequency.start(), Some(p.start_samples));
        assert_eq!(p.frequency.end(), Some(p.end_samples));
        for id in &p.supporting_atoms {
            assert!(seen.insert(*id), "atom {id:?} supports two partials");
        }
        assert!(p.normalized_energy >= 0.0);
    }
    let total: f64 = partials.iter().map(|p| p.normalized_energy).sum();
    assert!(total <= 1.0 + 1e-9, "{total}");
    let d = &a.diagnostics;
    assert_eq!(d.accepted_partials, partials.len());
    assert_eq!(d.candidate_ridges, d.accepted_partials + d.rejected_short + d.rejected_sparse);
    assert_eq!(d.observations + d.skipped.total(), d.input_atoms);
    assert_eq!(d.supporting_atoms + d.unsupported_atoms, d.observations);
}

#[test]
fn polyphony_gives_one_partial_per_harmonic() {
    let a = run(&polyphony());
    check_invariants(&a);
    // Twelve harmonics, two of which coincide (330×2 ≈ 220×3 at 660 Hz): at least eleven distinct
    // long partials.
    let long = a.book.partials.iter().filter(|p| p.duration_samples() > 48_000).count();
    assert!(long >= 11, "{long}: {:?}", a.diagnostics);
}

/// §45 regression: a real 5000-atom book, held to structural invariants rather than to values.
#[test]
fn a_real_book_obeys_the_invariants() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/books/book1.json");
    let book = rmp_core::book::read(&path).unwrap();
    let a = run(&book);
    check_invariants(&a);
    let n = a.book.partials.len();
    assert!((5..2000).contains(&n), "{n} partials from {} atoms", book.selections.len());
    assert!(a.diagnostics.supporting_atoms > 0);
}

#[test]
fn an_empty_book_has_no_partials_and_is_not_an_error() {
    let a = run(&Book::new(0.0, SR));
    assert!(a.book.partials.is_empty());
    check_invariants(&a);
}

#[test]
fn a_partial_book_round_trips_in_every_format() {
    let dir = std::env::temp_dir().join(format!("rmp-structure-io-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut a = run(&polyphony()).book;
    a.metadata.source_book = Some("poly.json.gz".into());
    for name in ["p.partials.json", "p.partials.json.gz", "p.partials.toml"] {
        let path = dir.join(name);
        a.write(&path).unwrap();
        assert_eq!(PartialBook::read(&path).unwrap(), a, "{name}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

