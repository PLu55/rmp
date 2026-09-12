//! The end-to-end HRMP gate, which needs both halves of the workspace.
//!
//! It lives here rather than beside the pursuit in `rmp_core::mp` because measuring the energy a
//! book puts into a silent gap means *rendering* the book, and rendering is this crate's job. A
//! dev-dependency from `rmp-core` back onto `rmp-synthesis` would have kept it next to the code it
//! tests, but the types do not unify across such a cycle: the `rmp_core` linked into a lib-test
//! target is a different compilation unit from the one `rmp-synthesis` was built against, so a
//! `Book` made by the test is not the `Book` `render_atoms` accepts.
//!
//! An integration test is the honest place for an assertion that spans the two crates, and it
//! leaves the dependency graph acyclic.

use rmp_core::book::Book;
use rmp_core::dict::{BlockConfig, Dictionary};
use rmp_core::fft::Planner;
use rmp_core::fof::{AtomParams, Envelope, EnvelopeParams};
use rmp_core::hrmp::HrmpConfig;
use rmp_core::mp::{Mp, MpConfig};
use rmp_core::signal::{self, Signal};

const SR: f32 = 48_000.0;

/// The end-to-end case HRMP exists for: energy invented in a gap.
///
/// The dictionary holds only the *long* shape, so ordinary MP has no choice but to explain two
/// separated bursts with an atom that spans the silence between them — the adversarial case the
/// spec describes. (Given the matching short shape, plain MP picks it and never bridges, so the
/// temptation has to be constructed deliberately.) HRMP must refuse to fill the gap.
#[test]
fn hrmp_stops_a_long_atom_inventing_energy_in_a_gap() {
    let mut planner = Planner::new();
    let cfg = BlockConfig { f_min: 300.0, f_max: 4000.0, ..BlockConfig::default() };
    let d = Dictionary::from_grid(&[(80.0, 0.001)], SR, &mut planner, &cfg).unwrap();
    let long_support = d.blocks[0].support_len();

    let f = d.blocks[0].bin_hz((d.blocks[0].k_lo + d.blocks[0].k_hi) / 2);
    let short = EnvelopeParams::new(2147.0, 0.0003);
    let short_len = Envelope::render(short, SR).unwrap().support_len();
    let (t1, t2) = (400i64, 400 + (long_support / 5) as i64);

    let mut sig = Signal::silence(long_support + 8_000, SR);
    for t0 in [t1, t2] {
        let a = AtomParams { t0, f, env: short.into(), phi: 0.4, amp: 1.0 };
        signal::add_at(&mut sig.samples, &a.render(SR).unwrap(), t0);
    }

    // The silence between the bursts, with a margin so neither burst leaks in.
    let gap = (t1 as usize + short_len + 64)..(t2 as usize - 64);
    assert!(gap.end > gap.start, "fixture has no gap");
    assert!(
        signal::energy_of(&sig.samples[gap.clone()]) < 1e-9,
        "the gap must actually be silent"
    );

    let base = MpConfig {
        max_atoms: 8,
        target_snr_db: f32::INFINITY,
        candidate_count: 4,
        ..Default::default()
    };
    let with_hr = MpConfig {
        hrmp: HrmpConfig { enabled: true, ..HrmpConfig::default() },
        ..base
    };

    let mut planner = Planner::new();
    let plain = Mp::new(&d, &sig, &mut planner).run(&base);
    let guarded = Mp::new(&d, &sig, &mut planner).run(&with_hr);
    assert!(!plain.is_empty(), "plain MP selected nothing");

    let gap_energy = |b: &Book| {
        rmp_synthesis::atoms::render_atoms(b, sig.len())
            .map(|r| signal::energy_of(&r.samples[gap.clone()]))
            .unwrap_or(f64::INFINITY)
    };
    let (plain_gap, guarded_gap) = (gap_energy(&plain), gap_energy(&guarded));

    assert!(
        plain_gap > 0.0,
        "the fixture is not adversarial: plain MP put no energy in the gap"
    );
    assert!(
        guarded_gap < 0.5 * plain_gap,
        "HRMP put {guarded_gap:.3e} into a silent gap against plain MP's {plain_gap:.3e}"
    );
    assert!(
        guarded.selections.iter().all(|s| s.hr_score.is_some()),
        "HRMP was enabled but recorded no verdicts"
    );

    // Whatever it selects, the pursuit must still be strictly decreasing.
    for w in guarded.selections.windows(2) {
        assert!(w[1].residual_energy < w[0].residual_energy, "residual rose under HRMP");
    }
}
