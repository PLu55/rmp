//! Per-stage cost of one decomposition.
//!
//! `examples/analyze` reports what a whole run costs; this splits that between the stages so a
//! change can be attributed. The arms are chosen to isolate one thing each:
//!
//! - `init` — correlate every frame of every block once, the fixed startup cost;
//! - `grid` — the plain pursuit, whose per-atom cost is dominated by `refresh_stale`;
//! - `refined` — the same with off-grid refinement, to price it against that dominant term;
//! - `hrmp` — the same with local-support validation on top;
//! - `full_update` — recomputing every frame each iteration, the reference the incremental path is
//!   validated against and the measure of what the local update buys.
//!
//! Every arm rebuilds the `Mp` state, so each includes one `init`. Per-*atom* cost is therefore
//! `(arm - init) / atoms`, and the arms select different numbers of atoms — criterion's raw
//! per-iteration times are not comparable between them. The atom counts are printed alongside so
//! the division can be done honestly.
//!
//! `CLAUDE.md`'s measurement discipline applies: run-to-run spread on this machine reaches ~5%, so
//! treat smaller differences as unresolved.

use criterion::{Criterion, criterion_group};
use rmp_core::dict::{BlockConfig, Dictionary};
use rmp_core::fft::Planner;
use rmp_core::fof::AtomParams;
use rmp_core::hrmp::HrmpConfig;
use rmp_core::mp::{Mp, MpConfig};
use rmp_core::refine::RefineConfig;
use rmp_core::signal::Signal;

const SR: f32 = 48_000.0;
/// Short enough to bench repeatedly, long enough that the local update has somewhere to be local.
const SECONDS: f32 = 0.25;
const MAX_ATOMS: usize = 60;

/// Atoms deliberately off the frequency and onset grids — the case refinement exists for, and the
/// only honest one to price it on.
fn planted(dict: &Dictionary, len: usize) -> Signal {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 16_777_216.0
    };
    let atoms: Vec<AtomParams> = (0..12)
        .map(|_| {
            let b = &dict.blocks[(rnd() * dict.blocks.len() as f32) as usize % dict.blocks.len()];
            let k = b.k_lo + (rnd() * (b.k_hi - b.k_lo) as f32) as usize;
            AtomParams {
                t0: ((rnd() * len as f32) as usize + b.hop / 2 + 1) as i64,
                f: b.bin_hz(k) + 0.5 * SR / b.fft_len as f32,
                env: b.env.params,
                phi: rnd() * std::f32::consts::TAU,
                amp: 0.3 + 0.7 * rnd(),
            }
        })
        .collect();
    Signal::from_atoms(&atoms, len, SR).unwrap()
}

fn bench_pursuit(c: &mut Criterion) {
    let mut planner = Planner::new();
    let dict = Dictionary::voice(SR, &mut planner, &BlockConfig::default()).unwrap();
    let len = (SECONDS * SR) as usize;
    let signal = planted(&dict, len);

    let grid = MpConfig {
        max_atoms: MAX_ATOMS,
        target_snr_db: f32::INFINITY,
        ..Default::default()
    };
    let refined = MpConfig {
        refine: RefineConfig { enabled: true, ..RefineConfig::default() },
        ..grid
    };
    let guarded = MpConfig {
        hrmp: HrmpConfig { enabled: true, ..HrmpConfig::default() },
        ..refined
    };
    let full = MpConfig { full_update: true, max_atoms: 8, ..grid };

    let mut g = c.benchmark_group("pursuit");
    g.sample_size(10);

    g.bench_function("init", |b| {
        b.iter(|| {
            let mut p = Planner::new();
            std::hint::black_box(Mp::new(&dict, &signal, &mut p));
        });
    });

    for (name, cfg) in [
        ("grid", &grid),
        ("refined", &refined),
        ("hrmp", &guarded),
        ("full_update", &full),
    ] {
        // Report the atom count once, so per-atom cost can be recovered from criterion's timings.
        let mut p = Planner::new();
        let atoms = Mp::new(&dict, &signal, &mut p).run(cfg).len();
        println!("pursuit/{name}: {atoms} atoms");

        g.bench_function(name, |b| {
            b.iter(|| {
                let mut p = Planner::new();
                let mut mp = Mp::new(&dict, &signal, &mut p);
                std::hint::black_box(mp.run(cfg));
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_pursuit);

fn main() {
    benches();
    Criterion::default().configure_from_args().final_summary();
}
