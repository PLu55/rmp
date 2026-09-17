//! The scoring loop: `fit::accumulate` and its cached-Gram form, per sample.
//!
//! Every refinement trial, every full-support score and HRMP's main fit go through this loop, and it
//! is the largest single cost of a refined analysis once the refresh is parallel. The arms are the
//! two specialisations refinement actually runs:
//!
//! - `with_gram` — [`fit::accumulate`], the data and the Gram together: every stage but the onset
//!   sweep;
//! - `data_only` — [`fit::accumulate_with`] handed a Gram, the onset sweep's case.
//!
//! across supports that real trials have: a short FOF (`alpha = 256`), and Gaussians of 15 ms,
//! 100 ms and 420 ms — the last about `refine.max_atom_samples = 150000` in the shipped configs.
//! Throughput is in samples, so criterion's rate reads directly as samples per second.
//!
//! `CLAUDE.md`'s measurement discipline applies: run-to-run spread on this machine reaches ~5%, so
//! treat smaller differences as unresolved, and compare against a saved baseline
//! (`--save-baseline` / `--baseline`) rather than across sessions.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use rmp_core::atom::Shape;
use rmp_core::fit;
use rmp_core::fof::{Envelope, EnvelopeParams};
use rmp_core::gauss::GaussianParams;

const SR: f32 = 48_000.0;

fn noise(n: usize) -> Vec<f32> {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 8_388_608.0 - 1.0
        })
        .collect()
}

fn bench_fit(c: &mut Criterion) {
    let shapes: [(&str, Shape); 4] = [
        ("fof_a256", EnvelopeParams::new(256.0, 0.001).into()),
        ("gauss_15ms", GaussianParams::new(0.015).into()),
        ("gauss_100ms", GaussianParams::new(0.1).into()),
        ("gauss_420ms", GaussianParams::new(0.42).into()),
    ];
    let omega = std::f64::consts::TAU * 440.0 / SR as f64;

    let mut g = c.benchmark_group("fit");
    for (name, shape) in shapes {
        let env = Envelope::render(shape, SR).unwrap().samples;
        let n = env.len();
        let residual = noise(n);
        let gram = fit::gram(&env, omega);
        println!("fit/{name}: {n} samples");

        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::new("with_gram", name), |b| {
            b.iter(|| std::hint::black_box(fit::accumulate(&residual, &env, 0, omega)));
        });
        g.bench_function(BenchmarkId::new("data_only", name), |b| {
            b.iter(|| {
                std::hint::black_box(fit::accumulate_with(&residual, &env, 0, omega, Some(gram)))
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_fit);

fn main() {
    benches();
    Criterion::default().configure_from_args().final_summary();
}
