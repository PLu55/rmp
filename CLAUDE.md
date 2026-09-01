# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rmp` ("rust matching pursuit") is an **audio analysis** engine: it decomposes a signal into FOF
atoms (Rodet's Formant Wave Function, from the CHANT synthesizer) using Matching Pursuit. The
tractability design follows Krstulovic & Gribonval, *"MPTK: Matching Pursuit Made Tractable"*
(ICASSP 2006); `notes.md` holds the full bibliography.

Analysis is the *inverse* of what `/home/plu/Projects/rfofs` does. rmp finds FOF parameters; rfofs
synthesizes from them. A decomposition is replayable through rfofs unchanged.

## Commands

```bash
cargo build --release
cargo test
cargo test <test_name>                       # single test
cargo clippy --all-targets

# the CLI
./target/release/rmp in.wav -o resynth.wav [-c settings.toml] [-r residual.wav] [-b book.toml]
./target/release/rmp --write-config > settings.toml

# end-to-end measurement against synthetic ground truth
cargo run --release --example analyze [seconds] [max_atoms] [grains_per_sec]

cargo bench --bench fft                      # FFTW planning = MEASURE (default)
FFTW_PLAN=patient cargo bench --bench fft
```

Benchmarks use criterion with `harness = false`, matching rfofs's convention.

## Architecture

The pipeline is: dictionary → correlate every frame → pick the best atom → subtract → repeat. The
parts that need reading together:

- **`fof`** — the bridge to rfofs. Nothing here reimplements FOF math; envelopes and support lengths
  come from *rendering a probe grain* and inspecting it.
- **`dict`** — blocks, one per `(alpha, beta)` envelope. Owns the Gram tables and the hop.
- **`corr`** — one envelope-windowed FFT per frame yields correlations against every frequency at
  once, then a closed-form 2-D projection.
- **`mp`** — the pursuit loop with the local update.
- **`select`** — max segment tree over frames.
- **`naive`** — brute-force oracle. Deliberately shares nothing with `dict`/`corr` beyond the atom
  definition and the search space.
- **`signal` / `book`** — f64 energy bookkeeping, and the decomposition result.
- **`config` / `audio` / `main`** — TOML settings, libsndfile I/O, the CLI.

### Invariants that are not locally obvious

**`E(t)` depends only on `(alpha, beta, fade_*)`, never on `f` or `phi`.** This is what makes one FFT
per frame serve every frequency, and it is the load-bearing fact of the whole design.

**Hop scales as `1/alpha`, not as a fraction of the window.** Onset capture falls off as
`exp(-2*alpha*|delta|)`, a width set by the decay rate. An MPTK-style `hop = L/2` captures ~1e-4 of an
atom's energy here. Hop is *measured* from each block's own envelope autocorrelation rather than
derived, so it accounts for `beta` and the fade tail. Because the loss depends on `alpha`, this is a
correctness property: a fixed hop would bias selection toward small-`alpha` blocks.

**Phase is solved, not searched.** The dictionary grids `(alpha, beta, t0, f)` only; `phi` and `amp`
come from `z = G^-1 d`. `amp = hypot(z)` is non-negative by construction, so there is no sign fixup
and no mod-pi ambiguity.

**Never use `sqrt(d_u^2 + d_v^2)` as the selection criterion.** Its relative error is `O(rho_k)` —
about 8% for a 500 Hz formant of 80 Hz bandwidth — which is fatal to a greedy algorithm whose top
candidates differ by a few percent.

**DC and Nyquist are excluded outright**; the sine basis vector is identically zero there, so `G` is
exactly rank-1. A `rho` gate disables the ill-conditioned low band as well.

**`rfofs::fof_amax` must stay out of the amplitude path.** The probe render already carries the
`1/amax` factor, so a fitted coefficient maps *directly* to `FofParams::amp`. It is public now (as
`rfofs_amax` across the C ABI) and applying it again would double-normalize. It is used only to screen
the `alpha*beta > 10` cliff before rendering.

**Residual energy is measured, never assumed.** The atom subtracted is what rfofs actually rendered,
not the ideal vector projected onto, so `||R_new||^2` is computed from the rendered atom. Buffers are
f32; accumulators are f64.

**`support_len` and `fft_len` are different fields.** The stale set and the envelope use support; the
frame read and the FFT use `fft_len` (rounded to an even 5-smooth length, not a power of two).

## Working with rfofs

`rfofs::fof` is pure math (imports only `wide` and its build-script sine LUT) and safe to depend on.
`engine`/`queue`/`shm`/`offline` are realtime and I/O infrastructure — not needed here.

Three traps when rendering: `fill_block` **accumulates** (`+=`), so zero the buffer first; render
each atom in **one** call because `decay_acc` is a running product carried across calls; and
`start_sample`/`block_start` must both be 0 or the sub-block offset logic silently emits nothing.

Derive support lengths by rendering, never by formula — rfofs clamps `decay_end` to at least
`attack_end`, and its death sample depends on internal rounding a separate formula will drift from.

## Testing discipline

Mirrors rfofs's `NaiveRef` pattern: an obviously-correct implementation asserted against the
optimized one. Three levels — closed form vs. brute-force scan, fast MP vs. `naive`, and incremental
update vs. full recompute.

**The stage-3 gate compares the whole frame table after every atom, not the atoms selected.** The
weaker form has no teeth: frames at the stale range's edges overlap the atom only where its envelope
has decayed to −60 dB, so an off-by-one perturbs energies by ~2e-7 and leaves the selection order
intact. Verified by mutation — the table comparison catches both a short `n_hi` and a high `n_lo`.

When adding a recovery test, assert planted atoms are inside `[k_lo, k_hi]`. An atom above `f_max` is
unrepresentable and presents as a decomposition failure rather than a bad test.

## Measured performance

48 kHz, 22-block voice dictionary, one second of audio:

| | on-grid | off-grid |
| --- | --- | --- |
| realtime factor | 1.0× | 9.5× |
| atoms to 40 dB | 31 | 378 |
| splitting factor | 1.4 | 17.2 |

**Per-atom cost is invariant to signal length** (~24 ms across a 10× change), confirming the local
update works as designed. **Off-grid input needs 12× more atoms for the same SNR** — this dominates
every other cost, and is the case for post-selection refinement over `(t0, f, alpha, beta)` being the
next thing built. Splitting is largely *caused* by grid mismatch rather than being independent, so
refinement should be measured before back-projection is added.

## Build configuration — three things that will bite

**`src/main.rs` must contain a `main`.** An empty file fails the whole build with `E0601`, including
bench targets, which makes `cargo bench` look broken for an unrelated reason.

**`fftw` must keep `features = ["system"]`.** The crate's default `source` feature vendors FFTW 3.3.8
with no SIMD flags at all — the generated `config.h` has `HAVE_AVX`, `HAVE_AVX2`, `HAVE_SSE2` all
`#undef`, producing scalar code ~3× slower. Requires `libfftw3-dev`. Note the **engine uses realfft
only**; `fftw` exists solely for `benches/fft.rs`.

**`.cargo/config.toml` sets `-C target-cpu=native`.** rfofs sets the same flag, and it does *not*
propagate across a path dependency. rfofs's `wide::f32x8` SIMD width decides which sine approximation
each sample gets (degree-9 polynomial for full lanes, LUT for the scalar tail), so a mismatched build
changes atom values between the crates. An explicit `RUSTFLAGS` env var overrides `build.rustflags`
rather than merging — don't set both.

`rustfft` is deliberately commented out in `Cargo.toml`; it still arrives transitively via `realfft`.

## The FFT benchmark

`benches/fft.rs` compares realfft against FFTW at 1024–32768, the range FOF supports land in
(`support ≈ 6.9·sr/alpha`).

- `FFTW_PLAN` selects `measure` (default) or `patient`; an unrecognized value panics. The mode is in
  the benchmark id so criterion keeps separate baselines.
- The summary table does **not** re-time anything — it reads back criterion's stored
  `estimates.json`, so columns can come from **different runs**; arms never run show `—`.
- `Flag` is bitflags 2.x deriving only `Default`, so it is **not `Copy`** — build it fresh per plan.
- Both planning modes overwrite the arrays, so fill input buffers *after* creating the plan.

## Measurement discipline

Run-to-run variance on this machine reaches ~5% (realfft is markedly less reproducible than FFTW —
17% spread at N=8192 across runs, against FFTW's 3.6%). **Treat sub-5% differences as unresolved**
unless they reproduce across several runs. Criterion's within-run confidence intervals are much
tighter than the true between-run spread and will overstate your confidence.

## Design history

An earlier written plan lives at `/home/plu/.claude/plans/the-purpose-of-this-replicated-moon.md`.
It is outside the repo and machine-local, so treat it as background rather than a reference: its
derivations have since been implemented and its numbers corrected in place (the dictionary is 22
blocks, not 23; the real-FFT fold is unreachable at the default `f_max`; the cost model was an order
of magnitude pessimistic). Where the two disagree, the code and this file are current.
