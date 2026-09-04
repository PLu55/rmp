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
./target/release/rmp in.wav -o resynth.wav -s 2.5 -d 0.5   # analyse one excerpt, in seconds
./target/release/rmp in.wav -o resynth.wav -b book.json.gz   # any book format, compressed
./target/release/rmp --write-config > settings.toml

# end-to-end measurement against synthetic ground truth
cargo run --release --example analyze [seconds] [max_atoms] [grains_per_sec] [candidates]

cargo bench --bench fft                      # FFTW planning = MEASURE (default)
cargo bench --bench pursuit                  # per-stage cost of one decomposition
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
- **`fit`** — the same exact projection *off* the grid, by direct f64 summation. Refinement and
  HRMP both need a score where no precomputed Gram exists.
- **`cand`** — coarse discovery: local time-frequency maxima, merged across blocks.
- **`refine`** — bounded 1-D search over `(t0, f, alpha, beta)` after selection.
- **`hrmp`** — local-support probes and the amplitude clamp.
- **`mp`** — the pursuit loop with the local update; owns the candidate → refine → validate → select
  pipeline.
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

**The fit-region score and the full-support score are not interchangeable.** `E_capt` is the squared
norm of an orthogonal projection of *one fixed* ambient vector — that is the only reason scores from
different blocks, onsets and bins are comparable. Truncating the envelope at `fit::fit_end` zeroes
the residual outside a *candidate-dependent* window, so a truncated score ranks candidates partly by
how much residual each was allowed to ignore. The fit region drives `refine`'s 1-D searches and never
leaves that module; everything else uses the full support.

**Refinement can lose, so it must be allowed to decline.** Because those two optima differ, the
parameters maximizing the search objective can score worse on the full support than the seed. Both
are scored through `fit::score` — same function, so the same Gram clipping — and the refined atom is
adopted only if it strictly wins.

**`fit` clips the Gram to the same range as the data; `corr` and `naive` do not.** They use a
whole-support Gram while reading the residual past the signal end as zeros, which understates a
boundary-overhanging atom's removable energy. The two agree to the last bit for interior atoms, but
a seed and a refined candidate must always be compared through `fit`.

**A rejected HRMP candidate must be demoted, not just skipped.** `mp::run` breaks the pursuit when
residual energy rises, so a rejection must never reach that line. Writing the rejected score back
into the frame table and its segment tree is what stops the next iteration recomputing the same
argmax forever — an infinite loop with no error message. The demotion happens whether or not some
*other* seed was selected that iteration: a rejection is a fact about the residual at that frame,
and leaving its energy standing means the next iteration promotes the same doomed seed again.

**`max_atoms` counts selected atoms, not iterations.** An iteration where HRMP rejects everything
adds nothing to the book, so charging it to the budget lets a strict setting spend the whole budget
on atoms it refused. `max_stalls` is what bounds a barren stretch; the atom cap must not double as
an iteration cap or the two limits interfere.

**`signal::overlap` is the single definition of which samples an atom occupies.** Writing it
(`add_at`, `subtract_at`), scoring it (`fit::accumulate`) and invalidating the frames it touched
(`refresh_stale`) all go through it. `refresh_stale` used to be passed the seed's frame onset, which
equals the atom's `t0` only until refinement can move it.

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

**An HRMP fixture has to be deliberately adversarial.** Given the matching short shape in the
dictionary, plain MP simply selects it and never bridges a gap — there is nothing for HRMP to
prevent, and a test built that way passes for the wrong reason. Offer only the long shape, then
measure the energy the book puts into a silent gap.

**Phase in a multi-atom fixture is referenced to each atom's own onset.** A second burst given
`phi = PI` is not in anti-phase with the first: it is rotated by `omega * (t2 - t1)` — tens of
carrier periods — and typically lands almost back in phase. `H_i` applies that rotation
automatically during the test, but a fixture has to apply it by hand.

## Measured performance

48 kHz, 22-block voice dictionary, one second of audio, 30 planted grains per second:

| | on-grid | off-grid, grid only | off-grid, refined |
| --- | --- | --- | --- |
| realtime factor | 1.0× | 9.6× | 2.2× |
| atoms to 40 dB | 31 | 378 | 91 |
| splitting factor | 1.4 | 17.2 | 4.5 |
| median \|Δf\| | 0.0 Hz | 17.8 Hz | 0.1 Hz |

**Per-atom cost is invariant to signal length** (~24 ms across a 10× change), confirming the local
update works as designed.

**Refinement is the largest single win and is nearly free.** Off-grid input needed 12× more atoms
than on-grid; refining `(t0, f, alpha, beta)` after selection cuts that to 3×, and cuts wall time
4.4×. It costs about 0.2 ms per candidate — roughly 2% of a 21 ms iteration — because `refresh_stale`
dominates everything. 99% of selected atoms move off the grid.

**Splitting is largely *caused* by grid mismatch**, which is why it falls with refinement rather than
needing back-projection. What remains (4.5) is the honest figure for a coherent dictionary.

**`candidate_count` buys nothing measurable.** 1, 4, 8 and 64 all reach 40 dB in 91 atoms with the
same splitting factor, while the candidate stage's cost scales linearly (5 ms at 1, 210 ms at 64,
over the same run): the strongest seed is also the seed that refines best. It stays configurable
because HRMP can *reject* a candidate rather than merely outscore it, and the loop then needs
somewhere to fall through to.

`cargo bench --bench pursuit` splits one decomposition by stage. At 0.25 s of off-grid audio, each
arm including its own `init` of 74 ms:

| arm | atoms | total | per atom |
| --- | --- | --- | --- |
| grid | 60 | 1.40 s | 22.2 ms |
| refined | 60 | 1.19 s | 18.6 ms |
| hrmp | 42 | 0.92 s | 20.2 ms |
| full_update | 8 | 0.67 s | 74.0 ms |

**The local update is worth 3.3× here**, and far more at longer durations — `full_update` scales with
the signal, the incremental path does not. Refinement is *cheaper* per atom than the plain grid, not
merely affordable: a refined atom's support differs from its seed block's, and the stale sets it
invalidates are on average smaller. HRMP selects fewer atoms in the same budget because it declines
the ones it cannot justify, which is the point of it.

Refinement's remaining error is concentrated in `(t0, alpha, beta)`, not `f`. All three shape the
attack, so they trade against each other along a shallow valley that coordinate descent walks down
but not along — a known cost of the one-dimensional method, bounded by a fit that still captures
99.9% of an isolated atom.

### HRMP on real material: the two settings that decide everything

The synthetic fixtures are sparse and isolated, and HRMP's original defaults were tuned there. Dense
polyphonic audio behaves differently, because a probe's local residual carries *other events*, so a
local phase far from the global fit is ordinary rather than evidence of a bridged gap. Measured on
0.5 s of solo piano at 48 kHz, `candidate_count = 1`, refinement on, target 40 dB:

| phase tolerance | depth | atoms | SNR | rejected |
| --- | --- | --- | --- | --- |
| 45° | 2 | 136 | 11.7 dB | 3864 |
| 60° | 2 | 1078 | 40.0 dB | 2791 |
| 90° | 2 | 1111 | 40.0 dB | 917 |
| 90° | 1 | 979 | 40.0 dB | 0 |
| HRMP off | — | 920 | 40.0 dB | — |

**`phase_tolerance_deg` saturates at 90.** The `dot > 0` sign rule rejects everything beyond a
quarter turn on its own, so 90 and 179 give bit-identical books; the setting only ever tightens.
45° is not a mild tightening — it is the difference between a decomposition and a failure.

**`depth` is the strictness knob, not just a resolution knob.** Rejection is "any probe disagrees",
so the rate climbs with the probe count: `2^depth` masks means `2^(depth+1) - 1` overlapping probes.
Depth 1 rejects nothing on this material and still clamps 85% of what it passes at a mean `rho` of
0.79 — HRMP is fully engaged, just not trigger-happy.

At depth 1 and 90°, HRMP costs about 6% more atoms than plain MP for the same SNR. That is the
honest price of the constraint; anything far above it means a setting is rejecting good atoms.

`noise_epsilon` moves the same dial from the other end: lowering it to 0.05 makes most probes
uninformative (698 of 745 here), so HRMP passes almost everything through unchanged. That reaches
40 dB in 897 atoms but is close to disabling HRMP rather than tuning it.

## Book size

A book is dominated by its per-atom record, and the text formats are extravagant about it. Measured
on a 3000-atom decomposition of 7.0 s of mono 48 kHz audio:

| encoding | bytes | per atom |
| --- | --- | --- |
| pretty JSON (`-b book.json`) | 1,597,609 | 532 |
| compact JSON | 1,009,592 | 336 |
| gzipped JSON (`-b book.json.gz`) | 229,319 | 76 |
| fixed-width binary, every field | 231,000 | 77 |
| replay-only `(t0, f, alpha, beta, phi, amp)` | 72,000 | 24 |

**The JSON book is 1.19× the size of the f32 WAV it decomposes**, so the decomposition expands
rather than compresses until something is done about it. A `.gz` or `.gzip` suffix on `--book`
gzips the output and is worth about 7×; the format is then read from the extension *beneath* the
suffix, so `book.json.gz` is JSON and a bare `book.gz` is TOML, matching the no-extension default.

**A binary format would buy the same 6.9× as gzip and no more** — its advantage would be parse-free
loading, not size. The size lives in the field list: only 24 of the 77 bytes/atom are replayable
parameters. `block`/`onset`/`bin` record the pre-refinement grid point, which refinement moved for
all but one atom of the 3000; the three energies are f64 for what is only ever printed as dB; and
`fade_level`/`fade_dur` are config constants re-encoded per atom (1 and 7 distinct values in 3000).
Splitting the replay stream from the diagnostics is where the remaining 22× is, not the encoding.

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
