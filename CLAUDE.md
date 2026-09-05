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
./target/release/rmp in.wav -b book.toml                    # analyse only, no resynthesis
./target/release/rmp -b book.toml -o resynth.wav            # synthesise a book, no analysis
./target/release/rmp --write-config > settings.toml

# statistics and visualization over a book
./target/release/rmpstat summary book.json [-c settings.toml]
./target/release/rmpstat diag    book.json -c settings.toml
./target/release/rmpstat hist    book.json --of alpha,bandwidth,f --weight energy
./target/release/rmpstat hist    book.json --of alpha,f -f svg -o plots/
./target/release/rmpstat snr     book.json -f svg -o snr.svg
./target/release/rmpstat wv      book.json -f png -o wv.png --log-freq --floor 65

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
- **`signal` / `book`** — f64 energy bookkeeping, and the decomposition result. `book::read` /
  `book::write` are the single definition of the on-disk format, used by both binaries.
- **`stats`** — aggregation over a book: derived quantities, weighted histograms, diagnostics.
  Nothing here renders.
- **`tfmap`** — the atom-based pseudo-Wigner time-frequency map (spec §21), as diagnostics only.
- **`config` / `audio` / `main`** — TOML settings, libsndfile I/O, the CLI.
- **`bin/rmpstat`** — the statistics CLI: clap, `plotters`, and text tables. A thin shell, so
  everything worth an oracle lives in `stats`/`tfmap` where `cargo test` reaches it.

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

**Never derive amplitude or phase in a bin scan.** `hypot` and `atan2` cost more than everything
else in the projection put together — at one call per bin per frame they were 60% of total runtime,
against 13% for the FFT the design treats as its inner loop. Only the energy chooses a bin, so the
scan compares energies and the full projection is solved once, for the winner. `corr::solve` and
`fit::Quad::solve_z` exist so the energy-only and full paths cannot drift apart. The same trap is
live in `refine`, whose 1-D searches read nothing but `.energy` a few hundred times per candidate.

**Blocks are independent, and that is load-bearing for speed.** Each owns its correlator, frame
table and tree; the only shared thing is the residual, read-only during a refresh. `mp::for_each_block`
runs them on rayon, which is why `Correlator` holds its own FFT plan and `RealFft::forward` takes
`&mut self`. Nothing crosses between blocks, so the parallel result is bit-identical and the
bit-identity gates keep their teeth. Below `PARALLEL_FRAME_THRESHOLD` it stays serial, because
rayon's per-task overhead is real next to the test fixtures' tiny stale sets.

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

**A book's `hr_score` is not a clamp severity.** It and `energy_removed` record the same
post-clamp energy and agree to 5e-3 on real material, so reading `hr_score < energy_removed` as
"HRMP clamped this atom" measures pure rounding — it reported 43% clamped on a book where the
figure is meaningless. The clamp is visible as `energy_removed / projected_energy`: median 0.98,
p5 0.76, min 0.24 on the 5000-atom piano book. On a book where HRMP did *not* run, that same
shortfall would be a parameter-mapping error instead, which is why the two readings need
separating.

**`--book` is an input or an output depending on whether an input soundfile is given.** With
one it is written; without one it is read and synthesised, and the whole analysis path — config,
dictionary, `--start`/`--duration`, `--residual` — is inapplicable rather than merely unused, so
those flags are errors in that mode. Analysis needs at least one of the three outputs; `--out`
alone is no longer mandatory, and skipping it also skips the resynthesis render.

**A synthesised book is longer than the excerpt it came from.** `Book::natural_len` sizes the
output by rendering each atom's envelope and taking the furthest death, where analysis sized the
residual by the input. The atom tails the analysis truncated at the excerpt end are audible again —
2.9% of the excerpt's energy on a 0.15 s piano fixture. Over the excerpt itself the two renders are
bit-identical, so this is a longer file, not a different one.

**A stale frame is bounded, not recomputed.** After a subtraction, frames overlapping the atom get
an upper bound — `(sqrt(E_old) + ||P a||)^2`, with `||P a||^2` bounded in O(1) by both `||a||^2` and
`(sum |a| E)^2 / lambda_min(G)` — and a dirty flag. Seeds are read off the *clean* frames only;
every dirty frame at or above the weakest seed is then recomputed as one parallel batch, and the loop
repeats until nothing dirty is above the line. The selected atom is identical to the eager update's
by construction. Two things that were measured, not reasoned: taking the threshold from the
*unmasked* table resolves one frame per pass and turns a 42 s run into a twenty-minute one; and the
envelope enters the bound **once**, through the basis — writing the change as `a*E` and bounding
with `E^2` double-counts it, and undercut by 1.1e-6. `lazy_bounds_never_undercut_the_exact_value`
checks every dirty frame after every atom against a fresh exact scan; keep it.

**The bound is one-sided, and that is the ceiling.** A bound never sits below `E_old`, so a frame
already near the top of the table is recomputed every time an atom touches it, however tight the
bound. A seven-second window over dense material is always near the top: the `alpha = 1` blocks
recompute 70% of what they bound. What remains for those blocks is the number of frames, which is
`capture_tolerance`.

**`signal::overlap` is the single definition of which samples an atom occupies.** Writing it
(`add_at`, `subtract_at`), scoring it (`fit::accumulate`) and invalidating the frames it touched
(`refresh_stale`) all go through it. `refresh_stale` used to be passed the seed's frame onset, which
equals the atom's `t0` only until refinement can move it.

### The pseudo-Wigner map

Spec §21 rules Wigner-Ville out of the pursuit and into diagnostics. `tfmap` builds
`E(t,f) = sum_k E_k * W~(t,f)` — a sum of *per-atom* distributions, so there are no cross-terms by
construction rather than by smoothing. Four facts that were measured, not assumed:

**The kernel is the separable product of each atom's exact time and frequency marginals.** Both
marginals are reproduced exactly; what is lost is their coupling. The true WVD of a decaying
exponential is a wedge whose frequency half-width to the first null is `1/(4t)`, so the product form
is too narrow over the first 23% of an atom's life and too wide over the rest. Since a product space
constrained only by its marginals has the product as its maximum-entropy solution, "the
least-committed joint density consistent with the atom's exact marginals" is a precise description,
which is what earns the *pseudo* label the spec demands.

**A closed-form Lorentzian frequency kernel is not viable, and the test suite knows it.** The
half-cosine attack cuts the tails hard: at `alpha*beta = 1` the exact spectrum is 5.5 dB below a
Lorentzian at five half-widths and 40 dB below at thirty. A Lorentzian would stay above a −60 dB
floor out to ~1000 half-widths — a full-height vertical smear under every atom with a real attack.
`the_real_book_does_not_haze_the_display` bounds the lit fraction at −60 dB (measured 21.3%) and is
the regression that encodes this.

**Transform the rendered *atom*, not an envelope spectrum shifted to `f_k`.** Shifting drops the
negative-frequency image, its cross term, and the Nyquist fold. 523 of `book1.json`'s 5000 atoms
have `2f < alpha/PI`, where the images overlap outright and the cross term is order one.

**Integrate mass over each bin; never sample a density at its centre.** This is what makes the alpha
range a non-issue with no multi-resolution grid: at 1200x800 over a real book, an `alpha = 19.7`
atom is 64 time bins by *under one* frequency bin while an `alpha = 3994` atom is *under one* time
bin by 196 frequency bins, and both are exact. The cumulative uses the **trapezoid** rule rather
than a left Riemann sum — same arithmetic, same transform, and it took the worst disagreement with
the direct oracle from 0.81 dB to 0.15 dB, which a Riemann cumulative would have needed a 16x longer
transform to match.

**An `(alpha, beta)` cache collapses nothing.** Refinement moves both continuously: 4752 of 5000
atoms have distinct envelope bits, 3125 even at 5% log quantization. The envelope *shape* depends
only on `(alpha*beta, fade_level)` with `alpha` a pure time scale — verified to 0.05 dB down to
−80 dB — but that identity is continuous-time and the sampled spectrum aliases, +13.7 dB at
`alpha = 3994`. `refine::EnvelopeCache` is also not reusable: its `get` is private and it is bounded
by `RefineConfig`. The module keeps only an exact-bits memo, worthless on a refined book and free on
an unrefined one.

**Kernels are computed in parallel and folded in serially, in book order.** Per-thread accumulators
would cost more to reduce than the fold takes (24 x 7.7 MB against ~100 ms), and fixing the f64
addition order makes the map bit-identical whatever the thread count — the same standard
`mp::for_each_block` is held to, and what gives the determinism gate teeth. 5000 atoms onto
1200x800 takes 0.13 s wall; the FFTs are the whole cost.

**`plotters` uses the `ab_glyph` backend, which has no font discovery.** `render::init_fonts` finds
a system sans font and registers it before any chart is drawn. The alternative, plotters' `ttf`
backend, does discover fonts but needs `libfontconfig1-dev` at build time — a system package to
install for a diagnostic tool, against reading one file at startup. Text output needs no font, so
`-f text` works regardless. The `colormaps` feature does **not** build without `full_palette`; the
heat ramp is local anyway.

**The heat map is blitted as one image, never drawn as rectangles.** A useful grid is ~10^6 cells,
and an SVG of that many `<rect>` elements would be tens of megabytes and would not open.
`SVGBackend` base64-embeds a PNG for a bitmap blit, so SVG and PNG take the identical path.

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

48 kHz, 22-block voice dictionary, one second of audio, 30 planted grains per second, 24 threads:

| | on-grid | off-grid, grid only | off-grid, refined |
| --- | --- | --- | --- |
| realtime factor | 0.1× | 0.6× | 0.3× |
| per atom | 1.40 ms | 1.54 ms | 2.95 ms |
| atoms to 40 dB | 31 | 378 | 91 |
| splitting factor | 1.4 | 17.2 | 4.5 |
| median \|Δf\| | 0.0 Hz | 17.8 Hz | 0.1 Hz |

**Analysis runs faster than realtime.** It did not always: three changes took the full refined
pipeline from 2.2× realtime to 0.3×, a factor of ten, with a bit-identical book. In order of size —
deriving `amp` and `phi` once per frame rather than once per bin (3.3×, see the invariant below),
refreshing the blocks in parallel (2.8×), and scanning the Gram rows without the per-bin liveness
check (1.06×).

**Per-atom cost is invariant to signal length**, confirming the local update works as designed.

**Refinement is the largest single win on quality.** Off-grid input needed 12× more atoms than
on-grid; refining `(t0, f, alpha, beta)` after selection cuts that to 3×, and 99% of selected atoms
move off the grid.

Refinement is now the *expensive* part of an iteration rather than a rounding error on it — 2.76 ms
per atom against 1.54 ms without. That is a reversal: it used to be the cheaper of the two because
`refresh_stale` dominated everything. Parallelising the refresh removed that cover, and refinement's
few hundred accumulation passes per candidate are still serial. It remains worth it four times over
on atom count, and it is where the next parallelism would go.

Caching `G` across the onset sweep — the one stage holding both envelope and carrier fixed, so `G`
cannot change — takes the refined arm from 266 ms to 248 ms, about 7% of that stage and 3%
end-to-end. Two failed attempts at the same idea are recorded in `fit`'s comments: splitting the
accumulation into a data pass and a Gram pass makes the cached case free but costs 15% overall,
because traversing the envelope twice is dearer than the arithmetic saved and every stage except the
onset sweep needs both halves anyway. The const-generic specialization is what works.

**Splitting is largely *caused* by grid mismatch**, which is why it falls with refinement rather than
needing back-projection. What remains (4.5) is the honest figure for a coherent dictionary.

**`candidate_count` buys nothing measurable.** 1, 4, 8 and 64 all reach 40 dB in 91 atoms with the
same splitting factor, while the candidate stage's cost scales linearly (5 ms at 1, 210 ms at 64,
over the same run): the strongest seed is also the seed that refines best. It stays configurable
because HRMP can *reject* a candidate rather than merely outscore it, and the loop then needs
somewhere to fall through to.

`cargo bench --bench pursuit` splits one decomposition by stage. At 0.25 s of off-grid audio, each
arm including its own `init` of 2.94 ms:

| arm | atoms | total | per atom |
| --- | --- | --- | --- |
| grid | 60 | 77.2 ms | 1.24 ms |
| refined | 60 | 81.8 ms | 1.32 ms |
| hrmp | 60 | 96.9 ms | 1.57 ms |
| full_update | 8 | 29.2 ms | 3.29 ms |

**The local update is worth 2.7× here**, and far more at longer durations — `full_update` scales with
the signal, the incremental path does not. Its margin narrowed when the refresh went parallel, since
recomputing everything is the more uniform workload and parallelises better.

Refinement costs little on this fixture (0.25 s, sparse) and a great deal on the denser one above
(1 s, 30 grains/s: 2.95 ms against 1.54 ms). It scales with candidates and rounds, not with the
signal, so its share grows as the parallel refresh shrinks everything around it.

Refinement's remaining error is concentrated in `(t0, alpha, beta)`, not `f`. All three shape the
attack, so they trade against each other along a shallow valley that coordinate descent walks down
but not along — a known cost of the one-dimensional method, bounded by a fit that still captures
99.9% of an isolated atom.

### The low-alpha regime, and what `capture_tolerance` is worth

The realistic configs in `data/config` reach down to `alpha = 1`: a 332k-sample support, a 337,500
point transform, and `support/hop` pinned near 134 by the default tolerance — so every atom refreshed
~130 frames of that transform per block. On 3 s of piano, `mp_1.toml`:

| | analysis | dictionary | realtime |
| --- | --- | --- | --- |
| before this work | 41.8 s | 1.6 s | 14.5× |
| lazy refresh, gallop hop | 28.4 s | 0.05 s | 9.5× |
| the same at `capture_tolerance = 0.5` | 6.6 s | 0.06 s | 2.2× |

**`capture_tolerance` is the lever, and it is a config trade rather than a free win.** A frame
count scales as `1/ln(1/tol)`, so 0.5 has 13× fewer frames than 0.95, and the profile is 85%
transforms either way. On piano it costs 8% more atoms to the same SNR (978 → 1057); 0.3 is *slower*
again (19 s) — so it is an optimum, not a monotone knob. On the adversarial synthetic, which plants
every atom exactly `hop/2` off-grid, 0.5 alone is a real regression (193 atoms to 40 dB against
102, splitting 10.5 against 4.5) because the coarser seed misranks; `candidate_count = 8` recovers
it fully (95 atoms, 4.8) — but on piano more candidates buy nothing (1057 / 1077 / 1073 for 1/4/8).
The defaults stay at 0.95 and 1, because the 0.95 hop is a correctness property without refinement;
with refinement on, set 0.5 for realistic material and expect the numbers above.

**Splitting the refresh by frame instead of by block was slower**, 30% at every thread count from 4
to 24, and chunking frames by block did not recover it. Same profile shape with more time in the
FFT's load-heavy butterflies — bouncing threads across six differently sized envelopes and buffer
sets. The block split's locality is worth more than its balance costs.

**The transform length is not the problem.** rustfft runs these lengths at ~1.1 ns/sample; a
factorisation with a small odd cofactor (345600 = 2⁹·675 over 337500 = 2²·3³·5⁵) is 7–14% faster on
the two largest blocks, a few percent overall, and would re-baseline every book. Not taken.

**`measure_hop` gallops and bisects.** The linear scan was 16k O(N) correlations per low-alpha block
at 0.5 — twenty seconds of dictionary build sitting invisibly ahead of a seven-second analysis,
because only the analysis line was being read. Monotonicity is checked, not assumed:
`hop_search_matches_the_linear_scan` pins it on every block of both dictionaries.

`RMP_REFRESH_DETAIL=1` prints per-block bounded/recomputed counts and the transform samples each
block cost; it is how every attribution above was made.

### Tuning a low-alpha config: what the settings are worth

Measured on 3 s of piano, `mp_1.toml`, every arm driven to the same 35 dB so atoms and wall clock
are comparable. Residual peak is the quality column that moves: at equal rms it says how well the
transients are handled, and it is where a badly-placed long atom shows up.

| arm | atoms | wall | resid peak |
| --- | --- | --- | --- |
| all six alpha rungs, `max_atom_samples = 65536` | 1631 | 9.9 s | −17.3 dB |
| alphas `[16,64,256]` only | 1735 | 11.2 s | −30.8 dB |
| **`max_atom_samples = 150000`, `alpha_bracket = 2.5`, `rounds = 2`** | **1528** | **12.6 s** | **−30.3 dB** |
| `max_atom_samples = 400000`, `alpha_bracket = 2.5` | 1441 | 27.4 s | −30.6 dB |
| `max_atom_samples = 400000`, `alpha_bracket = 1.6` | 1448 | 59.2 s | −29.4 dB |

**`refine.max_atom_samples` silently decides whether a block is refined at all.** `refine` asks the
envelope cache for the seed's own shape before anything else; the cache refuses a shape longer than
the cap, and refinement declines. The atom is still *selected*, just pinned to grid frequency, grid
onset and grid envelope. At the default 65536 against a dictionary reaching `alpha = 1`, that was 12
of 24 blocks and 34% of the atoms in the book — and worth 13 dB of residual peak, invisible in the
rms figure. `analyse` now prints how many blocks are affected and what fraction of atoms actually
moved off the grid.

**Capping it *deliberately* below the longest block is the best setting measured.** At `rounds = 2`,
150000 gives −30.3 dB against 400000's −24.9: letting refinement chase seven-second atoms it cannot
converge on in two rounds is worse than not letting it start. At `rounds = 3` the ordering reverses
(−29.7 against −30.6) but costs 17 s against 27 s. The cap is a regularizer, not just a budget.

**A wider `alpha_bracket` is faster, not slower.** 2.5 against 1.6 at `max_atom_samples = 400000`:
27.4 s against 59.2 s for the same atom count and peak. Golden section covers the range in fewer
rounds, so `score_tol` ends the search sooner. 4.0 is slower again (39.5 s) with no gain.

**The low rungs earn their keep on this material.** 90–96% of the removed energy sits in `alpha < 8`
— piano is sustained, and long atoms are right for it. Dropping them is a real trade, not free:
`[16,64,256]` needs 6% more atoms but has the fastest dictionary build by 26× (2.4 ms against 64 ms)
and holds the peak, so it is the right answer when the dictionary is rebuilt often.

**`candidate_count` and `golden_iters` do nothing here.** 4 candidates costs 42% more wall for the
same atoms; `golden_iters = 6` saves 13% and loses 4 dB of peak.

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

**Adding `src/bin/rmpstat/` needed no `[[bin]]` section.** Cargo auto-discovers `src/main.rs` and
`src/bin/*` together, so `rmp` is unaffected.

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

## Comparing two decompositions

**A WAV written twice is not byte-identical**, because libsndfile stamps a timestamp into the PEAK
chunk of a float file. Exactly one byte differs and the audio data is untouched, but `cmp` on the
file reports a difference and reads as nondeterminism. Compare the book, or the data past byte 72.

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
