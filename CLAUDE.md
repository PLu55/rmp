# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rmp` ("rust matching pursuit") is an **audio analysis** engine: it decomposes a signal into FOF
atoms (Rodet's Formant Wave Function, from the CHANT synthesizer) and Gaussian (Gabor) atoms using
Matching Pursuit. The tractability design follows Krstulovic & Gribonval, *"MPTK: Matching Pursuit
Made Tractable"* (ICASSP 2006); `notes.md` holds the full bibliography.

Analysis is the *inverse* of what `/home/plu/Projects/rfofs` does. rmp finds FOF parameters; rfofs
synthesizes from them. A book's FOF atoms replay through rfofs unchanged; its Gaussian atoms have no
rfofs representation and are defined and rendered by rmp itself. `rmp` only analyses — every render
of a book, atoms and stochastic residual alike, is `rmpsynth`'s.

## Commands

```bash
# a virtual workspace: these still act on all four crates from the root
cargo build --release
cargo test
cargo test <test_name>                       # single test
cargo clippy --all-targets
cargo build --release -p rmp-core            # one crate

# analysis: rmp writes a book, and never audio other than the residual
./target/release/rmp in.wav -b book.json.gz [-c settings.toml] [-r residual.wav]
./target/release/rmp in.wav -b book.toml -s 2.5 -d 0.5      # analyse one excerpt, in seconds
./target/release/rmp --write-config > settings.toml

# residual stochastic analysis (off by default)
./target/release/rmp in.wav -b book.json.gz --residual-analysis
./target/release/rmp in.wav -b book.json --residual-book bank.json.gz
RMP_RESIDUAL_DETAIL=1 ./target/release/rmp in.wav --residual-book bank.json.gz

# synthesis: rmpsynth renders a book's atoms (FOF through rfofs, Gaussian by rmp) and its residual
./target/release/rmpsynth -b book.json.gz -o resynth.wav                  # atoms + embedded residual
./target/release/rmpsynth -b book.json.gz --no-residual -o atoms.wav      # atoms only
./target/release/rmpsynth -b book.json --residual-book bank.json.gz -o mixed.wav
./target/release/rmpsynth -b bank.json.gz -o stochastic.wav               # residual book: noise only
./target/release/rmpsynth -b bank.json.gz -o out.wav --seed 7 --gain-db -6 --encoding pcm24

# statistics and visualization over a book
./target/release/rmpstat summary book.json [-c settings.toml]
./target/release/rmpstat diag    book.json -c settings.toml
./target/release/rmpstat hist    book.json --of alpha,bandwidth,f --weight energy
./target/release/rmpstat hist    book.json --of sigma,bandwidth              # gaussian atoms
./target/release/rmpstat hist    book.json --of alpha,f -f svg -o plots/
./target/release/rmpstat snr     book.json -f svg -o snr.svg
./target/release/rmpstat wv      book.json -f png -o wv.png --log-freq --floor 65

# the graphical front end (a scaffold)
cargo run --release -p rmp-gui

# end-to-end measurement against synthetic ground truth
cargo run --release -p rmp-cli --example analyze [seconds] [max_atoms] [grains_per_sec] [candidates]

cargo bench -p rmp-core --bench fft          # FFTW planning = MEASURE (default)
cargo bench -p rmp-core --bench pursuit      # per-stage cost of one decomposition
FFTW_PLAN=patient cargo bench -p rmp-core --bench fft
```

Benchmarks use criterion with `harness = false`, matching rfofs's convention.

`MANUAL.md` is the user-facing settings reference: every knob, what it does to the result and to
the clock, with the measured numbers. It is the right place for tuning advice; this file is the
right place for why the code is shaped as it is. Settings guidance added here should end up there.

## Architecture

### Crate layout

A four-crate workspace. The dependency direction is the boundary the split exists to enforce, and
it points one way only:

```
rmp-cli  ─┐
          ├─→  rmp-synthesis  ─→  rmp-core
rmp-gui  ─┘                          ↑
                                     └── (rmp-cli and rmp-gui also depend on it directly)
```

- **`rmp-core`** — all analysis, and everything both front ends need: the atoms, the dictionary, the
  pursuit, the book format, the settings document, the statistics, the time-frequency map, the ERB
  residual analysis, and libsndfile I/O. No clap, no plotters, no synthesis.
- **`rmp-synthesis`** — turning a book back into audio. Was `src/synth/`.
- **`rmp-cli`** — the three binaries, and the only crate that knows about clap or plotters.
- **`rmp-gui`** — an eframe front end. A scaffold; see its own module docs.

**Nothing points back up, and one test had to move to keep it that way.** `mp`'s HRMP gap fixture
measures the energy a book puts into a silent gap, which means rendering the book. A dev-dependency
from `rmp-core` onto `rmp-synthesis` *compiles* — cargo permits cycles through dev-dependencies —
but the types do not unify across one: the `rmp_core` linked into a lib-test target is a different
compilation unit from the one `rmp-synthesis` was built against, so a `Book` made by the test is not
the `Book` `render_atoms` accepts. The fixture is now `rmp-synthesis/tests/hrmp_gap.rs`. Do not
reintroduce the dev-dependency to move it back.

`residual::pseudo_noise` is public for the same reason: the residual synthesis tests in the other
crate compare against that exact stream, and two copies of a fixture whose values matter is the
drift this codebase avoids everywhere else.

Versions live in `[workspace.dependencies]` at the root; members write `dep.workspace = true`. Path
dependencies there resolve against the *workspace root*, which is why rfofs is still `../rfofs`.

### Modules

The pipeline is: dictionary → correlate every frame → pick the best atom → subtract → repeat. The
parts that need reading together, all in `rmp-core` unless said otherwise:

- **`atom`** — `Shape` (`Fof | Gaussian`), `AtomParams` and the rendered `Envelope`, dispatching to
  the two kinds. Everything downstream of it works from a rendered envelope and never asks which
  kind it holds; what differs between kinds is confined here, to `refine`'s searched parameters,
  `fit::fit_end`, HRMP's scaled probes, and `stats`.
- **`fof`** — the bridge to rfofs. Nothing here reimplements FOF math; envelopes and support lengths
  come from *rendering a probe grain* and inspecting it.
- **`gauss`** — the Gaussian atom. rmp owns this definition, so its envelope is the formula and its
  support a closed form.
- **`dict`** — blocks, one per envelope shape of either kind. Owns the Gram tables and the hop.
- **`corr`** — one envelope-windowed FFT per frame yields correlations against every frequency at
  once, then a closed-form 2-D projection.
- **`fit`** — the same exact projection *off* the grid, by direct f64 summation. Refinement and
  HRMP both need a score where no precomputed Gram exists.
- **`cand`** — coarse discovery: local time-frequency maxima, merged across blocks.
- **`refine`** — bounded 1-D search over `(t0, f, alpha, beta)` for a FOF, `(t0, f, sigma)` for a
  Gaussian, after selection.
- **`hrmp`** — local-support probes and the amplitude clamp.
- **`mp`** — the pursuit loop with the local update; owns the candidate → refine → validate → select
  pipeline, and the windowing that keeps its frame tables bounded on a long clip.
- **`naive`** — brute-force oracle. Deliberately shares nothing with `dict`/`corr` beyond the atom
  definition and the search space.
- **`signal` / `book`** — f64 energy bookkeeping, and the decomposition result. `book::read` /
  `book::write` are the single definition of the on-disk format, used by both binaries.
- **`stats`** — aggregation over a book: derived quantities, weighted histograms, diagnostics.
  Nothing here renders.
- **`tfmap`** — the atom-based pseudo-Wigner time-frequency map (spec §21), as diagnostics only.
- **`residual`** — stochastic analysis of the final residue: an ERB gammatone bank, one-pole band
  power, and a fixed-rate `ResidualBook`. A post-processing stage; it cannot touch the pursuit.
- **`pipeline`** — one analysis end to end: build the dictionary, plan the windows, run the
  pursuit, analyse the residue. It computes and reports facts and formats nothing, which is what
  lets the CLI and the GUI drive the same code without one of them dictating how the other reads.
- **`config` / `audio`** — TOML settings, libsndfile I/O.
- **`rmp-synthesis`** — all synthesis. `atoms` renders a book's atoms through the same per-atom
  render the pursuit subtracted; the rest is the inverse of `residual`: a power-complementary ERB
  bank driven by independent per-band noise at `sqrt(P_b)`, reusing `rmp_core::residual::filter`
  outright. `render` mixes the two on one timeline.
- **`rmp-cli/src/bin/rmp`** — the analysis CLI. Flag merging, file I/O and `report.rs`, which holds
  every line it prints and nothing else.
- **`rmp-cli/src/bin/rmpsynth`** — the synthesis CLI. A file and configuration front end over
  `rmp-synthesis`; no DSP lives in it.
- **`rmp-cli/src/bin/rmpstat`** — the statistics CLI: clap, `plotters`, and text tables. A thin
  shell, so everything worth an oracle lives in `stats`/`tfmap` where `cargo test` reaches it.
- **`rmp-gui`** — the eframe front end. The window is a strip of tabs, each an independent
  analysis: its own input file, settings, run, log and results, titled `NN filename.wav` with `NN`
  the lowest two-digit number no open tab is using. **A tab is one file, and opening a file is the
  only way a tab comes into being** — so there is no empty tab, not at startup and not after the
  last one closes, and no way to point a tab at a second file. Both are facts about `Session`
  rather than rules the UI has to keep remembering: `input` is a `PathBuf` set at construction, so
  there is no state for an empty tab to be in and nothing to write a second file into. The reason
  is that a tab's results, log and title all describe one file, and swapping it underneath would
  leave a book describing a file the tab no longer names. With no tabs the window shows an Open
  button and nothing else. `task` runs a decomposition off the UI thread; the panels inside a tab
  are stubs naming the `rmp-core` call each is a view of.

### Invariants that are not locally obvious

**`E(t)` depends only on the shape — `(alpha, beta, fade_*)` or `(sigma, cutoff_level)` — never on
`f` or `phi`.** This is what makes one FFT per frame serve every frequency, and it is the load-bearing
fact of the whole design. It is also the whole of what a second atom kind had to satisfy.

**Hop scales as `1/alpha`, not as a fraction of the window.** Onset capture falls off as
`exp(-2*alpha*|delta|)`, a width set by the decay rate. An MPTK-style `hop = L/2` captures ~1e-4 of an
atom's energy here. Hop is *measured* from each block's own envelope autocorrelation rather than
derived, so it accounts for `beta` and the fade tail. Because the loss depends on `alpha`, this is a
correctness property: a fixed hop would bias selection toward small-`alpha` blocks. A Gaussian's
capture falls as `exp(-delta^2 / 2 sigma^2)`, so its hop is proportional to `sigma` — and it is
measured by the same code, which is what keeps ranking across the two kinds fair.

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

**`rmp` analyses and `rmpsynth` renders; nothing crosses.** `rmp` used to render a book too, with
`--book` read rather than written when no soundfile was given. Both are gone: `--book` is always an
output, and a hidden `-o` survives only to point a stale command line at `rmpsynth`. Analysis needs
at least one of `--book`, `--residual`, `--residual-book`. The pursuit still renders every atom it
subtracts — that is analysis, not synthesis — and `rmp_synthesis::atoms` renders a book through
exactly that call, so the two cannot drift.

**The pipeline reports facts; the front end formats them.** `pipeline::analyse` is what the `rmp`
binary's `analyse` used to be, with every `eprintln!` taken out of it, and it is the reason a GUI
cannot decompose the same input differently from the command line. Two consequences are worth
keeping:

- **The `Event` stream carries only what happens while work is in progress** — the dictionary, the
  window plan, a finished window. Everything a front end reports at the *end* (atom counts, SNR,
  how many atoms refined, the residual's level, the refresh counters) is a field of `Analysis` or
  is derived from the book, so it needs no event and can be presented in any order. Adding an
  event for something already on `Analysis` is how the two halves start disagreeing about when it
  is true.
- **Nothing in `pipeline` opens a file.** Reading the soundfile, cutting the excerpt out of it and
  deciding where each result goes are the caller's, which is what leaves a front end free to
  analyse a buffer it never read from disk. `Analysis` therefore returns the `ResidualBook` beside
  the `Book` rather than embedded in it; whether the two share a file is `rmp`'s decision, not the
  pipeline's.

**Cancellation is sticky, and that is load-bearing.** `Reporter::cancelled` is polled once per
selected atom by `Mp::run_with` and once per window by `run_windowed`, and `Analysis::cancelled` is
decided by asking *again* after the run returns — because the book of an interrupted pursuit is a
perfectly ordinary book and nothing in it says it stopped early. A flag that could go back to false
would report a completed run. `Mp::run` delegates to `run_with` with a constant false, so every
existing call site, bench and bit-identity gate is untouched.

`task::Running` cancels in its `Drop`, which is what closing a GUI tab relies on. A closed channel
is not enough on its own: the worker only discovers that on its next send, and a single-window run
sends nothing between starting and finishing, so it would hold a core to the end of a decomposition
nobody is going to look at. `a_cancelled_run_selects_nothing_and_reports_that_it_was_cancelled` is
the gate, paired with an uncancelled run over the same fixture so it cannot pass for being barren.

**A synthesised book is longer than the excerpt it came from.**
`rmp_synthesis::atoms::natural_len` sizes the output by rendering each atom's envelope and taking
the furthest death, where analysis sized the residual by the input. The atom tails the analysis
truncated at the excerpt end are audible again — 2.9% of the excerpt's energy on a 0.15 s piano
fixture. Over the excerpt itself the two renders are bit-identical, so this is a longer file, not a
different one.

**A book records where its excerpt began.** `Book::start_sample` is skipped when zero, so a book
analysed from the start of its file is byte-identical to one written before the field existed. The
residual book always recorded it, so when the two meet in `rmpsynth` a zero on the book defers to the
residual — which is what lets an old book with an embedded residual still line up — and only two
different nonzero origins are an error.

**Memory scales with the signal, so per-frame bytes are a design constraint.** The frame tables
are `energy` (f64), `bin` (u32) and `dirty` (bool) — **13 bytes per frame** — and a dictionary needs
`sum_b 1/hop_b` frames per input sample, which at the default grid and tolerance is over five,
because `hop ∝ 1/alpha` makes the *short* atoms cost the most table. Three things that used to sit
on top of that are gone, and must not come back:

- A `SegTree` per block, at `2 * next_power_of_two(frames)` f64 — 16–32 bytes/frame, *more than the
  table itself*. Its only reader was `mp::global_argmax`, called from a `debug_assert!` and nowhere
  else, so a release build paid for it and never read it. `global_argmax` is now a linear scan;
  seeds come from `top_seeds`, which scans linearly anyway. Measured on piano at the default config,
  removing it took the frame tables from 3838 kB to 846 kB per second of audio, **4.5x**.
- A masked *copy* of every frame table, rebuilt on each pass of `top_candidates`. `cand::FrameTable`
  now carries `dirty` and masks on read — the same predicate, no allocation.
- `cand::top_seeds` collecting and sorting every local maximum. It keeps a capped heap of the
  strongest `max(64k, 1024)` instead; the retained prefix is identical because the tie-break is a
  strict total order, and the cap doubles and retries in the case suppression exhausts it.

None of the three changed a book by a bit, and the gates that prove it are the `-c mp_1.toml`
reference book and the whole test suite.

**The envelope cache is bounded, and that is what made long clips possible at all.**
`refine::EnvelopeCache` is keyed on exact shape bits (`Shape::cache_key`: `(alpha, beta)` for a FOF,
`(sigma, cutoff_level)` for a Gaussian), and refinement moves them
continuously, so its hit rate *across* atoms is nil — but it used to retain every entry for the life
of the pursuit. Each is a full envelope, hundreds of KB at a low `alpha_min`. Measured on
`lux-eterna-1.toml`, that cost **7 MB per selected atom**: 2.3 GB at 300 atoms, and the config's own
`max_atoms = 7500` would have needed ~50 GB. It now carries a 16 M-sample budget and clears itself
whole when exceeded, which is safe at any point because it is a pure memo. RSS is flat in atom count
afterwards. Do not "improve" the eviction into something per-entry: a whole search must fit, and
clearing whole is what keeps the current search's working set intact.

**A signal too large for the frame-table budget is windowed, and that is the one thing in the crate
that changes which atoms are selected.** `mp::WindowPlan` sizes a *core* from `max_memory_mb` and the
dictionary's own per-sample cost, plus a *guard* of one longest atom — the longest block support, or
`refine.max_atom_samples` when refinement is on. `mp::run_windowed` runs `Mp::with_core` over each,
writes the window's residual back before the next reads it, and translates onsets and the running
residual energy into global coordinates. Four things that are load-bearing:

- **A single window takes the original path verbatim**, so a signal under the budget is bit-identical
  to what the un-windowed pursuit produced. That is both the compatibility guarantee and what keeps
  every existing oracle gate meaningful; `a_signal_inside_the_budget_is_one_window_and_bit_identical`
  is the test.
- **The stopping rule reads the core, not the window.** The guard belongs to the *next* window and
  will be decomposed there, so counting its energy makes every window look permanently unfinished
  and spend its whole atom budget failing to finish. `signal::subtract_at_core` tracks both energies
  in one pass — the core's for the stopping rule, the whole atom's for `energy_removed`, because an
  atom's removal is a fact about the atom and not about which window selected it.
- **A window is never shorter than four guards**, whatever the budget says. Below that every window
  re-correlates more guard than core, and an atom deferred out of one window's tail lands in the next
  window's tail again. A low `refine.alpha_min` therefore sets a floor on how finely a clip can be
  windowed, and the CLI says so rather than silently thrashing.
- **The book comes out in time order rather than energy order.** `residual_energy` is still a global
  running total, so `rmpstat snr` reads a correct curve, but its *shape* is progress through the clip
  and "atoms to 20 dB" stops meaning what it did on a single-window book.

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

### Gaussian atoms

`amp * g[n] * sin(phi + omega n)`, `g[n] = exp(-(n-h)^2 / 2 s^2)` over `n = 0..2h`, with the peak
exactly 1 and the support cut where `g` falls below `cutoff_level`. Settings are
`[dictionary.gaussian] sigmas_ms` (empty by default) and `[refine] sigma_*`. Seven facts that are not
obvious from the code:

**rmp owns the definition, so the rule "derive by rendering, never by formula" does not apply.**
That rule exists because a formula for a FOF would drift from rfofs's rounding. There is no second
implementation of a Gaussian to drift from: the formula *is* the atom, for analysis and resynthesis
alike, and `half_len` is a closed form. The carrier is an exact f64 `sin`, not rfofs's LUT.

**`Shape` is serialised untagged, and that is what keeps every FOF book byte-identical.** A book's
`env` is the inner struct with no kind tag; the variants are told apart by their fields, and
`GaussianParams`'s `deny_unknown_fields` keeps the match unambiguous.
`a_fof_shape_serialises_exactly_as_its_envelope_did` pins it. A tagged enum would have rewritten
every book ever produced.

**Blocks are built FOF family first, and that is what keeps FOF decompositions bit-identical.** Block
index is both what a book records and the selection tie-break, so appending the Gaussian blocks
leaves every FOF block where it was. Verified: re-analysing the `mp_1.toml` piano reference after the
change gives a byte-identical book and residual.

**The `sigma` stage holds the centre, not `t0`.** `t0` is the first sample of the support for both
kinds, but a Gaussian's peak sits `half_len(sigma)` later. Searching `sigma` at a fixed `t0` would move
the peak by ~3.7 samples per sample of `sigma` and turn a width search into a coupled width-and-onset
search. `recovers_an_off_grid_gaussian_about_its_centre` seeds from the wrong rung and recovers the
centre to 4 samples. The seed there is centred on the frame grid, because that is where the coarse
argmax puts a symmetric atom — seeding the *start* of the support at the true start leaves the centre
hundreds of samples off, further than the onset stage's hop-derived radius can walk.

**A Gaussian's fit region is its whole support.** `fit::fit_end` exists to exclude a FOF's release,
which says nothing about `alpha` and `beta`. A Gaussian has no release; its tails are already at the
cutoff.

**The refinement cache key for a FOF ignores the release, and `adopt` keeps the seed's `fade_dur`.**
Both were true before there was a second kind and both are kept exactly, because changing either moves
every refined book: a hit serves whichever release the first request rendered, and a refined FOF
records its seed's `fade_dur` rather than the one its policy would derive.

**HRMP's legacy mode scales a Gaussian by `sigma / 2^depth`**, the one-parameter analogue of a FOF's
`alpha * 2^depth, beta / 2^depth`. `probe_shape` is the single definition, and
`a_long_gaussian_bridging_a_gap_is_caught_in_both_modes` covers both probe modes on a symmetric atom.

**Measured, on the first 3 s of `chopin-nocturne-2.wav` at `mp_1.toml` settings to 35 dB:** FOF only
833 atoms / 5.53 s / −32.2 dB residual peak; FOF + `sigmas_ms = [1, 2.5, 6, 15, 40]` 832 / 5.70 s /
−31.4 dB, with the Gaussians taking 110 atoms and 2% of the energy; Gaussian only 862 / **1.61 s** /
−33.8 dB. The Gaussian-only speed is the longest transform — 14,273 samples at 40 ms against 332,053 at
`alpha = 1` — not anything cheaper per frame. One run each. The atom render of the mixed book null-tests
against the input at 35.01 dB, the reported SNR.

### Synthesis in `rmpsynth`

**Not `rfofs::OfflineRenderer`.** It renders in engine blocks, and a FOF's `decay_acc` is a running
product across `fill_block` calls, so a block-split grain is not bit-identical to the single-call
render the pursuit subtracted. It also writes a float WAV directly — nothing can be mixed in — needs
monotonic non-negative onsets, and stops at a 30 s safety limit. `rmp_synthesis::atoms` uses
`FofState` per atom through `AtomParams::render`, and the gate is exact: the pre-change `rmp -o`
piano render and `rmpsynth -b` of the same book agree sample for sample over the excerpt.

**Atoms and residual share one timeline.** Both are placed at the excerpt's source sample, or both at
zero with `--trim-to-excerpt`, and whichever ends later sets the length. The mix is `(atoms +
residual) * gain` in f32, so at 0 dB an atoms-only render is the atom render to the last bit.

**The noise streams run through the timeline lead-in.** The residual renderer draws noise for every
output sample, including the silent ones before the excerpt, so a trimmed render is a different
stretch of the same stream from a placed one. Both are correct realisations; they are not
sample-comparable, and a test that compares them has to silence the residual.

### Residual ERB analysis

`rmp_residual_erb_analysis_spec.md` is the written specification. Five facts that are not obvious
from the code:

**The unit-noise-power gain is measured, not derived.** Each band's `g_b` is `1/sqrt(sum h[n]^2)`
over its own rendered impulse response, so `(1/2pi) * integral |H_b|^2 dw = 1` exactly, by Parseval.
A closed-form pole expression would have to model both the `2*Re(.)` negative-frequency image and
its cross term, and would be a second definition of a filter that already exists. This is the same
rule the rest of the crate follows for support lengths. Only the *truncation point* is a formula —
the envelope peaks at `(N-1)/decay` and running `64/decay` past it leaves a tail below 1e-23.

**Bands write disjoint columns, so the parallel result is bit-identical.** Each band owns its filter
state, detector state and output column; the columns are gathered in band order by an indexed
`collect`. `the_parallel_bank_matches_a_serial_reference` checks it against the spec's own
sample-major reference loop, which shares no structure with the chunked band-major one that runs.

**A frame is the detector state *after* its own sample, not before.** Chunking the residual by
`update_samples` is what makes that structural rather than an off-by-one waiting to happen: the
first sample of chunk `k` is sample `k*Nu`, and the frame is written the moment it has been through
the detector.

**`serde_json` needs `features = ["float_roundtrip"]`.** Its default parser is a fast path that is
not correctly rounded — it reads `1842.6232639284315` back as `...17`. Every f64 in a book goes
through it, the energies included, so without the feature a book does not survive its own write and
read. Found by the residual bank's centre frequencies, but it was always true.

**The summed band power is not the residual's energy.** The bands overlap and each reports a
density, so on 0.5 s of piano the sum sits 27 dB above the residual's variance — explained entirely
by a bandlimited residual read through 48 overlapping unit-noise-power bands. What it does do is
track the residual in time: 0.74 correlation against short-time power over 20 ms windows, peaks one
window apart. Both figures are on material with structure; on a flat noise floor there is nothing to
correlate and the figure means nothing.

Costs: 0.5 ms of analysis for 0.5 s of audio at 48 bands, against seconds for the pursuit. The book
is the expense — 48 bands at 1 ms is 48000 f32 per second of audio, roughly 12× the atom list on a
0.5 s piano excerpt, which is why `--residual-book` exists.

### Residual ERB synthesis

`rmp_residual_synthesis_spec.md` is the written specification. Five facts that are not obvious from
the code:

**The spec's §6 and §7 prescribe different levels, and applying both cancels.** §6 says
`g_b = sqrt(P_b / C_b)` with `C_b` the synthesis band's own noise power; §7 says scale the bands so
`sum_b |H_b|^2 = 1` and drive them at `sqrt(P_b)`. Dividing by `C_b` undoes exactly that scaling.
§7 is the one to follow, and the reason is the invariant already recorded above: a unit-noise-power
band makes `P_b` a weighted *average* of the residual's PSD, a density, not a share of its energy.
Output PSD is `sum_b g_b^2 |H_b|^2`, so `sqrt(P_b)` into a complementary bank reproduces the
residual's spectrum, while §6 read with `C_b = 1` reproduces the 27 dB overshoot instead. Measured
on 1 s of piano the reconstruction lands **0.45 dB** below the residual it came from;
`white_noise_survives_the_round_trip_at_its_own_level` is the regression that encodes it.

**The calibration is a least-squares fit, not a division at the band centres.** The obvious rule —
divide each band by `W(f_b)` — was tried first: it drives `W` to exactly 1 at the centres and leaves
the dips between them, scalloping the default 48-band bank by 0.55 dB. The ISRA multiplicative
update `c_b^2 *= sum_i m_b[i] / sum_i m_b[i] W[i]` weights by the band's own response, so a band can
see the gap it is meant to fill: 0.30 dB, RMS 0.072 → 0.044. Both are deterministic; §10 rules out
Monte Carlo, and a fixed 16384-point grid also means the reported diagnostics repeat exactly.

**How flat `W` comes out is a property of the bank, not of the fit.** 48 bands hold to 0.30 dB and
64 to 0.08, but 24 bands put the centres 1.7 ERB apart and *no* choice of scales fills between them
— 5 dB of scalloping, and an audible comb. `a_sparse_bank_cannot_be_made_complementary_and_says_so`
pins it, and the CLI warns past 1 dB. It is a reason to analyse with 48 bands or more.

**The noise stream's variance is exactly 1, and the calibration depends on it.** The bands are
scaled against unit-variance white noise, so the uniform `[-1, 1)` that `residual::pseudo_noise`
produces — variance 1/3 — would put the whole render 4.8 dB low. `rng` emits uniform
`[-sqrt(3), sqrt(3))` and `the_stream_has_unit_variance` is what stops that drifting.

**The block size cannot reach the output, and that is structural.** Each block is split at the
book's exact frame boundaries, and every band's noise stream, filter state and gain pole run
continuously across the splits — so a frame reload is idempotent and block boundaries need no
special case. `the_block_size_does_not_change_the_output` compares 1, 7, 48, 64, 100 and 256 bit for
bit.

Nothing here is parallel. 48 bands of a fourth-order complex cascade is ~1500 flops per output
sample: 8 s of audio renders in 0.22 s, of which 0.06 s is the one-off bank calibration — about 50x
realtime serially. Rayon would buy nothing and would cost the bit-identical determinism the seed
exists to provide.

**What it does not reproduce is the peak.** At matched rms the piano residual's reconstruction peaks
~10 dB lower: the impulsive part of the residue is exactly what a stochastic model does not carry.
That is the model's boundary, not a bug, and it is why the residual peak is worth reporting.

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

`cargo bench -p rmp-core --bench pursuit` splits one decomposition by stage. At 0.25 s of off-grid audio, each
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

## Build configuration — four things that will bite

**The root manifest is virtual: it has a `[workspace]` and no `[package]`.** `cargo build`, `cargo
test` and `cargo clippy --all-targets` from the root act on all four members, so the everyday
commands are unchanged; anything that names a *target* needs the crate that owns it
(`-p rmp-core --bench pursuit`, `-p rmp-cli --example analyze`).

**None of the three binaries needs a `[[bin]]` section.** Cargo auto-discovers `src/bin/<name>/`
directories, so `rmp`, `rmpstat` and `rmpsynth` are all found by their directory names.

**`cargo test` runs each crate with its own package root as the working directory, and
`CARGO_MANIFEST_DIR` one level below the workspace root.** The `data/` fixtures are at the root, so
the paths that reach them are `include_str!("../../data/config/...")` in `config.rs` and
`env!("CARGO_MANIFEST_DIR")` + `"../data/books/book1.json"` in `tfmap.rs`. A new fixture path has to
account for the same step.

**`fftw` must keep `features = ["system"]`.** The crate's default `source` feature vendors FFTW 3.3.8
with no SIMD flags at all — the generated `config.h` has `HAVE_AVX`, `HAVE_AVX2`, `HAVE_SSE2` all
`#undef`, producing scalar code ~3× slower. Requires `libfftw3-dev`. Note the **engine uses realfft
only**; `fftw` exists solely for `rmp-core/benches/fft.rs`.

**`.cargo/config.toml` sets `-C target-cpu=native`.** rfofs sets the same flag, and it does *not*
propagate across a path dependency. rfofs's `wide::f32x8` SIMD width decides which sine approximation
each sample gets (degree-9 polynomial for full lanes, LUT for the scalar tail), so a mismatched build
changes atom values between the crates. An explicit `RUSTFLAGS` env var overrides `build.rustflags`
rather than merging — don't set both.

`rustfft` is deliberately absent from `[workspace.dependencies]`; it still arrives transitively via
`realfft`.

## Comparing two decompositions

**A WAV written twice is not byte-identical**, because libsndfile stamps a timestamp into the PEAK
chunk of a float file. Exactly one byte differs and the audio data is untouched, but `cmp` on the
file reports a difference and reads as nondeterminism. Compare the book, or the data past byte 72.

## The FFT benchmark

`rmp-core/benches/fft.rs` compares realfft against FFTW at 1024–32768, the range FOF supports land
in (`support ≈ 6.9·sr/alpha`).

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
