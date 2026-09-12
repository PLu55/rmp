# rmp — user manual

`rmp` decomposes a soundfile into FOF atoms (Rodet's Formant Wave Function) and, optionally,
Gaussian atoms by Matching Pursuit, and writes them as a *book*. A book's FOF atoms replay through
the `rfofs` synthesizer unchanged; its Gaussian atoms are rmp's own.

Three binaries: `rmp` analyses, `rmpsynth` renders a book — its atoms, its stochastic residual, or
both (§10) — and `rmpstat` reports on it.

This manual covers the command line, every setting, and what each one costs. For the design and its
internals see `CLAUDE.md`; for the algorithm see `notes.md`.

---

## 1. The mental model

Three stages, and every setting belongs to exactly one of them.

**The dictionary** is a set of *blocks*. A block fixes one envelope shape and offers it at every
onset on a time grid and every frequency on an FFT grid. There are two kinds of shape, each its own
family: a **FOF** — a decay rate `alpha` and an attack duration `beta`, a sharp onset and an
exponential tail — and a **Gaussian** — a width `sigma`, symmetric about its peak.
`[dictionary.fof]`, `[dictionary.gaussian]`, `[envelope]` and `[blocks]` build it. Every block of
either kind competes for every atom on equal terms.

**The pursuit** repeatedly finds the single best-matching atom in the dictionary, subtracts it from
the residual, and repeats. `[pursuit]` decides when to stop.

**Refinement** takes each selected atom off the grid before subtracting it, adjusting
`(t0, f, alpha, beta)` for a FOF or `(t0, f, sigma)` for a Gaussian by local search. `[refine]`
controls it. `[hrmp]` optionally vetoes atoms the residual does not support across their whole length.

The relations worth carrying in your head, because most settings act through them:

```text
FOF       -3 dB bandwidth  ≈  alpha / pi          Hz        alpha = 100   ->   32 Hz wide
          atom length      ≈  6.9 * sr / alpha    samples   alpha = 100   ->  3300 samples at 48 kHz
          attack duration  =  beta                seconds
Gaussian  -3 dB bandwidth  ≈  0.265 / sigma       Hz        sigma = 5 ms  ->   53 Hz wide
          atom length      ≈  7.4 * sigma * sr    samples   sigma = 5 ms  ->  1785 samples at 48 kHz
```

So **`alpha` is the one parameter that sets both an atom's frequency width and its length**, in
opposite directions. Low `alpha` means a long, narrow, sustained atom; high `alpha` means a short,
wide, percussive one. Almost every cost in the program scales with the longest `alpha` in the
dictionary.

---

## 2. Running it

```bash
# analyse: needs at least one output
rmp in.wav -b book.json.gz                     # the atoms
rmp in.wav -b book.json.gz -r residual.wav     # and what they could not explain
rmp in.wav -b book.json -c settings.toml       # with settings
rmp in.wav -b book.json -s 2.5 -d 0.5          # one excerpt, in seconds
rmp in.wav -b book.json --residual-book bank.json.gz   # atoms, plus the ERB residual analysis

# render: rmpsynth, not rmp (§10)
rmpsynth -b book.json.gz -o resynth.wav

rmp --write-config > settings.toml             # a fully commented default document
```

`rmp` does not render books. An old `rmp ... -o out.wav` command line fails with the `rmpsynth`
command to use instead.

| flag | meaning |
| --- | --- |
| `-c`, `--config` | settings document. Defaults are used if omitted |
| `-r`, `--residual` | what the decomposition could not explain |
| `-b`, `--book` | the atoms, written here |
| `-s`, `--start` | offset into the file, seconds |
| `-d`, `--duration` | length to analyse, seconds |
| `-q`, `--quiet` | suppress the report |
| `--residual-analysis` | analyse the final residue into ERB power frames (§9) |
| `--no-residual-analysis` | skip it even if the settings document enables it |
| `--residual-update-ms` | frame interval, overriding `[residual] update_ms` |
| `--residual-book` | write that analysis to its own file instead of into `--book` |

Book format follows the extension: `.toml` or `.json`, either optionally `.gz`. Gzip is worth about
7× and costs nothing to read back. `book.json.gz` is JSON; a bare `book.gz` is TOML.

Multi-channel input is downmixed to mono by averaging, which partially cancels out-of-phase content
between channels. The run says so when it happens.

Atom onsets are relative to the excerpt, not the file: with `--start 2.5`, onsets in the book count
from that point. The book also records where the excerpt began, so `rmpsynth` puts the render back
at 2.5 s in the source's timeline — or at zero with `--trim-to-excerpt`.

### Reading the report

```text
in.wav: 29.07 s, 48000 Hz, 2 channel(s)
  analysing 5.000-8.000 s (144000 samples from 240000)
  downmixed to mono; out-of-phase content between channels partially cancels
dictionary: 24 blocks (24 fof) in 13.68ms
  note: 8 of 24 blocks are longer than refine.max_atom_samples (150000), so their
        atoms stay on the grid unrefined; longest support 332053 samples (fof alpha 1.000 beta 0.30 ms)
analysis: 1528 atoms, 35.0 dB in 12.64s (init 13.68ms, 4.2x realtime)
  refined: 1291/1528 atoms moved off the grid (84%)
  refresh: 1266438 frames bounded, 416712 recomputed (32.9%)
residual: -35.0 dB rms, -30.3 dB peak relative to input
  absolute: -73.1 dBFS rms, -57.2 dBFS peak (input -38.1 dBFS rms, -26.9 dBFS peak)
```

- **`4.2x realtime`** is wall clock over audio duration. Below 1.0 is faster than realtime.
- **`refined: N/M`** — how many atoms moved off the grid. Well under 100% with refinement enabled
  means `max_atom_samples` is blocking blocks; see §6.
- **`kinds:`** — printed only when the dictionary holds both families: how many atoms of each kind
  were selected, and what share of the removed energy each carries.
- **`refresh: bounded/recomputed`** — the lazy update's hit rate. Lower recomputed % is faster; it
  is diagnostic, not a setting.
- **`-35.0 dB rms`** is the negated SNR, so it restates the line above.
- **`-30.3 dB peak`** is the one that adds information. It is where the decomposition is *worst*
  rather than where it is on average, and it is the most useful single quality number in the report:
  a badly-placed atom shows up here and nowhere else.

---

## 3. `[dictionary.fof]` and `[dictionary.gaussian]` — which envelope shapes exist

The seed grid, one section per atom kind. With refinement on, these are starting points rather than
the final parameters, so each grid needs to *bracket* the material rather than resolve it.

Either family may be empty, but not both. The blocks are built FOF family first, so adding Gaussians
to a document leaves its FOF blocks — and their indices in the book — exactly where they were.

**A settings document from before the families existed** puts `alphas` directly under
`[dictionary]`. That is refused with a message saying so: add a `[dictionary.fof]` header above
`alphas`, `betas_ms` and `alpha_beta_max`.

### `[dictionary.fof] alphas` — default `[80, 128, 205, 328, 524, 839, 1342, 2147]`

Decay rates in s⁻¹, one block per `(alpha, beta)` pair.

**Result.** The range must cover the material. Sustained instruments want low `alpha` — on solo
piano, 90–96% of the removed energy sits in `alpha < 8`. Percussive material wants high. Spacing
matters much less than range once refinement is on: a ratio of about 1.6 between neighbours costs
roughly 5% of an atom's energy at the worst point between two rungs, and refinement recovers that.

**Cost.** This is the single most expensive setting in the file, and it is superlinear in the lowest
value. A block's transform is its atom length, `≈ 6.9·sr/alpha`, so `alpha = 1` at 48 kHz is a
332,053-sample support and a 337,500-point FFT — 80× the work per frame of `alpha = 80`. Adding one
low rung costs far more than adding several high ones.

Measured, 3 s of piano at a fixed 35 dB: dropping `alpha = 1` from a six-rung grid took the
dictionary build from 64 ms to 32 ms and the analysis from 9.9 s to 8.4 s. Keeping only
`[16, 64, 256]` built the dictionary 26× faster (2.4 ms) and cost 6% more atoms.

### `[dictionary.fof] betas_ms` — default `[0.3, 1.0, 3.0]`

Attack durations in milliseconds — the half-cosine rise at the atom's onset.

**Result.** Shapes only the first few milliseconds. It matters for transient character, not for the
body of the atom. Refinement moves it freely, so two or three well-spread rungs suffice.

**Cost.** Multiplies the block count directly. Fewer rungs is not reliably faster, though: measured,
cutting `[0.3, 1, 3, 12]` to `[1, 12]` made a piano run *slower* (16.3 s against 9.9 s), because the
seeds fit worse and the pursuit needed more of the expensive long blocks.

### `[dictionary.fof] alpha_beta_max` — default `4.0`

Drops `(alpha, beta)` combinations whose product exceeds this.

**Result.** A guard, not a tuning knob. rfofs renders `alpha*beta > 10` as silence outright, and its
amplitude normalisation is ill-conditioned well before that. Raising it above ~4 admits blocks whose
peak normalisation is unreliable. Leave it.

### `[dictionary.gaussian] sigmas_ms` — default `[]` (off)

Envelope standard deviations in milliseconds, one block per value. The atom is
`amp · exp(-(t-c)²/2σ²) · sin(2πft + φ)`, symmetric about its centre, with a peak of exactly `amp`.

**Result.** A FOF has a sharp attack and a long exponential tail; a Gaussian has neither, so it fits
what a FOF fits badly: events that swell and fade symmetrically, bowed or blown tones, and the parts
of a sustained sound between its attacks. A ratio of about 2.5 between rungs matches the default
`refine.sigma_bracket`, so `[1, 2.5, 6, 15, 40]` covers 6.6–265 Hz of bandwidth with no rung out of
refinement's reach.

**Cost.** Like `alphas`, set by the longest atom: a block's transform is `≈ 7.4·σ·sr` samples, so
`σ = 40 ms` is a 14,300-point transform — about a FOF at `alpha = 23`. Its hop grows with `σ` too,
so a wide rung costs fewer frames than its length suggests.

### `[dictionary.gaussian] cutoff_level` — default `0.001` (−60 dB)

Where the support is cut, relative to the peak.

**Result.** The cut is a hard step of this size, which sets a leakage floor near it. Leave it unless
the residual shows a floor at that level.

**Cost.** The length grows as `√ln(1/cutoff_level)`: halving it lengthens every Gaussian by about 5%.

### What a Gaussian family is worth, measured

The first 3 s of `data/audio/chopin-nocturne-2.wav` at `mp_1.toml`'s settings, every arm driven to
the same 35 dB. One run each, so treat differences under 5% as noise.

| dictionary | blocks | atoms | wall | residual peak |
| --- | --- | --- | --- | --- |
| FOF only — `mp_1.toml` as shipped | 24 | 833 | 5.53 s | −32.2 dB |
| FOF + `sigmas_ms = [1, 2.5, 6, 15, 40]` | 29 | 832 | 5.70 s | −31.4 dB |
| Gaussian only — the same ladder, `alphas = []` | 5 | 862 | 1.61 s | −33.8 dB |

**Adding the family to a FOF dictionary changed nothing measurable here.** The pursuit took 110
Gaussians, carrying 2% of the removed energy, for the same atom count and wall clock: piano partials
have a sharp attack and an exponential tail, which is a FOF's shape, so the FOFs win the competition.

**A Gaussian-only dictionary is the surprise.** It reached the same SNR with 3.5% more atoms, 3.4×
faster and with a better residual peak — its longest block is a 14,000-sample transform against the
FOF family's 332,000. On this material that makes it a serious fast arm rather than an add-on. It is
one excerpt of one instrument, so measure it on yours before relying on it.

`rmpstat diag` flagged 37% of the Gaussian seeds on the 1 ms and 40 ms rungs, and refinement walked
`sigma` out to 78 ms: on sustained material the ladder wants a wider top rung than the suggested one.

---

## 4. `[envelope]` — the release, fixed for the whole analysis

rfofs's release is a linear ramp to zero, entered where the exponential decay reaches `fade_level`.
The same policy is used for analysis and resynthesis, so a book always replays as it was fitted.

### `fade_level` — default `0.001` (−60 dB)

Where the ramp begins, relative to the atom's peak.

**Result.** Sets how much of the exponential tail is kept. Lower keeps more.

**Cost.** Enters the atom length directly: `length ≈ -ln(fade_level)·sr/alpha`. Halving it to 0.0005
lengthens every atom by 11% and every transform with it. Raising it to 0.01 shortens them by 33%.
This is a cheaper lever on cost than `alphas` and a blunter one on quality.

### `fade_dur_scale` — default `2.0`, with `fade_dur_min_ms` / `fade_dur_max_ms` — default `1.0` / `10.0`

The ramp lasts `fade_dur_scale / alpha` seconds, clamped to the bounds.

**Result.** Almost none. The release carries under 1e-4 of an atom's energy.

**Cost.** Adds to the atom length and so to every transform. It is derived from `alpha` rather than
fixed because a constant duration would run several times longer than the body of a short atom. Do
not set a large fixed release by pushing both clamps together.

---

## 5. `[blocks]` — the time and frequency grids

### `capture_tolerance` — default `0.95`

**The main speed lever.** The worst-case fraction of an atom's energy a frame must still capture
when the true onset falls between two hop positions. It sets the hop, and the hop sets how many
frames exist.

**Result.** A coarser hop means the coarse search ranks candidates from a worse starting point.
Without refinement this is a correctness property — a fixed hop would bias selection toward
low-`alpha` blocks and make cross-block ranking unfair — so **leave it at 0.95 if `refine.enabled`
is false.** With refinement on, the seed only has to be close enough to refine from.

**Cost.** Frame count scales as `1/ln(1/tolerance)`, so 0.5 has about 13× fewer frames than 0.95,
and the profile is ~85% transforms either way.

Measured on 3 s of piano with refinement on: 0.95 → 28.4 s and 978 atoms; 0.5 → 6.6 s and 1057
atoms. About 8% more atoms for 4× the speed. **0.3 is slower again** (19 s) — the seeds get bad
enough that the pursuit needs many more of them, so this is an optimum rather than a monotone knob.

The trade is sharper on adversarial input. On a synthetic that plants every atom exactly `hop/2`
off-grid, 0.5 alone is a real regression: 193 atoms to 40 dB against 102, with the splitting factor
at 10.5 against 4.5. `candidate_count = 8` recovers it fully (95 atoms, 4.8) — but on real material
extra candidates buy nothing. See §7.

### `f_min` / `f_max` — default `50.0` / `10000.0` Hz

The frequency range each block represents.

**Result.** Content outside the range is unrepresentable and simply stays in the residual. Setting
`f_max` below the material's brightness leaves a bright residual; setting it near Nyquist wastes
bins on content the dictionary handles badly.

**Cost.** Linear in the bin count, which is the bin scan — about a third of a refresh. Narrowing
`f_max` from 10 kHz to 3 kHz removes 70% of the bins.

DC and Nyquist are excluded unconditionally whatever you set: the sine basis vector is identically
zero there, so the projection is exactly rank-1.

### `rho_sq_max` — default `0.9999`

Disables bins where the sine and cosine basis vectors are nearly parallel and the 2×2 projection is
ill-conditioned. In practice this is the low-frequency edge, where an atom spans less than a carrier
period.

**Result.** A numerical guard. Lowering it disables more of the low band; raising it admits bins
whose amplitude and phase are unreliable. Leave it unless `rmpstat diag` reports ill-conditioned
atoms.

---

## 6. `[refine]` — moving atoms off the grid

**This is the largest single quality win available**, and it is on by default. Without it, off-grid
input needs about 12× more atoms than on-grid input to reach the same SNR, because each miss is
patched with a cluster of partial atoms instead of being represented by one. With it, that falls to
about 3×, and 99% of selected atoms move.

Refinement is now the *expensive* part of an iteration — about 2.8 ms per atom against 1.5 ms
without — because the refresh around it was parallelised and refinement was not.

### `enabled` — default `true`

Turning it off gives you the plain grid pursuit. Do that only to reproduce a grid-only book; then
restore `capture_tolerance = 0.95`, because the fine hop is doing the work refinement otherwise does.

### `max_atom_samples` — default `65536`

**The setting most likely to surprise you.** It rejects any refined envelope longer than this — but
`refine` asks the envelope cache for the *seed's own shape* before anything else, so a block whose
support already exceeds the cap can never be refined at all. Its atoms are still selected, just
pinned to grid frequency, grid onset and grid envelope.

At the default 65536 against a dictionary reaching `alpha = 1`, that was 12 of 24 blocks and 34% of
the atoms in a book — and worth **13 dB of residual peak**, entirely invisible in the rms figure.
The run now prints a note when blocks exceed the cap, and a `refined:` line saying what fraction of
atoms actually moved.

**But capping deliberately below the longest block is also the best setting measured.** At
`rounds = 2` on 3 s of piano, 150000 gives a −30.3 dB residual peak where 400000 gives −24.9:
letting refinement chase seven-second atoms it cannot converge on in two rounds is worse than not
letting it start. At `rounds = 3` the ordering reverses (−29.7 against −30.6) but costs 17 s against
27 s. **Treat it as a regularizer, not just a budget** — and read the `refined:` line to see what
you actually got.

Rule of thumb: set it a little above the *second*-longest block, not the longest.

### `alpha_bracket` — default `1.6` · `beta_bracket` — default `3.5` · `f_bracket_bins` — default `1.0`

How far from the seed each parameter is searched — multiplicative for `alpha` and `beta`, in FFT
bins for `f`.

**Result.** The defaults reach the neighbouring grid rung in each direction, so no true value is out
of reach of a correctly-bracketing dictionary. Widening helps when the dictionary is sparse.

**Cost. A wider `alpha_bracket` is *faster*, not slower** — the one genuinely counterintuitive
result here. Measured at `max_atom_samples = 400000`: 2.5 took 27.4 s where 1.6 took 59.2 s, for the
same atom count and residual peak. Golden section covers the range in fewer rounds, so the
`score_tol` early exit fires sooner. 4.0 is slower again (39.5 s) with no quality gain, so 2.5 is
about the optimum.

### `rounds` — default `3` · `golden_iters` — default `10` · `score_tol` — default `0.0001`

Work per candidate. A round is one `f → alpha → beta → t0` cycle for a FOF, `f → sigma → t0` for a
Gaussian; `golden_iters` is the
golden-section evaluations per parameter; `score_tol` ends the search when a whole round gains less
than that fraction.

**Result.** More rounds is not reliably better. Measured at `max_atom_samples = 150000`, `rounds = 2`
beat `rounds = 3` on *both* axes — 12.6 s and −30.3 dB against 17.0 s and −29.7 dB. Refinement's
remaining error sits in `(t0, alpha, beta)`, which all shape the attack and trade against each other
along a shallow valley that coordinate descent walks down but not along; extra rounds walk further
down the same valley without moving along it.

**Cost.** Both are close to linear. `golden_iters = 6` is the worse of the two levers: holding
everything else at `max_atom_samples = 400000, alpha_bracket = 2.5`, it saved 25% of wall clock and
cost **6 dB** of residual peak, where `rounds = 2` saved 26% and cost nothing. Prefer `rounds`.

### `alpha_min` / `alpha_max` — default `40.0` / `4000.0` · `beta_min_ms` / `beta_max_ms` — default `0.1` / `10.0`

Hard bounds on where refinement may go, deliberately wider than the dictionary grid at both ends.

**Result.** These stop the optimizer wandering somewhere non-physical. If your `alphas` reach below
`alpha_min`, refinement cannot even hold the seed value — set `alpha_min` at or below your lowest
rung. The frequency range and the `alpha*beta` cap are *not* here: they are shared with `[blocks]`
and `[dictionary.fof]` so the coarse and refined paths cannot disagree about what is representable.

### `sigma_min_ms` / `sigma_max_ms` — default `0.5` / `200.0` · `sigma_bracket` — default `2.5`

The Gaussian counterparts of the `alpha` bounds and bracket: where a refined `sigma` may go, and how
far from the seed it is searched, multiplicatively.

**Result.** The width is searched with the atom's *centre* held still, so it does not drag the onset
around while it moves. `max_atom_samples` applies to Gaussians as well, and a rung whose support
already exceeds it is never refined, exactly as for a FOF. Set `sigma_min_ms` at or below your
narrowest rung, or refinement cannot hold the seed.

### `t0_radius` — default `0`

Onset search radius in samples. `0` derives it from the block's own hop, which is what you want:
the seed is at most half a hop from the true onset by construction.

**Cost.** Set explicitly and large, this is pure waste — the search is over a range the seed already
brackets.

---

## 7. `[pursuit]` — when to stop

### `target_snr_db` — default `30.0` · `max_atoms` — default `1000`

The pursuit stops at whichever comes first. `max_atoms` counts *selected* atoms, not iterations, so
an iteration where HRMP rejects everything does not spend budget.

**Result.** The honest quality dial. Each 10 dB costs roughly a factor of 2–3 in atoms.

**Cost.** Directly proportional to atoms. If the report says `stopped short of the target`, the
`max_atoms` cap bound first and the SNR you asked for was not reached.

### `min_gain` — default `1e-9`

Stops when the best remaining atom would remove less than this fraction of the residual.

**Result.** A floor against grinding on numerical noise. Raise it to stop earlier on a plateau.

### `candidate_count` — default `1`

Local time-frequency maxima promoted to exact scoring and refinement each iteration. 1 is the plain
global argmax.

**Result. Usually nothing.** On real material, 1, 4 and 8 all reach the same SNR in 1057 / 1077 /
1073 atoms — the strongest seed is also the seed that refines best. It earns its keep in exactly two
situations: a coarse hop on adversarial input (`capture_tolerance = 0.5` on the synthetic needs 8 to
recover its quality), and HRMP, which can *reject* a candidate rather than merely outscore it, and
then the loop needs somewhere else to go.

**Cost.** Linear, and it is refinement cost — the expensive kind. On piano, 4 candidates cost 42%
more wall clock for the same atoms.

### `max_stalls` — default `4096`

How many consecutive iterations may pass with every candidate rejected before the pursuit gives up.
Only reachable with HRMP on.

**Result.** Must be generous. A rejection is a fact about one frame, and on dense material long runs
of them are ordinary; each stall demotes at least one frame, so a run of stalls is progress rather
than a spin. Setting it small turns a strict HRMP configuration into an early stop, which reads as
"HRMP produces far too few atoms".

### `max_memory_mb` — default `1024`

The frame-table budget, and so how long a stretch of signal the pursuit analyses at once.

The pursuit's working set is one frame table per block, and a block's frame count is
`samples / hop` with `hop ∝ 1/alpha` — so the tables grow linearly with the *signal*, at a
per-sample cost the dictionary sets. That cost is printed when it matters; on the default grid at
`capture_tolerance = 0.95` it is about **18 bytes per input sample**, which is 2.5 GB for nine
minutes at 48 kHz. Nothing about this can be streamed to disk: the tables are working state, not
results.

So a signal too large for the budget is analysed in **windows**. Each window owns a *core* it selects
atoms in, and carries a *guard* past the core — one full atom's support — so an atom starting at the
core's end is still scored on all of itself. The window's residual is written back before the next
window reads it, so every atom is subtracted exactly once and an atom the search wanted in the guard
is simply deferred to the window that owns it.

**Result.** A signal that fits the budget is one window and is decomposed exactly as it always was,
bit for bit. Past it, three things change:

- **Selection is greedy within a window, not across the clip.** A quiet passage no longer waits
  behind a loud one.
- **`target_snr_db`, `min_gain` and `max_atoms` become per-window.** Each window is driven to
  `target_snr_db` against *its own core's* energy, and `max_atoms` is shared out by duration. Local
  quality is uniform rather than front-loaded, which is usually what you want for long material.
- **The book is in time order, not energy order.** `rmpstat snr` still reads a correct global SNR
  curve — `residual_energy` is a global running total — but its shape is now progress through the
  clip rather than convergence, so "atoms to 20 dB" stops meaning what it did.

**Cost.** Each window re-correlates its own frames up front, so the initialisation is paid once per
window instead of once. That is reported separately from the pursuit time. The guard is re-correlated
by two windows, so a core much shorter than the guard wastes real work — which is why a window is
never allowed below four guards, even if the budget asks for it. You are told when that happens.

The guard is the longest atom the run can produce: the longest block support, or
`refine.max_atom_samples` when refinement is on, whichever is larger. **A low `alpha_min` therefore
sets a floor on how finely a clip can be windowed** — at `refine.max_atom_samples = 150000` the guard
is 3.1 s and no window can be shorter than 12.5 s.

### `window_seconds` — default `0` (use the budget)

Analyse windows of exactly this many seconds, ignoring `max_memory_mb`.

The budget picks a window from how much memory you have, which makes the book depend on the machine
that produced it. Set this when a run has to reproduce elsewhere. It is still raised to the
four-guard floor if it is below it.

---

## 8. `[hrmp]` — rejecting atoms the residual does not support

Off by default. Ordinary MP scores an atom by its *global* correlation, so a long atom can win by
summing evidence from two separated events and claiming the silence between them — pre-echo, and
energy invented in gaps. HRMP probes the atom's own extent and clamps or rejects it.

It is a *stricter* criterion, so it trades SNR per atom for atoms that describe events actually
present. At its best setting it costs about 6% more atoms than plain MP for the same SNR. **Anything
far above that means a setting is rejecting good atoms.**

### `phase_tolerance_deg` — default `90.0`

Rejects when a probe's local phase disagrees with the global fit by more than this.

**It saturates at 90 and only ever tightens.** The underlying sign rule rejects everything beyond a
quarter turn on its own, so 90 and 179 give bit-identical books. Going below 90 is not a mild
tightening: measured on 0.5 s of piano, 90° gave 1111 atoms at the 40 dB target, 60° gave 1078, and
**45° gave 136 atoms and 11.7 dB** — the difference between a decomposition and a failure. Dense
polyphonic material puts other events under every probe, so a local phase far from the global fit is
ordinary rather than evidence of a bridged gap.

### `depth` — default `2`

`2^depth` probes across the atom. **This is the strictness knob, not just a resolution knob**, because
rejection is "any probe disagrees" and the rate climbs with the probe count.

On the same piano fixture at 90°: depth 1 rejected *nothing* and still clamped 85% of what it passed,
at a mean amplitude ratio of 0.79 — fully engaged, just not trigger-happy — reaching 40 dB in 979
atoms against plain MP's 920. Depth 2 rejected 917 candidates and needed 1111 atoms. **On dense
material prefer depth 1.** In `legacy_scaled_fof` mode `depth` also sets the probe *scale*
(`2^-depth` of the main atom), and that mode needs a larger depth than `localized_candidate` to
reach the same locality.

### `mode` — default `"localized_candidate"`

`localized_candidate` masks the refined atom's own envelope — it tests the exact atom you are about
to subtract, and needs no second dictionary. `legacy_scaled_fof` uses smaller same-frequency FOFs,
reproducing the historical implementation. Prefer the default; use the other for comparison.

### `noise_epsilon` — default `0.2`

Each probe must see at least `1/noise_epsilon²` times the local residual noise power in atom energy
before its vote counts. This is the gate that does the work.

**Result.** Lowering it makes most probes uninformative — 0.05 left 698 of 745 uninformative, so
HRMP passed almost everything through unchanged and reached 40 dB in 897 atoms. That is close to
*disabling* HRMP rather than tuning it. Raising it makes HRMP stricter from the other end.

### `minimum_probe_energy` — default `0.001` · `min_mask_periods` — default `2.0` · `magnitude_policy` — default `"strict_min"`

Structural guards. `minimum_probe_energy` is a floor on a probe's share of the atom's energy before
it gets a vote. `min_mask_periods` skips HRMP entirely when a probe would span fewer carrier periods
than this — its 2×2 Gram cannot be conditioned, and an atom that short cannot bridge anything anyway.
`strict_min` is the original criterion. Leave all three.

---

## 9. `[residual]` — analysing what is left over

Off by default. The pursuit leaves `x = Σ FOF + r`; this turns `r` into a fixed-rate map of power
over ERB bands, which a later real-time noise bank can excite with `g_b = √P_b`. It is a strict
post-processing stage — it runs once, after the pursuit has stopped, on the pursuit's own residual
buffer, and **cannot change which atoms were selected**.

```bash
rmp in.wav -b book.json.gz --residual-analysis            # embedded in the book
rmp in.wav -b book.json --residual-book bank.json.gz      # kept in its own file
rmp in.wav --residual-book bank.json.gz                   # analysis only, no atom book
```

`--residual-analysis` / `--no-residual-analysis` and `--residual-update-ms` override
`[residual] enabled` and `update_ms`; everything else lives in the settings document. Giving
`--residual-book` turns the stage on by itself. Enabling it with nowhere to write it — no `--book`,
no `--residual-book` — warns and skips rather than doing the work for nothing.

### What a frame means

A frame is the state of every band's causal power detector **immediately after** processing the
sample at that frame's offset. With `update_ms = 1` at 48 kHz, frames sit at samples 0, 48, 96, …
Nothing is stored per frame but the powers: the grid is exact by construction, so a timestamp would
only be a second definition of it. Frames stop at the end of the residual; no decay tail is
appended.

### `[residual] update_ms` — default `1.0`

How often a frame is recorded, converted to an exact sample count once. Independent of the filter
bank, which always runs at the audio rate.

**This is the size knob.** 48 bands at 1 ms is 48 000 numbers per second of audio — far more than
the atom list. Measured on 0.5 s of piano: the atom book is 104 kB of pretty JSON, and the residual
section adds ~1.2 MB, or 129 kB gzipped. Give `--book` a `.gz` suffix, use `--residual-book` to keep
the two apart, or raise `update_ms`. There is no quantisation yet: the stored numbers are `f32`
linear power.

### `[residual.erb] bands`, `min_freq_hz`, `max_freq_hz` — defaults `48`, `50.0`, `20000.0`

Band centres are uniform on the ERB-rate scale, with the first and last sitting exactly on the
configured bounds. At the defaults that is 0.85 ERB per step, so neighbouring bands overlap
substantially — which is the point, but see the warning about summing them below.

`max_freq_hz` must stay under 0.98 × Nyquist. Above it you get an error, not a silent clamp: at
32 kHz the default 20 kHz top is refused, and you are meant to lower it deliberately. Setting
`max_freq_hz` near the analysis `f_max` is usually wrong — the residual is exactly where the content
*above* `f_max` ended up.

### `[residual.erb] order`, `filter`, `spacing`, `normalization` — defaults `4`, `gammatone`, `erb_rate`, `unit_noise_power`

`order` is the length of the complex one-pole cascade, 1 to 8. 4 is the classical gammatone; the
range exists because the cascade is generic, not because four recipes were written out. The other
three have one value each today and are in the document so a book never has to be guessed at.

`unit_noise_power` means each band's gain is measured from its own rendered impulse response, so
unit-variance white noise leaves the band with unit variance. That is what makes a band power
readable as a fraction of the residual's own variance, and what a synthesis bank needs in order to
turn one back into a gain.

### `[residual.power] mode` — default `"bandwidth_relative"`

The one-pole detector is the **only** temporal smoothing in the chain, deliberately: a residual
carries rhythm and transients the atom book does not, and blurring them throws away the part worth
keeping.

- `"fixed"` uses `tau_ms` for every band.
- `"bandwidth_relative"` uses `τ_b = clamp(tau_scale / ERB(f_b), tau_min_ms, tau_max_ms)`.

Bandwidth-relative is the default because the bands cannot all resolve the same events. A 25 Hz-wide
band rings for 40 ms on its own; asking its detector to track a half-millisecond event measures the
filter, not the signal. A 3 kHz-wide band can track it. One constant either over-smooths the top of
the bank or leaves the bottom reading its own envelope ripple.

`tau_min_ms` (default `0.5`) is what bounds how far a transient can smear; `tau_max_ms` (default
`10.0`) how long the narrow bands hold. `tau_scale` (default `1.0`) is dimensionless and moves the
whole set. At the defaults every band below about 100 Hz sits on the `tau_max` rail and everything
above about 2 kHz on `tau_min`.

### Reading the output

`RMP_RESIDUAL_DETAIL=1` prints the bank table — band, centre, bandwidth, `tau_ms`, normalisation
gain — and `rmpstat summary` reports the bank of a book that carries one.

**Do not read the summed band power as the residual's energy.** The bands overlap and are not an
orthogonal partition, and each reports a *density* rather than a share. On 0.5 s of piano the sum
sits 27 dB above the residual's own variance, all of it explained by a bandlimited residual measured
through 48 overlapping unit-noise-power bands. What the sum does do is track the residual in time:
against the residual's own short-time power over 20 ms windows, 0.74 correlation with the peaks one
window apart.

The analysis is cheap — 0.5 ms for 0.5 s of audio at 48 bands, against seconds for the pursuit — so
there is no cost argument for leaving it off once you want it.

To play the result back, see §10. If that is the plan, keep `bands` at 48 or above: a sparser bank
cannot be made power-complementary and the reconstruction combs.

---

## 10. `rmpsynth` — rendering a book

`rmp` analyses; `rmpsynth` renders. It reads a book of either kind — a full book, or a standalone
residual book — and writes a soundfile: the atoms, the stochastic residual, or both mixed.

```bash
rmpsynth -b book.json.gz -o resynth.wav                         # atoms + the book's own residual
rmpsynth -b book.json.gz --no-residual -o atoms.wav             # the atoms alone
rmpsynth -b book.json --residual-book bank.json.gz -o mix.wav   # atoms + a standalone residual
rmpsynth -b bank.json.gz -o stochastic.wav                      # a residual book: noise only
```

FOF atoms render through rfofs and Gaussian atoms through rmp's own definition, each exactly as the
analysis subtracted it: over the analysed excerpt an atoms-only render is, sample for sample, the
signal the decomposition explained. A full book with no residual section simply renders its atoms.

Which kind of book you passed is worked out from the document, so there is no flag for it. The atoms
and the residual must come from the same analysis — a sample-rate or excerpt-origin disagreement
between the book and `--residual-book` is an error.

| flag | meaning |
| --- | --- |
| `-b`, `--book` | the book to render. Same extension rules as `rmp --book` |
| `-o`, `--output` | output soundfile |
| `--residual-book` | a standalone residual book to render with the atoms, replacing any embedded one |
| `--no-atoms` / `--no-residual` | leave that component out |
| `--seed` — default `1` | the whole of the nondeterminism |
| `--gain-smoothing-ms` — default `1.0` | one-pole smoothing of the band gains |
| `--gain-smoothing-mode` — default `fixed` | or `bandwidth-relative` |
| `--gain-db` — default `0.0` | output gain, applied after mixing |
| `--encoding` — default `float32` | or `pcm24` |
| `--clip` / `--error-on-clip` | what to do about samples past full scale. Neither: write and count |
| `--trim-to-excerpt` | drop the leading silence a `--start` offset puts in (`--trim-to-residual` still works) |
| `-v`, `--verbose` | the per-band table. `RMP_RESIDUAL_DETAIL` does the same |

### How the level is decided, and why it is not obvious

A band power `P_b` is a spectral **density**, not a share of the residual's energy — the bands
overlap and each is normalised to unit noise power, which is exactly why summing them lands 27 dB
high (§9). So the reconstruction cannot simply run the analysis filters at `√P_b` and add them up.

Instead the synthesis bands are rescaled so that `Σ_b |H_b(ω)|² ≈ 1` — *power-complementary* — and
then driven at `g_b = √P_b`. With independent noise per band the output spectrum is `Σ_b g_b²|H_b|²`,
which is then the residual's own spectrum. Measured on 1 s of piano: the reconstruction lands
**0.45 dB** below the residual it came from. Without the rescaling it would be roughly 20 dB above.

The scaling is fitted at startup on a fixed 16 384-point frequency grid — a non-negative least
squares fit, no randomness — and the report prints how flat it came out:

```text
bank complementarity: -0.30 dB worst, rms 0.044, over 76 .. 18237 Hz
```

**That figure is a property of `[residual.erb] bands`, not of the fit.** 48 bands hold to 0.3 dB and
64 to 0.08 dB, but 24 bands put the centres 1.7 ERB apart and no choice of scales fills between
them — it scallops by 5 dB, and the reconstruction combs. `rmpsynth` warns past 1 dB. If you plan to
resynthesise, do not analyse with a sparse bank.

### What it does not reproduce

The peak. A stochastic model matches power, and at matched rms the reconstruction of the piano
residual peaks about 10 dB lower than the residual itself — the impulsive part of the residue is
precisely what a noise model does not carry. Everything below `min_freq_hz` and above `max_freq_hz`
is also simply absent.

### Timeline

Output sample 0 is source sample 0. A book analysed with `-s 2.0` therefore renders two seconds of
leading silence, and `--trim-to-excerpt` removes it. The atoms and the residual are always placed
together: both at the excerpt's source position, or both at zero.

The output is as long as the longer of the two. That is usually the atoms: the residual stops exactly
at the end of the analysed excerpt, with no filter tail, but the atoms' tails run on past it — the
parts the analysis truncated at the excerpt end are audible again. Nothing is ever normalised.

A trimmed render's *noise* is a different realisation from an untrimmed one's: the noise streams
also run through the leading silence. Both have the same spectrum and level; they are just not the
same samples.

### Cost

8 s of audio through 48 bands takes 0.22 s, of which 0.06 s is the one-off bank calibration: about
50× realtime for the DSP itself. Bands are summed serially in a fixed order, so the same seed gives
bit-identical samples and the block size cannot reach the output.

---

## 11. Tuning recipes

Measured on 3 s of solo piano at 48 kHz, all driven to the same 35 dB so atoms and wall clock are
comparable. The dictionary reaches `alpha = 1`.

| | atoms | wall | residual peak |
| --- | --- | --- | --- |
| **balanced** — `max_atom_samples 150000`, `alpha_bracket 2.5`, `rounds 2`, `capture_tolerance 0.5` | 1528 | 12.6 s | −30.3 dB |
| **fast** — as above but `alphas = [16, 64, 256]` | 1735 | 11.2 s | −30.8 dB |
| **sparsest** — `max_atom_samples 400000`, `alpha_bracket 2.5`, `alphas [4,16,64,256]` | 1421 | 22.0 s | −30.7 dB |
| *the trap* — same dictionary, `max_atom_samples 65536` | 1631 | 9.9 s | **−17.3 dB** |

**Start here.** Take `data/config/mp_1.toml` as the balanced row. Then:

- **Too slow?** Raise `capture_tolerance` toward 0.5 first (biggest lever, mild quality cost), then
  drop your lowest `alphas` rung (large lever, real quality cost on sustained material), then raise
  `fade_level`. Do not reach for `golden_iters`.
- **Not enough detail?** Raise `target_snr_db` before touching anything else. Then check the
  `refined:` line is near 100% — if it is not, `max_atom_samples` is the problem, not the dictionary.
- **Bad transients?** Watch the residual *peak*, not the rms. A peak much worse than the rms means
  atoms are landing badly: check `refined:`, then lower `capture_tolerance`, then `rounds`.
- **Pre-echo, or energy in silences?** Turn on `[hrmp]` at `depth = 1`, `phase_tolerance_deg = 90`.
  Expect ~6% more atoms. If you get far more, something is over-rejecting.

**Without refinement**, none of the above applies: set `capture_tolerance = 0.95`, use a dense
`alphas` ladder (ratio ~1.6), and expect roughly 12× the atoms.

---

## 12. Diagnostics

```bash
rmpstat summary book.json -c settings.toml   # atoms, energy, parameter spread
rmpstat diag    book.json -c settings.toml   # per-block seeds and energy share,
                                             #   refinement drift, conditioning
rmpstat hist    book.json --of alpha,bandwidth,f --weight energy
rmpstat hist    book.json --of sigma,bandwidth   # gaussian atoms; alpha/beta skip them
rmpstat snr     book.json -f svg -o snr.svg  # convergence curve
rmpstat wv      book.json -f png -o wv.png --log-freq --floor 65
```

`alpha`, `beta`, `alpha-beta`, `fade-dur` and `rho` describe FOF atoms only and `sigma` Gaussian atoms
only; on a mixed book a histogram of one of them counts the other kind separately, outside its total.
`bandwidth`, `q`, `support` and the placement and energy quantities apply to both.

`diag` answers the two questions that decide most settings: **which blocks are actually earning
their keep** (its energy-share column — if a rung takes 1% of energy for 5% of the seeds, drop it),
and **how far refinement had to move** (its drift table — drift consistently at the bracket edge
means the bracket is too narrow or the grid too sparse).

The `wv` map is an atom-based pseudo-Wigner display, built as a sum of per-atom distributions so it
has no cross-terms. It is diagnostics only and plays no part in the pursuit.

---

## 13. Things that will bite

**A WAV written twice is not byte-identical.** libsndfile stamps a timestamp into the PEAK chunk of
a float file. Exactly one byte differs and the audio is untouched — compare the book, or the data
past byte 72.

**A rendered book is longer than the excerpt it came from.** Analysis sizes the residual by the
input; `rmpsynth` sizes the output by the furthest atom death, so tails the analysis truncated at the
excerpt end become audible. Over the excerpt itself the two are identical.

**Unknown settings are rejected, not ignored.** A typo fails loudly, which is what you want in a
hand-edited document.

**`[dictionary] alphas` is refused.** The dictionary holds one section per atom kind now; put a
`[dictionary.fof]` header above `alphas`, `betas_ms` and `alpha_beta_max`. The shipped documents in
`data/config` are already migrated.

**`rmp` no longer renders.** `rmp in.wav -o out.wav` and `rmp -b book.json -o out.wav` both fail,
naming the `rmpsynth -b ... -o ...` command that replaces them. `--fof-audio` is gone from `rmpsynth`
because it renders the atoms itself.

**A mixed book does not fully replay through rfofs.** Its FOF atoms convert to rfofs parameters
unchanged; its Gaussian atoms have no rfofs representation. `rmpsynth` renders both.
