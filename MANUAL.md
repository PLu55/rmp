# rmp — user manual

`rmp` decomposes a soundfile into FOF atoms (Rodet's Formant Wave Function) by Matching Pursuit,
and writes them as a *book* that replays through the `rfofs` synthesizer unchanged.

This manual covers the command line, every setting, and what each one costs. For the design and its
internals see `CLAUDE.md`; for the algorithm see `notes.md`.

---

## 1. The mental model

Three stages, and every setting belongs to exactly one of them.

**The dictionary** is a set of *blocks*. A block fixes one envelope shape — a decay rate `alpha` and
an attack duration `beta` — and offers that shape at every onset on a time grid and every frequency
on an FFT grid. `[dictionary]`, `[envelope]` and `[blocks]` build it.

**The pursuit** repeatedly finds the single best-matching atom in the dictionary, subtracts it from
the residual, and repeats. `[pursuit]` decides when to stop.

**Refinement** takes each selected atom off the grid before subtracting it, adjusting
`(t0, f, alpha, beta)` by local search. `[refine]` controls it. `[hrmp]` optionally vetoes atoms the
residual does not support across their whole length.

Two relations are worth carrying in your head, because most settings act through them:

```text
-3 dB bandwidth  ≈  alpha / pi          Hz          alpha = 100  ->   32 Hz wide
atom length      ≈  6.9 * sr / alpha    samples     alpha = 100  ->  3300 samples at 48 kHz
attack duration  =  beta                seconds
```

So **`alpha` is the one parameter that sets both an atom's frequency width and its length**, in
opposite directions. Low `alpha` means a long, narrow, sustained atom; high `alpha` means a short,
wide, percussive one. Almost every cost in the program scales with the longest `alpha` in the
dictionary.

---

## 2. Running it

```bash
# analyse: needs at least one output
rmp in.wav -o resynth.wav                      # resynthesis
rmp in.wav -b book.json                        # atoms only, no render
rmp in.wav -o resynth.wav -r residual.wav -b book.json.gz
rmp in.wav -o out.wav -c settings.toml         # with settings
rmp in.wav -o out.wav -s 2.5 -d 0.5            # one excerpt, in seconds

# synthesise: no input soundfile, so --book is read rather than written
rmp -b book.json -o resynth.wav

rmp --write-config > settings.toml             # a fully commented default document
```

| flag | meaning |
| --- | --- |
| `-o`, `--out` | resynthesis, 32-bit float WAV. Optional when analysing |
| `-c`, `--config` | settings document. Defaults are used if omitted |
| `-r`, `--residual` | what the decomposition could not explain |
| `-b`, `--book` | the atoms. **Output when analysing, input when not** |
| `-s`, `--start` | offset into the file, seconds |
| `-d`, `--duration` | length to analyse, seconds |
| `-q`, `--quiet` | suppress the report |

Book format follows the extension: `.toml` or `.json`, either optionally `.gz`. Gzip is worth about
7× and costs nothing to read back. `book.json.gz` is JSON; a bare `book.gz` is TOML.

Multi-channel input is downmixed to mono by averaging, which partially cancels out-of-phase content
between channels. The run says so when it happens.

Everything is relative to the excerpt, not the file: with `--start 2.5`, atom onsets in the book
count from that point, and synthesising the book gives you the excerpt.

### Reading the report

```text
in.wav: 29.07 s, 48000 Hz, 2 channel(s)
  analysing 5.000-8.000 s (144000 samples from 240000)
  downmixed to mono; out-of-phase content between channels partially cancels
dictionary: 24 blocks in 13.68ms
  note: 8 of 24 blocks are longer than refine.max_atom_samples (150000), so their
        atoms stay on the grid unrefined; longest support 332053 samples (alpha 1.000)
analysis: 1528 atoms, 35.0 dB in 12.64s (init 13.68ms, 4.2x realtime)
  refined: 1291/1528 atoms moved off the grid (84%)
  refresh: 1266438 frames bounded, 416712 recomputed (32.9%)
residual: -35.0 dB rms, -30.3 dB peak relative to input
  absolute: -73.1 dBFS rms, -57.2 dBFS peak (input -38.1 dBFS rms, -26.9 dBFS peak)
```

- **`4.2x realtime`** is wall clock over audio duration. Below 1.0 is faster than realtime.
- **`refined: N/M`** — how many atoms moved off the grid. Well under 100% with refinement enabled
  means `max_atom_samples` is blocking blocks; see §6.
- **`refresh: bounded/recomputed`** — the lazy update's hit rate. Lower recomputed % is faster; it
  is diagnostic, not a setting.
- **`-35.0 dB rms`** is the negated SNR, so it restates the line above.
- **`-30.3 dB peak`** is the one that adds information. It is where the decomposition is *worst*
  rather than where it is on average, and it is the most useful single quality number in the report:
  a badly-placed atom shows up here and nowhere else.

---

## 3. `[dictionary]` — which envelope shapes exist

The seed grid. With refinement on, these are starting points rather than the final parameters, so
the grid needs to *bracket* the material rather than resolve it.

### `alphas` — default `[80, 128, 205, 328, 524, 839, 1342, 2147]`

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

### `betas_ms` — default `[0.3, 1.0, 3.0]`

Attack durations in milliseconds — the half-cosine rise at the atom's onset.

**Result.** Shapes only the first few milliseconds. It matters for transient character, not for the
body of the atom. Refinement moves it freely, so two or three well-spread rungs suffice.

**Cost.** Multiplies the block count directly. Fewer rungs is not reliably faster, though: measured,
cutting `[0.3, 1, 3, 12]` to `[1, 12]` made a piano run *slower* (16.3 s against 9.9 s), because the
seeds fit worse and the pursuit needed more of the expensive long blocks.

### `alpha_beta_max` — default `4.0`

Drops `(alpha, beta)` combinations whose product exceeds this.

**Result.** A guard, not a tuning knob. rfofs renders `alpha*beta > 10` as silence outright, and its
amplitude normalisation is ill-conditioned well before that. Raising it above ~4 admits blocks whose
peak normalisation is unreliable. Leave it.

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

Work per candidate. A round is one `f → alpha → beta → t0` cycle; `golden_iters` is the
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
and `[dictionary]` so the coarse and refined paths cannot disagree about what is representable.

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

## 9. Tuning recipes

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

## 10. Diagnostics

```bash
rmpstat summary book.json -c settings.toml   # atoms, energy, parameter spread
rmpstat diag    book.json -c settings.toml   # per-block seeds and energy share,
                                             #   refinement drift, conditioning
rmpstat hist    book.json --of alpha,bandwidth,f --weight energy
rmpstat snr     book.json -f svg -o snr.svg  # convergence curve
rmpstat wv      book.json -f png -o wv.png --log-freq --floor 65
```

`diag` answers the two questions that decide most settings: **which blocks are actually earning
their keep** (its energy-share column — if a rung takes 1% of energy for 5% of the seeds, drop it),
and **how far refinement had to move** (its drift table — drift consistently at the bracket edge
means the bracket is too narrow or the grid too sparse).

The `wv` map is an atom-based pseudo-Wigner display, built as a sum of per-atom distributions so it
has no cross-terms. It is diagnostics only and plays no part in the pursuit.

---

## 11. Things that will bite

**A WAV written twice is not byte-identical.** libsndfile stamps a timestamp into the PEAK chunk of
a float file. Exactly one byte differs and the audio is untouched — compare the book, or the data
past byte 72.

**A synthesised book is longer than the excerpt it came from.** Analysis sizes the residual by the
input; synthesis sizes the output by the furthest atom death, so tails the analysis truncated at the
excerpt end become audible. Over the excerpt itself the two renders are identical.

**Unknown settings are rejected, not ignored.** A typo fails loudly, which is what you want in a
hand-edited document.

**`--book` reverses direction with no other signal.** With an input soundfile it is written; without
one it is read and synthesised, and `--config`, `--start`, `--duration` and `--residual` become
errors rather than being silently ignored.
