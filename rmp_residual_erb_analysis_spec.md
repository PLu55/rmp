# RMP Residual ERB Analysis — Implementation Specification

## 1. Purpose

Extend **rmp** (Rust Matching Pursuit) with an optional **residual stochastic analysis stage** that runs after the FOF matching-pursuit decomposition has completed.

The deterministic decomposition remains:

\[
x[n] = \sum_k \mathrm{FOF}_k[n] + r[n]
\]

where \(r[n]\) is the final residual.

The new analysis stage converts the residual into a compact, time-varying ERB-band power representation suitable for later real-time stochastic resynthesis:

\[
r[n]
\rightarrow
\text{ERB analysis bank}
\rightarrow
\text{per-band power estimation}
\rightarrow
\text{fixed-rate residual book}
\]

The initial implementation covers **analysis only**. It must define data in a way that a later real-time power-complementary ERB synthesis bank can consume directly.

---

## 2. Scope

### Included

- Optional residual analysis after FOF Matching Pursuit.
- ERB-spaced analysis filter bank.
- Configurable number of bands.
- Configurable lower and upper analysis frequencies.
- Per-band power estimation.
- Configurable residual-book update interval.
- Band-dependent or fixed power-detector time constants.
- Storage of analyzed power frames in the output book.
- CLI options.
- Settings-file options.
- Validation.
- Diagnostics and summary output.
- Unit tests and integration tests.
- Deterministic/reproducible analysis.

### Not included in this stage

- Real-time residual synthesis.
- Noise generation.
- ERB synthesis bank.
- Adaptive update-rate encoding.
- Entropy compression of residual-book data.
- LPC/AR modeling.
- Wavelet analysis.
- Joint optimization of FOF atoms and stochastic residual.
- HRMP.

The residual analyzer runs only on the **final ordinary FOF-MP residue**.

---

# 3. High-level processing flow

```text
input audio
    |
    v
FOF Matching Pursuit
    |
    +----> FOF atom book
    |
    v
final residue r[n]
    |
    +---- residual analysis disabled ---> done
    |
    v
ERB analysis bank
    |
    v
band signals r_b[n]
    |
    v
power detectors P_b[n]
    |
    v
sample every update_interval
    |
    v
ResidualBook
```

The residual analysis must not alter the MP decomposition.

---

# 4. Terminology

## Residue / residual

`residue` is the waveform left after subtracting all selected FOF atoms from the source signal.

Use one term consistently in Rust identifiers. Recommended:

- user-facing CLI/config: `residual`
- internal Rust API: `Residual*`

## Residual book

A control-rate representation containing one power value per ERB band and update frame.

A frame is:

\[
\mathbf P[k]
=
[P_0[k], P_1[k], \ldots, P_{B-1}[k]]
\]

where each \(P_b[k]\) is a linear power value.

---

# 5. Design principles

1. **Preserve temporal structure.**  
   Residual signals may be stochastic at waveform level while retaining important rhythmic and transient structure. Temporal smoothing must therefore be minimal and explicit.

2. **Frequency smoothing comes mainly from the ERB bank.**  
   Do not add broad spectral smoothing after the bank unless explicitly configured in a future version.

3. **Analysis rate and book update rate are separate.**  
   The filter bank and power detector operate at audio rate. The book records their state at a configurable control rate.

4. **Store power, not instantaneous amplitude.**

5. **Use deterministic algorithms.**  
   Analysis must not depend on thread scheduling or random state.

6. **Do not couple residual analysis to the MP search loop.**  
   The first implementation is a post-processing stage.

7. **Prepare for later real-time synthesis.**  
   The band definition and normalization must be serialized sufficiently to reconstruct the matching synthesis bank.

---

# 6. Mathematical model

For final residual \(r[n]\), each ERB analysis filter produces:

\[
y_b[n] = H_b(z) r[n]
\]

for band:

\[
b = 0,\ldots,B-1.
\]

Instantaneous band power:

\[
q_b[n] = y_b^2[n].
\]

A one-pole power estimator produces:

\[
P_b[n]
=
a_b P_b[n-1]
+
(1-a_b)q_b[n].
\]

with:

\[
a_b = e^{-1/(\tau_b f_s)}.
\]

At residual-book update sample \(n_k\):

\[
n_k = k N_u
\]

store:

\[
P_b[k] = P_b[n_k].
\]

The update period is:

\[
N_u =
\operatorname{round}
\left(
f_s \Delta t
\right)
\]

where \(\Delta t\) is configurable.

---

# 7. ERB frequency scale

Use a standard ERB-rate mapping.

Recommended Glasberg/Moore-style equations:

\[
ERB(f)
=
24.7
\left(
1 + 4.37 f / 1000
\right)
\]

and ERB-rate:

\[
E(f)
=
21.4 \log_{10}
\left(
1 + 0.00437f
\right).
\]

Band centers are uniformly spaced on ERB-rate:

\[
E_b
=
E(f_{\min})
+
\frac{b}{B-1}
\left[
E(f_{\max})-E(f_{\min})
\right].
\]

Convert each \(E_b\) back to frequency.

The inverse is:

\[
f(E)
=
\frac{10^{E/21.4}-1}{0.00437}.
\]

---

# 8. Filter-bank requirements

## 8.1 Initial implementation

Use a fixed bank of low-order IIR filters.

Recommended first implementation:

- fourth-order gammatone-like or equivalent ERB band-pass response;
- fixed coefficients during analysis;
- one filter instance per band;
- `f64` coefficient generation;
- processing may use `f32` or `f64` according to existing `rmp` numeric conventions.

The filter implementation must be isolated behind an interface so another filter-bank design can replace it later.

Example:

```rust
pub trait AnalysisBand {
    fn reset(&mut self);
    fn process_sample(&mut self, x: f32) -> f32;
}
```

Or preferably a block API if `rmp` already uses block processing:

```rust
pub trait AnalysisBand {
    fn reset(&mut self);
    fn process_block(&mut self, input: &[f32], output: &mut [f32]);
}
```

## 8.2 Shared bank definition

Introduce a serializable bank definition:

```rust
pub struct ErbBankSpec {
    pub bands: usize,
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    pub spacing: ErbSpacing,
    pub filter_kind: ErbFilterKind,
    pub filter_order: u32,
    pub normalization: ErbNormalization,
}
```

Initial enum values may be constrained to one implementation:

```rust
pub enum ErbSpacing {
    ErbRate,
}

pub enum ErbFilterKind {
    Gammatone,
}

pub enum ErbNormalization {
    UnitNoisePower,
}
```

Keep enums even if only one value is initially implemented.

---

# 9. Filter normalization

The analysis representation is intended for a later stochastic synthesis bank.

Each analysis filter should therefore have a defined and serialized normalization.

Recommended target:

\[
C_b
=
\frac{1}{2\pi}
\int_{-\pi}^{\pi}
|H_b(e^{j\omega})|^2 d\omega
=
1.
\]

In other words, unit-variance white noise should produce approximately unit output variance in each normalized band.

Call this:

```text
unit-noise-power
```

This makes measured band powers easier to interpret.

If exact normalization is difficult analytically, compute a deterministic normalization factor from the designed coefficients.

The normalization process must not use random Monte Carlo estimation.

---

# 10. Power detector

Introduce:

```rust
pub enum ResidualPowerTimeMode {
    Fixed,
    BandwidthRelative,
}
```

## 10.1 Fixed mode

All bands use:

\[
\tau_b = \tau.
\]

Config:

```text
residual.power_tau_ms
```

## 10.2 Bandwidth-relative mode

Use:

\[
\tau_b
=
\operatorname{clamp}
\left(
K / B_b,
\tau_{\min},
\tau_{\max}
\right)
\]

where:

- \(B_b\) = effective bandwidth in Hz;
- \(K\) = dimensionless configurable scale;
- \(\tau\) is in seconds.

Configuration:

```text
residual.power_tau_mode = "bandwidth-relative"
residual.power_tau_scale = ...
residual.power_tau_min_ms = ...
residual.power_tau_max_ms = ...
```

A reasonable initial default may be chosen conservatively, but all defaults must be documented in the settings schema.

Do not silently apply additional temporal smoothing.

---

# 11. Update interval

Residual-book update interval is independently configurable.

Support at least:

```text
--residual-update-ms <FLOAT>
```

Settings equivalent:

```toml
[residual]
update_ms = 1.0
```

Convert to exact integer samples once:

```rust
let update_samples =
    ((update_ms * 0.001) * sample_rate as f64).round() as u64;
```

Require:

```text
update_samples >= 1
```

Store `update_samples` in the book.

Do not repeatedly convert milliseconds to samples during processing.

---

# 12. Proposed settings-file schema

Assuming TOML. Adapt field naming to the existing `rmp` settings conventions.

```toml
[residual]
enabled = true

# Residual-book control update interval.
update_ms = 1.0

# Stored representation.
storage = "f32-linear-power"

[residual.erb]
bands = 48
min_freq_hz = 50.0
max_freq_hz = 20000.0

spacing = "erb-rate"
filter = "gammatone"
order = 4
normalization = "unit-noise-power"

[residual.power]
mode = "bandwidth-relative"

# Used only for mode = "fixed".
tau_ms = 2.0

# Used for mode = "bandwidth-relative".
tau_scale = 1.0
tau_min_ms = 0.5
tau_max_ms = 10.0
```

The exact nesting should follow the existing project style if different.

---

# 13. CLI additions

Add a residual-analysis option group.

Suggested CLI:

```text
--residual-analysis
--no-residual-analysis

--residual-update-ms <MS>

--residual-erb-bands <N>
--residual-erb-min-hz <HZ>
--residual-erb-max-hz <HZ>

--residual-power-mode <fixed|bandwidth-relative>

--residual-power-tau-ms <MS>
--residual-power-tau-scale <FLOAT>
--residual-power-tau-min-ms <MS>
--residual-power-tau-max-ms <MS>
```

Optional future-proofing:

```text
--residual-erb-filter <gammatone>
--residual-erb-order <N>
--residual-erb-normalization <unit-noise-power>
```

If these values are not intended to vary initially, keep them in the settings file and omit them from the CLI until needed.

---

# 14. CLI precedence

Use normal configuration precedence:

```text
built-in defaults
    <
settings file
    <
CLI arguments
```

CLI arguments override only fields explicitly supplied.

Do not construct a separate configuration path for residual analysis.

After merging settings, validate once and pass a fully resolved immutable config to analysis.

---

# 15. Rust configuration model

Recommended resolved config:

```rust
#[derive(Debug, Clone)]
pub struct ResidualAnalysisConfig {
    pub enabled: bool,
    pub update_samples: usize,
    pub erb: ErbBankConfig,
    pub power: ResidualPowerConfig,
    pub storage: ResidualStorage,
}

#[derive(Debug, Clone)]
pub struct ErbBankConfig {
    pub bands: usize,
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    pub filter_kind: ErbFilterKind,
    pub filter_order: usize,
    pub normalization: ErbNormalization,
}

#[derive(Debug, Clone)]
pub struct ResidualPowerConfig {
    pub mode: ResidualPowerTimeMode,
    pub fixed_tau_seconds: f64,
    pub bandwidth_tau_scale: f64,
    pub tau_min_seconds: f64,
    pub tau_max_seconds: f64,
}
```

Keep raw user settings separate from resolved runtime settings if that pattern already exists in `rmp`.

---

# 16. Validation

Reject invalid configurations before starting MP.

## Required checks

### Update rate

```text
update_ms > 0
update_samples >= 1
```

### Band count

```text
bands >= 1
```

Recommended practical validation:

```text
bands >= 4
```

unless tests or special modes need fewer.

### Frequencies

```text
min_freq_hz > 0
max_freq_hz > min_freq_hz
max_freq_hz < Nyquist
```

Prefer a small guard below Nyquist if the selected filter implementation requires it.

### Power detector

All time constants:

```text
> 0
```

For bandwidth-relative:

```text
tau_min <= tau_max
tau_scale > 0
```

### Filter order

Must be supported by selected filter implementation.

Errors should identify the field and invalid value.

---

# 17. Analysis API

Recommended top-level API:

```rust
pub fn analyze_residual(
    residual: &[f32],
    sample_rate: u32,
    config: &ResidualAnalysisConfig,
) -> Result<ResidualBook, ResidualAnalysisError>;
```

If audio can be multichannel, do not hide policy inside this function.

Prefer either:

```rust
pub fn analyze_residual_channel(...)
```

or:

```rust
pub fn analyze_residual(
    residual: &AudioBuffer,
    ...
)
```

with explicit per-channel behavior.

The initial implementation should define whether residual analysis is:

- per channel;
- downmixed;
- disabled for multichannel.

Do not silently downmix.

---

# 18. Residual-book data model

Recommended structure:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualBook {
    pub version: u32,

    pub sample_rate: u32,
    pub start_sample: u64,
    pub source_samples: u64,

    pub update_samples: u32,

    pub bank: ResidualErbBankDescriptor,

    pub frames: Vec<ResidualFrame>,
}
```

Frame:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualFrame {
    pub power: Vec<f32>,
}
```

Do not store per-frame timestamps for fixed-rate books.

Frame `k` occurs at:

\[
n_k =
start\_sample + k \cdot update\_samples.
\]

For performance and compactness, the final internal representation may instead be flat:

```rust
pub struct ResidualBook {
    ...
    pub frame_count: u64,
    pub power: Vec<f32>,
}
```

with indexing:

```rust
power[frame * band_count + band]
```

This is preferred for serialization and cache locality.

---

# 19. Bank descriptor stored in the book

The book must contain enough metadata for the later synthesis engine to know what the power values mean.

Example:

```rust
pub struct ResidualErbBankDescriptor {
    pub band_count: u32,
    pub min_freq_hz: f32,
    pub max_freq_hz: f32,

    pub center_freq_hz: Vec<f32>,
    pub bandwidth_hz: Vec<f32>,

    pub spacing: ErbSpacing,
    pub filter_kind: ErbFilterKind,
    pub filter_order: u32,
    pub normalization: ErbNormalization,

    pub power_detector: ResidualPowerDescriptor,
}
```

Storing center frequencies explicitly is recommended even if they can be recomputed.

This protects the book against small future changes in ERB formulas or filter design.

---

# 20. Output-book integration

If `rmp` already has one top-level analysis book, extend it rather than creating unrelated output files.

Conceptually:

```rust
pub struct Book {
    pub atoms: Vec<FofAtom>,
    pub residual: Option<ResidualBook>,
    ...
}
```

Residual analysis disabled:

```rust
residual = None
```

Residual analysis enabled:

```rust
residual = Some(...)
```

If the existing format supports typed sections/chunks, use a separate residual section.

Do not interleave residual frames with FOF atom events.

They have different semantics:

- FOF atoms are sparse deterministic events.
- residual frames are regularly sampled stochastic-control data.

---

# 21. Initial analysis algorithm

Straightforward reference implementation:

```rust
fn analyze_residual(
    residual: &[f32],
    sample_rate: u32,
    cfg: &ResidualAnalysisConfig,
) -> Result<ResidualBook, Error> {
    let mut bank = ErbAnalysisBank::new(sample_rate, &cfg.erb)?;

    let mut power = vec![0.0_f32; cfg.erb.bands];

    let coeffs =
        compute_power_detector_coefficients(sample_rate, &bank, &cfg.power);

    let frame_count =
        residual.len().div_ceil(cfg.update_samples);

    let mut frames =
        Vec::with_capacity(frame_count * cfg.erb.bands);

    for (n, &x) in residual.iter().enumerate() {
        for b in 0..bank.len() {
            let y = bank.process_band_sample(b, x);

            let q = y * y;

            power[b] =
                coeffs[b] * power[b]
                + (1.0 - coeffs[b]) * q;
        }

        if n % cfg.update_samples == 0 {
            frames.extend_from_slice(&power);
        }
    }

    Ok(...)
}
```

This is a reference design, not necessarily the fastest offline implementation.

---

# 22. Preferred offline implementation layout

Because analysis is offline, use the layout that performs best and is easiest to parallelize.

A likely faster approach is band-major:

```text
for each ERB band:
    filter entire residual
    compute power trajectory
    sample power at book update positions
    write band column into output matrix
```

Advantages:

- contiguous filter processing;
- easier SIMD over samples;
- easy parallelism over bands;
- less per-sample branch/control overhead.

Conceptually:

```rust
frames = vec![0.0; frame_count * bands];

parallel_for_each_band(|b| {
    filter residual -> band_signal

    power = 0

    for n:
        update power

        if update sample:
            frames[frame_index * bands + b] = power
});
```

If using parallel execution, each band writes only to its own column to avoid synchronization.

The resulting output must be bitwise or numerically deterministic according to the project's existing determinism requirements.

---

# 23. Start and end behavior

Power filters need defined startup behavior.

Initial implementation:

```text
P_b[-1] = 0
```

This causes a startup rise determined by \(\tau_b\).

For a residual starting at sample zero, this is physically reasonable.

Do not pre-fill the detector with the first sample unless explicitly chosen and documented.

At the signal end:

- record the last regularly scheduled frame that falls inside the residual;
- do not append arbitrary decay frames after the residual in the first implementation.

Later synthesis can stop stochastic excitation at the source duration.

---

# 24. Frame convention

Define exactly what a frame means.

Recommended:

> A residual frame stores the state of each causal band-power detector immediately after processing the sample at the frame's sample offset.

Thus for:

```text
update_samples = 48
```

frames correspond to samples:

```text
0, 48, 96, 144, ...
```

This convention must be used consistently by later synthesis.

---

# 25. Precision

## Analysis

Prefer:

- filter coefficient design: `f64`;
- ERB center-frequency calculation: `f64`;
- normalization calculations: `f64`.

Filter processing may use:

```text
f32
```

if the MP residual itself is `f32`.

Power accumulation may benefit from `f64` during development/testing, but `f32` is likely adequate for the stored residual book.

Benchmark both only if needed.

## Book

Initial representation:

```text
f32 linear power
```

Do not quantize in the first implementation.

Later storage formats may include:

```text
u16 log power
u8 log power
delta coded log power
```

without changing the conceptual model.

---

# 26. Serialization

Book serialization must contain a format version.

Example:

```rust
pub const RESIDUAL_BOOK_VERSION: u32 = 1;
```

The serialized format must be able to reject incompatible future variants.

If the main RMP book already has a version, increment or extend it according to existing compatibility policy.

---

# 27. CLI summary output

When residual analysis is enabled, print or log a concise summary such as:

```text
Residual analysis:
  ERB bands:          48
  frequency range:    50.0 .. 20000.0 Hz
  update interval:    48 samples / 1.000 ms
  power mode:         bandwidth-relative
  tau range:          0.5 .. 10.0 ms
  residual frames:    183421
```

Optional diagnostics:

```text
Residual energy before stochastic analysis: ...
Stored residual control values: ...
```

Avoid dumping per-band values unless verbose/debug mode is enabled.

---

# 28. Diagnostics

Useful debug outputs:

## Bank table

```text
band
center_hz
bandwidth_hz
power_tau_ms
normalization_gain
```

Expose via a verbose flag or debug log.

## Residual-energy check

Calculate:

\[
E_r = \sum_n r^2[n].
\]

Optionally compare with a rough reconstructed ERB energy measure.

Do not expect exact equality because overlapping filters and temporal power detection mean the stored bands are not a strict orthogonal energy partition.

---

# 29. Testing

## 29.1 ERB center frequencies

Verify:

- strictly increasing;
- first approximately equals configured minimum;
- last approximately equals configured maximum;
- uniform spacing on ERB-rate.

## 29.2 Filter stability

Every configured band must remain stable for valid sample rates and frequency ranges.

Feed:

```text
impulse
silence
constant
white-noise fixture
```

and verify finite output.

## 29.3 Unit-noise-power normalization

For each filter, use a deterministic spectral/integral test.

Do not rely on random input in unit tests.

Verify normalization within a specified tolerance.

## 29.4 Power detector

For constant squared input \(q\), confirm convergence toward \(q\).

For an impulse/burst, verify expected exponential decay.

## 29.5 Update timing

For:

```text
sample_rate = 48000
update_ms = 1
```

verify:

```text
update_samples == 48
```

and frames occur exactly at:

```text
0, 48, 96, ...
```

## 29.6 Silence

For all-zero residual:

```text
all book powers == 0
```

within exact arithmetic expected from the implementation.

## 29.7 Single tone

Feed a sinusoidal residue at known frequency.

Verify:

- strongest power appears in nearest ERB band(s);
- adjacent overlap behaves as expected;
- distant bands remain substantially lower.

## 29.8 Broadband noise fixture

Use a deterministic pseudo-noise fixture checked into tests or generated from a fixed algorithm.

Verify approximately flat normalized band powers after startup, accounting for statistical tolerance.

## 29.9 Transient preservation

Feed a short broadband pulse/noise burst.

Verify the residual book shows a correspondingly short power event.

This test is important because temporal over-smoothing is specifically undesirable.

## 29.10 Serialization round trip

Serialize and deserialize:

```text
bank metadata
update interval
frame count
power matrix
```

and verify equality.

## 29.11 Disabled mode

When residual analysis is disabled:

- no ERB processing occurs;
- output book has no residual section;
- ordinary MP result is unchanged.

---

# 30. Performance

The analysis is offline, so correctness and model quality take priority over latency.

Still:

- allocate output buffers once;
- avoid allocation in inner filtering loops;
- precompute coefficients;
- precompute power-detector coefficients;
- process bands independently;
- allow parallel processing across ERB bands where useful;
- avoid recomputing ERB formulas per sample.

Residual analysis should occur after MP and should not increase the cost of atom search.

---

# 31. Error handling

Add a dedicated error type if appropriate:

```rust
pub enum ResidualAnalysisError {
    InvalidBandCount,
    InvalidFrequencyRange,
    FrequencyAboveNyquist,
    InvalidUpdateInterval,
    InvalidPowerTimeConstant,
    UnsupportedFilterOrder,
    FilterDesignFailed,
    ...
}
```

Integrate with the project's existing error stack rather than introducing a separate error-handling style.

---

# 32. Recommended initial defaults

These are starting values, not assumptions about the final model:

```toml
[residual]
enabled = false
update_ms = 1.0
storage = "f32-linear-power"

[residual.erb]
bands = 48
min_freq_hz = 50.0
max_freq_hz = 20000.0
spacing = "erb-rate"
filter = "gammatone"
order = 4
normalization = "unit-noise-power"

[residual.power]
mode = "bandwidth-relative"
tau_ms = 2.0
tau_scale = 1.0
tau_min_ms = 0.5
tau_max_ms = 10.0
```

Clamp `max_freq_hz` only if configuration explicitly supports an `auto` value.

Prefer validation errors over silent modification of explicit user values.

---

# 33. Future synthesis contract

Although synthesis is outside this implementation, analysis must preserve the following contract.

The future stochastic synthesizer will approximately compute:

\[
\hat r[n]
=
\sum_b g_b[n] H_b(z) w_b[n]
\]

with independent stochastic excitation \(w_b[n]\).

The synthesis filters will be designed as an ERB-spaced **power-complementary bank**:

\[
\sum_b |H_b(e^{j\omega})|^2 \approx 1.
\]

Book powers will become synthesis targets approximately through:

\[
g_b =
\sqrt{P_b}.
\]

Therefore analysis metadata must make the normalization and exact bank specification explicit.

A later implementation may introduce a fixed overlap-compensation matrix:

\[
\mathbf p \approx M \mathbf g^2
\]

to improve the mapping between analyzed band power and synthesis gains.

Do not implement this inverse correction in version 1.

---

# 34. Suggested module layout

Adapt to the existing crate structure.

```text
src/
    residual/
        mod.rs
        config.rs
        erb.rs
        filter.rs
        power.rs
        analyze.rs
        book.rs
        error.rs

    cli/
        ...

    settings/
        ...
```

Suggested responsibilities:

### `residual/config.rs`

- raw settings;
- resolved settings;
- validation.

### `residual/erb.rs`

- ERB conversion functions;
- center-frequency generation;
- bandwidth calculation;
- bank descriptor.

### `residual/filter.rs`

- IIR/gammatone filter design;
- normalization;
- filter processing.

### `residual/power.rs`

- one-pole power detector;
- bandwidth-relative time-constant calculation.

### `residual/analyze.rs`

- orchestration;
- residual-to-book conversion;
- parallel band processing.

### `residual/book.rs`

- serialized data structures.

---

# 35. Implementation sequence

## Phase 1 — configuration

1. Add residual settings structures.
2. Add CLI arguments.
3. Merge CLI/settings/defaults.
4. Validate resolved configuration.
5. Add configuration tests.

## Phase 2 — ERB bank

1. Implement ERB/ERB-rate conversions.
2. Generate band centers.
3. Calculate effective bandwidths.
4. Implement selected filter.
5. Implement deterministic unit-noise-power normalization.
6. Add bank/filter tests.

## Phase 3 — power analysis

1. Implement one-pole power detector.
2. Implement fixed-\(\tau\) mode.
3. Implement bandwidth-relative-\(\tau\) mode.
4. Add detector tests.

## Phase 4 — residual book

1. Add `ResidualBook`.
2. Add bank descriptor.
3. Add flat frame storage.
4. Add serialization.
5. Add round-trip tests.

## Phase 5 — integration

1. Obtain final MP residual.
2. Run residual analyzer if enabled.
3. Attach `ResidualBook` to main output book.
4. Add CLI summary.
5. Add end-to-end tests.

---

# 36. Acceptance criteria

Version 1 is complete when:

1. `rmp` can run exactly as before with residual analysis disabled.
2. Residual analysis can be enabled from both settings and CLI.
3. A configurable ERB bank analyzes the final MP residue.
4. Band-power control frames are written at an exact configurable sample interval.
5. Temporal power estimation has fixed and bandwidth-relative modes.
6. The output book stores all metadata needed to interpret the residual frames.
7. Silence, sinusoid, transient, and broadband deterministic fixtures pass.
8. Residual analysis introduces no changes to selected FOF atoms.
9. No per-sample heap allocation occurs in filter or power-detector loops.
10. Serialized residual books are versioned and round-trip correctly.

---

# 37. Example command

Illustrative only; adapt executable name and existing CLI conventions:

```bash
rmp analyze input.wav \
    --residual-analysis \
    --residual-update-ms 1.0 \
    --residual-erb-bands 48 \
    --residual-erb-min-hz 50 \
    --residual-erb-max-hz 20000 \
    --residual-power-mode bandwidth-relative \
    --residual-power-tau-min-ms 0.5 \
    --residual-power-tau-max-ms 10
```

---

# 38. Initial implementation decision summary

The initial residual analyzer is:

\[
\boxed{
\text{final MP residue}
\rightarrow
\text{48-ish ERB-spaced normalized IIR bands}
\rightarrow
\text{one-pole band-power estimation}
\rightarrow
\text{configurable fixed-rate power frames}
\rightarrow
\text{ResidualBook}
}
\]

The defining design choice is that the residual is treated as **stochastic but potentially strongly nonstationary**. The analyzer therefore avoids broad temporal smoothing and preserves fast rhythmic/transient variation in the per-band power trajectories.
