# Residual ERB Synthesis — Standalone Executable and Rust Library Specification

## 1. Purpose

Implement a standalone Rust synthesis component that reconstructs the stochastic residual from an RMP residual book and writes a sound file.

The same implementation must also be available as a Rust library callable from `rmp`.

The synthesizer accepts either:

1. a **residual book** containing only stochastic residual analysis data; or
2. a **full RMP book** containing FOF atoms and an embedded residual book.

Optionally, the synthesizer also accepts an existing sound file containing pre-rendered FOF synthesis and mixes the reconstructed stochastic residual into that signal.

The output is a new sound file.

The synthesizer is intended primarily for offline rendering, but its DSP core should be suitable for later real-time use.

---

# 2. High-level behavior

Supported modes:

```text
ResidualBook
    |
    v
stochastic ERB synthesis
    |
    v
output.wav
```

or:

```text
FullBook
    |
    +--> extract ResidualBook
    |
    v
stochastic ERB synthesis
    |
    +-----------------------------+
                                  |
optional FOF synthesis audio ----+--> mix --> output.wav
```

The standalone executable must not perform FOF atom synthesis itself in version 1.

If the user provides a full RMP book, only its residual section is used by the stochastic synthesizer.

FOF synthesis may be supplied separately as an audio file and mixed into the generated residual.

---

# 3. Scope

## Included

- Rust library API.
- Standalone CLI executable.
- Read residual book.
- Read full RMP book and extract embedded residual section.
- Reconstruct stochastic residual from ERB-band power trajectories.
- Configurable deterministic random seed.
- Per-band stochastic excitation.
- Power-complementary ERB synthesis bank.
- One-pole gain interpolation/smoothing.
- Optional input sound file containing synthesized FOF signal.
- Sample-aligned mixing of FOF audio and stochastic residual.
- WAV output.
- Output gain handling.
- Clipping diagnostics.
- Deterministic rendering.
- Unit tests and integration tests.

## Not included initially

- FOF synthesis from atom data.
- Real-time JACK/PipeWire client.
- FLAC output unless already supported by project dependencies.
- Adaptive residual-book update rates.
- LPC-based residual reconstruction.
- Wavelet synthesis.
- Multichannel ambisonic residual decorrelation policy beyond explicitly supported channel modes.
- Psychoacoustic post-processing.

---

# 4. Suggested crate structure

Recommended crate name:

```text
rmp-resynth
```

or another project-consistent name.

Structure:

```text
rmp-resynth/
    Cargo.toml

    src/
        lib.rs
        main.rs

        config.rs
        error.rs

        book.rs
        render.rs

        erb/
            mod.rs
            bank.rs
            filter.rs
            normalization.rs

        stochastic/
            mod.rs
            rng.rs
            gain.rs
            synth.rs

        audio/
            mod.rs
            reader.rs
            writer.rs
            mixer.rs

    tests/
        cli.rs
        render.rs
        determinism.rs
```

If book types already live in a shared `rmp` crate, reuse them rather than duplicating serialization definitions.

---

# 5. Library architecture

The executable must be a thin wrapper around the public library.

The library should expose:

```rust
pub fn render_residual_book(
    book: &ResidualBook,
    config: &RenderConfig,
) -> Result<AudioBuffer, RenderError>;
```

and:

```rust
pub fn render_full_book(
    book: &RmpBook,
    config: &RenderConfig,
) -> Result<AudioBuffer, RenderError>;
```

For file-oriented use:

```rust
pub fn render_to_file(
    request: &RenderRequest,
) -> Result<RenderReport, RenderError>;
```

Recommended request object:

```rust
pub struct RenderRequest {
    pub book: BookInput,
    pub fof_audio: Option<PathBuf>,
    pub output: PathBuf,
    pub config: RenderConfig,
}
```

Input enum:

```rust
pub enum BookInput {
    Residual(ResidualBook),
    Full(RmpBook),
}
```

File-loading helpers may additionally expose:

```rust
pub fn load_book(path: &Path) -> Result<BookInput, RenderError>;
```

---

# 6. Rendering model

For each ERB band `b`:

\[
y_b[n] = H_b(z) w_b[n]
\]

where:

- `H_b(z)` is a synthesis-band filter;
- `w_b[n]` is a deterministic pseudo-random white-noise sequence;
- each band uses an independent random stream.

The final stochastic residual is:

\[
r_s[n]
=
\sum_{b=0}^{B-1}
g_b[n] y_b[n].
\]

The band target power is taken from the residual book:

\[
P_b[k].
\]

The corresponding target amplitude gain is approximately:

\[
g_{b,\mathrm{target}}[k]
=
\sqrt{
\frac{P_b[k]}{C_b}
}
\]

where `C_b` is the calibrated output noise power of synthesis band `b`.

If filters are normalized to unit noise power:

\[
C_b = 1
\]

and therefore:

\[
g_{b,\mathrm{target}}[k]
=
\sqrt{P_b[k]}.
\]

---

# 7. Power-complementary ERB synthesis bank

The synthesis bank should be ERB-spaced using the same center frequencies and nominal band definitions stored in the residual book.

The desired stochastic reconstruction property is:

\[
\sum_b |H_b(e^{j\omega})|^2
\approx 1.
\]

This is a power-complementary requirement, not a waveform perfect-reconstruction requirement.

Because independent noise streams are used per band, expected output power is approximately:

\[
S_r(\omega,t)
=
\sum_b
g_b^2(t)
|H_b(e^{j\omega})|^2.
\]

This makes squared-magnitude complementarity the relevant design target.

---

# 8. Bank construction

The residual book must contain enough information to reconstruct the synthesis bank:

```rust
pub struct ResidualErbBankDescriptor {
    pub band_count: u32,
    pub center_freq_hz: Vec<f32>,
    pub bandwidth_hz: Vec<f32>,

    pub spacing: ErbSpacing,
    pub filter_kind: ErbFilterKind,
    pub filter_order: u32,
    pub normalization: ErbNormalization,
}
```

The synthesizer should not regenerate center frequencies from only `min_freq_hz`, `max_freq_hz`, and `band_count` if explicit centers are present.

Use explicit stored center frequencies as canonical.

---

# 9. Synthesis filter design

Version 1 should use the same general ERB filter family assumed by the analysis specification.

Recommended initial implementation:

```text
4th-order gammatone-like IIR
```

or the exact filter family adopted by RMP residual analysis.

The synthesis implementation must be encapsulated so the bank can later be replaced.

Suggested trait:

```rust
pub trait SynthesisBand {
    fn reset(&mut self);
    fn process_sample(&mut self, x: f32) -> f32;
    fn process_block(&mut self, input: &[f32], output: &mut [f32]);
}
```

---

# 10. Power-complementary calibration

Do not assume that a bank of normalized gammatone filters automatically satisfies:

\[
\sum_b |H_b|^2 = 1.
\]

Provide a deterministic calibration stage at bank construction.

Recommended approach:

1. evaluate each synthesis-band magnitude response on a dense deterministic frequency grid;
2. compute:
   \[
   W(\omega)=\sum_b|H_b(\omega)|^2;
   \]
3. derive per-band normalization/scaling factors to reduce systematic excess or deficit;
4. verify that:
   \[
   W(\omega)\approx1
   \]
   over the configured frequency range.

The exact first algorithm may be simple.

Do not use random Monte Carlo calibration.

Store or report diagnostics:

```text
min power sum
max power sum
RMS deviation from unity
```

---

# 11. Excitation

Each ERB band uses an independent deterministic PRNG state.

Recommended API:

```rust
pub trait NoiseSource {
    fn next_f32(&mut self) -> f32;
}
```

Recommended properties:

- uniform or approximately Gaussian white excitation;
- zero mean;
- known variance;
- no heap allocation;
- very small per-band state;
- reproducible from seed.

A fast generator such as `xoshiro`/`xoroshiro` class is appropriate.

Avoid cryptographic RNGs.

---

# 12. Seed handling

Rendering must be reproducible.

Top-level configuration:

```rust
pub struct RenderConfig {
    pub seed: u64,
    ...
}
```

Derive each band seed deterministically from:

```text
master seed
+
band index
+
channel index
```

using a stable hash or splitmix-style seed expansion.

Do not use band creation order as an implicit source of nondeterminism.

Given identical:

- residual book;
- render configuration;
- seed;
- sample rate;
- implementation version;

the renderer should produce identical or numerically stable output according to project policy.

---

# 13. Gain interpolation

Residual-book values occur at a fixed update interval:

```text
update_samples
```

At each book frame, load the target power vector.

Two implementation modes are useful.

## 13.1 Default mode: one-pole gain interpolation

Convert book power to target gain:

\[
g_t = \sqrt{P}.
\]

Then smooth:

\[
g[n]
=
a g[n-1]
+
(1-a)g_t[n].
\]

where:

\[
a=e^{-1/(\tau f_s)}.
\]

Configuration:

```text
gain_smoothing_ms
```

The smoothing time constant should be short enough not to erase transient residual structure.

Recommended initial default:

```text
1.0 ms
```

## 13.2 Future mode

Later versions may support:

```text
linear
hold
one-pole
```

For version 1, one-pole is sufficient.

---

# 14. Optional band-dependent gain smoothing

The library should leave room for:

\[
\tau_b
\propto
\frac{1}{B_b}
\]

where `B_b` is ERB bandwidth.

Initial enum:

```rust
pub enum GainSmoothingMode {
    Fixed,
    BandwidthRelative,
}
```

Version 1 may implement only `Fixed` if desired.

If both modes are implemented:

```rust
pub struct GainSmoothingConfig {
    pub mode: GainSmoothingMode,
    pub fixed_ms: f64,
    pub scale: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}
```

---

# 15. Book timing

The residual book stores:

```text
sample_rate
start_sample
source_samples
update_samples
```

Frame `k` becomes active at:

\[
n_k
=
start\_sample
+
k\cdot update\_samples.
\]

The renderer must honor exact sample positions.

Do not reconstruct timing from floating-point seconds.

---

# 16. Output duration

Default residual output duration:

```text
source_samples
```

from the residual book.

If `start_sample > 0`, output should include leading silence unless rendering is explicitly configured to trim it.

Recommended default behavior:

```text
timeline-preserving
```

meaning output sample 0 corresponds to source sample 0.

The residual contribution is zero before `start_sample`.

---

# 17. Optional FOF-audio input

The CLI and library may accept an existing audio file containing synthesized FOF output.

Example:

```text
fofs.wav
```

The stochastic residual is mixed into this signal:

\[
y[n]
=
x_{\mathrm{FOF}}[n]
+
r_s[n].
\]

The FOF file is assumed to be already rendered at the correct timeline position.

No automatic time stretching or resampling should occur unless explicitly configured.

---

# 18. FOF-audio validation

When optional FOF audio is supplied, validate:

1. sample rate matches residual-book sample rate;
2. channel count is compatible;
3. sample format is supported.

Default behavior on sample-rate mismatch:

```text
error
```

Do not silently resample.

Default duration behavior:

- if FOF audio is shorter than residual output, treat missing tail as zero;
- if FOF audio is longer than residual output, output duration becomes the longer of the two unless explicitly configured otherwise.

Recommended:

\[
N_\mathrm{out}
=
\max(
N_\mathrm{residual},
N_\mathrm{FOF}
).
\]

---

# 19. Channel model

Version 1 must explicitly define channel behavior.

Recommended initial policy:

## Mono residual book

If residual book is mono:

- synthesize one stochastic residual channel.

If FOF audio is mono:

- mix directly.

If FOF audio has more than one channel:

- either reject by default;
- or define a configurable mono-to-all-channels policy.

Prefer rejection unless the project's channel semantics are already established.

## Multichannel residual book

If RMP residual analysis later stores per-channel residual data, each channel should have:

- its own power trajectories;
- its own deterministic PRNG derivation;
- independent ERB bank state.

Do not implicitly share stochastic excitation across channels unless specified.

---

# 20. Audio output

Initial required format:

```text
WAV
```

Recommended supported encodings:

```text
float32
pcm24
```

Default:

```text
float32 WAV
```

Configuration:

```rust
pub enum OutputEncoding {
    Float32,
    Pcm24,
}
```

No normalization should occur silently.

---

# 21. Output gain

Provide explicit output gain:

```text
--gain-db <DB>
```

Applied after stochastic synthesis and optional FOF mixing:

\[
y_\mathrm{out}[n]
=
G y[n].
\]

where:

\[
G=10^{g_\mathrm{dB}/20}.
\]

Default:

```text
0 dB
```

---

# 22. Clipping behavior

For float output:

- do not clip by default;
- count samples exceeding `|1.0|`;
- report peak amplitude.

For PCM output:

- default behavior should be configurable.

Recommended initial behavior:

```text
error on clipping
```

or explicit hard clipping only when enabled.

Possible CLI:

```text
--clip
```

Do not silently normalize.

---

# 23. Render configuration

Suggested:

```rust
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub seed: u64,

    pub gain_smoothing: GainSmoothingConfig,

    pub output_gain_db: f64,
    pub output_encoding: OutputEncoding,

    pub clipping: ClippingPolicy,

    pub preserve_timeline: bool,
}
```

---

# 24. Render report

Return a useful report:

```rust
pub struct RenderReport {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples_written: u64,

    pub residual_peak: f32,
    pub mixed_peak: f32,

    pub clipped_samples: u64,

    pub seed: u64,

    pub book_type: RenderBookType,
}
```

This is useful both for CLI output and integration in `rmp`.

---

# 25. Standalone CLI

Suggested executable name:

```text
rmp-resynth
```

Basic usage:

```bash
rmp-resynth \
    --book analysis.book \
    --output reconstructed.wav
```

With pre-rendered FOF audio:

```bash
rmp-resynth \
    --book analysis.book \
    --fof-audio fofs.wav \
    --output reconstructed.wav
```

---

# 26. CLI options

Suggested initial CLI:

```text
--book <PATH>
--fof-audio <PATH>
--output <PATH>

--seed <U64>

--gain-smoothing-ms <MS>

--gain-db <DB>

--encoding <float32|pcm24>

--clip
--no-clip

--verbose
```

Optional:

```text
--preserve-timeline
--trim-to-residual
```

---

# 27. Input book detection

The executable should detect whether `--book` contains:

```text
ResidualBook
```

or:

```text
RmpBook
```

If full book:

```rust
let residual = full_book
    .residual
    .as_ref()
    .ok_or(RenderError::NoResidualBook)?;
```

Do not require separate CLI flags for residual/full book unless the serialization format makes automatic detection impossible.

---

# 28. Library use from RMP

`rmp` should be able to call the renderer directly without spawning the CLI process.

Example:

```rust
use rmp_resynth::{
    render_to_file,
    RenderRequest,
    RenderConfig,
};

let request = RenderRequest {
    book: BookInput::Full(book),
    fof_audio: Some(fof_path),
    output: output_path,
    config,
};

let report = render_to_file(&request)?;
```

The CLI should call the same function.

No DSP logic should live only in `main.rs`.

---

# 29. Block rendering

Even though rendering is offline, implement the DSP core in fixed blocks.

Recommended:

```text
64
128
256
512
```

samples.

The exact block size should not affect book timing.

A book update may occur:

- before block start;
- inside a block;
- multiple times inside a block.

The renderer must split control processing accordingly or process sample-accurate target changes within the block.

---

# 30. Preferred inner-loop structure

For each output block:

```text
clear residual output block

for each ERB band:
    generate noise block
    filter noise block
    apply gain trajectory
    accumulate into residual output block
```

This is preferred over sample-major:

```text
for each sample:
    for each band:
        ...
```

because band-major block processing is more cache/SIMD friendly.

---

# 31. Gain trajectory inside a block

Each band stores:

```rust
struct BandGainState {
    current: f32,
    target: f32,
    coeff: f32,
}
```

When a book frame becomes active:

```text
target <- sqrt(power)
```

The one-pole state continues sample by sample.

Book update events must occur at exact sample indices.

---

# 32. Initial gain state

At render start:

```text
current_gain = 0
target_gain = 0
```

Before the first residual-book frame, residual output is silence.

At first frame, gain rises according to smoothing.

No implicit pre-roll is introduced.

---

# 33. End behavior

At `source_samples`, residual generation stops by default.

Do not append filter tails unless explicitly configured.

Version 1 should truncate exactly at requested output duration.

This keeps alignment with the source timeline.

A future option may allow:

```text
--render-tail-ms
```

---

# 34. Residual-only output

The CLI should support generating only the stochastic component.

This is the default when `--fof-audio` is omitted.

This is important for:

- debugging;
- listening tests;
- comparing deterministic and stochastic components;
- validating the residual model.

---

# 35. Mixed output

When `--fof-audio` is present:

```text
output = FOF audio + residual synthesis
```

No automatic relative gain scaling is applied.

The residual power values are assumed to correspond to the original MP residue scale.

---

# 36. Numerical precision

Recommended:

- filter coefficient design: `f64`;
- bank calibration: `f64`;
- gain coefficient calculation: `f64`;
- runtime audio processing: `f32`;
- mixing: `f32` or `f64` depending on existing project practice.

For offline final rendering, `f64` accumulation of all ERB bands may be worth considering if band count is large.

A practical compromise:

```text
band processing: f32
sum accumulator: f64
output: f32
```

Benchmark before optimizing.

---

# 37. Performance targets

Even offline, the DSP core should be real-time capable on a modern desktop for likely configurations such as:

```text
48 ERB bands
48 kHz
mono
```

Reason:

The same library should later be reusable in an online synthesis engine.

Avoid:

- heap allocation in audio loops;
- per-sample dynamic dispatch;
- per-sample square roots;
- coefficient redesign during rendering;
- locks in audio loops.

---

# 38. Avoid per-sample square roots

Book powers arrive only at update frames.

Convert:

```text
power -> target_gain
```

only when a book frame is loaded.

Thus:

```rust
target_gain[b] = power[b].sqrt();
```

is evaluated at control rate, not audio rate.

---

# 39. Output file writing

Use streaming output rather than rendering the whole file into memory.

Preferred pipeline:

```text
load book metadata
open optional FOF audio reader
open output writer

for output blocks:
    synthesize residual block
    read FOF block if present
    mix
    apply output gain
    write block

finalize WAV
```

The library may also offer in-memory rendering for tests and short files.

---

# 40. Error model

Suggested errors:

```rust
pub enum RenderError {
    Io(...),
    InvalidBook(...),
    NoResidualBook,

    UnsupportedBookVersion,
    UnsupportedResidualBookVersion,

    UnsupportedFilterKind,
    UnsupportedFilterOrder,

    SampleRateMismatch {
        book: u32,
        fof_audio: u32,
    },

    ChannelMismatch,

    InvalidSeed,
    InvalidGainSmoothing,

    OutputEncodingUnsupported,

    ClippingDetected {
        count: u64,
        peak: f32,
    },

    ...
}
```

Integrate with project-wide error handling.

---

# 41. CLI diagnostics

Example:

```text
Residual synthesis:
  book:                 analysis.book
  source type:          full RMP book
  sample rate:          48000 Hz
  ERB bands:            48
  residual frames:      183421
  update interval:      48 samples / 1.000 ms
  seed:                 12345678
  gain smoothing:       1.0 ms

FOF audio:
  input:                fofs.wav
  channels:             1
  samples:              8804208

Output:
  file:                 reconstructed.wav
  encoding:             float32
  samples written:      8804208
  peak:                 0.91
  samples > 1.0:        0
```

---

# 42. Settings-file support

The standalone executable may optionally support a TOML settings file.

Suggested:

```toml
[synthesis]
seed = 1

gain_smoothing_ms = 1.0

output_gain_db = 0.0
encoding = "float32"
clip = false
preserve_timeline = true
```

CLI precedence:

```text
defaults
<
settings file
<
CLI
```

---

# 43. Determinism requirements

Given the same:

- input book;
- FOF audio;
- seed;
- render config;
- crate version;
- CPU floating-point model where relevant;

the output should be reproducible.

Parallel rendering must not change summation order unless the project explicitly accepts small floating-point differences.

For strict determinism, either:

- render bands in a fixed order;
- or use deterministic chunk reduction.

---

# 44. Tests

## 44.1 Zero residual

Residual book with all powers zero.

Expected output:

```text
all zeros
```

unless FOF audio is provided.

## 44.2 Constant equal band power

Set identical constant power in all bands.

Expected:

- approximately stationary broadband noise;
- no book-update artifacts.

## 44.3 Single active band

Activate one ERB band only.

Expected:

- noise localized around that band's center frequency.

## 44.4 Time-varying gain

Create a short band-power burst.

Expected:

- corresponding short stochastic burst;
- no zipper noise;
- timing matches book sample positions.

## 44.5 Multi-band transient

Activate many high-frequency bands for a few milliseconds.

Expected:

- broadband transient retained at correct time.

## 44.6 Determinism

Render twice with same seed.

Expected:

```text
identical output
```

within project's determinism standard.

Render with different seed.

Expected:

- different waveform;
- statistically similar spectral envelope.

## 44.7 Full-book extraction

Provide full RMP book with residual section.

Expected:

- same stochastic output as rendering embedded residual book directly.

## 44.8 Missing residual section

Full book without residual section.

Expected:

```text
NoResidualBook
```

## 44.9 FOF mix

Use known test waveform as FOF input and zero residual.

Expected output equals FOF file exactly, modulo output encoding conversion.

## 44.10 Sample-rate mismatch

Book at 48 kHz, FOF audio at 44.1 kHz.

Expected:

```text
error
```

## 44.11 Output duration

Verify output sample count follows:

```text
max(residual_duration, fof_audio_duration)
```

unless an explicit trim mode overrides it.

## 44.12 Clipping

Generate signal > 1.0.

Verify:

- float output reports overs;
- PCM output follows configured clipping policy.

---

# 45. Integration tests against analysis

Use synthetic source cases:

## A. White noise

1. analyze residual;
2. create residual book;
3. synthesize;
4. compare long-term ERB-band powers.

Expected:

```text
synthesized band powers approximately match analyzed band powers
```

## B. Colored noise

1. generate deterministic colored-noise fixture;
2. analyze;
3. synthesize;
4. compare ERB power trajectories.

Expected:

- approximate spectral match;
- waveform different;
- temporal power structure preserved.

## C. Noise burst

Expected:

- burst time and duration retained.

These are more meaningful than sample-by-sample waveform tests.

---

# 46. Acceptance criteria

Version 1 is complete when:

1. a standalone executable can render a residual book to WAV;
2. a full RMP book can be supplied directly;
3. an optional FOF synthesis sound file can be mixed into the residual;
4. the same rendering engine is callable as a Rust library;
5. rendering is deterministic for a fixed seed;
6. ERB synthesis uses independent stochastic excitation per band;
7. band gains follow book power trajectories sample-accurately;
8. gain transitions are smoothed with a configurable one-pole interpolator;
9. sample-rate mismatches are rejected;
10. output duration and timeline behavior are defined and tested;
11. output can be float32 WAV;
12. clipping is measured and reported;
13. no heap allocation occurs in inner DSP loops;
14. analysis/synthesis integration tests reproduce residual ERB power trajectories within defined tolerances.

---

# 47. Suggested initial defaults

```toml
[synthesis]
seed = 1

gain_smoothing_ms = 1.0

output_gain_db = 0.0
encoding = "float32"

clip = false
preserve_timeline = true
```

---

# 48. Initial architecture summary

The synthesis system is:

\[
\boxed{
\text{ResidualBook}
\rightarrow
\text{ERB power trajectories}
\rightarrow
\text{smoothed band gains}
\rightarrow
\text{independent white-noise sources}
\rightarrow
\text{power-complementary ERB synthesis bank}
\rightarrow
\text{stochastic residual}
}
\]

and optionally:

\[
\boxed{
\text{FOF audio}
+
\text{stochastic residual}
\rightarrow
\text{output sound file}
}
\]

The implementation should be developed as a reusable Rust library first, with the CLI executable acting only as a file/configuration front end.
