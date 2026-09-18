# RMP Structural Book Analysis Crate Specification

## 1. Purpose

Add a new Rust crate to the RMP workspace for extracting higher-level structure from an existing Matching Pursuit book.

The crate operates **after Matching Pursuit decomposition**. It must not modify the MP algorithm, atom selection, or residue calculation.

The primary transformation is:

```text
MP atoms
   ↓
persistent partials
   ↓
partial trajectories
   ↓
morphological relationships
   ↓
candidate stems / sources
```

The main goals are:

1. Extract persistent spectral partials from an MP book while suppressing tiny temporal and spectral details.
2. Represent each partial as a slowly varying frequency/amplitude trajectory rather than as a collection of individual atoms.
3. Detect relationships between partials using:
   - common onset,
   - common offset,
   - common amplitude morphology,
   - common frequency morphology,
   - harmonic relationships,
   - spectral-envelope relationships,
   - optional spatial/multichannel relationships.
4. Group related partials into candidate stems.
5. Preserve uncertainty rather than forcing every partial into exactly one stem.

The result is a simplified structural interpretation of an MP book.

---

## 2. Crate

Recommended workspace layout:

```text
rmp/
    Cargo.toml

    rmp-core/
    rmp-cli/
    rmp-gui/
    rmp-synthesis/
    rmp-structure/
```

Crate name:

```text
rmp-structure
```

The crate should primarily be a library.

`rmp-cli` may expose commands using this crate, but the structural-analysis algorithms should live entirely in `rmp-structure`.

Dependency direction should preferably be:

```text
rmp-core
   ↑
rmp-structure
   ↑
rmp-cli
   ↑
rmp-gui
```

`rmp-structure` may depend on the MP book and atom types exported by `rmp-core`.

It must not depend on `rmp-cli` or `rmp-gui`.

---

## 3. Scope

The first implementation should support both atom families currently used by RMP:

```text
FOF atoms
Gaussian atoms
```

The structural analysis must normalize these atom types into a common internal representation.

Atom-specific properties must remain available where useful, but later stages should not normally need to know whether an observation originated from a FOF or Gaussian atom.

---

## 4. Non-goals

The initial implementation does not attempt to:

- perform blind source separation directly on audio;
- modify the original MP decomposition;
- synthesize audio;
- reconstruct stems into audio files;
- recognize instruments;
- classify musical notes;
- infer musical scores;
- force every atom into a partial;
- force every partial into a stem;
- provide sample-accurate source boundaries.

The crate performs structural inference from an already existing MP book.

---

## 5. High-level architecture

The crate should contain approximately these modules:

```text
rmp-structure/src/

    lib.rs

    config.rs
    error.rs

    observation.rs

    partial/
        mod.rs
        accumulation.rs
        peaks.rs
        tracking.rs
        simplify.rs
        metrics.rs

    morphology/
        mod.rs
        amplitude.rs
        frequency.rs
        onset.rs
        harmonic.rs
        spectral.rs
        spatial.rs

    stem/
        mod.rs
        graph.rs
        clustering.rs
        refine.rs
        membership.rs

    output/
        mod.rs
        partial_book.rs
        stem_book.rs

    analysis.rs
```

The exact module layout may evolve, but atom normalization, partial extraction, morphology analysis, and stem clustering should remain separate concepts.

---

## 6. Processing stages

The full analysis pipeline is:

```text
MP Book
   │
   ▼
Atom normalization
   │
   ▼
Coarse time/frequency energy representation
   │
   ▼
Persistent spectral ridge detection
   │
   ▼
Partial tracking
   │
   ▼
Trajectory simplification
   │
   ▼
Partial Book
   │
   ▼
Pairwise/group morphology analysis
   │
   ▼
Affinity graph
   │
   ▼
Stem clustering
   │
   ▼
Stem refinement
   │
   ▼
Stem Book
```

Partial extraction and stem extraction must be independently callable.

---

## 7. Normalized atom observations

Every supported MP atom must first be converted into a common representation.

Suggested type:

```rust
pub struct AtomObservation {
    pub atom_id: AtomId,

    pub channel: Option<u32>,

    pub time_center_samples: u64,
    pub start_samples: u64,
    pub end_samples: u64,

    pub frequency_hz: f64,

    pub energy: f64,

    pub effective_bandwidth_hz: f64,
    pub effective_duration_samples: f64,

    pub phase: Option<f64>,

    pub atom_kind: AtomKind,
}
```

with:

```rust
pub enum AtomKind {
    Fof,
    Gaussian,
}
```

The normalization should calculate physically comparable quantities where possible.

In particular:

```text
center time
effective duration
center frequency
effective bandwidth
energy
```

must have consistent meanings across atom families.

The exact definition of effective bandwidth and duration should be documented for every atom type.

---

## 8. Principle of partial extraction

A persistent partial must not be defined as a single MP atom.

A partial is a **persistent concentration of energy near a continuously evolving frequency**.

Several weak atoms occurring consistently around the same frequency may constitute a significant partial even when none of the individual atoms passes a simple energy threshold.

Therefore, the implementation must aggregate evidence before rejecting low-level detail.

Do not begin partial extraction by globally deleting all low-energy atoms.

---

## 9. Time-frequency accumulation

The first partial-analysis stage builds a deliberately coarse time-frequency representation.

Suggested configurable defaults:

```text
time step:            10 ms
analysis window:      40 ms
frequency resolution: 20 cents
frequency scale:      logarithmic
```

Alternative frequency scales should eventually be supported:

```rust
pub enum FrequencyScale {
    LinearHz {
        bin_width_hz: f64,
    },

    LogCents {
        cents_per_bin: f64,
        reference_hz: f64,
    },

    Erb {
        bands_per_erb: f64,
    },
}
```

For the first implementation, `LogCents` is preferred.

---

## 10. Atom contribution

Atoms should contribute energy over an area rather than simply being assigned to one time-frequency cell.

Each observation contributes using kernels:

```text
K_t(t - t_i)
K_f(f - f_i)
```

so that:

```text
P(t,f) = Σ E_i K_t(t-t_i) K_f(f-f_i)
```

The exact kernels may initially be simple Gaussian kernels based on the atom's effective duration and bandwidth.

This representation serves as an intermediate statistical representation and does not need to reproduce the original signal.

---

## 11. Persistent ridge detection

Candidate partials are detected as ridges in the accumulated energy representation.

A candidate ridge is a sequence:

```text
(t0, f0, e0)
(t1, f1, e1)
...
(tn, fn, en)
```

where adjacent observations satisfy frequency-continuity constraints.

Typical conditions:

```text
maximum frequency jump per frame
maximum missing-frame gap
minimum accumulated energy
minimum duration
minimum persistence
```

Suggested configuration:

```rust
pub struct PartialTrackingConfig {
    pub max_jump_cents_per_frame: f64,
    pub max_gap_frames: usize,

    pub min_duration_ms: f64,
    pub min_persistence: f64,

    pub min_relative_level_db: f64,
}
```

Suggested initial defaults:

```text
max jump:             50 cents/frame
maximum gap:          2 frames
minimum duration:     100 ms
minimum persistence:  0.50
relative level floor: -60 dB
```

These values must remain configurable.

---

## 12. Persistence

Persistence measures how consistently a partial exists during its nominal lifetime.

One simple definition is:

```text
number of frames containing sufficient partial energy
------------------------------------------------------
number of frames between partial start and end
```

This should produce:

```text
0.0 <= persistence <= 1.0
```

Persistence should be one of the primary mechanisms used to reject tiny MP details.

---

## 13. Partial significance

A significance measure should be available for ranking and filtering.

Initial form:

```text
significance = normalized_energy × persistence
```

A more general configurable form may later include duration:

```text
S = E^a × P^b × D^c
```

The first version should avoid excessive parameterization.

Store the components separately so alternative significance measures can be tested later.

---

## 14. Partial representation

The extracted partial should contain both summary information and trajectories.

Suggested type:

```rust
pub struct Partial {
    pub id: PartialId,

    pub start_samples: u64,
    pub end_samples: u64,

    pub mean_frequency_hz: f64,
    pub geometric_mean_frequency_hz: f64,

    pub frequency_std_cents: f64,

    pub mean_amplitude: f64,
    pub energy: f64,

    pub persistence: f64,
    pub significance: f64,

    pub frequency: Trajectory,
    pub amplitude: Trajectory,

    pub supporting_atoms: Vec<AtomId>,
}
```

`supporting_atoms` is important for tracing structural results back to the original MP book.

For very large books, storing all IDs may later become optional.

---

## 15. Trajectory representation

Do not preserve frame-by-frame detail unnecessarily.

A partial should contain a simplified trajectory.

Suggested generic type:

```rust
pub struct Trajectory {
    pub points: Vec<TrajectoryPoint>,
}

pub struct TrajectoryPoint {
    pub time_samples: u64,
    pub value: f64,
}
```

The frequency trajectory should normally use Hz externally but may use log-frequency/cents internally.

Amplitude should normally be represented in logarithmic magnitude for analysis.

---

## 16. Trajectory simplification

After partial tracking, trajectories should be reduced to remove insignificant details.

Possible first implementation:

```text
low-pass smoothing
+
Ramer-Douglas-Peucker-style trajectory simplification
```

Separate tolerances should exist for:

```text
frequency deviation in cents
amplitude deviation in dB
```

Suggested initial tolerances:

```text
frequency: 5 cents
amplitude:  0.5 dB
```

These are starting values only.

---

## 17. Partial book

Partial extraction produces a new serializable object.

```rust
pub struct PartialBook {
    pub metadata: PartialBookMetadata,
    pub partials: Vec<Partial>,
}
```

Metadata should contain enough information to reproduce the analysis:

```rust
pub struct PartialBookMetadata {
    pub sample_rate: f64,
    pub source_book: Option<PathBuf>,

    pub config: PartialAnalysisConfig,

    pub version: String,
}
```

The original MP book must not be overwritten.

---

## 18. Morphological analysis

Stem inference works primarily from partial trajectories.

For each pair of temporally overlapping partials, calculate similarities for several independent cues.

Suggested representation:

```rust
pub struct PartialAffinity {
    pub a: PartialId,
    pub b: PartialId,

    pub onset: f32,
    pub offset: f32,

    pub amplitude_morphology: f32,
    pub frequency_morphology: f32,

    pub harmonic: f32,
    pub spectral: f32,

    pub spatial: Option<f32>,

    pub combined: f32,
}
```

Each score should normally be normalized to:

```text
0.0 .. 1.0
```

Scores must remain available separately even after calculating the combined affinity.

---

## 19. Common onset

Common onset is an important grouping cue.

For two partials:

```text
Δt = |onset_a - onset_b|
```

Convert this into a similarity using a smooth kernel rather than a hard threshold.

For example:

```text
S_onset = exp(-(Δt / τ)^2)
```

with configurable `τ`.

Suggested default:

```text
τ = 20 ms
```

Common onset should be strong evidence but must never be sufficient by itself to merge partials.

A chord or ensemble attack can produce many unrelated partials with a common onset.

---

## 20. Common offset

Offset/release similarity is calculated analogously to onset.

This can help distinguish partials belonging to sounds with different durations even when their attacks coincide.

---

## 21. Amplitude morphology

For overlapping portions of two partials, resample their amplitude trajectories onto a common low-rate grid.

Suggested analysis rate:

```text
100 Hz
```

Represent amplitude in dB.

Remove mean level so that morphology comparison is largely independent of absolute partial strength.

Compare:

```text
a'_i(t) = d/dt a_i(t)
```

rather than only comparing raw levels.

The score may initially be based on normalized cross-correlation.

This captures common:

```text
attack shape
decay
tremolo
amplitude modulation
phrase envelope
```

The comparison should tolerate small timing offsets.

A configurable ±20 ms lag search is appropriate for the first implementation.

---

## 22. Frequency morphology

Frequency morphology should be evaluated in cents relative to each partial's own central frequency.

Define:

```text
δf_i(t) = 1200 log2(f_i(t) / f̄_i)
```

Compare `δf_i(t)` or its derivative between overlapping partials.

This permits detection of shared:

```text
vibrato
pitch drift
glissando
frequency modulation
microtonal fluctuations
```

without requiring the partials to occupy similar absolute frequency ranges.

---

## 23. Harmonic relationship

For a pair or group of partials, determine whether frequencies are consistent with a common fundamental.

A pairwise score can begin with:

```text
ratio = f_high / f_low
```

and measure distance from a plausible low-order rational/harmonic relationship.

However, harmonicity should ultimately be evaluated at the group level because pairwise harmonic tests are often ambiguous.

The first implementation may include pairwise harmonic evidence but must not let it dominate all morphological evidence.

---

## 24. Common fundamental estimation

For candidate groups, optionally estimate an `f0` trajectory.

Do not require every stem to be harmonic.

The API should therefore use:

```rust
pub enum Fundamental {
    None,
    Static(f64),
    Trajectory(Trajectory),
}
```

or equivalent.

Inharmonic and noise-like stems must remain representable.

---

## 25. Spectral-envelope consistency

Once a provisional stem contains several partials, calculate its spectral envelope.

Compare the relative amplitudes of partials over time.

If a collection of partials maintains a coherent spectral shape, this is additional evidence that they belong to the same source.

This cue should initially be used during stem refinement rather than initial pairwise clustering.

---

## 26. Multichannel/spatial morphology

The design must permit future multichannel analysis.

When an MP book contains spatial/channel information, partials should optionally contain a spatial descriptor.

Potential cues include:

```text
channel energy ratios
inter-channel phase
estimated direction
ambisonic direction
spatial trajectory
```

The first implementation may return:

```rust
spatial: None
```

but the architecture must not assume mono input.

---

## 27. Affinity combination

The combined affinity between two partials should initially use a weighted sum:

```text
W(i,j) =
    w_onset  S_onset
  + w_offset S_offset
  + w_amp    S_amp
  + w_freq   S_freq
  + w_harm   S_harm
  + w_spec   S_spec
  + w_space  S_space
```

Weights belong in configuration.

Suggested initial emphasis:

```text
amplitude morphology   high
frequency morphology   high
onset                   medium-high
offset                  medium
harmonicity             medium
spectral envelope       medium
spatial                  high when available
```

Do not hard-code these qualitative priorities into the API.

---

## 28. Affinity graph

Stem extraction should operate on a graph:

```text
node = partial
edge = evidence that two partials belong to the same source
edge weight = combined affinity
```

Do not create edges for every pair in very large books.

Only compare partials whose lifetimes overlap sufficiently.

Suggested pruning criteria:

```text
minimum temporal overlap
reasonable frequency range
minimum individual significance
```

This should make graph construction approximately local in time rather than O(N²) for the complete book.

---

## 29. Seed groups

Initial groups should be constructed only from strong evidence.

For example:

```text
common onset
AND
strong amplitude morphology
```

or:

```text
strong frequency morphology
AND
compatible harmonic relationship
```

The exact rule should be configurable.

The purpose is to form small high-confidence groups before using weaker evidence.

---

## 30. Stem clustering

The first implementation should prefer a relatively simple graph clustering method over a sophisticated machine-learning approach.

Suitable starting methods include:

```text
connected components after a strong edge threshold
+
iterative agglomerative merging
```

or a weighted community clustering algorithm.

The algorithm must be deterministic for identical input/configuration.

Avoid adding a large ML dependency for the initial version.

---

## 31. Hierarchical grouping

Stem grouping should be hierarchical.

Recommended procedure:

```text
partials
   ↓
high-confidence seed groups
   ↓
group-level morphology calculation
   ↓
merge compatible groups
   ↓
attach weaker peripheral partials
   ↓
leave ambiguous partials uncertain/unassigned
```

This is preferable to immediately partitioning all partials globally.

---

## 32. Stem representation

Suggested result:

```rust
pub struct Stem {
    pub id: StemId,

    pub start_samples: u64,
    pub end_samples: u64,

    pub partials: Vec<StemPartialMembership>,

    pub fundamental: Fundamental,

    pub amplitude: Trajectory,

    pub confidence: f32,

    pub morphology: StemMorphology,
}
```

with:

```rust
pub struct StemPartialMembership {
    pub partial_id: PartialId,
    pub membership: f32,
}
```

Membership must be normalized:

```text
0.0 .. 1.0
```

---

## 33. Uncertain membership

Do not force every partial into exactly one stem.

The output should support:

```text
partial 31:
    stem 2       0.82
    stem 5       0.11
    unassigned   0.07
```

This can either be represented directly in `StemBook` or through an inverse membership table.

For version 1, it is acceptable to assign a primary stem while preserving an explicit confidence score.

The data model should nevertheless allow multiple memberships later without format redesign.

---

## 34. Stem morphology

Suggested type:

```rust
pub struct StemMorphology {
    pub mean_onset_samples: u64,
    pub onset_spread_samples: f64,

    pub mean_offset_samples: u64,
    pub offset_spread_samples: f64,

    pub harmonicity: f32,

    pub amplitude_coherence: f32,
    pub frequency_coherence: f32,

    pub spectral_coherence: f32,

    pub spatial_coherence: Option<f32>,
}
```

These values make stem grouping inspectable and useful in the GUI.

---

## 35. Stem book

The final output is:

```rust
pub struct StemBook {
    pub metadata: StemBookMetadata,
    pub stems: Vec<Stem>,
    pub unassigned_partials: Vec<PartialId>,
}
```

Metadata should record:

```text
source MP book
source partial book
analysis configuration
RMP version
rmp-structure version
```

---

## 36. Public API

The top-level API should be simple.

Suggested:

```rust
pub fn analyze_partials(
    book: &MpBook,
    config: &PartialAnalysisConfig,
) -> Result<PartialBook>;

pub fn analyze_stems(
    partials: &PartialBook,
    config: &StemAnalysisConfig,
) -> Result<StemBook>;

pub fn analyze_structure(
    book: &MpBook,
    config: &StructureAnalysisConfig,
) -> Result<StructureAnalysis>;
```

with:

```rust
pub struct StructureAnalysis {
    pub partials: PartialBook,
    pub stems: StemBook,
}
```

---

## 37. Configuration

Top-level configuration:

```rust
pub struct StructureAnalysisConfig {
    pub partials: PartialAnalysisConfig,
    pub stems: StemAnalysisConfig,
}
```

Example:

```toml
[structure.partials]
time_step_ms = 10.0
window_ms = 40.0
frequency_scale = "log-cents"
cents_per_bin = 20.0

min_duration_ms = 100.0
min_persistence = 0.50
min_relative_level_db = -60.0

max_jump_cents_per_frame = 50.0
max_gap_frames = 2

frequency_simplify_cents = 5.0
amplitude_simplify_db = 0.5

[structure.stems]
morphology_rate_hz = 100.0

onset_tau_ms = 20.0
offset_tau_ms = 30.0

max_morphology_lag_ms = 20.0

weight_onset = 1.0
weight_offset = 0.5
weight_amplitude = 2.0
weight_frequency = 2.0
weight_harmonic = 1.0
weight_spectral = 1.0
weight_spatial = 2.0

seed_affinity = 0.80
merge_affinity = 0.65

min_partial_overlap = 0.30
min_stem_confidence = 0.50
```

The existing RMP configuration mechanism should be reused rather than creating a separate incompatible parser.

---

## 38. CLI integration

`rmp-cli` should eventually expose:

```text
rmp analyze-structure input.rmpbook
```

Useful alternatives:

```text
rmp analyze-partials input.rmpbook
rmp analyze-stems input.partialbook
```

Possible output:

```text
input.partialbook
input.stembook
```

or JSON/MessagePack equivalents depending on the existing RMP book serialization mechanism.

CLI naming should follow existing RMP conventions.

---

## 39. Diagnostics

Analysis should produce optional diagnostics useful for development and GUI visualization.

Possible diagnostics include:

```text
number of input atoms
number of normalized observations
number of candidate ridges
number of accepted partials
number of rejected short ridges
number of rejected weak ridges

number of graph nodes
number of graph edges
number of seed groups
number of final stems
number of unassigned partials
```

Do not print from library code.

Expose diagnostics as returned structures or through the existing logging infrastructure.

---

## 40. GUI-oriented inspection data

The API should make it straightforward for `rmp-gui` to draw:

```text
MP atom cloud
partial trajectories
partial significance
partial/stem colors
stem memberships
affinity between selected partials
common onset groups
amplitude morphology
frequency morphology
```

For this reason, intermediate results should be inspectable rather than hidden entirely inside the analysis function.

Potential API:

```rust
pub struct StemAnalysisDiagnostics {
    pub affinities: Vec<PartialAffinity>,
    pub seed_groups: Vec<Vec<PartialId>>,
}
```

Diagnostics may be optionally generated to avoid memory cost in batch operation.

---

## 41. Determinism

The structural analysis must be deterministic.

Given:

```text
same MP book
same configuration
same version
```

it should produce the same result.

If parallel processing is used, floating-point reductions should be handled carefully enough that minor scheduling differences do not alter cluster topology.

Stable ordering should be imposed before IDs are generated.

---

## 42. IDs

`PartialId` and `StemId` should be stable within an analysis result.

A simple sequential representation is sufficient initially:

```rust
pub struct PartialId(pub u32);
pub struct StemId(pub u32);
```

IDs should be assigned after deterministic sorting.

For example, partials can be sorted by:

```text
start time
then mean frequency
then energy
```

---

## 43. Performance

Structural analysis is offline and does not have hard real-time requirements.

Nevertheless, typical analysis should avoid unnecessary quadratic behavior.

Important optimizations:

```text
time-index partials
only compare temporally overlapping partials
use sparse affinity graphs
avoid repeated trajectory resampling
cache morphology vectors
parallelize independent pair comparisons
```

Rayon may be used if already acceptable within the RMP dependency policy.

The design should comfortably handle books containing tens or hundreds of thousands of MP atoms.

---

## 44. Numerical representation

Use:

```text
f64
```

for analysis quantities unless profiling demonstrates a significant advantage from `f32`.

Use sample positions as integer sample indices where possible.

Do not accumulate timing internally using floating-point seconds.

---

## 45. Testing

Testing should be performed at several levels.

### Unit tests

Test independently:

```text
onset similarity
offset similarity
amplitude morphology correlation
frequency morphology correlation
harmonicity score
trajectory simplification
persistence
ridge linking
```

### Synthetic partial tests

Generate artificial partial trajectories with known relationships.

Examples:

```text
same onset + same envelope
same onset + unrelated envelopes
different onset + same vibrato
harmonic partials with common vibrato
two independent harmonic sources
crossing partials
weak persistent partial
strong isolated atom
```

The weak persistent partial should survive.

The strong isolated atom should normally be rejected.

### Atom-family tests

Construct equivalent sounds represented using:

```text
FOF atoms
Gaussian atoms
```

and verify that normalized partial extraction gives approximately equivalent structural results.

### Regression tests

Keep a small collection of MP books and serialize expected summary outputs.

Do not require bit-identical floating-point trajectory values unless appropriate.

Test structural invariants instead.

---

## 46. Important edge cases

The implementation must explicitly test:

### Crossing partials

Two partials crossing in frequency must not automatically exchange identities.

Morphology before and after the crossing should help maintain track identity.

### Vibrato

A partial with strong vibrato must remain one partial rather than fragmenting into several neighboring tracks.

### Tremolo

Shared amplitude modulation should increase stem affinity.

### Chords with simultaneous onset

Common onset alone must not collapse all partials into one stem.

### Quiet persistent partials

A low-energy but persistent partial should survive where possible.

### Loud transients

A high-energy short event should not automatically become a steady partial.

### Inharmonic sources

Stem formation must work without harmonicity.

### Missing partial sections

Short gaps in MP representation should not split an otherwise continuous partial.

---

## 47. Initial implementation phases

Implementation should proceed in this order.

### Phase 1 — normalized observations

Implement:

```text
FOF → AtomObservation
Gaussian → AtomObservation
```

with tests.

### Phase 2 — persistent partial extraction

Implement:

```text
TF accumulation
peak extraction
ridge tracking
persistence
trajectory construction
simplification
PartialBook
```

Do not implement stem extraction until this stage can be inspected and validated.

### Phase 3 — morphology metrics

Implement independently:

```text
onset
offset
amplitude morphology
frequency morphology
harmonicity
```

Provide pairwise diagnostic output.

### Phase 4 — affinity graph

Implement:

```text
temporal candidate selection
pairwise affinity
sparse graph
```

### Phase 5 — stem clustering

Implement:

```text
strong seed groups
agglomerative merging
membership/confidence
StemBook
```

### Phase 6 — refinement

Add:

```text
spectral-envelope consistency
group-level f0
spatial morphology
better ambiguity handling
```

---

## 48. Acceptance criteria for partial extraction

The first usable partial-analysis implementation should satisfy all of the following:

1. It reads an existing RMP MP book.
2. It supports both FOF and Gaussian atoms.
3. It extracts persistent frequency trajectories.
4. A steady sinusoidal component represented by many MP atoms becomes one partial.
5. Small atom-level fluctuations are removed.
6. Weak persistent components are not rejected merely because individual atoms are weak.
7. Short isolated details are strongly suppressed.
8. The resulting partials retain references to their supporting MP atoms.
9. Results are deterministic.
10. Parameters are configurable.

---

## 49. Acceptance criteria for stem extraction

The first usable stem-analysis implementation should demonstrate:

1. Partials with a common onset and strongly correlated amplitude morphology tend to group.
2. Partials with common vibrato tend to group even at different absolute frequencies.
3. Simultaneously starting but morphologically unrelated partials are not automatically grouped.
4. Harmonicity contributes useful evidence without being mandatory.
5. Two overlapping harmonic sources can produce two candidate stems.
6. Low-confidence partials may remain unassigned.
7. Every grouping decision can be inspected through component affinity scores.
8. Results are deterministic.

---

## 50. Design principle

The central principle is that RMP already contains a detailed description of the sound.

`rmp-structure` should not attempt to reproduce another detailed decomposition.

Its purpose is to deliberately throw away detail and expose longer-lived organization:

```text
MP atom
    ↓
local acoustic event

Partial
    ↓
persistent spectral component

Stem
    ↓
group of partials sharing temporal and morphological behavior
```

The distinction between these three levels should remain clear throughout the implementation.

The MP book remains the authoritative detailed representation.

The partial book and stem book are derived interpretations that can be regenerated with different structural-analysis settings.
