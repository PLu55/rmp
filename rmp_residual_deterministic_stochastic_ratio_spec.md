# RMP Residual Deterministic/Stochastic Ratio — Additional Implementation Specification

## 1. Purpose

Extend the residual-analysis stage in **rmp** with a diagnostic that estimates how much of the final residue remains **deterministic/predictable** versus **stochastic/noise-like**.

The metric is not intended to be a mathematically exact probability. It is a practical analysis quantity for:

- evaluating whether the final FOF residue is suitable for stochastic resynthesis;
- comparing pursuit settings;
- diagnosing missed deterministic structure;
- potentially informing a future FOF-MP stopping criterion.

The calculation should reuse the ERB-band residual signals already produced by the residual analysis.

## 2. Design goal

For each ERB band `b` and analysis frame `k`, estimate a local deterministic ratio

\[
D_b[k] \in [0,1]
\]

and corresponding stochastic ratio

\[
S_b[k] = 1 - D_b[k].
\]

Then derive a global energy-weighted ratio:

\[
D[k] =
\frac{
\sum_b P_b[k] D_b[k]
}{
\sum_b P_b[k]
}
\]

and

\[
S[k] = 1-D[k].
\]

Here `P_b[k]` is the already computed residual band power.

The first implementation should derive `D_b[k]` primarily from **linear-prediction gain**.

## 3. Why linear prediction

A deterministic or strongly structured signal is locally predictable.

A stochastic residual is less predictable after accounting for its local spectral coloration.

For ERB-band signal `r_b[n]`, fit an AR model:

\[
r_b[n]
=
-\sum_{i=1}^{p} a_i r_b[n-i]
+
e_b[n].
\]

Let

\[
\sigma_r^2 = \operatorname{var}(r_b)
\]

and

\[
\sigma_e^2 = \operatorname{var}(e_b).
\]

Prediction gain:

\[
G_{p,b}
=
10\log_{10}
\left(
\frac{\sigma_r^2}{\sigma_e^2}
\right).
\]

High `G_p` indicates strong predictability.

Low `G_p` indicates a more stochastic residual.

## 4. Frame-based analysis

The deterministic/stochastic ratio is calculated on short analysis windows.

This is separate from the residual-book update interval.

Introduce:

```text
residual.detstoch.window_ms
residual.detstoch.hop_ms
```

Recommended initial defaults:

```text
window_ms = 20.0
hop_ms = 5.0
```

The window must be long enough to estimate low-order LPC robustly, but short enough to track local changes.

## 5. Per-band LPC configuration

Add:

```text
residual.detstoch.lpc_order
```

Recommended initial default:

```text
lpc_order = 6
```

Expected useful range:

```text
4 .. 10
```

The first implementation should use the same LPC order in all bands.

Future versions may use band-dependent order.

## 6. LPC estimation method

Use autocorrelation LPC with a stable solver.

Recommended procedure:

1. Remove local DC mean from the analysis frame.
2. Compute autocorrelation `R[m]` for `m = 0..p`.
3. Solve the Yule-Walker system using Levinson-Durbin.
4. Compute prediction-error variance directly from the recursion.

Do not estimate prediction gain by explicitly filtering the whole frame unless needed for verification.

## 7. Prediction-gain normalization

Raw prediction gain in dB is unbounded and not directly a ratio.

Map it into:

\[
D_b[k] \in [0,1].
\]

Initial recommended mapping:

\[
D_b[k]
=
\operatorname{clamp}
\left(
\frac{G_{p,b}[k]-G_{\min}}
     {G_{\max}-G_{\min}},
0,
1
\right).
\]

Suggested initial defaults:

```text
G_min = 0 dB
G_max = 12 dB
```

Thus:

- `0 dB` prediction gain -> fully stochastic:
  \[
  D=0
  \]
- `12 dB` or more -> strongly deterministic:
  \[
  D=1
  \]

These thresholds are empirical tuning parameters and must be configurable.

Add:

```text
residual.detstoch.pred_gain_min_db
residual.detstoch.pred_gain_max_db
```

## 8. Silence / low-energy handling

Prediction gain becomes unreliable in very low-energy bands.

Introduce:

```text
residual.detstoch.min_power_db
```

Recommended behavior:

If

\[
P_b[k] < P_{\min},
\]

then:

- exclude band `b` from the global weighted ratio;
- do not classify silence as stochastic;
- mark the local diagnostic as inactive/invalid internally.

Quiet bands must not dominate the aggregate statistic.

## 9. Optional autocorrelation metric

A later or optional implementation may add:

\[
C_b[k]
=
\max_{\tau \in \mathcal T}
|\rho_b(\tau)|.
\]

This is useful for detecting:

- missed periodic components;
- residual ringing;
- quasi-harmonic structure.

If used, combine with prediction gain:

\[
D_b
=
w_p D_{p,b}
+
w_c D_{c,b},
\]

with

\[
w_p+w_c=1.
\]

For version 1, prediction gain alone is sufficient.

## 10. Global deterministic/stochastic ratio

For active bands:

\[
D[k]
=
\frac{
\sum_b P_b[k]D_b[k]
}{
\sum_b P_b[k]
}.
\]

Then

\[
S[k]=1-D[k].
\]

This energy weighting is essential.

Without weighting, very quiet bands could dominate the statistic.

## 11. Whole-file summary

Compute at least an energy-weighted mean deterministic ratio:

\[
\bar D
=
\frac{
\sum_k E[k]D[k]
}{
\sum_k E[k]
}
\]

where

\[
E[k]=\sum_b P_b[k].
\]

Then

\[
\bar S = 1-\bar D.
\]

Recommended additional summary statistics:

- mean deterministic ratio;
- median deterministic ratio;
- 90th percentile deterministic ratio;
- maximum local deterministic ratio;
- fraction of active frames above a configurable deterministic threshold.

Example:

```text
Residual deterministic/stochastic analysis:
  deterministic mean:      0.18
  stochastic mean:         0.82
  deterministic median:    0.11
  deterministic p90:       0.37
  max deterministic:       0.74
```

## 12. Configuration schema

Suggested TOML extension:

```toml
[residual.detstoch]
enabled = true

window_ms = 20.0
hop_ms = 5.0

lpc_order = 6

pred_gain_min_db = 0.0
pred_gain_max_db = 12.0

min_power_db = -80.0
```

Optional future fields:

```toml
use_autocorrelation = false
autocorr_min_lag_ms = 0.5
autocorr_max_lag_ms = 20.0

prediction_weight = 1.0
autocorr_weight = 0.0
```

## 13. CLI additions

Suggested CLI:

```text
--residual-detstoch
--no-residual-detstoch

--residual-detstoch-window-ms <MS>
--residual-detstoch-hop-ms <MS>

--residual-detstoch-lpc-order <N>

--residual-detstoch-pred-gain-min-db <DB>
--residual-detstoch-pred-gain-max-db <DB>

--residual-detstoch-min-power-db <DB>
```

CLI precedence follows existing configuration rules:

```text
defaults < settings file < CLI
```

## 14. Rust configuration model

Suggested:

```rust
#[derive(Debug, Clone)]
pub struct ResidualDetStochConfig {
    pub enabled: bool,

    pub window_samples: usize,
    pub hop_samples: usize,

    pub lpc_order: usize,

    pub pred_gain_min_db: f64,
    pub pred_gain_max_db: f64,

    pub min_power_linear: f64,
}
```

Validation:

```text
window_samples > lpc_order
hop_samples >= 1
lpc_order >= 1
pred_gain_max_db > pred_gain_min_db
min_power_linear >= 0
```

## 15. Data model

Add optional diagnostics to the residual book or analysis-report section.

Recommended:

```rust
pub struct ResidualBook {
    ...
    pub detstoch: Option<ResidualDetStochBook>,
}
```

Suggested:

```rust
pub struct ResidualDetStochBook {
    pub version: u32,

    pub window_samples: u32,
    pub hop_samples: u32,
    pub lpc_order: u32,

    pub pred_gain_min_db: f32,
    pub pred_gain_max_db: f32,

    pub frames: Vec<ResidualDetStochFrame>,

    pub summary: ResidualDetStochSummary,
}
```

Frame:

```rust
pub struct ResidualDetStochFrame {
    pub deterministic_ratio: f32,
    pub stochastic_ratio: f32,
}
```

The first implementation does not need to serialize all per-band deterministic ratios unless useful for debugging.

## 16. Relationship to residual-book frames

The deterministic/stochastic analysis may use a different hop interval from the residual power-book update interval.

Do not require:

```text
detstoch hop == residual update interval
```

The two analyses have different purposes.

Residual power book:

- optimized for later stochastic synthesis;
- possibly `1 ms` update.

Deterministic/stochastic diagnostic:

- needs a larger estimation window;
- may use `5 ms` hop.

Store its own timing metadata.

## 17. Reuse of ERB-bank output

Do not filter the residue twice if avoidable.

Preferred architecture:

```text
final residue
   |
   v
ERB analysis bank
   |
   +--> power detector --> ResidualBook
   |
   +--> frame buffers --> LPC/predictability analysis
                         |
                         v
                  DetStochBook
```

For band-major offline processing, each band can:

1. filter the complete residual;
2. write residual-power trajectory;
3. calculate local LPC prediction gain;
4. accumulate weighted deterministic statistics.

## 18. Per-band algorithm

For each band `b`:

```text
filter residue -> y_b[n]

for every diagnostic frame k:
    extract y_b[start .. start+window]

    remove mean

    compute frame power

    if below threshold:
        mark inactive
        continue

    compute autocorrelation R[0..p]

    solve LPC

    calculate prediction error variance

    Gp =
        10 * log10(signal_variance / error_variance)

    D =
        clamp(
            (Gp - Gmin) / (Gmax - Gmin),
            0,
            1
        )

    accumulate:
        weighted_D[k] += power * D
        total_power[k] += power
```

After all bands:

```text
if total_power[k] > 0:
    D_global[k] = weighted_D[k] / total_power[k]
    S_global[k] = 1 - D_global[k]
else:
    frame inactive
```

## 19. Numerical safeguards

Handle degenerate frames explicitly.

If:

```text
R[0] <= epsilon
```

mark inactive.

If Levinson-Durbin yields invalid or unstable coefficients:

- do not panic;
- mark the band/frame invalid;
- exclude it from the weighted aggregate;
- increment a diagnostic counter.

Clamp:

```text
error_variance >= epsilon
```

before logarithms.

Clamp final ratio:

```text
0 <= D <= 1
```

## 20. Interpretation

Document the ratio as:

> An energy-weighted predictability index for the residue, mapped to the interval 0..1.

Do not describe it as:

```text
probability that the signal is deterministic
```

Interpretation:

```text
D near 0:
    residue is weakly predictable / noise-like

D near 1:
    residue contains strong predictable structure
```

The corresponding stochastic ratio is:

```text
S = 1 - D
```

## 21. Potential future MP stopping criterion

Do not use this metric to stop MP in the first implementation.

Future possibility:

Stop pursuit when both residual energy and deterministic ratio have fallen sufficiently:

\[
\frac{\|r_n\|^2}{\|x\|^2} < \epsilon_E
\]

and

\[
\bar D_n < \epsilon_D.
\]

This would help avoid extracting increasingly noise-like atoms.

Keep the implementation modular so the statistic can later be evaluated at intermediate pursuit iterations.

## 22. Performance

For approximately:

```text
48 ERB bands
LPC order 6
20 ms windows
5 ms hops
```

the added work is mainly:

- short autocorrelations;
- tiny Levinson-Durbin solves.

This should be much cheaper than the FOF atom search and likely cheaper than the ERB filtering itself.

Avoid FFT-based LPC for the initial low-order implementation.

## 23. Parallelism

Band-major processing is naturally parallel.

Each worker may compute:

```text
band signal
power frames
prediction gain frames
```

for independent ERB bands.

Global ratio accumulation should avoid locks in the inner loop.

Preferred approaches:

- per-thread frame accumulators followed by reduction;
- one per-band result matrix followed by final reduction;
- deterministic ordered reduction if reproducibility requires it.

## 24. Tests

### White noise

Expected:

- low prediction gain;
- low deterministic ratio;
- high stochastic ratio.

### Pure sine

Expected:

- high prediction gain;
- high deterministic ratio.

### Decaying sinusoid / FOF-like resonance

Expected:

- high deterministic ratio while active;
- low weight after decay.

### Colored noise

Expected:

- some predictability from coloration;
- substantially less deterministic than a coherent sinusoid.

### Noise burst

Expected:

- high stochastic ratio while active;
- inactive outside burst.

### Sine + noise mixture

Sweep deterministic-to-stochastic energy ratio.

Expected:

- deterministic ratio increases monotonically as sine energy increases.

This is a key acceptance test.

### Low-energy band

Verify frames below threshold are excluded.

### Serialization

Round-trip timing, configuration, frame ratios, and summary values.

## 25. Recommended first implementation

Implement only:

1. reuse of ERB-band residual signals;
2. low-order autocorrelation LPC;
3. prediction gain;
4. configurable prediction-gain mapping;
5. energy-weighted global deterministic ratio;
6. stochastic ratio as `1-D`;
7. summary statistics.

Do not initially add:

- spectral flatness;
- phase coherence;
- autocorrelation peak metric;
- entropy measures;
- machine-learning classification.

## 26. Acceptance criteria

The feature is complete when:

1. deterministic/stochastic analysis is independently configurable;
2. it reuses the residual ERB analysis where practical;
3. each active diagnostic frame produces a ratio in `[0,1]`;
4. quiet bands do not dominate the statistic;
5. white-noise fixtures score substantially more stochastic than tonal fixtures;
6. sine/noise mixtures produce monotonic deterministic-ratio behavior;
7. no extra ERB filtering pass is required in the normal implementation;
8. analysis remains deterministic and serializable;
9. the metric is stored as a diagnostic and does not affect MP selection;
10. CLI summary reports whole-file deterministic and stochastic ratios.

## 27. Suggested output example

```text
Residual deterministic/stochastic analysis:
  method:                 ERB LPC prediction gain
  ERB bands:              48
  LPC order:              6
  window:                 20.0 ms
  hop:                    5.0 ms

  deterministic ratio:    0.17
  stochastic ratio:       0.83

  deterministic median:   0.10
  deterministic p90:      0.36
  max deterministic:      0.71
```

## 28. Summary

The first deterministic/stochastic estimator should be:

\[
\boxed{
\text{ERB residual bands}
\rightarrow
\text{local low-order LPC}
\rightarrow
\text{prediction gain}
\rightarrow
\text{normalized per-band predictability}
\rightarrow
\text{energy-weighted deterministic ratio}
}
\]

with

\[
\boxed{S = 1-D}.
\]

This gives `rmp` a cheap, interpretable diagnostic for determining whether the final FOF residue has become sufficiently noise-like for stochastic reconstruction.
