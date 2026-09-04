# FOF Matching Pursuit / High-Resolution Matching Pursuit

## Implementation Specification for Analysis and Resynthesis

**Status:** design specification for implementation  
**Purpose:** provide enough mathematical, algorithmic, and software detail for an AI coding agent to implement a FOF-based Matching Pursuit (MP) and High-Resolution Matching Pursuit (HRMP) analyzer/resynthesizer.  
**Primary historical references:** Gribonval et al. (HRMP), LastWave 2.0.4, and MPTK.

---

## 1. Goal

Implement an analysis system that decomposes an audio signal into a sparse sum of Rodet-style Formant-Wave-Function (FOF) atoms plus a residual:

\[
x[n] \approx \sum\_{k=0}^{K-1} g_k[n] + r_K[n].
\]

Each extracted FOF should have the parameters

\[
(t_0, f, \alpha, \beta, A, \phi),
\]

while the release/fade parameter \(\rho\) is **fixed globally at initialization** and is not estimated per atom.

The analyzer must support:

1. ordinary Matching Pursuit;
2. High-Resolution Matching Pursuit in the sense of Gribonval et al.;
3. efficient coarse candidate discovery using FFT/STFT-like correlations;
4. local continuous or quasi-continuous refinement of \(f\), \(\alpha\), and \(\beta\);
5. extraction of amplitude and phase without a phase dictionary dimension;
6. resynthesis with the same FOF envelope used by the analyzer;
7. eventual handoff of a low-level stochastic residual to a separate residual/noise model.

The core design principle is:

> **Do not construct a dense dictionary over \((t_0,f,\alpha,\beta,\phi,A)\).**

Use a cheap structured search to find promising regions, then refine the nonlinear envelope parameters locally.

---

## 2. Fixed design decisions

The following decisions are part of this specification.

### 2.1 Per-atom parameters

Estimate:

- \(t_0\): FOF start time, stored as an integer sample index in the first implementation;
- \(f\): carrier frequency in Hz;
- \(\alpha\): exponential decay coefficient in s\(^{-1}\);
- \(\beta\): attack/skirt angular coefficient in rad/s;
- \(A\): synthesis amplitude;
- \(\phi\): carrier phase in radians.

### 2.2 Global parameter

- \(\rho=\rho_0\) is fixed when the analyzer is initialized.
- The final fade is a **linear amplitude drop to zero**.
- The release is deliberately excluded from \(\alpha\)/\(\beta\) estimation because it occurs at a low level and contains little useful information for those parameters.

### 2.3 No phase search

Do not enumerate phase in the dictionary. Obtain phase from quadrature/complex correlation.

### 2.4 Coarse-to-fine envelope search

Use a small multiscale/prototype FOF bank for candidate discovery, inspired by LastWave/MPTK. After a candidate is found, estimate \(\alpha\) and \(\beta\) independently by local fitting.

### 2.5 HRMP is a local-support constraint

HRMP does not replace the fast correlation engine. It tests whether a proposed large atom is supported locally throughout its claimed support and limits/rejects the atom accordingly.

---

## 3. FOF signal model

Let

\[
\tau = \frac{n-t_0}{F_s},
\]

where \(F_s\) is the sample rate.

The real FOF is

\[
g[n] = A\,e(\tau;\alpha,\beta,\rho_0)
\cos(2\pi f\tau+\phi).
\]

### 3.1 Attack and exponential decay

For the part relevant to parameter estimation, use

\[
e_0(\tau;\alpha,\beta)=
\begin{cases}
\frac{1-\cos(\beta\tau)}{2}\,e^{-\alpha\tau},
&0\le \tau < T_a,\\[2mm]
e^{-\alpha\tau},
&\tau\ge T_a,
\end{cases}
\]

with

\[
T_a=\frac{\pi}{\beta}.
\]

The attack therefore reaches its full multiplicative factor at \(\beta\tau=\pi\).

### 3.2 Linear release

The release policy is global configuration. The analyzer and synthesizer **must call the same envelope implementation**.

A suitable reference convention is:

1. choose a fixed release level \(a_r\), normally specified in dB;
2. let \(t_r\) be the time at which the exponential decay reaches \(a_r\):

   \[
   t_r=-\frac{\ln a_r}{\alpha};
   \]

3. over the globally fixed release duration \(T\_\rho\) represented by \(\rho_0\), replace the exponential tail by a straight line from \(a_r\) to zero:

   \[
   e(\tau)=a*r\left(1-\frac{\tau-t_r}{T*\rho}\right),
   \qquad t*r\le\tau<t_r+T*\rho;
   \]

4. set the envelope to zero afterward.

If an existing FOF synthesizer uses a different interpretation of \(\rho_0\), that implementation is the source of truth. The analysis code must share it rather than duplicate it.

### 3.3 Peak convention

The raw envelope should have a well-defined synthesis amplitude convention. Prefer:

\[
\max_n e[n] = 1
\]

or preserve the existing synthesizer's convention and document it explicitly.

Do **not** confuse synthesis amplitude with an L2-normalized MP coefficient.

---

## 4. Exact amplitude and phase extraction from a real residual

A complex atom makes phase extraction easy, but converting the complete real signal to an analytic signal is not mathematically exact for very short/broadband FOFs. The reference implementation should therefore support an exact real-signal quadrature projection.

For a fixed candidate \(\theta=(t_0,f,\alpha,\beta)\), define

\[
h_c[n]=e[n]\cos(\omega n),
\qquad
h_s[n]=e[n]\sin(\omega n),
\]

where \(\omega=2\pi f/F_s\) and the phase reference uses **local atom time** \(n=0\) at \(t_0\).

Model the residual segment as

\[
r[n]\approx a h_c[n] + b h_s[n].
\]

Define

\[
G=
\begin{bmatrix}
\langle h_c,h_c\rangle & \langle h_c,h_s\rangle\\
\langle h_c,h_s\rangle & \langle h_s,h_s\rangle
\end{bmatrix},
\qquad
p=
\begin{bmatrix}
\langle r,h_c\rangle\\
\langle r,h_s\rangle
\end{bmatrix}.
\]

Then

\[
\begin{bmatrix}a\\b\end{bmatrix}=G^{-1}p.
\]

For the synthesis convention

\[
A e[n]\cos(\omega n+\phi),
\]

we have

\[
a=A\cos\phi,
\qquad
b=-A\sin\phi,
\]

therefore

\[
A=\sqrt{a^2+b^2},
\qquad
\phi=\operatorname{atan2}(-b,a).
\]

The energy removed by the orthogonal projection onto this two-dimensional quadrature subspace is

\[
E\_{\mathrm{capt}}=p^T G^{-1}p.
\]

Use this quantity as the exact MP score for a fixed envelope/time/frequency candidate.

### 4.1 Efficient Gram matrix from one precomputed spectrum

The quadrature Gram matrix does not require direct dot products at every frequency.

Let

\[
E=\sum_n e[n]^2
\]

and

\[
H_2(\omega)=\sum_n e[n]^2 e^{-j2\omega n}.
\]

Then

\[
G\_{cc}=\frac12(E+\Re H_2),
\]

\[
G\_{ss}=\frac12(E-\Re H_2),
\]

\[
G\_{cs}=-\frac12\Im H_2.
\]

If the complex correlation is

\[
C(\omega)=\sum_n r[n]e[n]e^{-j\omega n},
\]

then

\[
p_c=\Re C,
\qquad
p_s=-\Im C.
\]

Thus one FFT correlation supplies the two data projections, while \(G\) can be precomputed for every template/frequency bin.

For long atoms sufficiently far from DC and Nyquist, \(G\approx(E/2)I\). Do not rely on that approximation in the reference implementation.

### 4.2 Degenerate frequencies

At DC and Nyquist the sine/cosine quadratures become degenerate. Detect a small Gram determinant and use a one-dimensional projection or exclude such frequencies from the normal FOF search.

---

## 5. Ordinary Matching Pursuit

Initialize

\[
r_0[n]=x[n].
\]

At iteration \(k\):

1. search a coarse structured dictionary for promising candidates;
2. refine each promising candidate locally;
3. select the candidate with the largest captured energy;
4. synthesize the selected atom using its fitted \(A,\phi\);
5. subtract it from the residual:

   \[
   r\_{k+1}[n]=r_k[n]-g_k[n];
   \]

6. append the atom parameters to the decomposition book;
7. stop when a configured stopping criterion is met.

The decomposition book stores synthesis parameters, not merely normalized dictionary coefficients.

---

## 6. Coarse candidate discovery

### 6.1 Historical observation

LastWave and MPTK avoid an independent \((\alpha,\beta)\) grid. Their FOF window is essentially controlled by scale/window length.

In LastWave, for a window length \(N\) samples,

\[
\alpha_0(N)=\frac{F_s\ln D}{N},
\]

with legacy \(D=10^5\), and

\[
\beta_0(N)=\frac{4\pi F_s}{N},
\]

which gives an attack duration of approximately \(N/4\) samples.

MPTK carries essentially the same fixed-shape, scale-controlled FOF window.

### 6.2 Recommended use of the historical design

Use scale-coupled FOFs only as **seed templates**.

For example, create templates for window lengths

\[
N\in\{N*0,2N_0,4N_0,\ldots,N*{\max}\}.
\]

Each template supplies initial

\[
(\alpha_0,\beta_0).
\]

For each template:

1. run an FFT/STFT-like correlation over time and frequency;
2. compute the exact quadrature projection score where practical;
3. retain only local time-frequency maxima;
4. keep a global top-\(K\) candidate list.

Do not store a full four-dimensional correlation tensor.

### 6.3 Time grid

The coarse search may use a hop \(H>1\). After a candidate is selected for refinement, search all integer start samples in a small neighborhood of the coarse \(t_0\), e.g. within one coarse hop.

The first implementation should store \(t_0\) as an integer sample. Sub-sample onset estimation can be added later if the synthesizer can represent it.

### 6.4 Frequency grid

Use FFT bins for coarse discovery. Refine frequency continuously or by local interpolation afterward.

---

## 7. Efficient estimation of \(\alpha\) and \(\beta\)

The key simplification is that \(\alpha\) and \(\beta\) affect different portions of the FOF strongly:

- \(\alpha\) is visible throughout most of the attack and exponential tail;
- \(\beta\) is visible primarily in the attack;
- the fixed low-level linear release should be ignored during estimation.

The implementation should provide two estimators: a robust reference path and an optional faster complex-baseband path.

---

## 8. Robust reference estimator: variable projection on the real residual

This is the preferred correctness path.

For every trial \((\alpha,\beta)\):

1. generate the raw envelope over a local fit region;
2. construct the two quadratures;
3. solve the 2x2 linear least-squares problem for \((a,b)\);
4. evaluate

   \[
   E\_{\mathrm{capt}}(\alpha,\beta)=p^TG^{-1}p;
   \]

5. maximize this score over only the nonlinear parameters.

Amplitude and phase have been eliminated analytically; this is a variable-projection method.

### 8.1 Fit region

Do not include the low-level release. Restrict the fit to samples for which the predicted or observed envelope is above a configurable fitting floor.

A useful policy is:

- include the complete attack;
- include the exponential decay while it is comfortably above the configured release level;
- exclude the final linear release and samples dominated by the residual noise floor.

### 8.2 Coordinate-refinement strategy

A simple robust implementation is:

1. initialize \((\alpha,\beta)=(\alpha_0,\beta_0)\) from the coarse template;
2. refine \(f\) locally;
3. optimize \(\alpha\) in one dimension with \(\beta\) fixed;
4. optimize \(\beta\) in one dimension with \(\alpha\) fixed;
5. refine integer \(t_0\) locally;
6. repeat steps 2-5 for two or three rounds or until the score improvement is negligible.

Use bounded Brent search, safeguarded Newton, or another deterministic bounded 1-D optimizer. Parameterize positive variables in log-space if convenient:

\[
u=\ln\alpha,
\qquad
v=\ln\beta.
\]

A general multidimensional optimizer is not required for version 1.

### 8.3 Useful derivatives

Away from the attack boundary and before release,

\[
\frac{\partial e}{\partial\alpha}=-\tau e.
\]

During the attack,

\[
\frac{\partial e}{\partial\beta}
=\frac{\tau}{2}\sin(\beta\tau)e^{-\alpha\tau}.
\]

After the attack and before release,

\[
\frac{\partial e}{\partial\beta}=0.
\]

These derivatives make a Gauss-Newton refinement inexpensive if later required.

---

## 9. Optional fast estimator from a complex baseband

If a reliable analytic/baseband representation is available for a local candidate, \(\alpha\) and a residual frequency error can be estimated almost simultaneously.

After demodulation, suppose

\[
z[n]\approx c\,e^{-\alpha n/F_s}e^{j\Delta\omega n}
\]

in the exponential-tail region.

Fit the first-order complex relation

\[
z[n+1]\approx qz[n]
\]

by weighted least squares:

\[
\hat q=
\frac{\sum_n w_n z[n+1]z[n]^\*}
{\sum_n w_n|z[n]|^2}.
\]

Then

\[
\hat\alpha=-F_s\ln|\hat q|
\]

and

\[
\Delta\hat f=\frac{F_s}{2\pi}\arg\hat q.
\]

This estimator avoids logarithms of noisy magnitudes.

Use it only on the exponential tail and only above the release/noise floor.

### 9.1 Initial \(\beta\) from the attack

After compensating for the estimated exponential decay, the ideal attack factor is

\[
u(\tau)=\frac{1-\cos(\beta\tau)}{2}.
\]

Its 50% point obeys

\[
\beta t\_{50}=\frac{\pi}{2},
\]

hence

\[
\beta*0\approx\frac{\pi}{2t*{50}}.
\]

Use this only as an initializer. Final \(\beta\) should be obtained from a one-dimensional profile fit or variable-projection score.

---

## 10. High-Resolution Matching Pursuit

### 10.1 Purpose

Ordinary MP can select a long atom because its **global** correlation is large even if the residual does not support the atom over its whole temporal extent. This can bridge separated events and create pre-echo or artificial energy in gaps.

HRMP adds a local-support test.

### 10.2 Historical LastWave behavior

LastWave contains an explicit high-resolution implementation. For a proposed large atom it:

1. finds smaller same-frequency sub-atoms whose supports lie inside the large atom;
2. obtains the residual coefficient/phase of each sub-atom;
3. computes the inner product between the large atom and each sub-atom;
4. rejects the large atom if a sub-atom has zero support or an incompatible sign/phase;
5. otherwise clamps the large atom's squared coefficient to the minimum local bound.

In simplified notation, if the main atom initially has energy coefficient \(C^2\), LastWave applies bounds of the form

\[
C^2 \leftarrow
\min\left(C^2,
\frac{C_i^2}{|\langle g,g_i\rangle|^2}
\right).
\]

This is the concrete implementation counterpart of the HRMP criterion described by Gribonval et al.

### 10.3 Complex/generalized form

For a candidate main atom \(g\) and local probe/sub-atom \(p_i\), define

\[
d_i=\langle r,p_i\rangle
\]

and

\[
h_i=\langle g,p_i\rangle.
\]

When \(h_i\neq0\), the local implied coefficient of the main atom is

\[
q_i=\frac{d_i}{h_i}.
\]

If the candidate is genuinely present throughout its support, the \(q_i\) should have approximately the same phase and comparable magnitude.

Let the main fitted phasor be

\[
c*{\mathrm{main}}=A*{\mathrm{main}}e^{j\phi\_{\mathrm{main}}}.
\]

A strict generalized HRMP magnitude is

\[
A*{\mathrm{HR}}=
\min\left(A*{\mathrm{main}},\min_i|q_i|\right).
\]

Reject the candidate if local phases are inconsistent. Otherwise use

\[
c*{\mathrm{HR}}=
A*{\mathrm{HR}}e^{j\phi\_{\mathrm{main}}}.
\]

The HRMP candidate score is the energy captured using this clamped amplitude, not the unconstrained ordinary-MP amplitude.

### 10.4 Phase consistency

For each valid probe compute

\[
\Delta\phi*i=
\arg(q_i c*{\mathrm{main}}^\*).
\]

Strict mode should require

\[
|\Delta\phi*i|<\phi*{\max}
\]

for all informative probes.

For a real-atom reproduction of historical HRMP, the phase rule reduces to the original sign-consistency test.

### 10.5 Choice of sub-atoms/probes

Provide two modes.

#### Mode A: `legacy_scaled_fof`

Use smaller FOF atoms at the same carrier frequency, distributed over the support of the large atom. A depth of 1 corresponds naturally to a scale near half the main scale, following the LastWave design.

This mode is useful for:

- reproducing the historical algorithm;
- coarse multiscale FOF dictionaries;
- regression tests against LastWave behavior.

#### Mode B: `localized_candidate`

For a refined FOF with independent \(\alpha,\beta\), form probes by multiplying the actual candidate envelope by short localization windows/masks:

\[
p_i[n] \propto g[n]m_i[n].
\]

Distribute the masks over the useful portion of the FOF support.

This tests local support of **the exact refined candidate** rather than changing \(\alpha\) and \(\beta\) merely to create a smaller FOF.

This is the recommended extension for the final flexible FOF model.

### 10.6 Strict versus robust HRMP

Implement strict HRMP first:

\[
A\_{\mathrm{HR}}=\min_i |q_i|.
\]

Optionally add a robust mode later using a low quantile instead of the absolute minimum. Do not call the robust mode identical to the original HRMP; expose it as a separate policy.

---

## 11. Recommended complete extraction pipeline

For iteration \(k\):

### Step 1 - update/search coarse correlations

For every seed FOF template:

1. correlate the current residual using FFT/STFT machinery;
2. calculate a candidate score from the complex correlation and precomputed quadrature Gram data;
3. detect local maxima in time-frequency;
4. retain only strong maxima.

### Step 2 - form top candidates

Merge maxima across scales/templates and keep the best \(K\_{\mathrm{cand}}\) candidates.

Each candidate contains at least

```text
t0_coarse
f_coarse
alpha_seed
beta_seed
coarse_score
template_id
```

### Step 3 - local parameter refinement

For each retained candidate:

1. refine \(t_0\) to the best integer sample near the coarse hop;
2. refine frequency;
3. refine \(\alpha\);
4. refine \(\beta\);
5. solve the exact quadrature projection for \(A,\phi\);
6. compute the ordinary-MP captured energy.

### Step 4 - HRMP validation

For each refined candidate:

1. construct its HRMP sub-atoms/probes;
2. compute local implied coefficients;
3. test phase consistency;
4. compute the HRMP amplitude clamp;
5. compute the HRMP captured-energy score.

### Step 5 - select atom

Choose the candidate with the largest acceptable score.

For MP mode use the ordinary score.  
For HRMP mode use the HR-constrained score.

### Step 6 - synthesize and subtract

Generate the full FOF using the shared synthesis envelope, including the fixed linear release:

\[
g_k[n]=A_k e[n]\cos(2\pi f_k\tau+\phi_k).
\]

Update

\[
r\_{k+1}=r_k-g_k.
\]

### Step 7 - store atom

Store at minimum:

```text
t0_samples
frequency_hz
alpha_per_second
beta_rad_per_second
amplitude
phase_rad
rho_global_id_or_value
ordinary_score
hr_score
hr_accepted
source_template_id
```

### Step 8 - stopping test

Stop when any configured condition is met, for example:

- maximum number of atoms;
- maximum wall-clock/CPU budget;
- residual energy ratio below threshold;
- best candidate score below threshold;
- best candidate SNR below threshold;
- no candidate passes HRMP;
- residual is judged sufficiently stochastic for the residual-noise model.

---

## 12. Reference pseudocode

```text
function decompose(x, config):
    residual = copy(x)
    book = []

    precompute_seed_templates(config)
    precompute_template_gram_data(config)

    for iteration in 0 .. config.max_atoms:
        maxima = coarse_search_local_maxima(residual)
        candidates = global_top_k(maxima, config.candidate_count)

        best = null

        for c in candidates:
            c.t0 = refine_integer_start(residual, c)
            c.f  = refine_frequency(residual, c)
            c.alpha = refine_alpha(residual, c)
            c.beta  = refine_beta(residual, c)

            fit = fit_quadratures_exact(residual, c)
            c.amplitude = fit.amplitude
            c.phase = fit.phase
            c.mp_score = fit.captured_energy

            if config.mode == HRMP:
                hr = evaluate_hrmp(residual, c, fit, config)
                if not hr.accepted:
                    continue
                c.amplitude = hr.clamped_amplitude
                c.hr_score = hr.captured_energy
                score = c.hr_score
            else:
                score = c.mp_score

            if best == null or score > best.score:
                best = c
                best.score = score

        if best == null:
            break

        if stop_before_commit(best, residual, config):
            break

        atom = synthesize_fof(best, config.rho0, config.release_policy)
        residual -= atom
        book.push(best)

        update_search_state_after_subtraction(best, residual)

    return {book, residual}
```

---

## 13. Candidate search implementation

### 13.1 Correctness-first version

The first working version may recompute the complete coarse correlation bank after every subtraction. This is slow but easy to validate.

### 13.2 Fast version

After the reference implementation is correct, make the search incremental.

A selected atom only changes correlation frames whose windows overlap its support. Therefore:

1. identify affected frames for each template;
2. recompute only those FFT frames;
3. update local maxima only in the affected time-frequency neighborhood;
4. maintain a heap/priority queue of candidate maxima;
5. attach generation/version counters to heap entries so stale maxima can be discarded lazily.

This follows the general tractability idea of LastWave fast MP and MPTK without requiring their exact code.

### 13.3 Local maxima dictionary

LastWave's `fastmpd` explicitly uses a sub-dictionary of local time-frequency maxima and notes that the fast algorithm is optimized for Gaussian and FOF atoms.

Adopt the same idea:

- keep only local maxima from each correlation surface;
- keep roughly enough maxima to cover several future MP iterations;
- refresh maxima locally after a residual update.

The number of retained maxima is a performance/greediness tradeoff and must be configurable.

---

## 14. Numerical considerations

### 14.1 Precision

A reasonable mixed-precision policy is:

- audio and FFT buffers: `f32` initially;
- energy sums, Gram matrices, parameter refinements, and stopping-energy accounting: `f64`;
- convert only at well-defined boundaries.

Provide an all-`f64` reference mode for tests.

### 14.2 Gram determinant

For

\[
\det G=G*{cc}G*{ss}-G\_{cs}^2,
\]

reject or special-case candidates for which

\[
\det G < \epsilon_G E^2.
\]

### 14.3 Residual energy monotonicity

For ordinary MP with an exact least-squares projection and exact subtraction, residual energy should not increase beyond floating-point tolerance.

HRMP intentionally clamps the amplitude below the ordinary projection, but subtraction should still lower residual energy for an accepted, phase-consistent atom. Assert this in debug/test builds.

### 14.4 Parameter bounds

All local optimizers must use explicit bounds:

```text
frequency_min_hz
frequency_max_hz
alpha_min
alpha_max
beta_min
beta_max
minimum_attack_samples
maximum_atom_samples
```

Never allow an optimizer to wander into nonphysical or numerically meaningless regions.

### 14.5 Release exclusion

Do not let the low-level fixed release dominate fitting. Stop or strongly downweight the fit before the release begins.

### 14.6 Phase reference

Define phase at the FOF onset:

\[
\phi=\text{phase at }\tau=0.
\]

All FFT and refinement code must convert to this same reference. This prevents phase changes merely because \(t_0\) moves.

---

## 15. Suggested data structures

Language-neutral definitions:

```text
FofGlobalConfig
    sample_rate
    rho0
    release_level_db
    release_policy
    frequency_bounds
    alpha_bounds
    beta_bounds
    seed_scales
    coarse_hop_policy
    candidate_count
    hrmp_config
    stop_config

FofParams
    t0_samples
    frequency_hz
    alpha
    beta
    amplitude
    phase_rad

SeedTemplate
    id
    length_samples
    alpha_seed
    beta_seed
    envelope[]
    envelope_energy
    gram_frequency_data[]
    fft_plan_or_kernel_handle

Candidate
    params
    seed_template_id
    coarse_score
    mp_score
    hr_score
    accepted

HrmpConfig
    enabled
    mode                 # legacy_scaled_fof | localized_candidate
    depth
    phase_tolerance_rad
    magnitude_policy     # strict_min initially
    probe_overlap
    minimum_probe_energy

DecompositionResult
    atoms[]
    residual[]
    original_energy
    residual_energy
    iterations
    diagnostics
```

---

## 16. API boundaries

Keep these modules separate.

### `fof_envelope`

Source of truth for attack, decay, and fixed linear release.

Required operations:

```text
generate(params_without_A_phi, global_config, output)
attack_end_samples(beta)
release_start_samples(alpha, global_config)
support_samples(alpha, beta, global_config)
```

### `fof_projection`

Exact quadrature fit and score.

```text
fit_real_quadratures(residual_segment, envelope, frequency) -> ProjectionFit
```

### `fof_search`

Coarse FFT/STFT correlation and local maxima extraction.

### `fof_refine`

Local \(t_0,f,\alpha,\beta\) refinement.

### `hrmp`

Local-support probes and HR amplitude constraint.

### `pursuit`

Owns the residual, decomposition loop, stopping criteria, and book.

### `resynthesis`

Synthesizes the book independently of the analyzer. A decomposition must round-trip through this module.

---

## 17. Test plan

Tests are mandatory before performance optimization.

### 17.1 Envelope unit tests

For many \((\alpha,\beta)\):

- value at onset is zero;
- attack/decay boundary is continuous;
- attack end is approximately \(\pi/\beta\);
- exponential portion has the requested \(\alpha\);
- release is exactly linear to zero;
- support termination is correct;
- analyzer and synthesizer generate bit-identical or tolerance-identical envelopes.

### 17.2 Quadrature projection tests

Generate one synthetic FOF with known \(A,\phi\). Verify recovery of:

- amplitude;
- phase;
- captured energy;
- correct result for non-orthogonal cosine/sine quadratures;
- behavior close to DC and Nyquist;
- Gram data computed from \(H_2\) matches direct dot products.

### 17.3 Parameter recovery tests

Noise-free single FOF:

- seed from a deliberately imperfect coarse template;
- recover \(t_0,f,\alpha,\beta,A,\phi\);
- report absolute and relative errors.

Repeat with controlled noise and interfering FOFs.

### 17.4 MP decomposition tests

Synthetic sums of FOFs:

- non-overlapping events;
- overlapping events;
- close frequencies;
- same frequency, separated in time;
- different \(\alpha\) with similar \(\beta\);
- different \(\beta\) with similar \(\alpha\).

Check reconstruction error and parameter recovery.

### 17.5 HRMP-specific tests

#### Bridging test

Construct two short same-frequency events separated by silence. Include a long candidate capable of covering both.

Expected:

- ordinary MP may prefer the long candidate in a designed adversarial case;
- HRMP must reject or strongly clamp it because middle probes have no support.

#### Genuine long-event test

Construct one true long FOF.

Expected:

- sub-probes have consistent phase;
- HRMP accepts it;
- amplitude is not unnecessarily reduced.

#### Phase-inconsistency test

Construct local same-frequency components with incompatible phases.

Expected:

- strict HRMP rejects the candidate.

### 17.6 Historical regression tests

Where licensing and test data permit, reproduce behavior rather than copy code.

- LastWave/MPTK legacy FOF window shapes for selected scales;
- LastWave high-resolution coefficient-clamping behavior on synthetic real atoms;
- local-maxima candidate search behavior.

### 17.7 Determinism

Given identical input/configuration, the decomposition book must be reproducible. Specify deterministic tie-breaking for equal candidate scores.

---

## 18. Performance benchmarks

Measure each stage independently:

```text
coarse FFT search time
local-maxima extraction time
candidate refinement time
HRMP validation time
residual subtraction time
correlation-update time
atoms extracted per second
wall time per second of analyzed audio
peak memory
```

Benchmark both:

1. full recomputation after every atom;
2. incremental affected-region updates.

Do not optimize an unvalidated approximate scorer before the exact reference implementation passes all reconstruction tests.

---

## 19. Implementation phases for an AI coding agent

### Phase 0 - scalar reference

Implement:

- FOF envelope;
- real quadrature projection;
- direct dot-product candidate scoring;
- one-atom synthetic recovery tests.

No FFT optimization yet.

### Phase 1 - ordinary MP

Implement:

- a small explicit seed dictionary;
- residual loop;
- exact subtraction;
- decomposition book;
- reconstruction tests.

### Phase 2 - FFT coarse search

Replace brute-force frequency enumeration with FFT/STFT correlation while preserving the exact quadrature score via precomputed Gram data.

Add local-maxima extraction and top-\(K\) candidates.

### Phase 3 - independent \(\alpha,\beta\) refinement

Add bounded local refinement so the final atom is no longer constrained to the seed template's scale-coupled envelope.

### Phase 4 - HRMP

First implement `legacy_scaled_fof` HRMP and tests corresponding closely to the historical algorithm. Then add `localized_candidate` probes for independently refined FOFs.

### Phase 5 - incremental MP updates

Update only affected frames/candidates after each atom subtraction. Add a generation-tagged maxima heap.

### Phase 6 - optional optimizations

Only after profiling:

- vectorized envelope generation;
- cached local envelopes/derivatives;
- complex-baseband fast \(\alpha\)/frequency estimator;
- batched FFTs;
- parallel candidate refinement;
- analytic or semi-analytic inner products where useful.

---

## 20. Historical findings from LastWave and MPTK

These findings motivated the architecture but should not be treated as requirements to copy the old code.

### 20.1 LastWave FOF window

Archive path:

```text
LastWave_2_0_4/package_stft/src/stft_window.c
```

It defines a FOF window with

```text
decayFoF = 1e5
betaFoF  = pi / 0.25 = 4*pi
```

and scales both decay and attack with the window size. The window is L2-normalized for dictionary use.

### 20.2 LastWave analytic FOF inner products

Archive paths:

```text
LastWave_2_0_4/package_mp/src/atom_asyminnerprod.c
LastWave_2_0_4/package_mp/src/atom_innerprod.c
```

LastWave contains `CCAtomAnalyticFoF(...)`, an approximate/analytic FOF-FOF inner-product implementation used for sufficiently large non-chirped FOF atoms.

**Recommendation:** do not port this into version 1. The new design allows independent \(\alpha,\beta\), so the historical scale-coupled formula is not directly the general solution. Use validated numerical/FFT methods first.

### 20.3 LastWave HRMP

Archive path:

```text
LastWave_2_0_4/package_mp/src/mp_highres.c
```

`SetAtomCoeff2HighRes(...)` shows the actual local-support implementation:

- smaller same-frequency atoms inside the main atom;
- local coefficient and phase lookup;
- main/sub-atom inner products;
- sign/phase rejection;
- minimum coefficient-energy clamping.

This source is especially useful as an executable interpretation of the 1996 HRMP paper.

Some LastWave 2.0.4 plumbing around adding `HighResStft` dictionaries appears incomplete/disabled, so treat the source as research implementation evidence rather than a polished API specification.

### 20.4 LastWave fast MP

Archive path:

```text
LastWave_2_0_4/scripts/mp/MPDalgorithms
```

The `fastmpd` documentation states that Fast Matching Pursuit uses sub-dictionaries of local time-frequency maxima and is optimized for Gaussian and FOF atoms. This motivates retaining only maxima rather than a complete TF grid.

### 20.5 MPTK FOF port

Archive paths:

```text
mptk/src/utils/libdsp_windows/dsp_windows.c
mptk/src/utils/libdsp_windows/dsp_windows.h
mptk/src/tests/test_win.cpp
```

MPTK defines `DSP_FOF_WIN`, again with a fixed scale-controlled FOF shape. Its tests explicitly state that selected FOF windows match LastWave up to numerical accuracy.

FOF windows are also present in example MPTK dictionaries such as:

```text
mptk/reference/dictionary/dic_harmonic.xml
mptk/reference/dictionary/dic_mclt.xml
mptk/reference/dictionary/dic_mdst.xml
```

The important lesson is not to keep the fixed historical shape, but to preserve its computational advantage during candidate discovery and refine the physical FOF parameters afterward.

---

## 21. Wigner-Ville / time-frequency display

Wigner-Ville is not required for the pursuit itself.

If a time-frequency display is desired after decomposition, prefer an atom-based map

\[
E(t,f)=\sum*k E_k\,\widetilde W*{g_k}(t,f)
\]

rather than the raw Wigner-Ville distribution of the complete signal. Summing per-atom distributions avoids the cross-terms that arise in the bilinear Wigner-Ville distribution of a sum.

LastWave contains FOF-specific pseudo-Wigner display code, but comments indicate that the exact FOF Wigner-Ville form was not fully derived there. MPTK also contains incomplete/approximate paths for non-Gaussian windows.

Therefore:

- do not make Wigner-Ville part of atom extraction;
- implement it later as visualization/diagnostics if useful;
- label any separable or Gaussian-frequency approximation as a pseudo-Wigner representation.

---

## 22. Residual model boundary

The FOF pursuit should not be forced to explain stochastic residue with increasingly weak atoms.

Expose a clean stopping/handoff point:

```text
FOF decomposition -> structured atoms + residual
residual -> separate stochastic/filtered-noise analyzer
```

Potential handoff diagnostics can include:

- low best-atom SNR;
- low structured-energy reduction per iteration;
- residual spectral statistics;
- iteration budget.

The residual-noise model is outside the scope of this document.

---

## 23. Licensing note

LastWave and MPTK source code are useful historical and algorithmic references. Before copying source code or formulas expressed directly as code, verify license compatibility with the new project.

For a clean independent implementation:

- implement from the mathematical specification and published algorithms;
- use the old sources for behavioral comparison and regression tests;
- avoid line-by-line copying unless the target project's license and attribution requirements permit it.

---

## 24. References

1. R. Gribonval, E. Bacry, S. Mallat, Ph. Depalle, X. Rodet, **“Sound Signals Decomposition Using a High Resolution Matching Pursuit,”** Proceedings of the International Computer Music Conference (ICMC), 1996, pp. 293-296.
2. R. Gribonval and E. Bacry, **“Harmonic Decomposition of Audio Signals with Matching Pursuit,”** IEEE Transactions on Signal Processing, 2003.
3. LastWave 2.0.4 source tree, especially `package_stft/src/stft_window.c`, `package_mp/src/mp_highres.c`, `package_mp/src/atom_asyminnerprod.c`, and `scripts/mp/MPDalgorithms`.
4. MPTK source tree, especially `src/utils/libdsp_windows/dsp_windows.c`, `src/tests/test_win.cpp`, and the FOF-containing reference dictionaries.
5. X. Rodet, work on Formant-Wave-Function (FOF) synthesis, as the synthesis model underlying the atom family.

---

## 25. Compact implementation checklist

An implementation is not complete until all of the following are true:

- [ ] one shared FOF envelope implementation is used for analysis and resynthesis;
- [ ] \(\rho\) is fixed globally and the final release is linear;
- [ ] amplitude and phase are obtained analytically from quadratures, not searched;
- [ ] the exact 2x2 quadrature Gram correction is implemented;
- [ ] coarse candidate discovery is FFT/STFT-based;
- [ ] only local TF maxima are promoted to expensive refinement;
- [ ] \(\alpha\) and \(\beta\) are refined independently of the coarse scale template;
- [ ] release samples are excluded/downweighted during \(\alpha,\beta\) fitting;
- [ ] MP mode works and reconstructs synthetic FOF mixtures;
- [ ] HRMP implements local-support coefficient clamping and phase consistency;
- [ ] a bridging-event test demonstrates the difference between MP and HRMP;
- [ ] residual subtraction uses the same physical parameters stored in the book;
- [ ] stopping criteria prevent overfitting stochastic residue;
- [ ] deterministic tie-breaking and reproducibility are tested;
- [ ] the scalar/reference implementation remains available for regression testing;
- [ ] performance optimization is measured against the reference implementation, not substituted for it.
