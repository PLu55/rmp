//! High-Resolution Matching Pursuit: a local-support test on a proposed atom.
//!
//! Ordinary MP scores an atom by its *global* correlation with the residual, so a long atom can win
//! by summing evidence from two separated events and claiming the silence between them. The result
//! is pre-echo, and energy invented in gaps. HRMP asks a different question: is this atom supported
//! *everywhere* it claims to be?
//!
//! # The test, in the real sine convention
//!
//! For a probe basis `(u_i, v_i)`, with cross-Gram and data term
//!
//! ```text
//! H_i = [[<u,u_i>, <v,u_i>],        d_i = (<r,u_i>, <r,v_i>)
//!        [<u,v_i>, <v,v_i>]]
//! ```
//!
//! the real analogue of the literature's complex `q_i = d_i / h_i` is **`q_i = H_i^-1 d_i`**, a
//! 2-vector already expressed in the main atom's own `(cos, sin)` coordinates. The probe's own Gram
//! cancels out of the derivation. So `A_i = hypot(q_i)` and `phi_i = atan2(q_i.y, q_i.x)` — the same
//! shape as [`crate::corr::project`], with only the matrix changed.
//!
//! Because `H_i` is accumulated in global time with each basis carried at its own phase reference,
//! the rotation between a probe's onset and the atom's is baked in and `q_i` comes out already
//! rotated. There is no manual phase correction to forget.
//!
//! # Why masking is not just "project onto the masked atom"
//!
//! With `u_i = m_i * u`, the cross-Gram weights the mask **once** and the probe Gram weights it
//! **twice**. For a binary mask `m == m^2` and the two coincide, so the masked projection *is*
//! `q_i`. For a Hann mask they differ, and the naive masked projection overstates `A_i` by
//! `(sum h)/(sum h^2) = 4/3`. A 33% bias would go straight through `min_i A_i` into the score, so
//! both matrices are accumulated and `q_i = H_i^-1 d_i` is used.
//!
//! # Masks are placed by equal energy, not equal time
//!
//! A FOF decays 60 dB across its support: 99% of its energy is in the first 28-48% of it. Four
//! uniform-in-time masks see energy shares of roughly `0.99 / 0.01 / 1e-4 / 1e-6`, so the last is
//! pure noise — and a strict `min_i` over estimators with wildly unequal variance is decided by
//! whichever probe was noisiest, not by which region genuinely lacks support. Splitting at
//! cumulative-energy quantiles is the only placement giving every probe the same noise variance,
//! which is what makes the minimum a test of *shape*.
//!
//! The cost is that equal-energy masks are heavily front-loaded in time, so HRMP detects gaps near
//! the onset sharply and gaps in the decay tail weakly. That is the information content of the
//! signal, not a defect of the placement.
//!
//! # The score needs no new inner products
//!
//! Subtracting `A_HR * g_hat` with the phase held at `phi_main` removes
//! `2*A_HR*<r,g_hat> - A_HR^2*<g_hat,g_hat>`. Since `c = z/|z|`, both terms collapse:
//! `c'd = E_MP/A_main` and `c'Gc = E_MP/A_main^2`, so with `rho = A_HR/A_main`
//!
//! ```text
//! E_HR = E_MP * (2*rho - rho^2)
//! ```
//!
//! Under strict HRMP `A_HR <= A_main` and every `A_i >= 0`, so `rho` is in `[0, 1]` and `E_HR` can
//! never be negative. It could only go so in a robust mode that replaces the minimum with a
//! quantile, which is why that path must clamp to `A_main` too.

use crate::corr::Projection;
use crate::fit::Quad;
use crate::fof::Envelope;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeMode {
    /// Probes are the candidate's own envelope times short masks. Tests the support of the *exact*
    /// refined atom, and needs no second dictionary.
    LocalizedCandidate,
    /// Probes are smaller same-frequency FOFs inside the main support, as in the historical
    /// implementation.
    LegacyScaledFof,
}

/// How the local amplitudes are combined into a bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MagnitudePolicy {
    /// `A_HR = min_i A_i`, the original criterion.
    StrictMin,
}

#[derive(Clone, Copy, Debug)]
pub struct HrmpConfig {
    pub enabled: bool,
    pub mode: ProbeMode,
    /// `2^depth` masks. Depth 1 reproduces the historical "half the main scale".
    pub depth: u32,
    pub phase_tolerance_rad: f32,
    /// Structural floor on a probe's share of the atom's energy.
    pub min_probe_energy: f64,
    /// Bound on each local amplitude's relative standard error. A probe must see at least
    /// `1/epsilon^2` times the local residual noise power in atom energy to get a vote.
    pub noise_epsilon: f64,
    /// A mask spanning fewer carrier periods than this cannot condition its own Gram.
    pub min_mask_periods: f32,
    pub magnitude_policy: MagnitudePolicy,
    pub rho_sq_max: f64,
}

impl Default for HrmpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: ProbeMode::LocalizedCandidate,
            depth: 2,
            phase_tolerance_rad: std::f32::consts::FRAC_PI_4,
            min_probe_energy: 1e-3,
            noise_epsilon: 0.2,
            min_mask_periods: 2.0,
            magnitude_policy: MagnitudePolicy::StrictMin,
            rho_sq_max: 1.0 - 1e-4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The atom is too short for any mask to be conditioned, so HRMP has nothing to say.
    TooShort,
    /// No probe carried enough atom energy to be informative.
    Uninformative,
    /// Every informative probe agreed; the amplitude is unchanged or barely reduced.
    Accepted,
    /// Supported everywhere, but more weakly somewhere than the global fit claimed.
    Clamped,
    /// A probe's phase was inconsistent with the main fit.
    Rejected,
}

#[derive(Clone, Copy, Debug)]
pub struct Verdict {
    pub outcome: Outcome,
    /// The clamped amplitude. Equal to the main fit's when nothing was learned.
    pub amp: f32,
    /// Energy this atom would remove at `amp`, holding the phase at the main fit's.
    pub energy: f64,
    /// Probes that carried a vote.
    pub informative: usize,
    /// Worst phase disagreement seen, radians.
    pub worst_dphi: f32,
}

impl Verdict {
    pub fn accepted(&self) -> bool {
        self.outcome != Outcome::Rejected
    }

    fn pass(amp: f32, energy: f64, outcome: Outcome) -> Self {
        Self { outcome, amp, energy, informative: 0, worst_dphi: 0.0 }
    }
}

/// Where a candidate atom sits, and how much of it is worth testing.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub t0: i64,
    pub f: f32,
    /// Samples from the onset to test — [`crate::fit::fit_end`], so the release is excluded.
    pub fit_len: usize,
}

/// Test whether the residual supports this atom across its whole claimed extent.
///
/// `main` and `fit` are the ordinary-MP quadrature statistics for the same `(env, at)`, so the
/// caller pays for them once.
pub fn evaluate(
    residual: &[f32],
    env: &Envelope,
    at: Placement,
    main: &Quad,
    fit: &Projection,
    cfg: &HrmpConfig,
) -> Verdict {
    let Placement { t0, f, fit_len } = at;
    let e_mp = fit.energy;
    if e_mp <= 0.0 || fit.amp <= 0.0 {
        return Verdict::pass(fit.amp, e_mp, Outcome::Uninformative);
    }

    // A mask must span a couple of carrier periods or `m*u` and `m*v` are nearly parallel and the
    // local amplitude is arbitrary. Declining costs nothing: HRMP exists to stop a *long* atom
    // bridging separated events, and an atom this short cannot bridge anything.
    let periods = fit_len as f32 * f / env.sample_rate;
    let p = (1usize << cfg.depth).min((periods / cfg.min_mask_periods.max(1e-3)) as usize);
    if p < 2 {
        return Verdict::pass(fit.amp, e_mp, Outcome::TooShort);
    }

    let Some(bounds) = energy_quantiles(&env.samples[..fit_len], 2 * p) else {
        return Verdict::pass(fit.amp, e_mp, Outcome::TooShort);
    };

    let omega = std::f64::consts::TAU * f as f64 / env.sample_rate as f64;
    let (s_main, c_main) = (fit.phi as f64).sin_cos();
    // c'Gc for the unit main atom, from the statistics the caller already computed.
    let ggc = c_main * c_main * main.g_uu
        + 2.0 * c_main * s_main * main.g_uv
        + s_main * s_main * main.g_vv;
    let usable = ggc > 0.0;
    if !usable {
        return Verdict::pass(fit.amp, e_mp, Outcome::Uninformative);
    }

    let Some((_, r_start, n_total)) = crate::signal::overlap(residual.len(), fit_len, t0) else {
        return Verdict::pass(fit.amp, e_mp, Outcome::Uninformative);
    };
    let span = Span {
        e_start: if t0 < 0 { (-t0) as usize } else { 0 },
        r_start,
        n: n_total,
    };

    let mut a_hr = fit.amp as f64;
    let mut informative = 0usize;
    let mut worst_dphi = 0.0f32;

    // Mask i spans two quantile cells and starts one cell after its predecessor: 50% overlap, with
    // every mask carrying an equal 1/p share of the atom's energy. Stepping by two cells instead
    // would tile them edge to edge and leave the joins untested.
    for i in 0..(2 * p - 1) {
        let (lo, hi) = (bounds[i], bounds[i + 2]);
        if hi <= lo + 1 {
            continue;
        }
        let Some(pr) = probe(residual, &env.samples, span, lo, hi, omega) else {
            continue;
        };

        // Structural share of the atom's energy inside this probe's subspace.
        let kappa = (c_main * c_main * pr.g_uu
            + 2.0 * c_main * s_main * pr.g_uv
            + s_main * s_main * pr.g_vv)
            / ggc;
        if kappa < cfg.min_probe_energy {
            continue;
        }
        // Noise-relative gate: the local amplitude's relative standard error is
        // sigma / sqrt(kappa * E_MP), so demand that stay under `noise_epsilon`.
        let eps = cfg.noise_epsilon.max(1e-6);
        if kappa * e_mp < pr.noise_power / (eps * eps) {
            continue;
        }
        // A probe whose own basis is degenerate cannot report an amplitude.
        let det_g = pr.g_uu * pr.g_vv - pr.g_uv * pr.g_uv;
        let tr_g = pr.g_uu + pr.g_vv;
        let conditioned = det_g > 0.0 && tr_g > 0.0
            && 1.0 - 4.0 * det_g / (tr_g * tr_g) <= cfg.rho_sq_max;
        if !conditioned {
            continue;
        }

        // q_i = H_i^-1 d_i, in the main atom's coordinates.
        let det_h = pr.h00 * pr.h11 - pr.h01 * pr.h10;
        if det_h == 0.0 || !det_h.is_finite() {
            continue;
        }
        let q_x = (pr.h11 * pr.d0 - pr.h01 * pr.d1) / det_h;
        let q_y = (pr.h00 * pr.d1 - pr.h10 * pr.d0) / det_h;

        informative += 1;

        // Phase test. The explicit `dot > 0` is the historical sign-consistency rule and also makes
        // a degenerate q_i, where atan2 returns an arbitrary angle, fail loudly instead of passing.
        let dot = q_x * c_main + q_y * s_main;
        let cross = q_y * c_main - q_x * s_main;
        let dphi = cross.atan2(dot).abs();
        worst_dphi = worst_dphi.max(dphi as f32);
        if dot <= 0.0 || dphi > cfg.phase_tolerance_rad as f64 {
            return Verdict {
                outcome: Outcome::Rejected,
                amp: 0.0,
                energy: 0.0,
                informative,
                worst_dphi,
            };
        }

        match cfg.magnitude_policy {
            MagnitudePolicy::StrictMin => a_hr = a_hr.min(q_x.hypot(q_y)),
        }
    }

    if informative == 0 {
        // Absence of evidence is not evidence of absence: pass the atom through unchanged.
        return Verdict::pass(fit.amp, e_mp, Outcome::Uninformative);
    }

    let rho = (a_hr / fit.amp as f64).clamp(0.0, 1.0);
    let energy = e_mp * (2.0 * rho - rho * rho);
    Verdict {
        outcome: if rho > 0.999 { Outcome::Accepted } else { Outcome::Clamped },
        amp: (rho * fit.amp as f64) as f32,
        energy,
        informative,
        worst_dphi,
    }
}

/// Statistics for one masked probe.
struct Probe {
    h00: f64,
    h01: f64,
    h10: f64,
    h11: f64,
    g_uu: f64,
    g_uv: f64,
    g_vv: f64,
    d0: f64,
    d1: f64,
    /// Mean residual power under the mask.
    noise_power: f64,
}

/// Where the atom's samples sit against the residual's, from one `signal::overlap` call.
#[derive(Clone, Copy)]
struct Span {
    e_start: usize,
    r_start: usize,
    n: usize,
}

/// Accumulate one Hann-masked probe over envelope indices `[lo, hi)`.
///
/// `H` weights the mask once and `G` weights it twice — the distinction the module docs explain.
fn probe(
    residual: &[f32],
    env: &[f32],
    span: Span,
    lo: usize,
    hi: usize,
    omega: f64,
) -> Option<Probe> {
    let Span { e_start, r_start, n: n_total } = span;
    let span = (hi - lo) as f64;
    let (mut h00, mut h01, mut h11) = (0.0f64, 0.0f64, 0.0f64);
    let (mut g_uu, mut g_uv, mut g_vv) = (0.0f64, 0.0f64, 0.0f64);
    let (mut d0, mut d1) = (0.0f64, 0.0f64);
    let (mut noise, mut count) = (0.0f64, 0usize);

    for (j, &e) in env.iter().enumerate().take(hi).skip(lo) {
        if j < e_start || j - e_start >= n_total {
            continue;
        }
        let e = e as f64;
        let r = residual[r_start + (j - e_start)] as f64;
        // Hann across the mask span, so adjacent 50%-overlapped masks tile the envelope.
        let m = 0.5 * (1.0 - (std::f64::consts::TAU * (j - lo) as f64 / span).cos());

        let (s, c) = (omega * j as f64).sin_cos();
        let (u, v) = (e * s, e * c);
        let (m2, uu, uv, vv) = (m * m, u * u, u * v, v * v);

        h00 += m * uu;
        h01 += m * uv;
        h11 += m * vv;
        g_uu += m2 * uu;
        g_uv += m2 * uv;
        g_vv += m2 * vv;
        d0 += r * m * u;
        d1 += r * m * v;
        noise += r * r;
        count += 1;
    }

    if count == 0 {
        return None;
    }
    Some(Probe {
        // In this mode `u_i = m*u` and `v_i = m*v`, so H is symmetric: <v,u_i> == <u,v_i>.
        h00,
        h01,
        h10: h01,
        h11,
        g_uu,
        g_uv,
        g_vv,
        d0,
        d1,
        noise_power: noise / count as f64,
    })
}

/// Indices splitting `env` into `parts` runs of equal `E^2`.
///
/// Returns `parts + 1` boundaries, strictly increasing, or `None` if the support is too short to
/// divide.
fn energy_quantiles(env: &[f32], parts: usize) -> Option<Vec<usize>> {
    if parts == 0 || env.len() < parts * 2 {
        return None;
    }
    let mut cum = Vec::with_capacity(env.len());
    let mut acc = 0.0f64;
    for &e in env {
        acc += (e as f64) * (e as f64);
        cum.push(acc);
    }
    let total = acc;
    let usable = total > 0.0;
    if !usable {
        return None;
    }

    let mut bounds = Vec::with_capacity(parts + 1);
    bounds.push(0usize);
    for k in 1..parts {
        let want = total * k as f64 / parts as f64;
        let idx = cum.partition_point(|&c| c < want);
        // Keep the sequence strictly increasing even where one sample carries a whole quantile.
        let idx = idx.max(bounds[k - 1] + 1).min(env.len() - (parts - k));
        bounds.push(idx);
    }
    bounds.push(env.len());
    Some(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit;
    use crate::fof::{AtomParams, EnvelopeParams};
    use crate::signal::add_at;

    const SR: f32 = 48_000.0;
    const RHO: f64 = 1.0 - 1e-4;

    fn cfg() -> HrmpConfig {
        HrmpConfig { enabled: true, ..HrmpConfig::default() }
    }

    /// Score `atom`'s shape against `sig` and run HRMP on the result.
    fn verdict(sig: &[f32], env: &Envelope, t0: i64, f: f32, c: &HrmpConfig) -> Verdict {
        let omega = std::f64::consts::TAU * f as f64 / SR as f64;
        let q = fit::accumulate(sig, &env.samples, t0, omega).expect("overlap");
        let p = q.solve(RHO).expect("conditioned");
        let at = Placement { t0, f, fit_len: fit::fit_end(env) };
        evaluate(sig, env, at, &q, &p, c)
    }

    fn env_of(alpha: f32, beta: f32) -> Envelope {
        Envelope::render(EnvelopeParams::new(alpha, beta), SR).unwrap()
    }

    /// A genuine long event: every probe sees the atom, at one phase.
    #[test]
    fn a_real_long_atom_is_accepted_without_being_clamped() {
        let env = env_of(120.0, 0.001);
        let atom = AtomParams { t0: 500, f: 700.0, env: env.params, phi: 0.4, amp: 0.8 };
        let mut sig = vec![0.0f32; 40_000];
        add_at(&mut sig, &atom.render(SR).unwrap(), atom.t0);

        let v = verdict(&sig, &env, atom.t0, atom.f, &cfg());
        assert!(v.accepted(), "{v:?}");
        assert!(v.informative >= 2, "{v:?}");
        assert!(
            v.amp > 0.95 * atom.amp,
            "a supported atom must not be clamped: {v:?} vs amp {}",
            atom.amp
        );
        assert!(v.worst_dphi < 0.2, "{v:?}");
    }

    /// The bridging case HRMP exists for: two short bursts, silence between, and a long candidate
    /// that ordinary MP is happy to stretch across both.
    #[test]
    fn a_long_atom_bridging_a_gap_is_clamped_hard() {
        let f = 700.0;
        let short = EnvelopeParams::new(2000.0, 0.0003);
        // The long candidate has to actually span both events, or it is not bridging anything and
        // the test would pass for the wrong reason. alpha = 80 is the grid's longest, and the
        // second burst goes a fifth of the way into its support, with silence between.
        let long = env_of(80.0, 0.001);
        let t1 = 500i64;
        let t2 = t1 + (long.support_len() / 5) as i64;
        assert!(t2 + 200 < t1 + long.support_len() as i64, "second event must be inside");

        let mut sig = vec![0.0f32; 40_000];
        for t0 in [t1, t2] {
            let a = AtomParams { t0, f, env: short, phi: 0.4, amp: 1.0 };
            add_at(&mut sig, &a.render(SR).unwrap(), t0);
        }

        let mp = {
            let omega = std::f64::consts::TAU * f as f64 / SR as f64;
            fit::accumulate(&sig, &long.samples, t1, omega).unwrap().solve(RHO).unwrap()
        };
        let v = verdict(&sig, &long, t1, f, &cfg());

        // Ordinary MP would take the whole projected amplitude; HRMP must not.
        assert!(v.informative >= 2, "{v:?}");
        assert!(
            v.energy < 0.5 * mp.energy,
            "bridging atom kept {:.3} of its {:.3} ordinary score: {v:?}",
            v.energy,
            mp.energy
        );
    }

    /// Same frequency, incompatible phases: strict HRMP must refuse rather than average them.
    #[test]
    fn phase_inconsistency_is_rejected() {
        let f = 700.0;
        let short = EnvelopeParams::new(2000.0, 0.0003);
        let long = env_of(80.0, 0.001);
        // Both bursts inside the long candidate's support, or there is no conflict to detect.
        let (t1, t2) = (500i64, 500 + (long.support_len() / 6) as i64);
        let mut sig = vec![0.0f32; 40_000];
        add_at(&mut sig, &AtomParams { t0: t1, f, env: short, phi: 0.0, amp: 1.0 }.render(SR).unwrap(), t1);

        // Anti-phase means anti-phase *in the main atom's frame*, not in the second atom's own.
        // Each FOF references its phase to its own onset, so a bare `phi = PI` here would be
        // rotated by omega*(t2-t1) -- 36.5 carrier periods -- and land almost back in phase. This
        // is the correction the module docs say `H_i` applies automatically during the test; the
        // fixture has to apply it by hand.
        let omega = std::f32::consts::TAU * f / SR;
        let phi2 = std::f32::consts::PI + omega * (t2 - t1) as f32;
        add_at(&mut sig, &AtomParams { t0: t2, f, env: short, phi: phi2, amp: 1.0 }.render(SR).unwrap(), t2);

        let v = verdict(&sig, &long, t1, f, &HrmpConfig { depth: 3, ..cfg() });
        assert_eq!(v.outcome, Outcome::Rejected, "{v:?}");
        assert!(!v.accepted(), "{v:?}");
    }

    #[test]
    fn a_short_atom_is_skipped_rather_than_judged() {
        // At 2147 s^-1 and 60 Hz the whole support spans well under four carrier periods.
        let env = env_of(2147.0, 0.0003);
        let atom = AtomParams { t0: 100, f: 60.0, env: env.params, phi: 0.2, amp: 1.0 };
        let mut sig = vec![0.0f32; 8_000];
        add_at(&mut sig, &atom.render(SR).unwrap(), atom.t0);

        let v = verdict(&sig, &env, atom.t0, atom.f, &cfg());
        assert_eq!(v.outcome, Outcome::TooShort, "{v:?}");
        assert_eq!(v.amp, {
            let omega = std::f64::consts::TAU * atom.f as f64 / SR as f64;
            fit::accumulate(&sig, &env.samples, atom.t0, omega).unwrap().solve(RHO).unwrap().amp
        });
    }

    #[test]
    fn silence_under_the_probes_is_uninformative_not_rejected() {
        let env = env_of(120.0, 0.001);
        let sig = vec![0.0f32; 40_000];
        let omega = std::f64::consts::TAU * 700.0 / SR as f64;
        let q = fit::accumulate(&sig, &env.samples, 0, omega).unwrap();
        let p = q.solve(RHO).unwrap();
        let at = Placement { t0: 0, f: 700.0, fit_len: fit::fit_end(&env) };
        let v = evaluate(&sig, &env, at, &q, &p, &cfg());
        assert!(v.accepted(), "{v:?}");
        assert_eq!(v.outcome, Outcome::Uninformative);
    }

    /// The energy law the score relies on, checked against the projection it is derived from.
    #[test]
    fn the_clamped_energy_matches_the_quadrature_form() {
        let env = env_of(300.0, 0.001);
        let atom = AtomParams { t0: 800, f: 1_100.0, env: env.params, phi: -0.9, amp: 0.5 };
        let mut sig = vec![0.0f32; 30_000];
        add_at(&mut sig, &atom.render(SR).unwrap(), atom.t0);

        let omega = std::f64::consts::TAU * atom.f as f64 / SR as f64;
        let q = fit::accumulate(&sig, &env.samples, atom.t0, omega).unwrap();
        let p = q.solve(RHO).unwrap();
        for rho in [0.1f64, 0.5, 0.9, 1.0] {
            let want = q.energy_at(rho * p.amp as f64, p.phi as f64);
            let got = p.energy * (2.0 * rho - rho * rho);
            assert!(
                (got - want).abs() <= 1e-6 * p.energy,
                "rho={rho}: law {got} vs quadrature {want}"
            );
        }
    }

    #[test]
    fn equal_energy_quantiles_are_strictly_increasing_and_balanced() {
        let env = env_of(251.0, 0.001);
        let cut = fit::fit_end(&env);
        let b = energy_quantiles(&env.samples[..cut], 8).unwrap();
        assert_eq!(b.len(), 9);
        assert_eq!(b[0], 0);
        assert_eq!(b[8], cut);
        for w in b.windows(2) {
            assert!(w[1] > w[0], "{b:?}");
        }
        // Each cell carries about an eighth of the energy — that equality is the whole point.
        let total: f64 = env.samples[..cut].iter().map(|&e| (e as f64).powi(2)).sum();
        for w in b.windows(2) {
            let part: f64 = env.samples[w[0]..w[1]].iter().map(|&e| (e as f64).powi(2)).sum();
            assert!(
                (part / total - 0.125).abs() < 0.02,
                "cell {w:?} carries {:.4}",
                part / total
            );
        }
        // Front-loaded in time, which is the documented cost of the placement.
        assert!(b[4] < cut / 4, "median energy should sit early: {b:?}");
    }

    #[test]
    fn quantiles_refuse_a_support_too_short_to_divide() {
        assert!(energy_quantiles(&[1.0, 1.0, 1.0], 8).is_none());
        assert!(energy_quantiles(&[0.0; 64], 4).is_none());
    }
}
