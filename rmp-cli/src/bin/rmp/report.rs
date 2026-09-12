//! Everything `rmp` prints about a run, and nothing else.
//!
//! The pipeline reports facts and formats nothing (see [`rmp_core::pipeline`]), so this is the only
//! place that knows what an analysis *reads like*. It is split out for the same reason the pipeline
//! was: so that adding a second front end cannot quietly change what the first one says.
//!
//! Progress arrives two ways, and the split is not arbitrary. Whatever happens *while* the pursuit
//! is working comes through [`Reporter::event`] and has to be printed as it arrives; everything
//! else is read off the finished [`Analysis`] afterwards, in whatever order reads best. Today that
//! order is the one the single-function version printed in, which is what lets the two be diffed.

use rmp_core::atom::AtomKind;
use rmp_core::book::Book;
use rmp_core::dict::Dictionary;
use rmp_core::mp::MpConfig;
use rmp_core::pipeline::{Analysis, Event, Reporter, unrefinable_blocks};
use rmp_core::residual::book::ResidualBook;
use rmp_core::signal::{Signal, db_fs, peak_of, rms_of};
use std::path::Path;

/// Progress to stderr.
///
/// [`Reporter::cancelled`] is left at its default, so a command-line run is never cut short: there
/// is no interrupt source here to drive it. The machinery exists for a front end that has one —
/// which is why [`Cli::analysis`] still reports [`Analysis::cancelled`] rather than assuming it.
pub struct Cli {
    pub quiet: bool,
    /// The excerpt's sample rate. The window plan is in samples and reads in seconds.
    pub sr: f32,
    /// Needed by the progress lines themselves: which blocks refinement will decline, and what
    /// target the run fell short of.
    pub mp_cfg: MpConfig,
    /// Only to quote back in the over-budget note.
    pub max_memory_mb: usize,
}

impl Reporter for Cli {
    fn event(&mut self, e: Event<'_>) {
        match e {
            Event::Dictionary { dict, elapsed } => {
                let kinds = kinds_of(dict);
                self.say(&format!(
                    "dictionary: {} blocks ({}) in {elapsed:.2?}",
                    dict.blocks.len(),
                    kinds
                        .iter()
                        .map(|(k, n)| format!("{n} {k}"))
                        .collect::<Vec<_>>()
                        .join(", "),
                ));

                // A fact rather than a fault; `pipeline::unrefinable_blocks` documents why it cuts
                // both ways, and the `refined:` line below says what actually happened.
                let stuck = unrefinable_blocks(dict, &self.mp_cfg);
                if let Some(longest) = stuck.iter().max_by_key(|b| b.support_len()) {
                    self.say(&format!(
                        "  note: {} of {} blocks are longer than refine.max_atom_samples ({}), so \
                         their atoms stay on the grid unrefined; longest support {} samples ({})",
                        stuck.len(),
                        dict.blocks.len(),
                        self.mp_cfg.refine.max_atom_samples,
                        longest.support_len(),
                        longest.env.params.describe(),
                    ));
                }
            }

            Event::Windows { plan } => {
                let sr = self.sr;
                self.say(&format!(
                    "windows: {} x {:.2} s core + {:.2} s guard ({:.1} MiB of frame table each, \
                     {:.2} B/sample)",
                    plan.count,
                    plan.core_len as f32 / sr,
                    plan.guard_len as f32 / sr,
                    plan.window_bytes() / (1 << 20) as f64,
                    plan.bytes_per_sample,
                ));
                self.say(
                    "  selection is greedy within a window, not across the clip; target_snr_db, \
                     min_gain and max_atoms apply per window",
                );
                if plan.over_budget {
                    self.say(&format!(
                        "  note: a window cannot be shorter than four guards, so the core was \
                         raised to {:.2} s past what max_memory_mb ({} MiB) or window_seconds \
                         asked for. Shorten the longest atom -- a higher dictionary alpha, or a \
                         lower refine.max_atom_samples -- to window more finely",
                        plan.core_len as f32 / sr,
                        self.max_memory_mb,
                    ));
                }
            }

            Event::Window { index, of, atoms } => {
                if of > 1 {
                    self.say(&format!("  window {index}/{of}: {atoms} atoms so far"));
                }
            }
        }
    }
}

impl Cli {
    pub fn say(&self, m: &str) {
        if !self.quiet {
            eprintln!("{m}");
        }
    }

    /// What was read, and what part of it is being analysed.
    pub fn input(
        &self,
        path: &Path,
        whole_len: usize,
        channels: usize,
        downmixed: bool,
        offset: usize,
        excerpt_len: usize,
    ) {
        let sr = self.sr;
        self.say(&format!(
            "{}: {:.2} s, {} Hz, {} channel(s)",
            path.display(),
            whole_len as f32 / sr,
            sr as u32,
            channels
        ));
        if excerpt_len != whole_len {
            // Everything downstream — atom onsets in the book, the resynthesis, the residual — is
            // relative to this excerpt, not to the file.
            self.say(&format!(
                "  analysing {:.3}-{:.3} s ({} samples from {})",
                offset as f32 / sr,
                (offset + excerpt_len) as f32 / sr,
                excerpt_len,
                offset
            ));
        }
        if downmixed {
            self.say("  downmixed to mono; out-of-phase content between channels partially cancels");
        }
    }

    /// The decomposition itself: what it selected, and what the refresh cost.
    pub fn analysis(&self, a: &Analysis, duration: f32) {
        let (init, pursuit) = (a.timing.init, a.timing.pursuit);
        let book = &a.book;
        self.say(&format!(
            "analysis: {} atoms, {:.1} dB in {pursuit:.2?} (init {init:.2?}, {:.1}x realtime)",
            book.len(),
            book.snr_db(),
            (init + pursuit).as_secs_f32() / duration.max(1e-9)
        ));
        if a.cancelled {
            self.say("  interrupted: the book holds only what had been selected by then");
        }
        if !book.is_empty() && self.mp_cfg.refine.enabled {
            let refined = book.selections.iter().filter(|s| s.refined).count();
            self.say(&format!(
                "  refined: {refined}/{} atoms moved off the grid ({:.0}%)",
                book.len(),
                100.0 * refined as f64 / book.len() as f64
            ));
        }
        // Only worth a line when the dictionary offered a choice.
        let kinds = kinds_of(&a.dict);
        if kinds.len() > 1 && !book.is_empty() {
            self.say(&format!("  kinds: {}", per_kind(book, &kinds).join(", ")));
        }
        if book.is_empty() {
            self.say("  warning: no atoms selected — check the dictionary covers the signal's content");
        } else if !a.cancelled && book.snr_db() < self.mp_cfg.target_snr_db {
            self.say(&format!(
                "  stopped short of the {:.1} dB target (max_atoms = {})",
                self.mp_cfg.target_snr_db, self.mp_cfg.max_atoms
            ));
        }

        let (marked, resolved) = (a.refresh.marked, a.refresh.resolved);
        if marked > 0 {
            self.say(&format!(
                "  refresh: {marked} frames bounded, {resolved} recomputed ({:.1}%)",
                100.0 * resolved as f64 / marked as f64
            ));
            if std::env::var_os("RMP_REFRESH_DETAIL").is_some() {
                for (bi, &(m, r)) in a.refresh.per_block.iter().enumerate() {
                    let b = &a.dict.blocks[bi];
                    self.say(&format!(
                        "    block {bi:>2} {:<30} fft {:>7}: {m:>8} bounded {r:>8} recomputed ({:>5.1}%)  ~{:.0} Msamples",
                        b.env.params.describe(), b.fft_len, 100.0 * r as f64 / m.max(1) as f64,
                        (r * b.fft_len) as f64 / 1e6
                    ));
                }
            }
        }
    }

    /// The level of what the decomposition could not explain.
    ///
    /// Reported relative to the input first, because that is the figure you act on — an absolute
    /// dBFS residual means nothing without knowing how loud the input was. The rms ratio is the
    /// negated SNR by construction; it is restated here so the two absolute levels beside it do not
    /// have to be subtracted by eye. The peak ratio is the one that carries new information: it is
    /// where the decomposition is worst rather than where it is on average.
    pub fn residual_levels(&self, residual: &[f32], signal: &Signal) {
        let (r_rms, r_peak) = (rms_of(residual), peak_of(residual) as f64);
        let (s_rms, s_peak) = (signal.rms(), signal.peak() as f64);
        self.say(&format!(
            "residual: {:+.1} dB rms, {:+.1} dB peak relative to input",
            db_fs(r_rms) - db_fs(s_rms),
            db_fs(r_peak) - db_fs(s_peak),
        ));
        self.say(&format!(
            "  absolute: {:.1} dBFS rms, {:.1} dBFS peak (input {:.1} dBFS rms, {:.1} dBFS peak)",
            db_fs(r_rms),
            db_fs(r_peak),
            db_fs(s_rms),
            db_fs(s_peak),
        ));
    }

    /// The ERB analysis of the residue.
    pub fn residual_analysis(&self, rb: &ResidualBook, elapsed: std::time::Duration, sr: f32) {
        let taus = &rb.bank.power_detector.tau_seconds;
        let (tau_lo, tau_hi) = (
            taus.iter().cloned().fold(f64::INFINITY, f64::min) * 1e3,
            taus.iter().cloned().fold(0.0, f64::max) * 1e3,
        );
        self.say(&format!(
            "residual analysis: {} ERB bands, {:.1} .. {:.1} Hz, order {} gammatone, in {elapsed:.2?}",
            rb.band_count, rb.bank.min_freq_hz, rb.bank.max_freq_hz, rb.bank.filter_order
        ));
        self.say(&format!(
            "  update: {} samples / {:.3} ms -> {} frames ({} values)",
            rb.update_samples,
            rb.update_samples as f64 * 1e3 / sr as f64,
            rb.frame_count,
            rb.power.len()
        ));
        self.say(&format!(
            "  power:  {}, tau {tau_lo:.2} .. {tau_hi:.2} ms",
            rb.bank.power_detector.mode
        ));
        if std::env::var_os("RMP_RESIDUAL_DETAIL").is_some() {
            self.say("    band   center_hz  bandwidth_hz  tau_ms   norm_gain");
            for (b, &tau) in taus.iter().enumerate() {
                self.say(&format!(
                    "    {b:>4}  {:>10.2}  {:>12.2}  {:>6.2}  {:>10.3e}",
                    rb.bank.center_freq_hz[b],
                    rb.bank.bandwidth_hz[b],
                    tau * 1e3,
                    rb.bank.normalization_gain[b],
                ));
            }
        }
    }
}

/// Atom kinds the dictionary actually holds, in `AtomKind::ALL` order, with their block counts.
fn kinds_of(dict: &Dictionary) -> Vec<(AtomKind, usize)> {
    AtomKind::ALL
        .iter()
        .map(|&k| (k, dict.blocks.iter().filter(|b| b.env.kind() == k).count()))
        .filter(|&(_, n)| n > 0)
        .collect()
}

/// How the removed energy divided between the kinds on offer.
fn per_kind(book: &Book, kinds: &[(AtomKind, usize)]) -> Vec<String> {
    let total: f64 = book.selections.iter().map(|s| s.energy_removed).sum();
    kinds
        .iter()
        .map(|&(k, _)| {
            let picked: Vec<_> = book.selections.iter().filter(|s| s.atom.kind() == k).collect();
            let energy: f64 = picked.iter().map(|s| s.energy_removed).sum();
            format!(
                "{} {k} ({:.0}% of the energy removed)",
                picked.len(),
                100.0 * energy / total.max(f64::MIN_POSITIVE)
            )
        })
        .collect()
}
