//! `rmpstat` — statistics and visualization for decomposition books.
//!
//! ```text
//! rmpstat summary book.json
//! rmpstat hist    book.json --of alpha,bandwidth,f --weight energy
//! rmpstat hist    book.json --of alpha,f -f svg -o plots/
//! rmpstat diag    book.json -c settings.toml
//! rmpstat snr     book.json -f svg -o snr.svg
//! rmpstat wv      book.json -f png -o wv.png --log-freq --overlay
//! ```
//!
//! Reports go to stdout, progress and warnings to stderr, matching `rmp`.

mod render;

use clap::{Parser, Subcommand, ValueEnum};
use render::{
    Image, Table, histogram_chart, histogram_text, init_fonts, num, snr_chart, summary_line,
    wv_chart,
};
use rmp_core::book::{self, Book};
use rmp_core::config::Config;
use rmp_core::dict::Dictionary;
use rmp_core::fft::Planner;
use rmp_core::stats::{self, Evaluator, Quantity, Weight};
use rmp_core::tfmap::{self, MapGrid, MapOptions, Reference};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;

#[derive(Parser, Debug)]
#[command(
    name = "rmpstat",
    about = "Statistics and visualization for rmp decomposition books",
    version
)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Headline figures: atoms, SNR, convergence, refinement and HRMP rates, energy accounting.
    Summary {
        book: PathBuf,
        /// Settings the run used, for the dictionary the seed indices refer to.
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
    /// Distributions of atom parameters.
    Hist {
        book: PathBuf,
        /// Comma-separated quantities: alpha, bandwidth, beta, alpha-beta, sigma, f, amp, energy,
        /// t0, support, fade-dur, q, rho, periods, block. alpha, beta, alpha-beta, fade-dur and rho
        /// describe FOF atoms only, sigma Gaussian atoms only.
        #[arg(long, value_name = "LIST", default_value = "alpha,beta,f,amp")]
        of: String,
        #[arg(long, default_value_t = 24)]
        bins: usize,
        /// Force geometric bins. The default is per quantity — see `Quantity::log_by_default`.
        #[arg(long, conflicts_with = "linear")]
        log: bool,
        /// Force linear bins.
        #[arg(long)]
        linear: bool,
        /// Restrict the axis, as `LO,HI`.
        #[arg(long, value_name = "LO,HI")]
        range: Option<String>,
        #[arg(long, value_enum, default_value_t = WeightArg::Count)]
        weight: WeightArg,
        /// Estimate support from `alpha` instead of rendering. Only affects `support`/`periods`.
        #[arg(long)]
        fast_support: bool,
        #[command(flatten)]
        out: OutArgs,
    },
    /// Pursuit diagnostics: refinement drift, HRMP clamping, grid fit, conditioning.
    Diag {
        book: PathBuf,
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
    /// The convergence curve.
    Snr {
        book: PathBuf,
        #[command(flatten)]
        out: OutArgs,
    },
    /// The separable-marginal pseudo-Wigner time-frequency map.
    Wv {
        book: PathBuf,
        /// Grid size as `WIDTHxHEIGHT`, in cells.
        #[arg(long, value_name = "WxH", default_value = "1200x800")]
        size: String,
        /// Geometric frequency axis.
        #[arg(long)]
        log_freq: bool,
        /// Display range below the reference, in dB.
        #[arg(long, default_value_t = 60.0)]
        floor: f32,
        #[arg(long, value_enum, default_value_t = RefArg::Max)]
        reference: RefArg,
        #[arg(long, value_enum, default_value_t = WvWeight::Energy)]
        weight: WvWeight,
        /// Restrict the time axis, in seconds.
        #[arg(short, long, value_name = "SECONDS")]
        start: Option<f64>,
        #[arg(short, long, value_name = "SECONDS")]
        duration: Option<f64>,
        /// Mark each atom's `(t0, f)` — if the heat is not under the dots, something is wrong.
        #[arg(long)]
        overlay: bool,
        #[command(flatten)]
        out: OutArgs,
    },
}

#[derive(clap::Args, Debug)]
struct OutArgs {
    #[arg(short, long, value_enum, default_value_t = Format::Text)]
    format: Format,
    /// Output path. For `hist` with several quantities and an image format, a directory.
    #[arg(short, long)]
    out: Option<PathBuf>,
    /// Image size in pixels, as `WIDTHxHEIGHT`.
    #[arg(long, value_name = "WxH", default_value = "1000x620")]
    pixels: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Format {
    Text,
    Svg,
    Png,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WeightArg {
    Count,
    Energy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum RefArg {
    Max,
    Initial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WvWeight {
    Energy,
    Hr,
    Atom,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rmpstat: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    match args.cmd {
        Cmd::Summary { book, config } => {
            let b = book::read(&book)?;
            cmd_summary(&b, config.as_deref())
        }
        Cmd::Hist {
            book,
            of,
            bins,
            log,
            linear,
            range,
            weight,
            fast_support,
            out,
        } => {
            let b = book::read(&book)?;
            let log = match (log, linear) {
                (true, _) => Some(true),
                (_, true) => Some(false),
                _ => None,
            };
            cmd_hist(&b, &of, bins, log, range.as_deref(), weight, fast_support, &out)
        }
        Cmd::Diag { book, config } => {
            let b = book::read(&book)?;
            cmd_diag(&b, config.as_deref())
        }
        Cmd::Snr { book, out } => {
            let b = book::read(&book)?;
            cmd_snr(&b, &out)
        }
        Cmd::Wv {
            book,
            size,
            log_freq,
            floor,
            reference,
            weight,
            start,
            duration,
            overlay,
            out,
        } => {
            let b = book::read(&book)?;
            cmd_wv(
                &b, &size, log_freq, floor, reference, weight, start, duration, overlay, &out,
            )
        }
    }
}

// ---------------------------------------------------------------------------------------------

/// The dictionary the seed indices refer to, at the book's own sample rate, plus a note on where
/// the settings came from.
///
/// The book records only the seed's block index, never the grid itself, and the block list depends
/// on the sample rate through the supports. A config that does not match the run gives silently
/// meaningless drift figures, so the caller is always told which settings were used.
fn dictionary_at(sr: f32, config: Option<&Path>) -> Result<(Dictionary, Config, String), String> {
    let (cfg, note) = match config {
        Some(p) => {
            let text =
                std::fs::read_to_string(p).map_err(|e| format!("reading {}: {e}", p.display()))?;
            let cfg =
                Config::from_toml(&text).map_err(|e| format!("parsing {}: {e}", p.display()))?;
            (cfg, format!("dictionary from {}", p.display()))
        }
        None => (
            Config::default(),
            "dictionary from the built-in defaults — pass -c to match the run that wrote this book"
                .to_string(),
        ),
    };
    cfg.validate()?;
    let mut planner = Planner::new();
    let dict =
        Dictionary::from_shapes(&cfg.dictionary_shapes(), sr, &mut planner, &cfg.block_config())
            .map_err(|e| format!("building the dictionary: {e}"))?;
    Ok((dict, cfg, note))
}

fn cmd_summary(book: &Book, config: Option<&Path>) -> Result<(), String> {
    let mut ev = Evaluator::new(book);
    let s = stats::summarize(book, &mut ev).map_err(|e| e.to_string())?;
    let sr = book.sample_rate as f64;

    println!("book");
    println!("  atoms               {}", s.atoms);
    println!("  sample rate         {:.0} Hz", s.sample_rate);
    println!(
        "  span                {:.3} .. {:.3} s  ({:.3} s)",
        s.span.0 as f64 / sr,
        s.span.1 as f64 / sr,
        (s.span.1 - s.span.0) as f64 / sr
    );

    println!("\nconvergence");
    println!("  final SNR           {:.1} dB", s.snr_db);
    for (target, at) in &s.atoms_to_reach {
        match at {
            Some(n) => println!("  {target:>4.0} dB reached at  {n} atoms"),
            None => println!("  {target:>4.0} dB              not reached"),
        }
    }

    println!("\nenergy");
    println!("  initial             {:.6e}", s.initial_energy);
    println!("  residual            {:.6e}", s.residual_energy);
    println!(
        "  sum of atoms        {:.1}% of what the pursuit removed",
        100.0 * s.deposited_frac
    );
    println!("  per atom          {}", summary_line(&s.energy_removed));

    println!("\npursuit");
    println!(
        "  refined             {}/{} ({:.1}%)",
        (s.refined_frac * s.atoms as f64).round() as usize,
        s.atoms,
        100.0 * s.refined_frac
    );
    let r = &s.removed_over_projected;
    println!(
        "  removed/projected   median {:.3}, p5 {:.3}, min {:.3}, max {:.3}",
        r.median, r.p5, r.min, r.max
    );
    println!(
        "  fell short of it    {}/{} atoms by more than 1%",
        s.shortfall_atoms, s.atoms
    );
    if s.hrmp_atoms == 0 {
        println!("  HRMP                not run");
        // With no clamp in the path, removed and projected are the same energy computed two ways,
        // so a real shortfall here is a parameter-mapping error rather than a setting.
        if r.n > 0 && r.p5 < 0.9 {
            eprintln!(
                "rmpstat: warning: HRMP did not run, yet 5% of atoms removed under {:.0}% of \
                 their projection — with no clamp in the path that points at a parameter-mapping \
                 error, not a setting",
                100.0 * r.p5
            );
        }
    } else {
        println!(
            "  HRMP ran on         {}/{} atoms — the shortfall above is its amplitude clamp",
            s.hrmp_atoms, s.atoms
        );
        if s.hr_consistency > 0.05 {
            eprintln!(
                "rmpstat: warning: hr_score and energy_removed differ by up to {:.1}% — they \
                 record the same post-clamp energy and should agree",
                100.0 * s.hr_consistency
            );
        }
    }

    // The stochastic half, when the book carries one. Only what it is, not what is in it: the
    // power matrix is a different kind of object from everything above and wants its own reader.
    if let Some(rb) = &book.residual {
        println!("\nresidual analysis");
        println!(
            "  bank                {} ERB bands, {:.0} .. {:.0} Hz, order {} {}",
            rb.band_count,
            rb.bank.min_freq_hz,
            rb.bank.max_freq_hz,
            rb.bank.filter_order,
            rb.bank.filter_kind
        );
        println!(
            "  frames              {} every {} samples ({:.3} ms)",
            rb.frame_count,
            rb.update_samples,
            rb.update_samples as f64 * 1e3 / rb.sample_rate
        );
        let taus = &rb.bank.power_detector.tau_seconds;
        println!(
            "  power detector      {}, tau {:.2} .. {:.2} ms",
            rb.bank.power_detector.mode,
            taus.iter().cloned().fold(f64::INFINITY, f64::min) * 1e3,
            taus.iter().cloned().fold(0.0, f64::max) * 1e3
        );
    }

    // The two pictures of the same book: the grid it searched, and where it ended up.
    println!("\nparameters");
    // Per-kind quantities print only when some atom has them, so a FOF-only book reads as it did.
    let kinds: Vec<String> = rmp_core::atom::AtomKind::ALL
        .iter()
        .map(|&k| (k, book.selections.iter().filter(|s| s.atom.kind() == k).count()))
        .filter(|&(_, n)| n > 0)
        .map(|(k, n)| format!("{n} {k}"))
        .collect();
    if kinds.len() > 1 {
        println!("  kinds               {}", kinds.join(", "));
    }
    for q in [
        Quantity::Alpha,
        Quantity::Bandwidth,
        Quantity::Beta,
        Quantity::AlphaBeta,
        Quantity::Sigma,
        Quantity::Freq,
        Quantity::AmpDb,
    ] {
        let col = ev.column(book, q).map_err(|e| e.to_string())?;
        let st = stats::Summary::of(
            &col.iter().flatten().copied().filter(|v| v.is_finite()).collect::<Vec<_>>(),
        );
        if st.n == 0 && col.iter().all(Option::is_none) {
            continue;
        }
        println!("  {:<22}{}", q.label(), summary_line(&st).trim_start());
    }

    if let Ok((dict, cfg, note)) = dictionary_at(book.sample_rate, config) {
        let d = stats::diagnose(book, &dict, cfg.blocks.rho_sq_max);
        println!("\nseed grid ({note})");
        if let Some(w) = &d.edge_pileup {
            println!("  WARNING: {w}");
        }
        if d.d_ln_alpha.n > 0 || d.d_ln_sigma.n == 0 {
            println!(
                "  refinement moved    |ln a/a0| median {:.3}, |ln b/b0| median {:.3}",
                d.d_ln_alpha.median, d.d_ln_beta.median
            );
        }
        if d.d_ln_sigma.n > 0 {
            println!("  refinement moved    |ln s/s0| median {:.3}", d.d_ln_sigma.median);
        }
    }
    Ok(())
}

fn cmd_diag(book: &Book, config: Option<&Path>) -> Result<(), String> {
    let (dict, cfg, note) = dictionary_at(book.sample_rate, config)?;
    let d = stats::diagnose(book, &dict, cfg.blocks.rho_sq_max);
    println!("{note}\n");

    if let Some(w) = &d.edge_pileup {
        println!("WARNING: {w}\n");
    }

    println!("blocks");
    let mut t = Table::new(&[
        "block", "kind", "alpha", "beta_ms", "sigma_ms", "support", "seeds", "%", "energy %",
    ]);
    let total: usize = d.blocks.iter().map(|b| b.count).sum();
    let dash = || "-".to_string();
    for b in &d.blocks {
        let (fof, gauss) = (b.shape.as_fof(), b.shape.as_gaussian());
        t.row(vec![
            b.index.to_string(),
            b.shape.kind().to_string(),
            fof.map_or_else(dash, |p| format!("{:.0}", p.alpha)),
            fof.map_or_else(dash, |p| format!("{:.2}", p.beta * 1e3)),
            gauss.map_or_else(dash, |g| format!("{:.2}", g.sigma * 1e3)),
            b.support_len.to_string(),
            b.count.to_string(),
            format!(
                "{:.1}",
                if total > 0 {
                    100.0 * b.count as f64 / total as f64
                } else {
                    0.0
                }
            ),
            format!("{:.1}", 100.0 * b.energy_share),
        ]);
    }
    print!("{}", t.render("  "));

    // Refinement's whole job is to leave the grid, so these are the numbers that say whether it
    // did. One alpha rung is ln 1.6 = 0.47; one beta rung about ln 3.3 = 1.19.
    println!("\nrefinement drift from the seed");
    println!("  |ln alpha/alpha_0|  {}", summary_line(&d.d_ln_alpha).trim_start());
    println!("  |ln beta/beta_0|    {}", summary_line(&d.d_ln_beta).trim_start());
    if d.d_ln_sigma.n > 0 {
        println!("  |ln sigma/sigma_0|  {}", summary_line(&d.d_ln_sigma).trim_start());
    }
    println!("  |f - bin| Hz        {}", summary_line(&d.d_f_hz).trim_start());
    println!("  |t0 - onset| samp   {}", summary_line(&d.d_t0).trim_start());
    println!(
        "  (one alpha rung is ln 1.6 = 0.470; one beta rung about ln 3.3 = 1.19)"
    );

    println!("\nconditioning");
    println!(
        "  ill-conditioned     {} atoms whose seed bin had rho^2 above {}",
        d.ill_conditioned, cfg.blocks.rho_sq_max
    );
    if d.off_grid_seeds > 0 {
        println!(
            "  off-grid seeds      {} — the config probably does not match the run",
            d.off_grid_seeds
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_hist(
    book: &Book,
    of: &str,
    bins: usize,
    log: Option<bool>,
    range: Option<&str>,
    weight: WeightArg,
    fast_support: bool,
    out: &OutArgs,
) -> Result<(), String> {
    let quantities: Vec<Quantity> = of
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(Quantity::from_str)
        .collect::<Result<_, _>>()?;
    if quantities.is_empty() {
        return Err("--of listed no quantities".into());
    }
    let range = match range {
        Some(r) => {
            let (a, b) = r
                .split_once(',')
                .ok_or("--range wants LO,HI")?;
            Some((
                a.trim().parse::<f64>().map_err(|e| format!("--range: {e}"))?,
                b.trim().parse::<f64>().map_err(|e| format!("--range: {e}"))?,
            ))
        }
        None => None,
    };
    let w = match weight {
        WeightArg::Count => Weight::Count,
        WeightArg::Energy => Weight::Energy,
    };

    let mut ev = Evaluator::new(book);
    ev.fast_support = fast_support;

    let mut first = true;
    for q in quantities.iter().copied() {
        let h = stats::histogram(book, &mut ev, q, bins, log, range, w)
            .map_err(|e| e.to_string())?;
        match out.format {
            Format::Text => {
                if !first {
                    println!();
                }
                print!("{}", histogram_text(&h));
            }
            f => {
                begin_images()?;
                let path = image_path(out.out.as_deref(), &slug(q), f, quantities.len() > 1)?;
                histogram_chart(&h, &path, image_of(f), parse_size(&out.pixels)?)?;
                eprintln!("wrote {}", path.display());
            }
        }
        first = false;
    }
    Ok(())
}

fn cmd_snr(book: &Book, out: &OutArgs) -> Result<(), String> {
    let trace = book.snr_trace();
    if trace.is_empty() {
        return Err("the book has no atoms".into());
    }
    match out.format {
        Format::Text => {
            println!("convergence  ({} atoms, final {:.1} dB)", trace.len(), book.snr_db());
            // Decile checkpoints: the whole trace is one line per atom and nobody reads 5000 of
            // them, but the shape — where it flattens — is the diagnostic.
            let mut t = Table::new(&["atoms", "SNR dB"]);
            for i in 0..=10 {
                let k = (i * (trace.len() - 1)) / 10;
                t.row(vec![(k + 1).to_string(), format!("{:.2}", trace[k])]);
            }
            print!("{}", t.render("  "));
            for target in [10.0f32, 20.0, 30.0, 40.0] {
                match book.atoms_to_reach(target) {
                    Some(n) => println!("  {target:>4.0} dB at {n} atoms"),
                    None => println!("  {target:>4.0} dB not reached"),
                }
            }
        }
        f => {
            begin_images()?;
            let path = image_path(out.out.as_deref(), "snr", f, false)?;
            snr_chart(&trace, &path, image_of(f), parse_size(&out.pixels)?)?;
            eprintln!("wrote {}", path.display());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_wv(
    book: &Book,
    size: &str,
    log_freq: bool,
    floor: f32,
    reference: RefArg,
    weight: WvWeight,
    start: Option<f64>,
    duration: Option<f64>,
    overlay: bool,
    out: &OutArgs,
) -> Result<(), String> {
    if book.is_empty() {
        return Err("the book has no atoms".into());
    }
    if !(floor.is_finite() && floor > 0.0) {
        return Err("--floor wants a positive number of dB".into());
    }
    let (n_t, n_f) = {
        let (w, h) = parse_size(size)?;
        (w as usize, h as usize)
    };
    let opts = MapOptions {
        weight: match weight {
            WvWeight::Energy => tfmap::Weight::EnergyRemoved,
            WvWeight::Hr => tfmap::Weight::HrScore,
            WvWeight::Atom => tfmap::Weight::AtomEnergy,
        },
        ..Default::default()
    };

    let mut grid =
        MapGrid::covering(book, n_t, n_f, log_freq).map_err(|e| e.to_string())?;
    // `--start`/`--duration` narrow the time axis, in seconds, as they do on `rmp`.
    if start.is_some() || duration.is_some() {
        let sr = book.sample_rate as f64;
        let t0 = start.map_or(grid.t_edges[0], |s| s * sr);
        let t1 = duration.map_or(grid.t_edges[grid.n_t()], |d| t0 + d * sr);
        if t1 <= t0 {
            return Err("--duration must be positive".into());
        }
        let (f0, f1) = (grid.f_edges[0], grid.f_edges[grid.n_f()]);
        grid = if log_freq {
            MapGrid::log_freq(t0..t1, n_t, f0..f1, n_f, book.sample_rate)
        } else {
            MapGrid::linear(t0..t1, n_t, f0..f1, n_f, book.sample_rate)
        };
    }

    let map = tfmap::compute(book, grid, &opts).map_err(|e| e.to_string())?;
    let captured = book.initial_energy - book.residual_energy();
    let accounted = map.deposited + map.clipped;

    let caption = format!(
        "{} atoms, {:.1}% of the removed energy on the grid — separable-marginal pseudo-Wigner, \
         floor {:.0} dB",
        map.atoms,
        100.0 * map.deposited / accounted.max(f64::MIN_POSITIVE),
        floor
    );

    match out.format {
        Format::Text => {
            println!("pseudo-Wigner map  {}x{} cells", map.grid.n_t(), map.grid.n_f());
            println!(
                "  time              {:.3} .. {:.3} s",
                map.grid.t_edges[0] / book.sample_rate as f64,
                map.grid.t_edges[map.grid.n_t()] / book.sample_rate as f64
            );
            println!(
                "  frequency         {:.0} .. {:.0} Hz{}",
                map.grid.f_edges[0],
                map.grid.f_edges[map.grid.n_f()],
                if log_freq { " (log)" } else { "" }
            );
            println!("  atoms             {} drawn, {} skipped", map.atoms, map.skipped);
            println!("  deposited         {:.6e}", map.deposited);
            println!("  clipped           {:.6e}", map.clipped);
            println!("  book removed      {captured:.6e}");
            report_accounting(accounted, captured);
            println!("\n  the map is a separable-marginal pseudo-Wigner representation: each atom");
            println!("  contributes the product of its exact time and frequency marginals, summed");
            println!("  per atom so there are no cross-terms.");

            let tm = map.time_marginal();
            let fm = map.freq_marginal();
            println!("\ntime marginal (deciles)");
            print!("{}", marginal_table(&tm, |i| {
                format!(
                    "{:.3}",
                    (map.grid.t_edges[i] / book.sample_rate as f64)
                )
            }));
            println!("\nfrequency marginal (deciles)");
            print!("{}", marginal_table(&fm, |i| format!("{:.0}", map.grid.f_edges[i])));
        }
        f => {
            begin_images()?;
            let path = image_path(out.out.as_deref(), "wv", f, false)?;
            let dots: Vec<(f64, f32)> = if overlay {
                book.selections
                    .iter()
                    .map(|s| (s.atom.t0 as f64 / book.sample_rate as f64, s.atom.f))
                    .collect()
            } else {
                Vec::new()
            };
            // The image is one pixel per cell plus the axis furniture around it.
            let px = (n_t as u32 + 110, n_f as u32 + 110);
            wv_chart(
                &map,
                floor,
                match reference {
                    RefArg::Max => Reference::Max,
                    RefArg::Initial => Reference::Initial,
                },
                &dots,
                &caption,
                &path,
                image_of(f),
                px,
            )?;
            eprintln!("wrote {} ({}x{} cells)", path.display(), n_t, n_f);
            eprintln!("{caption}");
            report_accounting(accounted, captured);
        }
    }
    Ok(())
}

/// The map's total against the book's own figure.
///
/// These agree to rounding when the weight is `energy_removed`, because that is the quantity the
/// book's residual trace is written in. A visible gap means the two disagree about what was
/// removed, which is worth saying out loud rather than leaving in a picture.
fn report_accounting(accounted: f64, captured: f64) {
    if captured <= 0.0 {
        return;
    }
    let rel = (accounted - captured).abs() / captured;
    if rel > 1e-6 {
        eprintln!(
            "rmpstat: note: atoms sum to {:.4}x what the pursuit removed — greedy MP re-removes \
             energy earlier atoms had already taken, so below 1 is ordinary",
            accounted / captured
        );
    }
}

/// Ten evenly-spaced entries of a marginal, with the bin's lower edge.
fn marginal_table(v: &[f64], edge: impl Fn(usize) -> String) -> String {
    let mut t = Table::new(&["from", "energy", "%"]);
    let total: f64 = v.iter().sum();
    let step = (v.len() / 10).max(1);
    for i in (0..v.len()).step_by(step) {
        let chunk: f64 = v[i..(i + step).min(v.len())].iter().sum();
        t.row(vec![
            edge(i),
            num(chunk),
            format!("{:.1}", if total > 0.0 { 100.0 * chunk / total } else { 0.0 }),
        ]);
    }
    t.render("  ")
}

/// Prepare for image output, or explain why it cannot happen.
fn begin_images() -> Result<(), String> {
    init_fonts()
}

fn image_of(f: Format) -> Image {
    match f {
        Format::Png => Image::Png,
        _ => Image::Svg,
    }
}

/// Where an image goes.
///
/// With several quantities, `-o` names a directory and each gets its own file; with one, `-o` may
/// name the file directly. Without `-o` the file lands in the working directory under the
/// quantity's own name, so nothing is silently overwritten by a differently-named run.
fn image_path(out: Option<&Path>, name: &str, f: Format, many: bool) -> Result<PathBuf, String> {
    let ext = if f == Format::Png { "png" } else { "svg" };
    let file = format!("{name}.{ext}");
    Ok(match out {
        None => PathBuf::from(file),
        Some(p) => {
            let as_dir = many || p.is_dir() || p.extension().is_none();
            if as_dir {
                std::fs::create_dir_all(p)
                    .map_err(|e| format!("creating {}: {e}", p.display()))?;
                p.join(file)
            } else {
                p.to_path_buf()
            }
        }
    })
}

fn slug(q: Quantity) -> String {
    match q {
        Quantity::Alpha => "alpha",
        Quantity::Bandwidth => "bandwidth",
        Quantity::Beta => "beta",
        Quantity::Sigma => "sigma",
        Quantity::AlphaBeta => "alpha-beta",
        Quantity::Freq => "f",
        Quantity::AmpDb => "amp",
        Quantity::EnergyDb => "energy",
        Quantity::T0 => "t0",
        Quantity::SupportMs => "support",
        Quantity::FadeDurMs => "fade-dur",
        Quantity::Q => "q",
        Quantity::Rho => "rho",
        Quantity::Periods => "periods",
        Quantity::Block => "block",
    }
    .to_string()
}

fn parse_size(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("size '{s}' wants WIDTHxHEIGHT"))?;
    let w: u32 = w.trim().parse().map_err(|_| format!("bad width in '{s}'"))?;
    let h: u32 = h.trim().parse().map_err(|_| format!("bad height in '{s}'"))?;
    if w == 0 || h == 0 {
        return Err(format!("size '{s}' must be positive"));
    }
    Ok((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_parse_or_explain_themselves() {
        assert_eq!(parse_size("1200x800").unwrap(), (1200, 800));
        assert_eq!(parse_size(" 64 X 48 ").unwrap(), (64, 48));
        assert!(parse_size("1200").is_err());
        assert!(parse_size("0x10").is_err());
        assert!(parse_size("axb").is_err());
    }

    #[test]
    fn a_single_quantity_may_name_its_file_directly() {
        let p = image_path(Some(Path::new("/tmp/x/plot.svg")), "alpha", Format::Svg, false).unwrap();
        assert_eq!(p, Path::new("/tmp/x/plot.svg"));
    }

    #[test]
    fn several_quantities_name_a_directory() {
        let dir = std::env::temp_dir().join(format!("rmpstat-{}", std::process::id()));
        let p = image_path(Some(&dir), "alpha", Format::Png, true).unwrap();
        assert_eq!(p, dir.join("alpha.png"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn without_out_the_file_is_named_for_the_quantity() {
        assert_eq!(
            image_path(None, "bandwidth", Format::Svg, false).unwrap(),
            Path::new("bandwidth.svg")
        );
    }

    /// Every quantity has a distinct filename, or two histograms in one run would overwrite.
    #[test]
    fn slugs_are_unique() {
        let all = [
            Quantity::Alpha,
            Quantity::Bandwidth,
            Quantity::Beta,
            Quantity::AlphaBeta,
            Quantity::Freq,
            Quantity::AmpDb,
            Quantity::EnergyDb,
            Quantity::T0,
            Quantity::SupportMs,
            Quantity::FadeDurMs,
            Quantity::Q,
            Quantity::Rho,
            Quantity::Periods,
            Quantity::Block,
        ];
        let mut seen: Vec<String> = all.iter().map(|&q| slug(q)).collect();
        seen.sort();
        let n = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), n);
        // And each round-trips back through the parser it advertises.
        for q in all {
            assert_eq!(Quantity::from_str(&slug(q)).unwrap(), q);
        }
    }

    #[test]
    fn cli_parses_the_documented_invocations() {
        use clap::CommandFactory;
        Args::command().debug_assert();
        for argv in [
            vec!["rmpstat", "summary", "b.json"],
            vec!["rmpstat", "hist", "b.json", "--of", "alpha,f", "--weight", "energy"],
            vec!["rmpstat", "hist", "b.json", "--of", "alpha", "-f", "svg", "-o", "p/"],
            vec!["rmpstat", "diag", "b.json", "-c", "s.toml"],
            vec!["rmpstat", "snr", "b.json", "-f", "svg", "-o", "s.svg"],
            vec!["rmpstat", "wv", "b.json", "-f", "png", "-o", "w.png", "--log-freq", "--overlay"],
        ] {
            Args::try_parse_from(&argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        }
        // --log and --linear are contradictory and must be refused, not silently ordered.
        assert!(
            Args::try_parse_from(["rmpstat", "hist", "b.json", "--log", "--linear"]).is_err()
        );
    }
}
