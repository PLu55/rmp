//! End-to-end run at realistic scale.
//!
//! Synthesizes a signal from known FOF parameters, analyzes it with the full voice dictionary at
//! 48 kHz, and reports cost, convergence, and parameter recovery.
//!
//! ```text
//! cargo run --release --example analyze [seconds] [max_atoms] [grains_per_sec] [candidates] [capture_tol]
//! ```
//!
//! Everything the plan predicted but never measured lands here: the per-iteration cost model, how
//! badly a coherent dictionary splits one true grain across several atoms, and what the frequency
//! and onset grids actually cost against off-grid input.

use rmp::book::Book;
use rmp::dict::{BlockConfig, Dictionary};
use rmp::fft::Planner;
use rmp::fof::AtomParams;
use rmp::mp::{Mp, MpConfig};
use rmp::refine::RefineConfig;
use rmp::signal::Signal;
use std::time::Instant;

const SR: f32 = 48_000.0;

fn main() {
    let mut args = std::env::args().skip(1);
    let seconds: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.25);
    let max_atoms: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(400);
    let density: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let candidates: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let capture_tol: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.95);
    let len = (seconds * SR) as usize;

    let mut planner = Planner::new();

    println!("building voice dictionary at {SR} Hz");
    let t = Instant::now();
    let cfg = BlockConfig { capture_tol, ..BlockConfig::default() };
    let dict = Dictionary::voice(SR, &mut planner, &cfg).unwrap();
    println!(
        "  built {} blocks in {:.2?} (capture_tolerance {capture_tol})\n",
        dict.blocks.len(),
        t.elapsed()
    );

    report_dictionary(&dict, len);

    // Ground truth: atoms placed on the dictionary grid, so recovery should be near-exact.
    let truth = plant_atoms(&dict, len, false, density);
    println!("\nplanted {} atoms over {seconds} s ({len} samples)", truth.len());
    let signal = Signal::from_atoms(&truth, len, SR).unwrap();
    println!("  signal energy {:.4e}", signal.energy());

    let grid = MpConfig {
        max_atoms,
        target_snr_db: 40.0,
        ..Default::default()
    };
    let refined = MpConfig {
        candidate_count: candidates,
        refine: RefineConfig::default(),
        ..grid
    };

    let book = run(&dict, &signal, &grid, &mut planner);
    report_convergence(&book, &dict, &signal);
    report_recovery(&book, &truth);
    report_roundtrip(&book, &signal);

    // Off-grid: the honest measure, since real signals do not sit on grid points. Run it both ways,
    // because the whole case for refinement is the difference between these two.
    println!("\n{:=<72}", "");
    println!("OFF-GRID: same atoms, shifted off the frequency and onset grids");
    println!("{:=<72}", "");
    let truth = plant_atoms(&dict, len, true, density);
    let signal = Signal::from_atoms(&truth, len, SR).unwrap();

    println!("\n-- grid only --");
    let book = run(&dict, &signal, &grid, &mut planner);
    report_convergence(&book, &dict, &signal);
    report_recovery(&book, &truth);

    println!("\n-- refined --");
    let book = run(&dict, &signal, &refined, &mut planner);
    report_convergence(&book, &dict, &signal);
    report_refinement(&book, &dict);
    report_recovery(&book, &truth);
    report_roundtrip(&book, &signal);
}

fn run(dict: &Dictionary, signal: &Signal, cfg: &MpConfig, planner: &mut Planner) -> Book {
    let t = Instant::now();
    let mut mp = Mp::new(dict, signal, planner);
    let init = t.elapsed();

    let t = Instant::now();
    let book = mp.run(cfg);
    let pursuit = t.elapsed();

    println!("\ncost");
    println!("  init (all frames)   {init:.2?}");
    println!("  pursuit             {pursuit:.2?}  for {} atoms", book.len());
    if !book.is_empty() {
        println!(
            "  per atom            {:.2?}",
            pursuit / book.len() as u32
        );
    }
    let audio_s = signal.len() as f32 / SR;
    println!(
        "  realtime factor     {:.1}x  (init+pursuit vs {audio_s:.2} s of audio)",
        (init + pursuit).as_secs_f32() / audio_s
    );
    book
}

/// How far refinement actually moved the atoms it accepted.
fn report_refinement(book: &Book, dict: &Dictionary) {
    if book.is_empty() {
        return;
    }
    let (mut moved, mut d_alpha, mut d_beta) = (0usize, 0.0f64, 0.0f64);
    for s in &book.selections {
        // `Selection::block` is the seed's provenance, so the block's own envelope is the
        // before-picture and `atom.env` is whatever refinement settled on.
        let (Some(seed), Some(atom)) = (dict.blocks[s.block].env.params.as_fof(), s.atom.env.as_fof())
        else {
            continue;
        };
        if seed.alpha != atom.alpha || seed.beta != atom.beta {
            moved += 1;
        }
        d_alpha += (atom.alpha as f64 / seed.alpha as f64).ln().abs();
        d_beta += (atom.beta as f64 / seed.beta as f64).ln().abs();
    }
    let n = book.len() as f64;
    println!("\nrefinement");
    println!(
        "  atoms moved off the grid   {moved}/{} ({:.0}%)",
        book.len(),
        100.0 * moved as f64 / n
    );
    println!("  mean |ln alpha/alpha_0|    {:.3}", d_alpha / n);
    println!("  mean |ln beta/beta_0|      {:.3}", d_beta / n);
}

fn report_dictionary(dict: &Dictionary, len: usize) {
    println!(
        "{:<28} {:>7} {:>6} {:>8} {:>7} {:>9}",
        "shape", "support", "hop", "fft_len", "bins", "frames"
    );
    let mut total_frames = 0usize;
    for b in &dict.blocks {
        let frames = b.frame_count(len);
        total_frames += frames;
        println!(
            "{:<28} {:>7} {:>6} {:>8} {:>7} {:>9}",
            b.env.params.describe(),
            b.support_len(),
            b.hop,
            b.fft_len,
            b.k_hi - b.k_lo + 1,
            frames
        );
    }
    println!("  total frames to correlate at init: {total_frames}");
}

/// A spread of atoms across the dictionary, at a fixed density per second so longer runs are
/// denser rather than merely longer. With `off_grid`, frequencies land half a bin up and onsets
/// half a hop late — the worst case the grid has to cover.
fn plant_atoms(dict: &Dictionary, len: usize, off_grid: bool, per_second: f32) -> Vec<AtomParams> {
    let count = ((len as f32 / SR) * per_second).round().max(1.0) as usize;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 16_777_216.0
    };

    (0..count)
        .map(|_| {
            let b = &dict.blocks[(rnd() * dict.blocks.len() as f32) as usize % dict.blocks.len()];
            let span = b.k_hi - b.k_lo;
            let k = b.k_lo + (rnd() * span as f32) as usize;
            let onset = (rnd() * len as f32) as usize;
            let (f, t0) = if off_grid {
                (
                    b.bin_hz(k) + 0.5 * SR / b.fft_len as f32,
                    onset + b.hop / 2 + 1,
                )
            } else {
                (b.bin_hz(k), onset - onset % b.hop)
            };
            AtomParams {
                t0: t0 as i64,
                f,
                env: b.env.params,
                phi: rnd() * std::f32::consts::TAU,
                amp: 0.3 + 0.7 * rnd(),
            }
        })
        .collect()
}

fn report_convergence(book: &Book, dict: &Dictionary, signal: &Signal) {
    println!("\nconvergence");
    println!("  final SNR           {:.1} dB", book.snr_db());
    for target in [10.0f32, 20.0, 30.0, 40.0] {
        match book.atoms_to_reach(target) {
            Some(n) => println!("  {target:>4.0} dB reached at  {n} atoms"),
            None => println!("  {target:>4.0} dB              not reached"),
        }
    }
    let _ = signal;

    let hist = book.block_histogram(dict.blocks.len());
    let used: Vec<String> = hist
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(i, &c)| match dict.blocks[i].env.params {
            rmp::Shape::Fof(p) => format!("a{:.0}/b{:.1}:{c}", p.alpha, p.beta * 1000.0),
            rmp::Shape::Gaussian(g) => format!("s{:.1}:{c}", g.sigma * 1000.0),
        })
        .collect();
    println!("  blocks used         {}", used.join("  "));

    // Selections piling up at an alpha edge mean the ladder is mis-sized.
    let edge = hist[0] + hist[hist.len() - 1];
    let total: usize = hist.iter().sum();
    if total > 0 && edge * 4 > total {
        println!(
            "  WARNING: {:.0}% of picks are at an alpha edge — grid may be mis-sized",
            100.0 * edge as f32 / total as f32
        );
    }
}

/// How many selected atoms cluster around each planted one, and how well the best of them matches.
///
/// The count is the interesting number: a coherent dictionary represents one true grain with
/// several nearby atoms whose amplitudes partly cancel, so a splitting factor well above 1 is the
/// signature that back-projection would pay.
fn report_recovery(book: &Book, truth: &[AtomParams]) {
    println!("\nrecovery");
    println!(
        "  {:>9} {:>8} {:>9} {:>8} {:>8} {:>7}",
        "true f", "d_f Hz", "true t0", "d_t0", "d_amp %", "split"
    );

    let mut splits = Vec::new();
    let mut df: Vec<f32> = Vec::new();
    for want in truth.iter().take(12) {
        let support = want.render(SR).map(|v| v.len()).unwrap_or(512) as i64;
        let window = (support / 2).max(32);

        // Every selected atom whose onset falls inside this grain's span.
        let near: Vec<_> = book
            .selections
            .iter()
            .filter(|s| (s.atom.t0 - want.t0).abs() <= window)
            .collect();
        if near.is_empty() {
            println!("  {:>9.1} {:>8} {:>9} {:>8} {:>8} {:>7}", want.f, "-", want.t0, "-", "-", 0);
            continue;
        }
        // The strongest of them is the one standing in for the true atom.
        let best = near
            .iter()
            .max_by(|a, b| a.atom.amp.abs().total_cmp(&b.atom.amp.abs()))
            .unwrap();

        splits.push(near.len());
        df.push((best.atom.f - want.f).abs());
        println!(
            "  {:>9.1} {:>8.1} {:>9} {:>8} {:>8.1} {:>7}",
            want.f,
            best.atom.f - want.f,
            want.t0,
            best.atom.t0 - want.t0,
            100.0 * (best.atom.amp - want.amp) / want.amp,
            near.len()
        );
    }

    if !splits.is_empty() {
        let mean = splits.iter().sum::<usize>() as f32 / splits.len() as f32;
        println!("  mean splitting factor {mean:.1} atoms per planted grain");
        // Median, not mean: where grains overlap, the time window can catch a neighbouring grain
        // and the frequency difference then says nothing about recovery accuracy.
        df.sort_by(f32::total_cmp);
        println!("  median |d_f| of best match {:.1} Hz", df[df.len() / 2]);
    }
}

fn report_roundtrip(book: &Book, signal: &Signal) {
    let resynth = rmp::synth::atoms::render_atoms(book, signal.len()).unwrap();
    let err: f64 = signal
        .samples
        .iter()
        .zip(&resynth.samples)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    println!(
        "\nround trip: analyse -> book -> render = {:.1} dB",
        rmp::signal::snr_db(signal.energy(), err)
    );
}
