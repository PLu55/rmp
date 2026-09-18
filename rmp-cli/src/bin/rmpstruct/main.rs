//! `rmpstruct` — structural analysis of a decomposition book.
//!
//! ```text
//! rmpstruct partials book.json.gz [-c structure.toml] [-o book.partials.json.gz]
//! rmpstruct show     book.partials.json.gz [-n 20]
//! rmpstruct --write-config > structure.toml
//! ```
//!
//! A file and configuration front end over `rmp-structure`: every number it prints is a field the
//! library returned. It reads a book and writes a derived document beside it; the book itself is
//! never modified. Reports go to stdout, progress and warnings to stderr, matching `rmp`.

use clap::{Parser, Subcommand};
use rmp_structure::config::DEFAULT_CONFIG_HEADER;
use rmp_structure::{PartialBook, PartialDiagnostics, StructureAnalysisConfig, analyze_partials};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(
    name = "rmpstruct",
    about = "Structural analysis of rmp books: persistent partials (stems to follow)",
    version
)]
struct Args {
    #[command(subcommand)]
    cmd: Option<Cmd>,

    /// Print the default structure settings as TOML and exit.
    #[arg(long)]
    write_config: bool,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Extract persistent partials from an MP book and write a partial book.
    Partials {
        book: PathBuf,
        /// Structure settings (`[structure.partials]`). Defaults when omitted.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Where the partial book goes. Defaults to `<book>.partials.json.gz` beside the book.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Also print the strongest partials.
        #[arg(short = 'n', long, value_name = "N")]
        show: Option<usize>,
    },
    /// Print a partial book: its provenance and its strongest partials.
    Show {
        partials: PathBuf,
        /// How many partials to list, by descending significance.
        #[arg(short = 'n', long, default_value_t = 20)]
        count: usize,
    },
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rmpstruct: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    if args.write_config {
        print!("{DEFAULT_CONFIG_HEADER}{}", StructureAnalysisConfig::default().to_toml());
        return Ok(());
    }
    match args.cmd {
        None => Err("nothing to do: give a subcommand (see --help) or --write-config".into()),
        Some(Cmd::Partials { book, config, output, show }) => {
            cmd_partials(&book, config.as_deref(), output, show)
        }
        Some(Cmd::Show { partials, count }) => {
            let pb = PartialBook::read(&partials).map_err(|e| e.to_string())?;
            print_provenance(&pb);
            print_partials(&pb, count);
            Ok(())
        }
    }
}

fn load_config(path: Option<&Path>) -> Result<StructureAnalysisConfig, String> {
    let cfg = match path {
        None => StructureAnalysisConfig::default(),
        Some(p) => {
            let text =
                std::fs::read_to_string(p).map_err(|e| format!("reading {}: {e}", p.display()))?;
            StructureAnalysisConfig::from_toml(&text).map_err(|e| format!("{}: {e}", p.display()))?
        }
    };
    cfg.validate().map_err(|e| e.to_string())?;
    Ok(cfg)
}

/// `dir/name.json.gz` → `dir/name.partials.json.gz`: the book's name without its format and
/// compression suffixes, then the derived document's own.
fn default_output(book: &Path) -> PathBuf {
    let mut stem = book.file_name().and_then(|n| n.to_str()).unwrap_or("book").to_string();
    for suffix in [".gz", ".gzip", ".json", ".toml"] {
        if let Some(s) = stem.strip_suffix(suffix) {
            stem = s.to_string();
        }
    }
    book.with_file_name(format!("{stem}.partials.json.gz"))
}

fn cmd_partials(
    book_path: &Path,
    config: Option<&Path>,
    output: Option<PathBuf>,
    show: Option<usize>,
) -> Result<(), String> {
    let cfg = load_config(config)?;
    let book = rmp_core::book::read(book_path)?;
    let out = output.unwrap_or_else(|| default_output(book_path));
    if out == book_path {
        return Err(format!("refusing to overwrite the MP book {}", out.display()));
    }

    eprintln!("analysing {} atoms from {}", book.selections.len(), book_path.display());
    let started = std::time::Instant::now();
    let mut analysis = analyze_partials(&book, &cfg.partials).map_err(|e| e.to_string())?;
    let elapsed = started.elapsed();
    analysis.book.metadata.source_book = Some(book_path.to_path_buf());

    analysis.book.write(&out).map_err(|e| e.to_string())?;
    print_diagnostics(&analysis.diagnostics);
    println!("time             {:.3} s", elapsed.as_secs_f64());
    println!("wrote            {}", out.display());
    if let Some(n) = show {
        print_partials(&analysis.book, n);
    }
    Ok(())
}

fn print_diagnostics(d: &PartialDiagnostics) {
    println!("input atoms      {}", d.input_atoms);
    println!("observations     {}", d.observations);
    if d.skipped.total() > 0 {
        println!(
            "skipped          {} (no energy {}, bad frequency {}, bad shape {})",
            d.skipped.total(),
            d.skipped.no_energy,
            d.skipped.bad_frequency,
            d.skipped.bad_shape
        );
    }
    if d.outside_grid > 0 {
        println!("outside grid     {}", d.outside_grid);
    }
    println!("grid             {} frames x {} bins, {} peaks", d.frames, d.bins, d.peaks);
    println!("candidate ridges {}", d.candidate_ridges);
    println!("  rejected short {}", d.rejected_short);
    println!("  rejected sparse {}", d.rejected_sparse);
    println!("partials         {}", d.accepted_partials);
    let pct = |n: usize| 100.0 * n as f64 / d.observations.max(1) as f64;
    println!(
        "supporting atoms {} ({:.1}%), unsupported {} ({:.1}%)",
        d.supporting_atoms,
        pct(d.supporting_atoms),
        d.unsupported_atoms,
        pct(d.unsupported_atoms)
    );
}

fn print_provenance(pb: &PartialBook) {
    let m = &pb.metadata;
    println!(
        "partial book v{}  rmp-structure {}  rmp {}",
        m.format_version, m.version, m.rmp_version
    );
    if let Some(src) = &m.source_book {
        println!("source book      {}", src.display());
    }
    println!("sample rate      {} Hz", m.sample_rate);
    println!("partials         {}", pb.partials.len());
}

fn print_partials(pb: &PartialBook, n: usize) {
    let sr = pb.metadata.sample_rate;
    println!();
    println!(
        "{:>5} {:>8} {:>8} {:>10} {:>7} {:>7} {:>8} {:>9} {:>5} {:>5}",
        "id", "start s", "dur s", "f Hz", "std c", "persist", "energy%", "signif", "pts", "atoms"
    );
    for p in pb.by_significance().into_iter().take(n) {
        println!(
            "{:>5} {:>8.3} {:>8.3} {:>10.2} {:>7.1} {:>7.2} {:>8.3} {:>9.2e} {:>5} {:>5}",
            p.id.0,
            p.start_samples as f64 / sr,
            p.duration_samples() as f64 / sr,
            p.geometric_mean_frequency_hz,
            p.frequency_std_cents,
            p.persistence,
            100.0 * p.normalized_energy,
            p.significance,
            p.frequency.points.len(),
            p.supporting_atoms.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_output_sits_beside_the_book_under_its_own_name() {
        for (book, want) in [
            ("dir/p.json.gz", "dir/p.partials.json.gz"),
            ("p.json", "p.partials.json.gz"),
            ("a.b.toml", "a.b.partials.json.gz"),
            ("book", "book.partials.json.gz"),
        ] {
            assert_eq!(default_output(Path::new(book)), PathBuf::from(want), "{book}");
        }
    }
}
