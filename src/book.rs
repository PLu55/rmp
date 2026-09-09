//! The decomposition result.
//!
//! A [`Book`] is the output of a pursuit: the atoms selected, in order, with enough provenance to
//! diagnose a bad decomposition and enough parameters to replay it through rfofs.

use crate::fof::{AtomParams, Envelope, FofError};
use crate::residual::ResidualBook;
use crate::signal::snr_db;
use serde::Serialize;
use serde::de::DeserializeOwned;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use rfofs::fof::FofParams;
use std::io::{Read, Write};
use std::path::Path;

/// One selected atom, with where it came from and what it actually removed.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Selection {
    pub atom: AtomParams,
    /// Index into the dictionary's block list.
    pub block: usize,
    /// Onset in samples (the grid position before any refinement).
    pub onset: usize,
    /// Frequency bin.
    pub bin: usize,
    /// Energy the projection predicted.
    pub projected_energy: f64,
    /// Energy actually removed, measured from the rendered atom. Divergence from
    /// `projected_energy` indicates a parameter-mapping error.
    pub energy_removed: f64,
    /// Residual energy after this atom was subtracted.
    pub residual_energy: f64,
    /// Energy this atom removed after HRMP clamped its amplitude, when HRMP ran.
    ///
    /// `None` means ordinary MP: the atom was subtracted at its full projected amplitude.
    #[serde(default)]
    pub hr_score: Option<f64>,
    /// Whether refinement moved the parameters off the seed block's grid point.
    #[serde(default)]
    pub refined: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Book {
    pub selections: Vec<Selection>,
    pub initial_energy: f64,
    pub sample_rate: f32,
    /// Stochastic analysis of the final residue, when `[residual]` was enabled.
    ///
    /// A section of its own rather than anything mixed into `selections`: atoms are sparse
    /// deterministic events, residual frames are a matrix on a fixed grid, and neither ordering
    /// means anything to the other. `skip_serializing_if` keeps a book written with the stage off
    /// byte-identical to one written before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual: Option<ResidualBook>,
}

impl Book {
    pub fn new(initial_energy: f64, sample_rate: f32) -> Self {
        Self {
            selections: Vec::new(),
            initial_energy,
            sample_rate,
            residual: None,
        }
    }

    pub fn len(&self) -> usize {
        self.selections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.selections.is_empty()
    }

    /// Residual energy after the last atom, or the initial energy if none were selected.
    pub fn residual_energy(&self) -> f64 {
        self.selections
            .last()
            .map_or(self.initial_energy, |s| s.residual_energy)
    }

    pub fn snr_db(&self) -> f32 {
        snr_db(self.initial_energy, self.residual_energy())
    }

    /// SNR in dB after each atom — the convergence curve.
    pub fn snr_trace(&self) -> Vec<f32> {
        self.selections
            .iter()
            .map(|s| snr_db(self.initial_energy, s.residual_energy))
            .collect()
    }

    /// Atoms needed to first reach `target_db`, if reached.
    pub fn atoms_to_reach(&self, target_db: f32) -> Option<usize> {
        self.snr_trace()
            .iter()
            .position(|&db| db >= target_db)
            .map(|i| i + 1)
    }

    /// Render the book back to a signal — the analysis inverted.
    ///
    /// Uses the same rfofs path the pursuit subtracted with, so a book that reached N dB against
    /// its input reproduces that input to N dB here.
    pub fn resynthesize(&self, len: usize) -> Result<crate::signal::Signal, crate::fof::FofError> {
        let atoms: Vec<AtomParams> = self.selections.iter().map(|s| s.atom).collect();
        crate::signal::Signal::from_atoms(&atoms, len, self.sample_rate)
    }

    /// Samples this book occupies, from the analysis origin to the death of the last atom.
    ///
    /// This is what [`resynthesize`](Self::resynthesize) needs when there is no input signal to
    /// take the length from. Atoms with a negative `t0` began before the analysed excerpt and are
    /// clipped at the origin here exactly as `resynthesize` clips them, so a book replays in the
    /// same frame it was analysed in. Support lengths come from rendering, never from a formula —
    /// the same rule the rest of the crate follows.
    pub fn natural_len(&self) -> Result<usize, FofError> {
        let mut len = 0i64;
        for s in &self.selections {
            let env = Envelope::render(s.atom.env, self.sample_rate)?;
            len = len.max(s.atom.t0 + env.support_len() as i64);
        }
        Ok(len.max(0) as usize)
    }

    /// Replayable parameters, for rendering through rfofs.
    pub fn to_fof_params(&self, origin: u64) -> Vec<FofParams> {
        self.selections
            .iter()
            .map(|s| s.atom.to_fof_params(origin))
            .collect()
    }

    /// How often each block was selected — the best diagnostic of a mis-sized grid. Piling up at an
    /// `alpha` edge means the ladder does not reach far enough.
    pub fn block_histogram(&self, n_blocks: usize) -> Vec<usize> {
        let mut counts = vec![0; n_blocks];
        for s in &self.selections {
            if s.block < n_blocks {
                counts[s.block] += 1;
            }
        }
        counts
    }
}


/// The wire format a path names: TOML or JSON, and whether it is gzipped.
///
/// A trailing `.gz` or `.gzip` compresses, and the format is then read from the extension
/// *beneath* it: `book.json.gz` is gzipped JSON, a bare `book.gz` gzipped TOML. No extension at
/// all is TOML. This is the single definition of those rules — [`read`] and [`write`] must not
/// each grow their own, or the two binaries will disagree about what `book.gz` means.
fn format_of(path: &Path) -> Result<(bool, bool), String> {
    let gzip = matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("gz" | "gzip")
    );
    // Strip the .gz to expose the format extension. Only the extension is ever read from this, so
    // losing the directory to `file_stem` does not matter.
    let stem = path.file_stem().unwrap_or_default();
    let format_path = if gzip { Path::new(stem) } else { path };

    let json = match format_path.extension().and_then(|e| e.to_str()) {
        Some("json") => true,
        Some("toml") | None => false,
        Some(other) => {
            return Err(format!(
                "unknown book format '.{other}' — use .toml or .json, optionally with a .gz suffix"
            ));
        }
    };
    Ok((json, gzip))
}

/// Serialise the book, picking the format from the file extension.
///
/// A book is mostly repeated field names and decimal digits, so the `.gz` suffix is worth about 7×.
/// With residual analysis on it is worth a great deal more: a 48-band bank at 1 ms writes 48000
/// numbers per second of audio, which dwarfs the atom list.
pub fn write(path: &Path, book: &Book) -> Result<(), String> {
    write_doc(path, book)
}

/// Read a book back, by the same extension rules [`write`] uses.
///
/// `Selection`'s `#[serde(default)]` on `hr_score` and `refined`, and `Book`'s on `residual`, is
/// what keeps books written before those fields existed readable.
pub fn read(path: &Path) -> Result<Book, String> {
    let book: Book = read_doc(path)?;
    if let Some(r) = &book.residual {
        r.validate()
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(book)
}

/// Write any serialisable document by the same extension rules a book follows.
///
/// [`format_of`] is the single definition of those rules and this is the single user of it, so the
/// standalone residual book written by `--residual-book` cannot drift from the main one.
pub fn write_doc<T: Serialize>(path: &Path, doc: &T) -> Result<(), String> {
    let (json, gzip) = format_of(path)?;

    let text = if json {
        serde_json::to_string_pretty(doc).map_err(|e| format!("serialising book: {e}"))?
    } else {
        toml::to_string_pretty(doc).map_err(|e| format!("serialising book: {e}"))?
    };

    let bytes = if gzip {
        let mut enc = GzEncoder::new(Vec::new(), Compression::best());
        enc.write_all(text.as_bytes())
            .and_then(|()| enc.finish())
            .map_err(|e| format!("compressing book: {e}"))?
    } else {
        text.into_bytes()
    };
    std::fs::write(path, bytes).map_err(|e| format!("writing {}: {e}", path.display()))
}

/// Read any deserialisable document by the same extension rules [`write_doc`] uses.
pub fn read_doc<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let (json, gzip) = format_of(path)?;

    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let text = if gzip {
        let mut out = String::new();
        GzDecoder::new(&bytes[..])
            .read_to_string(&mut out)
            .map_err(|e| format!("decompressing {}: {e}", path.display()))?;
        out
    } else {
        String::from_utf8(bytes).map_err(|e| format!("reading {}: {e}", path.display()))?
    };

    if json {
        serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
    } else {
        toml::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fof::EnvelopeParams;

    fn sel(block: usize, residual: f64) -> Selection {
        Selection {
            atom: AtomParams {
                t0: 0,
                f: 1000.0,
                env: EnvelopeParams::new(251.0, 0.001),
                phi: 0.0,
                amp: 1.0,
            },
            block,
            onset: 0,
            bin: 10,
            projected_energy: 1.0,
            energy_removed: 1.0,
            residual_energy: residual,
            hr_score: None,
            refined: false,
        }
    }

    #[test]
    fn snr_trace_and_targets() {
        let mut b = Book::new(100.0, 48_000.0);
        b.selections.push(sel(0, 10.0)); // 10 dB
        b.selections.push(sel(1, 1.0)); // 20 dB
        b.selections.push(sel(0, 0.1)); // 30 dB

        let trace = b.snr_trace();
        assert!((trace[0] - 10.0).abs() < 1e-3);
        assert!((trace[2] - 30.0).abs() < 1e-3);
        assert!((b.snr_db() - 30.0).abs() < 1e-3);

        assert_eq!(b.atoms_to_reach(20.0), Some(2));
        assert_eq!(b.atoms_to_reach(99.0), None);
    }

    #[test]
    fn empty_book_reports_initial_energy() {
        let b = Book::new(50.0, 48_000.0);
        assert_eq!(b.residual_energy(), 50.0);
        assert!((b.snr_db() - 0.0).abs() < 1e-6);
        assert!(b.is_empty());
    }

    #[test]
    fn block_histogram_counts_selections() {
        let mut b = Book::new(1.0, 48_000.0);
        b.selections.push(sel(0, 0.5));
        b.selections.push(sel(2, 0.25));
        b.selections.push(sel(0, 0.1));
        assert_eq!(b.block_histogram(3), vec![2, 0, 1]);
    }

    /// The .gz is stripped before the format is read, the bytes on disk are a gzip member when
    /// they should be, and every combination round-trips back to the same book.
    #[test]
    fn a_gz_suffix_compresses_and_the_format_comes_from_beneath_it() {
        let dir = std::env::temp_dir().join(format!("rmp-book-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut book = Book::new(1.0, 48_000.0);
        book.selections.push(Selection {
            hr_score: Some(0.4),
            ..sel(3, 0.5)
        });

        for (name, gzipped) in [
            ("b.json", false),
            ("b.json.gz", true),
            ("b.toml", false),
            ("b.toml.gzip", true),
            ("b.gz", true),
            ("b", false),
        ] {
            let path = dir.join(name);
            write(&path, &book).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(bytes.starts_with(&[0x1f, 0x8b]), gzipped, "{name}");
            // A bare .gz falls through to TOML, the same as no extension at all.
            assert_eq!(read(&path).unwrap(), book, "{name}");
        }

        assert!(write(&dir.join("b.yaml"), &book).is_err());
        assert!(write(&dir.join("b.yaml.gz"), &book).is_err());
        assert!(read(&dir.join("b.yaml")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `natural_len` has to cover every atom's whole life and no more: rendering into a longer
    /// buffer must add nothing past it, and the samples just inside it must still be live.
    ///
    /// Also the reason a negative `t0` cannot extend it — `resynthesize` clips such an atom at the
    /// origin, so counting its pre-origin part would pad the output with silence the book does not
    /// contain.
    #[test]
    fn natural_len_is_exactly_where_the_last_atom_dies() {
        let at = |t0: i64| Selection {
            atom: AtomParams {
                t0,
                ..sel(0, 0.5).atom
            },
            ..sel(0, 0.5)
        };

        let mut b = Book::new(1.0, 48_000.0);
        assert_eq!(b.natural_len().unwrap(), 0, "an empty book occupies nothing");

        let support = Envelope::render(at(0).atom.env, b.sample_rate).unwrap().support_len();
        b.selections.push(at(-(support as i64) - 10));
        assert_eq!(b.natural_len().unwrap(), 0, "an atom entirely before the origin is clipped");

        b.selections.push(at(-200));
        assert_eq!(b.natural_len().unwrap(), support - 200);

        b.selections.push(at(5_000));
        let len = b.natural_len().unwrap();
        assert_eq!(len, support + 5_000);

        // Rendered with room to spare, the book is silent past its own length and audible inside.
        let s = b.resynthesize(len + 4_096).unwrap();
        assert!(s.samples[len..].iter().all(|&v| v == 0.0), "energy past natural_len");
        assert!(s.samples[len - 64..len].iter().any(|&v| v != 0.0), "dead before natural_len");
    }

    /// A book written before `hr_score` and `refined` existed still reads.
    #[test]
    fn missing_optional_fields_default() {
        let json = r#"{"selections":[{"atom":{"t0":0,"f":440.0,
            "env":{"alpha":251.0,"beta":0.001,"fade_level":0.001,"fade_dur":0.008},
            "phi":0.0,"amp":1.0},"block":0,"onset":0,"bin":9,
            "projected_energy":1.0,"energy_removed":1.0,"residual_energy":0.5}],
            "initial_energy":1.0,"sample_rate":48000.0}"#;
        let b: Book = serde_json::from_str(json).unwrap();
        assert_eq!(b.selections[0].hr_score, None);
        assert!(!b.selections[0].refined);
        assert_eq!(b.residual, None);
    }

    /// §29.11: with residual analysis off, the book on disk is exactly what it was before the
    /// section existed. `skip_serializing_if` is what buys this, and it is worth a test because
    /// losing it would silently rewrite every book ever produced.
    #[test]
    fn a_book_without_residual_analysis_is_unchanged() {
        let mut book = Book::new(1.0, 48_000.0);
        book.selections.push(sel(0, 0.5));
        for text in [
            serde_json::to_string_pretty(&book).unwrap(),
            toml::to_string_pretty(&book).unwrap(),
        ] {
            // `residual_energy` is a Selection field, so only the section's own key counts.
            assert!(!text.contains("\"residual\""), "{text}");
            assert!(!text.contains("[residual"), "{text}");
        }
    }

    /// §29.10 end to end: the residual section survives every format the book supports.
    #[test]
    fn the_residual_section_round_trips_in_every_format() {
        let dir = std::env::temp_dir().join(format!("rmp-residual-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut book = Book::new(1.0, 48_000.0);
        book.selections.push(sel(1, 0.25));
        book.residual = Some(
            crate::residual::analyze_residual(
                &crate::residual::pseudo_noise(4800),
                48_000.0,
                960,
                &crate::residual::ResidualAnalysisConfig {
                    enabled: true,
                    ..Default::default()
                },
            )
            .unwrap(),
        );

        for name in ["r.json", "r.json.gz", "r.toml", "r.gz"] {
            let path = dir.join(name);
            write(&path, &book).unwrap();
            assert_eq!(read(&path).unwrap(), book, "{name}");
        }

        // The standalone writer takes the same path through `format_of`.
        let alone = dir.join("bank.json.gz");
        write_doc(&alone, book.residual.as_ref().unwrap()).unwrap();
        let back: crate::residual::ResidualBook = read_doc(&alone).unwrap();
        assert_eq!(&back, book.residual.as_ref().unwrap());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A book claiming a residual section this build cannot read is refused at the door, not half
    /// interpreted.
    #[test]
    fn an_unreadable_residual_section_is_rejected() {
        let dir = std::env::temp_dir().join(format!("rmp-residual-ver-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("future.json");

        let mut book = Book::new(1.0, 48_000.0);
        book.residual = Some(crate::residual::ResidualBook {
            version: crate::residual::RESIDUAL_BOOK_VERSION + 1,
            ..crate::residual::analyze_residual(
                &[0.0; 480],
                48_000.0,
                0,
                &crate::residual::ResidualAnalysisConfig::default(),
            )
            .unwrap()
        });
        std::fs::write(&path, serde_json::to_string(&book).unwrap()).unwrap();

        let err = read(&path).unwrap_err();
        assert!(err.contains("newer than this build"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
