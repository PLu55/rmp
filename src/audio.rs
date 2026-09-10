//! Soundfile input and output.
//!
//! Uses libsndfile (already a build requirement via rfofs), so WAV, AIFF and FLAC all work as
//! input. Output is 32-bit float WAV by default — the residual routinely contains values that would
//! clip or quantise badly in a fixed-point format. [`write_pcm24`] exists for `rmpsynth`, whose
//! output is a finished mix rather than a residual.

use crate::signal::Signal;
use sndfile::{
    Endian, MajorFormat, OpenOptions, ReadOptions, SndFile, SndFileIO, SubtypeFormat, WriteOptions,
};
use std::path::Path;

#[derive(Debug)]
pub enum AudioError {
    Open(String),
    Read(String),
    Write(String),
    Empty(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open(m) => write!(f, "could not open {m}"),
            Self::Read(m) => write!(f, "could not read {m}"),
            Self::Write(m) => write!(f, "could not write {m}"),
            Self::Empty(m) => write!(f, "{m} contains no samples"),
        }
    }
}

impl std::error::Error for AudioError {}

/// What was read, and what had to be done to it.
pub struct Input {
    pub signal: Signal,
    /// Channel count of the file on disk, before any downmix.
    pub channels: usize,
}

impl Input {
    pub fn was_downmixed(&self) -> bool {
        self.channels > 1
    }
}

/// Read a soundfile, downmixing to mono.
///
/// Multi-channel input is averaged. That is a lossy choice — out-of-phase content between channels
/// partially cancels, so a stereo file can decompose worse than either channel alone would — and it
/// is reported by [`Input::was_downmixed`] so the caller can say so.
///
/// The seam for real multi-channel support is here: return one [`Signal`] per channel instead of
/// folding them, and let the caller analyse each. Nothing downstream of this function assumes mono,
/// so that change stays local.
pub fn read(path: impl AsRef<Path>) -> Result<Input, AudioError> {
    let path = path.as_ref();
    let name = path.display().to_string();

    let mut snd: SndFile = OpenOptions::ReadOnly(ReadOptions::Auto)
        .from_path(path)
        .map_err(|e| AudioError::Open(format!("{name}: {e:?}")))?;

    let channels = snd.get_channels();
    let sample_rate = snd.get_samplerate() as f32;

    let interleaved: Vec<f32> = <SndFile as SndFileIO<f32>>::read_all_to_vec(&mut snd)
        .map_err(|_| AudioError::Read(name.clone()))?;

    if interleaved.is_empty() {
        return Err(AudioError::Empty(name));
    }

    // Downmixed in place, into the front of the buffer it reads from, and then truncated. The
    // obvious `collect()` allocates a second buffer and holds both until the interleaved one is
    // dropped — an extra `channels/(channels+1)` of the file's samples at the peak, which for a
    // nine-minute stereo file is 100 MB that nothing ever reads twice. The read is strictly ahead
    // of the write (`frame*channels >= frame`), so the aliasing is safe by index.
    let samples = if channels <= 1 {
        interleaved
    } else {
        let mut interleaved = interleaved;
        let scale = 1.0 / channels as f32;
        let frames = interleaved.len() / channels;
        for frame in 0..frames {
            let sum: f32 = interleaved[frame * channels..][..channels].iter().sum();
            interleaved[frame] = sum * scale;
        }
        interleaved.truncate(frames);
        interleaved.shrink_to_fit();
        interleaved
    };

    Ok(Input {
        signal: Signal::new(samples, sample_rate),
        channels,
    })
}

/// Write a mono signal as 32-bit float WAV.
pub fn write(path: impl AsRef<Path>, signal: &Signal) -> Result<(), AudioError> {
    write_samples(path, &signal.samples, signal.sample_rate)
}

/// [`write`] for samples that are not already owned by a [`Signal`].
///
/// The pursuit's residual is a borrowed slice of its working buffer, and wrapping it in a `Signal`
/// to write it copied the whole thing — a second full-length `f32` buffer live at the one moment
/// the resynthesis is also resident.
pub fn write_samples(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: f32,
) -> Result<(), AudioError> {
    write_encoded(path, samples, sample_rate, SubtypeFormat::FLOAT)
}

/// Write a mono signal as 24-bit PCM WAV.
///
/// Offered for the residual synthesiser, whose output is a finished mix rather than a residual and
/// so has a defined full scale. Anything past `[-1, 1)` wraps in a fixed-point format, which is why
/// [`crate::synth`] measures and reports overs before it gets here.
pub fn write_pcm24(path: impl AsRef<Path>, signal: &Signal) -> Result<(), AudioError> {
    write_encoded(path, &signal.samples, signal.sample_rate, SubtypeFormat::PCM_24)
}

fn write_encoded(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: f32,
    subtype: SubtypeFormat,
) -> Result<(), AudioError> {
    let path = path.as_ref();
    let name = path.display().to_string();

    let mut snd = OpenOptions::WriteOnly(WriteOptions::new(
        MajorFormat::WAV,
        subtype,
        Endian::File,
        sample_rate as usize,
        1,
    ))
    .from_path(path)
    .map_err(|e| AudioError::Open(format!("{name}: {e:?}")))?;

    <SndFile as SndFileIO<f32>>::write_from_slice(&mut snd, samples)
        .map_err(|_| AudioError::Write(name))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rmp_audio_test_{name}.wav"))
    }

    #[test]
    fn round_trips_a_mono_signal() {
        let path = tmp("mono");
        let want = Signal::new((0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect(), 48_000.0);
        write(&path, &want).unwrap();

        let got = read(&path).unwrap();
        assert_eq!(got.channels, 1);
        assert!(!got.was_downmixed());
        assert_eq!(got.signal.sample_rate, 48_000.0);
        assert_eq!(got.signal.len(), want.len());
        for (a, b) in want.samples.iter().zip(&got.signal.samples) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn preserves_sample_rate_other_than_48k() {
        let path = tmp("rate");
        write(&path, &Signal::new(vec![0.1; 100], 44_100.0)).unwrap();
        assert_eq!(read(&path).unwrap().signal.sample_rate, 44_100.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        // Written by hand as interleaved stereo, since `write` only emits mono.
        let path = tmp("stereo");
        let mut snd = OpenOptions::WriteOnly(WriteOptions::new(
            MajorFormat::WAV,
            SubtypeFormat::FLOAT,
            Endian::File,
            48_000,
            2,
        ))
        .from_path(&path)
        .unwrap();
        // Left = 1.0, right = 0.0 -> mono 0.5.
        let interleaved: Vec<f32> = (0..200).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        <SndFile as SndFileIO<f32>>::write_from_slice(&mut snd, &interleaved).unwrap();
        drop(snd);

        let got = read(&path).unwrap();
        assert_eq!(got.channels, 2);
        assert!(got.was_downmixed());
        assert_eq!(got.signal.len(), 100);
        for &s in &got.signal.samples {
            assert!((s - 0.5).abs() < 1e-6, "expected 0.5, got {s}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn out_of_phase_stereo_cancels() {
        // Documents the cost of downmixing: this file is not silent, but its mono fold is.
        let path = tmp("antiphase");
        let mut snd = OpenOptions::WriteOnly(WriteOptions::new(
            MajorFormat::WAV,
            SubtypeFormat::FLOAT,
            Endian::File,
            48_000,
            2,
        ))
        .from_path(&path)
        .unwrap();
        let interleaved: Vec<f32> = (0..200).map(|i| if i % 2 == 0 { 0.7 } else { -0.7 }).collect();
        <SndFile as SndFileIO<f32>>::write_from_slice(&mut snd, &interleaved).unwrap();
        drop(snd);

        let got = read(&path).unwrap();
        assert!(got.signal.energy() < 1e-9, "energy {}", got.signal.energy());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        assert!(read("/nonexistent/rmp/definitely_not_here.wav").is_err());
    }
}
