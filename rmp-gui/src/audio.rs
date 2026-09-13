//! Playing a rendered soundfile back, through rodio.
//!
//! Deliberately the smallest thing that answers "did that sound right": one device, one track at a
//! time, start from the beginning. Everything a real transport would add — a position, a scrub bar,
//! pause — is absent because none of it is needed to check a render, and each would be state to
//! keep true against a file that can be rewritten underneath it.
//!
//! **The device is opened on first use, not at startup.** Opening it eagerly would make a machine
//! with no sound card, or a busy exclusive-mode device, fail at launch over a feature the user may
//! never touch — and analysis does not need a speaker. A failure here is a line in the log, not a
//! refusal to run.
//!
//! **`MixerDeviceSink` must outlive playback.** rodio stops the moment it is dropped, so it is held
//! here rather than in the function that opened it, which is the one thing about this API that
//! bites if you do not know it.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub struct Audio {
    /// Held, not used directly: dropping it silences everything.
    _device: rodio::MixerDeviceSink,
    player: rodio::Player,
    /// What was last handed to `play`. Kept so a tab can ask whether *its* render is the one
    /// sounding — with one device and several tabs, "something is playing" is not the same
    /// question as "this tab is playing".
    current: Option<std::path::PathBuf>,
}

impl Audio {
    /// Open the default output device.
    pub fn open() -> Result<Self, String> {
        let device = rodio::DeviceSinkBuilder::open_default_sink()
            .map_err(|e| format!("no audio output: {e}"))?;
        let player = rodio::Player::connect_new(device.mixer());
        Ok(Self { _device: device, player, current: None })
    }

    /// Play `path` from the start, replacing whatever was playing.
    ///
    /// Replacing rather than mixing: two renders of the same excerpt on top of each other is not a
    /// comparison, it is a mess. Pressing play again restarts, which is also how you listen to the
    /// same passage twice.
    pub fn play(&mut self, path: &Path) -> Result<(), String> {
        self.player.stop();
        let file = File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
        let source = rodio::Decoder::try_from(BufReader::new(file))
            .map_err(|e| format!("decoding {}: {e}", path.display()))?;
        self.player.append(source);
        self.player.play();
        self.current = Some(path.to_path_buf());
        Ok(())
    }

    pub fn stop(&mut self) {
        self.player.stop();
        self.current = None;
    }

    /// Whether a sound is going.
    ///
    /// Both halves are needed, and each covers what the other misses. `player.empty()` catches a
    /// track that reached its end on its own, which nothing else would notice. But rodio's `stop()`
    /// only *sets a flag* — `sound_count` is decremented later by the audio thread — so `empty()`
    /// stays false for a moment after a stop, and a Stop button that lingers after the sound was
    /// told to stop is a button that appears not to work. `current`, cleared synchronously, is what
    /// makes the answer immediate in that direction.
    pub fn playing(&self) -> bool {
        self.current.is_some() && !self.player.empty()
    }

    /// The file sounding right now, or `None` when nothing is.
    pub fn playing_path(&self) -> Option<&std::path::Path> {
        self.playing().then_some(self.current.as_deref()).flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A second of quiet noise as a real WAV, written through the same code the renders go through.
    fn a_wav(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("rmp-gui-audio-{}-{name}.wav", std::process::id()));
        let samples = rmp_core::residual::pseudo_noise(48_000);
        let quiet: Vec<f32> = samples.iter().map(|s| s * 0.05).collect();
        rmp_core::audio::write_samples(&path, &quiet, 48_000.0).expect("writing the fixture");
        path
    }

    /// Needs a sound device, which a headless machine has not got, so a failure to *open* is not a
    /// failure of this test — what is under test is everything after that: decoding a real render
    /// and reporting honestly which file is sounding.
    #[test]
    fn a_rendered_file_decodes_and_reports_itself_as_playing() {
        let Ok(mut audio) = Audio::open() else {
            eprintln!("no audio device here; skipping the playback test");
            return;
        };
        assert!(!audio.playing(), "nothing plays before anything is asked for");
        assert!(audio.playing_path().is_none());

        let path = a_wav("play");
        audio.play(&path).expect("a wav rmp itself wrote must decode");
        assert!(audio.playing());
        assert_eq!(audio.playing_path(), Some(path.as_path()), "and says which file");

        audio.stop();
        assert!(!audio.playing());
        assert!(audio.playing_path().is_none(), "stopping clears the file too");
        std::fs::remove_file(&path).ok();
    }

    /// A missing or unreadable file is a line in the log, not a panic — the render it names can
    /// have been moved or deleted since.
    #[test]
    fn playing_a_file_that_is_not_there_is_an_error_not_a_panic() {
        let Ok(mut audio) = Audio::open() else { return };
        let err = audio
            .play(std::path::Path::new("/nonexistent/never-rendered.wav"))
            .expect_err("that file does not exist");
        assert!(err.contains("never-rendered.wav"), "the message should name it: {err}");
        assert!(!audio.playing());
    }
}
