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


pub struct Audio {
    /// Held, not used directly: dropping it silences everything.
    _device: rodio::MixerDeviceSink,
    player: rodio::Player,
    /// Whether a sound was started and not yet stopped. See [`Audio::playing`] for why this is
    /// tracked here rather than asked of the player.
    active: bool,
}

impl Audio {
    /// Open the default output device.
    pub fn open() -> Result<Self, String> {
        let device = rodio::DeviceSinkBuilder::open_default_sink()
            .map_err(|e| format!("no audio output: {e}"))?;
        let player = rodio::Player::connect_new(device.mixer());
        Ok(Self { _device: device, player, active: false })
    }

    /// Play samples already in memory.
    ///
    /// What Play uses: the mix it wants exists only as a buffer, and writing it to a file first
    /// would mean naming and saving something before you could hear it. Which *tab* the samples
    /// came from is the window's business, not this module's — one device serves all of them.
    pub fn play_samples(&mut self, signal: &rmp_core::signal::Signal) {
        let rate = std::num::NonZero::new(signal.sample_rate as u32)
            .unwrap_or(std::num::NonZero::new(48_000).expect("48000 is not zero"));
        let mono = std::num::NonZero::new(1u16).expect("1 is not zero");
        let buffer = rodio::buffer::SamplesBuffer::new(mono, rate, signal.samples.clone());
        self.start(buffer);
    }

    fn start<S>(&mut self, source: S)
    where
        S: rodio::Source + Send + 'static,
    {
        self.player.stop();
        // A stopped player will not take a new source: `stop` latches a flag that the audio thread
        // clears when it drains. A fresh one is the reliable way to start again, and the old one is
        // dropped here.
        self.player = rodio::Player::connect_new(self._device.mixer());
        self.player.append(source);
        self.player.play();
        self.active = true;
    }

    pub fn stop(&mut self) {
        self.player.stop();
        self.active = false;
    }

    /// Whether a sound is going.
    ///
    /// Both halves are needed, and each covers what the other misses. `player.empty()` catches a
    /// track that reached its end on its own, which nothing else would notice. But rodio's `stop()`
    /// only *sets a flag* — `sound_count` is decremented later by the audio thread — so `empty()`
    /// stays false for a moment after a stop, and a Stop button that lingers after the sound was
    /// told to stop is a button that appears not to work. `active`, cleared synchronously, is what
    /// makes the answer immediate in that direction.
    pub fn playing(&self) -> bool {
        self.active && !self.player.empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A second of quiet noise, as a `Signal` — which is the form Play hands over.
    fn a_signal() -> rmp_core::signal::Signal {
        let samples: Vec<f32> =
            rmp_core::residual::pseudo_noise(48_000).iter().map(|s| s * 0.05).collect();
        rmp_core::signal::Signal::new(samples, 48_000.0)
    }

    /// Needs a sound device, which a headless machine has not got, so a failure to *open* is not a
    /// failure of this test — what is under test is everything after that.
    #[test]
    fn samples_play_and_stopping_is_noticed_at_once() {
        let Ok(mut audio) = Audio::open() else {
            eprintln!("no audio device here; skipping the playback test");
            return;
        };
        assert!(!audio.playing(), "nothing plays before anything is asked for");

        audio.play_samples(&a_signal());
        assert!(audio.playing());

        // The point of tracking `active` ourselves: rodio's `stop` only latches a flag, and
        // `player.empty()` stays false until the audio thread drains. A Stop button that lingers
        // after the sound was told to stop looks broken.
        audio.stop();
        assert!(!audio.playing(), "stopping is visible immediately, not eventually");
    }

    /// Starting a second sound after a stop must work. rodio will not take a new source on a
    /// stopped player, which is why `start` connects a fresh one.
    #[test]
    fn a_second_sound_plays_after_the_first_was_stopped() {
        let Ok(mut audio) = Audio::open() else {
            eprintln!("no audio device here; skipping the playback test");
            return;
        };
        audio.play_samples(&a_signal());
        audio.stop();
        audio.play_samples(&a_signal());
        assert!(audio.playing(), "the player was stopped and never came back");
    }
}
