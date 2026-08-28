# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rmp` ("rust matching pursuit") is an **audio analysis** engine: it decomposes a signal into FOF
atoms (Rodet's Formant Wave Function, from the CHANT synthesizer) using Matching Pursuit. Reference
work is Gribonval's thesis and Krstulovic & Gribonval, *"MPTK: Matching Pursuit Made Tractable"*
(ICASSP 2006).

None of this is discoverable from the current source — `src/lib.rs` is still the cargo template. The
only substantive code is `benches/fft.rs`, which exists because FFT-based correlation is the
engine's inner loop.

**The design is written up in `/home/plu/.claude/plans/the-purpose-of-this-replicated-moon.md`.**
Read it before implementing engine code; it carries numerically verified derivations and several
non-obvious corrections.

## Relationship to rfofs

`/home/plu/Projects/rfofs` is the sibling **synthesis** project — a mature real-time FOF synthesizer
(JACK, shared memory, a C client library). rmp is the inverse direction, and takes a path dependency
on it so that analysis atoms and synthesis atoms are bit-identical.

- `rfofs::fof` is pure math (imports only `wide` + its build-script sine LUT) and is safe to depend
  on. `engine`/`queue`/`shm`/`offline` are realtime and I/O infrastructure — not needed here.
- Render an atom with `FofState::spawn(params, sr)` then `fill_block(sr, 0, &mut buf)`. Three traps:
  `fill_block` **accumulates** (`+=`), so zero the buffer first; render each atom in **one** call
  because `decay_acc` is a running product carried across calls; and `start_sample`/`block_start`
  must both be 0 or the sub-block offset logic silently emits nothing.
- Derive envelopes and support lengths **by rendering a probe**, never by reimplementing the
  formula — rfofs's death sample depends on internal rounding that a separate formula will drift
  away from.

## Commands

```bash
cargo build --release
cargo test
cargo test <test_name>              # single test
cargo bench --bench fft             # FFTW planning = MEASURE (default)
FFTW_PLAN=patient cargo bench --bench fft
```

Benchmarks use criterion with `harness = false`, matching rfofs's convention.

## Build configuration — three things that will bite

**`src/main.rs` must contain at least `fn main() {}`.** An empty file fails the whole build with
`E0601`, including bench targets, which makes `cargo bench` look broken for an unrelated reason.

**`fftw` must keep `features = ["system"]`.** The crate's default `source` feature vendors FFTW
3.3.8 and configures it with no SIMD flags at all — the generated `config.h` has `HAVE_AVX`,
`HAVE_AVX2`, `HAVE_SSE2` all `#undef`, producing scalar code roughly **3× slower** than the system
library. Reverting to defaults silently destroys FFTW's performance and makes any comparison against
realfft meaningless. Requires `libfftw3-dev`.

**`.cargo/config.toml` sets `-C target-cpu=native`.** Keep it, and note rfofs sets the same flag in
its own config — that setting does *not* propagate across a path dependency. rfofs's `wide::f32x8`
SIMD width determines which sine approximation each sample gets (a degree-9 polynomial for full
lanes, an LUT for the scalar tail), so a mismatched build changes atom values between the two
crates. An explicit `RUSTFLAGS` env var overrides `build.rustflags` rather than merging, so don't
set both.

`rustfft` is deliberately commented out in `Cargo.toml`; it still arrives transitively via `realfft`.

## The FFT benchmark

`benches/fft.rs` compares realfft against FFTW for real-to-complex f32 transforms at
1024/2048/4096/8192/16384/32768 — the range FOF atom lengths land in, since support is `≈ 6.9·sr/α`.

- `FFTW_PLAN` selects `measure` (default) or `patient`; an unrecognized value panics. The mode is
  baked into the benchmark id (`fftw3-measure` / `fftw3-patient`) so criterion keeps separate
  baselines instead of one overwriting the other.
- After the run it prints a comparison table. This does **not** re-time anything — it reads back the
  mean from criterion's own `target/criterion/<group>/<variant>/<N>/new/estimates.json`. A single run
  only exercises one FFTW mode, so columns can come from **different runs**; arms never run show `—`.
- `Flag` is bitflags 2.x deriving only `Default`, so it is **not `Copy`** — build it fresh per plan
  or it moves on the first loop iteration.
- Both MEASURE and PATIENT overwrite the arrays while planning, so fill input buffers *after*
  creating the plan.

## Measurement discipline

Run-to-run variance on this machine reaches ~5% (realfft is markedly less reproducible than FFTW —
17% spread at N=8192 across runs, against FFTW's 3.6%). **Treat sub-5% differences as unresolved**
unless they reproduce across several runs. When a result matters, run it repeatedly and compare
medians rather than trusting one `cargo bench` line — criterion's within-run confidence intervals
are much tighter than the true between-run spread and will overstate your confidence.

## Other agent configs

An OpenAI Codex config exists at `~/.codex/config.toml`. To bring anything over (MCP servers, slash
commands, subagents, skills, instructions), reply `/import` to scan and list what's importable, then
`/import --yes=<digest>` using the digest the scan prints. If `/import` isn't available on this
surface, run `claude import` from a terminal.
