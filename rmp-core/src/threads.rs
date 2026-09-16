//! Running on the cores that make an analysis faster, which is not all of them.
//!
//! # Why not one thread per hardware thread
//!
//! Rayon's default is one worker per hardware thread, anywhere the scheduler likes. On a hybrid CPU
//! with hyperthreading that is measurably slower than using fewer, better-placed threads. The
//! pursuit's parallel stages are memory-bound rather than compute-bound — a 337,500-point transform
//! walks about 9 MB of buffers — and refinement's batches wait for their slowest trial. Measured on
//! an i7-13700KF, 8 performance cores with hyperthreading and 8 efficiency cores, 24 hardware
//! threads:
//!
//! - **Transform throughput peaks at 10–12 concurrent 337,500-point correlations (4.9x one) and falls
//!   to 3.2x at 24.** Hyperthread siblings share a core's caches, and the efficiency cores share an
//!   L2 per cluster of four, so past the physical performance cores each thread added slows the
//!   others more than it contributes. Transforms of 86,400 points and below keep scaling to about
//!   10.5x, but the large blocks are where the refresh spends its time.
//! - **A pool of 8 was the fastest end to end**, in interleaved runs: 3 s of piano 3.6 s against
//!   4.2 s at 24, 10 s of piano 32.2 s against 35.7 s, and the Gaussian, mixed and `zyklus`
//!   configurations 6–16% faster. 10, 12 and 16 all landed between the two.
//! - **Where the thread driving the pursuit runs matters as much as the pool.** It does refinement's
//!   serial half, the subtraction and the scans. Left to the scheduler while eight workers hold the
//!   performance cores, it lands on an efficiency core: keeping it and the workers on the
//!   performance cores' CPUs took a further 10–13% (3 s of piano 4.0 s → 3.5 s, the Gaussian run
//!   7.3 → 6.6, `zyklus` 5.1 → 4.6). Pinning the workers *without* the driver was worse than
//!   nothing on the Gaussian run, 7.8 s against 7.3, for exactly that reason.
//!
//! So the default pool is **one thread per physical core of the fastest class**, its workers kept to
//! that class's CPUs, and a front end calls [`prefer_fast_cores`] on the thread that drives an
//! analysis. None of it can change a result: every parallel stage is bit-identical at any thread
//! count and on any core. Setting `RAYON_NUM_THREADS` turns all of it off, pool size and placement
//! both, and leaves threads to the caller.
//!
//! **The CPUs are a set, not one per worker.** Pinning each worker to its own core measured another
//! 3–5%, but two `rmp` processes run side by side would then pin their workers to the same eight
//! CPUs and leave the rest idle. Kept to a set, the scheduler still balances within it.
//!
//! # Detection
//!
//! Linux only, from sysfs: the CPUs this process may run on come from `/proc/self/status` — so a
//! `taskset` mask is respected — a CPU's physical core from `core_cpus_list`, which hyperthread
//! siblings share, and its class from `cpu_capacity` or `cpuinfo_max_freq`, whichever actually tells
//! the CPUs apart. That is not a formality: the 13700KF's kernel reports a capacity of 1024 for every
//! CPU, efficiency cores included, and trusting it counted 16 fast cores. Its frequencies do separate
//! them, 5.3 and 5.4 GHz for the performance cores (two are favoured) against 4.2 for the efficiency
//! cores, and a CPU within 10% of the fastest counts as fast. Anything unreadable leaves rayon's own
//! default in place. Elsewhere nothing here does anything.

use std::fs;

/// Fraction of the fastest CPU's speed a CPU must reach to count as fast.
const FAST_FRACTION: f64 = 0.9;

/// One schedulable CPU as detection sees it.
#[derive(Clone, Debug, PartialEq)]
struct Cpu {
    /// The CPU number the kernel uses.
    id: usize,
    /// Identifies the physical core; hyperthread siblings share it.
    core: String,
    /// `cpu_capacity`, where the kernel provides it.
    capacity: Option<u64>,
    /// `cpuinfo_max_freq`, kHz, where the kernel provides it.
    max_freq: Option<u64>,
}

/// Build the global pool with one thread per fast physical core, its workers kept to those cores'
/// CPUs, and return the pool size in effect.
///
/// Leaves the pool alone when `RAYON_NUM_THREADS` is set, when the pool already exists, or when
/// the cores cannot be told apart. Call it before anything touches rayon. It does not move the
/// calling thread; see [`prefer_fast_cores`].
pub fn configure_pool() -> usize {
    if !overridden() && let Some(cores) = fast_cores() {
        let cpus: Vec<usize> = cores.concat();
        // Fails only if the pool was built already, which leaves it as it was.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(cores.len())
            .start_handler(move |_| restrict_to(&cpus))
            .build_global();
    }
    pool_size()
}

/// Keep the calling thread on the fast cores' CPUs, for a thread that drives an analysis.
///
/// The command line calls it on its main thread. A GUI calls it on the threads it starts work on,
/// and not on its UI thread, which has no business competing for those cores.
pub fn prefer_fast_cores() {
    if !overridden() && let Some(cores) = fast_cores() {
        restrict_to(&cores.concat());
    }
}

/// Threads in the global pool — what [`configure_pool`] settled on, or rayon's own default.
pub fn pool_size() -> usize {
    rayon::current_num_threads()
}

/// Physical cores of the fastest class among the CPUs this process may run on, if that can be told.
pub fn fast_physical_cores() -> Option<usize> {
    fast_cores().map(|cores| cores.len())
}

/// `RAYON_NUM_THREADS` is set, so threads are the caller's business.
fn overridden() -> bool {
    std::env::var_os("RAYON_NUM_THREADS").is_some_and(|v| !v.is_empty())
}

/// The CPUs of each fast physical core this process may run on.
fn fast_cores() -> Option<Vec<Vec<usize>>> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let allowed = status.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:"))?;
    let cpus: Vec<Cpu> = parse_cpu_list(allowed.trim())?
        .into_iter()
        .map(|id| {
            let dir = format!("/sys/devices/system/cpu/cpu{id}");
            let read = |p: &str| fs::read_to_string(format!("{dir}/{p}")).ok();
            let number = |p: &str| read(p).and_then(|s| s.trim().parse().ok());
            let core = read("topology/core_cpus_list")
                .or_else(|| read("topology/thread_siblings_list"))
                .map_or_else(|| id.to_string(), |s| s.trim().to_string());
            Cpu {
                id,
                core,
                capacity: number("cpu_capacity"),
                max_freq: number("cpufreq/cpuinfo_max_freq"),
            }
        })
        .collect();
    let cores = group_fast_cores(&cpus);
    (!cores.is_empty()).then_some(cores)
}

/// The CPUs within [`FAST_FRACTION`] of the fastest, grouped by physical core, in first-seen order.
///
/// Speed is `cpu_capacity` where it differs between CPUs, otherwise `cpuinfo_max_freq`, otherwise
/// the same for all: a measure that reads equal everywhere says nothing either way.
fn group_fast_cores(cpus: &[Cpu]) -> Vec<Vec<usize>> {
    let tells_apart = |get: fn(&Cpu) -> Option<u64>| {
        let values: Option<Vec<u64>> = cpus.iter().map(get).collect();
        values.filter(|v| v.iter().any(|&x| x != v[0]))
    };
    let speeds = tells_apart(|c| c.capacity)
        .or_else(|| tells_apart(|c| c.max_freq))
        .unwrap_or_else(|| vec![1; cpus.len()]);
    let Some(&fastest) = speeds.iter().max() else {
        return Vec::new();
    };

    let mut cores: Vec<(&str, Vec<usize>)> = Vec::new();
    for (cpu, &speed) in cpus.iter().zip(&speeds) {
        if (speed as f64) < FAST_FRACTION * fastest as f64 {
            continue;
        }
        match cores.iter_mut().find(|(core, _)| *core == cpu.core) {
            Some((_, ids)) => ids.push(cpu.id),
            None => cores.push((&cpu.core, vec![cpu.id])),
        }
    }
    cores.into_iter().map(|(_, ids)| ids).collect()
}

/// Restrict the calling thread to `cpus`. Best effort: a refusal leaves it where it was.
#[cfg(target_os = "linux")]
fn restrict_to(cpus: &[usize]) {
    // SAFETY: `cpu_set_t` is a plain bitmask for which all-zero is the empty set, `CPU_SET` only
    // writes bits inside it (ids past its capacity are skipped first), and pid 0 names this thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let capacity = 8 * std::mem::size_of::<libc::cpu_set_t>();
        for &cpu in cpus.iter().filter(|&&c| c < capacity) {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn restrict_to(_cpus: &[usize]) {}

/// Parse a kernel CPU list such as `0-3,8,10-11`. `None` if any part is malformed.
fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in list.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.parse().ok()?, b.parse().ok()?);
                if a > b {
                    return None;
                }
                out.extend(a..=b);
            }
            None => out.push(part.parse().ok()?),
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A CPU as x86 reports it: every capacity equal, the class only in the frequency.
    fn cpu(id: usize, core: &str, max_freq: u64) -> Cpu {
        Cpu { id, core: core.to_string(), capacity: Some(1024), max_freq: Some(max_freq) }
    }

    #[test]
    fn cpu_lists_parse_as_the_kernel_writes_them() {
        assert_eq!(parse_cpu_list("0-3,8,10-11"), Some(vec![0, 1, 2, 3, 8, 10, 11]));
        assert_eq!(parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(parse_cpu_list("0-23"), Some((0..24).collect()));
        assert_eq!(parse_cpu_list(""), Some(vec![]));
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("a-b"), None);
    }

    /// The machine the rule was measured on: 8 hyperthreaded performance cores, two of them
    /// favoured at 5.4 GHz, and 8 efficiency cores at 4.2 GHz. Eight cores, sixteen CPUs.
    #[test]
    fn a_hybrid_cpu_groups_its_performance_cores_by_sibling() {
        let mut cpus = Vec::new();
        for core in 0..8 {
            let freq = if core == 4 || core == 5 { 5_400_000 } else { 5_300_000 };
            let siblings = format!("{}-{}", 2 * core, 2 * core + 1);
            cpus.push(cpu(2 * core, &siblings, freq));
            cpus.push(cpu(2 * core + 1, &siblings, freq));
        }
        for e in 16..24 {
            cpus.push(cpu(e, &e.to_string(), 4_200_000));
        }
        let cores = group_fast_cores(&cpus);
        assert_eq!(cores.len(), 8);
        assert_eq!(cores[0], vec![0, 1]);
        assert_eq!(cores.concat(), (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn an_smt_machine_counts_physical_cores_and_a_plain_one_counts_cpus() {
        let smt: Vec<Cpu> = (0..16).map(|n| cpu(n, &format!("{}-{}", n & !1, n | 1), 100)).collect();
        assert_eq!(group_fast_cores(&smt).len(), 8);
        let plain: Vec<Cpu> = (0..6).map(|n| cpu(n, &n.to_string(), 100)).collect();
        assert_eq!(group_fast_cores(&plain).len(), 6);
    }

    /// ARM reports the class in the capacity and may leave the frequencies equal or absent.
    #[test]
    fn a_capacity_that_varies_decides_the_class() {
        let big_little: Vec<Cpu> = (0..8)
            .map(|id| Cpu {
                id,
                core: id.to_string(),
                capacity: Some(if id < 2 { 1024 } else { 430 }),
                max_freq: None,
            })
            .collect();
        assert_eq!(group_fast_cores(&big_little), vec![vec![0], vec![1]]);
        let unknown: Vec<Cpu> = (0..4)
            .map(|id| Cpu { id, core: id.to_string(), capacity: None, max_freq: None })
            .collect();
        assert_eq!(group_fast_cores(&unknown).len(), 4);
    }

    #[test]
    fn an_affinity_mask_limits_what_is_counted() {
        // Only the efficiency cores are allowed, so they are the fastest there is.
        let cpus: Vec<Cpu> = (16..20).map(|n| cpu(n, &n.to_string(), 4_200_000)).collect();
        assert_eq!(group_fast_cores(&cpus).len(), 4);
        assert!(group_fast_cores(&[]).is_empty());
    }
}
