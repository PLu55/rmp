use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group,
};
use fftw::{
    array::AlignedVec,
    plan::{R2CPlan, R2CPlan32},
    types::{Flag, c32},
};
use realfft::RealFftPlanner;

const SIZES: [usize; 6] = [1024, 2048, 4096, 8192, 16384, 32768];

const GROUP: &str = "fft r2c f32";

/// Every arm the summary table knows how to report on. A run only produces one
/// of the two FFTW modes, so the table fills in whichever are on disk.
const VARIANTS: [&str; 3] = ["realfft", "fftw3-measure", "fftw3-patient"];

fn signal(i: usize) -> f32 {
    (i as f32 * 0.001).sin()
}

/// FFTW planning effort, selected with `FFTW_PLAN=measure|patient`
/// (default `measure`). PATIENT searches a wider space of algorithms at plan
/// time, so it plans much more slowly but may transform faster.
fn fftw_plan_flag() -> (Flag, &'static str) {
    match std::env::var("FFTW_PLAN")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "" | "measure" => (Flag::MEASURE, "measure"),
        "patient" => (Flag::PATIENT, "patient"),
        other => panic!(
            "unknown FFTW_PLAN {other:?}, expected \"measure\" or \"patient\""
        ),
    }
}

fn bench_r2c(c: &mut Criterion) {
    let mut group = c.benchmark_group(GROUP);
    let mut planner = RealFftPlanner::<f32>::new();

    for n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        // realfft: hoist the scratch buffer out of the loop, since `process`
        // allocates a fresh one on every call.
        let fft = planner.plan_fft_forward(n);
        let mut rf_input = fft.make_input_vec();
        let mut rf_output = fft.make_output_vec();
        let mut rf_scratch = fft.make_scratch_vec();

        for (i, x) in rf_input.iter_mut().enumerate() {
            *x = signal(i);
        }

        group.bench_function(BenchmarkId::new("realfft", n), |b| {
            b.iter(|| {
                fft.process_with_scratch(
                    black_box(&mut rf_input),
                    black_box(&mut rf_output),
                    black_box(&mut rf_scratch),
                )
                .unwrap();

                black_box(&rf_output);
            });
        });

        // fftw3: both MEASURE and PATIENT overwrite the arrays while planning,
        // so fill the input afterwards.
        let mut fw_input = AlignedVec::<f32>::new(n);
        let mut fw_output = AlignedVec::<c32>::new(n / 2 + 1);
        // `Flag` is not `Copy`, so build it fresh for each plan.
        let (fftw_flag, fftw_label) = fftw_plan_flag();
        let mut plan: R2CPlan32 = R2CPlan32::aligned(&[n], fftw_flag)
            .expect("FFTW plan creation failed");

        for (i, x) in fw_input.iter_mut().enumerate() {
            *x = signal(i);
        }

        group.bench_function(BenchmarkId::new(format!("fftw3-{fftw_label}"), n), |b| {
            b.iter(|| {
                plan.r2c(
                    black_box(&mut fw_input),
                    black_box(&mut fw_output),
                )
                .unwrap();

                black_box(&fw_output);
            });
        });
    }

    group.finish();
}

/// Mean time in nanoseconds that criterion last recorded for one arm, or
/// `None` if that arm has never been run.
fn stored_mean_ns(variant: &str, n: usize) -> Option<f64> {
    let home = std::env::var("CRITERION_HOME")
        .unwrap_or_else(|_| "target/criterion".to_string());
    let path = format!("{home}/{GROUP}/{variant}/{n}/new/estimates.json");
    let text = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    json["mean"]["point_estimate"].as_f64()
}

fn fmt_time(ns: f64) -> String {
    if ns < 1000.0 {
        format!("{ns:.1} ns")
    } else {
        format!("{:.3} \u{b5}s", ns / 1000.0)
    }
}

/// Print a comparison of every arm criterion has results for. Figures are the
/// most recent stored run of each arm, so the FFTW columns are only filled in
/// once that planning mode has actually been benchmarked.
fn print_summary_table() {
    let rows: Vec<(usize, Vec<Option<f64>>)> = SIZES
        .iter()
        .map(|&n| (n, VARIANTS.iter().map(|v| stored_mean_ns(v, n)).collect()))
        .collect();

    if rows.iter().all(|(_, m)| m.iter().all(Option::is_none)) {
        return;
    }

    let widths: Vec<usize> = VARIANTS
        .iter()
        .enumerate()
        .map(|(i, v)| {
            rows.iter()
                .filter_map(|(_, m)| m[i].map(|ns| fmt_time(ns).chars().count()))
                .chain(std::iter::once(v.chars().count()))
                .max()
                .unwrap_or(1)
        })
        .collect();

    println!("\n{GROUP} \u{2014} mean time per transform, lower is better\n");

    print!("{:>7}", "N");
    for (v, w) in VARIANTS.iter().zip(&widths) {
        print!("  {v:>w$}");
    }
    println!("  {:>16}", "fastest");

    print!("{:->7}", "");
    for w in &widths {
        print!("  {:->w$}", "");
    }
    println!("  {:->16}", "");

    for (n, means) in &rows {
        print!("{n:>7}");
        for (m, w) in means.iter().zip(&widths) {
            match m {
                Some(ns) => print!("  {:>w$}", fmt_time(*ns)),
                None => print!("  {:>w$}", "\u{2014}"),
            }
        }

        // Name the winner, and by how much it beats the runner-up.
        let mut ranked: Vec<(&str, f64)> = VARIANTS
            .iter()
            .zip(means)
            .filter_map(|(v, m)| m.map(|ns| (*v, ns)))
            .collect();
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));

        match ranked.as_slice() {
            [] => println!("  {:>16}", "\u{2014}"),
            [(only, _)] => println!("  {only:>16}"),
            [(best, bt), (_, next), ..] => {
                let margin = (next - bt) / next * 100.0;
                println!("  {best:>16}  {margin:>5.1}% ahead");
            }
        }
    }

    println!(
        "\nRun both `cargo bench` and `FFTW_PLAN=patient cargo bench` to fill \
         every column."
    );
}

criterion_group!(benches, bench_r2c);

fn main() {
    benches();
    Criterion::default().configure_from_args().final_summary();
    print_summary_table();
}
