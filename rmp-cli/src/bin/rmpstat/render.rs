//! Turning statistics into something to look at — fixed-width text, or a chart.

use plotters::prelude::*;
use rmp_core::stats::{Histogram, Summary};
use rmp_core::tfmap::{Reference, TfMap};
use std::path::Path;

/// Width the bar column gets in a text histogram.
const BAR_WIDTH: usize = 34;

/// A fixed-width table whose column widths come from the content.
///
/// The same renderer `benches/fft.rs` uses for its summary, lifted here so a third one does not
/// get written. Cells are formatted by the caller; this only aligns them.
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    /// Columns to left-align. Everything else is right-aligned, which is what numbers want.
    left: Vec<usize>,
}

impl Table {
    pub fn new(headers: &[&str]) -> Self {
        Self {
            headers: headers.iter().map(|h| h.to_string()).collect(),
            rows: Vec::new(),
            left: Vec::new(),
        }
    }

    pub fn left_align(mut self, cols: &[usize]) -> Self {
        self.left = cols.to_vec();
        self
    }

    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }

    pub fn render(&self, indent: &str) -> String {
        let n = self.headers.len();
        let mut w: Vec<usize> = self.headers.iter().map(|h| h.chars().count()).collect();
        for r in &self.rows {
            for (i, c) in r.iter().enumerate().take(n) {
                w[i] = w[i].max(c.chars().count());
            }
        }

        let mut out = String::new();
        let line = |out: &mut String, cells: &[String], w: &[usize], left: &[usize]| {
            out.push_str(indent);
            for (i, c) in cells.iter().enumerate().take(w.len()) {
                if i > 0 {
                    out.push(' ');
                }
                if left.contains(&i) {
                    out.push_str(&format!("{:<width$}", c, width = w[i]));
                } else {
                    out.push_str(&format!("{:>width$}", c, width = w[i]));
                }
            }
            // Left-aligned trailing cells leave ragged whitespace behind.
            while out.ends_with(' ') {
                out.pop();
            }
            out.push('\n');
        };

        line(&mut out, &self.headers, &w, &self.left);
        out.push_str(indent);
        out.push_str(
            &w.iter()
                .map(|&x| "-".repeat(x))
                .collect::<Vec<_>>()
                .join(" "),
        );
        out.push('\n');
        for r in &self.rows {
            line(&mut out, r, &w, &self.left);
        }
        out
    }
}

/// A number in as few characters as stays readable across the ranges these quantities span.
pub fn num(x: f64) -> String {
    let a = x.abs();
    if x == 0.0 {
        "0".into()
    } else if !x.is_finite() {
        format!("{x}")
    } else if !(1e-3..1e5).contains(&a) {
        format!("{x:.3e}")
    } else if a >= 1e3 {
        format!("{x:.0}")
    } else if a >= 10.0 {
        format!("{x:.1}")
    } else if a >= 1.0 {
        format!("{x:.2}")
    } else {
        format!("{x:.4}")
    }
}

/// The order statistics as one line.
pub fn summary_line(s: &Summary) -> String {
    if s.n == 0 {
        return "  (no data)".into();
    }
    format!(
        "  n {}   min {}   p25 {}   median {}   p75 {}   max {}   mean {}",
        s.n,
        num(s.min),
        num(s.p25),
        num(s.median),
        num(s.p75),
        num(s.max),
        num(s.mean)
    )
}

/// A histogram as a table of bins with a bar column.
pub fn histogram_text(h: &Histogram) -> String {
    let mut out = format!("{}{}\n", h.quantity.label(), if h.log { "  [log]" } else { "" });
    out.push_str(&summary_line(&h.stats));
    out.push('\n');

    let peak = h.peak();
    let mut t = Table::new(&["from", "to", "weight", "%", "bar"]).left_align(&[4]);
    for (i, &c) in h.counts.iter().enumerate() {
        // Blank rather than a zero-length bar, so an empty stretch of the axis reads as empty.
        let n = if peak > 0.0 {
            ((c / peak) * BAR_WIDTH as f64).round() as usize
        } else {
            0
        };
        t.row(vec![
            num(h.edges[i]),
            num(h.edges[i + 1]),
            num(c),
            format!(
                "{:.1}",
                if h.total > 0.0 { 100.0 * c / h.total } else { 0.0 }
            ),
            "#".repeat(n),
        ]);
    }
    out.push_str(&t.render("  "));

    // Mass off the ends is reported rather than folded into the end bins, so the axis never lies
    // about what it covers.
    for (label, v) in [
        ("below range", h.below),
        ("above range", h.above),
        ("not finite", h.skipped),
    ] {
        if v > 0.0 {
            out.push_str(&format!(
                "  {label}: {} ({:.1}%)\n",
                num(v),
                100.0 * v / h.total
            ));
        }
    }
    // Atoms of a kind this quantity does not describe are outside `total` altogether, so they are
    // stated as a count rather than a share of it.
    if h.inapplicable > 0.0 {
        out.push_str(&format!(
            "  other atom kinds: {} (not described by this quantity, not in the total)\n",
            num(h.inapplicable)
        ));
    }
    out
}

// ---------------------------------------------------------------------------------------------
// charts
// ---------------------------------------------------------------------------------------------

/// SVG or PNG. Both go through the same drawing code; only the backend differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Image {
    Svg,
    Png,
}

/// Run a drawing body against whichever backend the format names.
///
/// A macro rather than a function taking a closure, because the two backends are different types
/// and one closure cannot implement `FnOnce` for both. The body is type-checked once per backend,
/// so every chart below is still written once. `SVGBackend` base64-embeds a PNG for bitmap blits,
/// which is what lets the heat map take the identical path into both formats.
macro_rules! draw_to {
    ($path:expr, $fmt:expr, $size:expr, |$root:ident| $body:block) => {{
        match $fmt {
            Image::Png => {
                let $root = BitMapBackend::new($path, $size).into_drawing_area();
                $root.fill(&WHITE).map_err(|e| e.to_string())?;
                $body
                $root.present().map_err(|e| e.to_string())
            }
            Image::Svg => {
                let $root = SVGBackend::new($path, $size).into_drawing_area();
                $root.fill(&WHITE).map_err(|e| e.to_string())?;
                $body
                $root.present().map_err(|e| e.to_string())
            }
        }
    }};
}

pub const FONT: &str = "sans-serif";
const INK: RGBColor = RGBColor(60, 110, 190);

/// Where a sans font is likely to live, in preference order.
///
/// plotters' `ab_glyph` backend has no system font discovery — it renders only fonts handed to it
/// as bytes — so one has to be found and registered before any chart is drawn. The alternative,
/// plotters' `ttf` backend, does discover fonts but pulls in font-kit and a build-time dependency
/// on `libfontconfig1-dev`, which is a system package to install for a diagnostic tool. Reading
/// one file at startup is the cheaper trade.
const FONT_CANDIDATES: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
];

/// Register a text font, so charts can be labelled.
///
/// Only needed for image output; the text reports do not care. Leaking the bytes is deliberate —
/// `register_font` wants `&'static [u8]`, the font lives as long as the process, and this runs once.
pub fn init_fonts() -> Result<(), String> {
    let path = FONT_CANDIDATES
        .iter()
        .find(|p| Path::new(p).is_file())
        .ok_or_else(|| {
            format!(
                "no text font found — looked for {}. Charts need one; the text reports do not, so \
                 `-f text` still works.",
                FONT_CANDIDATES.join(", ")
            )
        })?;
    let bytes: &'static [u8] = Box::leak(
        std::fs::read(path)
            .map_err(|e| format!("reading the font {path}: {e}"))?
            .into_boxed_slice(),
    );
    for style in [
        FontStyle::Normal,
        FontStyle::Bold,
        FontStyle::Italic,
        FontStyle::Oblique,
    ] {
        // Every style maps to the one regular face: plotters asks for whichever the caller named,
        // and an unregistered style is a hard error rather than a synthesized fallback.
        plotters::style::register_font(FONT, style, bytes)
            .map_err(|_| format!("{path} is not a usable font"))?;
    }
    Ok(())
}

/// A histogram as a bar chart.
///
/// Bars are drawn against bin *index*, with the edge values as tick labels. A log histogram is
/// already geometric in its edges, so bending the axis as well would double the transform for no
/// gain — the bins *are* the axis.
pub fn histogram_chart(
    h: &Histogram,
    path: &Path,
    fmt: Image,
    size: (u32, u32),
) -> Result<(), String> {
    let n = h.counts.len();
    let peak = h.peak().max(f64::MIN_POSITIVE);
    let title = format!(
        "{}{}   n={}",
        h.quantity.label(),
        if h.log { "  (log bins)" } else { "" },
        h.stats.n
    );

    draw_to!(path, fmt, size, |root| {
        let mut chart = ChartBuilder::on(&root)
            .caption(&title, (FONT, 18))
            .margin(12)
            .x_label_area_size(48)
            .y_label_area_size(70)
            .build_cartesian_2d(0f64..n as f64, 0f64..peak * 1.08)
            .map_err(|e| e.to_string())?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .x_labels(8.min(n).max(2))
            .x_label_formatter(&|v: &f64| {
                let i = (v.round().max(0.0) as usize).min(h.edges.len() - 1);
                num(h.edges[i])
            })
            .y_desc("weight")
            .label_style((FONT, 13))
            .draw()
            .map_err(|e| e.to_string())?;

        chart
            .draw_series(h.counts.iter().enumerate().map(|(i, &c)| {
                Rectangle::new([(i as f64 + 0.06, 0.0), (i as f64 + 0.94, c)], INK.filled())
            }))
            .map_err(|e| e.to_string())?;
    })
}

/// The convergence curve: SNR in dB against atom index.
pub fn snr_chart(trace: &[f32], path: &Path, fmt: Image, size: (u32, u32)) -> Result<(), String> {
    if trace.is_empty() {
        return Err("the book has no atoms to plot".into());
    }
    let finite: Vec<(f64, f64)> = trace
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (i as f64 + 1.0, v as f64))
        .collect();
    if finite.is_empty() {
        return Err("the convergence trace has no finite values".into());
    }
    let hi = finite.iter().map(|p| p.1).fold(f64::MIN, f64::max).max(1.0);
    let lo = finite.iter().map(|p| p.1).fold(f64::MAX, f64::min).min(0.0);
    let n = trace.len() as f64;

    draw_to!(path, fmt, size, |root| {
        let mut chart = ChartBuilder::on(&root)
            .caption("convergence", (FONT, 18))
            .margin(12)
            .x_label_area_size(48)
            .y_label_area_size(70)
            .build_cartesian_2d(0f64..n, lo..hi * 1.05)
            .map_err(|e| e.to_string())?;
        chart
            .configure_mesh()
            .x_desc("atoms")
            .y_desc("SNR (dB)")
            .label_style((FONT, 13))
            .draw()
            .map_err(|e| e.to_string())?;

        // Marks at the targets the pursuit is usually asked for, so "where did it reach 40 dB" is
        // answerable off the picture rather than only off the table.
        for target in [10.0f64, 20.0, 30.0, 40.0] {
            if target > lo && target <= hi {
                chart
                    .draw_series(std::iter::once(PathElement::new(
                        vec![(0.0, target), (n, target)],
                        RGBColor(205, 205, 205),
                    )))
                    .map_err(|e| e.to_string())?;
            }
        }
        chart
            .draw_series(LineSeries::new(finite.iter().copied(), INK.stroke_width(2)))
            .map_err(|e| e.to_string())?;
    })
}

/// Black through violet and orange to a pale yellow — a spectrogram ramp anchored at true black,
/// so silence reads as empty rather than as the coloured field viridis's dark blue would give.
fn heat(u: f64) -> RGBColor {
    // Piecewise-linear through five stops, monotone in luminance, which is the property that makes
    // a level readable off the map.
    const STOPS: [(f64, f64, f64, f64); 5] = [
        (0.00, 0.0, 0.0, 0.0),
        (0.30, 40.0, 20.0, 110.0),
        (0.55, 150.0, 30.0, 110.0),
        (0.80, 240.0, 110.0, 40.0),
        (1.00, 255.0, 255.0, 210.0),
    ];
    let u = u.clamp(0.0, 1.0);
    let mut i = 0;
    while i + 2 < STOPS.len() && u > STOPS[i + 1].0 {
        i += 1;
    }
    let (u0, r0, g0, b0) = STOPS[i];
    let (u1, r1, g1, b1) = STOPS[i + 1];
    let t = ((u - u0) / (u1 - u0)).clamp(0.0, 1.0);
    RGBColor(
        (r0 + (r1 - r0) * t) as u8,
        (g0 + (g1 - g0) * t) as u8,
        (b0 + (b1 - b0) * t) as u8,
    )
}

/// The pseudo-Wigner map as a heat map.
///
/// The map is blitted as one image rather than drawn as rectangles. A grid of any useful size is
/// hundreds of thousands of cells, and an SVG carrying that many `<rect>` elements would be tens of
/// megabytes and would not open; `SVGBackend` embeds the blit as a base64 PNG instead, so both
/// formats go through this one path.
///
/// The frequency axis is drawn linearly in *bin index* and labelled with the grid's own edges. That
/// is what makes a log-frequency grid work without resampling: the pixels were laid out on the
/// grid's edges, so labelling from the same edges keeps image and axis on one scale, and a
/// geometric grid then shows equal-height octaves.
#[allow(clippy::too_many_arguments)]
pub fn wv_chart(
    map: &TfMap,
    floor_db: f32,
    reference: Reference,
    overlay: &[(f64, f32)],
    caption: &str,
    path: &Path,
    fmt: Image,
    size: (u32, u32),
) -> Result<(), String> {
    let (n_t, n_f) = (map.grid.n_t(), map.grid.n_f());
    let db = map.to_db(floor_db, reference);
    let sr = map.grid.sample_rate as f64;
    let (t0, t1) = (map.grid.t_edges[0] / sr, map.grid.t_edges[n_t] / sr);

    // Row-major RGB, top row = highest frequency, which is how a spectrogram is read.
    let mut rgb = vec![0u8; n_t * n_f * 3];
    for j in 0..n_f {
        let y = n_f - 1 - j;
        for i in 0..n_t {
            let c = heat((db[i * n_f + j] as f64 + floor_db as f64) / floor_db as f64);
            let o = (y * n_t + i) * 3;
            rgb[o] = c.0;
            rgb[o + 1] = c.1;
            rgb[o + 2] = c.2;
        }
    }

    // Bin index of a frequency, for placing the overlay dots on the same axis as the pixels.
    let f_index = |hz: f32| -> f64 {
        match map.grid.f_edges.binary_search_by(|e| e.total_cmp(&hz)) {
            Ok(i) => i as f64,
            Err(0) => -1.0,
            Err(i) if i > n_f => -1.0,
            Err(i) => {
                let (a, b) = (map.grid.f_edges[i - 1], map.grid.f_edges[i]);
                (i - 1) as f64 + ((hz - a) / (b - a)) as f64
            }
        }
    };

    draw_to!(path, fmt, size, |root| {
        let (top, bottom) = root.split_vertically(size.1 as i32 - 24);
        bottom
            .titled(caption, (FONT, 11).into_font().color(&RGBColor(90, 90, 90)))
            .map_err(|e| e.to_string())?;

        let mut chart = ChartBuilder::on(&top)
            .caption("separable-marginal pseudo-Wigner map", (FONT, 18))
            .margin(12)
            .x_label_area_size(46)
            .y_label_area_size(74)
            .build_cartesian_2d(t0..t1, 0f64..n_f as f64)
            .map_err(|e| e.to_string())?;
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("time (s)")
            .y_desc("frequency (Hz)")
            .y_labels(10)
            .y_label_formatter(&|v: &f64| {
                let i = (v.round().max(0.0) as usize).min(n_f);
                format!("{:.0}", map.grid.f_edges[i])
            })
            .label_style((FONT, 13))
            .draw()
            .map_err(|e| e.to_string())?;

        let elem = BitMapElement::with_owned_buffer(
            (t0, n_f as f64),
            (n_t as u32, n_f as u32),
            rgb.clone(),
        )
        .ok_or("could not build the heat-map image")?;
        chart
            .draw_series(std::iter::once(elem))
            .map_err(|e| e.to_string())?;

        // Sanity overlay: if the heat is not under the dots, the accumulator has an indexing bug.
        if !overlay.is_empty() {
            chart
                .draw_series(
                    overlay
                        .iter()
                        .map(|&(t, f)| (t, f_index(f)))
                        .filter(|&(t, y)| t >= t0 && t <= t1 && y >= 0.0)
                        .map(|(t, y)| Circle::new((t, y), 1, RGBColor(120, 255, 160).filled())),
                )
                .map_err(|e| e.to_string())?;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_widths_come_from_the_content() {
        let mut t = Table::new(&["a", "bb"]);
        t.row(vec!["xxxx".into(), "y".into()]);
        let out = t.render("");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "   a bb");
        assert_eq!(lines[1], "---- --");
        assert_eq!(lines[2], "xxxx  y");
    }

    #[test]
    fn numbers_stay_short_across_the_ranges_these_quantities_span() {
        assert_eq!(num(0.0), "0");
        assert_eq!(num(2147.0), "2147");
        assert_eq!(num(12.34), "12.3");
        assert_eq!(num(0.3), "0.3000");
        assert_eq!(num(1.2e-5), "1.200e-5");
        assert_eq!(num(4.8e6), "4.800e6");
    }

    /// The ramp starts at black and ends light, monotonically in luminance — the property that
    /// makes a level readable off the picture.
    #[test]
    fn the_heat_ramp_is_monotone_in_luminance() {
        let lum = |c: RGBColor| 0.2126 * c.0 as f64 + 0.7152 * c.1 as f64 + 0.0722 * c.2 as f64;
        assert_eq!((heat(0.0).0, heat(0.0).1, heat(0.0).2), (0, 0, 0));
        let mut prev = -1.0;
        for i in 0..=64 {
            let l = lum(heat(i as f64 / 64.0));
            assert!(l > prev, "luminance fell at {i}");
            prev = l;
        }
    }

}
