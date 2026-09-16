//! `rmp-gui` — a graphical front end over the whole system: analyse, inspect, resynthesise.
//!
//! A scaffold. The window, the settings panel and the result tabs are stubs that name the
//! `rmp-core` and `rmp-synthesis` calls that will fill them; what is finished is [`task`], which
//! runs a decomposition off the UI thread and can stop one.
//!
//! It shares its analysis with `rmp` the command line rather than reimplementing it: both drive
//! [`rmp_core::pipeline::analyse`], which is why they cannot drift into two different
//! decompositions of the same input.

mod app;
mod audio;
mod help;
mod playback;
mod settings;
mod task;
mod view;

/// Fonts with enough coverage for the manual, tried in order.
///
/// The same list `rmpstat` uses to label its charts, for the same reason: no font discovery, so a
/// path it is. eframe's own fonts cover Latin but not the arrows, Greek and maths `MANUAL.md`
/// writes its formulas in — `→ ≈ √ ∝ Σ σ π` all come out as missing-glyph boxes without this, and
/// a help window full of boxes where the operators should be is worse than no help window.
const FONT_CANDIDATES: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
];

/// Add a broad-coverage system font behind the built-in ones.
///
/// Appended rather than prepended: egui walks a family in order and takes the first face that has
/// the glyph, so this changes nothing that already rendered and only fills the gaps. Missing
/// entirely is survivable — the window works, some symbols are boxes — so a failure here is not
/// worth refusing to start over.
fn install_fallback_font(ctx: &egui::Context) {
    let Some(path) = FONT_CANDIDATES.iter().find(|p| std::path::Path::new(p).is_file()) else {
        return;
    };
    let Ok(bytes) = std::fs::read(path) else { return };

    let mut fonts = egui::FontDefinitions::default();
    fonts
        .font_data
        .insert("fallback".into(), std::sync::Arc::new(egui::FontData::from_owned(bytes)));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("fallback".into());
    }
    ctx.set_fonts(fonts);
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 860.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("rmp"),
        ..Default::default()
    };
    eframe::run_native(
        "rmp",
        options,
        Box::new(|cc| {
            install_fallback_font(&cc.egui_ctx);
            Ok(Box::<app::RmpApp>::default())
        }),
    )
}
