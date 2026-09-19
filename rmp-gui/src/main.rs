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
mod project;
mod results;
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

/// `rmp-gui`'s one flag and one positional argument.
///
/// No clap here: the CLI surface is one optional path, and `rmp-cli` is where this workspace's
/// argument parsing lives — see `CLAUDE.md`'s crate layout. Scanned for `-h`/`--help` wherever it
/// appears, the usual convention, rather than only in the first position.
fn print_help() {
    println!(
        "rmp-gui — graphical front end for rmp: analyse, inspect and resynthesise\n\
         \n\
         Usage:\n\
         \x20 rmp-gui [PROJECT]\n\
         \n\
         Arguments:\n\
         \x20 PROJECT   a project directory to open at startup (one `File > Save Project`\n\
         \x20           already wrote). If given, it is opened the same way `File > Open\n\
         \x20           Project` does; a directory that does not exist, or one with no\n\
         \x20           readable `project.toml`, is reported in the window rather than\n\
         \x20           refused here.\n\
         \n\
         Options:\n\
         \x20 -h, --help   print this message and exit"
    );
}

/// The startup project path, or a usage error for anything this cannot make sense of.
///
/// A second positional argument is refused rather than silently ignored — a typo'd flag landing
/// here as a second path is a mistake worth saying something about, not a project to open.
fn parse_args(args: &[String]) -> Result<Option<std::path::PathBuf>, String> {
    match args {
        [] => Ok(None),
        [path] => Ok(Some(std::path::PathBuf::from(path))),
        _ => Err(format!("too many arguments (expected at most one: a project directory), got {args:?}")),
    }
}

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return Ok(());
    }
    let project = match parse_args(&args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("rmp-gui: {e}\n");
            print_help();
            std::process::exit(2);
        }
    };

    rmp_core::threads::configure_pool();
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
        Box::new(move |cc| {
            install_fallback_font(&cc.egui_ctx);
            let mut rmp_app = app::RmpApp::default();
            if let Some(dir) = project {
                rmp_app.open_project_at(dir);
            }
            Ok(Box::new(rmp_app))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_arguments_is_no_project() {
        assert_eq!(parse_args(&[]).unwrap(), None);
    }

    #[test]
    fn one_argument_is_the_project_path() {
        let args = ["a-project".to_string()];
        assert_eq!(parse_args(&args).unwrap(), Some(std::path::PathBuf::from("a-project")));
    }

    #[test]
    fn a_second_argument_is_refused_rather_than_ignored() {
        let args = ["a".to_string(), "b".to_string()];
        let err = parse_args(&args).expect_err("two positional arguments must not be accepted");
        assert!(!err.is_empty());
    }
}
