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
mod settings;
mod task;

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
        Box::new(|_cc| Ok(Box::<app::RmpApp>::default())),
    )
}
