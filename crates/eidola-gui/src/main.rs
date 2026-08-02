use eidola_gui::lifecycle::{LaunchOptions, USAGE};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if LaunchOptions::wants_help(&args) {
        print!("{USAGE}");
        return;
    }
    eidola_gui::run_with(LaunchOptions::parse(&args));
}
