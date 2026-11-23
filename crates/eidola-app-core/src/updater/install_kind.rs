//! What kind of install the running executable belongs to, and so what an
//! update would replace.
//!
//! Every shipped payload has a fixed layout, and the running executable's
//! own path says which one it is in. Nothing here asks a package manager or
//! reads a marker file: a marker is one more thing a manual install over the
//! top would leave stale, and the layout is the fact the update has to act
//! on anyway. The mobile app has no executable path at all: it runs as wasm
//! in a webview, and the platform it was compiled for is the whole answer.
//!
//! | Layout | Kind | What an update replaces |
//! |---|---|---|
//! | `<x>.app/Contents/MacOS/<exe>` | [`InstallKind::MacApp`] | the whole bundle, where it is |
//! | the same, under an `AppTranslocation` mount | [`InstallKind::MacAppTranslocated`] | nothing: the path is randomized and read-only |
//! | `/usr/libexec/eidola/<exe>` | [`InstallKind::Deb`] | the package, through the system's package tool |
//! | under `/nix/store/` | [`InstallKind::Nix`] | nothing: the profile that installed it updates it |
//! | the CLI anywhere else on Linux | [`InstallKind::StaticCli`] | that one file |
//! | wasm32, in a native shell's webview | [`InstallKind::MobileBundle`] | the shell's active bundle, through its bundle store (`super::mobile`) |
//! | anything else | [`InstallKind::Unmanaged`] | nothing |
//!
//! "Anything else" includes every development build. A debug build is
//! always unmanaged, and so is anything under a `target` directory, which
//! is where Cargo puts a release build too: an updater that replaced the
//! binary a developer just built would be replacing their work with ours.

use std::path::{Component, Path, PathBuf};

/// The `.app`'s main executable, named by `CFBundleExecutable`
/// (`scripts/package-gui-app.sh`).
pub const MAC_BUNDLE_EXECUTABLE: &str = "Eidola";
/// Where the `.deb` puts every executable it ships (`flake.nix`,
/// `packages.eidola-linux-deb`).
pub const DEB_LIBEXEC: &str = "/usr/libexec/eidola";
/// The `.deb`'s main executable, inside [`DEB_LIBEXEC`].
pub const DEB_MAIN_EXECUTABLE: &str = "eidola-gui";

/// Which of the two shipped programs is asking. The static CLI is a layout
/// only the CLI has: a GUI binary loose on disk is an unpacked archive, not
/// an install anything updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Program {
    Gui,
    Cli,
}

/// The platform whose layouts apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    MacOs,
    Linux,
    /// wasm32 in a webview: the mobile app, hosted by a native shell.
    Webview,
    Other,
}

impl Platform {
    pub fn current() -> Self {
        if cfg!(target_family = "wasm") {
            Platform::Webview
        } else if cfg!(target_os = "macos") {
            Platform::MacOs
        } else if cfg!(target_os = "linux") {
            Platform::Linux
        } else {
            Platform::Other
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallKind {
    /// A macOS app bundle, wherever it was put: `/Applications`,
    /// `~/Applications`, or anywhere else. The GUI and the CLI inside it
    /// are one install.
    MacApp { bundle: PathBuf },
    /// macOS is running the bundle from an App Translocation mount, because
    /// the app was opened where it was downloaded while still quarantined.
    /// The path is randomized and read-only and goes away, so nothing can
    /// be replaced there; the app has to be moved first.
    MacAppTranslocated { bundle: PathBuf },
    /// The `.deb`'s layout. The package database owns these files.
    Deb,
    /// A Nix store path. Store paths are immutable, and whatever put this
    /// one in a profile is what moves the profile on.
    Nix,
    /// The standalone static Linux CLI, at `path`.
    StaticCli { path: PathBuf },
    /// The mobile app: a bundle a native shell serves. An update stages a
    /// new bundle in the shell's store and activates it there
    /// (`super::mobile`); there is no executable to replace or to hash.
    MobileBundle,
    /// A development build, or a layout no release ships.
    Unmanaged,
}

impl InstallKind {
    /// The executable whose bytes identify this install: the one an update
    /// replaces, and the one the installed-release record is keyed by
    /// (`super::installed`). One per install, whichever of its programs is
    /// asking, so the GUI and the CLI from one bundle agree on what is
    /// installed. `None` for an install the updater never writes.
    pub fn main_executable(&self) -> Option<PathBuf> {
        match self {
            InstallKind::MacApp { bundle } => Some(
                bundle
                    .join("Contents")
                    .join("MacOS")
                    .join(MAC_BUNDLE_EXECUTABLE),
            ),
            InstallKind::Deb => Some(Path::new(DEB_LIBEXEC).join(DEB_MAIN_EXECUTABLE)),
            InstallKind::StaticCli { path } => Some(path.clone()),
            InstallKind::MacAppTranslocated { .. }
            | InstallKind::Nix
            | InstallKind::MobileBundle
            | InstallKind::Unmanaged => None,
        }
    }
}

static RUNNING_PROGRAM: std::sync::OnceLock<Program> = std::sync::OnceLock::new();

/// Say which program this process is. The CLI calls it once at startup;
/// a process that never does is taken to be the GUI. Only the first call
/// counts.
pub fn declare_running_program(program: Program) {
    let _ = RUNNING_PROGRAM.set(program);
}

/// The program [`declare_running_program`] named, or the GUI.
pub fn running_program() -> Program {
    RUNNING_PROGRAM.get().copied().unwrap_or(Program::Gui)
}

/// The install the running executable belongs to.
///
/// `std::env::current_exe` names the executable as it was invoked on
/// macOS, which may be a link (`/usr/local/bin/eidola` into a bundle), so
/// the path is resolved before its shape is read. Linux's `/proc/self/exe`
/// is resolved already. A path that cannot be read at all is
/// [`InstallKind::Unmanaged`]: an update the process cannot locate is not
/// one it can perform.
pub fn detect(program: Program) -> InstallKind {
    if cfg!(debug_assertions) {
        return InstallKind::Unmanaged;
    }
    if Platform::current() == Platform::Webview {
        return InstallKind::MobileBundle;
    }
    let Ok(exe) = std::env::current_exe() else {
        return InstallKind::Unmanaged;
    };
    let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
    classify(&exe, program, Platform::current())
}

/// The install `exe` belongs to, by its path alone. Pure: reads nothing on
/// disk, so every layout is testable on every platform. `exe` should
/// already be canonical.
pub fn classify(exe: &Path, program: Program, platform: Platform) -> InstallKind {
    // A webview has no executable path; whatever was passed names nothing.
    if platform == Platform::Webview {
        return InstallKind::MobileBundle;
    }
    if !exe.is_absolute() || under_cargo_target(exe) {
        return InstallKind::Unmanaged;
    }
    match platform {
        Platform::MacOs => match app_bundle_of(exe) {
            Some(bundle) if translocated(&bundle) => InstallKind::MacAppTranslocated { bundle },
            Some(bundle) => InstallKind::MacApp { bundle },
            None => InstallKind::Unmanaged,
        },
        Platform::Linux => {
            if exe.starts_with("/nix/store") {
                InstallKind::Nix
            } else if exe.parent() == Some(Path::new(DEB_LIBEXEC)) {
                InstallKind::Deb
            } else if program == Program::Cli {
                InstallKind::StaticCli {
                    path: exe.to_path_buf(),
                }
            } else {
                InstallKind::Unmanaged
            }
        }
        // Answered above, before the path was read.
        Platform::Webview => InstallKind::MobileBundle,
        Platform::Other => InstallKind::Unmanaged,
    }
}

/// The `.app` directory `exe` is the `Contents/MacOS/` executable of.
fn app_bundle_of(exe: &Path) -> Option<PathBuf> {
    let macos = exe.parent()?;
    let contents = macos.parent()?;
    let bundle = contents.parent()?;
    let is_bundle = macos.file_name().is_some_and(|n| n == "MacOS")
        && contents.file_name().is_some_and(|n| n == "Contents")
        && bundle.extension().is_some_and(|e| e == "app");
    is_bundle.then(|| bundle.to_path_buf())
}

fn translocated(bundle: &Path) -> bool {
    bundle
        .components()
        .any(|c| matches!(c, Component::Normal(n) if n == "AppTranslocation"))
}

/// Whether `exe` sits in a Cargo target directory, where every local build
/// lands, debug or release.
fn under_cargo_target(exe: &Path) -> bool {
    exe.components()
        .any(|c| matches!(c, Component::Normal(n) if n == "target"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mac(path: &str) -> InstallKind {
        classify(Path::new(path), Program::Gui, Platform::MacOs)
    }

    fn linux(path: &str, program: Program) -> InstallKind {
        classify(Path::new(path), program, Platform::Linux)
    }

    #[test]
    fn a_bundle_is_an_app_wherever_it_was_put() {
        for bundle in [
            "/Applications/Eidola.app",
            "/Users/someone/Applications/Eidola.app",
            "/Users/someone/Desktop/Eidola 2.app",
        ] {
            assert_eq!(
                mac(&format!("{bundle}/Contents/MacOS/Eidola")),
                InstallKind::MacApp {
                    bundle: bundle.into()
                }
            );
        }
    }

    #[test]
    fn the_cli_inside_a_bundle_is_the_same_install_as_the_app() {
        let gui = mac("/Applications/Eidola.app/Contents/MacOS/Eidola");
        let cli = classify(
            Path::new("/Applications/Eidola.app/Contents/MacOS/eidola-cli"),
            Program::Cli,
            Platform::MacOs,
        );
        assert_eq!(gui, cli);
        assert_eq!(
            cli.main_executable().unwrap(),
            Path::new("/Applications/Eidola.app/Contents/MacOS/Eidola")
        );
    }

    #[test]
    fn a_translocated_bundle_cannot_be_replaced() {
        let kind = mac(
            "/private/var/folders/x/T/AppTranslocation/1234-ABCD/d/Eidola.app/Contents/MacOS/Eidola",
        );
        assert!(matches!(kind, InstallKind::MacAppTranslocated { .. }));
        assert_eq!(kind.main_executable(), None);
    }

    #[test]
    fn a_mac_executable_outside_a_bundle_is_unmanaged() {
        assert_eq!(mac("/usr/local/bin/eidola"), InstallKind::Unmanaged);
        assert_eq!(
            mac("/Applications/Eidola.app/Contents/Resources/Eidola"),
            InstallKind::Unmanaged
        );
    }

    #[test]
    fn the_deb_layout_is_the_package_for_either_program() {
        for (exe, program) in [
            ("/usr/libexec/eidola/eidola-gui", Program::Gui),
            ("/usr/libexec/eidola/eidola", Program::Cli),
        ] {
            let kind = linux(exe, program);
            assert_eq!(kind, InstallKind::Deb);
            assert_eq!(
                kind.main_executable().unwrap(),
                Path::new("/usr/libexec/eidola/eidola-gui")
            );
        }
        // Only that directory: a subdirectory of it is not the package's.
        assert_eq!(
            linux("/usr/libexec/eidola/old/eidola-gui", Program::Gui),
            InstallKind::Unmanaged
        );
    }

    #[test]
    fn a_store_path_belongs_to_nix() {
        assert_eq!(
            linux(
                "/nix/store/abc-eidola-gui/bin/.eidola-gui-wrapped",
                Program::Gui
            ),
            InstallKind::Nix
        );
        assert_eq!(
            linux("/nix/store/abc-eidola/bin/eidola", Program::Cli),
            InstallKind::Nix
        );
    }

    #[test]
    fn a_loose_cli_on_linux_is_the_static_cli() {
        for path in ["/usr/local/bin/eidola", "/home/someone/bin/eidola-cli"] {
            assert_eq!(
                linux(path, Program::Cli),
                InstallKind::StaticCli { path: path.into() }
            );
        }
        // The GUI has no standalone layout.
        assert_eq!(
            linux("/home/someone/bin/eidola-gui", Program::Gui),
            InstallKind::Unmanaged
        );
    }

    #[test]
    fn a_cargo_build_is_never_managed() {
        assert_eq!(
            linux("/home/someone/eidola/target/release/eidola", Program::Cli),
            InstallKind::Unmanaged
        );
        assert_eq!(
            linux(
                "/home/someone/eidola/target/x86_64-unknown-linux-musl/release/eidola",
                Program::Cli
            ),
            InstallKind::Unmanaged
        );
        assert_eq!(
            mac("/Users/someone/eidola/target/release/Eidola.app/Contents/MacOS/Eidola"),
            InstallKind::Unmanaged
        );
    }

    #[test]
    fn a_relative_path_is_not_a_location() {
        assert_eq!(linux("eidola", Program::Cli), InstallKind::Unmanaged);
    }

    #[test]
    fn the_webview_is_the_mobile_bundle_whatever_the_path() {
        for path in ["", "/", "eidola", "/target/release/eidola"] {
            let kind = classify(Path::new(path), Program::Gui, Platform::Webview);
            assert_eq!(kind, InstallKind::MobileBundle, "{path:?}");
            assert_eq!(kind.main_executable(), None);
        }
    }

    #[test]
    fn other_platforms_have_no_managed_layout() {
        assert_eq!(
            classify(
                Path::new("/usr/local/bin/eidola"),
                Program::Cli,
                Platform::Other
            ),
            InstallKind::Unmanaged
        );
    }
}
