//! `eidola service {install,start,stop}` — the Linux long-lived-app sugar
//! (task 17, wave 2).
//!
//! On Linux the idiomatic long-lived per-user process is a **systemd user
//! service**, not a tray icon (StatusNotifierItem is fragmented across
//! desktops and a tray must never be load-bearing). So we ship a unit file as
//! data and give it three verbs. Everything here is deliberately thin: the
//! unit is a template with one substitution, install writes it and asks
//! systemd to reload, and start/stop are `systemctl --user` calls whose
//! output is passed straight through. There is nothing clever to get wrong,
//! and a user who prefers to manage the unit by hand loses nothing.
//!
//! The service runs the **GUI binary** in `--windowless` mode — one process,
//! not a daemon split: TCC-style permissions and the future screen-capture
//! capability attach to the app the user can see and quit, and engines, the
//! invalidation bus and the single-writer local database already live in one
//! `AppCore`.
//!
//! **The honest caveat, printed at install time:** while the service holds
//! the local database's exclusive lock, an `eidola …` invocation refuses to
//! start (`AppError::DatabaseInUse`). The CLI becoming a client of the
//! running app is the IPC wave; until then, `eidola service stop` is the way
//! to hand the database back.

use std::path::{Path, PathBuf};
use std::process::Command;

use eidola_app_core::error::AppError;

/// The unit's file name — also its `systemctl --user` name.
pub const UNIT_NAME: &str = "eidola.service";

/// The shipped unit, with `@EXEC@` standing in for the resolved executable.
///
/// A `const` rather than an `include_str!` of a `data/` file on purpose: the
/// Nix build's source filter is crane's, which keeps Rust sources plus a
/// short allowlist of extensions (`.sql`, `.ttf`), so a `.service` asset
/// would silently vanish from the packaged source and break the release
/// build. One string in one crate needs no packaging story at all.
const UNIT_TEMPLATE: &str = "\
# Eidola as a systemd user service.
#
# Written by `eidola service install` into
# $XDG_CONFIG_HOME/systemd/user/eidola.service (default ~/.config/systemd/user).
#
# The unit runs the GUI binary in `--windowless` mode: the process hosts the
# app — loaded local inference engines, background polling, and (later) the
# OpenAI-compatible proxy — with no window. Windows are opened by launching
# the app normally; until the IPC socket lands, a second launch cannot ask
# this process for one, and will refuse to start because this one holds the
# local database's exclusive lock.

[Unit]
Description=Eidola
Documentation=https://www.eidola.ai/docs/
# Prefer to start after the desktop session so the process inherits
# WAYLAND_DISPLAY and can show a window later. Deliberately a soft ordering,
# not a requirement: on a headless box there is no graphical session and the
# service should still run.
After=graphical-session.target

[Service]
Type=simple
ExecStart=@EXEC@ --windowless
Restart=on-failure
RestartSec=5
# The process owns loaded inference engines; SIGTERM is translated into an
# ordinary quit so they are torn down rather than orphaned.
KillSignal=SIGTERM
TimeoutStopSec=20

[Install]
WantedBy=default.target
";

/// Render the unit for a concrete executable path.
pub fn render_unit(exec: &str) -> String {
    UNIT_TEMPLATE.replace("@EXEC@", &systemd_quote(exec))
}

/// Quote a path for a systemd command line.
///
/// `ExecStart=` is **split on whitespace**, so a bare
/// `/home/me/Eidola Builds/eidola-gui` would run `/home/me/Eidola` with
/// `Builds/eidola-gui` as its first argument — and a path with a space in it
/// is entirely ordinary on a desktop. Systemd's rules, all three of which
/// apply here:
///
/// - a double-quoted word is one argument;
/// - inside it, `\` escapes, so a literal `\` or `"` must be doubled/escaped;
/// - `%` introduces a *specifier* anywhere in the unit — including inside
///   quotes — so a literal `%` must be written `%%`.
///
/// Quoting unconditionally rather than only when needed: one code path is one
/// fewer thing to get subtly wrong, and systemd accepts a quoted executable
/// exactly as it accepts a bare one.
fn systemd_quote(path: &str) -> String {
    let mut out = String::with_capacity(path.len() + 2);
    out.push('"');
    for ch in path.chars() {
        match ch {
            '\\' | '"' => {
                out.push('\\');
                out.push(ch);
            }
            '%' => out.push_str("%%"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

/// Where the unit belongs, under a given XDG config home.
pub fn unit_path(config_home: &Path) -> PathBuf {
    config_home.join("systemd").join("user").join(UNIT_NAME)
}

/// `$XDG_CONFIG_HOME`, else `$HOME/.config`.
pub fn config_home() -> Result<PathBuf, AppError> {
    if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    let home = std::env::var_os("HOME").filter(|v| !v.is_empty());
    home.map(|h| PathBuf::from(h).join(".config"))
        .ok_or_else(|| AppError::Config {
            message: "neither XDG_CONFIG_HOME nor HOME is set, so there is no \
                      user unit directory to install into"
                .into(),
        })
}

/// Write the rendered unit under `config_home`, creating the directory.
/// Returns the path written.
pub fn write_unit(config_home: &Path, exec: &str) -> std::io::Result<PathBuf> {
    let path = unit_path(config_home);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, render_unit(exec))?;
    Ok(path)
}

/// Find the GUI binary the unit should start.
///
/// An explicit `--exec` wins. Otherwise look for `eidola-gui` beside this
/// binary — the layout every one of our artifacts uses (the Nix output, the
/// cargo target dir, a distro package's `bin/`) — and fall back to a bare
/// name resolved against `PATH`. A guess that turns out wrong fails loudly
/// at `systemctl start`, which is the right place for it to fail; refusing to
/// write a unit at all would be worse.
pub fn resolve_exec(explicit: Option<String>) -> Result<String, AppError> {
    if let Some(path) = explicit {
        return Ok(path);
    }
    if let Ok(me) = std::env::current_exe()
        && let Some(dir) = me.parent()
    {
        let sibling = dir.join("eidola-gui");
        if sibling.is_file() {
            return Ok(sibling.to_string_lossy().into_owned());
        }
    }
    if let Some(found) = which_on_path("eidola-gui") {
        return Ok(found.to_string_lossy().into_owned());
    }
    Err(AppError::Config {
        message: "could not find the `eidola-gui` binary (looked beside this \
                  executable and on PATH) — pass `--exec <path>`"
            .into(),
    })
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// Run `systemctl --user <args>`, passing its output through.
///
/// A missing `systemctl` is the interesting failure and gets its own
/// message: this is a Linux/systemd facility and saying so plainly beats an
/// "No such file or directory (os error 2)".
fn systemctl(args: &[&str]) -> Result<(), AppError> {
    let status = Command::new("systemctl").arg("--user").args(args).status();
    match status {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(AppError::Config {
            message: format!("`systemctl --user {}` failed ({status})", args.join(" ")),
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(AppError::Config {
            message: "`systemctl` was not found — user services are a \
                      systemd (Linux) facility. On macOS the app already \
                      outlives its windows; run it normally."
                .into(),
        }),
        Err(e) => Err(AppError::Config {
            message: format!("could not run systemctl: {e}"),
        }),
    }
}

/// Refuse before touching the filesystem when there is no systemd to install
/// into — writing a unit nobody will ever read is worse than saying so.
fn require_systemd() -> Result<(), AppError> {
    match Command::new("systemctl").arg("--version").output() {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(AppError::Config {
            message: "`systemctl` was not found — user services are a \
                      systemd (Linux) facility. On macOS the app already \
                      outlives its windows; run it normally."
                .into(),
        }),
        Err(e) => Err(AppError::Config {
            message: format!("could not run systemctl: {e}"),
        }),
    }
}

/// Write the unit, reload systemd, and enable it for login.
pub fn install(exec: Option<String>) -> Result<(), AppError> {
    require_systemd()?;
    let exec = resolve_exec(exec)?;
    let home = config_home()?;
    let path = write_unit(&home, &exec).map_err(|e| AppError::Config {
        message: format!("could not write {}: {e}", unit_path(&home).display()),
    })?;
    println!("Wrote {}", path.display());
    println!("  ExecStart={exec} --windowless");

    systemctl(&["daemon-reload"])?;
    systemctl(&["enable", UNIT_NAME])?;
    println!("Enabled {UNIT_NAME} (starts at login). Start it now with:");
    println!("  eidola service start");
    println!();
    println!(
        "Note: while the service runs it holds the local database's exclusive \
         lock, so other `eidola` commands will refuse to start until the CLI \
         can talk to the running app. `eidola service stop` hands it back."
    );
    Ok(())
}

/// Start the service now.
pub fn start() -> Result<(), AppError> {
    systemctl(&["start", UNIT_NAME])?;
    println!("Started {UNIT_NAME}.");
    Ok(())
}

/// Stop the service (SIGTERM → an ordinary quit, engines torn down).
pub fn stop() -> Result<(), AppError> {
    systemctl(&["stop", UNIT_NAME])?;
    println!("Stopped {UNIT_NAME}.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rendered_unit_starts_the_gui_windowless() {
        let unit = render_unit("/opt/eidola/bin/eidola-gui");
        assert!(
            unit.contains(r#"ExecStart="/opt/eidola/bin/eidola-gui" --windowless"#),
            "unit must launch the resolved binary in windowless mode:\n{unit}"
        );
        assert!(!unit.contains("@EXEC@"), "every placeholder is substituted");
        assert!(unit.contains("[Install]") && unit.contains("WantedBy=default.target"));
        // SIGTERM is what the windowless mode translates into a quit; a
        // KillMode that skipped it would orphan the engines.
        assert!(unit.contains("KillSignal=SIGTERM"));
    }

    #[test]
    fn install_writes_into_the_xdg_user_unit_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_unit(tmp.path(), "/usr/bin/eidola-gui").expect("write unit");

        assert_eq!(path, tmp.path().join("systemd/user/eidola.service"));
        let written = std::fs::read_to_string(&path).expect("read back");
        assert_eq!(written, render_unit("/usr/bin/eidola-gui"));
    }

    #[test]
    fn writing_the_unit_twice_replaces_it() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_unit(tmp.path(), "/old/eidola-gui").expect("first write");
        let path = write_unit(tmp.path(), "/new/eidola-gui").expect("second write");
        let written = std::fs::read_to_string(&path).expect("read back");
        assert!(written.contains("/new/eidola-gui"));
        assert!(!written.contains("/old/eidola-gui"));
    }

    #[test]
    fn a_path_with_whitespace_stays_one_argument() {
        // Unquoted, systemd would split this into the executable
        // `/home/me/Eidola` with `Builds/eidola-gui` as its first argument.
        let unit = render_unit("/home/me/Eidola Builds/eidola-gui");
        assert!(
            unit.contains(r#"ExecStart="/home/me/Eidola Builds/eidola-gui" --windowless"#),
            "the whole path must be one quoted argument:\n{unit}"
        );
    }

    #[test]
    fn quoting_escapes_what_systemd_would_otherwise_read() {
        // `%` is a specifier anywhere in the unit — even inside quotes — and
        // `"`/`\` are the quoted word's own escapes.
        assert_eq!(
            systemd_quote("/opt/100%/eidola-gui"),
            r#""/opt/100%%/eidola-gui""#
        );
        assert_eq!(
            systemd_quote(r#"/opt/a"b/eidola-gui"#),
            r#""/opt/a\"b/eidola-gui""#
        );
        assert_eq!(
            systemd_quote(r"/opt/a\b/eidola-gui"),
            r#""/opt/a\\b/eidola-gui""#
        );
        // The template itself must stay specifier-free, or `%%` in a path
        // would be the only escaped one.
        assert!(!UNIT_TEMPLATE.replace("@EXEC@", "").contains('%'));
    }

    #[test]
    fn an_explicit_exec_wins_over_discovery() {
        let resolved = resolve_exec(Some("/somewhere/else/eidola-gui".into())).expect("resolve");
        assert_eq!(resolved, "/somewhere/else/eidola-gui");
    }
}
