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
    UNIT_TEMPLATE.replace("@EXEC@", &systemd_quote_exec(exec))
}

/// Quote the **executable word** of a systemd command line.
///
/// The name says `_exec` because the escaping rules are not the same for the
/// program and for its arguments, and getting that backwards is precisely the
/// bug this function had. Two things must be escaped here, and one must not:
///
/// - **Whitespace** → quote the word. `ExecStart=` is split on whitespace, so
///   a bare `/home/me/Eidola Builds/eidola-gui` would run `/home/me/Eidola`
///   with `Builds/eidola-gui` as its first argument, and a space in a desktop
///   path is entirely ordinary. Inside a double-quoted word `\` escapes, so a
///   literal `\` or `"` must be escaped in turn.
/// - **`%` → `%%`.** A specifier is expanded *everywhere in the unit*,
///   including inside quotes and including the executable word. Verified:
///   `ExecStart="/opt/100%dir/eidola-gui"` is rejected by
///   `systemd-analyze verify` as
///   `Command /opt/100/run/credentials/….serviceir/eidola-gui is not
///   executable` — `%d` had been substituted mid-path.
/// - **`$` must be left alone.** Environment expansion does *not* apply to
///   the program: "Note that the first argument (i.e. the program to execute)
///   may not be a variable" (systemd.service(5), Command Lines). Escaping it
///   is therefore not merely unnecessary but *corrupting* — systemd takes the
///   `$$` literally. Verified: `ExecStart="/srv/$app/eidola-gui"` passes
///   `systemd-analyze verify` while `"/srv/$$app/eidola-gui"` is rejected as
///   not existing.
///
/// (For a later *argument* the rule flips — `$FOO`/`${FOO}` are expanded there
/// and `$$` is the documented literal-dollar escape — which is why this
/// function is scoped to the executable and the unit's only other word is the
/// fixed literal `--windowless`.)
///
/// Quoting unconditionally rather than only when needed: one code path is one
/// fewer thing to get subtly wrong, and systemd accepts a quoted executable
/// exactly as it accepts a bare one.
fn systemd_quote_exec(path: &str) -> String {
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
/// name resolved against `PATH`.
///
/// Every branch goes through [`absolute_existing_exec`], because a unit's
/// `ExecStart` **must** be absolute and a `--exec` typed at a shell is very
/// often not (`--exec ./target/release/eidola-gui` is the dev loop).
pub fn resolve_exec(explicit: Option<String>) -> Result<String, AppError> {
    if let Some(path) = explicit {
        return absolute_existing_exec(Path::new(&path));
    }
    if let Ok(me) = std::env::current_exe()
        && let Some(dir) = me.parent()
    {
        let sibling = dir.join("eidola-gui");
        if sibling.is_file() {
            return absolute_existing_exec(&sibling);
        }
    }
    if let Some(found) = which_on_path("eidola-gui") {
        return absolute_existing_exec(&found);
    }
    Err(AppError::Config {
        message: "could not find the `eidola-gui` binary (looked beside this \
                  executable and on PATH) — pass `--exec <path>`"
            .into(),
    })
}

/// Make an executable path absolute (against the install-time CWD) and
/// refuse one that isn't there.
///
/// **Absolutize lexically; do not `canonicalize`.** Canonicalization resolves
/// symlinks, and the two places a `--exec` most often points are symlinks
/// *on purpose*: a Nix profile entry, and `target/release/eidola-gui` under a
/// symlinked checkout. Pinning the unit to today's link target means a
/// rebuild swaps the link and `systemctl restart` keeps launching the old
/// binary — silently, which is the worst way to be wrong. Writing the path
/// the user named leaves the indirection they asked for intact.
/// `std::path::absolute` is purely lexical: it joins the CWD and drops `.`
/// components but leaves `..` alone, precisely because resolving `..` is
/// unsound through a symlink. The kernel resolves both at exec time.
///
/// Existence is checked because the alternative is a unit that can only ever
/// fail, discovered later at `systemctl start` with systemd's own wording
/// rather than here with ours. Discovery already only yields paths it found,
/// so this bites exactly the hand-typed `--exec`.
fn absolute_existing_exec(path: &Path) -> Result<String, AppError> {
    let absolute = std::path::absolute(path).map_err(|e| AppError::Config {
        message: format!("could not resolve `{}`: {e}", path.display()),
    })?;
    if !absolute.is_file() {
        return Err(AppError::Config {
            message: format!(
                "no executable at `{}` — systemd needs an absolute path to a \
                 file that exists",
                absolute.display()
            ),
        });
    }
    if !is_executable(&absolute) {
        return Err(AppError::Config {
            message: format!(
                "`{}` is not executable — systemd would refuse to start the \
                 unit (`chmod +x` it, or point `--exec` elsewhere)",
                absolute.display()
            ),
        });
    }
    Ok(absolute.to_string_lossy().into_owned())
}

/// Whether the file at `path` can be executed.
///
/// Existing is not enough: systemd rejects a mode-644 file outright
/// (`Command … is not executable: Permission denied`, verified with
/// `systemd-analyze verify`), so a unit naming one can never start. Checking
/// the bits here keeps the refusal at the same before-any-write seam as the
/// missing-file case.
///
/// The mode test is any-of-`u+x`/`g+x`/`o+x` rather than "executable *by me*":
/// the unit runs as this user, so `u+x` is what matters in practice, but a
/// binary shipped mode 555 or 775 is perfectly startable and refusing it would
/// be the wrong kind of strict. Off unix there are no mode bits and no systemd
/// either — `service` is Linux-only and its callers are refused earlier by the
/// `systemctl` preflight — so existence is the honest answer there.
#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
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

/// Refuse before touching the filesystem when there is no **reachable user
/// manager** to install into — writing a unit nobody will ever read is worse
/// than saying so, and a half-install (unit written, never reloaded or
/// enabled) is worse than both.
///
/// The probe is `systemctl --user daemon-reload`, not `systemctl --version`.
/// The version query is answered by the binary itself and succeeds anywhere
/// the package is present — inside a container, under WSL, on a box where no
/// per-user systemd instance is running — so it proved only that a *file*
/// existed while the very next call failed to connect to the bus, after the
/// unit had already been written. `daemon-reload` is the cheapest operation
/// that must actually talk to the user manager, it is idempotent, and it is
/// the same call the install has to make anyway.
fn require_systemd() -> Result<(), AppError> {
    let probe = Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .output();
    match probe {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => Err(AppError::Config {
            message: format!(
                "no systemd user manager is reachable ({}){}",
                out.status,
                match String::from_utf8_lossy(&out.stderr).trim() {
                    "" => String::new(),
                    detail => format!(": {detail}"),
                }
            ),
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

    /// A fixture binary: present *and* executable, since `resolve_exec`
    /// refuses a mode-644 file (systemd would too).
    fn write_executable(path: &Path) {
        std::fs::write(path, b"#!/bin/sh\n").expect("write fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod fixture");
        }
    }

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
            systemd_quote_exec("/opt/100%/eidola-gui"),
            r#""/opt/100%%/eidola-gui""#
        );
        assert_eq!(
            systemd_quote_exec(r#"/opt/a"b/eidola-gui"#),
            r#""/opt/a\"b/eidola-gui""#
        );
        assert_eq!(
            systemd_quote_exec(r"/opt/a\b/eidola-gui"),
            r#""/opt/a\\b/eidola-gui""#
        );
        // `$` is NOT escaped: environment expansion does not apply to the
        // program word ("the first argument ... may not be a variable"), so
        // `$$` would be taken literally and corrupt the path. Verified with
        // `systemd-analyze verify` — the literal form passes, the escaped
        // form is rejected as nonexistent.
        assert_eq!(
            systemd_quote_exec("/srv/$app/eidola-gui"),
            r#""/srv/$app/eidola-gui""#
        );
        assert_eq!(
            systemd_quote_exec("/srv/${HOME}/eidola-gui"),
            r#""/srv/${HOME}/eidola-gui""#
        );
        // The template's own *directives* must stay specifier-free, or a
        // path's `%%` would be the only escaped one. Comment lines are exempt
        // — systemd expands nothing in them.
        for line in UNIT_TEMPLATE.replace("@EXEC@", "").lines() {
            if line.starts_with('#') {
                continue;
            }
            assert!(!line.contains('%'), "unescaped specifier in: {line}");
        }
    }

    #[test]
    fn an_explicit_exec_wins_over_discovery() {
        // A real file, so the existence gate passes; the point is that
        // discovery is never consulted.
        let tmp = tempfile::tempdir().expect("tempdir");
        let exe = tmp.path().join("eidola-gui");
        write_executable(&exe);
        let resolved = resolve_exec(Some(exe.to_string_lossy().into_owned())).expect("resolve");
        assert_eq!(resolved, exe.to_string_lossy());
    }

    #[test]
    fn a_relative_exec_is_resolved_against_the_cwd_not_written_verbatim() {
        // `--exec ./target/release/eidola-gui` is the dev loop, and systemd
        // requires an absolute path. Read-only on the CWD — `set_current_dir`
        // is process-global and these tests run in parallel.
        let cwd = std::env::current_dir().expect("cwd");
        let err = resolve_exec(Some("./target/release/eidola-gui-absent".into()))
            .expect_err("a relative path is resolved, then checked");
        let message = err.to_string();
        assert!(
            message.contains(
                &cwd.join("target/release/eidola-gui-absent")
                    .to_string_lossy()
                    .to_string()
            ),
            "the path must be absolutized against the CWD before use; got {message}"
        );
    }

    #[test]
    fn a_resolved_exec_is_absolute_in_the_unit() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exe = tmp.path().join("eidola-gui");
        write_executable(&exe);
        let resolved = resolve_exec(Some(exe.to_string_lossy().into_owned())).expect("resolve");
        assert!(Path::new(&resolved).is_absolute(), "got {resolved}");
        assert!(
            render_unit(&resolved).contains(&format!(r#"ExecStart="{resolved}" --windowless"#))
        );
    }

    #[test]
    fn a_symlinked_exec_keeps_the_link_not_its_target() {
        // The indirection is the point: a Nix profile entry or a
        // `target/release` symlink is swapped by a rebuild, and a unit
        // pinned to today's target would keep launching the old binary.
        let tmp = tempfile::tempdir().expect("tempdir");
        let real = tmp.path().join("eidola-gui-v1");
        write_executable(&real);
        let link = tmp.path().join("eidola-gui");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");

        let resolved = resolve_exec(Some(link.to_string_lossy().into_owned())).expect("resolve");
        assert!(resolved.ends_with("/eidola-gui"), "got {resolved}");
        assert!(
            !resolved.ends_with("eidola-gui-v1"),
            "the symlink must not be canonicalized away: {resolved}"
        );
    }

    #[test]
    fn a_dollar_bearing_path_reaches_the_unit_intact() {
        // The program word is not subject to environment expansion, so it
        // must be passed through verbatim. Escaping it to `$$` (which an
        // earlier revision did) made systemd look for a path that contains
        // two literal dollars — confirmed rejected by `systemd-analyze
        // verify`, while this form is accepted.
        let unit = render_unit("/srv/$app/eidola-gui");
        assert!(
            unit.contains(r#"ExecStart="/srv/$app/eidola-gui" --windowless"#),
            "the path must survive verbatim:\n{unit}"
        );
        assert!(
            !unit.contains("$$"),
            "no dollar escaping in the program word"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_non_executable_file_is_refused_rather_than_written() {
        // Existing is not enough: systemd rejects a mode-644 file with
        // `Command … is not executable: Permission denied`, so a unit naming
        // one could never start.
        let tmp = tempfile::tempdir().expect("tempdir");
        let exe = tmp.path().join("eidola-gui");
        std::fs::write(&exe, b"#!/bin/sh\n").expect("write");
        // Deliberately mode 644.
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o644)).expect("chmod");

        let err = resolve_exec(Some(exe.to_string_lossy().into_owned()))
            .expect_err("a unit that could never start must be refused here");
        assert!(err.to_string().contains("not executable"), "got {err}");

        // And the same file, made executable, is accepted — so the gate is
        // the mode bits and not the path.
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        resolve_exec(Some(exe.to_string_lossy().into_owned())).expect("now executable");
    }

    #[test]
    fn a_missing_exec_is_refused_rather_than_written() {
        let err = resolve_exec(Some("/definitely/not/here/eidola-gui".into()))
            .expect_err("a unit that can never start must be refused here");
        assert!(err.to_string().contains("no executable at"), "got {err}");
    }
}
