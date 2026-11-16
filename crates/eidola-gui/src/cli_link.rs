//! "Install command line tool" — an `eidola` link on the user's `PATH`.
//!
//! Every macOS bundle carries the command-line client from its own build at
//! `Contents/MacOS/eidola-cli` (not `eidola`: the default APFS volume is
//! case-insensitive and that name is the main binary `Eidola`). On Linux the
//! `.deb` installs `/usr/bin/eidola` and a Nix profile puts `bin/eidola` on
//! `PATH`, so only macOS needs a way to get the command onto `PATH`: a symbolic
//! link named `eidola` in [`LINK_DIR`], pointing at the **running** bundle's
//! CLI. app-core resolves the running executable through links before taking
//! its `llama-server` sibling, so the linked CLI still finds this bundle's
//! engine.
//!
//! **The split.** Everything that decides is pure and unit-tested on every
//! platform: where the running bundle's CLI is ([`bundle_cli`]), whether a
//! link names an Eidola bundle's CLI ([`is_bundle_cli`]), what the occupant of
//! the link path means ([`classify`]), and what an install would do
//! ([`decide`]). The edge is three reads ([`snapshot`]) and one write
//! ([`install`]) — the `status_item` shape, where the pure model compiles and
//! is tested everywhere and only the system calls are platform-bound.
//!
//! **The system is the source of truth.** Like the login item, nothing is
//! persisted app-side: the link on disk *is* the installation, so the row reads
//! it and re-reads after every write.
//!
//! **Never clobber.** Only a link whose own text names an Eidola bundle's CLI
//! (`<something>.app/Contents/MacOS/eidola-cli`) is ever replaced — a moved or
//! reinstalled app leaves exactly that behind. A regular file (a user's own
//! build, the standalone CLI download), a directory, or a link to anything else
//! is refused and named. The rule is enforced twice: by [`decide`] before the
//! button is offered, and again by [`INSTALL_SCRIPT`] at the moment of writing,
//! so something appearing at the path between the two is still refused rather
//! than replaced.
//!
//! **Privilege.** [`LINK_DIR`] is on the default macOS `PATH`, but it may not
//! exist and is usually root-owned. When the nearest existing directory on the
//! way to it is writable the script runs directly; otherwise it runs through
//! AppleScript's `do shell script … with administrator privileges`, the
//! standard macOS authorization prompt. **No path is ever spliced into code**:
//! the script is a constant, and the AppleScript passes the script and every
//! path as `argv`, shell-quoted by AppleScript's own `quoted form of`. A bundle
//! path containing quotes, `$( … )`, `;` or a newline therefore reaches `ln` as
//! one literal argument — pinned by running the very same AppleScript, without
//! the privilege clause, against hostile temporary paths
//! (`the_applescript_route_passes_hostile_paths_literally`).

use std::path::{Component, Path, PathBuf};

/// Where the link goes: on the default macOS `PATH` (`/etc/paths`).
pub const LINK_DIR: &str = "/usr/local/bin";
/// The command's name on `PATH`.
pub const LINK_NAME: &str = "eidola";
/// The CLI's file name inside a bundle's `Contents/MacOS/`.
pub const BUNDLE_CLI_NAME: &str = "eidola-cli";

/// Why this process cannot offer to install the command at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unavailable {
    /// The running executable is not a bundle's `Contents/MacOS/` binary — a
    /// bare `cargo run`, a test binary, or Linux.
    NotInBundle,
    /// macOS is running the bundle from an App Translocation mount (a
    /// quarantined app launched where it was downloaded). That path is
    /// randomized and read-only and goes away, so a link to it would dangle.
    Translocated,
    /// The bundle is real but has no CLI inside it.
    CliMissing,
}

/// The CLI inside the bundle the executable `exe` belongs to. Pure: reads the
/// path's shape and nothing on disk. `exe` should already be canonical.
pub fn bundle_cli(exe: &Path) -> Result<PathBuf, Unavailable> {
    let macos = exe.parent().ok_or(Unavailable::NotInBundle)?;
    let contents = macos.parent().ok_or(Unavailable::NotInBundle)?;
    let bundle = contents.parent().ok_or(Unavailable::NotInBundle)?;
    let is_bundle = macos.file_name().is_some_and(|n| n == "MacOS")
        && contents.file_name().is_some_and(|n| n == "Contents")
        && bundle.extension().is_some_and(|e| e == "app");
    if !is_bundle {
        return Err(Unavailable::NotInBundle);
    }
    if bundle
        .components()
        .any(|c| matches!(c, Component::Normal(n) if n == "AppTranslocation"))
    {
        return Err(Unavailable::Translocated);
    }
    Ok(macos.join(BUNDLE_CLI_NAME))
}

/// Whether `path` names an Eidola bundle's CLI by its shape:
/// `<anything>.app/Contents/MacOS/eidola-cli`. Lexical on purpose — a link left
/// by a moved app no longer resolves, and it is exactly the link this feature
/// may replace. The install script's `case` pattern is the same rule.
pub fn is_bundle_cli(path: &Path) -> bool {
    let mut names = path.components().rev().filter_map(|c| match c {
        Component::Normal(n) => Some(n),
        _ => None,
    });
    let (Some(cli), Some(macos), Some(contents), Some(bundle)) =
        (names.next(), names.next(), names.next(), names.next())
    else {
        return false;
    };
    cli == BUNDLE_CLI_NAME
        && macos == "MacOS"
        && contents == "Contents"
        && bundle.to_string_lossy().ends_with(".app")
}

/// What is at the link path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Occupant {
    Nothing,
    /// A symbolic link. `target` is its stored text, made absolute against the
    /// link's directory; `resolves_to` is the canonical file it reaches, when
    /// it reaches one.
    Link {
        target: PathBuf,
        resolves_to: Option<PathBuf>,
    },
    File,
    Directory,
    /// A socket, a FIFO, a device.
    Other,
}

/// What stands at the link path when it is not ours to replace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Blocker {
    File,
    Directory,
    /// A link to something that is not an Eidola bundle's CLI.
    Link(PathBuf),
    Other,
}

/// The command's state as the reader should see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkState {
    Unavailable(Unavailable),
    NotInstalled,
    /// The link resolves to this bundle's CLI.
    Installed,
    /// A link into an Eidola bundle that is not this one. `dangling` when it
    /// resolves to nothing — the app it named moved or was deleted.
    Stale {
        target: PathBuf,
        dangling: bool,
    },
    Occupied(Blocker),
    /// The link path itself could not be examined.
    Unreadable(String),
}

/// What the occupant means for the CLI at `source` (canonical).
pub fn classify(source: &Path, occupant: &Occupant) -> LinkState {
    match occupant {
        Occupant::Nothing => LinkState::NotInstalled,
        Occupant::Link {
            resolves_to: Some(resolved),
            ..
        } if resolved == source => LinkState::Installed,
        Occupant::Link {
            target,
            resolves_to,
        } if is_bundle_cli(target) => LinkState::Stale {
            target: target.clone(),
            dangling: resolves_to.is_none(),
        },
        Occupant::Link { target, .. } => LinkState::Occupied(Blocker::Link(target.clone())),
        Occupant::File => LinkState::Occupied(Blocker::File),
        Occupant::Directory => LinkState::Occupied(Blocker::Directory),
        Occupant::Other => LinkState::Occupied(Blocker::Other),
    }
}

/// Whether this user can write where the link goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirAccess {
    /// The link directory exists (otherwise the install creates it).
    pub exists: bool,
    /// The nearest existing directory on the way to it is writable by this
    /// user — the directory itself when it exists.
    pub writable: bool,
}

/// What pressing the verb would do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Nothing to do; offer no verb.
    Nothing,
    /// Create the link, or replace an Eidola-bundle link with it.
    Install { replace: bool, admin: bool },
    /// Something not ours is there; never replace it.
    Refuse(Blocker),
}

/// The pure decision: create, replace, refuse — and whether it needs an
/// administrator.
pub fn decide(state: &LinkState, access: DirAccess) -> Decision {
    let admin = !access.writable;
    match state {
        LinkState::NotInstalled => Decision::Install {
            replace: false,
            admin,
        },
        LinkState::Stale { .. } => Decision::Install {
            replace: true,
            admin,
        },
        LinkState::Occupied(blocker) => Decision::Refuse(blocker.clone()),
        LinkState::Installed | LinkState::Unavailable(_) | LinkState::Unreadable(_) => {
            Decision::Nothing
        }
    }
}

/// Everything the row needs, read in one go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// The running bundle's CLI (canonical), when there is one.
    pub source: Option<PathBuf>,
    /// The directory the link goes in.
    pub dir: PathBuf,
    pub state: LinkState,
    pub access: DirAccess,
}

impl Snapshot {
    /// The full path of the link.
    pub fn link(&self) -> PathBuf {
        self.dir.join(LINK_NAME)
    }

    pub fn decision(&self) -> Decision {
        decide(&self.state, self.access)
    }

    /// The commands a reader can run in Terminal to do what the verb would
    /// have done — offered when they decline the password prompt or the
    /// install fails. `None` when the verb would do nothing or refuse: this
    /// never suggests replacing something that is not ours.
    pub fn manual_command(&self) -> Option<String> {
        let Decision::Install { replace, admin } = self.decision() else {
            return None;
        };
        let source = self.source.as_deref()?;
        Some(manual_command(
            source,
            &self.dir,
            replace,
            admin,
            !self.access.exists,
        ))
    }
}

/// Quote `s` for a POSIX shell: one single-quoted word, with each `'` closed,
/// escaped and reopened.
pub fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// The Terminal fallback for an install. Pure, so the quoting is tested.
pub fn manual_command(
    source: &Path,
    dir: &Path,
    replace: bool,
    admin: bool,
    create_dir: bool,
) -> String {
    let sudo = if admin { "sudo " } else { "" };
    let link = dir.join(LINK_NAME);
    let ln = format!(
        "{sudo}ln -s{} {} {}",
        if replace { "f" } else { "" },
        shell_quote(&source.to_string_lossy()),
        shell_quote(&link.to_string_lossy()),
    );
    if create_dir {
        format!(
            "{sudo}mkdir -p {} && {ln}",
            shell_quote(&dir.to_string_lossy())
        )
    } else {
        ln
    }
}

/// The one write, as a constant `sh` script taking `dir src dst` as `$1 $2
/// $3`. It re-applies the never-clobber rule at the moment of writing: an
/// existing link is removed only when its text names an Eidola bundle's CLI
/// (the [`is_bundle_cli`] shape), anything else exits 3, and `ln -s` without
/// `-f` refuses whatever might appear in between.
pub const INSTALL_SCRIPT: &str = r#"dir=$1 src=$2 dst=$3
/bin/mkdir -p -- "$dir" || exit 1
if [ -L "$dst" ]; then
  case $(/usr/bin/readlink -- "$dst") in
    *.app/Contents/MacOS/eidola-cli) /bin/rm -f -- "$dst" || exit 1 ;;
    *) exit 3 ;;
  esac
fi
if [ -e "$dst" ] || [ -L "$dst" ]; then exit 3; fi
exec /bin/ln -s -- "$src" "$dst"
"#;

/// The exit status [`INSTALL_SCRIPT`] uses for "something not ours is there".
const REFUSED_STATUS: i32 = 3;
/// AppleScript's "User canceled." — the password prompt was dismissed.
const USER_CANCELED: i32 = -128;

/// The AppleScript that runs [`INSTALL_SCRIPT`]. Every input arrives as
/// `argv` and is shell-quoted by `quoted form of`; nothing is interpolated
/// into source. `privileged` adds the authorization clause — the only
/// difference between what ships and what the tests run.
pub fn applescript(privileged: bool) -> String {
    format!(
        "on run argv\n\
         do shell script \"/bin/sh -c \" & quoted form of (item 1 of argv) & \" eidola-install \" \
         & quoted form of (item 2 of argv) & \" \" & quoted form of (item 3 of argv) & \" \" \
         & quoted form of (item 4 of argv){}\n\
         end run",
        if privileged {
            " with administrator privileges"
        } else {
            ""
        }
    )
}

/// How an install did not happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallFailure {
    /// The reader dismissed the administrator prompt. Nothing changed.
    Declined,
    /// Something that is not an Eidola link is at the path. Nothing changed.
    Refused,
    /// Anything else, in the words of whatever failed.
    Failed(String),
}

/// How [`install`] runs the script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// `/bin/sh` as this user.
    Direct,
    /// Through `osascript`; `privileged` asks for an administrator.
    AppleScript { privileged: bool },
}

impl Route {
    /// The route a decision calls for.
    pub fn for_admin(admin: bool) -> Self {
        if admin {
            Route::AppleScript { privileged: true }
        } else {
            Route::Direct
        }
    }
}

/// Map an `osascript` failure: its stderr ends `… (N)`, with `N` the error
/// number — the script's own exit status, or AppleScript's.
pub fn applescript_failure(stderr: &str) -> InstallFailure {
    let text = stderr.trim();
    let code = text
        .strip_suffix(')')
        .and_then(|t| t.rsplit_once('('))
        .and_then(|(_, n)| n.parse::<i32>().ok());
    match code {
        Some(USER_CANCELED) => InstallFailure::Declined,
        Some(REFUSED_STATUS) => InstallFailure::Refused,
        _ => InstallFailure::Failed(
            text.split_once("execution error: ")
                .map_or(text, |(_, m)| m)
                .to_string(),
        ),
    }
}

/// Link `dir/eidola` to `source`. Blocking: the privileged route waits on the
/// reader's password, so callers run this off the main thread.
#[cfg(unix)]
pub fn install(source: &Path, dir: &Path, route: Route) -> Result<(), InstallFailure> {
    use std::process::Command;
    let link = dir.join(LINK_NAME);
    let args = [dir.as_os_str(), source.as_os_str(), link.as_os_str()];
    let output = match route {
        Route::Direct => Command::new("/bin/sh")
            .arg("-c")
            .arg(INSTALL_SCRIPT)
            .arg("eidola-install")
            .args(args)
            .output(),
        Route::AppleScript { privileged } => Command::new("/usr/bin/osascript")
            .arg("-e")
            .arg(applescript(privileged))
            .arg(INSTALL_SCRIPT)
            .args(args)
            .output(),
    }
    .map_err(|e| InstallFailure::Failed(e.to_string()))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    match route {
        Route::Direct if output.status.code() == Some(REFUSED_STATUS) => {
            Err(InstallFailure::Refused)
        }
        Route::Direct => Err(InstallFailure::Failed(stderr.trim().to_string())),
        Route::AppleScript { .. } => Err(applescript_failure(&stderr)),
    }
}

/// Read what is at `link`.
#[cfg(unix)]
pub fn observe(link: &Path) -> std::io::Result<Occupant> {
    let meta = match std::fs::symlink_metadata(link) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Occupant::Nothing),
        Err(e) => return Err(e),
    };
    let kind = meta.file_type();
    if kind.is_symlink() {
        let stored = std::fs::read_link(link)?;
        let target = match link.parent() {
            Some(parent) => parent.join(&stored),
            None => stored,
        };
        let resolves_to = std::fs::canonicalize(link).ok();
        Ok(Occupant::Link {
            target,
            resolves_to,
        })
    } else if kind.is_dir() {
        Ok(Occupant::Directory)
    } else if kind.is_file() {
        Ok(Occupant::File)
    } else {
        Ok(Occupant::Other)
    }
}

/// Whether this user can write into `dir`, or create it.
#[cfg(unix)]
pub fn dir_access(dir: &Path) -> DirAccess {
    use std::os::unix::ffi::OsStrExt as _;
    let exists = dir.is_dir();
    let mut probe = Some(dir);
    while let Some(candidate) = probe {
        if candidate.is_dir() {
            let writable = std::ffi::CString::new(candidate.as_os_str().as_bytes())
                // SAFETY: a valid NUL-terminated path; `access` only reads it.
                .map(|c| unsafe { libc::access(c.as_ptr(), libc::W_OK) } == 0)
                .unwrap_or(false);
            return DirAccess { exists, writable };
        }
        probe = candidate.parent();
    }
    DirAccess {
        exists,
        writable: false,
    }
}

/// Read the state for the CLI belonging to the executable `exe`, linked into
/// `dir`. The production call is [`snapshot`]; this is its seam.
#[cfg(unix)]
pub fn snapshot_for(exe: &Path, dir: &Path) -> Snapshot {
    let unavailable = |why| Snapshot {
        source: None,
        dir: dir.to_path_buf(),
        state: LinkState::Unavailable(why),
        access: DirAccess {
            exists: false,
            writable: false,
        },
    };
    let exe = std::fs::canonicalize(exe).unwrap_or_else(|_| exe.to_path_buf());
    let source = match bundle_cli(&exe) {
        Ok(source) => source,
        Err(why) => return unavailable(why),
    };
    let Ok(source) = std::fs::canonicalize(&source) else {
        return unavailable(Unavailable::CliMissing);
    };
    if !source.is_file() {
        return unavailable(Unavailable::CliMissing);
    }
    let state = match observe(&dir.join(LINK_NAME)) {
        Ok(occupant) => classify(&source, &occupant),
        Err(e) => LinkState::Unreadable(e.to_string()),
    };
    Snapshot {
        source: Some(source),
        dir: dir.to_path_buf(),
        state,
        access: dir_access(dir),
    }
}

/// The state for the running process, linked into [`LINK_DIR`].
#[cfg(unix)]
pub fn snapshot() -> Snapshot {
    match std::env::current_exe() {
        Ok(exe) => snapshot_for(&exe, Path::new(LINK_DIR)),
        Err(_) => Snapshot {
            source: None,
            dir: PathBuf::from(LINK_DIR),
            state: LinkState::Unavailable(Unavailable::NotInBundle),
            access: DirAccess {
                exists: false,
                writable: false,
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLI: &str = "/Applications/Eidola.app/Contents/MacOS/eidola-cli";

    #[test]
    fn the_cli_is_found_beside_a_bundles_main_binary() {
        assert_eq!(
            bundle_cli(Path::new("/Applications/Eidola.app/Contents/MacOS/Eidola")),
            Ok(PathBuf::from(CLI))
        );
        // Wherever the bundle is — the running one, never `/Applications`.
        assert_eq!(
            bundle_cli(Path::new(
                "/Users/r/Apps/Eidola 2.app/Contents/MacOS/Eidola"
            )),
            Ok(PathBuf::from(
                "/Users/r/Apps/Eidola 2.app/Contents/MacOS/eidola-cli"
            ))
        );
    }

    #[test]
    fn a_binary_outside_a_bundle_offers_nothing() {
        for exe in [
            "/Users/r/eidola/target/debug/eidola-gui",
            "/usr/libexec/eidola/eidola-gui",
            "/Eidola",
            "/Contents/MacOS/Eidola",
            "/x/Eidola/Contents/MacOS/Eidola",
            "/x/Eidola.app/Contents/Resources/Eidola",
        ] {
            assert_eq!(
                bundle_cli(Path::new(exe)),
                Err(Unavailable::NotInBundle),
                "{exe}"
            );
        }
    }

    #[test]
    fn a_translocated_bundle_is_not_linked_to() {
        assert_eq!(
            bundle_cli(Path::new(
                "/private/var/folders/x/T/AppTranslocation/1234-ABCD/d/Eidola.app/Contents/MacOS/Eidola"
            )),
            Err(Unavailable::Translocated)
        );
    }

    #[test]
    fn only_an_eidola_bundles_cli_counts_as_ours() {
        assert!(is_bundle_cli(Path::new(CLI)));
        assert!(is_bundle_cli(Path::new(
            "/usr/local/bin/../../Volumes/Old/Eidola.app/Contents/MacOS/eidola-cli"
        )));
        for other in [
            "/Users/r/eidola/target/release/eidola-cli",
            "/Users/r/bin/eidola",
            "/Applications/Eidola.app/Contents/MacOS/Eidola",
            "/Applications/Eidola/Contents/MacOS/eidola-cli",
            "/Applications/Eidola.app/Contents/Resources/eidola-cli",
            "eidola-cli",
        ] {
            assert!(!is_bundle_cli(Path::new(other)), "{other}");
        }
    }

    fn link(target: &str, resolves_to: Option<&str>) -> Occupant {
        Occupant::Link {
            target: PathBuf::from(target),
            resolves_to: resolves_to.map(PathBuf::from),
        }
    }

    #[test]
    fn the_occupant_is_classified_against_the_running_bundle() {
        let source = Path::new(CLI);
        assert_eq!(
            classify(source, &Occupant::Nothing),
            LinkState::NotInstalled
        );
        assert_eq!(
            classify(source, &link(CLI, Some(CLI))),
            LinkState::Installed
        );
        // A link that reaches this CLI by another spelling is still installed.
        assert_eq!(
            classify(
                source,
                &link(
                    "/usr/local/bin/../../Applications/Eidola.app/Contents/MacOS/eidola-cli",
                    Some(CLI)
                )
            ),
            LinkState::Installed
        );
        // The app moved: the old link dangles.
        let moved = "/Users/r/Downloads/Eidola.app/Contents/MacOS/eidola-cli";
        assert_eq!(
            classify(source, &link(moved, None)),
            LinkState::Stale {
                target: PathBuf::from(moved),
                dangling: true
            }
        );
        // Another copy that still exists.
        assert_eq!(
            classify(source, &link(moved, Some(moved))),
            LinkState::Stale {
                target: PathBuf::from(moved),
                dangling: false
            }
        );
    }

    #[test]
    fn what_is_not_ours_is_named_and_never_replaced() {
        let source = Path::new(CLI);
        let own_build = "/Users/r/eidola/target/release/eidola-cli";
        for (occupant, blocker) in [
            (Occupant::File, Blocker::File),
            (Occupant::Directory, Blocker::Directory),
            (Occupant::Other, Blocker::Other),
            (
                link(own_build, Some(own_build)),
                Blocker::Link(PathBuf::from(own_build)),
            ),
            (
                link(own_build, None),
                Blocker::Link(PathBuf::from(own_build)),
            ),
        ] {
            let state = classify(source, &occupant);
            assert_eq!(state, LinkState::Occupied(blocker.clone()));
            for writable in [true, false] {
                assert_eq!(
                    decide(
                        &state,
                        DirAccess {
                            exists: true,
                            writable
                        }
                    ),
                    Decision::Refuse(blocker.clone())
                );
            }
        }
    }

    #[test]
    fn the_decision_says_create_replace_or_nothing_and_who_may_do_it() {
        let writable = DirAccess {
            exists: true,
            writable: true,
        };
        let locked = DirAccess {
            exists: true,
            writable: false,
        };
        assert_eq!(
            decide(&LinkState::NotInstalled, writable),
            Decision::Install {
                replace: false,
                admin: false
            }
        );
        assert_eq!(
            decide(&LinkState::NotInstalled, locked),
            Decision::Install {
                replace: false,
                admin: true
            }
        );
        let stale = LinkState::Stale {
            target: PathBuf::from(CLI),
            dangling: true,
        };
        assert_eq!(
            decide(&stale, locked),
            Decision::Install {
                replace: true,
                admin: true
            }
        );
        for state in [
            LinkState::Installed,
            LinkState::Unavailable(Unavailable::NotInBundle),
            LinkState::Unreadable("denied".into()),
        ] {
            assert_eq!(decide(&state, writable), Decision::Nothing);
        }
    }

    #[test]
    fn the_terminal_fallback_quotes_every_path() {
        let source = Path::new("/Users/r/it's here/Eidola.app/Contents/MacOS/eidola-cli");
        assert_eq!(
            manual_command(source, Path::new("/usr/local/bin"), false, true, true),
            "sudo mkdir -p '/usr/local/bin' && sudo ln -s \
             '/Users/r/it'\\''s here/Eidola.app/Contents/MacOS/eidola-cli' '/usr/local/bin/eidola'"
        );
        assert_eq!(
            manual_command(Path::new(CLI), Path::new("/opt/bin"), true, false, false),
            format!("ln -sf '{CLI}' '/opt/bin/eidola'")
        );
    }

    #[test]
    fn applescript_failures_are_read_by_their_number() {
        assert_eq!(
            applescript_failure("0:17: execution error: User canceled. (-128)\n"),
            InstallFailure::Declined
        );
        assert_eq!(
            applescript_failure(
                "7:119: execution error: The command exited with a non-zero status. (3)"
            ),
            InstallFailure::Refused
        );
        assert_eq!(
            applescript_failure("7:119: execution error: mkdir: /x: Read-only file system (1)"),
            InstallFailure::Failed("mkdir: /x: Read-only file system (1)".into())
        );
    }

    #[test]
    fn only_the_shipped_script_asks_for_privileges() {
        assert!(applescript(true).contains("with administrator privileges"));
        assert!(!applescript(false).contains("administrator"));
        // Nothing but `argv` reaches the shell, and only through
        // `quoted form of`.
        let script = applescript(true);
        assert_eq!(script.matches("quoted form of").count(), 4);
        assert_eq!(script.matches("item ").count(), 4);
    }

    /// The edge, against temporary directories only — never `LINK_DIR`, and
    /// never the privileged route (that is a real authorization prompt; the
    /// manual-verification checklist covers it). macOS only: the script names
    /// `/bin` and `/usr/bin` tools a Nix build sandbox does not have, and
    /// macOS is the only platform that runs it.
    #[cfg(target_os = "macos")]
    mod edge {
        use super::super::*;
        use std::fs;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Scratch(PathBuf);
        impl Scratch {
            fn new() -> Self {
                static N: AtomicUsize = AtomicUsize::new(0);
                let dir = std::env::temp_dir().join(format!(
                    "eidola-cli-link-{}-{}",
                    std::process::id(),
                    N.fetch_add(1, Ordering::Relaxed)
                ));
                fs::create_dir_all(&dir).unwrap();
                Self(fs::canonicalize(&dir).unwrap())
            }
        }
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }

        /// A bundle at `root` with a main binary and a CLI; returns the main
        /// binary.
        fn bundle(root: &Path) -> PathBuf {
            let macos = root.join("Contents/MacOS");
            fs::create_dir_all(&macos).unwrap();
            fs::write(macos.join("Eidola"), b"").unwrap();
            fs::write(macos.join(BUNDLE_CLI_NAME), b"").unwrap();
            macos.join("Eidola")
        }

        #[test]
        fn a_snapshot_follows_the_link_through_install_move_and_reinstall() {
            let s = Scratch::new();
            let dir = s.0.join("missing/bin");
            let first = bundle(&s.0.join("Eidola.app"));

            let snap = snapshot_for(&first, &dir);
            assert_eq!(snap.state, LinkState::NotInstalled);
            assert!(!snap.access.exists && snap.access.writable);
            assert_eq!(
                snap.decision(),
                Decision::Install {
                    replace: false,
                    admin: false
                }
            );
            install(snap.source.as_deref().unwrap(), &dir, Route::Direct).unwrap();
            assert_eq!(snapshot_for(&first, &dir).state, LinkState::Installed);

            // The app moves: the link dangles, and the moved app replaces it.
            let moved_root = s.0.join("Moved/Eidola.app");
            fs::create_dir_all(moved_root.parent().unwrap()).unwrap();
            fs::rename(s.0.join("Eidola.app"), &moved_root).unwrap();
            let moved = moved_root.join("Contents/MacOS/Eidola");
            let snap = snapshot_for(&moved, &dir);
            assert_eq!(
                snap.state,
                LinkState::Stale {
                    target: s.0.join("Eidola.app/Contents/MacOS/eidola-cli"),
                    dangling: true
                }
            );
            install(snap.source.as_deref().unwrap(), &dir, Route::Direct).unwrap();
            assert_eq!(snapshot_for(&moved, &dir).state, LinkState::Installed);

            // A second, still-present copy is "another copy", and is replaced
            // the same way.
            let other = bundle(&s.0.join("Other/Eidola.app"));
            let snap = snapshot_for(&other, &dir);
            assert!(matches!(
                snap.state,
                LinkState::Stale {
                    dangling: false,
                    ..
                }
            ));
        }

        #[test]
        fn the_script_refuses_what_is_not_ours_even_when_asked() {
            let s = Scratch::new();
            let exe = bundle(&s.0.join("Eidola.app"));
            let source = s.0.join("Eidola.app/Contents/MacOS/eidola-cli");
            let dir = s.0.join("bin");
            fs::create_dir_all(&dir).unwrap();
            let link = dir.join(LINK_NAME);

            // A regular file — the standalone CLI, or the reader's own build.
            fs::write(&link, b"#!/bin/sh\necho mine\n").unwrap();
            assert_eq!(
                snapshot_for(&exe, &dir).state,
                LinkState::Occupied(Blocker::File)
            );
            // Asked anyway (the race the decision cannot see), the script still
            // refuses, and the file is untouched.
            assert_eq!(
                install(&source, &dir, Route::Direct),
                Err(InstallFailure::Refused)
            );
            assert_eq!(fs::read(&link).unwrap(), b"#!/bin/sh\necho mine\n");

            // A link to something else.
            fs::remove_file(&link).unwrap();
            let own = s.0.join("target/release/eidola-cli");
            std::os::unix::fs::symlink(&own, &link).unwrap();
            assert_eq!(
                snapshot_for(&exe, &dir).state,
                LinkState::Occupied(Blocker::Link(own.clone()))
            );
            assert_eq!(
                install(&source, &dir, Route::Direct),
                Err(InstallFailure::Refused)
            );
            assert_eq!(fs::read_link(&link).unwrap(), own);

            // A directory.
            fs::remove_file(&link).unwrap();
            fs::create_dir(&link).unwrap();
            assert_eq!(
                snapshot_for(&exe, &dir).state,
                LinkState::Occupied(Blocker::Directory)
            );
            assert_eq!(
                install(&source, &dir, Route::Direct),
                Err(InstallFailure::Refused)
            );
            assert!(link.is_dir());
        }

        #[test]
        fn a_read_only_directory_needs_an_administrator() {
            use std::os::unix::fs::PermissionsExt as _;
            let s = Scratch::new();
            let exe = bundle(&s.0.join("Eidola.app"));
            let dir = s.0.join("locked");
            fs::create_dir_all(&dir).unwrap();
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
            let snap = snapshot_for(&exe, &dir);
            let below = snapshot_for(&exe, &dir.join("bin"));
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
            assert_eq!(
                snap.decision(),
                Decision::Install {
                    replace: false,
                    admin: true
                }
            );
            // A missing directory under a locked one needs one too.
            assert!(!below.access.exists && !below.access.writable);
        }

        /// The quoting the privileged route depends on, exercised through the
        /// exact AppleScript that ships minus its authorization clause. The
        /// bundle and link paths carry every shell and AppleScript
        /// metacharacter, including two command substitutions that would each
        /// create a file if any layer interpolated them.
        #[test]
        fn the_applescript_route_passes_hostile_paths_literally() {
            let s = Scratch::new();
            let base = s.0.to_string_lossy().into_owned();
            let hostile =
                format!("q'uo\"te $(touch {base}/pwned) ; `touch {base}/pwned2` \\ & \n|");
            let root = s.0.join(&hostile).join("Eidola.app");
            let exe = bundle(&root);
            let dir = s.0.join("bin \"' $(touch x)");
            let snap = snapshot_for(&exe, &dir);
            let source = snap.source.clone().unwrap();
            assert!(source.to_string_lossy().contains("$(touch"));

            install(&source, &dir, Route::AppleScript { privileged: false }).unwrap();

            assert_eq!(fs::read_link(dir.join(LINK_NAME)).unwrap(), source);
            assert_eq!(snapshot_for(&exe, &dir).state, LinkState::Installed);
            assert!(!s.0.join("pwned").exists(), "a $( ) was evaluated");
            assert!(!s.0.join("pwned2").exists(), "a backtick was evaluated");

            // The refusal travels back through AppleScript as its own number.
            fs::remove_file(dir.join(LINK_NAME)).unwrap();
            fs::write(dir.join(LINK_NAME), b"").unwrap();
            assert_eq!(
                install(&source, &dir, Route::AppleScript { privileged: false }),
                Err(InstallFailure::Refused)
            );
        }
    }
}
