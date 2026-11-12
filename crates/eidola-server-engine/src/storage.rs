//! The `verified-readonly` storage check: refuse weights that are not on read-only,
//! immutable-looking storage, before any weights file is opened.
//!
//! # What this establishes, and what it does not
//!
//! It refuses deployments that are clearly not immutable. It does not make the bytes
//! immutable and does not verify them: in production the weights are a dm-verity volume,
//! and the kernel checking every read against the measured root hash is what binds the
//! bytes this process uses (at boot and on every later read of the memory-mapped shards)
//! to the bytes the weights hash covered.
//!
//! * **Every platform**: each path's mount is read-only (`statvfs` `ST_RDONLY`). That is a
//!   per-mount flag: a read-only bind mount can sit over a filesystem still writable
//!   through another mount of the same superblock, so on its own it is dev-grade. macOS
//!   (development only) stops here.
//! * **Linux**: additionally, each path resolves (`/proc/self/mountinfo`, by the file's
//!   device id and the longest mount point containing its canonical path, the last such
//!   mount winning as it shadows earlier ones) to a mount whose mount options **and
//!   superblock options** both contain `ro`. A filesystem on a dm-verity device is always
//!   mounted with a read-only superblock (the device is read-only), so production passes;
//!   the bind-mount alias above fails. A superblock that is read-only now can still be
//!   remounted read-write by a privileged process in the same VM later; nothing in this
//!   process can rule that out, which is why dm-verity, not this check, is the binding.

use std::path::{Path, PathBuf};

use crate::model::ModelError;

/// Refuses unless `path` is on read-only storage as described in the module docs. Only
/// metadata is read (`statvfs`, `stat`, `/proc/self/mountinfo`); `path` is not opened.
/// `name` is what refusals call it.
pub fn require_immutable(path: &Path, name: &str) -> Result<(), ModelError> {
    let stat = rustix::fs::statvfs(path)
        .map_err(|e| ModelError(format!("cannot inspect the filesystem of {name}: {e}")))?;
    if !stat.f_flag.contains(rustix::fs::StatVfsMountFlags::RDONLY) {
        return Err(ModelError(format!(
            "{name} is on a writable filesystem; verified-readonly weights must be on a \
             read-only mount"
        )));
    }
    #[cfg(target_os = "linux")]
    require_read_only_superblock(path, name)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn require_read_only_superblock(path: &Path, name: &str) -> Result<(), ModelError> {
    use std::os::unix::fs::MetadataExt;
    // The device from `path` itself (which may be a pinned `/proc/self/fd` path); the
    // canonical text only places it among the mount points.
    let dev = std::fs::metadata(path)
        .map_err(|e| ModelError(format!("cannot inspect {name}: {}", e.kind())))?
        .dev();
    let canonical = std::fs::canonicalize(path)
        .map_err(|e| ModelError(format!("cannot resolve {name}: {}", e.kind())))?;
    // Bytes, not text: mount points are byte strings, and an unrelated mount whose path is
    // not UTF-8 must not stop this check.
    let text = std::fs::read("/proc/self/mountinfo")
        .map_err(|e| ModelError(format!("cannot read /proc/self/mountinfo: {}", e.kind())))?;
    let mounts =
        mountinfo::parse(&text).map_err(|e| ModelError(format!("/proc/self/mountinfo: {e}")))?;
    let mount = mountinfo::resolve(&mounts, &canonical, mountinfo::linux_dev(dev))
        .ok_or_else(|| ModelError(format!("cannot find the mount holding {name}")))?;
    mountinfo::check(mount).map_err(|e| ModelError(format!("{name}: {e}")))
}

/// `/proc/self/mountinfo` parsing and path resolution (pure, so it is tested on every
/// platform against captured samples).
pub mod mountinfo {
    use super::*;

    /// One mount, as far as this check needs it.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Mount {
        /// `major:minor` of the mounted filesystem.
        pub dev: (u32, u32),
        /// Where it is mounted (octal escapes decoded).
        pub mount_point: PathBuf,
        /// Per-mount options contain `ro`.
        pub mount_ro: bool,
        /// Superblock options contain `ro`.
        pub super_ro: bool,
    }

    /// Parses mountinfo text (proc(5): `id parent maj:min root mount-point options
    /// [optional fields…] - fstype source super-options`).
    pub fn parse(text: &[u8]) -> Result<Vec<Mount>, String> {
        text.split(|b| *b == b'\n')
            .filter(|l| !l.trim_ascii().is_empty())
            .enumerate()
            .map(|(i, line)| parse_line(line).ok_or_else(|| format!("line {} is malformed", i + 1)))
            .collect()
    }

    fn parse_line(line: &[u8]) -> Option<Mount> {
        use std::os::unix::ffi::OsStringExt;
        let fields: Vec<&[u8]> = line.split(|b| *b == b' ').collect();
        let (major, minor) = std::str::from_utf8(fields.get(2)?).ok()?.split_once(':')?;
        let dev = (major.parse().ok()?, minor.parse().ok()?);
        let mount_point = PathBuf::from(std::ffi::OsString::from_vec(unescape(fields.get(4)?)?));
        let mount_ro = has_ro(fields.get(5)?);
        let sep = fields.iter().skip(6).position(|f| *f == b"-")? + 6;
        let super_ro = has_ro(fields.get(sep + 3)?);
        Some(Mount {
            dev,
            mount_point,
            mount_ro,
            super_ro,
        })
    }

    fn has_ro(options: &[u8]) -> bool {
        options.split(|b| *b == b',').any(|o| o == b"ro")
    }

    /// Decodes the kernel's `\ooo` octal escapes (space, tab, newline, backslash).
    fn unescape(b: &[u8]) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'\\' {
                let digits = std::str::from_utf8(b.get(i + 1..i + 4)?).ok()?;
                out.push(u8::from_str_radix(digits, 8).ok()?);
                i += 4;
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        Some(out)
    }

    /// The mount holding `canonical` (an absolute, canonical path) on device `dev`: the
    /// longest mount point that contains it, the last one listed winning a tie (a later
    /// mount on the same point shadows an earlier one).
    pub fn resolve<'a>(
        mounts: &'a [Mount],
        canonical: &Path,
        dev: (u32, u32),
    ) -> Option<&'a Mount> {
        let mut best: Option<&Mount> = None;
        for m in mounts {
            if m.dev != dev || !canonical.starts_with(&m.mount_point) {
                continue;
            }
            let longer = best.is_none_or(|b| {
                m.mount_point.components().count() >= b.mount_point.components().count()
            });
            if longer {
                best = Some(m);
            }
        }
        best
    }

    /// Refuses a mount that is not read-only both per mount and per superblock.
    pub fn check(m: &Mount) -> Result<(), String> {
        match (m.mount_ro, m.super_ro) {
            (true, true) => Ok(()),
            (false, _) => Err(format!("{} is mounted read-write", m.mount_point.display())),
            (true, false) => Err(format!(
                "{} is a read-only mount of a filesystem whose superblock is writable \
                 (another mount could change the weights)",
                m.mount_point.display()
            )),
        }
    }

    /// Splits a Linux `st_dev` into `(major, minor)` (glibc's encoding).
    pub fn linux_dev(dev: u64) -> (u32, u32) {
        let major = ((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff);
        let minor = (dev & 0xff) | ((dev >> 12) & !0xff);
        (major as u32, minor as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::mountinfo::*;
    use super::*;

    /// Captured from a Linux VM (abridged), plus the cases below.
    const SAMPLE: &str = "\
22 1 252:1 / / rw,relatime shared:1 - ext4 /dev/vda1 rw,errors=remount-ro
25 22 0:22 / /proc rw,nosuid,nodev,noexec,relatime shared:12 - proc proc rw
30 22 252:1 /srv/weights /weights ro,relatime shared:1 - ext4 /dev/vda1 rw,errors=remount-ro
31 22 253:3 / /verity ro,relatime shared:20 - erofs /dev/mapper/weights ro,user_xattr,acl
32 22 253:4 / /mnt/my\\040weights ro,noatime - squashfs /dev/loop4 ro
33 22 7:9 / /stack rw,relatime - ext4 /dev/loop9 rw
34 22 7:9 / /stack ro,relatime - ext4 /dev/loop9 ro
";

    fn mounts() -> Vec<Mount> {
        parse(SAMPLE.as_bytes()).unwrap()
    }

    fn verdict(path: &str, dev: (u32, u32)) -> Result<(), String> {
        let m = mounts();
        let mount = resolve(&m, Path::new(path), dev).ok_or("unresolved")?;
        check(mount)
    }

    #[test]
    fn a_read_only_bind_over_a_writable_superblock_is_refused() {
        let e = verdict("/weights/model.safetensors", (252, 1)).unwrap_err();
        assert!(e.contains("superblock is writable"), "{e}");
    }

    #[test]
    fn a_read_only_superblock_is_accepted() {
        verdict("/verity/model.safetensors", (253, 3)).unwrap();
        verdict("/mnt/my weights/config.json", (253, 4)).unwrap();
    }

    #[test]
    fn a_read_write_mount_is_refused() {
        let e = verdict("/home/x/model.safetensors", (252, 1)).unwrap_err();
        assert!(e.contains("read-write"), "{e}");
    }

    #[test]
    fn the_last_mount_on_a_point_shadows_earlier_ones() {
        verdict("/stack/a", (7, 9)).unwrap();
        let m: Vec<Mount> = mounts()
            .into_iter()
            .filter(|m| m.mount_point != Path::new("/stack") || !m.super_ro)
            .collect();
        assert!(check(resolve(&m, Path::new("/stack/a"), (7, 9)).unwrap()).is_err());
    }

    #[test]
    fn resolution_needs_the_device_and_a_component_prefix() {
        let m = mounts();
        assert!(
            resolve(&m, Path::new("/verity/x"), (252, 1))
                .is_none_or(|r| r.mount_point == Path::new("/"))
        );
        // `/weightsx` is not under `/weights`.
        let r = resolve(&m, Path::new("/weightsx/a"), (252, 1)).unwrap();
        assert_eq!(r.mount_point, Path::new("/"));
        assert_eq!(resolve(&m, Path::new("/verity/x"), (9, 9)), None);
    }

    #[test]
    fn a_non_utf8_mount_point_elsewhere_does_not_block_resolution() {
        let mut text = SAMPLE.as_bytes().to_vec();
        text.extend_from_slice(b"40 22 8:1 / /media/caf\xe9 rw - vfat /dev/sdb1 rw\n");
        let m = parse(&text).unwrap();
        check(resolve(&m, Path::new("/verity/model.safetensors"), (253, 3)).unwrap()).unwrap();
        use std::os::unix::ffi::OsStrExt;
        assert_eq!(
            m.last().unwrap().mount_point.as_os_str().as_bytes(),
            b"/media/caf\xe9"
        );
    }

    #[test]
    fn malformed_lines_and_devices() {
        assert!(parse(b"garbage").is_err());
        assert!(
            parse(b"1 2 3:4 / /x rw - ext4 src").is_err(),
            "no super options"
        );
        assert_eq!(linux_dev(0xfd03), (253, 3));
        let (major, minor) = (0x0012_3abc_u64, 0x0006_7845_u64);
        let dev = (minor & 0xff)
            | ((major & 0xfff) << 8)
            | ((minor & !0xff) << 12)
            | ((major & !0xfff) << 32);
        assert_eq!(linux_dev(dev), (0x0012_3abc, 0x0006_7845));
    }

    #[test]
    fn a_writable_directory_is_refused() {
        let e = require_immutable(&std::env::temp_dir(), "tmp").unwrap_err();
        assert!(e.to_string().contains("writable"), "{e}");
    }

    /// Runs where a read-only, superblock-read-only mount is available (a container or
    /// CI: mount one and set `EIDOLA_TEST_READ_ONLY_DIR`); skipped otherwise.
    #[test]
    fn a_read_only_mount_is_accepted() {
        let Some(dir) = std::env::var_os("EIDOLA_TEST_READ_ONLY_DIR") else {
            return;
        };
        require_immutable(Path::new(&dir), "dir").unwrap();
    }

    /// The other half, where such a mount is available: a read-only bind of a writable
    /// filesystem (`EIDOLA_TEST_READ_ONLY_BIND_DIR`) is refused.
    #[test]
    fn a_read_only_bind_of_a_writable_filesystem_is_refused() {
        let Some(dir) = std::env::var_os("EIDOLA_TEST_READ_ONLY_BIND_DIR") else {
            return;
        };
        let e = require_immutable(Path::new(&dir), "dir").unwrap_err();
        assert!(
            cfg!(not(target_os = "linux")) || e.to_string().contains("superblock"),
            "{e}"
        );
    }
}
