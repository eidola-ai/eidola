//! The node never runs where its memory can be written to durable media: at boot it
//! refuses a guest with any active swap area.
//!
//! # Why
//!
//! The prefix cache's host tier keeps idle KV in the confidential VM's own memory, and
//! the process holds prompts, outputs and the keys of requests in flight in host memory
//! anyway. That memory is encrypted by the platform and never leaves the VM, which is the
//! threat model's whole point; a swap area would page it out to a disk the VM does not
//! control. The host tier's pools are also page-locked (pinned with `cuMemHostAlloc` by
//! the CUDA executor, `mlock`ed by the CPU executor), so they could not be swapped out
//! even with swap, but nothing locks the rest of the process. Refusing swap closes it for
//! everything.
//!
//! Hibernation (suspend to disk) is not something the guest supports: the kernel writes
//! its image to a swap area, so a guest with none has nowhere to hibernate to, whatever the
//! platform allows. The swap check is the enforced part; nothing here relies on the
//! platform refusing guest hibernation on its own.
//!
//! # What is checked
//!
//! `/proc/swaps` lists the active swap areas after a header line; any line after it
//! refuses (a swap file, a partition, zram: the check makes no exception for swap that
//! happens to be in memory). A kernel built without swap support has no `/proc/swaps`,
//! and no swap: that is accepted. Any other failure to read it refuses.
//!
//! It runs on Linux, where production nodes run, for every `verified-readonly` node and
//! for any node with a host tier. A `dev-writable` node without a host tier (the dev model
//! on a developer's machine, the in-process tests) is not checked, because development
//! machines and CI runners usually have swap and those nodes serve no production traffic;
//! the gateway never routes to a `dev-writable` node. On other systems (macOS, which is
//! development only and always has a dynamic pager) there is no `/proc/swaps` to read and
//! the check does nothing; no runtime switch turns it off.

use std::path::Path;

use crate::BootError;
use crate::config::WeightsStorage;

/// The kernel's list of active swap areas.
pub const PROC_SWAPS: &str = "/proc/swaps";

/// Refuses unless the swap list at `path` (normally [`PROC_SWAPS`]) names no active swap
/// area. A missing file means the kernel has no swap support, and passes.
pub fn check_no_swap(path: &Path) -> Result<(), BootError> {
    let text = match std::fs::read(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(BootError(format!(
                "cannot read {} to check for swap: {}",
                path.display(),
                e.kind()
            )));
        }
    };
    let mut lines = text
        .split(|&b| b == b'\n')
        .filter(|l| !l.iter().all(u8::is_ascii_whitespace));
    match lines.next() {
        Some(header) if header.starts_with(b"Filename") => {}
        // Empty (no header at all) or not the format this check knows: refuse rather than
        // guess.
        None => {
            return Err(BootError(format!(
                "{} is empty; cannot tell whether swap is active",
                path.display()
            )));
        }
        Some(_) => {
            return Err(BootError(format!(
                "{} is not in the expected format; cannot tell whether swap is active",
                path.display()
            )));
        }
    }
    let active = lines.count();
    if active > 0 {
        return Err(BootError(format!(
            "the guest has {active} active swap area(s) ({}): the node's memory, the prefix \
             cache's host tier included, must never be paged to durable media; disable swap",
            path.display()
        )));
    }
    Ok(())
}

/// The boot check (see the module docs): on Linux, for a `verified-readonly` node or one
/// with a host tier (`kv_host_bytes` > 0), refuses active swap.
pub fn check_guest(storage: WeightsStorage, kv_host_bytes: u64) -> Result<(), BootError> {
    #[cfg(target_os = "linux")]
    if storage == WeightsStorage::VerifiedReadonly || kv_host_bytes > 0 {
        check_no_swap(Path::new(PROC_SWAPS))?;
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (storage, kv_host_bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swaps(text: &str) -> Result<(), BootError> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("swaps");
        std::fs::write(&path, text).unwrap();
        check_no_swap(&path)
    }

    const HEADER: &str = "Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n";

    #[test]
    fn active_swap_is_refused() {
        for area in [
            "/swapfile                               file\t\t4194300\t\t0\t\t-2\n",
            "/dev/sda2                               partition\t8388604\t\t1024\t\t-2\n",
            "/dev/zram0                              partition\t1000000\t\t0\t\t100\n",
        ] {
            let e = swaps(&format!("{HEADER}{area}")).unwrap_err();
            assert!(e.0.contains("1 active swap area"), "{e}");
        }
        let two = format!("{HEADER}/swapfile file 1 0 -2\n/dev/sda2 partition 2 0 -3\n");
        assert!(swaps(&two).unwrap_err().0.contains("2 active"));
    }

    #[test]
    fn no_swap_passes() {
        swaps(HEADER).unwrap();
        swaps(&format!("{HEADER}\n  \n")).unwrap();
        // A kernel without swap support has no list at all.
        let dir = tempfile::tempdir().unwrap();
        check_no_swap(&dir.path().join("absent")).unwrap();
    }

    #[test]
    fn an_unreadable_or_unknown_list_is_refused() {
        assert!(swaps("").is_err());
        assert!(swaps("/swapfile file 1 0 -2\n").is_err(), "no header");
        assert!(swaps("garbage\n").is_err());
        // A directory where the file should be: unreadable, refused.
        let dir = tempfile::tempdir().unwrap();
        assert!(check_no_swap(dir.path()).is_err());
    }

    /// On this machine's real list, a `dev-writable` node without a host tier is never
    /// checked; the others are, wherever `/proc/swaps` exists.
    #[test]
    fn only_production_or_tiered_nodes_are_checked() {
        check_guest(WeightsStorage::DevWritable, 0).unwrap();
        let real = check_no_swap(Path::new(PROC_SWAPS));
        let checked = cfg!(target_os = "linux");
        for (storage, bytes) in [
            (WeightsStorage::VerifiedReadonly, 0),
            (WeightsStorage::DevWritable, 1 << 30),
        ] {
            let r = check_guest(storage, bytes);
            if checked {
                assert_eq!(r.is_ok(), real.is_ok());
            } else {
                assert!(r.is_ok());
            }
        }
    }
}
