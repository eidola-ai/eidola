//! A device the Qwen3.8 executor runs on: memory, copies, the idle-KV cipher, the host
//! tier, and the kernel families ([`QwenKernels`]).
//!
//! Two implementations: [`super::cuda::CudaDevice`] (one GPU, the kernel images of the
//! committed manifest) and [`super::host::HostDevice`] (host memory, each launch run by a
//! host function: the kernels' own bodies compiled for the host, in the tests). An address
//! is a `u64`: a device address, or on the host a pointer.

use std::any::Any;

use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::secret::KvKey;

use super::kernels::QwenKernels;
use crate::Result;

/// Which kernels compute the forward (`eidola-engine-cuda/AGENTS.md` → Qwen3.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QwenProfile {
    /// Every kernel equal to the CPU reference executor bit for bit: the verification
    /// mode.
    #[default]
    Exact,
    /// The FP8 and BF16 linears on tensor cores (CUTLASS) and attention by FA2: the
    /// serving mode, within a measured bound of [`QwenProfile::Exact`] and invariant to
    /// the batch, the chunking and the cache.
    Fast,
}

impl QwenProfile {
    pub fn name(self) -> &'static str {
        match self {
            QwenProfile::Exact => "exact",
            QwenProfile::Fast => "fast",
        }
    }

    /// `exact` or `fast`, exactly.
    pub fn parse(s: &str) -> Option<QwenProfile> {
        match s {
            "exact" => Some(QwenProfile::Exact),
            "fast" => Some(QwenProfile::Fast),
            _ => None,
        }
    }
}

/// One range of [`QwenDevice::cipher_ranges`]: `words` 32-bit words at `addr` and the
/// keystream of (`key`, `nonce`) from block `counter` on.
#[derive(Clone, Copy)]
pub struct CipherRange<'k> {
    pub addr: u64,
    pub words: u32,
    pub counter: u32,
    pub key: &'k KvKey,
    pub nonce: KvNonce,
}

/// Memory, copies and kernels.
pub trait QwenDevice: QwenKernels {
    /// The profile this device computes in.
    fn profile(&self) -> QwenProfile;
    /// `bytes` zeroed bytes at an address aligned to 256, and what keeps them allocated.
    fn alloc(&self, bytes: usize) -> Result<(u64, Box<dyn Any>)>;
    /// # Safety
    ///
    /// `addr .. addr + bytes` lies inside one live allocation of this device.
    unsafe fn zero(&self, addr: u64, bytes: usize) -> Result<()>;
    /// # Safety
    ///
    /// As [`QwenDevice::zero`], for `src.len()` bytes.
    unsafe fn upload(&self, addr: u64, src: &[u8]) -> Result<()>;
    /// Copies out, after everything queued before it.
    ///
    /// # Safety
    ///
    /// As [`QwenDevice::zero`], for `dst.len()` bytes.
    unsafe fn download(&self, addr: u64, dst: &mut [u8]) -> Result<()>;
    /// # Safety
    ///
    /// Both ranges lie inside live allocations of this device and do not overlap.
    unsafe fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()>;
    /// XORs `words` 32-bit words at `addr` with the keystream of (`key`, `nonce`) from
    /// block `counter` (`eidola_engine::kv_cipher`), keeping no copy of the key.
    ///
    /// # Safety
    ///
    /// As [`QwenDevice::zero`], for `4 · words` bytes at a word-aligned address.
    unsafe fn cipher(
        &self,
        addr: u64,
        words: u32,
        counter: u32,
        key: &KvKey,
        nonce: &KvNonce,
    ) -> Result<()>;
    /// [`QwenDevice::cipher`] over every range of `ranges`, which do not overlap: on a
    /// GPU in one launch of the cipher kernel (one item list, zeroed behind the launch).
    ///
    /// # Safety
    ///
    /// As [`QwenDevice::cipher`], for each range.
    unsafe fn cipher_ranges(&self, ranges: &[CipherRange<'_>]) -> Result<()> {
        for r in ranges {
            // SAFETY: the caller's.
            unsafe { self.cipher(r.addr, r.words, r.counter, r.key, &r.nonce)? };
        }
        Ok(())
    }
    /// `bytes` zeroed bytes of host memory for the host tier (page-locked on a CUDA
    /// device), and what keeps them allocated.
    fn alloc_host(&self, bytes: usize) -> Result<(*mut u8, Box<dyn Any>)>;
    /// Queues a copy of device bytes to host memory.
    ///
    /// # Safety
    ///
    /// `src` as [`QwenDevice::zero`]; `dst .. dst + bytes` inside one live
    /// [`QwenDevice::alloc_host`] allocation nothing else touches until
    /// [`QwenDevice::synchronize`].
    unsafe fn copy_to_host(&self, dst: *mut u8, src: u64, bytes: usize) -> Result<()>;
    /// Queues a copy of host memory to device bytes.
    ///
    /// # Safety
    ///
    /// As [`QwenDevice::copy_to_host`], the other way.
    unsafe fn copy_from_host(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()>;
    /// Waits for everything queued.
    fn synchronize(&self) -> Result<()>;
    /// Records the launches `run` queues into a graph, instead of running them, and
    /// returns it instantiated and uploaded. A device that cannot capture refuses.
    fn capture(&self, run: &mut dyn FnMut() -> Result<()>) -> Result<Box<dyn Any>>;
    /// Queues a graph [`QwenDevice::capture`] returned.
    fn replay(&self, graph: &dyn Any) -> Result<()>;
    /// Free device memory, in bytes (0 where there is no device).
    fn free_memory(&self) -> Result<i64>;
}
