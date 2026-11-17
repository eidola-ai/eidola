//! [`HostDevice`]: host memory, each kernel launch run by a host function.
//!
//! The executor's tests give it the kernels' own bodies compiled for the host
//! (`oracle/qwen_launch_host.cc`): every launch first goes through its family's grid
//! function ([`super::kernels`]), exactly as a device launch does, and then runs the body
//! over that grid. So without a GPU the executor's scenarios exercise the structs it
//! fills, the wrappers' preconditions, the grids and the C++ a GPU compiles. It computes
//! only the exact profile.

use std::any::Any;
use std::ffi::c_void;

use eidola_engine::kv_cipher::{KvKeystream, KvNonce};
use eidola_engine::secret::KvKey;

use super::abi::*;
use super::device::{QwenDevice, QwenProfile};
use super::kernels::*;
use crate::Result;

/// Runs one launch: the family's name (the oracle's `qwen_host_<family>`), its parameter
/// struct, and the grid's x and y.
pub type HostLaunch = dyn Fn(&'static str, *const c_void, u32, u32) + Send + Sync;

/// Host memory with a host function per kernel family.
pub struct HostDevice {
    launch: Box<HostLaunch>,
}

impl HostDevice {
    pub fn new(launch: Box<HostLaunch>) -> HostDevice {
        HostDevice { launch }
    }

    fn run<T>(&self, family: &'static str, p: &T, grid: Result<[u32; 3]>) -> Result<()> {
        let g = grid?;
        (self.launch)(family, (p as *const T).cast(), g[0], g[1]);
        Ok(())
    }
}

/// A host slice at `addr`.
///
/// # Safety
///
/// `addr .. addr + len · size_of::<T>()` lies inside one live allocation, aligned for
/// `T`, and nothing else holds a reference into it.
unsafe fn wr<'a, T>(addr: u64, len: usize) -> &'a mut [T] {
    // SAFETY: the caller's range.
    unsafe { std::slice::from_raw_parts_mut(addr as *mut T, len) }
}

impl QwenKernels for HostDevice {
    unsafe fn embed(&self, p: &EmbedParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("embed", p, embed_grid(p))
    }
    unsafe fn norm(&self, p: &NormParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("norm", p, norm_grid(p))
    }
    unsafe fn quant(&self, p: &QuantParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("quant", p, quant_grid(p))
    }
    unsafe fn lin8(&self, p: &Lin8Params) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("lin8", p, lin8_grid(p))
    }
    unsafe fn linear(&self, p: &LinearParams, _large: bool) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("linear", p, linear_grid(p))
    }
    unsafe fn gate(&self, p: &GateParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("gate", p, gate_grid(p))
    }
    unsafe fn kv_write(&self, p: &KvWriteParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("kv_write", p, kv_write_grid(p))
    }
    unsafe fn attention(&self, p: &AttnParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("attn", p, attention_grid(p))
    }
    unsafe fn dn_prep(&self, p: &DnPrepParams) -> Result<()> {
        if p.positions == 0 {
            return Ok(());
        }
        self.run("dn_prep", p, dn_prep_grid(p))
    }
    unsafe fn dn_rec(&self, p: &DnRecParams) -> Result<()> {
        if p.seq_count == 0 {
            return Ok(());
        }
        self.run("dn_rec", p, dn_rec_grid(p))
    }
    unsafe fn dn_norm(&self, p: &DnNormParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("dn_norm", p, dn_norm_grid(p))
    }
    unsafe fn widen(&self, p: &WidenParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("widen", p, widen_grid(p))
    }
    unsafe fn round(&self, p: &RoundParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("round", p, round_grid(p))
    }
}

/// Host allocations are aligned to this, as device ones are.
const ALIGN: usize = 256;

struct HostBuf(Vec<u8>);

fn host_alloc(bytes: usize) -> (u64, Box<dyn Any>) {
    let buf = HostBuf(vec![0u8; bytes + ALIGN]);
    let base = buf.0.as_ptr() as usize;
    let addr = base.next_multiple_of(ALIGN);
    (addr as u64, Box::new(buf))
}

impl QwenDevice for HostDevice {
    fn profile(&self) -> QwenProfile {
        QwenProfile::Exact
    }

    fn alloc(&self, bytes: usize) -> Result<(u64, Box<dyn Any>)> {
        Ok(host_alloc(bytes))
    }

    unsafe fn zero(&self, addr: u64, bytes: usize) -> Result<()> {
        // SAFETY: the caller's range, for this and every method below.
        unsafe { wr::<u8>(addr, bytes).fill(0) };
        Ok(())
    }

    unsafe fn upload(&self, addr: u64, src: &[u8]) -> Result<()> {
        unsafe { wr::<u8>(addr, src.len()).copy_from_slice(src) };
        Ok(())
    }

    unsafe fn download(&self, addr: u64, dst: &mut [u8]) -> Result<()> {
        dst.copy_from_slice(unsafe { wr::<u8>(addr, dst.len()) });
        Ok(())
    }

    unsafe fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        // SAFETY: non-overlapping ranges of live allocations (the caller's).
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes) };
        Ok(())
    }

    unsafe fn cipher(
        &self,
        addr: u64,
        words: u32,
        counter: u32,
        key: &KvKey,
        nonce: &KvNonce,
    ) -> Result<()> {
        KvKeystream::with_counter(key, nonce, counter)
            .apply_words(unsafe { wr::<u32>(addr, words as usize) });
        Ok(())
    }

    fn alloc_host(&self, bytes: usize) -> Result<(*mut u8, Box<dyn Any>)> {
        let (addr, keep) = host_alloc(bytes);
        Ok((addr as *mut u8, keep))
    }

    unsafe fn copy_to_host(&self, dst: *mut u8, src: u64, bytes: usize) -> Result<()> {
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst, bytes) };
        Ok(())
    }

    unsafe fn copy_from_host(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()> {
        unsafe { std::ptr::copy_nonoverlapping(src, dst as *mut u8, bytes) };
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
