//! [`CudaDevice`]: one GPU, the Qwen3.8 kernel images of the committed manifest, in the
//! exact or the fast profile.
//!
//! Every launch runs its family's grid function first ([`super::kernels`]) and refuses
//! what its kernel cannot take; a launch over no rows launches nothing. In the fast
//! profile the FP8 linears run CUTLASS's blockwise tcgen05 GEMM
//! (`cutlass_fp8_blockwise_gemm`, the same image MiMo's executor runs: f32 scales per
//! 1×128 of the activations and 128×128 of the weights, exactly the kernels' layouts),
//! and the large BF16 linears (the head, the drafter's input projection) CUTLASS's BF16
//! GEMM into f32, rounded to BF16 by `eidola_qwen_round`. Attention stays the exact
//! kernel in both profiles until the FA2 head-dim-256 path is wired (its images are built:
//! `flashinfer_fa2_sink_paged_d256`).

use std::any::Any;
use std::ffi::c_void;
use std::sync::Mutex;

use cudarc::driver::sys;
use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::secret::KvKey;

use super::abi::*;
use super::device::{QwenDevice, QwenProfile};
use super::kernels::*;
use crate::kv_cipher::{CipherItem, CipherItems, KvCipher};
use crate::launch::dptr;
use crate::mimo::gemm::{Gemm, GemmArgs, GemmKind};
use crate::module::{Kernel, KernelDir, KernelModule};
use crate::{CudaError, Gpu, Result};

/// The Qwen3.8 kernel images.
pub const IMAGES: [&str; 3] = ["qwen_ops", "qwen_ops_div", "qwen_deltanet"];

struct Kernels {
    embed: Kernel,
    norm: Kernel,
    quant: Kernel,
    lin8: Kernel,
    linear: Kernel,
    gate: Kernel,
    kv_write: Kernel,
    attn: Kernel,
    dn_prep: Kernel,
    dn_rec: Kernel,
    dn_norm: Kernel,
    widen: Kernel,
    round: Kernel,
}

/// The fast profile's GEMMs.
struct Fast {
    fp8: Gemm,
    bf16: Gemm,
    _modules: Vec<KernelModule>,
}

/// One GPU running the Qwen3.8 executor.
pub struct CudaDevice {
    gpu: Gpu,
    profile: QwenProfile,
    k: Kernels,
    fast: Option<Fast>,
    cipher: Mutex<KvCipher>,
    _modules: Vec<KernelModule>,
}

fn bound<P>(module: &KernelModule, image: &str, symbol: &str) -> Result<Kernel> {
    let k = module.bound_kernel(image, symbol)?;
    if k.meta().params_bytes as usize != std::mem::size_of::<P>() {
        return Err(CudaError::new(format!(
            "{symbol}: the image's parameter struct is {} bytes, the host mirror {}",
            k.meta().params_bytes,
            std::mem::size_of::<P>()
        )));
    }
    Ok(k)
}

impl CudaDevice {
    /// Opens GPU `ordinal` and loads and verifies every image the profile runs.
    pub fn open(ordinal: usize, dir: &KernelDir, profile: QwenProfile) -> Result<CudaDevice> {
        let gpu = Gpu::open(ordinal)?;
        let ops = KernelModule::load(&gpu, dir, "qwen_ops")?;
        let div = KernelModule::load(&gpu, dir, "qwen_ops_div")?;
        let dn = KernelModule::load(&gpu, dir, "qwen_deltanet")?;
        let k = Kernels {
            embed: bound::<EmbedParams>(&ops, "qwen_ops", "eidola_qwen_embed")?,
            lin8: bound::<Lin8Params>(&ops, "qwen_ops", "eidola_qwen_lin8")?,
            linear: bound::<LinearParams>(&ops, "qwen_ops", "eidola_qwen_linear")?,
            kv_write: bound::<KvWriteParams>(&ops, "qwen_ops", "eidola_qwen_kv_write")?,
            widen: bound::<WidenParams>(&ops, "qwen_ops", "eidola_qwen_widen")?,
            round: bound::<RoundParams>(&ops, "qwen_ops", "eidola_qwen_round")?,
            norm: bound::<NormParams>(&div, "qwen_ops_div", "eidola_qwen_norm")?,
            quant: bound::<QuantParams>(&div, "qwen_ops_div", "eidola_qwen_quant")?,
            gate: bound::<GateParams>(&div, "qwen_ops_div", "eidola_qwen_gate")?,
            attn: bound::<AttnParams>(&div, "qwen_ops_div", "eidola_qwen_attn")?,
            dn_prep: bound::<DnPrepParams>(&div, "qwen_ops_div", "eidola_qwen_dn_prep")?,
            dn_norm: bound::<DnNormParams>(&div, "qwen_ops_div", "eidola_qwen_dn_norm")?,
            dn_rec: bound::<DnRecParams>(&dn, "qwen_deltanet", "eidola_qwen_dn_rec")?,
        };
        let cipher = KvCipher::load(&gpu, dir)?;
        let fast = match profile {
            QwenProfile::Exact => None,
            QwenProfile::Fast => {
                let f = KernelModule::load(&gpu, dir, GemmKind::Fp8Blockwise.kernel().0)?;
                let b = KernelModule::load(&gpu, dir, GemmKind::Bf16.kernel().0)?;
                Some(Fast {
                    fp8: Gemm::from_module(GemmKind::Fp8Blockwise, &f)?,
                    bf16: Gemm::from_module(GemmKind::Bf16, &b)?,
                    _modules: vec![f, b],
                })
            }
        };
        Ok(CudaDevice {
            gpu,
            profile,
            k,
            fast,
            cipher: Mutex::new(cipher),
            _modules: vec![ops, div, dn],
        })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn bind(&self) -> Result<()> {
        self.gpu.context().bind_to_thread()?;
        Ok(())
    }

    /// Launches `kernel` with `p` by value over `grid`.
    ///
    /// # Safety
    ///
    /// `p` is the kernel's parameter struct and every address in it is valid for what the
    /// kernel does with it.
    unsafe fn go<P: Copy>(&self, kernel: &Kernel, p: &P, grid: [u32; 3]) -> Result<()> {
        let mut v = *p;
        let mut args = [(&mut v as *mut P).cast::<c_void>()];
        // SAFETY: the caller's.
        unsafe { kernel.launch(self.gpu.stream(), grid, &mut args) }
    }
}

impl QwenKernels for CudaDevice {
    unsafe fn embed(&self, p: &EmbedParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = embed_grid(p)?;
        unsafe { self.go(&self.k.embed, p, g) }
    }
    unsafe fn norm(&self, p: &NormParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = norm_grid(p)?;
        unsafe { self.go(&self.k.norm, p, g) }
    }
    unsafe fn quant(&self, p: &QuantParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = quant_grid(p)?;
        unsafe { self.go(&self.k.quant, p, g) }
    }
    unsafe fn lin8(&self, p: &Lin8Params) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = lin8_grid(p)?;
        match &self.fast {
            None => unsafe { self.go(&self.k.lin8, p, g) },
            Some(f) => {
                if p.out_stride != p.n {
                    return Err(CudaError::new("the fast FP8 linear writes dense rows"));
                }
                let a = GemmArgs {
                    m: p.m_pad,
                    n: p.n,
                    k: p.k,
                    a: p.a,
                    b: p.w,
                    d: p.out,
                    sfa: p.a_scales,
                    sfb: p.w_scales,
                    alpha: 1.0,
                };
                GemmKind::Fp8Blockwise.check(&a)?;
                // SAFETY: `m_pad` rows of codes and scales, the weight and `m_pad · n`
                // outputs (the padding rows' outputs are written and never read).
                unsafe { f.fp8.launch(&self.gpu, &a) }
            }
        }
    }
    unsafe fn linear(&self, p: &LinearParams, large: bool) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = linear_grid(p)?;
        match (&self.fast, large) {
            (Some(f), true) => {
                if p.x_stride != p.k {
                    return Err(CudaError::new("the fast BF16 linear reads dense rows"));
                }
                let s = self.gpu.stream();
                let tmp = s.alloc_zeros::<f32>(p.rows as usize * p.n as usize)?;
                let a = GemmArgs {
                    m: p.rows,
                    n: p.n,
                    k: p.k,
                    a: p.x,
                    b: p.w,
                    d: dptr(&tmp, s),
                    sfa: 0,
                    sfb: 0,
                    alpha: 1.0,
                };
                GemmKind::Bf16.check(&a)?;
                let r = RoundParams {
                    x: a.d,
                    out: p.out,
                    rows: p.rows,
                    width: p.n,
                    x_stride: p.n,
                    out_stride: p.out_stride,
                    out_f32: p.out_f32,
                    reserved: 0,
                };
                let rg = round_grid(&r)?;
                // SAFETY: `rows` input rows, the weight, the f32 scratch above.
                unsafe {
                    f.bf16.launch(&self.gpu, &a)?;
                    self.go(&self.k.round, &r, rg)?;
                }
                // The scratch is freed in stream order.
                drop(tmp);
                Ok(())
            }
            _ => unsafe { self.go(&self.k.linear, p, g) },
        }
    }
    unsafe fn gate(&self, p: &GateParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = gate_grid(p)?;
        unsafe { self.go(&self.k.gate, p, g) }
    }
    unsafe fn kv_write(&self, p: &KvWriteParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = kv_write_grid(p)?;
        unsafe { self.go(&self.k.kv_write, p, g) }
    }
    unsafe fn attention(&self, p: &AttnParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = attention_grid(p)?;
        unsafe { self.go(&self.k.attn, p, g) }
    }
    unsafe fn dn_prep(&self, p: &DnPrepParams) -> Result<()> {
        if p.positions == 0 {
            return Ok(());
        }
        let g = dn_prep_grid(p)?;
        unsafe { self.go(&self.k.dn_prep, p, g) }
    }
    unsafe fn dn_rec(&self, p: &DnRecParams) -> Result<()> {
        if p.seq_count == 0 {
            return Ok(());
        }
        let g = dn_rec_grid(p)?;
        unsafe { self.go(&self.k.dn_rec, p, g) }
    }
    unsafe fn dn_norm(&self, p: &DnNormParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = dn_norm_grid(p)?;
        unsafe { self.go(&self.k.dn_norm, p, g) }
    }
    unsafe fn widen(&self, p: &WidenParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = widen_grid(p)?;
        unsafe { self.go(&self.k.widen, p, g) }
    }
    unsafe fn round(&self, p: &RoundParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let g = round_grid(p)?;
        unsafe { self.go(&self.k.round, p, g) }
    }
}

/// Pieces a weight upload goes in (a tensor may be gigabytes of mapped file).
const UPLOAD_PIECE: usize = 256 << 20;

impl QwenDevice for CudaDevice {
    fn profile(&self) -> QwenProfile {
        self.profile
    }

    fn alloc(&self, bytes: usize) -> Result<(u64, Box<dyn Any>)> {
        let s = self.gpu.stream();
        let buf = s.alloc_zeros::<u64>(bytes.div_ceil(8))?;
        Ok((dptr(&buf, s), Box::new(buf)))
    }

    unsafe fn zero(&self, addr: u64, bytes: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller's range.
        unsafe { sys::cuMemsetD8Async(addr, 0, bytes, self.gpu.stream().cu_stream()) }.result()?;
        Ok(())
    }

    unsafe fn upload(&self, addr: u64, src: &[u8]) -> Result<()> {
        self.bind()?;
        let mut at = addr;
        for piece in src.chunks(UPLOAD_PIECE) {
            // SAFETY: the caller's range; a pageable copy has left `piece` on return.
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    at,
                    piece.as_ptr().cast(),
                    piece.len(),
                    self.gpu.stream().cu_stream(),
                )
            }
            .result()?;
            at += piece.len() as u64;
        }
        Ok(())
    }

    unsafe fn download(&self, addr: u64, dst: &mut [u8]) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller's range; the stream is drained before `dst` is read.
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr().cast(),
                addr,
                dst.len(),
                self.gpu.stream().cu_stream(),
            )
        }
        .result()?;
        self.gpu.synchronize()
    }

    unsafe fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller's ranges.
        unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, self.gpu.stream().cu_stream()) }
            .result()?;
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
        let mut items = CipherItems::with_capacity(1);
        items.push(CipherItem::new(addr, words, counter, key, nonce));
        let mut c = self
            .cipher
            .lock()
            .map_err(|_| CudaError::new("cipher lock"))?;
        // SAFETY: one range, the caller's.
        unsafe { c.apply(&self.gpu, &mut items) }
    }

    fn alloc_host(&self, bytes: usize) -> Result<(*mut u8, Box<dyn Any>)> {
        // SAFETY: zeroed below before anything reads it.
        let mut mem = unsafe {
            self.gpu
                .context()
                .alloc_pinned_with_flags::<u8>(bytes.max(1), 0)
        }?;
        let base = mem.as_mut_ptr()?;
        // SAFETY: the allocation just made.
        unsafe { std::ptr::write_bytes(base, 0, bytes.max(1)) };
        Ok((base, Box::new(mem)))
    }

    unsafe fn copy_to_host(&self, dst: *mut u8, src: u64, bytes: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller's ranges.
        unsafe { sys::cuMemcpyDtoHAsync_v2(dst.cast(), src, bytes, self.gpu.stream().cu_stream()) }
            .result()?;
        Ok(())
    }

    unsafe fn copy_from_host(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller's ranges.
        unsafe { sys::cuMemcpyHtoDAsync_v2(dst, src.cast(), bytes, self.gpu.stream().cu_stream()) }
            .result()?;
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.gpu.synchronize()
    }
}
