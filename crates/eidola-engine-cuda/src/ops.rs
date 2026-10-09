//! Typed launch wrappers for our own kernels.

use std::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::device::Gpu;
use crate::module::{Kernel, KernelModule};
use crate::{CudaError, Result, narrow};

/// The RMSNorm kernel (`eidola_rmsnorm_bf16` of the `rmsnorm` image), bound
/// at construction: no other entry can be launched through it.
pub struct RmsNorm {
    kernel: Kernel,
}

impl RmsNorm {
    pub const IMAGE: &str = "rmsnorm";
    pub const SYMBOL: &str = "eidola_rmsnorm_bf16";

    pub fn from_module(module: &KernelModule) -> Result<RmsNorm> {
        Ok(RmsNorm {
            kernel: module.bound_kernel(Self::IMAGE, Self::SYMBOL)?,
        })
    }

    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// `out[r] = weight * x[r] / sqrt(mean(x[r]^2) + eps)` for each BF16 row of
    /// `hidden` elements in `x` (`x.len() / hidden` rows), accumulated in f32
    /// (`eidola_rmsnorm_bf16`).
    pub fn launch(
        &self,
        gpu: &Gpu,
        out: &mut CudaSlice<u16>,
        x: &CudaSlice<u16>,
        weight: &CudaSlice<u16>,
        hidden: u32,
        eps: f32,
    ) -> Result<()> {
        if hidden == 0 || !x.len().is_multiple_of(hidden as usize) {
            return Err(CudaError::new("rmsnorm: input is not whole rows"));
        }
        let rows: u32 = narrow(x.len() / hidden as usize, "rmsnorm rows")?;
        let n = x.len();
        if out.len() < n || weight.len() < hidden as usize {
            return Err(CudaError::new("rmsnorm: buffer too small"));
        }
        if rows == 0 {
            return Ok(());
        }
        let stream = gpu.stream();
        let (out_ptr, _o) = out.device_ptr_mut(stream);
        let (x_ptr, _x) = x.device_ptr(stream);
        let (w_ptr, _w) = weight.device_ptr(stream);
        let (mut out_ptr, mut x_ptr, mut w_ptr, mut hidden, mut eps) =
            (out_ptr, x_ptr, w_ptr, hidden, eps);
        let mut args = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut hidden as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        // SAFETY: the argument list matches `eidola_rmsnorm_bf16`'s signature and
        // the buffers were checked to cover `rows * hidden` elements.
        unsafe { self.kernel.launch(stream, [rows, 1, 1], &mut args) }
    }
}

/// The SM count DeepGEMM's persistent scheduler is instantiated with
/// (`kNumSms` in the kernels crate's `deepgemm_fp8_fp4_grouped.cu`). It is the
/// scheduler's grid stride, so the grid must be exactly this.
pub const DEEPGEMM_INSTANCE_SMS: u32 = 148;

/// The grid for a DeepGEMM launch on this device: the instance's SM count,
/// refused on a part with fewer SMs (blocks would be scheduled in waves the
/// persistent scheduler does not expect). A part with more SMs leaves the
/// rest idle.
pub fn deepgemm_grid(gpu: &Gpu) -> Result<[u32; 3]> {
    let sms = gpu.info().sm_count;
    if sms < DEEPGEMM_INSTANCE_SMS {
        return Err(CudaError::new(format!(
            "DeepGEMM instances need {DEEPGEMM_INSTANCE_SMS} SMs; the device has {sms}"
        )));
    }
    Ok([DEEPGEMM_INSTANCE_SMS, 1, 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_engine_kernels::Manifest;

    /// The constant matches the template argument baked into every DeepGEMM
    /// instance (`..., kNumMulticast, kIsMulticastOnA, kNumSMs, ...`).
    #[test]
    fn deepgemm_sm_count_matches_the_instances() {
        let needle = format!("ELb1ELj{DEEPGEMM_INSTANCE_SMS}ELb1ELb0E");
        let mut seen = 0;
        for cubin in &Manifest::embedded().kernels {
            for entry in cubin
                .entries
                .iter()
                .filter(|e| e.symbol.contains("deep_gemm"))
            {
                assert!(entry.symbol.contains(&needle), "{}", entry.symbol);
                seen += 1;
            }
        }
        assert_eq!(seen, 6, "two instances, three images");
    }
}
