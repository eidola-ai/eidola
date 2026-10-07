//! Opening a device and choosing which kernel image it runs.

use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaStream};

use crate::{CudaError, Result};

/// The kernel image a device runs: the exact-match architecture-specific
/// cubin where one is built, else the family cubin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageArch {
    /// B200 (compute capability 10.0).
    Sm100a,
    /// B300 (compute capability 10.3).
    Sm103a,
    /// Any compute capability 10.x part.
    Sm100f,
}

impl ImageArch {
    /// The architecture name the kernel manifest uses.
    pub fn as_str(self) -> &'static str {
        match self {
            ImageArch::Sm100a => "sm_100a",
            ImageArch::Sm103a => "sm_103a",
            ImageArch::Sm100f => "sm_100f",
        }
    }

    /// The image a part of this compute capability runs, or `None` when no
    /// image in the build can run on it (anything outside CC 10.x).
    pub fn for_compute_capability(major: i32, minor: i32) -> Option<ImageArch> {
        match (major, minor) {
            (10, 0) => Some(ImageArch::Sm100a),
            (10, 3) => Some(ImageArch::Sm103a),
            (10, _) => Some(ImageArch::Sm100f),
            _ => None,
        }
    }
}

/// What the driver reports about the device and itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    /// Compute capability `(major, minor)`.
    pub compute_capability: (i32, i32),
    /// Streaming multiprocessors. Persistent kernels size their grid by it.
    pub sm_count: u32,
    /// `cuDriverGetVersion`: `1000 * major + 10 * minor`.
    pub driver_version: i32,
    pub total_mem_bytes: usize,
    /// Opt-in dynamic shared memory per block.
    pub max_smem_per_block_optin: u32,
}

/// An open device: its primary context and the stream every launch uses.
pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    info: DeviceInfo,
}

impl Gpu {
    /// Whether the CUDA driver library can be loaded and reports at least
    /// one device. False (never a panic) on machines without a driver.
    pub fn available() -> bool {
        // SAFETY: probes for the shared library without calling into it.
        if !unsafe { sys::is_culib_present() } {
            return false;
        }
        matches!(CudaContext::device_count(), Ok(n) if n > 0)
    }

    /// Open device `ordinal` (its primary context) with a fresh stream.
    pub fn open(ordinal: usize) -> Result<Gpu> {
        if !Gpu::available() {
            return Err(CudaError::new("no CUDA driver or device"));
        }
        let ctx = CudaContext::new(ordinal)?;
        let attr = |a| ctx.attribute(a).map_err(CudaError::from);
        let mut driver_version = 0;
        // SAFETY: writes one int.
        unsafe { sys::cuDriverGetVersion(&mut driver_version) }.result()?;
        let info = DeviceInfo {
            ordinal,
            name: ctx.name()?,
            compute_capability: ctx.compute_capability()?,
            sm_count: attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
                as u32,
            driver_version,
            total_mem_bytes: ctx.total_mem()?,
            max_smem_per_block_optin: attr(
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )? as u32,
        };
        let stream = ctx.new_stream()?;
        Ok(Gpu { ctx, stream, info })
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The exact-match image for this device, if the build has one.
    pub fn image_arch(&self) -> Option<ImageArch> {
        let (major, minor) = self.info.compute_capability;
        ImageArch::for_compute_capability(major, minor)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_choice_by_compute_capability() {
        assert_eq!(
            ImageArch::for_compute_capability(10, 0),
            Some(ImageArch::Sm100a)
        );
        assert_eq!(
            ImageArch::for_compute_capability(10, 3),
            Some(ImageArch::Sm103a)
        );
        assert_eq!(
            ImageArch::for_compute_capability(10, 1),
            Some(ImageArch::Sm100f)
        );
        assert_eq!(ImageArch::for_compute_capability(12, 0), None);
        assert_eq!(ImageArch::for_compute_capability(9, 0), None);
    }
}
