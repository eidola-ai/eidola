//! Loading manifest-verified kernel images and launching their entries.
//!
//! Images reach the driver only through [`ArtifactDir`], which hands out bytes
//! whose size and SHA-256 match the compiled-in kernel manifest. The images
//! are SASS for the exact target with no PTX, so the driver has nothing to
//! JIT: an image for the wrong architecture fails to load rather than being
//! recompiled.

use std::ffi::{CString, c_void};
use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaStream};
use eidola_engine_kernels::{ArtifactDir, Cubin, KernelMeta, Manifest};

use crate::device::{Gpu, ImageArch};
use crate::{CudaError, Result};

/// Where a module's image came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageSource {
    /// One architecture's cubin.
    Cubin(ImageArch),
    /// The kernel's fatbin; the driver picks the exact-match cubin, which
    /// must be the one named here.
    Fatbin(ImageArch),
}

impl ImageSource {
    pub fn arch(self) -> ImageArch {
        match self {
            ImageSource::Cubin(a) | ImageSource::Fatbin(a) => a,
        }
    }
}

/// One loaded kernel image.
pub struct KernelModule {
    ctx: Arc<CudaContext>,
    module: sys::CUmodule,
    cubin: &'static Cubin,
    source: ImageSource,
}

// SAFETY: a CUmodule handle is usable from any thread once its context is
// current; every use below binds the context first.
unsafe impl Send for KernelModule {}
unsafe impl Sync for KernelModule {}

impl KernelModule {
    /// Load kernel `name` for this device's exact architecture from its
    /// cubin.
    pub fn load(gpu: &Gpu, dir: &ArtifactDir<'static>, name: &str) -> Result<KernelModule> {
        let arch = gpu.image_arch().ok_or_else(|| {
            CudaError::new(format!(
                "no kernel image for compute capability {:?}",
                gpu.info().compute_capability
            ))
        })?;
        KernelModule::load_from(gpu, dir, name, ImageSource::Cubin(arch))
    }

    /// Load kernel `name` from an explicit image.
    pub fn load_from(
        gpu: &Gpu,
        dir: &ArtifactDir<'static>,
        name: &str,
        source: ImageSource,
    ) -> Result<KernelModule> {
        let manifest = Manifest::embedded();
        let arch = source.arch().as_str();
        let cubin = manifest
            .cubin(name, arch)
            .ok_or_else(|| CudaError::new(format!("no cubin {name} {arch} in the manifest")))?;
        let bytes = match source {
            ImageSource::Cubin(_) => dir.cubin(name, arch),
            ImageSource::Fatbin(_) => dir.fatbin(name),
        }
        .map_err(|e| CudaError::new(e.to_string()))?;
        gpu.context().bind_to_thread()?;
        let mut module = std::ptr::null_mut();
        // SAFETY: `bytes` is a complete, manifest-verified ELF or fatbin image
        // that outlives the call; the driver copies what it keeps.
        unsafe { sys::cuModuleLoadData(&mut module, bytes.as_ptr() as *const c_void) }
            .result()
            .map_err(|e| CudaError::new(format!("loading {name} ({source:?}): {e:?}")))?;
        Ok(KernelModule {
            ctx: gpu.context().clone(),
            module,
            cubin,
            source,
        })
    }

    pub fn source(&self) -> ImageSource {
        self.source
    }

    /// The manifest record of the cubin this module runs.
    pub fn cubin(&self) -> &'static Cubin {
        self.cubin
    }

    /// Resolve an entry by symbol and read its launch contract out of the
    /// image. Opts the function into its dynamic shared memory when that is
    /// above the 48 KiB default.
    pub fn kernel(&self, symbol: &str) -> Result<Kernel> {
        let entry = self
            .cubin
            .entry(symbol)
            .ok_or_else(|| CudaError::new(format!("no entry {symbol} in {}", self.cubin.name)))?;
        self.ctx.bind_to_thread()?;
        let c_symbol = CString::new(symbol).expect("symbols have no NUL");
        let mut func = std::ptr::null_mut();
        // SAFETY: valid module and NUL-terminated name.
        unsafe { sys::cuModuleGetFunction(&mut func, self.module, c_symbol.as_ptr()) }
            .result()
            .map_err(|e| CudaError::new(format!("resolving {symbol}: {e:?}")))?;
        let meta = self.read_meta(&entry.meta)?;
        if meta.dynamic_smem_bytes > 48 * 1024 {
            // SAFETY: valid function handle.
            unsafe {
                sys::cuFuncSetAttribute(
                    func,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    meta.dynamic_smem_bytes as i32,
                )
            }
            .result()?;
        }
        Ok(Kernel {
            func,
            meta,
            symbol: symbol.to_owned(),
        })
    }

    /// The launch contract a record names, copied from the device global.
    pub fn read_meta(&self, record: &str) -> Result<KernelMeta> {
        self.ctx.bind_to_thread()?;
        let c_record = CString::new(record).expect("symbols have no NUL");
        let mut ptr = 0;
        let mut size = 0usize;
        // SAFETY: valid module and NUL-terminated name.
        unsafe { sys::cuModuleGetGlobal_v2(&mut ptr, &mut size, self.module, c_record.as_ptr()) }
            .result()
            .map_err(|e| CudaError::new(format!("resolving {record}: {e:?}")))?;
        if size != KernelMeta::SIZE {
            return Err(CudaError::new(format!(
                "{record} is {size} bytes, not {}",
                KernelMeta::SIZE
            )));
        }
        let mut bytes = [0u8; KernelMeta::SIZE];
        // SAFETY: the global is exactly `bytes.len()` bytes.
        unsafe { sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut c_void, ptr, bytes.len()) }
            .result()?;
        Ok(KernelMeta::from_bytes(&bytes))
    }
}

impl Drop for KernelModule {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            // SAFETY: the module is ours and no longer used.
            unsafe {
                let _ = sys::cuModuleUnload(self.module);
            }
        }
    }
}

/// A resolved kernel entry and its launch contract.
pub struct Kernel {
    func: sys::CUfunction,
    meta: KernelMeta,
    symbol: String,
}

// SAFETY: a CUfunction is a context-scoped handle with no thread affinity.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    pub fn meta(&self) -> &KernelMeta {
        &self.meta
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn raw(&self) -> sys::CUfunction {
        self.func
    }

    /// Launch with the block shape, dynamic shared memory and cluster shape
    /// of the launch contract.
    ///
    /// # Safety
    ///
    /// `args` must point to one value per kernel parameter, each of exactly
    /// the parameter's type, and every device pointer among them must be
    /// valid for what the kernel does with it until the launch completes.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        grid: [u32; 3],
        args: &mut [*mut c_void],
    ) -> Result<()> {
        let m = &self.meta;
        let mut attrs = Vec::with_capacity(1);
        if m.cluster != [1, 1, 1] {
            let mut attr: sys::CUlaunchAttribute = unsafe { std::mem::zeroed() };
            attr.id = sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            attr.value.clusterDim.x = m.cluster[0];
            attr.value.clusterDim.y = m.cluster[1];
            attr.value.clusterDim.z = m.cluster[2];
            attrs.push(attr);
        }
        let config = sys::CUlaunchConfig {
            gridDimX: grid[0],
            gridDimY: grid[1],
            gridDimZ: grid[2],
            blockDimX: m.block[0],
            blockDimY: m.block[1],
            blockDimZ: m.block[2],
            sharedMemBytes: m.dynamic_smem_bytes,
            hStream: stream.cu_stream(),
            attrs: attrs.as_mut_ptr(),
            numAttrs: attrs.len() as u32,
        };
        // SAFETY: forwarded to the caller.
        unsafe {
            sys::cuLaunchKernelEx(&config, self.func, args.as_mut_ptr(), std::ptr::null_mut())
        }
        .result()
        .map_err(|e| CudaError::new(format!("launching {}: {e:?}", self.symbol)))
    }
}
