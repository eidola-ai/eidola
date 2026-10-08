//! Loading manifest-verified kernel images and launching their entries.
//!
//! Images reach the driver only through [`KernelDir`], which hands out bytes
//! whose size and SHA-256 match the compiled-in kernel manifest: no other
//! manifest can approve bytes, because none can be supplied. The images
//! are SASS for the exact target with no PTX, so the driver has nothing to
//! JIT: an image for the wrong architecture fails to load rather than being
//! recompiled.

use std::ffi::{CString, c_void};
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaStream};
use eidola_engine_kernels::{ArtifactDir, Cubin, KernelMeta, Manifest};

use crate::device::{Gpu, ImageArch};
use crate::{CudaError, Result, narrow};

/// A kernel build output, read only through the compiled-in manifest
/// ([`Manifest::embedded`]).
#[derive(Clone, Debug)]
pub struct KernelDir(ArtifactDir<'static>);

impl KernelDir {
    pub fn new(root: impl Into<PathBuf>) -> KernelDir {
        KernelDir(ArtifactDir::new(root, Manifest::embedded()))
    }

    /// Verify every image the manifest lists.
    pub fn verify_all(&self) -> Result<()> {
        self.0
            .verify_all()
            .map_err(|e| CudaError::new(e.to_string()))
    }
}

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

/// Refuse unless `cubin` is image `image` and (when given) holds entry
/// `symbol`.
pub fn check_entry(cubin: &Cubin, image: &str, symbol: Option<&str>) -> Result<()> {
    if cubin.name != image {
        return Err(CudaError::new(format!(
            "image {} where {image} is required",
            cubin.name
        )));
    }
    if let Some(symbol) = symbol
        && cubin.entry(symbol).is_none()
    {
        return Err(CudaError::new(format!("no entry {symbol} in {image}")));
    }
    Ok(())
}

/// One loaded kernel image.
pub struct KernelModule {
    loaded: Arc<Loaded>,
    cubin: &'static Cubin,
    source: ImageSource,
}

/// A loaded image, unloaded when the last owner goes: the module and every
/// [`Kernel`] resolved from it share it, so no function handle outlives its
/// image.
struct Loaded {
    ctx: Arc<CudaContext>,
    module: sys::CUmodule,
}

// SAFETY: a CUmodule handle is usable from any thread once its context is
// current; every use binds the context first.
unsafe impl Send for Loaded {}
unsafe impl Sync for Loaded {}

impl Drop for Loaded {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            // SAFETY: the module is ours, and with the last owner gone no
            // function resolved from it remains.
            unsafe {
                let _ = sys::cuModuleUnload(self.module);
            }
        }
    }
}

impl KernelModule {
    /// Load kernel `name` for this device's exact architecture from its
    /// cubin.
    pub fn load(gpu: &Gpu, dir: &KernelDir, name: &str) -> Result<KernelModule> {
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
        dir: &KernelDir,
        name: &str,
        source: ImageSource,
    ) -> Result<KernelModule> {
        let manifest = Manifest::embedded();
        let arch = source.arch().as_str();
        let cubin = manifest
            .cubin(name, arch)
            .ok_or_else(|| CudaError::new(format!("no cubin {name} {arch} in the manifest")))?;
        let bytes = match source {
            ImageSource::Cubin(_) => dir.0.cubin(name, arch),
            ImageSource::Fatbin(_) => dir.0.fatbin(name),
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
            loaded: Arc::new(Loaded {
                ctx: gpu.context().clone(),
                module,
            }),
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

    /// Refuse unless this module runs image `image`: a typed wrapper's
    /// argument layouts are its image's.
    pub fn expect_image(&self, image: &str) -> Result<()> {
        check_entry(self.cubin, image, None)
    }

    /// Resolve `symbol` of image `image`, refused for any other image or
    /// entry: what a typed launch wrapper takes its kernel from.
    pub fn bound_kernel(&self, image: &str, symbol: &str) -> Result<Kernel> {
        check_entry(self.cubin, image, Some(symbol))?;
        self.kernel(symbol)
    }

    /// Resolve an entry by symbol and read its launch contract out of the
    /// image. Opts the function into its dynamic shared memory when that is
    /// above the 48 KiB default.
    pub fn kernel(&self, symbol: &str) -> Result<Kernel> {
        let entry = self
            .cubin
            .entry(symbol)
            .ok_or_else(|| CudaError::new(format!("no entry {symbol} in {}", self.cubin.name)))?;
        self.loaded.ctx.bind_to_thread()?;
        let c_symbol = CString::new(symbol).expect("symbols have no NUL");
        let mut func = std::ptr::null_mut();
        // SAFETY: valid module and NUL-terminated name.
        unsafe { sys::cuModuleGetFunction(&mut func, self.loaded.module, c_symbol.as_ptr()) }
            .result()
            .map_err(|e| CudaError::new(format!("resolving {symbol}: {e:?}")))?;
        let meta = self.read_meta(&entry.meta)?;
        if meta.dynamic_smem_bytes > 48 * 1024 {
            // SAFETY: valid function handle.
            unsafe {
                sys::cuFuncSetAttribute(
                    func,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    narrow(meta.dynamic_smem_bytes, "dynamic shared memory")?,
                )
            }
            .result()?;
        }
        Ok(Kernel {
            loaded: self.loaded.clone(),
            func,
            meta,
            symbol: symbol.to_owned(),
        })
    }

    /// The launch contract a record names, copied from the device global.
    pub fn read_meta(&self, record: &str) -> Result<KernelMeta> {
        self.loaded.ctx.bind_to_thread()?;
        let c_record = CString::new(record).expect("symbols have no NUL");
        let mut ptr = 0;
        let mut size = 0usize;
        // SAFETY: valid module and NUL-terminated name.
        unsafe {
            sys::cuModuleGetGlobal_v2(&mut ptr, &mut size, self.loaded.module, c_record.as_ptr())
        }
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

/// A resolved kernel entry and its launch contract.
pub struct Kernel {
    /// Keeps the image `func` lives in loaded.
    loaded: Arc<Loaded>,
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

    /// Static shared memory the image declares (beside the launch contract's
    /// dynamic amount).
    pub fn static_smem_bytes(&self) -> Result<u32> {
        self.loaded.ctx.bind_to_thread()?;
        let mut v = 0;
        // SAFETY: valid function handle; writes one int.
        unsafe {
            sys::cuFuncGetAttribute(
                &mut v,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                self.func,
            )
        }
        .result()?;
        narrow(v, "static shared memory")
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
        // The current context is per thread: whichever thread launches binds
        // the module's own.
        self.loaded.ctx.bind_to_thread()?;
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
            numAttrs: u32::try_from(attrs.len()).expect("at most two launch attributes"),
        };
        // SAFETY: forwarded to the caller.
        unsafe {
            sys::cuLaunchKernelEx(&config, self.func, args.as_mut_ptr(), std::ptr::null_mut())
        }
        .result()
        .map_err(|e| CudaError::new(format!("launching {}: {e:?}", self.symbol)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubin(name: &str) -> &'static Cubin {
        Manifest::embedded()
            .cubin(name, "sm_103a")
            .unwrap_or_else(|| panic!("{name}"))
    }

    /// A typed wrapper's kernel is bound by image and entry: another image's
    /// module, or another entry of the right image, is refused.
    #[test]
    fn entries_are_bound_to_their_image() {
        let rms = cubin("rmsnorm");
        check_entry(rms, "rmsnorm", Some("eidola_rmsnorm_bf16")).unwrap();
        check_entry(rms, "rmsnorm", None).unwrap();
        assert!(check_entry(rms, "rmsnorm", Some("eidola_embed")).is_err());
        let ops = cubin("engine_ops");
        assert!(ops.entry("eidola_embed").is_some());
        assert!(check_entry(ops, "rmsnorm", Some("eidola_rmsnorm_bf16")).is_err());
        assert!(check_entry(ops, "rmsnorm", None).is_err());
        assert!(check_entry(rms, "engine_ops", Some("eidola_embed")).is_err());
        // Each GEMM kind binds its own image only: both share a Params size,
        // so the size check alone would not tell them apart.
        use crate::gemm::GemmKind;
        let kinds = [GemmKind::Fp8Blockwise, GemmKind::Bf16];
        for kind in kinds {
            let (image, entry) = kind.kernel();
            check_entry(cubin(image), image, Some(entry)).unwrap();
            for other in kinds.iter().filter(|&&k| k != kind) {
                assert!(check_entry(cubin(other.kernel().0), image, Some(entry)).is_err());
            }
        }
        // The images the module-taking wrappers require exist.
        for image in [
            crate::ops::RmsNorm::IMAGE,
            "engine_ops",
            "flashinfer_fa2_sink_paged",
            "sampling",
            "deepgemm_fp8_fp4_grouped",
        ] {
            check_entry(cubin(image), image, None).unwrap();
        }
        check_entry(
            cubin(crate::ops::RmsNorm::IMAGE),
            crate::ops::RmsNorm::IMAGE,
            Some(crate::ops::RmsNorm::SYMBOL),
        )
        .unwrap();
    }
}
