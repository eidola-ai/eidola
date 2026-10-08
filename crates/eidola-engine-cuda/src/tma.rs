//! TMA descriptors (`CUtensorMap`), encoded on the host by the driver's
//! `cuTensorMapEncodeTiled`. The GEMM launch paths describe each operand as a
//! [`TmaSpec`] exactly as the upstream host code would, and embed the 128-byte
//! result in the kernel's parameters.

use cudarc::driver::sys;

use crate::{CudaError, Result};

/// `CUtensorMapDataType` values used here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TmaType {
    U8,
    I32,
    F32,
    Bf16,
    /// Packed e2m1 pairs in memory, unpacked to one byte each in shared
    /// memory (`CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`); extents count 4-bit
    /// elements.
    Fp4Unpacked,
}

/// `CUtensorMapL2promotion`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TmaL2 {
    B128,
    B256,
}

/// `CUtensorMapSwizzle`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TmaSwizzle {
    None,
    B32,
    B64,
    B128,
}

/// One `cuTensorMapEncodeTiled` call (no interleave, element strides 1,
/// zero out-of-bounds fill: what CUTLASS and DeepGEMM use throughout).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TmaSpec {
    pub dtype: TmaType,
    pub addr: u64,
    /// Extents, innermost first.
    pub dims: Vec<u64>,
    /// Byte strides of dimensions `1..rank`.
    pub strides: Vec<u64>,
    pub box_dims: Vec<u32>,
    pub swizzle: TmaSwizzle,
    pub l2: TmaL2,
}

impl TmaSpec {
    fn bits_per_element(&self) -> u64 {
        match self.dtype {
            TmaType::Fp4Unpacked => 4,
            TmaType::U8 => 8,
            TmaType::Bf16 => 16,
            TmaType::I32 | TmaType::F32 => 32,
        }
    }

    /// Encode with the driver (needs a current context).
    pub fn encode(&self) -> Result<[u8; 128]> {
        let rank = self.dims.len();
        if rank == 0 || rank > 5 || self.strides.len() + 1 != rank || self.box_dims.len() != rank {
            return Err(CudaError::new(format!("bad TMA spec {self:?}")));
        }
        let dtype = match self.dtype {
            TmaType::U8 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            TmaType::I32 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
            TmaType::F32 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            TmaType::Bf16 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            TmaType::Fp4Unpacked => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        };
        let l2 = match self.l2 {
            TmaL2::B128 => sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            TmaL2::B256 => sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        };
        let swizzle = match self.swizzle {
            TmaSwizzle::None => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            TmaSwizzle::B32 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
            TmaSwizzle::B64 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
            TmaSwizzle::B128 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        };
        let elem_strides = vec![1u32; rank];
        let mut map = sys::CUtensorMap { opaque: [0; 16] };
        // SAFETY: every array holds `rank` (strides: `rank - 1`) entries and
        // outlives the call; the driver writes the 128-byte map.
        unsafe {
            sys::cuTensorMapEncodeTiled(
                &mut map,
                dtype,
                u32::try_from(rank).expect("a rank of at most 5"),
                self.addr as *mut std::ffi::c_void,
                self.dims.as_ptr(),
                self.strides.as_ptr(),
                self.box_dims.as_ptr(),
                elem_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle,
                l2,
                sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        }
        .result()
        .map_err(|e| CudaError::new(format!("cuTensorMapEncodeTiled({self:?}): {e:?}")))?;
        let mut out = [0u8; 128];
        for (i, w) in map.opaque.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        Ok(out)
    }

    /// CuTe's post-processing of an encoded descriptor (`make_tma_copy_desc`):
    /// unless the first 128 KiB of the tensor are one dense run from its base
    /// address, bit 21 of the descriptor's second word is cleared.
    pub fn cute_fixup(&self, desc: &mut [u8; 128]) {
        if !self.prefix_is_contiguous() {
            let mut w = u64::from_le_bytes(desc[8..16].try_into().expect("8 bytes"));
            w &= !(1u64 << 21);
            desc[8..16].copy_from_slice(&w.to_le_bytes());
        }
    }

    /// `cute::detail::tma_gmem_prefix_is_contiguous` with its 128 KiB
    /// threshold.
    pub fn prefix_is_contiguous(&self) -> bool {
        const THRESHOLD: u64 = 128 * 1024;
        let mut modes: Vec<(u64, u64)> = self
            .dims
            .iter()
            .skip(1)
            .zip(&self.strides)
            .filter(|&(&s, &d)| s > 1 && d != 0)
            .map(|(&s, &d)| (s, d))
            .collect();
        modes.sort_by_key(|&(_, d)| d);
        let mut covered = self.dims[0] * self.bits_per_element() / 8;
        for (s, d) in modes {
            if d != covered {
                break;
            }
            if covered >= THRESHOLD || covered.checked_mul(s).is_none() {
                return true;
            }
            covered *= s;
        }
        covered >= THRESHOLD
    }
}
