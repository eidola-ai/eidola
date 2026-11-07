//! Raw device pointers and kernel argument lists.

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr};

/// The device address of a buffer, for a kernel argument. Event tracking is off
/// (see [`crate::Gpu::open`]), so taking it records nothing.
pub fn dptr<T: DeviceRepr>(slice: &CudaSlice<T>, stream: &CudaStream) -> u64 {
    slice.device_ptr(stream).0
}

/// The device address of element `offset` of a buffer.
pub fn dptr_at<T: DeviceRepr>(slice: &CudaSlice<T>, stream: &CudaStream, offset: usize) -> u64 {
    assert!(
        offset <= slice.len(),
        "offset {offset} past {}",
        slice.len()
    );
    dptr(slice, stream) + (offset * std::mem::size_of::<T>()) as u64
}

/// Launch `kernel` on `gpu`'s stream over `grid` with the listed arguments, each
/// passed by value exactly as written (give integer literals their parameter's
/// type: `3u32`, `0u64`).
///
/// # Safety
///
/// The same as [`crate::Kernel::launch`]: the argument types must be the
/// kernel's parameter types, in order, and every device address must be valid
/// for what the kernel does with it.
#[macro_export]
macro_rules! launch {
    ($gpu:expr, $kernel:expr, $grid:expr $(, $arg:expr)* $(,)?) => {{
        #[allow(unused_mut)]
        let mut args: Vec<*mut ::std::ffi::c_void> = Vec::new();
        $(
            let mut a = $arg;
            args.push(&mut a as *mut _ as *mut ::std::ffi::c_void);
        )*
        $kernel.launch($gpu.stream(), $grid, &mut args)
    }};
}
