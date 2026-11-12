//! The node's own check that its GPUs are in confidential-computing mode, run before the
//! CUDA executor opens a device when the measured configuration says
//! `EIDOLA_ENGINE_GPU_CC=required`.
//!
//! The platform's GPU image already refuses to open the GPUs to work unless NVIDIA's local
//! verifier has passed for every one of them at boot; this makes the same claim the node's
//! too, and adds what that appraisal does not state: confidential computing on, DevTools
//! mode off, and the production environment. The values come from NVML, the driver's
//! management library, loaded at run time (`libnvidia-ml.so.1`, the driver's own soname;
//! nothing is linked at build time), and are judged by [`check`], a pure function, so the
//! decision is tested on any machine.
//!
//! The NVML names, values and layouts are those of `nvml.h` from the driver release the
//! platform's GPU image ships, 595.71.05 (CUDA 13.2.2's `cuda_nvml_dev` 13.2.86):
//! `nvmlSystemGetConfComputeState` filling `nvmlConfComputeSystemState_t`
//! (`environment`, `ccFeature`, `devToolsMode`, three `unsigned int`s, in that order), and
//! `nvmlSystemGetConfComputeGpusReadyState` filling one `unsigned int`. Both are system-wide
//! queries: the CC mode and the ready state are set for every GPU of the system together.

/// `NVML_CC_SYSTEM_FEATURE_ENABLED`: confidential computing is on.
pub const CC_FEATURE_ENABLED: u32 = 1;
/// `NVML_CC_SYSTEM_DEVTOOLS_MODE_OFF`: DevTools mode (CC with debugging and profiling
/// allowed) is off.
pub const DEVTOOLS_MODE_OFF: u32 = 0;
/// `NVML_CC_SYSTEM_ENVIRONMENT_PROD`: the production environment (not the simulation one).
pub const ENVIRONMENT_PROD: u32 = 2;
/// `NVML_CC_ACCEPTING_CLIENT_REQUESTS_TRUE`: the GPUs accept work, which the platform's
/// boot sets only after their attestation passed.
pub const ACCEPTING_CLIENT_REQUESTS_TRUE: u32 = 1;

/// What the driver reports about the system's confidential-computing state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CcState {
    /// `nvmlConfComputeSystemState_t::environment`.
    pub environment: u32,
    /// `nvmlConfComputeSystemState_t::ccFeature`.
    pub cc_feature: u32,
    /// `nvmlConfComputeSystemState_t::devToolsMode`.
    pub devtools_mode: u32,
    /// `nvmlSystemGetConfComputeGpusReadyState`'s `isAcceptingWork`.
    pub ready_state: u32,
}

impl CcState {
    /// The state [`check`] accepts, and the only one.
    pub const REQUIRED: CcState = CcState {
        environment: ENVIRONMENT_PROD,
        cc_feature: CC_FEATURE_ENABLED,
        devtools_mode: DEVTOOLS_MODE_OFF,
        ready_state: ACCEPTING_CLIENT_REQUESTS_TRUE,
    };
}

/// Whether the reported state is the one a production node runs in: confidential computing
/// enabled, DevTools mode off, the production environment, and the GPUs accepting work.
/// Every other value of each field, an unknown one included, is refused, and the refusal
/// names the first field that differs.
pub fn check(state: &CcState) -> Result<(), String> {
    let wanted = CcState::REQUIRED;
    let fields = [
        (
            "confidential computing is not enabled (ccFeature",
            state.cc_feature,
            wanted.cc_feature,
        ),
        (
            "DevTools mode is not off (devToolsMode",
            state.devtools_mode,
            wanted.devtools_mode,
        ),
        (
            "the environment is not production (environment",
            state.environment,
            wanted.environment,
        ),
        (
            "the GPUs are not accepting work (ready state",
            state.ready_state,
            wanted.ready_state,
        ),
    ];
    match fields.iter().find(|(_, got, want)| got != want) {
        None => Ok(()),
        Some((what, got, want)) => Err(format!("{what} is {got}, not {want})")),
    }
}

/// Queries the driver through NVML (`cuda` feature).
#[cfg(feature = "cuda")]
pub fn query() -> Result<CcState, String> {
    nvml::query()
}

#[cfg(feature = "cuda")]
mod nvml {
    //! The few NVML entry points the check needs, resolved from the driver's library at run
    //! time. Declarations follow `nvml.h` (driver 595.71.05): every function returns
    //! `nvmlReturn_t`, a C enum whose `NVML_SUCCESS` is 0.

    use std::ffi::{c_char, c_int, c_uint};

    use super::CcState;

    /// The driver's NVML, by soname (what the NVIDIA container runtime mounts).
    const LIBRARY: &str = "libnvidia-ml.so.1";

    /// `NVML_SUCCESS`.
    const SUCCESS: c_int = 0;

    /// `nvmlConfComputeSystemState_t`.
    #[repr(C)]
    #[derive(Default)]
    struct SystemState {
        environment: c_uint,
        cc_feature: c_uint,
        dev_tools_mode: c_uint,
    }

    type Init = unsafe extern "C" fn() -> c_int;
    type Shutdown = unsafe extern "C" fn() -> c_int;
    type ErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
    type GetState = unsafe extern "C" fn(*mut SystemState) -> c_int;
    type GetReadyState = unsafe extern "C" fn(*mut c_uint) -> c_int;

    pub(super) fn query() -> Result<CcState, String> {
        // SAFETY: loading the driver's NVML runs its initialisers; it is the vendor's
        // library for this purpose, loaded by its documented soname.
        let lib = unsafe { libloading::Library::new(LIBRARY) }
            .map_err(|e| format!("cannot load the NVIDIA management library ({LIBRARY}): {e}"))?;
        // SAFETY: each symbol is resolved with the signature `nvml.h` declares for it.
        let (init, shutdown, error_string, get_state, get_ready) = unsafe {
            let sym = |name: &str| format!("{LIBRARY} has no {name}");
            (
                *lib.get::<Init>("nvmlInit_v2")
                    .map_err(|_| sym("nvmlInit_v2"))?,
                *lib.get::<Shutdown>("nvmlShutdown")
                    .map_err(|_| sym("nvmlShutdown"))?,
                *lib.get::<ErrorString>("nvmlErrorString")
                    .map_err(|_| sym("nvmlErrorString"))?,
                *lib.get::<GetState>("nvmlSystemGetConfComputeState")
                    .map_err(|_| sym("nvmlSystemGetConfComputeState"))?,
                *lib.get::<GetReadyState>("nvmlSystemGetConfComputeGpusReadyState")
                    .map_err(|_| sym("nvmlSystemGetConfComputeGpusReadyState"))?,
            )
        };
        let failed = |call: &str, code: c_int| {
            // SAFETY: `nvmlErrorString` returns a static, NUL-terminated string for any
            // code (or null, which is handled).
            let text = unsafe {
                let p = error_string(code);
                if p.is_null() {
                    String::new()
                } else {
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            format!("{call} failed ({code}: {text})")
        };

        // SAFETY: no arguments.
        let code = unsafe { init() };
        if code != SUCCESS {
            return Err(failed("nvmlInit_v2", code));
        }
        let mut state = SystemState::default();
        let mut ready: c_uint = 0;
        // SAFETY: NVML is initialised, and each pointer is to a live value of the type the
        // function fills.
        let result = unsafe {
            match get_state(&mut state) {
                SUCCESS => match get_ready(&mut ready) {
                    SUCCESS => Ok(()),
                    code => Err(failed("nvmlSystemGetConfComputeGpusReadyState", code)),
                },
                code => Err(failed("nvmlSystemGetConfComputeState", code)),
            }
        };
        // SAFETY: balances the successful `nvmlInit_v2` above.
        unsafe { shutdown() };
        result?;
        Ok(CcState {
            environment: state.environment,
            cc_feature: state.cc_feature,
            devtools_mode: state.dev_tools_mode,
            ready_state: ready,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_production_cc_state_is_accepted() {
        assert_eq!(check(&CcState::REQUIRED), Ok(()));
        let base = CcState::REQUIRED;
        for (state, needle) in [
            (
                CcState {
                    cc_feature: 0,
                    ..base
                },
                "confidential computing is not enabled (ccFeature is 0, not 1)",
            ),
            (
                CcState {
                    devtools_mode: 1,
                    ..base
                },
                "DevTools mode is not off (devToolsMode is 1, not 0)",
            ),
            (
                CcState {
                    environment: 1,
                    ..base
                },
                "the environment is not production (environment is 1, not 2)",
            ),
            (
                CcState {
                    environment: 0,
                    ..base
                },
                "environment is 0",
            ),
            (
                CcState {
                    ready_state: 0,
                    ..base
                },
                "the GPUs are not accepting work (ready state is 0, not 1)",
            ),
            // Values the header does not define are refused too.
            (
                CcState {
                    cc_feature: 7,
                    ..base
                },
                "ccFeature is 7",
            ),
            (
                CcState {
                    devtools_mode: u32::MAX,
                    ..base
                },
                "devToolsMode is 4294967295",
            ),
            (
                CcState {
                    ready_state: 2,
                    ..base
                },
                "ready state is 2",
            ),
        ] {
            let err = check(&state).unwrap_err();
            assert!(err.contains(needle), "{state:?}: {err}");
        }
    }

    /// With every field off, the refusal names the feature first.
    #[test]
    fn a_gpu_without_cc_is_refused_for_its_feature_first() {
        let off = CcState {
            environment: 0,
            cc_feature: 0,
            devtools_mode: 0,
            ready_state: 0,
        };
        assert!(
            check(&off)
                .unwrap_err()
                .starts_with("confidential computing")
        );
    }
}
