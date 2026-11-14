//! Ahead-of-time compiled GPU kernels and their reproducibility manifest.
//!
//! The kernels are compiled outside Cargo, by the Nix build in `nix/`, on a
//! GPU-less Linux builder: pinned CUDA toolkit, pinned upstream sources,
//! explicit flags, one cubin per target architecture plus a per-kernel fatbin.
//! That build writes `manifest.json`; the committed copy,
//! `kernels.manifest.json`, is compiled into this crate.
//!
//! The crate never compiles CUDA and needs no CUDA toolkit, so it builds and
//! tests anywhere. At run time a host points [`ArtifactDir`] at a build output
//! and gets back image bytes only after their size and SHA-256 match the
//! compiled-in manifest; those bytes are what a driver-API loader hands to
//! `cuModuleLoadData`. Each kernel entry point publishes its launch contract
//! inside the image as a device global, decoded by [`KernelMeta::from_bytes`].
//! The manifest binds every entry to exactly one such record and every record
//! to exactly one entry ([`Entry::meta`], [`Cubin::entry`],
//! [`Cubin::entry_for_meta`]): `<entry>_meta` for entries we name ourselves,
//! an explicit alias for mangled template instances.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// The manifest of the kernel build this crate was compiled against.
pub const MANIFEST_JSON: &str = include_str!("../kernels.manifest.json");

/// The manifest schema this crate reads.
pub const SCHEMA_VERSION: u32 = 2;

/// A kernel build: toolchain, upstream pins, flags, inputs, and every image.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub toolchain: Toolchain,
    pub sources: BTreeMap<String, SourcePin>,
    pub compile: CompileFlags,
    /// Every committed file the build read, with the rendered FlashInfer
    /// configuration it generated.
    pub inputs: Vec<InputFile>,
    pub kernels: Vec<Cubin>,
    pub fatbins: Vec<Fatbin>,
}

/// The pinned toolchain. Deliberately free of anything that names the build
/// host: builders of different CPU architectures must produce, and compare
/// against, the same manifest.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Toolchain {
    /// nixpkgs revision providing the CUDA toolkit and host compiler.
    pub nixpkgs: String,
    /// CUDA toolkit (nvcc) version.
    pub cuda: String,
    /// Host compiler nvcc drives for preprocessing.
    pub host_compiler: String,
    /// Version of every CUDA toolkit component in the build.
    pub components: BTreeMap<String, String>,
}

/// An upstream source tree, fetched by commit and verified by NAR hash.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourcePin {
    pub repository: String,
    pub rev: String,
    pub nar_hash: String,
    pub license: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompileFlags {
    /// Flags added per source file (`<source>` is its path in the crate).
    pub per_source_flags: Vec<String>,
    pub common_flags: Vec<String>,
    pub profiles: BTreeMap<String, Profile>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub flags: Vec<String>,
    /// Include roots, relative to the build's include tree (upstream trees
    /// by name, plus `generated/`).
    pub includes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputFile {
    pub path: String,
    pub sha256: String,
}

/// One kernel translation unit compiled for one target architecture.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cubin {
    pub name: String,
    /// `sm_100a`, `sm_103a`, or `sm_100f`.
    pub arch: String,
    pub source: String,
    pub profile: String,
    /// Path relative to the build output.
    pub file: String,
    pub sha256: String,
    pub size: u64,
    /// Kernel entry points, by symbol (as `cuModuleGetFunction` takes it),
    /// each bound to its launch-contract record.
    pub entries: Vec<Entry>,
    /// Every launch-contract global in the image (32 bytes each). In a valid
    /// manifest these correspond one to one with the entries' records.
    pub meta: Vec<String>,
    /// Count of tensor-core instructions in the SASS, by opcode family
    /// (`UTC*` is tcgen05, `HMMA` is `mma.sync`).
    pub mma_sass: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub symbol: String,
    pub demangled: String,
    /// The device global holding this entry's launch contract (read with
    /// `cuModuleGetGlobal`): `<symbol>_meta` for entries we name ourselves,
    /// or the alias `csrc/kernels.json` binds to a mangled entry.
    pub meta: String,
}

/// The arch-specific cubins of one kernel, bundled for the driver to pick
/// the exact-match image.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fatbin {
    pub name: String,
    pub file: String,
    pub sha256: String,
    pub size: u64,
    pub archs: Vec<String>,
}

impl Manifest {
    /// The compiled-in manifest. Its validity is a build invariant checked by
    /// this crate's tests, so a failure here is a packaging bug.
    pub fn embedded() -> &'static Manifest {
        static MANIFEST: OnceLock<Manifest> = OnceLock::new();
        MANIFEST.get_or_init(|| {
            Manifest::parse(MANIFEST_JSON).expect("compiled-in kernel manifest is valid")
        })
    }

    /// Parse and structurally validate a manifest.
    pub fn parse(json: &str) -> Result<Manifest, ManifestError> {
        let manifest: Manifest = serde_json::from_str(json).map_err(ManifestError::Json)?;
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<(), ManifestError> {
        let invalid = |what: String| Err(ManifestError::Invalid(what));
        if self.schema_version != SCHEMA_VERSION {
            return invalid(format!("schema_version {}", self.schema_version));
        }
        let mut seen = BTreeSet::new();
        for cubin in &self.kernels {
            if !seen.insert((cubin.name.as_str(), cubin.arch.as_str())) {
                return invalid(format!("duplicate cubin {} {}", cubin.name, cubin.arch));
            }
            if !self.compile.profiles.contains_key(&cubin.profile) {
                return invalid(format!("{}: unknown profile {}", cubin.name, cubin.profile));
            }
            check_artifact(&cubin.file, &cubin.sha256)?;
            if cubin.entries.is_empty() {
                return invalid(format!("{} {}: no entry points", cubin.name, cubin.arch));
            }
            let records: BTreeSet<&str> = cubin.meta.iter().map(String::as_str).collect();
            let bound: BTreeSet<&str> = cubin.entries.iter().map(|e| e.meta.as_str()).collect();
            let symbols: BTreeSet<&str> = cubin.entries.iter().map(|e| e.symbol.as_str()).collect();
            if records.len() != cubin.meta.len()
                || bound.len() != cubin.entries.len()
                || symbols.len() != cubin.entries.len()
                || bound != records
            {
                return invalid(format!(
                    "{} {}: entries and launch-contract records do not correspond one to one",
                    cubin.name, cubin.arch
                ));
            }
        }
        let mut fatbin_names = BTreeSet::new();
        for fatbin in &self.fatbins {
            if !fatbin_names.insert(fatbin.name.as_str()) {
                return invalid(format!("duplicate fatbin {}", fatbin.name));
            }
            check_artifact(&fatbin.file, &fatbin.sha256)?;
            for arch in &fatbin.archs {
                if self.cubin(&fatbin.name, arch).is_none() {
                    return invalid(format!("fatbin {} lacks cubin {arch}", fatbin.name));
                }
            }
        }
        for input in &self.inputs {
            check_sha256(&input.sha256)?;
        }
        Ok(())
    }

    pub fn cubin(&self, name: &str, arch: &str) -> Option<&Cubin> {
        self.kernels
            .iter()
            .find(|c| c.name == name && c.arch == arch)
    }

    pub fn fatbin(&self, name: &str) -> Option<&Fatbin> {
        self.fatbins.iter().find(|f| f.name == name)
    }
}

impl Cubin {
    /// The entry with this symbol, carrying the name of its launch-contract
    /// record.
    pub fn entry(&self, symbol: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.symbol == symbol)
    }

    /// The entry a launch-contract record describes (the inverse of
    /// [`Entry::meta`]): how a host that names kernels by their record finds
    /// the mangled symbol to pass to `cuModuleGetFunction`.
    pub fn entry_for_meta(&self, meta: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.meta == meta)
    }
}

fn check_artifact(file: &str, sha256: &str) -> Result<(), ManifestError> {
    let path = Path::new(file);
    if !path.components().all(|c| matches!(c, Component::Normal(_))) {
        return Err(ManifestError::Invalid(format!("artifact path {file}")));
    }
    check_sha256(sha256)
}

fn check_sha256(sha256: &str) -> Result<(), ManifestError> {
    if sha256.len() == 64
        && sha256
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        Ok(())
    } else {
        Err(ManifestError::Invalid(format!("sha256 {sha256:?}")))
    }
}

/// Lowercase hex SHA-256, the manifest's digest encoding.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// A kernel build output on disk, read only through the manifest.
#[derive(Debug, Clone)]
pub struct ArtifactDir<'m> {
    root: PathBuf,
    manifest: &'m Manifest,
}

impl<'m> ArtifactDir<'m> {
    pub fn new(root: impl Into<PathBuf>, manifest: &'m Manifest) -> Self {
        ArtifactDir {
            root: root.into(),
            manifest,
        }
    }

    /// The verified bytes of one cubin.
    pub fn cubin(&self, name: &str, arch: &str) -> Result<Vec<u8>, LoadError> {
        let cubin = self
            .manifest
            .cubin(name, arch)
            .ok_or_else(|| LoadError::Unknown(format!("{name} {arch}")))?;
        self.read_verified(&cubin.file, cubin.size, &cubin.sha256)
    }

    /// The verified bytes of one kernel's fatbin.
    pub fn fatbin(&self, name: &str) -> Result<Vec<u8>, LoadError> {
        let fatbin = self
            .manifest
            .fatbin(name)
            .ok_or_else(|| LoadError::Unknown(name.to_owned()))?;
        self.read_verified(&fatbin.file, fatbin.size, &fatbin.sha256)
    }

    /// Verify every image the manifest lists.
    pub fn verify_all(&self) -> Result<(), LoadError> {
        for cubin in &self.manifest.kernels {
            self.read_verified(&cubin.file, cubin.size, &cubin.sha256)?;
        }
        for fatbin in &self.manifest.fatbins {
            self.read_verified(&fatbin.file, fatbin.size, &fatbin.sha256)?;
        }
        Ok(())
    }

    fn read_verified(&self, file: &str, size: u64, sha256: &str) -> Result<Vec<u8>, LoadError> {
        let path = self.root.join(file);
        let bytes = std::fs::read(&path).map_err(|source| LoadError::Io {
            path: path.clone(),
            source,
        })?;
        if bytes.len() as u64 != size {
            return Err(LoadError::SizeMismatch {
                path,
                expected: size,
                actual: bytes.len() as u64,
            });
        }
        let actual = sha256_hex(&bytes);
        if actual != sha256 {
            return Err(LoadError::DigestMismatch {
                path,
                expected: sha256.to_owned(),
                actual,
            });
        }
        Ok(bytes)
    }
}

/// The launch contract a kernel publishes as a device global, the one its
/// manifest [`Entry::meta`] names (`EidolaKernelMeta` in
/// `csrc/eidola_kernel.cuh`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelMeta {
    pub block: [u32; 3],
    /// Dynamic shared memory the launch requests; above 48 KiB the function
    /// must first opt in via `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
    pub dynamic_smem_bytes: u32,
    /// Thread-block cluster shape; `[1, 1, 1]` means no cluster attribute.
    pub cluster: [u32; 3],
    /// Size of the single by-value parameter struct, or 0 for kernels that
    /// take a plain argument list.
    pub params_bytes: u32,
}

impl KernelMeta {
    pub const SIZE: usize = 32;

    /// Decode the 32 bytes copied out of the device global. GPU memory is
    /// little-endian.
    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> KernelMeta {
        let word = |i: usize| {
            u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4-byte slice"))
        };
        KernelMeta {
            block: [word(0), word(1), word(2)],
            dynamic_smem_bytes: word(3),
            cluster: [word(4), word(5), word(6)],
            params_bytes: word(7),
        }
    }
}

#[derive(Debug)]
pub enum ManifestError {
    Json(serde_json::Error),
    Invalid(String),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestError::Json(e) => write!(f, "kernel manifest is not valid JSON: {e}"),
            ManifestError::Invalid(what) => write!(f, "kernel manifest is invalid: {what}"),
        }
    }
}

impl std::error::Error for ManifestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ManifestError::Json(e) => Some(e),
            ManifestError::Invalid(_) => None,
        }
    }
}

#[derive(Debug)]
pub enum LoadError {
    /// The manifest has no such kernel (or no such kernel for that arch).
    Unknown(String),
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    SizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    DigestMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Unknown(what) => write!(f, "no kernel image {what} in the manifest"),
            LoadError::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            LoadError::SizeMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "{}: {actual} bytes, manifest says {expected}",
                path.display()
            ),
            LoadError::DigestMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "{}: sha256 {actual}, manifest says {expected}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
