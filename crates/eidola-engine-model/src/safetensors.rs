//! Memory-mapped safetensors files with per-tensor lazy access.
//!
//! A [`WeightSet`] is every `*.safetensors` file in a directory. Nothing is
//! read until a tensor is asked for, and a directory holding only some shards
//! (or files re-packed with a subset of tensors) is fine: tensors that are not
//! present are an error only when something asks for them.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::numeric::bf16_to_f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    F8E4M3,
    U8,
    I32,
    I64,
}

impl Dtype {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "BF16" => Dtype::Bf16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "F8_E4M3" => Dtype::F8E4M3,
            "U8" => Dtype::U8,
            "I32" => Dtype::I32,
            "I64" => Dtype::I64,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::I64 => 8,
            Dtype::F8E4M3 | Dtype::U8 => 1,
        }
    }
}

#[derive(Debug, Clone)]
struct TensorEntry {
    file: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    /// Absolute byte range in the file.
    start: usize,
    end: usize,
}

struct MappedFile {
    path: PathBuf,
    name: String,
    map: Mmap,
    metadata: BTreeMap<String, String>,
}

/// A borrowed tensor: raw little-endian bytes plus dtype and shape.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Decode a BF16, F16 or F32 tensor to f32 (exact for all three).
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        match self.dtype {
            Dtype::F32 => Ok(self
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect()),
            Dtype::Bf16 => Ok(self
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| bf16_to_f32(u16::from_le_bytes(b)))
                .collect()),
            Dtype::F16 => Ok(self
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| f16_to_f32(u16::from_le_bytes(b)))
                .collect()),
            other => Err(Error::layout(
                name,
                format!("expected a float tensor, found {other:?}"),
            )),
        }
    }

    /// Decode an I64 or I32 tensor.
    pub fn to_i64(&self, name: &str) -> Result<Vec<i64>> {
        match self.dtype {
            Dtype::I64 => Ok(self
                .data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&b| i64::from_le_bytes(b))
                .collect()),
            Dtype::I32 => Ok(self
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| i32::from_le_bytes(b) as i64)
                .collect()),
            other => Err(Error::layout(
                name,
                format!("expected an integer tensor, found {other:?}"),
            )),
        }
    }

    pub fn expect(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<()> {
        if self.dtype != dtype || self.shape != shape {
            return Err(Error::layout(
                name,
                format!(
                    "expected {dtype:?} {shape:?}, found {:?} {:?}",
                    self.dtype, self.shape
                ),
            ));
        }
        Ok(())
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal: renormalise.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// File names of the non-weight files whose bytes change what a checkpoint
/// directory loads as: the model configuration, and the index (whose metadata,
/// such as `tp_size`, decides how fused QKV rows are de-interleaved).
pub const SEMANTIC_FILES: [&str; 2] = ["config.json", "model.safetensors.index.json"];

/// Every safetensors file in one directory, indexed by tensor name.
///
/// **Integrity covers every semantic input.** [`WeightSet::open_dir`] reads the
/// [`SEMANTIC_FILES`] present in the directory once, keeps their bytes, and
/// uses only those bytes afterwards ([`WeightSet::model_config`], index
/// metadata). [`WeightSet::sha256_manifest`] and [`WeightSet::verify_sha256`]
/// cover them alongside the shards, so one manifest pins everything the loader
/// reads.
pub struct WeightSet {
    dir: PathBuf,
    files: Vec<MappedFile>,
    tensors: HashMap<String, TensorEntry>,
    index_metadata: BTreeMap<String, String>,
    /// The semantic files read, by file name, exactly as hashed.
    semantic: BTreeMap<String, Vec<u8>>,
}

impl std::fmt::Debug for WeightSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightSet")
            .field("dir", &self.dir)
            .field("files", &self.files.len())
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

impl WeightSet {
    /// Open every `*.safetensors` file in `dir`. Also reads
    /// `model.safetensors.index.json` metadata when present (for example the
    /// checkpoint's `tp_size`); its weight map is not required to be complete.
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| Error::io(dir, e))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        let mut set = Self::open_files(&paths)?;
        set.dir = dir.to_path_buf();

        for name in SEMANTIC_FILES {
            let path = dir.join(name);
            if path.exists() {
                let bytes = std::fs::read(&path).map_err(|e| Error::io(&path, e))?;
                set.semantic.insert(name.to_string(), bytes);
            }
        }
        let index = dir.join("model.safetensors.index.json");
        if let Some(bytes) = set.semantic.get("model.safetensors.index.json") {
            let v: Value = serde_json::from_slice(bytes).map_err(|e| Error::Json {
                what: index.display().to_string(),
                source: e,
            })?;
            if let Some(m) = v.get("metadata").and_then(Value::as_object) {
                for (k, val) in m {
                    set.index_metadata.insert(k.clone(), value_to_string(val));
                }
            }
        }
        Ok(set)
    }

    pub fn open_files(paths: &[PathBuf]) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        for (fi, path) in paths.iter().enumerate() {
            let (mapped, entries) = map_file(path, fi)?;
            for (name, entry) in entries {
                if tensors.contains_key(&name) {
                    return Err(Error::Safetensors {
                        path: path.clone(),
                        reason: format!("tensor {name} also appears in another file"),
                    });
                }
                tensors.insert(name, entry);
            }
            files.push(mapped);
        }
        let dir = paths
            .first()
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
            .unwrap_or_default();
        Ok(WeightSet {
            dir,
            files,
            tensors,
            index_metadata: BTreeMap::new(),
            semantic: BTreeMap::new(),
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<TensorView<'_>> {
        let e = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        Ok(TensorView {
            dtype: e.dtype,
            shape: &e.shape,
            data: &self.files[e.file].map[e.start..e.end],
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// A metadata value from the index file, else from any shard's
    /// `__metadata__`.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        if let Some(v) = self.index_metadata.get(key) {
            return Some(v);
        }
        self.files
            .iter()
            .find_map(|f| f.metadata.get(key))
            .map(String::as_str)
    }

    pub fn file_names(&self) -> Vec<&str> {
        self.files.iter().map(|f| f.name.as_str()).collect()
    }

    /// The model configuration from the `config.json` this set read (and hashed)
    /// in [`WeightSet::open_dir`].
    pub fn model_config(&self) -> Result<crate::ModelConfig> {
        let bytes = self
            .semantic
            .get("config.json")
            .ok_or_else(|| Error::Safetensors {
                path: self.dir.join("config.json"),
                reason: "no config.json was read with these weights".into(),
            })?;
        crate::ModelConfig::from_json_bytes(bytes, "config.json")
    }

    /// sha256 of every loaded file (shards and semantic files), keyed by file
    /// name.
    pub fn sha256_manifest(&self) -> BTreeMap<String, String> {
        let mut out: BTreeMap<String, String> = self
            .files
            .par_iter()
            .map(|f| (f.name.clone(), hex::encode(Sha256::digest(&f.map[..]))))
            .collect();
        for (name, bytes) in &self.semantic {
            out.insert(name.clone(), hex::encode(Sha256::digest(bytes)));
        }
        out
    }

    /// Check loaded files against an expected `file name → sha256 hex`
    /// manifest. Every manifest entry must name a loaded file, and every
    /// loaded file must be in the manifest.
    pub fn verify_sha256(&self, expected: &BTreeMap<String, String>) -> Result<()> {
        for name in expected.keys() {
            if !self.files.iter().any(|f| &f.name == name) && !self.semantic.contains_key(name) {
                return Err(Error::IntegrityUnknownFile(name.clone()));
            }
        }
        let actual = self.sha256_manifest();
        for (name, digest) in &actual {
            match expected.get(name) {
                None => {
                    return Err(Error::Integrity {
                        file: name.clone(),
                        expected: "(not in manifest)".into(),
                        actual: digest.clone(),
                    });
                }
                Some(want) if !want.eq_ignore_ascii_case(digest) => {
                    return Err(Error::Integrity {
                        file: name.clone(),
                        expected: want.clone(),
                        actual: digest.clone(),
                    });
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn path_of(&self, file_name: &str) -> Option<&Path> {
        self.files
            .iter()
            .find(|f| f.name == file_name)
            .map(|f| f.path.as_path())
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn map_file(path: &Path, file_index: usize) -> Result<(MappedFile, Vec<(String, TensorEntry)>)> {
    let bad = |reason: String| Error::Safetensors {
        path: path.to_path_buf(),
        reason,
    };
    let file = File::open(path).map_err(|e| Error::io(path, e))?;
    // SAFETY: the mapping is read-only and the weight files are treated as
    // immutable for the lifetime of the `WeightSet`; modifying or truncating a
    // file while it is mapped is outside this type's contract.
    let map = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(path, e))?;
    if map.len() < 8 {
        return Err(bad("shorter than the 8-byte header length".into()));
    }
    let header_len = u64::from_le_bytes(map[..8].try_into().unwrap());
    let data_start = usize::try_from(header_len)
        .ok()
        .and_then(|h| h.checked_add(8))
        .filter(|&s| s <= map.len())
        .ok_or_else(|| bad(format!("header length {header_len} exceeds file size")))?;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&map[8..data_start])
        .map_err(|e| bad(format!("header is not a JSON object: {e}")))?;

    let mut metadata = BTreeMap::new();
    let mut entries = Vec::new();
    for (name, v) in header {
        if name == "__metadata__" {
            if let Some(m) = v.as_object() {
                for (k, val) in m {
                    metadata.insert(k.clone(), value_to_string(val));
                }
            }
            continue;
        }
        let dtype_s = v
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| bad(format!("{name}: missing dtype")))?;
        let dtype = Dtype::parse(dtype_s).ok_or_else(|| bad(format!("{name}: dtype {dtype_s}")))?;
        let shape: Vec<usize> = v
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| bad(format!("{name}: missing shape")))?
            .iter()
            .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
            .collect::<Option<_>>()
            .ok_or_else(|| bad(format!("{name}: non-integer shape")))?;
        let offs = v
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .and_then(|a| Some((a[0].as_u64()?, a[1].as_u64()?)))
            .ok_or_else(|| bad(format!("{name}: missing data_offsets")))?;
        let (start, end) = tensor_range(data_start, offs, &shape, dtype.size(), map.len())
            .ok_or_else(|| {
                bad(format!(
                    "{name}: data_offsets {offs:?} do not hold {dtype:?} {shape:?}"
                ))
            })?;
        entries.push((
            name,
            TensorEntry {
                file: file_index,
                dtype,
                shape,
                start,
                end,
            },
        ));
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok((
        MappedFile {
            path: path.to_path_buf(),
            name,
            map,
            metadata,
        },
        entries,
    ))
}

/// The absolute byte range `[start, end)` of a tensor whose header gives
/// `offs` (relative to the data section at `data_start`), or `None` unless the
/// range lies inside a file of `file_len` bytes and holds exactly
/// `product(shape) * elem_size` bytes. Every header-derived quantity is
/// untrusted, so all arithmetic is checked: an overflowing header is refused,
/// never wrapped into a range over unrelated bytes.
fn tensor_range(
    data_start: usize,
    offs: (u64, u64),
    shape: &[usize],
    elem_size: usize,
    file_len: usize,
) -> Option<(usize, usize)> {
    let rel_start = usize::try_from(offs.0).ok()?;
    let rel_end = usize::try_from(offs.1).ok()?;
    let start = data_start.checked_add(rel_start)?;
    let end = data_start.checked_add(rel_end)?;
    let want = shape
        .iter()
        .try_fold(elem_size, |acc, &d| acc.checked_mul(d))?;
    (start <= end && end <= file_len && end - start == want).then_some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a safetensors file with `header` and `data` and open it.
    fn open_with_header(tag: &str, header: &str, data: &[u8]) -> Result<WeightSet> {
        let dir = std::env::temp_dir().join(format!(
            "eidola-engine-model-st-{}-{tag}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(&path, bytes).unwrap();
        let result = WeightSet::open_files(std::slice::from_ref(&path));
        std::fs::remove_dir_all(&dir).unwrap();
        result
    }

    fn assert_refused(tag: &str, header: &str) {
        match open_with_header(tag, header, &[0; 16]) {
            Err(Error::Safetensors { .. }) => {}
            Err(other) => panic!("{tag}: wrong error {other}"),
            Ok(_) => panic!("{tag}: malformed header accepted"),
        }
    }

    #[test]
    fn well_formed_header_opens() {
        let set = open_with_header(
            "ok",
            r#"{"t":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#,
            &[0; 16],
        )
        .unwrap();
        assert_eq!(set.get("t").unwrap().data.len(), 16);
    }

    /// Header-derived sizes and offsets that overflow `usize` arithmetic are
    /// refused with `Error::Safetensors`; unchecked, the debug build panics
    /// and the release build wraps into a range that passes the bounds check.
    #[test]
    fn overflowing_headers_are_refused() {
        let max = u64::MAX;
        // The shape product overflows; wrapped, 2^62 * 2^2 * 4 bytes is 0,
        // which an empty range would satisfy.
        assert_refused(
            "shape-product",
            r#"{"t":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,0]}}"#,
        );
        // The element-size multiplication overflows (2^62 * 4 = 2^64).
        assert_refused(
            "elem-size",
            r#"{"t":{"dtype":"F32","shape":[4611686018427387904],"data_offsets":[0,0]}}"#,
        );
        // `data_start + offset` overflows for both ends; wrapped, the range
        // lands just inside the header and holds exactly 16 bytes.
        assert_refused(
            "offset-add",
            &format!(
                r#"{{"t":{{"dtype":"F32","shape":[4],"data_offsets":[{},{max}]}}}}"#,
                max - 16
            ),
        );
        // Offsets out of order.
        assert_refused(
            "reversed",
            r#"{"t":{"dtype":"F32","shape":[0],"data_offsets":[8,0]}}"#,
        );
        // Past the end of the file.
        assert_refused(
            "past-end",
            r#"{"t":{"dtype":"F32","shape":[8],"data_offsets":[0,32]}}"#,
        );
    }

    #[test]
    fn oversized_header_length_is_refused() {
        let dir = std::env::temp_dir().join(format!(
            "eidola-engine-model-st-{}-header-len",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut bytes = u64::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(&path, bytes).unwrap();
        let result = WeightSet::open_files(std::slice::from_ref(&path));
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(matches!(result, Err(Error::Safetensors { .. })));
    }

    #[test]
    fn f16_decode() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
