use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::apply;
use crate::format::{MachOEntry, PlacementRecord};
use crate::fs_guard;
use crate::macho;
use crate::{ApplyError, DetachError, sha256_hex};

const RECORD_NAME: &str = "eidola-placement.json";

pub(crate) fn detach(
    signed_bundle: &Path,
    unsigned_bundle: &Path,
    output_dir: &Path,
) -> Result<PathBuf, DetachError> {
    fs_guard::root(signed_bundle)?;
    fs_guard::root(unsigned_bundle)?;
    let bundle_name = signed_bundle
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| DetachError::InvalidDestination {
            path: signed_bundle.to_path_buf(),
            reason: "signed bundle has no UTF-8 basename".into(),
        })?;
    if unsigned_bundle.file_name().and_then(|name| name.to_str()) != Some(bundle_name) {
        return Err(DetachError::InvalidDestination {
            path: unsigned_bundle.to_path_buf(),
            reason: format!("unsigned bundle must also be named `{bundle_name}`"),
        });
    }

    validate_destination(signed_bundle, unsigned_bundle, output_dir, bundle_name)?;

    let mut record = PlacementRecord {
        schema_version: 1,
        bundle: bundle_name.into(),
        inputs: BTreeMap::new(),
        machos: BTreeMap::new(),
        files: BTreeMap::new(),
    };
    for (relative, path) in fs_guard::regular_files(unsigned_bundle)? {
        let relative = relative
            .to_str()
            .ok_or_else(|| DetachError::InvalidDestination {
                path: path.clone(),
                reason: "unsigned bundle path is not UTF-8".into(),
            })?
            .to_owned();
        record
            .inputs
            .insert(relative, format!("sha256:{}", sha256_hex(&read(&path)?)));
    }
    let mut signatures = Vec::new();
    let macos = fs_guard::existing(signed_bundle, Path::new("Contents/MacOS"))?;
    let mut entries: Vec<_> = fs::read_dir(&macos)
        .map_err(|source| DetachError::Read {
            path: macos.clone(),
            source,
        })?
        .collect::<Result<_, _>>()
        .map_err(|source| DetachError::Read {
            path: macos.clone(),
            source,
        })?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let file_type = entry.file_type().map_err(|source| DetachError::Read {
            path: entry.path(),
            source,
        })?;
        if file_type.is_symlink() {
            return Err(fs_guard::GuardError::Symlink(entry.path()).into());
        }
        if !file_type.is_file() {
            continue;
        }
        let name =
            entry
                .file_name()
                .into_string()
                .map_err(|_| DetachError::InvalidDestination {
                    path: entry.path(),
                    reason: "Mach-O filename is not UTF-8".into(),
                })?;
        let relative = format!("Contents/MacOS/{name}");
        let signed_path = fs_guard::existing(signed_bundle, Path::new(&relative))?;
        let signed = read(&signed_path)?;
        let unsigned_path = fs_guard::existing(unsigned_bundle, Path::new(&relative))?;
        let unsigned = read(&unsigned_path)?;
        let parsed = macho::parse(&signed).map_err(|reason| DetachError::InvalidMachO {
            path: PathBuf::from(&relative),
            reason,
        })?;
        let mut suffixes = BTreeSet::new();
        let mut signature_blobs = BTreeMap::new();
        for slice in &parsed.slices {
            let code_signature =
                slice
                    .facts
                    .code_signature
                    .as_ref()
                    .ok_or_else(|| DetachError::UnsignedSlice {
                        path: PathBuf::from(&relative),
                        arch: slice.facts.arch.clone(),
                    })?;
            let suffix =
                arch_suffix(&slice.facts.arch).ok_or_else(|| DetachError::InvalidSignature {
                    path: PathBuf::from(&relative),
                    arch: slice.facts.arch.clone(),
                    reason: "unsupported detached-signature architecture".into(),
                })?;
            if !suffixes.insert(suffix) {
                return Err(DetachError::InvalidSignature {
                    path: PathBuf::from(&relative),
                    arch: slice.facts.arch.clone(),
                    reason: "architecture shares a signapple filename with another slice".into(),
                });
            }
            let start = usize::try_from(slice.facts.header_offset)
                .ok()
                .and_then(|base| base.checked_add(code_signature.dataoff as usize))
                .ok_or_else(|| DetachError::InvalidSignature {
                    path: PathBuf::from(&relative),
                    arch: slice.facts.arch.clone(),
                    reason: "signature offset overflow".into(),
                })?;
            let end = start
                .checked_add(code_signature.datasize as usize)
                .ok_or_else(|| DetachError::InvalidSignature {
                    path: PathBuf::from(&relative),
                    arch: slice.facts.arch.clone(),
                    reason: "signature range overflow".into(),
                })?;
            let blob = signed
                .get(start..end)
                .ok_or_else(|| DetachError::InvalidSignature {
                    path: PathBuf::from(&relative),
                    arch: slice.facts.arch.clone(),
                    reason: "signature extends beyond file".into(),
                })?
                .to_vec();
            signatures.push((format!("{name}.{suffix}sign"), blob.clone()));
            signature_blobs.insert(slice.facts.arch.clone(), blob);
        }
        let unsigned_parsed =
            macho::parse(&unsigned).map_err(|reason| DetachError::InvalidMachO {
                path: PathBuf::from(&relative),
                reason,
            })?;
        let macho_entry = MachOEntry {
            input_sha256: sha256_hex(&unsigned),
            kind: parsed.kind,
            slices: parsed.slices.into_iter().map(|slice| slice.facts).collect(),
            output_sha256: sha256_hex(&signed),
            output_len: signed.len() as u64,
        };
        let rebuilt = apply::rebuild(
            &relative,
            &unsigned,
            &unsigned_parsed,
            &macho_entry,
            &signature_blobs,
        )
        .map_err(|error| map_rebuild_error(&relative, error))?;
        if rebuilt != signed {
            return Err(DetachError::IncompatibleInput {
                path: PathBuf::from(&relative),
                arch: "all".into(),
                reason: format!(
                    "reconstruction produced {} bytes hashing {}, signed target is {} bytes hashing {}",
                    rebuilt.len(),
                    sha256_hex(&rebuilt),
                    signed.len(),
                    macho_entry.output_sha256
                ),
            });
        }
        record.machos.insert(relative, macho_entry);
    }
    if record.machos.is_empty() {
        return Err(DetachError::InvalidDestination {
            path: macos,
            reason: "bundle contains no regular Mach-O files".into(),
        });
    }

    let mut plain_files = Vec::new();
    for relative in [
        "Contents/_CodeSignature/CodeResources",
        "Contents/CodeResources",
    ] {
        if let Some(path) = fs_guard::optional_existing(signed_bundle, Path::new(relative))?
            && path.is_file()
        {
            let data = read(&path)?;
            record
                .files
                .insert(relative.into(), format!("sha256:{}", sha256_hex(&data)));
            plain_files.push((relative, data));
        }
    }
    validate_signed_tree(signed_bundle, &record)?;
    let json = serde_json::to_vec_pretty(&record)?;

    let previous = validate_previous_output(output_dir, signed_bundle, unsigned_bundle)?;
    fs::create_dir_all(output_dir).map_err(|source| DetachError::Write {
        path: output_dir.to_path_buf(),
        source,
    })?;
    fs_guard::root(output_dir)?;
    clear_previous(output_dir, previous.as_deref())?;
    let material_root = fs_guard::for_write(output_dir, Path::new(bundle_name))?;
    let macos_out = material_root.join("Contents/MacOS");
    fs::create_dir_all(&macos_out).map_err(|source| DetachError::Write {
        path: macos_out.clone(),
        source,
    })?;
    fs_guard::existing(output_dir, Path::new(bundle_name))?;
    fs_guard::existing(&material_root, Path::new("Contents/MacOS"))?;
    for (name, data) in signatures {
        let relative = PathBuf::from("Contents/MacOS").join(name);
        let path = fs_guard::for_write(&material_root, &relative)?;
        fs::write(&path, data).map_err(|source| DetachError::Write { path, source })?;
    }
    for (relative, data) in plain_files {
        let relative = Path::new(relative);
        let path = fs_guard::for_write(&material_root, relative)?;
        let parent = path.parent().expect("detached plain file has a parent");
        fs::create_dir_all(parent).map_err(|source| DetachError::Write {
            path: parent.to_path_buf(),
            source,
        })?;
        fs_guard::for_write(&material_root, relative)?;
        fs::write(&path, data).map_err(|source| DetachError::Write { path, source })?;
    }
    let record_path = fs_guard::for_write(output_dir, Path::new(RECORD_NAME))?;
    let mut json_with_newline = json;
    json_with_newline.push(b'\n');
    fs::write(&record_path, json_with_newline).map_err(|source| DetachError::Write {
        path: record_path,
        source,
    })?;
    Ok(material_root)
}

fn validate_previous_output(
    output_dir: &Path,
    signed_bundle: &Path,
    unsigned_bundle: &Path,
) -> Result<Option<String>, DetachError> {
    match fs::symlink_metadata(output_dir) {
        Ok(_) => fs_guard::root(output_dir)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(DetachError::Read {
                path: output_dir.to_path_buf(),
                source,
            });
        }
    }
    let mut entries: Vec<_> = fs::read_dir(output_dir)
        .map_err(|source| DetachError::Read {
            path: output_dir.to_path_buf(),
            source,
        })?
        .collect::<Result<_, _>>()
        .map_err(|source| DetachError::Read {
            path: output_dir.to_path_buf(),
            source,
        })?;
    entries.sort_by_key(fs::DirEntry::file_name);
    if entries.is_empty() {
        return Ok(None);
    }
    for entry in &entries {
        if entry
            .file_type()
            .map_err(|source| DetachError::Read {
                path: entry.path(),
                source,
            })?
            .is_symlink()
        {
            return Err(fs_guard::GuardError::Symlink(entry.path()).into());
        }
    }
    let actual: BTreeSet<_> = entries
        .iter()
        .map(|entry| PathBuf::from(entry.file_name()))
        .collect();
    let Some(record_path) = fs_guard::optional_existing(output_dir, Path::new(RECORD_NAME))? else {
        let entry = actual.first().expect("nonempty output has a first entry");
        return Err(DetachError::InvalidDestination {
            path: output_dir.to_path_buf(),
            reason: format!("unexpected detached output entry `{}`", entry.display()),
        });
    };
    if !record_path.is_file() {
        return Err(DetachError::InvalidDestination {
            path: record_path,
            reason: "previous placement record is not a regular file".into(),
        });
    }
    let bytes = read(&record_path)?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| DetachError::InvalidDestination {
            path: record_path.clone(),
            reason: format!("invalid previous placement record: {error}"),
        })?;
    let previous = value
        .get("bundle")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| DetachError::InvalidDestination {
            path: record_path.clone(),
            reason: "invalid previous placement record: bundle is missing".into(),
        })?;
    if Path::new(previous)
        .file_name()
        .and_then(|name| name.to_str())
        != Some(previous)
        || previous.is_empty()
    {
        return Err(DetachError::InvalidDestination {
            path: record_path,
            reason: "invalid previous placement record: bundle must be one basename".into(),
        });
    }
    let expected = BTreeSet::from([PathBuf::from(RECORD_NAME), PathBuf::from(previous)]);
    if let Some(entry) = actual.difference(&expected).next() {
        return Err(DetachError::InvalidDestination {
            path: output_dir.to_path_buf(),
            reason: format!("unexpected detached output entry `{}`", entry.display()),
        });
    }
    if let Some(entry) = expected.difference(&actual).next() {
        return Err(DetachError::InvalidDestination {
            path: output_dir.to_path_buf(),
            reason: format!("previous detached output is missing `{}`", entry.display()),
        });
    }
    let Some(previous_root) = fs_guard::optional_existing(output_dir, Path::new(previous))? else {
        unreachable!("the exact entry set contains the previous app")
    };
    if !previous_root.is_dir() {
        return Err(DetachError::InvalidDestination {
            path: previous_root,
            reason: "previous detached app is not a directory".into(),
        });
    }
    let canonical_previous =
        fs::canonicalize(&previous_root).map_err(|source| DetachError::Read {
            path: previous_root.clone(),
            source,
        })?;
    for source_path in [signed_bundle, unsigned_bundle] {
        let source = fs::canonicalize(source_path).map_err(|error| DetachError::Read {
            path: source_path.to_path_buf(),
            source: error,
        })?;
        if paths_overlap(&canonical_previous, &source) {
            return Err(DetachError::InvalidDestination {
                path: output_dir.to_path_buf(),
                reason: format!(
                    "previous detached root `{}` overlaps source `{}`",
                    previous_root.display(),
                    source.display()
                ),
            });
        }
    }
    Ok(Some(previous.to_owned()))
}

fn validate_signed_tree(signed_bundle: &Path, record: &PlacementRecord) -> Result<(), DetachError> {
    let mut actual = BTreeMap::new();
    for (relative, path) in fs_guard::regular_files(signed_bundle)? {
        actual.insert(relative, format!("sha256:{}", sha256_hex(&read(&path)?)));
    }
    let expected_paths: BTreeSet<_> = record.inputs.keys().chain(record.files.keys()).collect();
    for relative in expected_paths {
        let path = PathBuf::from(relative);
        let Some(actual_hash) = actual.remove(&path) else {
            return Err(DetachError::IncompatibleInput {
                path,
                arch: "all".into(),
                reason: "signed target is missing a bound file".into(),
            });
        };
        if record.machos.contains_key(relative) {
            continue;
        }
        let expected_hash = record
            .files
            .get(relative)
            .or_else(|| record.inputs.get(relative))
            .expect("expected signed path came from one of these maps");
        if actual_hash != *expected_hash {
            return Err(DetachError::IncompatibleInput {
                path,
                arch: "all".into(),
                reason: format!(
                    "signed target hashes {actual_hash}, bound input is {expected_hash}"
                ),
            });
        }
    }
    if let Some((path, _)) = actual.into_iter().next() {
        return Err(DetachError::IncompatibleInput {
            path,
            arch: "all".into(),
            reason: "signed target contains an unbound regular file".into(),
        });
    }
    Ok(())
}

fn map_rebuild_error(relative: &str, error: ApplyError) -> DetachError {
    match error {
        ApplyError::UnsignedSlice { path, arch } => DetachError::UnsignedSlice { path, arch },
        ApplyError::Placement { path, arch, reason }
        | ApplyError::DetachedSignature { path, arch, reason } => {
            DetachError::IncompatibleInput { path, arch, reason }
        }
        other => DetachError::IncompatibleInput {
            path: PathBuf::from(relative),
            arch: "all".into(),
            reason: other.to_string(),
        },
    }
}

fn validate_destination(
    signed_bundle: &Path,
    unsigned_bundle: &Path,
    output_dir: &Path,
    bundle_name: &str,
) -> Result<(), DetachError> {
    let signed = fs::canonicalize(signed_bundle).map_err(|source| DetachError::Read {
        path: signed_bundle.to_path_buf(),
        source,
    })?;
    let unsigned = fs::canonicalize(unsigned_bundle).map_err(|source| DetachError::Read {
        path: unsigned_bundle.to_path_buf(),
        source,
    })?;
    let output = canonical_future(output_dir)?;
    let material = output.join(bundle_name);
    for source in [&signed, &unsigned] {
        if output.starts_with(source) || paths_overlap(&material, source) {
            return Err(DetachError::InvalidDestination {
                path: output_dir.to_path_buf(),
                reason: format!("detached output overlaps source `{}`", source.display()),
            });
        }
    }
    Ok(())
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

fn canonical_future(path: &Path) -> Result<PathBuf, DetachError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| DetachError::Read {
                path: PathBuf::from("."),
                source,
            })?
            .join(path)
    };
    let absolute = normalize_absolute(&absolute, path)?;
    let mut existing = absolute.as_path();
    let mut missing = Vec::new();
    loop {
        match fs::symlink_metadata(existing) {
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(DetachError::Read {
                    path: existing.to_path_buf(),
                    source,
                });
            }
        }
        let name = existing
            .file_name()
            .ok_or_else(|| DetachError::InvalidDestination {
                path: path.to_path_buf(),
                reason: "destination has no existing ancestor".into(),
            })?;
        missing.push(name.to_os_string());
        existing = existing
            .parent()
            .ok_or_else(|| DetachError::InvalidDestination {
                path: path.to_path_buf(),
                reason: "destination has no existing ancestor".into(),
            })?;
    }
    let mut resolved = fs::canonicalize(existing).map_err(|source| DetachError::Read {
        path: existing.to_path_buf(),
        source,
    })?;
    for name in missing.into_iter().rev() {
        resolved.push(name);
    }
    Ok(resolved)
}

fn normalize_absolute(path: &Path, original: &Path) -> Result<PathBuf, DetachError> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(DetachError::InvalidDestination {
                        path: original.to_path_buf(),
                        reason: "destination escapes the filesystem root".into(),
                    });
                }
            }
        }
    }
    Ok(normalized)
}

fn clear_previous(output_dir: &Path, previous: Option<&str>) -> Result<(), DetachError> {
    if let Some(previous) = previous {
        let previous_root = fs_guard::existing(output_dir, Path::new(previous))?;
        fs::remove_dir_all(&previous_root).map_err(|source| DetachError::Write {
            path: previous_root,
            source,
        })?;
        let record_path = fs_guard::existing(output_dir, Path::new(RECORD_NAME))?;
        fs::remove_file(&record_path).map_err(|source| DetachError::Write {
            path: record_path,
            source,
        })?;
    }
    Ok(())
}

fn arch_suffix(arch: &str) -> Option<&'static str> {
    match arch {
        "arm64" | "arm64e" => Some("arm64"),
        "x86_64" => Some("x86_64"),
        _ => None,
    }
}

fn read(path: &Path) -> Result<Vec<u8>, DetachError> {
    fs::read(path).map_err(|source| DetachError::Read {
        path: path.to_path_buf(),
        source,
    })
}
