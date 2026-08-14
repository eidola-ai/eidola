use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::format::{MachOEntry, MachOKind, PlacementRecord, SliceFacts};
use crate::fs_guard;
use crate::macho::{self, ParsedMachO, ParsedSlice};
use crate::{ApplyError, sha256_hex};

const RECORD_NAME: &str = "eidola-placement.json";
const MH_FLAGS_OFFSET: usize = 24;
const LC_DATAOFF_OFFSET: usize = 8;

pub(crate) fn apply(unsigned_bundle: &Path, detached: &Path) -> Result<(), ApplyError> {
    let (detached_root, record_path) = locate_record(detached)?;
    fs_guard::root(unsigned_bundle)?;
    let record_relative = record_path
        .strip_prefix(&detached_root)
        .expect("located record is beneath detached root");
    let record_bytes = read_guarded(&detached_root, record_relative)?;
    let record: PlacementRecord =
        serde_json::from_slice(&record_bytes).map_err(|source| ApplyError::RecordJson {
            path: record_path.clone(),
            source,
        })?;
    validate_record(&record, &record_path, unsigned_bundle)?;
    validate_detached_root(&detached_root, &record)?;
    let material_root = fs_guard::existing(&detached_root, Path::new(&record.bundle))?;
    validate_detached_material(&material_root, &record)?;

    let mut macho_writes = Vec::with_capacity(record.machos.len());
    for (relative, entry) in &record.machos {
        let relative_path = safe_relative(relative, &record_path)?;
        let path = fs_guard::existing(unsigned_bundle, relative_path)?;
        let source = read(&path)?;
        let actual_input = sha256_hex(&source);
        if actual_input != entry.input_sha256 {
            return Err(ApplyError::WrongInput {
                path: PathBuf::from(relative),
                expected: entry.input_sha256.clone(),
                actual: actual_input,
            });
        }
        let parsed = macho::parse(&source).map_err(|reason| ApplyError::InvalidMachO {
            path: PathBuf::from(relative),
            reason,
        })?;
        let signatures = read_signatures(&material_root, relative, entry)?;
        let output = rebuild(relative, &source, &parsed, entry, &signatures)?;
        let actual_hash = sha256_hex(&output);
        if output.len() as u64 != entry.output_len || actual_hash != entry.output_sha256 {
            return Err(ApplyError::OutputMismatch {
                path: PathBuf::from(relative),
                expected_len: entry.output_len,
                expected_hash: entry.output_sha256.clone(),
                actual_len: output.len() as u64,
                actual_hash,
            });
        }
        macho_writes.push((path, output));
    }

    validate_unsigned_inputs(unsigned_bundle, &record)?;

    let mut file_writes = Vec::with_capacity(record.files.len());
    for (relative, expected) in &record.files {
        let relative_path = safe_relative(relative, &record_path)?;
        let source_path = fs_guard::existing(&material_root, relative_path)?;
        let data = read(&source_path)?;
        let actual = format!("sha256:{}", sha256_hex(&data));
        if actual != *expected {
            return Err(ApplyError::DetachedFileHash {
                path: PathBuf::from(relative),
                expected: expected.clone(),
                actual,
            });
        }
        file_writes.push((
            fs_guard::for_write(unsigned_bundle, relative_path)?,
            relative_path.to_path_buf(),
            data,
        ));
    }

    validate_plain_file_paths(unsigned_bundle, &record)?;

    commit_mutations(
        unsigned_bundle,
        &record,
        macho_writes,
        file_writes,
        prepare_mutation_targets,
    )
}

fn commit_mutations<F>(
    unsigned_bundle: &Path,
    record: &PlacementRecord,
    macho_writes: Vec<(PathBuf, Vec<u8>)>,
    file_writes: Vec<(PathBuf, PathBuf, Vec<u8>)>,
    prepare: F,
) -> Result<(), ApplyError>
where
    F: FnOnce(&Path, &PlacementRecord) -> Result<(), ApplyError>,
{
    // Content validation above is deliberately complete before preparation.
    prepare(unsigned_bundle, record)?;
    for (path, data) in macho_writes {
        let relative = path
            .strip_prefix(unsigned_bundle)
            .expect("validated Mach-O is beneath bundle root");
        fs_guard::existing(unsigned_bundle, relative)?;
        fs::write(&path, data).map_err(|source| ApplyError::Write { path, source })?;
    }
    remove_unrecorded_signature_files(unsigned_bundle, record)?;
    for (path, relative, data) in file_writes {
        fs_guard::for_write(unsigned_bundle, &relative)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| ApplyError::Write {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        fs::write(&path, data).map_err(|source| ApplyError::Write { path, source })?;
    }
    Ok(())
}

fn prepare_mutation_targets(bundle: &Path, record: &PlacementRecord) -> Result<(), ApplyError> {
    let mut directories = BTreeSet::new();
    let mut creatable_directories = BTreeSet::new();
    let mut writable_files = BTreeSet::new();

    for relative in record.machos.keys() {
        let relative = Path::new(relative);
        writable_files.insert(relative.to_path_buf());
        add_parent_directories(
            relative,
            false,
            &mut directories,
            &mut creatable_directories,
        );
    }
    for relative in record.files.keys() {
        let relative = Path::new(relative);
        if fs_guard::optional_existing(bundle, relative)?.is_some() {
            writable_files.insert(relative.to_path_buf());
        }
        add_parent_directories(relative, true, &mut directories, &mut creatable_directories);
    }

    let seal = Path::new("Contents/_CodeSignature/CodeResources");
    if !record
        .files
        .contains_key("Contents/_CodeSignature/CodeResources")
        && fs_guard::optional_existing(bundle, seal)?.is_some()
    {
        add_parent_directories(seal, false, &mut directories, &mut creatable_directories);
    }

    let ticket = Path::new("Contents/CodeResources");
    if !record.files.contains_key("Contents/CodeResources")
        && fs_guard::optional_existing(bundle, ticket)?.is_some()
    {
        add_parent_directories(ticket, false, &mut directories, &mut creatable_directories);
    }

    let mut directories: Vec<_> = directories.into_iter().collect();
    directories.sort_by(|left, right| {
        left.components()
            .count()
            .cmp(&right.components().count())
            .then_with(|| left.cmp(right))
    });
    for relative in directories {
        match fs_guard::optional_existing(bundle, &relative)? {
            Some(path) => {
                let metadata = fs::metadata(&path).map_err(|source| ApplyError::Read {
                    path: path.clone(),
                    source,
                })?;
                if !metadata.is_dir() {
                    return Err(ApplyError::PlainFileTarget {
                        path: relative,
                        reason: "mutation parent is not a directory".into(),
                    });
                }
                make_writable(&path, true)?;
            }
            None if creatable_directories.contains(&relative) => {
                let path = fs_guard::for_write(bundle, &relative)?;
                fs::create_dir(&path).map_err(|source| ApplyError::Write {
                    path: path.clone(),
                    source,
                })?;
                make_writable(&path, true)?;
            }
            None => {
                return Err(ApplyError::PlainFileTarget {
                    path: relative,
                    reason: "mutation parent is missing".into(),
                });
            }
        }
    }

    for relative in writable_files {
        let path = fs_guard::existing(bundle, &relative)?;
        make_writable(&path, false)?;
    }
    Ok(())
}

fn add_parent_directories(
    relative: &Path,
    creatable: bool,
    directories: &mut BTreeSet<PathBuf>,
    creatable_directories: &mut BTreeSet<PathBuf>,
) {
    let mut current = PathBuf::new();
    let Some(parent) = relative.parent() else {
        return;
    };
    for component in parent.components() {
        current.push(component.as_os_str());
        directories.insert(current.clone());
        if creatable {
            creatable_directories.insert(current.clone());
        }
    }
}

fn locate_record(detached: &Path) -> Result<(PathBuf, PathBuf), ApplyError> {
    fs_guard::root(detached)?;
    if let Some(direct) = fs_guard::optional_existing(detached, Path::new(RECORD_NAME))?
        && direct.is_file()
    {
        return Ok((detached.to_path_buf(), direct));
    }
    let parent = detached.parent().unwrap_or(detached);
    fs_guard::root(parent)?;
    if let Some(name) = detached.file_name() {
        fs_guard::existing(parent, Path::new(name))?;
    }
    if let Some(beside) = fs_guard::optional_existing(parent, Path::new(RECORD_NAME))?
        && beside.is_file()
    {
        return Ok((parent.to_path_buf(), beside));
    }
    Err(ApplyError::Read {
        path: detached.join(RECORD_NAME),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "placement record not found"),
    })
}

fn validate_detached_root(
    detached_root: &Path,
    record: &PlacementRecord,
) -> Result<(), ApplyError> {
    let entries = fs::read_dir(detached_root).map_err(|source| ApplyError::Read {
        path: detached_root.to_path_buf(),
        source,
    })?;
    let mut entries =
        entries
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| ApplyError::Read {
                path: detached_root.to_path_buf(),
                source,
            })?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let relative = PathBuf::from(entry.file_name());
        let file_type = entry.file_type().map_err(|source| ApplyError::Read {
            path: entry.path(),
            source,
        })?;
        if relative == Path::new(RECORD_NAME) {
            if file_type.is_file() {
                continue;
            }
            return Err(ApplyError::DetachedInputInvalid {
                path: relative,
                reason: "placement record is not a regular file".into(),
            });
        }
        if relative == Path::new(&record.bundle) {
            if file_type.is_dir() {
                continue;
            }
            return Err(ApplyError::DetachedInputInvalid {
                path: relative,
                reason: "recorded app tree is not a directory".into(),
            });
        }
        if file_type.is_file() {
            return Err(ApplyError::DetachedInputUnexpected { path: relative });
        }
        let reason = if file_type.is_symlink() {
            "symbolic link"
        } else if file_type.is_dir() {
            "unexpected directory"
        } else {
            "not a regular file or directory"
        };
        return Err(ApplyError::DetachedInputInvalid {
            path: relative,
            reason: reason.into(),
        });
    }
    Ok(())
}

fn validate_record(
    record: &PlacementRecord,
    record_path: &Path,
    unsigned_bundle: &Path,
) -> Result<(), ApplyError> {
    if record.schema_version != 1 {
        return invalid_record(
            record_path,
            format!("unsupported schema version {}", record.schema_version),
        );
    }
    if record.bundle.is_empty()
        || !record.bundle.ends_with(".app")
        || Path::new(&record.bundle)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(record.bundle.as_str())
    {
        return invalid_record(record_path, "bundle must be one basename");
    }
    if unsigned_bundle.file_name().and_then(|name| name.to_str()) != Some(record.bundle.as_str()) {
        return invalid_record(
            record_path,
            format!(
                "record names bundle `{}`, input is `{}`",
                record.bundle,
                unsigned_bundle.display()
            ),
        );
    }
    if record.machos.is_empty() {
        return invalid_record(record_path, "record contains no Mach-O files");
    }
    for relative in record
        .inputs
        .keys()
        .chain(record.machos.keys())
        .chain(record.files.keys())
    {
        safe_relative(relative, record_path)?;
    }
    for (relative, entry) in &record.machos {
        let expected = format!("sha256:{}", entry.input_sha256);
        if record.inputs.get(relative) != Some(&expected) {
            return invalid_record(
                record_path,
                format!("Mach-O input `{relative}` is not identically bound in inputs"),
            );
        }
    }
    for relative in record.machos.keys() {
        let path = Path::new(relative);
        if path.parent() != Some(Path::new("Contents/MacOS")) || path.file_name().is_none() {
            return invalid_record(
                record_path,
                format!("Mach-O path `{relative}` is not directly under Contents/MacOS"),
            );
        }
    }
    for relative in record.files.keys() {
        if !matches!(
            relative.as_str(),
            "Contents/_CodeSignature/CodeResources" | "Contents/CodeResources"
        ) {
            return invalid_record(
                record_path,
                format!("detached plain-file path `{relative}` is not part of the format"),
            );
        }
    }
    Ok(())
}

fn validate_unsigned_inputs(
    unsigned_bundle: &Path,
    record: &PlacementRecord,
) -> Result<(), ApplyError> {
    let mut actual = BTreeMap::new();
    for (relative, path) in fs_guard::regular_files(unsigned_bundle)? {
        let data = read(&path)?;
        actual.insert(relative, format!("sha256:{}", sha256_hex(&data)));
    }
    for (relative, expected) in &record.inputs {
        let path = PathBuf::from(relative);
        let Some(value) = actual.remove(&path) else {
            return Err(ApplyError::UnsignedInputMissing { path });
        };
        if value != *expected {
            return Err(ApplyError::UnsignedInputHash {
                path,
                expected: expected.clone(),
                actual: value,
            });
        }
    }
    if let Some((path, _)) = actual.into_iter().next() {
        return Err(ApplyError::UnsignedInputUnexpected { path });
    }
    Ok(())
}

fn safe_relative<'a>(relative: &'a str, record_path: &Path) -> Result<&'a Path, ApplyError> {
    let path = Path::new(relative);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return invalid_record(
            record_path,
            format!("path `{relative}` is not a normalized relative path"),
        );
    }
    Ok(path)
}

fn read_signatures(
    material_root: &Path,
    relative: &str,
    entry: &MachOEntry,
) -> Result<BTreeMap<String, Vec<u8>>, ApplyError> {
    let name = Path::new(relative)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ApplyError::Placement {
            path: PathBuf::from(relative),
            arch: "unknown".into(),
            reason: "Mach-O path has no UTF-8 basename".into(),
        })?;
    let mut signatures = BTreeMap::new();
    for target in &entry.slices {
        let suffix = arch_suffix(&target.arch).ok_or_else(|| ApplyError::Placement {
            path: PathBuf::from(relative),
            arch: target.arch.clone(),
            reason: "unsupported detached-signature architecture".into(),
        })?;
        let signature_relative =
            PathBuf::from("Contents/MacOS").join(format!("{name}.{suffix}sign"));
        let path = fs_guard::existing(material_root, &signature_relative)?;
        let blob = read(&path)?;
        if signatures.insert(target.arch.clone(), blob).is_some() {
            return Err(ApplyError::Placement {
                path: PathBuf::from(relative),
                arch: target.arch.clone(),
                reason: "duplicate architecture in placement record".into(),
            });
        }
    }
    Ok(signatures)
}

pub(crate) fn rebuild(
    relative: &str,
    source: &[u8],
    parsed: &ParsedMachO,
    record: &MachOEntry,
    signatures: &BTreeMap<String, Vec<u8>>,
) -> Result<Vec<u8>, ApplyError> {
    if parsed.kind != record.kind {
        return Err(ApplyError::Placement {
            path: PathBuf::from(relative),
            arch: "all".into(),
            reason: format!(
                "input is {:?}, record targets {:?}",
                parsed.kind, record.kind
            ),
        });
    }
    let by_arch: HashMap<&str, &ParsedSlice> = parsed
        .slices
        .iter()
        .map(|slice| (slice.facts.arch.as_str(), slice))
        .collect();
    if by_arch.len() != parsed.slices.len() || record.slices.len() != parsed.slices.len() {
        return Err(ApplyError::Placement {
            path: PathBuf::from(relative),
            arch: "all".into(),
            reason: "input and record slices are not a one-to-one architecture set".into(),
        });
    }
    let mut planned = Vec::with_capacity(record.slices.len());
    let mut ranges = Vec::new();

    for target in &record.slices {
        let source_slice =
            by_arch
                .get(target.arch.as_str())
                .copied()
                .ok_or_else(|| ApplyError::Placement {
                    path: PathBuf::from(relative),
                    arch: target.arch.clone(),
                    reason: "input has no matching slice".into(),
                })?;
        let target_cs = target
            .code_signature
            .as_ref()
            .ok_or_else(|| ApplyError::Placement {
                path: PathBuf::from(relative),
                arch: target.arch.clone(),
                reason: "record has no LC_CODE_SIGNATURE facts".into(),
            })?;
        let source_cs = source_slice.facts.code_signature.as_ref().ok_or_else(|| {
            ApplyError::UnsignedSlice {
                path: PathBuf::from(relative),
                arch: target.arch.clone(),
            }
        })?;
        let blob = signatures
            .get(&target.arch)
            .ok_or_else(|| ApplyError::DetachedSignature {
                path: PathBuf::from(relative),
                arch: target.arch.clone(),
                reason: "signature file is missing".into(),
            })?;
        if blob.len() as u64 != u64::from(target_cs.datasize) {
            return Err(ApplyError::DetachedSignature {
                path: PathBuf::from(relative),
                arch: target.arch.clone(),
                reason: format!("{} bytes, record says {}", blob.len(), target_cs.datasize),
            });
        }
        let blob_hash = sha256_hex(blob);
        if blob_hash != target_cs.superblob_sha256 {
            return Err(ApplyError::DetachedSignature {
                path: PathBuf::from(relative),
                arch: target.arch.clone(),
                reason: format!(
                    "hash {blob_hash}, record says {}",
                    target_cs.superblob_sha256
                ),
            });
        }

        let target_base = target.header_offset;
        let source_base = source_slice.facts.header_offset;
        if source_cs.dataoff != target_cs.dataoff
            || relative_offset(source_cs.lc_offset, source_base)
                != relative_offset(target_cs.lc_offset, target_base)
        {
            return Err(ApplyError::Placement {
                path: PathBuf::from(relative),
                arch: target.arch.clone(),
                reason: format!(
                    "signing moved LC_CODE_SIGNATURE or its data offset ({:#x}/{:#x} to {:#x}/{:#x})",
                    relative_offset(source_cs.lc_offset, source_base).unwrap_or(u64::MAX),
                    source_cs.dataoff,
                    relative_offset(target_cs.lc_offset, target_base).unwrap_or(u64::MAX),
                    target_cs.dataoff
                ),
            });
        }
        let head_start = usize_from_u64(source_base, relative, target)?;
        let head_len = usize::try_from(target_cs.dataoff).map_err(|_| {
            placement(
                relative,
                target,
                "signature data offset does not fit memory",
            )
        })?;
        let head_end = head_start
            .checked_add(head_len)
            .ok_or_else(|| placement(relative, target, "input head range overflow"))?;
        let mut head = source
            .get(head_start..head_end)
            .ok_or_else(|| {
                placement(
                    relative,
                    target,
                    "input ends before recorded signature offset",
                )
            })?
            .to_vec();

        let linkedit = target
            .linkedit
            .as_ref()
            .ok_or_else(|| placement(relative, target, "record has no __LINKEDIT segment"))?;
        write_u64_field(
            &mut head,
            linkedit.vmsize_field_offset,
            target_base,
            linkedit.vmsize,
        )
        .map_err(|reason| placement(relative, target, &reason))?;
        write_u64_field(
            &mut head,
            linkedit.fileoff_field_offset,
            target_base,
            linkedit.fileoff,
        )
        .map_err(|reason| placement(relative, target, &reason))?;
        write_u64_field(
            &mut head,
            linkedit.filesize_field_offset,
            target_base,
            linkedit.filesize,
        )
        .map_err(|reason| placement(relative, target, &reason))?;
        let lc_relative = relative_offset(target_cs.lc_offset, target_base)
            .ok_or_else(|| placement(relative, target, "LC_CODE_SIGNATURE precedes slice"))?;
        let lc_dataoff = usize_from_u64(lc_relative, relative, target)?
            .checked_add(LC_DATAOFF_OFFSET)
            .ok_or_else(|| placement(relative, target, "LC_CODE_SIGNATURE field overflow"))?;
        write_le_u32(&mut head, lc_dataoff, target_cs.dataoff)
            .map_err(|reason| placement(relative, target, &reason))?;
        write_le_u32(&mut head, lc_dataoff + 4, target_cs.datasize)
            .map_err(|reason| placement(relative, target, &reason))?;
        let flags = parse_hex_u32(&target.mh_flags)
            .map_err(|reason| placement(relative, target, &reason))?;
        write_le_u32(&mut head, MH_FLAGS_OFFSET, flags)
            .map_err(|reason| placement(relative, target, &reason))?;

        let target_start = usize_from_u64(target_base, relative, target)?;
        let body_len = head
            .len()
            .checked_add(blob.len())
            .ok_or_else(|| placement(relative, target, "slice length overflow"))?;
        let target_end = target_start
            .checked_add(body_len)
            .ok_or_else(|| placement(relative, target, "slice range overflow"))?;
        if ranges
            .iter()
            .any(|(start, end)| target_start < *end && *start < target_end)
        {
            return Err(placement(
                relative,
                target,
                "slice overlaps another recorded slice",
            ));
        }
        let recorded_size = match record.kind {
            MachOKind::Fat => target
                .fat_size
                .ok_or_else(|| placement(relative, target, "fat slice has no recorded size"))?,
            MachOKind::Thin => body_len as u64,
        };
        if recorded_size != body_len as u64 {
            return Err(placement(
                relative,
                target,
                &format!("slice is {body_len} bytes, record says {recorded_size}"),
            ));
        }
        ranges.push((target_start, target_end));
        planned.push((target_start, target_end, head, blob));
    }

    let derived_len = derive_output_len(relative, source.len(), parsed, record, &ranges)?;
    if record.output_len != derived_len as u64 {
        return Err(ApplyError::Placement {
            path: PathBuf::from(relative),
            arch: "all".into(),
            reason: format!(
                "recorded output length {} does not equal reconstructed end {derived_len}",
                record.output_len
            ),
        });
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(derived_len)
        .map_err(|_| ApplyError::Placement {
            path: PathBuf::from(relative),
            arch: "all".into(),
            reason: format!("cannot allocate reconstructed output length {derived_len}"),
        })?;
    output.resize(derived_len, 0u8);
    for (target_start, target_end, head, blob) in planned {
        output[target_start..target_start + head.len()].copy_from_slice(&head);
        output[target_start + head.len()..target_end].copy_from_slice(blob);
    }
    if record.kind == MachOKind::Fat {
        rebuild_fat_header(relative, &mut output, parsed, &record.slices)?;
    }
    Ok(output)
}

fn validate_detached_material(
    material_root: &Path,
    record: &PlacementRecord,
) -> Result<(), ApplyError> {
    let mut expected = BTreeMap::new();
    for (relative, entry) in &record.machos {
        let name = Path::new(relative)
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| ApplyError::Placement {
                path: PathBuf::from(relative),
                arch: "unknown".into(),
                reason: "Mach-O path has no UTF-8 basename".into(),
            })?;
        for slice in &entry.slices {
            let suffix = arch_suffix(&slice.arch).ok_or_else(|| ApplyError::Placement {
                path: PathBuf::from(relative),
                arch: slice.arch.clone(),
                reason: "unsupported detached-signature architecture".into(),
            })?;
            expected.insert(
                PathBuf::from("Contents/MacOS").join(format!("{name}.{suffix}sign")),
                (),
            );
        }
    }
    expected.extend(record.files.keys().map(|path| (PathBuf::from(path), ())));

    let actual = fs_guard::regular_files(material_root).map_err(|error| {
        let (path, reason) = match error {
            fs_guard::GuardError::Symlink(path) => (path, "symbolic link"),
            fs_guard::GuardError::NotRegular(path) => (path, "not a regular file"),
            other => return ApplyError::UnsafePath(other),
        };
        let relative = path
            .strip_prefix(material_root)
            .unwrap_or(&path)
            .to_path_buf();
        ApplyError::DetachedInputInvalid {
            path: relative,
            reason: reason.into(),
        }
    })?;
    for (relative, _) in actual {
        if !expected.contains_key(&relative) {
            return Err(ApplyError::DetachedInputUnexpected { path: relative });
        }
    }
    Ok(())
}

fn rebuild_fat_header(
    relative: &str,
    output: &mut [u8],
    parsed: &ParsedMachO,
    targets: &[SliceFacts],
) -> Result<(), ApplyError> {
    let magic = parsed.fat_magic.ok_or_else(|| ApplyError::Placement {
        path: PathBuf::from(relative),
        arch: "all".into(),
        reason: "fat input has no fat magic".into(),
    })?;
    let is_64 = magic == 0xcafebabf;
    let entry_size = if is_64 { 32 } else { 20 };
    write_be_u32(output, 0, magic).map_err(|reason| ApplyError::Placement {
        path: PathBuf::from(relative),
        arch: "all".into(),
        reason,
    })?;
    write_be_u32(output, 4, targets.len() as u32).map_err(|reason| ApplyError::Placement {
        path: PathBuf::from(relative),
        arch: "all".into(),
        reason,
    })?;
    for (index, target) in targets.iter().enumerate() {
        let source = parsed
            .slices
            .iter()
            .find(|slice| slice.facts.arch == target.arch)
            .ok_or_else(|| placement(relative, target, "source CPU values are missing"))?;
        let offset = target
            .fat_offset
            .ok_or_else(|| placement(relative, target, "fat offset is missing"))?;
        let size = target
            .fat_size
            .ok_or_else(|| placement(relative, target, "fat size is missing"))?;
        let align = target
            .fat_align
            .ok_or_else(|| placement(relative, target, "fat alignment is missing"))?;
        if offset != target.header_offset {
            return Err(placement(
                relative,
                target,
                "fat offset and Mach header offset disagree",
            ));
        }
        let pos = 8 + index * entry_size;
        write_be_i32(output, pos, source.cputype)
            .and_then(|()| write_be_i32(output, pos + 4, source.cpusubtype))
            .map_err(|reason| placement(relative, target, &reason))?;
        if is_64 {
            write_be_u64(output, pos + 8, offset)
                .and_then(|()| write_be_u64(output, pos + 16, size))
                .and_then(|()| write_be_u32(output, pos + 24, align))
                .and_then(|()| write_be_u32(output, pos + 28, 0))
                .map_err(|reason| placement(relative, target, &reason))?;
        } else {
            let offset = u32::try_from(offset)
                .map_err(|_| placement(relative, target, "fat offset exceeds 32 bits"))?;
            let size = u32::try_from(size)
                .map_err(|_| placement(relative, target, "fat size exceeds 32 bits"))?;
            write_be_u32(output, pos + 8, offset)
                .and_then(|()| write_be_u32(output, pos + 12, size))
                .and_then(|()| write_be_u32(output, pos + 16, align))
                .map_err(|reason| placement(relative, target, &reason))?;
        }
    }
    Ok(())
}

fn derive_output_len(
    relative: &str,
    source_len: usize,
    parsed: &ParsedMachO,
    record: &MachOEntry,
    ranges: &[(usize, usize)],
) -> Result<usize, ApplyError> {
    if record.kind == MachOKind::Thin {
        if ranges.len() != 1 || ranges[0].0 != 0 {
            return Err(ApplyError::Placement {
                path: PathBuf::from(relative),
                arch: "all".into(),
                reason: "thin output is not exactly one slice at file offset zero".into(),
            });
        }
        return Ok(ranges[0].1);
    }

    let magic = parsed.fat_magic.ok_or_else(|| ApplyError::Placement {
        path: PathBuf::from(relative),
        arch: "all".into(),
        reason: "fat input has no fat magic".into(),
    })?;
    let entry_size = if magic == 0xcafebabf { 32 } else { 20 };
    let table_end = 8usize
        .checked_add(record.slices.len().checked_mul(entry_size).ok_or_else(|| {
            ApplyError::Placement {
                path: PathBuf::from(relative),
                arch: "all".into(),
                reason: "fat table size overflow".into(),
            }
        })?)
        .ok_or_else(|| ApplyError::Placement {
            path: PathBuf::from(relative),
            arch: "all".into(),
            reason: "fat table size overflow".into(),
        })?;

    let mut source_start = table_end;
    let mut input_max_align = 0;
    for source in &parsed.slices {
        let facts = &source.facts;
        let offset = facts
            .fat_offset
            .ok_or_else(|| placement(relative, facts, "input fat offset is missing"))?;
        if offset != facts.header_offset {
            return Err(placement(
                relative,
                facts,
                "input fat offset and Mach header offset disagree",
            ));
        }
        let align = facts
            .fat_align
            .ok_or_else(|| placement(relative, facts, "input fat alignment is missing"))?;
        let alignment = 1usize.checked_shl(align).ok_or_else(|| {
            placement(relative, facts, "input fat alignment exponent is too large")
        })?;
        input_max_align = input_max_align.max(align);
        source_start = source_start
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
            .ok_or_else(|| placement(relative, facts, "input fat slice alignment overflow"))?;
        let actual_start = usize_from_u64(offset, relative, facts)?;
        if actual_start != source_start {
            return Err(placement(
                relative,
                facts,
                "input fat table is not canonically packed",
            ));
        }
        let size = facts
            .fat_size
            .ok_or_else(|| placement(relative, facts, "input fat size is missing"))?;
        source_start = source_start
            .checked_add(usize_from_u64(size, relative, facts)?)
            .ok_or_else(|| placement(relative, facts, "input fat slice range overflow"))?;
    }
    if source_start != source_len {
        return Err(ApplyError::Placement {
            path: PathBuf::from(relative),
            arch: "all".into(),
            reason: format!("input fat slices end at {source_start}, file length is {source_len}"),
        });
    }

    let mut expected_start = table_end;

    for ((target, &(start, end)), source) in record.slices.iter().zip(ranges).zip(&parsed.slices) {
        if target.arch != source.facts.arch {
            return Err(placement(
                relative,
                target,
                "slice order disagrees with the input fat table",
            ));
        }
        let offset = target
            .fat_offset
            .ok_or_else(|| placement(relative, target, "fat offset is missing"))?;
        if offset != target.header_offset {
            return Err(placement(
                relative,
                target,
                "fat offset and Mach header offset disagree",
            ));
        }
        let align = target
            .fat_align
            .ok_or_else(|| placement(relative, target, "fat alignment is missing"))?;
        if align > input_max_align {
            return Err(placement(
                relative,
                target,
                &format!("fat alignment exponent {align} exceeds input maximum {input_max_align}"),
            ));
        }
        let alignment = 1usize
            .checked_shl(align)
            .ok_or_else(|| placement(relative, target, "fat alignment exponent is too large"))?;
        expected_start = expected_start
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
            .ok_or_else(|| placement(relative, target, "fat slice alignment overflow"))?;
        if start != expected_start {
            return Err(placement(
                relative,
                target,
                &format!(
                    "fat slice starts at {start}, canonical packing requires {expected_start}"
                ),
            ));
        }
        expected_start = end;
    }
    Ok(expected_start)
}

fn remove_unrecorded_signature_files(
    bundle: &Path,
    record: &PlacementRecord,
) -> Result<(), ApplyError> {
    let seal = "Contents/_CodeSignature/CodeResources";
    if !record.files.contains_key(seal) {
        let relative = Path::new(seal);
        if let Some(path) = fs_guard::optional_existing(bundle, relative)? {
            fs::remove_file(&path).map_err(|source| ApplyError::Write { path, source })?;
        }
    }
    let ticket = "Contents/CodeResources";
    if !record.files.contains_key(ticket)
        && let Some(path) = fs_guard::optional_existing(bundle, Path::new(ticket))?
    {
        fs::remove_file(&path).map_err(|source| ApplyError::Write { path, source })?;
    }
    Ok(())
}

fn validate_plain_file_paths(bundle: &Path, record: &PlacementRecord) -> Result<(), ApplyError> {
    for relative in record.files.keys() {
        if let Some(path) = fs_guard::optional_existing(bundle, Path::new(relative))?
            && !path.is_file()
        {
            return Err(ApplyError::PlainFileTarget {
                path: PathBuf::from(relative),
                reason: "recorded file destination exists but is not a regular file".into(),
            });
        }
    }

    let seal = "Contents/_CodeSignature/CodeResources";
    if !record.files.contains_key(seal) {
        if let Some(path) =
            fs_guard::optional_existing(bundle, Path::new("Contents/_CodeSignature"))?
            && !path.is_dir()
        {
            return Err(ApplyError::PlainFileTarget {
                path: PathBuf::from("Contents/_CodeSignature"),
                reason: "unrecorded seal parent is not a directory".into(),
            });
        }
        if let Some(path) = fs_guard::optional_existing(bundle, Path::new(seal))?
            && !path.is_file()
        {
            return Err(ApplyError::PlainFileTarget {
                path: PathBuf::from(seal),
                reason: "unrecorded seal removal target is not a regular file".into(),
            });
        }
    }

    let ticket = "Contents/CodeResources";
    if !record.files.contains_key(ticket)
        && let Some(path) = fs_guard::optional_existing(bundle, Path::new(ticket))?
        && !path.is_file()
    {
        return Err(ApplyError::PlainFileTarget {
            path: PathBuf::from(ticket),
            reason: "unrecorded ticket removal target is not a regular file".into(),
        });
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

fn relative_offset(absolute: u64, base: u64) -> Option<u64> {
    absolute.checked_sub(base)
}

fn write_u64_field(bytes: &mut [u8], absolute: u64, base: u64, value: u64) -> Result<(), String> {
    let relative = absolute
        .checked_sub(base)
        .ok_or("recorded field precedes its slice")?;
    let offset = usize::try_from(relative).map_err(|_| "field offset does not fit memory")?;
    write_le_u64(bytes, offset, value)
}

fn parse_hex_u32(value: &str) -> Result<u32, String> {
    let digits = value
        .strip_prefix("0x")
        .ok_or("Mach header flags must start with 0x")?;
    u32::from_str_radix(digits, 16).map_err(|_| "invalid Mach header flags".into())
}

fn usize_from_u64(value: u64, relative: &str, target: &SliceFacts) -> Result<usize, ApplyError> {
    usize::try_from(value)
        .map_err(|_| placement(relative, target, "file offset does not fit memory"))
}

fn placement(relative: &str, target: &SliceFacts, reason: &str) -> ApplyError {
    ApplyError::Placement {
        path: PathBuf::from(relative),
        arch: target.arch.clone(),
        reason: reason.into(),
    }
}

fn invalid_record<T>(path: &Path, reason: impl Into<String>) -> Result<T, ApplyError> {
    Err(ApplyError::InvalidRecord {
        path: path.to_path_buf(),
        reason: reason.into(),
    })
}

fn read(path: &Path) -> Result<Vec<u8>, ApplyError> {
    fs::read(path).map_err(|source| ApplyError::Read {
        path: path.to_path_buf(),
        source,
    })
}

fn read_guarded(root: &Path, relative: &Path) -> Result<Vec<u8>, ApplyError> {
    let path = fs_guard::existing(root, relative)?;
    read(&path)
}

#[cfg(unix)]
fn make_writable(path: &Path, directory: bool) -> Result<(), ApplyError> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = fs::metadata(path).map_err(|source| ApplyError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut permissions = metadata.permissions();
    let required = if directory { 0o300 } else { 0o200 };
    permissions.set_mode(permissions.mode() | required);
    fs::set_permissions(path, permissions).map_err(|source| ApplyError::Write {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(not(unix))]
fn make_writable(path: &Path, _directory: bool) -> Result<(), ApplyError> {
    let metadata = fs::metadata(path).map_err(|source| ApplyError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut permissions = metadata.permissions();
    permissions.set_readonly(false);
    fs::set_permissions(path, permissions).map_err(|source| ApplyError::Write {
        path: path.to_path_buf(),
        source,
    })
}

macro_rules! write_num {
    ($name:ident, $ty:ty, $convert:ident) => {
        fn $name(bytes: &mut [u8], offset: usize, value: $ty) -> Result<(), String> {
            let encoded = value.$convert();
            let end = offset
                .checked_add(encoded.len())
                .ok_or("integer field range overflow")?;
            let destination = bytes
                .get_mut(offset..end)
                .ok_or("integer field lies outside reconstructed prefix")?;
            destination.copy_from_slice(&encoded);
            Ok(())
        }
    };
}

write_num!(write_le_u32, u32, to_le_bytes);
write_num!(write_le_u64, u64, to_le_bytes);
write_num!(write_be_u32, u32, to_be_bytes);
write_num!(write_be_i32, i32, to_be_bytes);
write_num!(write_be_u64, u64, to_be_bytes);

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::io;

    use super::*;

    #[test]
    fn preparation_failure_precedes_every_content_write() {
        let temporary = tempfile::tempdir().unwrap();
        let bundle = temporary.path().join("Fixture.app");
        let executable = bundle.join("Contents/MacOS/Fixture");
        fs::create_dir_all(executable.parent().unwrap()).unwrap();
        fs::write(&executable, b"before").unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&executable, permissions).unwrap();
        let record = PlacementRecord {
            schema_version: 1,
            bundle: "Fixture.app".into(),
            inputs: BTreeMap::new(),
            machos: BTreeMap::new(),
            files: BTreeMap::new(),
        };
        let preparation_target = executable.clone();
        let failure_path = bundle.join("Contents");

        let error = commit_mutations(
            &bundle,
            &record,
            vec![(executable.clone(), b"after".to_vec())],
            Vec::new(),
            |_, _| {
                make_writable(&preparation_target, false)?;
                Err(ApplyError::Write {
                    path: failure_path.clone(),
                    source: io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "injected preparation failure",
                    ),
                })
            },
        )
        .unwrap_err();

        assert!(matches!(error, ApplyError::Write { ref path, .. } if path == &failure_path));
        assert_eq!(fs::read(executable).unwrap(), b"before");
    }
}
