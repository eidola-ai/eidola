use std::fs;
use std::path::{Path, PathBuf};

use crate::codesign;
use crate::fs_guard;
use crate::macho;
use crate::{InspectError, SignatureFacts};

pub(crate) fn inspect(bundle: &Path) -> Result<SignatureFacts, InspectError> {
    fs_guard::root(bundle)?;
    let plist_path = fs_guard::existing(bundle, Path::new("Contents/Info.plist"))?;
    let plist = plist::Value::from_file(&plist_path).map_err(|error| InspectError::Plist {
        path: plist_path.clone(),
        reason: error.to_string(),
    })?;
    let executable = plist
        .as_dictionary()
        .and_then(|dictionary| dictionary.get("CFBundleExecutable"))
        .and_then(plist::Value::as_string)
        .ok_or_else(|| InspectError::Plist {
            path: plist_path,
            reason: "CFBundleExecutable is missing or is not a string".into(),
        })?;
    if executable.is_empty()
        || Path::new(executable)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(executable)
    {
        return Err(InspectError::Plist {
            path: bundle.join("Contents/Info.plist"),
            reason: "CFBundleExecutable must be one basename".into(),
        });
    }
    let relative = PathBuf::from("Contents/MacOS").join(executable);
    let path = fs_guard::existing(bundle, &relative)?;
    let data = fs::read(&path).map_err(|source| InspectError::Read {
        path: relative.clone(),
        source,
    })?;
    let parsed = macho::parse(&data).map_err(|reason| InspectError::InvalidMachO {
        path: relative.clone(),
        reason,
    })?;
    let mut common = None;
    for slice in parsed.slices {
        let signature =
            slice
                .facts
                .code_signature
                .as_ref()
                .ok_or_else(|| InspectError::UnsignedSlice {
                    path: relative.clone(),
                    arch: slice.facts.arch.clone(),
                })?;
        let base =
            usize::try_from(slice.facts.header_offset).map_err(|_| InspectError::InvalidMachO {
                path: relative.clone(),
                reason: "slice offset does not fit memory".into(),
            })?;
        let start = base
            .checked_add(signature.dataoff as usize)
            .ok_or_else(|| InspectError::InvalidMachO {
                path: relative.clone(),
                reason: "signature offset overflow".into(),
            })?;
        let end = start
            .checked_add(signature.datasize as usize)
            .ok_or_else(|| InspectError::InvalidMachO {
                path: relative.clone(),
                reason: "signature range overflow".into(),
            })?;
        let superblob = data
            .get(start..end)
            .ok_or_else(|| InspectError::InvalidMachO {
                path: relative.clone(),
                reason: "signature extends beyond file".into(),
            })?;
        let facts = codesign::inspect_superblob(superblob).map_err(|reason| {
            InspectError::InvalidSignature {
                path: relative.clone(),
                arch: slice.facts.arch.clone(),
                reason,
            }
        })?;
        if let Some(expected) = &common {
            if expected != &facts {
                return Err(InspectError::SliceMismatch {
                    path: relative.clone(),
                    reason: format!(
                        "{} reports {:?}, previous slice reports {:?}",
                        slice.facts.arch, facts, expected
                    ),
                });
            }
        } else {
            common = Some(facts);
        }
    }
    let mut facts = common.ok_or_else(|| InspectError::InvalidMachO {
        path: relative,
        reason: "Mach-O contains no slices".into(),
    })?;
    facts.has_notarization_ticket =
        fs_guard::optional_existing(bundle, Path::new("Contents/CodeResources"))?
            .is_some_and(|path| path.is_file());
    Ok(facts)
}
