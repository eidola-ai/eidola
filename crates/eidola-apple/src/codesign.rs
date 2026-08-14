use crate::SignatureFacts;

const CSMAGIC_EMBEDDED_SIGNATURE: u32 = 0xfade0cc0;
const CSMAGIC_CODEDIRECTORY: u32 = 0xfade0c02;
const CSMAGIC_EMBEDDED_ENTITLEMENTS: u32 = 0xfade7171;
const CSSLOT_CODEDIRECTORY: u32 = 0;
const CSSLOT_ENTITLEMENTS: u32 = 5;
const CS_RUNTIME: u32 = 0x0001_0000;
const CODEDIRECTORY_TEAM_VERSION: u32 = 0x0002_0200;

pub(crate) fn inspect_superblob(blob: &[u8]) -> Result<SignatureFacts, String> {
    if read_u32(blob, 0)? != CSMAGIC_EMBEDDED_SIGNATURE {
        return Err("outer blob is not an embedded-signature SuperBlob".into());
    }
    let length = usize_from_u32(read_u32(blob, 4)?)?;
    let superblob = blob
        .get(..length)
        .ok_or("SuperBlob length exceeds LC_CODE_SIGNATURE data")?;
    let count = usize_from_u32(read_u32(superblob, 8)?)?;
    let index_end = 12usize
        .checked_add(count.checked_mul(8).ok_or("SuperBlob index overflow")?)
        .ok_or("SuperBlob index overflow")?;
    if index_end > superblob.len() {
        return Err("SuperBlob index exceeds its declared length".into());
    }

    let mut code_directory = None;
    let mut entitlements = None;
    for index in 0..count {
        let pos = 12 + index * 8;
        let slot = read_u32(superblob, pos)?;
        let offset = usize_from_u32(read_u32(superblob, pos + 4)?)?;
        let nested = nested_blob(superblob, offset)?;
        if slot == CSSLOT_CODEDIRECTORY && code_directory.replace(nested).is_some() {
            return Err("SuperBlob has multiple primary CodeDirectories".into());
        }
        if slot == CSSLOT_ENTITLEMENTS && entitlements.replace(nested).is_some() {
            return Err("SuperBlob has multiple entitlement blobs".into());
        }
    }
    let code_directory = code_directory.ok_or("SuperBlob has no primary CodeDirectory")?;
    if read_u32(code_directory, 0)? != CSMAGIC_CODEDIRECTORY {
        return Err("primary slot is not a CodeDirectory".into());
    }
    if code_directory.len() < 44 {
        return Err("CodeDirectory is shorter than its base header".into());
    }
    let version = read_u32(code_directory, 8)?;
    let flags = read_u32(code_directory, 12)?;
    let identifier = c_string_at(code_directory, read_u32(code_directory, 20)?, "identifier")?;
    let team_id = if version >= CODEDIRECTORY_TEAM_VERSION {
        if code_directory.len() < 52 {
            return Err("CodeDirectory version requires a team offset it does not contain".into());
        }
        let offset = read_u32(code_directory, 48)?;
        if offset == 0 {
            None
        } else {
            Some(c_string_at(code_directory, offset, "Team ID")?)
        }
    } else {
        None
    };
    let entitlements_sha256 = entitlements
        .map(entitlements_payload)
        .transpose()?
        .map(crate::sha256_hex);

    Ok(SignatureFacts {
        team_id,
        identifier,
        hardened_runtime: flags & CS_RUNTIME != 0,
        entitlements_sha256,
        has_notarization_ticket: false,
    })
}

fn nested_blob(container: &[u8], offset: usize) -> Result<&[u8], String> {
    let length = usize_from_u32(read_u32(container, offset + 4)?)?;
    let end = offset
        .checked_add(length)
        .ok_or("nested blob range overflow")?;
    container
        .get(offset..end)
        .ok_or_else(|| "nested blob exceeds the SuperBlob".into())
}

fn entitlements_payload(blob: &[u8]) -> Result<&[u8], String> {
    if read_u32(blob, 0)? != CSMAGIC_EMBEDDED_ENTITLEMENTS {
        return Err("entitlements slot has the wrong blob magic".into());
    }
    blob.get(8..)
        .ok_or_else(|| "entitlements blob is shorter than its header".into())
}

fn c_string_at(blob: &[u8], offset: u32, label: &str) -> Result<String, String> {
    let offset = usize_from_u32(offset)?;
    let tail = blob
        .get(offset..)
        .ok_or_else(|| format!("CodeDirectory {label} offset exceeds its length"))?;
    let end = tail
        .iter()
        .position(|byte| *byte == 0)
        .ok_or_else(|| format!("CodeDirectory {label} is not NUL-terminated"))?;
    if end == 0 {
        return Err(format!("CodeDirectory {label} is empty"));
    }
    std::str::from_utf8(&tail[..end])
        .map(str::to_owned)
        .map_err(|_| format!("CodeDirectory {label} is not UTF-8"))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset.checked_add(4).ok_or("integer range overflow")?;
    let encoded: [u8; 4] = bytes
        .get(offset..end)
        .ok_or("integer field exceeds blob")?
        .try_into()
        .map_err(|_| "integer width mismatch")?;
    Ok(u32::from_be_bytes(encoded))
}

fn usize_from_u32(value: u32) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| "blob offset does not fit memory".into())
}
