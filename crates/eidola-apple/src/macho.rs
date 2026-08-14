use crate::format::{CodeSignatureFacts, LinkeditFacts, MachOKind, SliceFacts};
use crate::sha256_hex;

const FAT_MAGIC: u32 = 0xcafebabe;
const FAT_MAGIC_64: u32 = 0xcafebabf;
const MH_MAGIC_64: u32 = 0xfeedfacf;
const LC_SEGMENT_64: u32 = 0x19;
const LC_CODE_SIGNATURE: u32 = 0x1d;

#[derive(Clone, Debug)]
pub(crate) struct ParsedMachO {
    pub kind: MachOKind,
    pub fat_magic: Option<u32>,
    pub slices: Vec<ParsedSlice>,
}

#[derive(Clone, Debug)]
pub(crate) struct ParsedSlice {
    pub facts: SliceFacts,
    pub cputype: i32,
    pub cpusubtype: i32,
}

pub(crate) fn parse(data: &[u8]) -> Result<ParsedMachO, String> {
    let magic = read_be_u32(data, 0)?;
    if matches!(magic, FAT_MAGIC | FAT_MAGIC_64) {
        parse_fat(data, magic)
    } else {
        let slice = parse_slice(data, 0, data.len())?;
        Ok(ParsedMachO {
            kind: MachOKind::Thin,
            fat_magic: None,
            slices: vec![slice],
        })
    }
}

fn parse_fat(data: &[u8], magic: u32) -> Result<ParsedMachO, String> {
    let count = usize::try_from(read_be_u32(data, 4)?).map_err(|_| "fat slice count overflow")?;
    let entry_size = if magic == FAT_MAGIC_64 { 32 } else { 20 };
    let table_end = 8usize
        .checked_add(
            count
                .checked_mul(entry_size)
                .ok_or("fat table size overflow")?,
        )
        .ok_or("fat table size overflow")?;
    require(data, 0, table_end)?;

    let mut slices = Vec::with_capacity(count);
    for index in 0..count {
        let pos = 8 + index * entry_size;
        let cputype = read_be_i32(data, pos)?;
        let cpusubtype = read_be_i32(data, pos + 4)?;
        let (offset, size, align) = if magic == FAT_MAGIC_64 {
            (
                read_be_u64(data, pos + 8)?,
                read_be_u64(data, pos + 16)?,
                read_be_u32(data, pos + 24)?,
            )
        } else {
            (
                u64::from(read_be_u32(data, pos + 8)?),
                u64::from(read_be_u32(data, pos + 12)?),
                read_be_u32(data, pos + 16)?,
            )
        };
        let base = usize::try_from(offset).map_err(|_| "fat slice offset does not fit memory")?;
        let size_usize = usize::try_from(size).map_err(|_| "fat slice size does not fit memory")?;
        let end = base
            .checked_add(size_usize)
            .ok_or("fat slice range overflow")?;
        require(data, base, end)?;
        let mut slice = parse_slice(data, base, size_usize)?;
        if slice.cputype != cputype || slice.cpusubtype != cpusubtype {
            return Err(format!(
                "fat entry {index} CPU values disagree with its Mach header"
            ));
        }
        slice.facts.fat_offset = Some(offset);
        slice.facts.fat_size = Some(size);
        slice.facts.fat_align = Some(align);
        slices.push(slice);
    }

    Ok(ParsedMachO {
        kind: MachOKind::Fat,
        fat_magic: Some(magic),
        slices,
    })
}

fn parse_slice(data: &[u8], base: usize, slice_size: usize) -> Result<ParsedSlice, String> {
    let slice_end = base.checked_add(slice_size).ok_or("slice range overflow")?;
    require(data, base, slice_end)?;
    if read_le_u32(data, base)? != MH_MAGIC_64 {
        return Err(format!("unsupported Mach-O magic at {base:#x}"));
    }
    require(
        data,
        base,
        base.checked_add(32).ok_or("Mach header overflow")?,
    )?;
    let cputype = read_le_i32(data, base + 4)?;
    let cpusubtype = read_le_i32(data, base + 8)?;
    let ncmds = usize::try_from(read_le_u32(data, base + 16)?)
        .map_err(|_| "load-command count overflow")?;
    let sizeofcmds =
        usize::try_from(read_le_u32(data, base + 20)?).map_err(|_| "load-command size overflow")?;
    let flags = read_le_u32(data, base + 24)?;
    let commands_end = base
        .checked_add(32)
        .and_then(|value| value.checked_add(sizeofcmds))
        .ok_or("load-command range overflow")?;
    if commands_end > slice_end {
        return Err("load commands extend beyond the slice".into());
    }

    let mut linkedit = None;
    let mut code_signature = None;
    let mut pos = base + 32;
    for index in 0..ncmds {
        require(
            data,
            pos,
            pos.checked_add(8).ok_or("load command overflow")?,
        )?;
        let cmd = read_le_u32(data, pos)?;
        let cmdsize = usize::try_from(read_le_u32(data, pos + 4)?)
            .map_err(|_| "load-command length overflow")?;
        if cmdsize < 8 {
            return Err(format!("load command {index} has invalid size {cmdsize}"));
        }
        let next = pos
            .checked_add(cmdsize)
            .ok_or("load-command range overflow")?;
        if next > commands_end {
            return Err(format!("load command {index} extends beyond sizeofcmds"));
        }
        if cmd == LC_SEGMENT_64 {
            if cmdsize < 72 {
                return Err("LC_SEGMENT_64 is shorter than its fixed fields".into());
            }
            let name = require(data, pos + 8, pos + 24)?;
            if name.split(|byte| *byte == 0).next() == Some(b"__LINKEDIT") {
                if linkedit.is_some() {
                    return Err("multiple __LINKEDIT segments".into());
                }
                linkedit = Some(LinkeditFacts {
                    vmaddr: read_le_u64(data, pos + 24)?,
                    vmsize: read_le_u64(data, pos + 32)?,
                    fileoff: read_le_u64(data, pos + 40)?,
                    filesize: read_le_u64(data, pos + 48)?,
                    vmsize_field_offset: u64_from_usize(pos + 32)?,
                    fileoff_field_offset: u64_from_usize(pos + 40)?,
                    filesize_field_offset: u64_from_usize(pos + 48)?,
                });
            }
        } else if cmd == LC_CODE_SIGNATURE {
            if cmdsize < 16 {
                return Err("LC_CODE_SIGNATURE is shorter than linkedit_data_command".into());
            }
            if code_signature.is_some() {
                return Err("multiple LC_CODE_SIGNATURE commands".into());
            }
            let dataoff = read_le_u32(data, pos + 8)?;
            let datasize = read_le_u32(data, pos + 12)?;
            let blob_start = base
                .checked_add(usize::try_from(dataoff).map_err(|_| "signature offset overflow")?)
                .ok_or("signature offset overflow")?;
            let blob_end = blob_start
                .checked_add(usize::try_from(datasize).map_err(|_| "signature size overflow")?)
                .ok_or("signature range overflow")?;
            if blob_end > slice_end {
                return Err("LC_CODE_SIGNATURE points beyond its slice".into());
            }
            code_signature = Some(CodeSignatureFacts {
                dataoff,
                datasize,
                lc_offset: u64_from_usize(pos)?,
                superblob_sha256: sha256_hex(&data[blob_start..blob_end]),
            });
        }
        pos = next;
    }
    if pos != commands_end {
        return Err("load-command count does not consume sizeofcmds".into());
    }

    Ok(ParsedSlice {
        facts: SliceFacts {
            arch: cpu_name(cputype, cpusubtype),
            header_offset: u64_from_usize(base)?,
            mh_flags: format!("{flags:#x}"),
            linkedit,
            code_signature,
            fat_offset: None,
            fat_size: None,
            fat_align: None,
        },
        cputype,
        cpusubtype,
    })
}

pub(crate) fn cpu_name(cputype: i32, cpusubtype: i32) -> String {
    match (cputype as u32, cpusubtype as u32 & 0x00ff_ffff) {
        (0x0100_000c, 0) => "arm64".into(),
        (0x0100_000c, 2) => "arm64e".into(),
        (0x0100_0007, 3) => "x86_64".into(),
        _ => format!("{cputype}/{cpusubtype}"),
    }
}

fn require(data: &[u8], start: usize, end: usize) -> Result<&[u8], String> {
    data.get(start..end).ok_or_else(|| {
        format!(
            "range {start:#x}..{end:#x} exceeds file length {:#x}",
            data.len()
        )
    })
}

fn u64_from_usize(value: usize) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| "file offset does not fit u64".into())
}

macro_rules! read_num {
    ($name:ident, $ty:ty, $size:expr, $convert:ident) => {
        fn $name(data: &[u8], offset: usize) -> Result<$ty, String> {
            let end = offset.checked_add($size).ok_or("integer range overflow")?;
            let bytes: [u8; $size] = require(data, offset, end)?
                .try_into()
                .map_err(|_| "integer width mismatch")?;
            Ok(<$ty>::$convert(bytes))
        }
    };
}

read_num!(read_be_u32, u32, 4, from_be_bytes);
read_num!(read_be_i32, i32, 4, from_be_bytes);
read_num!(read_be_u64, u64, 8, from_be_bytes);
read_num!(read_le_u32, u32, 4, from_le_bytes);
read_num!(read_le_i32, i32, 4, from_le_bytes);
read_num!(read_le_u64, u64, 8, from_le_bytes);
