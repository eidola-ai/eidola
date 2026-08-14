use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementRecord {
    pub schema_version: u32,
    pub bundle: String,
    pub inputs: BTreeMap<String, String>,
    pub machos: BTreeMap<String, MachOEntry>,
    pub files: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachOEntry {
    pub input_sha256: String,
    pub kind: MachOKind,
    pub slices: Vec<SliceFacts>,
    pub output_sha256: String,
    pub output_len: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MachOKind {
    Fat,
    Thin,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SliceFacts {
    pub arch: String,
    pub header_offset: u64,
    pub mh_flags: String,
    pub linkedit: Option<LinkeditFacts>,
    pub code_signature: Option<CodeSignatureFacts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fat_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fat_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fat_align: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkeditFacts {
    pub vmaddr: u64,
    pub vmsize: u64,
    pub fileoff: u64,
    pub filesize: u64,
    pub vmsize_field_offset: u64,
    pub fileoff_field_offset: u64,
    pub filesize_field_offset: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodeSignatureFacts {
    pub dataoff: u32,
    pub datasize: u32,
    pub lc_offset: u64,
    pub superblob_sha256: String,
}
