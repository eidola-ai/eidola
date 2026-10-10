//! Device evidence policy, independent of the CPU platform.
//!
//! A pinned entry may declare how many NVIDIA GPUs its enclave runs with.
//! The CPU measurement already vouches for boot code that refuses to start
//! unless the GPUs attest in confidential-computing mode; this check adds the
//! per-handshake half: the document must carry exactly that many GPU
//! evidence items, each produced for this handshake's nonce. The items sit
//! inside the endorsed `device_evidence` section, so the CPU quote's
//! `REPORT_DATA` already binds their bytes. NVIDIA's certificate chain, the
//! SPDM report signature and the reference measurements are not appraised.

use serde::Deserialize;

use crate::Error;
use crate::bundle::{self, DeviceEvidence, NVIDIA_GPU_EVIDENCE_V1_FORMAT};

const KIND_GPU: &str = "gpu";
const VENDOR_NVIDIA: &str = "nvidia";

/// The `nvidia-gpu-evidence/v1` payload.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NvidiaGpuEvidenceV1 {
    #[allow(dead_code)]
    arch: String,
    #[allow(dead_code)]
    certificate: String,
    #[allow(dead_code)]
    evidence: String,
    nonce: String,
}

/// Require exactly `expected` nonce-bound NVIDIA GPU evidence items.
pub(crate) fn check_gpu_evidence(
    items: &[DeviceEvidence],
    expected: u32,
    nonce: &[u8; bundle::NONCE_LEN],
) -> Result<(), Error> {
    let mut count: u32 = 0;
    for item in items {
        let is_gpu_format = item.format == NVIDIA_GPU_EVIDENCE_V1_FORMAT;
        if item.kind == KIND_GPU && !is_gpu_format {
            // A GPU the policy cannot count must not ride along uncounted.
            return Err(Error::DeviceEvidence(format!(
                "device evidence item {:?} is a GPU in unsupported format {:?}",
                item.id, item.format
            )));
        }
        if !is_gpu_format {
            continue;
        }
        if item.kind != KIND_GPU || item.vendor != VENDOR_NVIDIA {
            return Err(Error::DeviceEvidence(format!(
                "device evidence item {:?} has format nvidia-gpu-evidence/v1 but kind {:?} \
                 vendor {:?}",
                item.id, item.kind, item.vendor
            )));
        }
        let payload: NvidiaGpuEvidenceV1 =
            serde_json::from_value(item.evidence.clone()).map_err(|e| {
                Error::DeviceEvidence(format!("device evidence item {:?}: {e}", item.id))
            })?;
        let item_nonce = bundle::decode_lower_hex_array::<{ bundle::NONCE_LEN }>(
            &payload.nonce,
            &format!("device evidence item {:?} nonce", item.id),
        )
        .map_err(|e| Error::DeviceEvidence(e.to_string()))?;
        if item_nonce != *nonce {
            return Err(Error::DeviceEvidence(format!(
                "device evidence item {:?} was produced for nonce {}, not this handshake's",
                item.id, payload.nonce
            )));
        }
        count += 1;
    }
    if count != expected {
        return Err(Error::DeviceEvidence(format!(
            "expected {expected} NVIDIA GPU evidence items, the document carries {count}"
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn gpu(id: &str, nonce: &[u8; 32]) -> DeviceEvidence {
        DeviceEvidence {
            id: id.to_string(),
            kind: "gpu".to_string(),
            vendor: "nvidia".to_string(),
            format: NVIDIA_GPU_EVIDENCE_V1_FORMAT.to_string(),
            evidence: serde_json::json!({
                "arch": "HOPPER",
                "certificate": "Y2VydA==",
                "evidence": "ZXZpZGVuY2U=",
                "nonce": hex::encode(nonce),
            }),
        }
    }

    fn nvswitch() -> DeviceEvidence {
        DeviceEvidence {
            id: "nvswitch0".to_string(),
            kind: "nvswitch".to_string(),
            vendor: "nvidia".to_string(),
            format: "https://tinfoil.sh/format/nvidia-nvswitch-evidence/v1".to_string(),
            evidence: serde_json::json!({"anything": true}),
        }
    }

    const NONCE: [u8; 32] = [0xab; 32];

    #[test]
    fn accepts_exactly_the_expected_nonce_bound_gpus() {
        let items = vec![gpu("gpu0", &NONCE), gpu("gpu1", &NONCE), nvswitch()];
        check_gpu_evidence(&items, 2, &NONCE).expect("two bound GPUs");
        check_gpu_evidence(&[nvswitch()], 0, &NONCE).expect("no GPUs expected");
    }

    #[test]
    fn refuses_wrong_counts_stale_nonces_and_disguised_items() {
        let stale = gpu("gpu1", &[8; 32]);
        let mut wrong_vendor = gpu("gpu1", &NONCE);
        wrong_vendor.vendor = "amd".to_string();
        let mut other_format = gpu("gpu1", &NONCE);
        other_format.format = "https://tinfoil.sh/format/nvidia-gpu-evidence/v2".to_string();
        let mut extra_member = gpu("gpu1", &NONCE);
        extra_member.evidence["unexpected"] = serde_json::Value::Bool(true);
        let mut upper_nonce = gpu("gpu1", &NONCE);
        upper_nonce.evidence["nonce"] = hex::encode_upper(NONCE).into();

        let cases: Vec<(&str, Vec<DeviceEvidence>, u32)> = vec![
            ("expected 2", vec![gpu("gpu0", &NONCE)], 2),
            (
                "expected 1",
                vec![gpu("gpu0", &NONCE), gpu("gpu1", &NONCE)],
                1,
            ),
            ("expected 1", vec![], 1),
            ("not this handshake", vec![gpu("gpu0", &NONCE), stale], 2),
            ("kind", vec![gpu("gpu0", &NONCE), wrong_vendor], 2),
            (
                "unsupported format",
                vec![gpu("gpu0", &NONCE), other_format],
                1,
            ),
            ("unknown field", vec![gpu("gpu0", &NONCE), extra_member], 2),
            ("lowercase hex", vec![gpu("gpu0", &NONCE), upper_nonce], 2),
        ];
        for (needle, items, expected) in cases {
            let err = check_gpu_evidence(&items, expected, &NONCE)
                .expect_err(needle)
                .to_string();
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }
}
