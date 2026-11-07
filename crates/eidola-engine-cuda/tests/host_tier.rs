//! The host tier through the store's seam operations on a device: `Demote`
//! copies a block's whole image (a target block's one range; a drafter
//! block's plane rows and its boundary tap) into page-locked host memory byte
//! for byte, `Promote` copies it back byte for byte, ordered on the stream
//! before the decryption that follows it and after the zero before it, and
//! `ZeroHost` leaves zeros once the stream drains. Nothing but these moves a
//! host block.

mod common;

use common::{Lcg, setup};
use eidola_engine::executor::Maintenance;
use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::secret::KvKey;
use eidola_engine_cuda::kv::{GroupGeometry, KvLayout, KvStore};
use eidola_engine_cuda::kv_cipher::KvCipher;

fn geometry() -> Vec<GroupGeometry> {
    vec![
        GroupGeometry {
            num_layers: 3,
            num_kv_heads: 4,
            head_dim_qk: 192,
            head_dim_v: 128,
            block_size: 16,
            num_blocks: 6,
            layout: KvLayout::Blocked,
        },
        GroupGeometry {
            num_layers: 2,
            num_kv_heads: 8,
            head_dim_qk: 192,
            head_dim_v: 128,
            block_size: 16,
            num_blocks: 5,
            layout: KvLayout::Planar {
                taps: 2,
                hidden: 64,
            },
        },
    ]
}

fn bytes(image: &[u16]) -> Vec<u8> {
    image.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[test]
#[allow(clippy::needless_range_loop)]
fn demote_and_promote_move_whole_ciphertext_images() {
    let Some(su) = setup() else { return };
    let mut rng = Lcg(91);
    for &arch in &su.archs {
        let mut store = KvStore::new(&su.gpu, geometry(), 2, 4, 0).unwrap();
        store.set_cipher(KvCipher::from_module(su.module(KvCipher::IMAGE, arch)).unwrap());
        store.set_host_tier(&su.gpu, &[4, 4]).unwrap();
        let geoms = store.geometry().to_vec();
        for (g, geom) in geoms.iter().enumerate() {
            for h in 0..4 {
                assert!(
                    store
                        .read_host_block(&su.gpu, g, h)
                        .unwrap()
                        .iter()
                        .all(|&x| x == 0),
                    "{arch:?}: the host tier starts zeroed"
                );
            }
            for b in 1..geom.num_blocks {
                let data: Vec<u16> = (0..geom.block_bytes() / 2)
                    .map(|_| u16::try_from(rng.next_u64() >> 48).unwrap())
                    .collect();
                store.write_block(&su.gpu, g, b, &data).unwrap();
            }
        }
        let key = [KvKey::from_bytes([5; 32])];
        let nonce = |g: u32| KvNonce::from_words([g, 0x51, 0x5eed]);
        // Block 2 of each group, encrypted (as the core holds an idle block).
        let enc: Vec<Maintenance> = (0..2u32)
            .map(|group| Maintenance::Encrypt {
                group,
                block: 2,
                key: 0,
                nonce: nonce(group),
            })
            .collect();
        let plain: Vec<Vec<u16>> = (0..2)
            .map(|g| store.read_block(&su.gpu, g, 2).unwrap())
            .collect();
        store.apply_keyed(&su.gpu, &enc, &key, &[]).unwrap();
        let cipher: Vec<Vec<u16>> = (0..2)
            .map(|g| store.read_block(&su.gpu, g, 2).unwrap())
            .collect();
        let untouched: Vec<Vec<u16>> = (0..2)
            .map(|g| store.read_block(&su.gpu, g, 3).unwrap())
            .collect();
        // Demote to host block 1, then zero the device block (the core's order).
        let mut m = Vec::new();
        for group in 0..2u32 {
            m.push(Maintenance::Demote {
                group,
                block: 2,
                host: 1,
            });
        }
        for group in 0..2u32 {
            m.push(Maintenance::Zero { group, block: 2 });
        }
        store.apply(&su.gpu, &m, &[]).unwrap();
        for g in 0..2usize {
            let host = store.read_host_block(&su.gpu, g, 1).unwrap();
            assert_eq!(
                host,
                bytes(&cipher[g]),
                "{arch:?}: group {g}: ciphertext on the host"
            );
            assert_ne!(host, bytes(&plain[g]));
            assert!(
                store
                    .read_block(&su.gpu, g, 2)
                    .unwrap()
                    .iter()
                    .all(|&x| x == 0)
            );
            assert_eq!(store.read_block(&su.gpu, g, 3).unwrap(), untouched[g]);
            for h in [0u32, 2, 3] {
                assert!(
                    store
                        .read_host_block(&su.gpu, g, h)
                        .unwrap()
                        .iter()
                        .all(|&x| x == 0)
                );
            }
        }
        // Zero block 4, promote host block 1 into it, decrypt it, zero the host
        // block: one step, in the core's order.
        let mut m = Vec::new();
        for group in 0..2u32 {
            m.push(Maintenance::Zero { group, block: 4 });
        }
        for group in 0..2u32 {
            m.push(Maintenance::Promote {
                group,
                host: 1,
                block: 4,
            });
        }
        for group in 0..2u32 {
            m.push(Maintenance::Decrypt {
                group,
                block: 4,
                key: 0,
                nonce: nonce(group),
            });
        }
        for group in 0..2u32 {
            m.push(Maintenance::ZeroHost { group, host: 1 });
        }
        store.apply_keyed(&su.gpu, &m, &key, &[]).unwrap();
        store.finish_host(&su.gpu).unwrap();
        for g in 0..2usize {
            assert_eq!(
                store.read_block(&su.gpu, g, 4).unwrap(),
                plain[g],
                "{arch:?}: group {g}: promoted and decrypted, bit for bit"
            );
            assert!(
                store
                    .read_host_block(&su.gpu, g, 1)
                    .unwrap()
                    .iter()
                    .all(|&x| x == 0)
            );
        }
    }
}
