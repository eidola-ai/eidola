//! The idle-KV cipher on the device against `eidola_engine::kv_cipher`, the definition the
//! CPU reference executor runs, byte for byte, on the device's exact cubin and the family
//! cubin: RFC 8439's vectors; random ranges whose lengths straddle keystream-block edges,
//! at 16-byte aligned and unaligned bases (the kernel's vector and scalar paths), from
//! counter 0 and mid-stream; and whole KV blocks through the store's `Encrypt` and
//! `Decrypt`, target and drafter layouts (every piece from a fresh keystream block, a
//! drafter block's tap included), with the other blocks untouched and decryption
//! restoring every bit. The device item list (which carries the keys) is zero after every
//! launch. Every case's host side (data, items, expected output) is built and checked
//! against the RFC without a device too (`cases_are_built_without_a_device`).

mod common;

use common::{Lcg, setup};
use eidola_engine::executor::Maintenance;
use eidola_engine::kv_cipher::{KvKeystream, KvNonce};
use eidola_engine::secret::KvKey;
use eidola_engine_cuda::kv::{GroupGeometry, KvLayout, KvStore};
use eidola_engine_cuda::kv_cipher::{CipherItem, CipherItems, KvCipher};
use eidola_engine_cuda::launch::dptr;

struct Case {
    key: [u8; 32],
    nonce: [u32; 3],
    counter: u32,
    /// Word offset of the range in the shared buffer.
    at: usize,
    data: Vec<u32>,
}

impl Case {
    fn expected(&self) -> Vec<u32> {
        let mut out = self.data.clone();
        KvKeystream::with_counter(
            &KvKey::from_bytes(self.key),
            &KvNonce::from_words(self.nonce),
            self.counter,
        )
        .apply_words(&mut out);
        out
    }
}

fn hex_words(s: &str) -> Vec<u32> {
    let s: String = s.split_whitespace().collect();
    (0..s.len())
        .step_by(8)
        .map(|i| {
            let b: Vec<u8> = (0..4)
                .map(|j| u8::from_str_radix(&s[i + 2 * j..i + 2 * j + 2], 16).unwrap())
                .collect();
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })
        .collect()
}

/// RFC 8439 §2.3.2: the block function's serialized output for key 00..1f, nonce
/// 000000090000004a00000000, counter 1, as the XOR of 16 zero words.
fn rfc_block_case() -> (Case, Vec<u32>) {
    let key = std::array::from_fn(|i| u8::try_from(i).unwrap());
    let case = Case {
        key,
        nonce: [0x0900_0000, 0x4a00_0000, 0],
        counter: 1,
        at: 0,
        data: vec![0; 16],
    };
    let want = hex_words(
        "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e
         d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e",
    );
    (case, want)
}

/// The cases, laid out back to back in one buffer: lengths around 16-word edges, bases
/// alternately 16-byte aligned and one word off.
fn cases() -> Vec<Case> {
    let mut rng = Lcg(0xC1FE);
    let mut out = vec![rfc_block_case().0];
    let mut at: usize = 16;
    for (i, words) in [1usize, 15, 16, 17, 31, 32, 33, 100, 4096, 4097, 65_537]
        .into_iter()
        .enumerate()
    {
        // Odd cases start one word past a 16-byte boundary.
        at = at.next_multiple_of(4) + (i % 2);
        let key = std::array::from_fn(|_| rng.next_u64().to_le_bytes()[3]);
        let nonce = std::array::from_fn(|_| u32::try_from(rng.next_u64() >> 32).unwrap());
        let counter = if i % 3 == 0 {
            0
        } else {
            u32::try_from(rng.below(1 << 20)).unwrap()
        };
        let data = (0..words)
            .map(|_| u32::try_from(rng.next_u64() >> 32).unwrap())
            .collect();
        out.push(Case {
            key,
            nonce,
            counter,
            at,
            data,
        });
        at += words;
    }
    out
}

fn buffer_len(cases: &[Case]) -> usize {
    cases.iter().map(|c| c.at + c.data.len()).max().unwrap() + 4
}

#[test]
fn cases_are_built_without_a_device() {
    let (case, want) = rfc_block_case();
    assert_eq!(case.expected(), want);
    let cs = cases();
    for w in cs.windows(2) {
        assert!(w[0].at + w[0].data.len() <= w[1].at, "cases overlap");
    }
    assert!(cs.iter().any(|c| c.at % 4 == 0 && c.data.len() >= 16));
    assert!(cs.iter().any(|c| c.at % 4 == 1 && c.data.len() >= 16));
    for c in &cs[1..] {
        let e = c.expected();
        let same = e.iter().zip(&c.data).filter(|(a, b)| a == b).count();
        assert!(
            same * 100 < c.data.len().max(100),
            "the keystream changes the data"
        );
    }
}

#[test]
fn the_kernel_matches_the_reference_byte_for_byte() {
    let Some(su) = setup() else { return };
    let s = su.gpu.stream();
    let cs = cases();
    let mut host = vec![0xA5A5_A5A5u32; buffer_len(&cs)];
    for c in &cs {
        host[c.at..c.at + c.data.len()].copy_from_slice(&c.data);
    }
    let mut want = host.clone();
    for c in &cs {
        want[c.at..c.at + c.data.len()].copy_from_slice(&c.expected());
    }
    for &arch in &su.archs {
        let mut cipher = KvCipher::from_module(su.module(KvCipher::IMAGE, arch)).unwrap();
        let buf = s.clone_htod(&host).unwrap();
        let base = dptr(&buf, s);
        let mut items: CipherItems = cs
            .iter()
            .map(|c| {
                CipherItem::new(
                    base + 4 * c.at as u64,
                    u32::try_from(c.data.len()).unwrap(),
                    c.counter,
                    &KvKey::from_bytes(c.key),
                    &KvNonce::from_words(c.nonce),
                )
            })
            .collect();
        cipher.apply(&su.gpu, &mut items).unwrap();
        assert!(items.is_empty(), "the host item list is consumed");
        let got = s.clone_dtoh(&buf).unwrap();
        for (i, c) in cs.iter().enumerate() {
            assert_eq!(
                got[c.at..c.at + c.data.len()],
                want[c.at..c.at + c.data.len()],
                "{arch:?}: case {i} ({} words at {})",
                c.data.len(),
                c.at
            );
        }
        assert_eq!(got, want, "{arch:?}: bytes outside the ranges changed");
        assert!(
            cipher
                .read_items(&su.gpu)
                .unwrap()
                .iter()
                .all(|i| *i == CipherItem::default()),
            "{arch:?}: the device item list (the keys) is not zero"
        );
    }
}

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

/// A block as `read_block` returns it, encrypted piece by piece with the reference.
fn encrypt_image(
    g: &GroupGeometry,
    block: u32,
    image: &[u16],
    key: &KvKey,
    nonce: &KvNonce,
) -> Vec<u16> {
    let mut words: Vec<u32> = image
        .as_chunks::<2>()
        .0
        .iter()
        .map(|h| u32::from(h[0]) | (u32::from(h[1]) << 16))
        .collect();
    let mut ks = KvKeystream::new(key, nonce);
    let mut at = 0;
    for r in g.block_ranges(block) {
        let n = r.bytes().1 / 4;
        ks.apply_words(&mut words[at..at + n]);
        at += n;
    }
    assert_eq!(at, words.len());
    words
        .iter()
        .flat_map(|w| {
            let [a, b, c, d] = w.to_le_bytes();
            [u16::from_le_bytes([a, b]), u16::from_le_bytes([c, d])]
        })
        .collect()
}

#[test]
fn the_store_encrypts_whole_blocks_like_the_reference() {
    let Some(su) = setup() else { return };
    let mut rng = Lcg(77);
    for &arch in &su.archs {
        let mut store = KvStore::new(&su.gpu, geometry(), 2, 4, 0).unwrap();
        store.set_cipher(KvCipher::from_module(su.module(KvCipher::IMAGE, arch)).unwrap());
        let mut plain = Vec::new();
        for (g, geom) in store.geometry().to_vec().iter().enumerate() {
            let mut blocks = Vec::new();
            for b in 0..geom.num_blocks {
                let data: Vec<u16> = (0..geom.block_bytes() / 2)
                    .map(|_| u16::try_from(rng.next_u64() >> 48).unwrap())
                    .collect();
                store.write_block(&su.gpu, g, b, &data).unwrap();
                blocks.push(data);
            }
            plain.push(blocks);
        }
        let keys = [KvKey::from_bytes([3; 32]), KvKey::from_bytes([4; 32])];
        let nonce = |g: u32, b: u32| KvNonce::from_words([g, b, 0x5eed]);
        let crypt = |encrypt: bool| -> Vec<Maintenance> {
            [(0u32, 2u32, 0u32), (0, 4, 1), (1, 1, 0), (1, 3, 1)]
                .into_iter()
                .map(|(group, block, key)| {
                    let nonce = nonce(group, block);
                    if encrypt {
                        Maintenance::Encrypt {
                            group,
                            block,
                            key,
                            nonce,
                        }
                    } else {
                        Maintenance::Decrypt {
                            group,
                            block,
                            key,
                            nonce,
                        }
                    }
                })
                .collect()
        };
        store
            .apply_keyed(&su.gpu, &crypt(true), &keys, &[])
            .unwrap();
        let geoms = store.geometry().to_vec();
        for (g, geom) in geoms.iter().enumerate() {
            for b in 0..geom.num_blocks {
                let got = store.read_block(&su.gpu, g, b).unwrap();
                let named = [(0, 2, 0), (0, 4, 1), (1, 1, 0), (1, 3, 1)]
                    .into_iter()
                    .find(|&(gg, bb, _)| gg == g && bb == b);
                let want = match named {
                    Some((_, _, k)) => encrypt_image(
                        geom,
                        b,
                        &plain[g][b as usize],
                        &keys[k],
                        &nonce(u32::try_from(g).unwrap(), b),
                    ),
                    None => plain[g][b as usize].clone(),
                };
                assert_eq!(got, want, "{arch:?}: group {g} block {b}");
            }
        }
        assert!(
            store
                .cipher()
                .unwrap()
                .read_items(&su.gpu)
                .unwrap()
                .iter()
                .all(|i| *i == CipherItem::default()),
            "{arch:?}: the device item list is not zero"
        );
        store
            .apply_keyed(&su.gpu, &crypt(false), &keys, &[])
            .unwrap();
        for (g, geom) in geoms.iter().enumerate() {
            for b in 0..geom.num_blocks {
                assert_eq!(
                    store.read_block(&su.gpu, g, b).unwrap(),
                    plain[g][b as usize],
                    "{arch:?}: group {g} block {b} after decryption"
                );
            }
        }
    }
}
