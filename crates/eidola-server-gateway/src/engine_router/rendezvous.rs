//! Which upstream a request tries first.
//!
//! **With a cache key**, rendezvous (highest-random-weight) hashing: every
//! request carrying the same key ranks the same upstreams in the same order,
//! so a conversation's requests land on the engine holding its prefix cache,
//! and when an upstream joins or leaves only the keys it wins move. The key is
//! first reduced to a route key, `SHA-256("eidola/route/v1" ‖ cache_key)`,
//! and an upstream's weight is the first eight bytes (big-endian) of
//! `SHA-256(route_key ‖ len(base_url) ‖ base_url)`. No secret is needed:
//! the key is 256 random bits the client chose, so the ranking reveals
//! nothing an observer of the routing could not already compute only by
//! holding the key. Every buffer the key or a value derived from it passes
//! through here is scrubbed: the hashers (`sha2`'s `zeroize` feature), the
//! route key, and the weights.
//!
//! **Without one**, the upstream with the fewest requests in flight, ties
//! broken by a rotating offset so equal upstreams share the load.

use std::sync::Arc;

use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use super::Upstream;
use crate::types::CacheKey;

/// The domain separator of the route key.
pub const ROUTE_DOMAIN: &[u8] = b"eidola/route/v1";

/// `SHA-256(ROUTE_DOMAIN ‖ cache_key)`, in a scrubbed buffer.
fn route_key(key: &CacheKey) -> Zeroizing<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(ROUTE_DOMAIN);
    hasher.update(key.as_bytes());
    let mut out = Zeroizing::new([0u8; 32]);
    out.copy_from_slice(&hasher.finalize());
    out
}

/// An upstream's weight under `route_key`.
fn weight(route_key: &[u8; 32], base_url: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(route_key);
    hasher.update((base_url.len() as u64).to_be_bytes());
    hasher.update(base_url.as_bytes());
    let mut digest = Zeroizing::new([0u8; 32]);
    digest.copy_from_slice(&hasher.finalize());
    let mut first = [0u8; 8];
    first.copy_from_slice(&digest[..8]);
    let weight = u64::from_be_bytes(first);
    first.zeroize();
    weight
}

/// `upstreams` in the order a request with `key` tries them: by descending
/// weight, ties (vanishingly unlikely) by base URL.
pub fn rank_by_key(key: &CacheKey, upstreams: Vec<Arc<Upstream>>) -> Vec<Arc<Upstream>> {
    let route_key = route_key(key);
    let mut weighted: Vec<(u64, Arc<Upstream>)> = upstreams
        .into_iter()
        .map(|u| (weight(&route_key, &u.base_url), u))
        .collect();
    weighted.sort_by(|(wa, a), (wb, b)| wb.cmp(wa).then_with(|| a.base_url.cmp(&b.base_url)));
    let ranked = weighted.iter().map(|(_, u)| u.clone()).collect();
    for (w, _) in &mut weighted {
        w.zeroize();
    }
    ranked
}

/// `upstreams` by fewest requests in flight, equal counts in an order rotated
/// by `rotation`.
pub fn rank_by_load(upstreams: Vec<Arc<Upstream>>, rotation: usize) -> Vec<Arc<Upstream>> {
    let n = upstreams.len().max(1);
    let mut indexed: Vec<(usize, usize, Arc<Upstream>)> = upstreams
        .into_iter()
        .enumerate()
        .map(|(i, u)| (u.in_flight(), (i + n - rotation % n) % n, u))
        .collect();
    indexed.sort_by_key(|(load, order, _)| (*load, *order));
    indexed.into_iter().map(|(_, _, u)| u).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> CacheKey {
        CacheKey::from_bytes([byte; 32])
    }

    fn keys(n: usize) -> impl Iterator<Item = CacheKey> {
        (0..n).map(|i| {
            let mut bytes = [0u8; 32];
            bytes[..8].copy_from_slice(&(i as u64).to_le_bytes());
            CacheKey::from_bytes(bytes)
        })
    }

    fn upstreams(n: usize) -> Vec<Arc<Upstream>> {
        (0..n)
            .map(|i| Arc::new(Upstream::fixture(&format!("https://engine-{i}.test"))))
            .collect()
    }

    fn first(key: &CacheKey, ups: &[Arc<Upstream>]) -> String {
        rank_by_key(key, ups.to_vec())[0].base_url.clone()
    }

    /// The weight is the documented function, so another implementation can
    /// reproduce a ranking from the description alone.
    #[test]
    fn the_weight_is_the_documented_hash() {
        let k = key(7);
        let route: [u8; 32] = Sha256::digest([ROUTE_DOMAIN, &[7u8; 32][..]].concat()).into();
        let url = "https://engine-0.test";
        let mut input = route.to_vec();
        input.extend_from_slice(&(url.len() as u64).to_be_bytes());
        input.extend_from_slice(url.as_bytes());
        let digest = Sha256::digest(&input);
        let expected = u64::from_be_bytes(digest[..8].try_into().unwrap());
        assert_eq!(weight(&route_key(&k), url), expected);
    }

    /// One key always ranks the same way, whatever order the upstreams are
    /// listed in.
    #[test]
    fn a_key_always_ranks_the_same_way() {
        let ups = upstreams(5);
        let mut reversed = ups.clone();
        reversed.reverse();
        for k in keys(50) {
            let a: Vec<String> = rank_by_key(&k, ups.clone())
                .iter()
                .map(|u| u.base_url.clone())
                .collect();
            let b: Vec<String> = rank_by_key(&k, reversed.clone())
                .iter()
                .map(|u| u.base_url.clone())
                .collect();
            assert_eq!(a, b);
        }
    }

    /// Removing an upstream moves only the keys it won, and they move to what
    /// was each key's second choice; the keys spread over every upstream.
    #[test]
    fn removing_an_upstream_moves_only_its_keys() {
        let ups = upstreams(4);
        let removed = ups[1].base_url.clone();
        let rest: Vec<Arc<Upstream>> = ups
            .iter()
            .filter(|u| u.base_url != removed)
            .cloned()
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        for k in keys(400) {
            let ranked = rank_by_key(&k, ups.clone());
            seen.insert(ranked[0].base_url.clone());
            let after = first(&k, &rest);
            if ranked[0].base_url == removed {
                assert_eq!(after, ranked[1].base_url);
            } else {
                assert_eq!(after, ranked[0].base_url);
            }
        }
        assert_eq!(seen.len(), 4, "400 keys reach every one of 4 upstreams");
    }

    #[test]
    fn without_a_key_the_least_loaded_goes_first_and_ties_rotate() {
        let ups = upstreams(3);
        let _busy = ups[0].begin();
        let _busier = (ups[1].begin(), ups[1].begin());
        let ranked = rank_by_load(ups.clone(), 0);
        assert_eq!(ranked[0].base_url, ups[2].base_url);
        assert_eq!(ranked[2].base_url, ups[1].base_url);

        let idle = upstreams(3);
        let firsts: std::collections::BTreeSet<String> = (0..3)
            .map(|r| rank_by_load(idle.clone(), r)[0].base_url.clone())
            .collect();
        assert_eq!(firsts.len(), 3, "equal upstreams take turns");
    }
}
