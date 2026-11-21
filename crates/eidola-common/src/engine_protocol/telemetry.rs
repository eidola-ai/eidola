//! A node's operational snapshot: the `telemetry` member of its
//! `GET /v1/engine/info` answer.
//!
//! A node has no egress of its own, so what an operator learns about how it
//! runs reaches them through the gateway: the gateway reads this snapshot
//! when it probes the node, over the same attested, token-authenticated
//! connection, and exports it through its own telemetry, labelled by the
//! model's weights hash. The node writes it with [`NodeTelemetry::to_value`]
//! and the gateway reads it with [`NodeTelemetry::from_value`], which refuses
//! every other shape: an unknown member, a missing one, a value of another
//! type, more pools than [`MAX_POOLS`], or an occupancy past its pool's size.
//! The member itself is optional on the info answer (a node built before it
//! answers without it), and the info answer's other members do not depend on
//! it.
//!
//! **Nothing in it is per request.** Every value is a process-wide aggregate:
//! a count since the process started ([`NodeCounters`], cumulative and
//! monotonic, so the reader turns two snapshots into a delta), or a level now
//! ([`NodeGauges`]). There is no request id, cache key, lineage, salt,
//! token, timestamp of an event, or size of any one request. Two kinds of
//! value are coarsened on the node, before they leave it, because nothing
//! the gateway exports already says as much: the KV pools' occupancy, rounded
//! to [`OCCUPANCY_STEPS`] steps of its pool ([`quantize_occupancy`]), and the
//! prefix cache's hit and miss token counts, floored to
//! [`PREFIX_TOKEN_QUANTUM`] ([`quantize_prefix_tokens`]).
//! `crates/eidola-server-engine/AGENTS.md` → Telemetry has the reasoning
//! for each member.
//!
//! `boot_id` is random per process: a reader holding an earlier snapshot
//! with another one knows the node restarted and its counts started again.

use serde_json::{Map, Value, json};

/// The info answer's member the snapshot travels in.
pub const INFO_TELEMETRY_MEMBER: &str = "telemetry";

/// Bytes of a [`NodeTelemetry::boot_id`], written as lowercase hex.
pub const BOOT_ID_BYTES: usize = 16;

/// The most pools a snapshot reports. The engine's pools are its KV groups,
/// in the order its architecture declares them, then its recurrent-state
/// pool if it has one; no architecture has more than six.
pub const MAX_POOLS: usize = 8;

/// The steps a pool's occupancy is reported in: each count is rounded to the
/// nearest multiple of `ceil(total / OCCUPANCY_STEPS)` blocks.
pub const OCCUPANCY_STEPS: u64 = 256;

/// The step the prefix cache's hit and miss token counts are floored to.
pub const PREFIX_TOKEN_QUANTUM: u64 = 1024;

/// `count` blocks of a pool of `total`, rounded to the nearest multiple of
/// the pool's step (`ceil(total / OCCUPANCY_STEPS)`, at least one block),
/// and never past `total`.
pub const fn quantize_occupancy(count: u64, total: u64) -> u64 {
    let step = total.div_ceil(OCCUPANCY_STEPS);
    let step = if step == 0 { 1 } else { step };
    let rounded = (count.saturating_add(step / 2) / step) * step;
    if rounded > total { total } else { rounded }
}

/// A cumulative prefix-cache token count, floored to
/// [`PREFIX_TOKEN_QUANTUM`]. Floored rather than rounded, so the reported
/// value never decreases while the count grows.
pub const fn quantize_prefix_tokens(count: u64) -> u64 {
    count / PREFIX_TOKEN_QUANTUM * PREFIX_TOKEN_QUANTUM
}

macro_rules! counters {
    ($($(#[$doc:meta])* $name:ident,)*) => {
        /// Counts since the process started: each only ever grows within one
        /// `boot_id`.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct NodeCounters {
            $($(#[$doc])* pub $name: u64,)*
        }

        impl NodeCounters {
            /// The members, as they are spelled on the wire.
            pub const MEMBERS: &'static [&'static str] = &[$(stringify!($name)),*];

            /// Each member's name and value, in [`NodeCounters::MEMBERS`] order.
            pub fn entries(&self) -> Vec<(&'static str, u64)> {
                vec![$((stringify!($name), self.$name)),*]
            }

            /// Each member's growth since `earlier`, zero where it did not
            /// grow.
            pub fn since(&self, earlier: &Self) -> Self {
                Self { $($name: self.$name.saturating_sub(earlier.$name),)* }
            }

            /// Whether no member is below its value in `earlier`.
            pub fn covers(&self, earlier: &Self) -> bool {
                true $(&& self.$name >= earlier.$name)*
            }

            fn to_value(self) -> Value {
                let mut object = Map::new();
                $(object.insert(stringify!($name).into(), self.$name.into());)*
                Value::Object(object)
            }

            fn from_value(value: &Value) -> Option<Self> {
                let object = value.as_object()?;
                if object.len() != Self::MEMBERS.len() {
                    return None;
                }
                Some(Self { $($name: object.get(stringify!($name))?.as_u64()?,)* })
            }
        }
    };
}

counters! {
    /// Requests the engine accepted.
    requests_accepted,
    /// Requests that finished on their own (an end of sequence, a length).
    requests_finished,
    /// Requests cancelled: a disconnect, a stop sequence, a reader cut off.
    requests_cancelled,
    /// Of those, readers cut off because they stopped draining their output.
    requests_cut_off,
    /// Requests the engine refused at submission.
    requests_rejected,
    /// Requests refused at first admission because the prefix cache did not
    /// hold their declared prefix.
    requests_prefix_refused,
    /// Requests the node refused before reading or admitting them because
    /// every read or admission slot was taken (`overloaded`).
    requests_overloaded,
    /// Engine steps that ran at least one sequence.
    steps,
    /// Wall time of those steps, in microseconds.
    step_micros,
    /// Sequences those steps ran, summed over the steps: over `steps`, the
    /// mean batch.
    step_rows,
    /// Prompt positions computed: the prompts' uncached parts, and the
    /// recomputation of a preempted sequence.
    prefill_tokens,
    /// Tokens generated.
    generated_tokens,
    /// Speculative draft tokens proposed.
    drafted_tokens,
    /// Of those, the ones accepted.
    accepted_drafts,
    /// Running sequences preempted to make room.
    preemptions,
    /// Requests admitted for the first time, each with one prefix-cache
    /// lookup.
    prefix_lookups,
    /// Prompt tokens those lookups found in the prefix cache, floored to
    /// [`PREFIX_TOKEN_QUANTUM`].
    prefix_hit_tokens,
    /// Prompt tokens they did not, floored to [`PREFIX_TOKEN_QUANTUM`].
    prefix_miss_tokens,
    /// Cache entries dropped from the device pools to make room, with no
    /// host tier to take them.
    cache_device_evictions,
    /// Cache entries dropped from the host tier to make room.
    cache_host_evictions,
    /// Blocks demoted from the device pools to the host tier.
    cache_demoted_blocks,
    /// Blocks promoted from the host tier back to the device for a hit.
    cache_promoted_blocks,
}

/// One pool's occupancy, its counts quantized ([`quantize_occupancy`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolOccupancy {
    /// Blocks on the device, exact (the pool's size, fixed at boot).
    pub total: u64,
    /// Blocks a running sequence maps.
    pub in_use: u64,
    /// Blocks the prefix cache holds that no sequence maps (idle, encrypted).
    pub idle: u64,
    /// Blocks in the host tier, exact; zero without one.
    pub host_total: u64,
    /// Host-tier blocks holding a demoted entry.
    pub host_used: u64,
}

impl PoolOccupancy {
    /// The members, as they are spelled on the wire.
    pub const MEMBERS: &'static [&'static str] =
        &["total", "in_use", "idle", "host_total", "host_used"];

    /// Quantizes exact counts for the wire. Each rounds on its own, and the
    /// idle count gives way where the two would round past the pool.
    pub const fn quantized(
        total: u64,
        in_use: u64,
        idle: u64,
        host_total: u64,
        host_used: u64,
    ) -> Self {
        let in_use = quantize_occupancy(in_use, total);
        let idle = quantize_occupancy(idle, total);
        let room = total - in_use;
        Self {
            total,
            in_use,
            idle: if idle > room { room } else { idle },
            host_total,
            host_used: quantize_occupancy(host_used, host_total),
        }
    }

    /// Blocks neither in use nor idle.
    pub const fn free(&self) -> u64 {
        self.total
            .saturating_sub(self.in_use)
            .saturating_sub(self.idle)
    }

    /// Host-tier blocks holding nothing.
    pub const fn host_free(&self) -> u64 {
        self.host_total.saturating_sub(self.host_used)
    }

    fn to_value(self) -> Value {
        json!({
            "total": self.total,
            "in_use": self.in_use,
            "idle": self.idle,
            "host_total": self.host_total,
            "host_used": self.host_used,
        })
    }

    fn from_value(value: &Value) -> Option<Self> {
        let object = value.as_object()?;
        if object.len() != Self::MEMBERS.len() {
            return None;
        }
        let member = |name: &str| object.get(name)?.as_u64();
        let pool = Self {
            total: member("total")?,
            in_use: member("in_use")?,
            idle: member("idle")?,
            host_total: member("host_total")?,
            host_used: member("host_used")?,
        };
        let fits =
            pool.in_use.checked_add(pool.idle)? <= pool.total && pool.host_used <= pool.host_total;
        fits.then_some(pool)
    }
}

/// Levels at the moment of the snapshot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeGauges {
    /// Requests the engine holds that are waiting for a seat.
    pub sequences_waiting: u64,
    /// Requests it is running.
    pub sequences_running: u64,
    /// Its pools, at most [`MAX_POOLS`].
    pub pools: Vec<PoolOccupancy>,
}

impl NodeGauges {
    /// The members, as they are spelled on the wire.
    pub const MEMBERS: &'static [&'static str] =
        &["sequences_waiting", "sequences_running", "pools"];

    fn to_value(&self) -> Value {
        json!({
            "sequences_waiting": self.sequences_waiting,
            "sequences_running": self.sequences_running,
            "pools": self.pools.iter().map(|p| p.to_value()).collect::<Vec<_>>(),
        })
    }

    fn from_value(value: &Value) -> Option<Self> {
        let object = value.as_object()?;
        if object.len() != Self::MEMBERS.len() {
            return None;
        }
        let pools = object.get("pools")?.as_array()?;
        if pools.len() > MAX_POOLS {
            return None;
        }
        Some(Self {
            sequences_waiting: object.get("sequences_waiting")?.as_u64()?,
            sequences_running: object.get("sequences_running")?.as_u64()?,
            pools: pools
                .iter()
                .map(PoolOccupancy::from_value)
                .collect::<Option<_>>()?,
        })
    }
}

/// A node's snapshot: the `telemetry` member of its info answer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeTelemetry {
    /// Random per process.
    pub boot_id: [u8; BOOT_ID_BYTES],
    pub counters: NodeCounters,
    pub gauges: NodeGauges,
}

impl NodeTelemetry {
    /// The members, as they are spelled on the wire.
    pub const MEMBERS: &'static [&'static str] = &["boot_id", "counters", "gauges"];

    /// The member's value as a node writes it.
    pub fn to_value(&self) -> Value {
        json!({
            "boot_id": hex(&self.boot_id),
            "counters": self.counters.to_value(),
            "gauges": self.gauges.to_value(),
        })
    }

    /// Reads the member's value: exactly [`NodeTelemetry::MEMBERS`], each of
    /// exactly its own members, `boot_id` as [`BOOT_ID_BYTES`] of lowercase
    /// hex, every number an integer that fits `u64`, at most [`MAX_POOLS`]
    /// pools, none occupied past its size. `None` for anything else.
    pub fn from_value(value: &Value) -> Option<Self> {
        let object = value.as_object()?;
        if object.len() != Self::MEMBERS.len() {
            return None;
        }
        Some(Self {
            boot_id: unhex(object.get("boot_id")?.as_str()?)?,
            counters: NodeCounters::from_value(object.get("counters")?)?,
            gauges: NodeGauges::from_value(object.get("gauges")?)?,
        })
    }
}

fn hex(bytes: &[u8; BOOT_ID_BYTES]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(BOOT_ID_BYTES * 2);
    for &b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0xf) as usize] as char);
    }
    out
}

fn unhex(text: &str) -> Option<[u8; BOOT_ID_BYTES]> {
    let digit = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    let text = text.as_bytes();
    if text.len() != BOOT_ID_BYTES * 2 {
        return None;
    }
    let mut out = [0u8; BOOT_ID_BYTES];
    for (i, pair) in text.chunks_exact(2).enumerate() {
        out[i] = (digit(pair[0])? << 4) | digit(pair[1])?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> NodeTelemetry {
        let mut counters = NodeCounters::default();
        for (i, name) in NodeCounters::MEMBERS.iter().enumerate() {
            // A distinct value per member, through the wire names.
            let mut value = counters.to_value();
            value[*name] = (i as u64 + 1).into();
            counters = NodeCounters::from_value(&value).unwrap();
        }
        NodeTelemetry {
            boot_id: [0xa5; BOOT_ID_BYTES],
            counters,
            gauges: NodeGauges {
                sequences_waiting: 2,
                sequences_running: 3,
                pools: vec![
                    PoolOccupancy::quantized(1000, 300, 200, 0, 0),
                    PoolOccupancy::quantized(64, 5, 7, 512, 100),
                ],
            },
        }
    }

    /// The members are exactly these: a new one is a change to the contract,
    /// and to what leaves the node, so it is made here on purpose. None of
    /// them names or measures a single request.
    #[test]
    fn the_snapshot_has_exactly_these_members() {
        assert_eq!(NodeTelemetry::MEMBERS, &["boot_id", "counters", "gauges"]);
        assert_eq!(
            NodeCounters::MEMBERS,
            &[
                "requests_accepted",
                "requests_finished",
                "requests_cancelled",
                "requests_cut_off",
                "requests_rejected",
                "requests_prefix_refused",
                "requests_overloaded",
                "steps",
                "step_micros",
                "step_rows",
                "prefill_tokens",
                "generated_tokens",
                "drafted_tokens",
                "accepted_drafts",
                "preemptions",
                "prefix_lookups",
                "prefix_hit_tokens",
                "prefix_miss_tokens",
                "cache_device_evictions",
                "cache_host_evictions",
                "cache_demoted_blocks",
                "cache_promoted_blocks",
            ]
        );
        assert_eq!(
            NodeGauges::MEMBERS,
            &["sequences_waiting", "sequences_running", "pools"]
        );
        assert_eq!(
            PoolOccupancy::MEMBERS,
            &["total", "in_use", "idle", "host_total", "host_used"]
        );
        // What is written is exactly the members, at every level.
        let value = sample().to_value();
        let keys = |v: &Value| -> Vec<String> {
            let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        let sorted = |m: &[&str]| -> Vec<String> {
            let mut k: Vec<String> = m.iter().map(|s| s.to_string()).collect();
            k.sort();
            k
        };
        assert_eq!(keys(&value), sorted(NodeTelemetry::MEMBERS));
        assert_eq!(keys(&value["counters"]), sorted(NodeCounters::MEMBERS));
        assert_eq!(keys(&value["gauges"]), sorted(NodeGauges::MEMBERS));
        for pool in value["gauges"]["pools"].as_array().unwrap() {
            assert_eq!(keys(pool), sorted(PoolOccupancy::MEMBERS));
        }
    }

    #[test]
    fn a_snapshot_reads_back_as_written() {
        let snapshot = sample();
        let value = snapshot.to_value();
        assert_eq!(value["boot_id"], "a5".repeat(BOOT_ID_BYTES));
        assert_eq!(NodeTelemetry::from_value(&value), Some(snapshot));
    }

    #[test]
    fn any_other_shape_is_refused() {
        let good = sample().to_value();
        let refused = |edit: &dyn Fn(&mut Value)| {
            let mut value = good.clone();
            edit(&mut value);
            NodeTelemetry::from_value(&value)
        };
        assert!(refused(&|_| {}).is_some());
        type Edit = Box<dyn Fn(&mut Value)>;
        let cases: Vec<(&str, Edit)> = vec![
            ("not an object", Box::new(|v| *v = json!([]))),
            (
                "unknown top member",
                Box::new(|v| v["request_id"] = 1.into()),
            ),
            (
                "missing top member",
                Box::new(|v| {
                    v.as_object_mut().unwrap().remove("gauges");
                }),
            ),
            (
                "unknown counter",
                Box::new(|v| v["counters"]["cache_keys"] = 1.into()),
            ),
            (
                "missing counter",
                Box::new(|v| {
                    v["counters"].as_object_mut().unwrap().remove("steps");
                }),
            ),
            (
                "negative counter",
                Box::new(|v| v["counters"]["steps"] = (-1).into()),
            ),
            (
                "fractional counter",
                Box::new(|v| v["counters"]["steps"] = 1.5.into()),
            ),
            (
                "string counter",
                Box::new(|v| v["counters"]["steps"] = "1".into()),
            ),
            (
                "unknown gauge",
                Box::new(|v| v["gauges"]["lineages"] = 1.into()),
            ),
            (
                "unknown pool member",
                Box::new(|v| v["gauges"]["pools"][0]["name"] = "kv".into()),
            ),
            (
                "occupied past its size",
                Box::new(|v| v["gauges"]["pools"][0]["in_use"] = 2000.into()),
            ),
            (
                "host tier past its size",
                Box::new(|v| v["gauges"]["pools"][1]["host_used"] = 513.into()),
            ),
            (
                "too many pools",
                Box::new(|v| {
                    let pool = v["gauges"]["pools"][0].clone();
                    v["gauges"]["pools"] = Value::Array(vec![pool; MAX_POOLS + 1]);
                }),
            ),
            ("short boot id", Box::new(|v| v["boot_id"] = "a5".into())),
            (
                "uppercase boot id",
                Box::new(|v| v["boot_id"] = "A5".repeat(BOOT_ID_BYTES).into()),
            ),
            (
                "boot id not hex",
                Box::new(|v| v["boot_id"] = "zz".repeat(BOOT_ID_BYTES).into()),
            ),
            ("boot id a number", Box::new(|v| v["boot_id"] = 7.into())),
        ];
        for (what, edit) in cases {
            assert_eq!(refused(&*edit), None, "{what}");
        }
        // As many pools as allowed, or none, are fine.
        let mut most = good.clone();
        let pool = most["gauges"]["pools"][0].clone();
        most["gauges"]["pools"] = Value::Array(vec![pool; MAX_POOLS]);
        assert!(NodeTelemetry::from_value(&most).is_some());
        let mut none = good;
        none["gauges"]["pools"] = json!([]);
        assert!(NodeTelemetry::from_value(&none).is_some());
    }

    #[test]
    fn deltas_are_growth_since_an_earlier_snapshot() {
        let earlier = sample().counters;
        let mut later = earlier;
        later.steps += 10;
        later.generated_tokens += 40;
        assert!(later.covers(&earlier));
        let delta = later.since(&earlier);
        assert_eq!(delta.steps, 10);
        assert_eq!(delta.generated_tokens, 40);
        assert_eq!(
            delta.entries().iter().map(|(_, v)| v).sum::<u64>(),
            50,
            "nothing else grew"
        );
        // A count that went down (a restart's) is not covered, and is no
        // growth.
        let restarted = NodeCounters::default();
        assert!(!restarted.covers(&earlier));
        assert_eq!(restarted.since(&earlier), NodeCounters::default());
    }

    #[test]
    fn occupancy_is_rounded_to_its_pool_steps() {
        // A pool of 1024 blocks reports in steps of 4.
        assert_eq!(quantize_occupancy(0, 1024), 0);
        assert_eq!(quantize_occupancy(1, 1024), 0);
        assert_eq!(quantize_occupancy(2, 1024), 4);
        assert_eq!(quantize_occupancy(5, 1024), 4);
        assert_eq!(quantize_occupancy(1024, 1024), 1024);
        // A pool of 100,000 blocks in steps of 391: one sequence of a few
        // hundred blocks is not told apart from none.
        assert_eq!(quantize_occupancy(150, 100_000), 0);
        assert_eq!(quantize_occupancy(200, 100_000), 391);
        // Never past the pool, and a small pool is exact.
        assert_eq!(quantize_occupancy(1000, 1000), 1000);
        assert_eq!(quantize_occupancy(999, 1000), 1000);
        assert_eq!(quantize_occupancy(7, 64), 7);
        assert_eq!(quantize_occupancy(0, 0), 0);
        // Quantized counts always fit their pool together.
        for total in [1u64, 63, 64, 255, 256, 257, 1000, 100_000] {
            for in_use in [
                0,
                1.min(total),
                total / 3,
                total / 2,
                total.saturating_sub(2),
                total,
            ] {
                let idle = total - in_use;
                let pool = PoolOccupancy::quantized(total, in_use, idle, total, in_use);
                assert_eq!(
                    PoolOccupancy::from_value(&pool.to_value()),
                    Some(pool),
                    "{total} {in_use}"
                );
            }
        }
    }

    #[test]
    fn prefix_token_counts_are_floored() {
        assert_eq!(quantize_prefix_tokens(0), 0);
        assert_eq!(quantize_prefix_tokens(PREFIX_TOKEN_QUANTUM - 1), 0);
        assert_eq!(
            quantize_prefix_tokens(PREFIX_TOKEN_QUANTUM),
            PREFIX_TOKEN_QUANTUM
        );
        assert_eq!(
            quantize_prefix_tokens(3 * PREFIX_TOKEN_QUANTUM + 5),
            3 * PREFIX_TOKEN_QUANTUM
        );
    }
}
