//! Serving core: executor interface, scheduler, paged KV cache, prefix cache, and speculative decoding.
//!
//! * [`executor`] — the seam between the host engine and whatever runs the model.
//! * [`spec`] — the model/memory description an executor reports.
//! * [`kv`] — paged KV manager and salted prefix cache.
//! * [`engine`] — the scheduler and request lifecycle.
//! * [`sampling`] — CPU reference sampling and speculative acceptance.
//! * [`secret`], [`hash`] — cache keys, salts, and the block-hash chain.
//! * [`mock`] — a deterministic mock executor for tests and simulation.
//!
//! Doctrine lives in this crate's `AGENTS.md`.

pub mod engine;
pub mod executor;
pub mod hash;
pub mod kv;
pub mod mock;
pub mod sampling;
pub mod secret;
pub mod spec;
