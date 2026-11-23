//! Eidola's mobile app: `eidola-app-core` compiled to wasm32, hosted by the
//! native shells (`mobile/`) and reaching everything outside the webview
//! through the shell contract (`mobile/shell-contract.md`).
//!
//! - [`protocol`]: the contract's wire format and typed errors.
//! - [`client`]: a typed call for every method, over a [`client::Transport`].
//! - [`socket`]: the TCP pipe as app-core's [`tinfoil_verifier::pipe::Dialer`].
//! - [`files`]: the file directory as app-core's
//!   [`eidola_app_core::storage::HostStorage`].
//! - [`boot`]: hello, the profile, the core, and `bundle.confirm`.
//! - [`update`]: the shell's bundle store as app-core's
//!   [`eidola_app_core::updater::mobile::BundleStore`], and the self-update
//!   on it.
//! - `web` (wasm32): the webview's bridge, and the bundle's entry point.
//!
//! Everything but `web` is target-independent, so the native tests drive it
//! against a shell written in Rust.

pub mod boot;
pub mod client;
pub mod files;
pub mod protocol;
#[cfg(feature = "self-test")]
pub mod selftest;
pub mod socket;
pub mod update;
#[cfg(target_family = "wasm")]
mod web;
