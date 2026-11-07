//! Writes the synthetic dev model (the committed fixture with its head widened to the
//! real tokenizer; see `tests/common/fixture.rs`) into a directory and prints its weights
//! hash, for serving it locally:
//!
//! ```sh
//! cargo run -p eidola-server-engine --example dev_model -- target/engine-dev-model
//! ```
//!
//! `just run engine-server` does this and starts the node on it.

#[path = "../tests/common/fixture.rs"]
mod fixture;

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: dev_model <output directory>");
    let out = std::path::Path::new(&out);
    fixture::ensure_dev_model(out).expect("write the dev model");
    println!("{}", fixture::weights_hash_of(out));
}
