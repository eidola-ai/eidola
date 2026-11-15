//! Writes the synthetic dev model (the committed fixture with its head widened to the
//! real tokenizer; see `tests/common/fixture.rs`) into a directory and prints its weights
//! hash, for serving it locally:
//!
//! ```sh
//! cargo run -p eidola-server-engine --example dev_model -- target/engine-dev-model
//! cargo run -p eidola-server-engine --example dev_model -- --deepseek target/engine-dev-model-deepseek
//! ```
//!
//! `just run engine-server` does the first and starts the node on it; `--deepseek`
//! writes the DeepSeek-V4.1 dev model instead.

#[path = "../tests/common/fixture.rs"]
mod fixture;

fn main() {
    let usage = "usage: dev_model [--deepseek] <output directory>";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let deepseek = args.first().is_some_and(|a| a == "--deepseek");
    if deepseek {
        args.remove(0);
    }
    let [out] = args.as_slice() else {
        panic!("{usage}");
    };
    let out = std::path::Path::new(out);
    if deepseek {
        fixture::ensure_deepseek_dev_model(out).expect("write the dev model");
    } else {
        fixture::ensure_dev_model(out).expect("write the dev model");
    }
    println!("{}", fixture::weights_hash_of(out));
}
