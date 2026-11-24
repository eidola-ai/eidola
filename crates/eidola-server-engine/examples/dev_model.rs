//! Writes the synthetic dev model (the committed fixture with its head widened to the
//! real tokenizer; see `tests/common/fixture.rs`) into a directory and prints its weights
//! hash, for serving it locally:
//!
//! ```sh
//! cargo run -p eidola-server-engine --example dev_model -- target/engine-dev-model
//! cargo run -p eidola-server-engine --example dev_model -- --deepseek target/engine-dev-model-deepseek
//! cargo run -p eidola-server-engine --example dev_model -- --qwen target/engine-dev-model-qwen
//! cargo run -p eidola-server-engine --example dev_model -- --qwen35 target/engine-dev-model-qwen35
//! ```
//!
//! `just run engine-server` does the first and starts the node on it; `--deepseek`,
//! `--qwen` and `--qwen35` write the DeepSeek-V4.1, Qwen3.8 and Qwen3.5 dev models instead.

#[path = "../tests/common/fixture.rs"]
mod fixture;

fn main() {
    let usage = "usage: dev_model [--deepseek | --qwen | --qwen35] <output directory>";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let family = match args.first().map(String::as_str) {
        Some(flag @ ("--deepseek" | "--qwen" | "--qwen35")) => {
            let flag = flag.to_string();
            args.remove(0);
            Some(flag)
        }
        _ => None,
    };
    let [out] = args.as_slice() else {
        panic!("{usage}");
    };
    let out = std::path::Path::new(out);
    match family.as_deref() {
        Some("--deepseek") => fixture::ensure_deepseek_dev_model(out),
        Some("--qwen") => fixture::ensure_qwen_dev_model(out),
        Some("--qwen35") => fixture::ensure_qwen35_dev_model(out),
        _ => fixture::ensure_dev_model(out),
    }
    .expect("write the dev model");
    println!("{}", fixture::weights_hash_of(out));
}
