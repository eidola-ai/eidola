//! Generate Argon2id hashes of secrets for measured secret verification.
//!
//! These hashes are committed in `tinfoil-config.yml` as `*_HASH` env vars,
//! binding injected secrets to the enclave measurement.
//!
//! Usage:
//!   cargo run -p hash-secret                    # reads from stdin
//!   cargo run -p hash-secret -- "my-secret"     # from argument
//!   echo "my-secret" | cargo run -p hash-secret # piped

use argon2::password_hash::SaltString;
use argon2::{Argon2, PasswordHasher};
use rand_core::OsRng;

fn main() {
    let secret = match std::env::args().nth(1) {
        Some(arg) => arg,
        None => {
            let mut input = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)
                .expect("failed to read stdin");
            input.trim_end().to_string()
        }
    };

    if secret.is_empty() {
        eprintln!("error: empty secret");
        std::process::exit(1);
    }

    let salt = SaltString::generate(&mut OsRng);
    let hash = Argon2::default()
        .hash_password(secret.as_bytes(), &salt)
        .expect("failed to hash secret");

    println!("{hash}");
}
