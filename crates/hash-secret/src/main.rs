//! Generate Argon2id hashes of secrets for measured secret verification.
//!
//! A hash committed in a deployment's `tinfoil-config.yml` as a `*_HASH` env
//! var binds the secret injected beside it to the enclave measurement: an
//! engine deployment measures `GATEWAY_TOKEN_HASH` this way.
//!
//! Usage:
//!   cargo run -p hash-secret                    # reads from stdin
//!   cargo run -p hash-secret -- "my-secret"     # from argument
//!   echo "my-secret" | cargo run -p hash-secret # piped

use argon2::{Argon2, PasswordHasher};

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

    let hash = Argon2::default()
        .hash_password(secret.as_bytes())
        .expect("failed to hash secret");

    println!("{hash}");
}
