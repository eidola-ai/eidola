//! Generate OpenAPI specification for the Eidola API.
//!
//! This binary outputs the OpenAPI JSON specification to stdout.
//! It is used by the build system to generate the committed openapi.json file.

fn main() {
    // Must install the crypto provider before constructing RedPillBackend (reqwest needs it).
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());

    // Build the full router so OpenApiRouter collects paths from handler annotations.
    let (_, spec) = eidola_server::build_router().split_for_parts();

    let json = spec
        .to_pretty_json()
        .expect("Failed to serialize OpenAPI spec");
    println!("{}", json);
}
