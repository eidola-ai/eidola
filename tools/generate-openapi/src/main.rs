//! Generate OpenAPI specification for the Eidolons API.
//!
//! This binary outputs the OpenAPI JSON specification to stdout.
//! It is used by the build system to generate the committed openapi.json file.

use eidolons_server::api_doc::ApiDoc;
use utoipa::OpenApi;

fn main() {
    let spec = ApiDoc::openapi()
        .to_pretty_json()
        .expect("Failed to serialize OpenAPI spec");
    println!("{}", spec);
}
