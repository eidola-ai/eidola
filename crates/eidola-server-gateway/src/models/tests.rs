//! The model catalog's format, its build-time checks (`file.rs`, the code
//! `build.rs` runs) over fixture files, and the reference syntax.

use eidola_common::architecture::Architecture;

use super::file::{self, Lifecycle, PinView, Row};
use super::{CatalogRow, ModelRefError, ModelReference};

const FIXTURE_PATH: &str = "models/fixture-family/flash.toml";
const FIXTURE: &str = include_str!("../../tests/fixtures/models/fixture-family/flash.toml");
const FIXTURE_SHA: &str = "a13d92b1d4bdf20dcffe052429ac50ca86b8423b8cea5682ae55f90b286072cf";

fn parse(text: &str) -> Result<Row, String> {
    file::parse(FIXTURE_PATH, text.as_bytes())
}

fn fixture() -> Row {
    parse(FIXTURE).unwrap()
}

/// A second, distinct row: the fixture with one shard's hash changed and the
/// aggregate recomputed.
fn other_row(tags: &[&str]) -> Row {
    let mut row = fixture();
    row.sha256 = "b".repeat(64);
    row.path = "models/fixture-family/other.toml".into();
    row.tags = tags.iter().map(|t| (*t).to_string()).collect();
    row
}

#[test]
fn the_fixture_parses_whole() {
    let row = fixture();
    assert_eq!(row.sha256, FIXTURE_SHA);
    assert_eq!(row.architecture, Architecture::MimoV26Flash);
    assert_eq!(row.tags, ["fixture-flash", "fixture-latest"]);
    assert_eq!(row.repo, "example/fixture-flash");
    assert_eq!(row.files.len(), 7);
    assert!(
        row.files
            .windows(2)
            .all(|w| w[0].path.as_bytes() < w[1].path.as_bytes())
    );
    assert_eq!(row.context_length, 1_048_576);
    assert_eq!(row.max_output_tokens, Some(32_768));
    assert_eq!(
        row.price,
        file::Price::PerToken {
            input_per_m: 1.80,
            output_per_m: 5.75
        }
    );
    assert_eq!(row.lifecycle, Lifecycle::Current);
}

/// The manifest's aggregate is the row's hash: the SHA-256 of
/// `<sha256>␠␠<path>\n` over the files sorted by path, exactly what the
/// engine computes over its weights directory and `sha256sum | sha256sum`
/// reproduces.
#[test]
fn the_hash_is_the_manifests_aggregate() {
    use sha2::{Digest, Sha256};
    let row = fixture();
    let manifest: String = row
        .files
        .iter()
        .map(|f| format!("{}  {}\n", f.sha256, f.path))
        .collect();
    let aggregate: String = Sha256::digest(manifest.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(aggregate, row.sha256);

    // Change one file's hash, or its name, and the aggregate no longer
    // matches; the build refuses.
    let first_file_sha = &row.files[0].sha256;
    let edited = FIXTURE.replacen(first_file_sha, &"0".repeat(64), 1);
    assert!(
        parse(&edited)
            .unwrap_err()
            .contains("manifest of files hashes to")
    );
    let wrong = FIXTURE.replace(FIXTURE_SHA, &"0".repeat(64));
    assert!(
        parse(&wrong)
            .unwrap_err()
            .contains("manifest of files hashes to")
    );
    // Sizes are published for checking a download, and are not hashed.
    let resized = FIXTURE.replacen("size = 9\n", "size = 10\n", 1);
    assert_ne!(resized, FIXTURE);
    assert!(parse(&resized).is_ok());
}

/// The fixture's files exist (`tests/fixtures/models/fixture-family/weights/`)
/// and hash, one by one and in aggregate, to what the row says: the same
/// directory `scripts/test-weights-hash.sh` runs the standalone recompute
/// script over, so the script and this code are held to one hash.
#[test]
fn the_fixtures_weights_directory_hashes_to_its_row() {
    use sha2::{Digest, Sha256};
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/models/fixture-family/weights");
    let row = fixture();
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    on_disk.sort();
    let mut listed: Vec<String> = row.files.iter().map(|f| f.path.clone()).collect();
    listed.sort();
    assert_eq!(on_disk, listed);
    let hex = |bytes: &[u8]| -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() };
    let mut manifest = String::new();
    for file in &row.files {
        let bytes = std::fs::read(dir.join(&file.path)).unwrap();
        assert_eq!(bytes.len() as u64, file.size, "{}", file.path);
        assert_eq!(hex(&Sha256::digest(&bytes)), file.sha256, "{}", file.path);
        manifest.push_str(&format!("{}  {}\n", file.sha256, file.path));
    }
    assert_eq!(hex(&Sha256::digest(manifest.as_bytes())), FIXTURE_SHA);
}

#[test]
fn the_file_set_is_what_the_engine_hashes() {
    // A file the engine does not read.
    let readme = format!(
        "{FIXTURE}\n[[files]]\npath = \"README.md\"\nsize = 1\nsha256 = \"{}\"\n",
        "1".repeat(64)
    );
    assert!(parse(&readme).unwrap_err().contains("README.md"));
    // A required file missing.
    let start = FIXTURE
        .find("[[files]]\npath = \"tokenizer.json\"")
        .unwrap();
    let end = start + FIXTURE[start..].find("\n\n").unwrap();
    let without = format!("{}{}", &FIXTURE[..start], &FIXTURE[end + 2..]);
    assert!(
        parse(&without)
            .unwrap_err()
            .contains("tokenizer.json is missing")
    );
}

#[test]
fn only_an_implemented_architecture_may_be_named() {
    let pro = FIXTURE.replace("\"MimoV26Flash\"", "\"MimoV26Pro\"");
    assert!(parse(&pro).unwrap_err().contains("not implemented"));
    let unknown = FIXTURE.replace("\"MimoV26Flash\"", "\"Llama\"");
    assert!(
        parse(&unknown)
            .unwrap_err()
            .contains("not one the engine knows")
    );
}

#[test]
fn tags_are_plain_names() {
    for bad in [
        "has@at",
        "has:colon",
        "has space",
        "",
        "a/b",
        &"x".repeat(65),
    ] {
        let text = FIXTURE.replace("\"fixture-latest\"", &format!("{bad:?}"));
        assert!(parse(&text).unwrap_err().contains("tag"), "{bad:?}");
    }
    for good in ["A.b_c-9", "mimo-v2.6-flash", &"x".repeat(64)] {
        let text = FIXTURE.replace("\"fixture-latest\"", &format!("{good:?}"));
        assert!(parse(&text).is_ok(), "{good:?}");
    }
    let twice = FIXTURE.replace("\"fixture-latest\"", "\"fixture-flash\"");
    assert!(parse(&twice).unwrap_err().contains("listed twice"));
}

#[test]
fn the_shape_is_strict() {
    let unknown_top = format!("extra = 1\n{FIXTURE}");
    assert!(parse(&unknown_top).is_err());
    let unknown_nested = FIXTURE.replace("[price]\n", "[price]\ndiscount = 0.5\n");
    assert!(parse(&unknown_nested).is_err());
    let repeated = FIXTURE.replace("[price]\n", "[price]\ninput_per_m = 2.0\n");
    assert!(parse(&repeated).is_err());
    // TOML integers only, as sizes and limits are.
    let float_size = FIXTURE.replacen("size = 9\n", "size = 9.0\n", 1);
    assert!(parse(&float_size).is_err());
}

#[test]
fn display_capabilities_and_price_are_complete() {
    let no_name = FIXTURE.replace("name = \"Fixture Flash\"", "name = \" \"");
    assert!(parse(&no_name).is_err());
    let no_window = FIXTURE.replace("context_length = 1048576", "context_length = 0");
    assert!(parse(&no_window).is_err());
    let ceiling = FIXTURE.replace("max_output_tokens = 32768", "max_output_tokens = 1048576");
    assert!(parse(&ceiling).unwrap_err().contains("max_output_tokens"));
    let no_text = FIXTURE.replace(
        "input_modalities = [\"text\"]",
        "input_modalities = [\"image\"]",
    );
    assert!(parse(&no_text).unwrap_err().contains("must include text"));
    let unknown_modality = FIXTURE.replace(
        "input_modalities = [\"text\"]",
        "input_modalities = [\"text\", \"video\"]",
    );
    assert!(parse(&unknown_modality).is_err());
    let class = FIXTURE.replace("\"reasoning\"\n", "\"extended\"\n");
    assert!(parse(&class).unwrap_err().contains("output_budget_class"));
    let negative = FIXTURE.replace("input_per_m = 1.80", "input_per_m = -1.0");
    assert!(parse(&negative).unwrap_err().contains("price"));
    let both = FIXTURE.replace("[price]\n", "[price]\nper_request = 0.01\n");
    assert!(parse(&both).unwrap_err().contains("price"));
    let per_request = FIXTURE.replace(
        "input_per_m = 1.80\noutput_per_m = 5.75",
        "per_request = 0.01",
    );
    assert_eq!(
        parse(&per_request).unwrap().price,
        file::Price::PerRequest { usd: 0.01 }
    );
    let bad_repo = FIXTURE.replace("example/fixture-flash", "fixture-flash");
    assert!(parse(&bad_repo).unwrap_err().contains("source.repo"));
    let bad_revision = FIXTURE.replace(&"4".repeat(40), &"4".repeat(39));
    assert!(
        parse(&bad_revision)
            .unwrap_err()
            .contains("source.revision")
    );
}

#[test]
fn the_lifecycle_is_consistent() {
    let status = |s: &str| FIXTURE.replace("status = \"current\"", s);
    assert_eq!(
        parse(&status(
            "status = \"deprecated\"\nretire_after = \"2027-02-28\""
        ))
        .unwrap()
        .lifecycle,
        Lifecycle::Deprecated {
            retire_after: "2027-02-28".into()
        }
    );
    assert!(parse(&status("status = \"deprecated\"")).is_err());
    assert!(
        parse(&status(
            "status = \"deprecated\"\nretire_after = \"2027-02-29\""
        ))
        .is_err()
    );
    assert!(
        parse(&status(
            "status = \"deprecated\"\nretire_after = \"2028-02-29\""
        ))
        .is_ok()
    );
    assert!(
        parse(&status(
            "status = \"current\"\nretire_after = \"2027-01-01\""
        ))
        .is_err()
    );
    assert_eq!(
        parse(&status("status = \"retired\"")).unwrap().lifecycle,
        Lifecycle::Retired { retire_after: None }
    );
    assert!(parse(&status("status = \"sunset\"")).is_err());
    for bad in [
        "2027-1-01",
        "2027-13-01",
        "2027-00-10",
        "27-01-01",
        "2027/01/01",
    ] {
        assert!(!file::is_date(bad), "{bad}");
    }
}

#[test]
fn a_catalog_file_lives_at_family_and_label() {
    assert!(file::is_catalog_path("models/mimo-v2.6-flash/base.toml"));
    for bad in [
        "models/flash.toml",
        "models/Mimo/base.toml",
        "models/mimo/base.json",
        "models/mimo/sub/base.toml",
        "other/mimo/base.toml",
        "models/mimo/.toml",
    ] {
        assert!(!file::is_catalog_path(bad), "{bad}");
        assert!(file::parse(bad, FIXTURE.as_bytes()).is_err(), "{bad}");
    }
}

/// Across files: a hash is described once, and a tag names exactly one hash.
#[test]
fn each_tag_names_one_hash_and_each_hash_one_row() {
    let row = fixture();
    assert!(file::check_catalog(&[row.clone(), other_row(&["other"])]).is_ok());
    let err =
        file::check_catalog(&[row.clone(), other_row(&["other", "fixture-latest"])]).unwrap_err();
    assert!(
        err.contains("fixture-latest") && err.contains("exactly one hash"),
        "{err}"
    );
    let mut same_hash = other_row(&["other"]);
    same_hash.sha256 = row.sha256.clone();
    assert!(
        file::check_catalog(&[row, same_hash])
            .unwrap_err()
            .contains("listed once")
    );
}

/// Rows and engine pins bind one to one by hash.
#[test]
fn rows_and_pins_bind_one_to_one() {
    let row = fixture();
    let pin = PinView {
        sha256: FIXTURE_SHA,
        architecture: Architecture::MimoV26Flash,
        repo: "example/fixture-flash",
        revision: "4444444444444444444444444444444444444444",
    };
    assert!(file::check_binding(&[], &[]).is_ok());
    assert!(file::check_binding(std::slice::from_ref(&row), &[pin]).is_ok());

    // A served row without its pin; a pin without a row.
    assert!(
        file::check_binding(std::slice::from_ref(&row), &[])
            .unwrap_err()
            .contains("pins no deployment")
    );
    assert!(
        file::check_binding(&[], &[pin])
            .unwrap_err()
            .contains("no served row")
    );
    // A retired row is not pinned, and a pin cannot be for one.
    let mut retired = row.clone();
    retired.lifecycle = Lifecycle::Retired { retire_after: None };
    assert!(file::check_binding(std::slice::from_ref(&retired), &[]).is_ok());
    assert!(
        file::check_binding(std::slice::from_ref(&retired), &[pin])
            .unwrap_err()
            .contains("retired")
    );
    // The pin's sidecar and the row agree on architecture and source.
    let other_arch = PinView {
        architecture: Architecture::MimoV26Pro,
        ..pin
    };
    assert!(
        file::check_binding(std::slice::from_ref(&row), &[other_arch])
            .unwrap_err()
            .contains("deployment.json")
    );
    let other_source = PinView {
        revision: "5555555555555555555555555555555555555555",
        ..pin
    };
    assert!(
        file::check_binding(std::slice::from_ref(&row), &[other_source])
            .unwrap_err()
            .contains("source")
    );
}

/// The committed catalog is what `build.rs` checked: every row's hash is
/// its manifest's aggregate, and its tags parse as references.
#[test]
fn the_compiled_catalog_holds_its_invariants() {
    for row in super::MODEL_CATALOG {
        assert!(row.architecture.is_implemented(), "{}", row.sha256);
        for tag in row.tags {
            assert_eq!(
                ModelReference::parse(tag),
                Ok(ModelReference::Follow { tag })
            );
        }
    }
}

#[test]
fn a_reference_is_a_tag_a_pin_or_a_hash() {
    let hash = "a".repeat(64);
    assert_eq!(
        ModelReference::parse("mimo-v2.6-flash"),
        Ok(ModelReference::Follow {
            tag: "mimo-v2.6-flash"
        })
    );
    assert_eq!(
        ModelReference::parse(&format!("mimo-v2.6-flash@sha256:{hash}")),
        Ok(ModelReference::Pin {
            tag: "mimo-v2.6-flash",
            sha256: &hash
        })
    );
    assert_eq!(
        ModelReference::parse(&format!("sha256:{hash}")),
        Ok(ModelReference::Hash { sha256: &hash })
    );
    // Only full, lowercase digests, and only these forms.
    for bad in [
        String::new(),
        "tag@".into(),
        "@sha256:".to_string() + &hash,
        format!("tag@sha256:{}", &hash[..63]),
        format!("tag@sha256:{}", hash.to_uppercase()),
        format!("tag@sha512:{hash}"),
        format!("tag@{hash}"),
        format!("sha256:{}", &hash[..12]),
        format!("tag@sha256:{hash}@eidola"),
        "has space".into(),
        "a:b".into(),
    ] {
        assert_eq!(
            ModelReference::parse(&bad),
            Err(ModelRefError::Malformed),
            "{bad:?}"
        );
    }
}

/// The typed refusals: a fixed message each, never the reference.
#[test]
fn refusals_are_typed_and_say_nothing_of_the_reference() {
    for (e, status, kind) in [
        (ModelRefError::Malformed, 400, "invalid_model_reference"),
        (ModelRefError::Unknown, 404, "model_not_found"),
        (ModelRefError::Retired, 410, "model_retired"),
    ] {
        assert_eq!((e.status(), e.error_type()), (status, kind));
        let server = crate::error::ServerError::ModelReference(e);
        assert_eq!(server.status_code().as_u16(), status);
        assert_eq!(server.to_error_response().error.error_type, kind);
    }
}

const ROWS: &[CatalogRow] = &[
    CatalogRow {
        tags: &["flash", "flash-latest"],
        ..crate::backend::tests::FIXTURE_CATALOG[0]
    },
    CatalogRow {
        tags: &["flash-previous"],
        ..crate::backend::tests::FIXTURE_CATALOG[1]
    },
    CatalogRow {
        tags: &["gone"],
        ..crate::backend::tests::FIXTURE_CATALOG[2]
    },
];

/// The resolution rule: a bare tag follows; in a reference with a digest
/// the digest is authoritative and the tag only a label; a retired row is
/// refused however it is named.
#[test]
fn references_resolve_by_the_documented_rule() {
    let current = ROWS[0].sha256;
    let previous = ROWS[1].sha256;
    let gone = ROWS[2].sha256;
    let resolve = |r: &str| super::resolve(ROWS, r).map(|(row, echo)| (row.sha256, echo));

    assert_eq!(
        resolve("flash-latest"),
        Ok((current, format!("flash-latest@sha256:{current}")))
    );
    assert_eq!(
        resolve(&format!("flash@sha256:{current}")),
        Ok((current, format!("flash@sha256:{current}")))
    );
    // The tag resolves elsewhere now, or never named that hash at all: the
    // digest decides, and the tag is echoed as the client sent it.
    assert_eq!(
        resolve(&format!("flash@sha256:{previous}")),
        Ok((previous, format!("flash@sha256:{previous}")))
    );
    assert_eq!(
        resolve(&format!("nothing@sha256:{current}")),
        Ok((current, format!("nothing@sha256:{current}")))
    );
    assert_eq!(
        resolve(&format!("sha256:{previous}")),
        Ok((previous, format!("sha256:{previous}")))
    );
    // Retired, by tag, pin or hash.
    assert_eq!(resolve("gone"), Err(ModelRefError::Retired));
    assert_eq!(
        resolve(&format!("gone@sha256:{gone}")),
        Err(ModelRefError::Retired)
    );
    assert_eq!(
        resolve(&format!("sha256:{gone}")),
        Err(ModelRefError::Retired)
    );
    // Unknown.
    assert_eq!(resolve("nothing"), Err(ModelRefError::Unknown));
    assert_eq!(
        resolve(&format!("flash@sha256:{}", "e".repeat(64))),
        Err(ModelRefError::Unknown)
    );
    // A label past the tag bound is malformed, so an echo stays bounded.
    assert_eq!(
        resolve(&format!("{}@sha256:{current}", "x".repeat(65))),
        Err(ModelRefError::Malformed)
    );
    assert_eq!(
        resolve(&format!("sha256:{}", "e".repeat(64))),
        Err(ModelRefError::Unknown)
    );
    assert_eq!(resolve("bad@ref"), Err(ModelRefError::Malformed));
}

#[test]
fn a_listed_id_never_holds_an_at_sign() {
    for row in ROWS {
        assert!(!row.list_id().contains('@'));
        assert!(row.pinned_reference().ends_with(row.sha256));
    }
    let untagged = CatalogRow {
        tags: &[],
        ..ROWS[0]
    };
    assert_eq!(untagged.list_id(), format!("sha256:{}", ROWS[0].sha256));
    assert_eq!(
        untagged.pinned_reference(),
        format!("sha256:{}", ROWS[0].sha256)
    );
}

/// The runtime's tag grammar is the one the build checks rows with.
#[test]
fn the_runtime_and_the_build_read_tags_alike() {
    assert_eq!(super::MAX_TAG_LEN, file::MAX_TAG_LEN);
    let long = "x".repeat(65);
    for tag in [
        "a",
        "A.b_c-9",
        "mimo-v2.6-flash",
        "",
        "a@b",
        "a:b",
        "a b",
        "a/b",
        "é",
        &long,
    ] {
        assert_eq!(super::is_tag(tag), file::is_tag(tag), "{tag:?}");
    }
}
