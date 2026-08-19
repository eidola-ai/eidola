//! Terms acceptance transmits *the snapshot that was presented*.
//!
//! Consent is obtained by a UI showing the user a set of documents; app-core
//! only transmits it. The two must describe the same bytes, which is why
//! `AppCore::accept_terms` and `AppCore::account_create` take the snapshot as
//! an argument instead of re-reading `GET /v1/terms` on the way out: a
//! document version that advances between the two would otherwise be recorded
//! as accepted without anyone having read it.
//!
//! The server side of the contract (`eidola-server`'s `POST /v1/account/terms`)
//! records a `(document, sha256)` pair only while that pair is still required
//! and answers a stale one with 409 Conflict. The mock here reproduces exactly
//! that rule, so what these tests pin is the client half: which pair goes on
//! the wire, and what the caller is told when the server refuses it.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use eidola_app_core::{AppCore, TermsDocument};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// The published hash of `terms_of_service` at a given version. Any stable
/// version→hash function does; these tests only care that two versions carry
/// two different hashes.
fn sha_for_version(version: i64) -> String {
    format!("{version:064x}")
}

/// `GET /v1/terms` — answers whatever version the test has advanced to, and
/// counts how many times it was asked.
struct CurrentTerms {
    version: Arc<AtomicI64>,
    hits: Arc<AtomicU64>,
}

impl Respond for CurrentTerms {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        self.hits.fetch_add(1, Ordering::SeqCst);
        let version = self.version.load(Ordering::SeqCst);
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "documents": [{
                "document": "terms_of_service",
                "version": version,
                "url": "https://example.invalid/terms/",
                "sha256": sha_for_version(version),
            }]
        }))
    }
}

/// `POST /v1/account/terms` — the server's rule: record the pair only while
/// it is the currently required one, 409 otherwise. Every submitted hash is
/// recorded either way, so a test can see what the client actually sent.
struct AcceptTerms {
    version: Arc<AtomicI64>,
    submitted: Arc<std::sync::Mutex<Vec<String>>>,
    recorded: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Respond for AcceptTerms {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let body: serde_json::Value = serde_json::from_slice(&req.body).expect("json body");
        let sha = body["sha256"].as_str().unwrap_or_default().to_string();
        self.submitted.lock().unwrap().push(sha.clone());
        if sha == sha_for_version(self.version.load(Ordering::SeqCst)) {
            self.recorded.lock().unwrap().push(sha);
            ResponseTemplate::new(204)
        } else {
            ResponseTemplate::new(409).set_body_json(serde_json::json!({
                "error": {
                    "message": "the submitted document/hash pair is not a currently \
                                required version",
                    "type": "conflict",
                }
            }))
        }
    }
}

struct Harness {
    core: AppCore,
    _server: MockServer,
    _dir: tempfile::TempDir,
    /// The version `GET /v1/terms` currently answers with.
    version: Arc<AtomicI64>,
    /// `GET /v1/terms` request count.
    terms_hits: Arc<AtomicU64>,
    /// Hashes the client submitted to `POST /v1/account/terms`, in order.
    submitted: Arc<std::sync::Mutex<Vec<String>>>,
    /// Hashes the server actually recorded (the subset it still required).
    recorded: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Harness {
    fn start() -> Self {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let dir = tempfile::tempdir().expect("tempdir");
        let client = reqwest::Client::builder().build().expect("plain client");
        let core = AppCore::with_test_http_client(
            dir.path().to_path_buf(),
            dir.path().join("data"),
            client,
        )
        .expect("open core");

        let version = Arc::new(AtomicI64::new(1));
        let terms_hits = Arc::new(AtomicU64::new(0));
        let submitted = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));

        let server = core.runtime().block_on({
            let version = version.clone();
            let terms_hits = terms_hits.clone();
            let submitted = submitted.clone();
            let recorded = recorded.clone();
            async move {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(path("/v1/terms"))
                    .respond_with(CurrentTerms {
                        version: version.clone(),
                        hits: terms_hits,
                    })
                    .mount(&server)
                    .await;
                Mock::given(method("POST"))
                    .and(path("/v1/account/terms"))
                    .respond_with(AcceptTerms {
                        version: version.clone(),
                        submitted,
                        recorded,
                    })
                    .mount(&server)
                    .await;
                Mock::given(method("POST"))
                    .and(path("/v1/account"))
                    .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                        "account_id": "00000000-0000-0000-0000-000000000001",
                        "secret": "mock-secret",
                        "created_at": "2025-01-01T00:00:00Z",
                    })))
                    .mount(&server)
                    .await;
                server
            }
        });
        core.runtime()
            .block_on(core.set_base_url(server.uri()))
            .expect("set base url");

        Self {
            core,
            _server: server,
            _dir: dir,
            version,
            terms_hits,
            submitted,
            recorded,
        }
    }

    /// Publish a new version of the document — the mid-flow advance the whole
    /// contract exists to survive.
    fn advance_to_version(&self, version: i64) {
        self.version.store(version, Ordering::SeqCst);
    }

    fn current_terms(&self) -> Vec<TermsDocument> {
        self.core
            .runtime()
            .block_on(self.core.current_terms())
            .expect("fetch current terms")
    }

    fn submitted(&self) -> Vec<String> {
        self.submitted.lock().unwrap().clone()
    }

    fn recorded(&self) -> Vec<String> {
        self.recorded.lock().unwrap().clone()
    }
}

#[test]
fn acceptance_submits_the_presented_snapshot_not_a_fresh_read() {
    let h = Harness::start();
    h.core
        .set_account_credentials(
            "00000000-0000-0000-0000-000000000001".into(),
            "mock-secret".into(),
        )
        .expect("configure credentials");

    // What a consent screen fetched and showed.
    let presented = h.current_terms();
    assert_eq!(presented.len(), 1);
    assert_eq!(presented[0].sha256, sha_for_version(1));
    let fetches_after_presenting = h.terms_hits.load(Ordering::SeqCst);

    // A new version is published while the user is reading.
    h.advance_to_version(2);

    // Accepting must speak about version 1 — the text that was on screen —
    // and must therefore be refused, because the server no longer requires it.
    let err = h
        .core
        .runtime()
        .block_on(h.core.accept_terms(presented))
        .expect_err("a stale snapshot must be refused, not silently upgraded");
    assert!(
        format!("{err}").contains("not a currently required version"),
        "the server's conflict must reach the caller: {err}"
    );

    assert_eq!(
        h.submitted(),
        vec![sha_for_version(1)],
        "the hash on the wire is the one that was presented"
    );
    assert!(
        h.recorded().is_empty(),
        "nothing may be recorded as accepted when the presented version went stale"
    );
    assert_eq!(
        h.terms_hits.load(Ordering::SeqCst),
        fetches_after_presenting,
        "accepting must not re-read the current documents — that read is what \
         would substitute unseen text for the snapshot the user agreed to"
    );
}

#[test]
fn account_creation_records_only_what_the_caller_presented() {
    let h = Harness::start();

    // The consent screen's snapshot, then a publish while it was on screen.
    let presented = h.current_terms();
    h.advance_to_version(2);

    let created = h
        .core
        .runtime()
        .block_on(h.core.account_create(presented))
        .expect("the account is still created — acceptance is best-effort");

    assert!(
        !created.terms_recorded,
        "the account exists, but its acceptance was refused as stale — and the \
         caller is told so rather than left to assume it landed"
    );
    assert_eq!(
        h.submitted(),
        vec![sha_for_version(1)],
        "creation submits the presented snapshot, never a fresh read"
    );
    assert!(
        h.recorded().is_empty(),
        "no acceptance may be recorded for text the user was never shown"
    );
}

#[test]
fn account_creation_records_the_snapshot_when_it_is_still_current() {
    let h = Harness::start();

    let presented = h.current_terms();
    let created = h
        .core
        .runtime()
        .block_on(h.core.account_create(presented))
        .expect("create account");

    assert!(created.terms_recorded, "the ordinary path still records");
    assert_eq!(h.recorded(), vec![sha_for_version(1)]);
}

#[test]
fn an_empty_snapshot_sends_nothing_and_counts_as_recorded() {
    // A server with no acceptance gate configured: `current_terms` answers
    // empty, so there is nothing to submit and nothing to be stale about.
    let h = Harness::start();
    h.core
        .set_account_credentials(
            "00000000-0000-0000-0000-000000000001".into(),
            "mock-secret".into(),
        )
        .expect("configure credentials");

    h.core
        .runtime()
        .block_on(h.core.accept_terms(Vec::new()))
        .expect("an empty snapshot is a no-op, not an error");
    assert!(h.submitted().is_empty());
    assert_eq!(
        h.terms_hits.load(Ordering::SeqCst),
        0,
        "and it does not fetch anything to decide that"
    );
}
