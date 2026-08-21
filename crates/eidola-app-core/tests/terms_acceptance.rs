//! Terms acceptance: what is *submitted*, and what is *reported*.
//!
//! Two contracts meet here, and they pull in opposite directions.
//!
//! **Submission takes the snapshot that was presented.** Consent is obtained
//! by a UI showing the user a set of documents; app-core only transmits it. So
//! `AppCore::accept_terms` and `AppCore::account_create` take the snapshot as
//! an argument instead of re-reading `GET /v1/terms` on the way out — a
//! version that advanced between the two would otherwise be recorded as
//! accepted without anyone having read it.
//!
//! **The report comes from reading the server back.** Acceptance is recorded
//! one document at a time and each submission is judged alone, so "every
//! document I submitted was accepted" is silent about a document that became
//! required *after* the snapshot was taken — most starkly when the snapshot
//! was empty because the gate was not switched on yet. The purchase gate asks
//! a different question, and `TermsAcceptance` is the answer to that one.
//!
//! The read that reports is not the read that was removed: it happens after
//! the submission, and its answer is document **names**, which carry no hash
//! and so can never be submitted.
//!
//! The mock reproduces the server's own rules — `POST /v1/account/terms`
//! records a `(document, sha256)` pair only while it is still required and
//! answers a stale one with 409 — so what these tests pin is the client half.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use eidola_app_core::{AppCore, TermsAcceptance, TermsDocument};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const ACCOUNT_ID: &str = "00000000-0000-0000-0000-000000000001";
const TOS: &str = "terms_of_service";
const PRIVACY: &str = "privacy_policy";

/// The published hash of a document at a given version. Any stable
/// `(document, version) -> hash` function does; the tests only need distinct
/// documents and distinct versions to carry distinct hashes.
fn sha_for(document: &str, version: i64) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in document.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:032x}{version:032x}")
}

/// The server's required-document table, as `(document, version)`.
type Required = Arc<std::sync::Mutex<Vec<(String, i64)>>>;

/// `GET /v1/terms` — answers the current required set, counts how many times
/// it was asked, and can be made to fail on cue.
struct CurrentTerms {
    required: Required,
    hits: Arc<AtomicU64>,
    broken: Arc<AtomicBool>,
}

impl Respond for CurrentTerms {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        self.hits.fetch_add(1, Ordering::SeqCst);
        if self.broken.load(Ordering::SeqCst) {
            return ResponseTemplate::new(500).set_body_string("terms feed unavailable");
        }
        let documents: Vec<serde_json::Value> = self
            .required
            .lock()
            .unwrap()
            .iter()
            .map(|(document, version)| {
                serde_json::json!({
                    "document": document,
                    "version": version,
                    "url": format!("https://example.invalid/{document}/"),
                    "sha256": sha_for(document, *version),
                })
            })
            .collect();
        ResponseTemplate::new(200).set_body_json(serde_json::json!({ "documents": documents }))
    }
}

/// `POST /v1/account/terms` — the server's rule: record the pair only while it
/// is a currently required one, 409 otherwise. Every submitted pair is noted
/// either way, so a test can see what the client actually sent.
struct AcceptTerms {
    required: Required,
    submitted: Arc<std::sync::Mutex<Vec<(String, String)>>>,
    recorded: Arc<std::sync::Mutex<Vec<(String, String)>>>,
}

impl Respond for AcceptTerms {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let body: serde_json::Value = serde_json::from_slice(&req.body).expect("json body");
        let document = body["document"].as_str().unwrap_or_default().to_string();
        let sha = body["sha256"].as_str().unwrap_or_default().to_string();
        self.submitted
            .lock()
            .unwrap()
            .push((document.clone(), sha.clone()));
        let is_current = self
            .required
            .lock()
            .unwrap()
            .iter()
            .any(|(d, v)| *d == document && sha_for(d, *v) == sha);
        if is_current {
            self.recorded.lock().unwrap().push((document, sha));
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
    required: Required,
    /// `GET /v1/terms` request count.
    terms_hits: Arc<AtomicU64>,
    /// Makes `GET /v1/terms` fail, so the read-back has nothing to report.
    terms_broken: Arc<AtomicBool>,
    submitted: Arc<std::sync::Mutex<Vec<(String, String)>>>,
    recorded: Arc<std::sync::Mutex<Vec<(String, String)>>>,
}

impl Harness {
    /// A server whose gate requires exactly `required`.
    fn start(required: &[(&str, i64)]) -> Self {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let dir = tempfile::tempdir().expect("tempdir");
        let client = reqwest::Client::builder().build().expect("plain client");
        let core = AppCore::with_test_http_client(
            dir.path().to_path_buf(),
            dir.path().join("data"),
            client,
        )
        .expect("open core");

        let required: Required = Arc::new(std::sync::Mutex::new(
            required
                .iter()
                .map(|(d, v)| ((*d).to_string(), *v))
                .collect(),
        ));
        let terms_hits = Arc::new(AtomicU64::new(0));
        let terms_broken = Arc::new(AtomicBool::new(false));
        let submitted = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));

        let server = core.runtime().block_on({
            let required = required.clone();
            let terms_hits = terms_hits.clone();
            let terms_broken = terms_broken.clone();
            let submitted = submitted.clone();
            let recorded = recorded.clone();
            async move {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(path("/v1/terms"))
                    .respond_with(CurrentTerms {
                        required: required.clone(),
                        hits: terms_hits,
                        broken: terms_broken,
                    })
                    .mount(&server)
                    .await;
                Mock::given(method("POST"))
                    .and(path("/v1/account/terms"))
                    .respond_with(AcceptTerms {
                        required: required.clone(),
                        submitted,
                        recorded,
                    })
                    .mount(&server)
                    .await;
                Mock::given(method("POST"))
                    .and(path("/v1/account"))
                    .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                        "account_id": ACCOUNT_ID,
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
            required,
            terms_hits,
            terms_broken,
            submitted,
            recorded,
        }
    }

    fn with_credentials(self) -> Self {
        self.core
            .set_account_credentials(ACCOUNT_ID.into(), "mock-secret".into())
            .expect("configure credentials");
        self
    }

    /// Replace the server's required set — a publish, mid-flow.
    fn now_requires(&self, required: &[(&str, i64)]) {
        *self.required.lock().unwrap() = required
            .iter()
            .map(|(d, v)| ((*d).to_string(), *v))
            .collect();
    }

    fn current_terms(&self) -> Vec<TermsDocument> {
        self.core
            .runtime()
            .block_on(self.core.current_terms())
            .expect("fetch current terms")
    }

    fn accept(&self, presented: Vec<TermsDocument>) -> Result<TermsAcceptance, String> {
        self.core
            .runtime()
            .block_on(self.core.accept_terms(presented))
            .map_err(|e| e.to_string())
    }

    fn create_account(&self, presented: Vec<TermsDocument>) -> TermsAcceptance {
        self.core
            .runtime()
            .block_on(self.core.account_create(presented))
            .expect("the account is created regardless — acceptance is best-effort")
            .terms
    }

    fn terms_hits(&self) -> u64 {
        self.terms_hits.load(Ordering::SeqCst)
    }

    /// `(document, sha256)` pairs that went on the wire.
    fn submitted(&self) -> Vec<(String, String)> {
        self.submitted.lock().unwrap().clone()
    }

    /// The subset the server actually recorded.
    fn recorded(&self) -> Vec<(String, String)> {
        self.recorded.lock().unwrap().clone()
    }
}

fn outstanding(standing: &TermsAcceptance) -> Vec<String> {
    match standing {
        TermsAcceptance::Outstanding { documents } => documents.clone(),
        other => panic!("expected Outstanding, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// What is submitted: the presented snapshot, never a fresh read
// ---------------------------------------------------------------------------

#[test]
fn acceptance_submits_the_presented_snapshot_not_a_fresh_read() {
    let h = Harness::start(&[(TOS, 1)]).with_credentials();

    // What a consent screen fetched and showed.
    let presented = h.current_terms();
    assert_eq!(presented.len(), 1);
    assert_eq!(presented[0].sha256, sha_for(TOS, 1));
    let fetches_after_presenting = h.terms_hits();

    // A new version is published while the user is reading.
    h.now_requires(&[(TOS, 2)]);

    // Accepting must speak about version 1 — the text that was on screen —
    // and must therefore be refused, because the server no longer requires it.
    let err = h
        .accept(presented)
        .expect_err("a stale snapshot must be refused, not silently upgraded");
    assert!(
        err.contains("not a currently required version"),
        "the server's conflict must reach the caller: {err}"
    );

    assert_eq!(
        h.submitted(),
        vec![(TOS.to_string(), sha_for(TOS, 1))],
        "the pair on the wire is the one that was presented"
    );
    assert!(
        h.recorded().is_empty(),
        "nothing may be recorded as accepted when the presented version went stale"
    );
    assert_eq!(
        h.terms_hits(),
        fetches_after_presenting,
        "nothing is read on the way to the submission — a read there would \
         substitute unseen text for the snapshot the user agreed to. (The \
         read-back that reports the standing runs only after a submission \
         succeeds, and this one did not.)"
    );
}

#[test]
fn account_creation_records_only_what_the_caller_presented() {
    let h = Harness::start(&[(TOS, 1)]);

    // The consent screen's snapshot, then a publish while it was on screen.
    let presented = h.current_terms();
    h.now_requires(&[(TOS, 2)]);

    let standing = h.create_account(presented);

    assert_eq!(
        outstanding(&standing),
        vec![TOS.to_string()],
        "the account exists, but its acceptance was refused as stale — and the \
         caller is told what is outstanding rather than left to assume it landed"
    );
    assert_eq!(
        h.submitted(),
        vec![(TOS.to_string(), sha_for(TOS, 1))],
        "creation submits the presented snapshot, never a fresh read"
    );
    assert!(
        h.recorded().is_empty(),
        "no acceptance may be recorded for text the user was never shown"
    );
}

#[test]
fn account_creation_records_the_snapshot_when_it_is_still_current() {
    let h = Harness::start(&[(TOS, 1)]);

    let presented = h.current_terms();
    let standing = h.create_account(presented);

    assert!(
        standing.is_complete(),
        "the ordinary path records and reads back clean: {standing:?}"
    );
    assert_eq!(h.recorded(), vec![(TOS.to_string(), sha_for(TOS, 1))]);
}

#[test]
fn an_empty_snapshot_submits_nothing() {
    // A server with no acceptance gate: `current_terms` answers empty, so
    // there is nothing to put on the wire.
    let h = Harness::start(&[]).with_credentials();

    let standing = h.accept(Vec::new()).expect("an empty snapshot is a no-op");
    assert!(standing.is_complete(), "nothing required, nothing missing");
    assert!(h.submitted().is_empty());
}

// ---------------------------------------------------------------------------
// What is reported: the standing, read back from the server
// ---------------------------------------------------------------------------

#[test]
fn a_document_added_after_the_snapshot_is_reported_outstanding() {
    // Every submission in the snapshot succeeds — so a report assembled from
    // those per-document outcomes reads as complete while the account has
    // never so much as been shown the new document, and the next purchase is
    // the first anyone hears of it.
    let h = Harness::start(&[(TOS, 1)]);
    let presented = h.current_terms();
    h.now_requires(&[(TOS, 1), (PRIVACY, 1)]);

    let standing = h.create_account(presented);

    assert_eq!(
        h.recorded(),
        vec![(TOS.to_string(), sha_for(TOS, 1))],
        "the presented document really was accepted"
    );
    assert_eq!(
        outstanding(&standing),
        vec![PRIVACY.to_string()],
        "and the one that appeared afterwards is named, not silently omitted"
    );
}

#[test]
fn a_gate_switched_on_after_an_empty_snapshot_is_reported_outstanding() {
    // The starkest case: nothing was submitted at all, so there is no
    // per-document outcome to be right about, and every submission
    // "succeeded" vacuously.
    let h = Harness::start(&[]).with_credentials();
    let presented = h.current_terms();
    assert!(presented.is_empty());

    h.now_requires(&[(TOS, 1)]);

    let standing = h
        .accept(presented)
        .expect("nothing to submit, nothing fails");
    assert!(h.submitted().is_empty());
    assert_eq!(
        outstanding(&standing),
        vec![TOS.to_string()],
        "an empty submission that succeeded says nothing about a gate that \
         opened since"
    );
}

#[test]
fn accepting_everything_currently_required_reports_complete() {
    let h = Harness::start(&[(TOS, 1), (PRIVACY, 2)]).with_credentials();
    let presented = h.current_terms();
    let before = h.terms_hits();

    let standing = h.accept(presented).expect("accept");

    assert!(standing.is_complete(), "{standing:?}");
    assert_eq!(
        h.terms_hits(),
        before + 1,
        "the standing is established by reading the server back, once, after \
         the submission — not inferred from the submissions succeeding"
    );
    assert_eq!(h.recorded().len(), 2);
}

#[test]
fn a_standing_that_cannot_be_read_is_unknown_not_complete() {
    // "I could not check" is an answer. Reporting completeness here would be
    // the same over-claim in a different costume.
    let h = Harness::start(&[(TOS, 1)]).with_credentials();
    let presented = h.current_terms();

    h.terms_broken.store(true, Ordering::SeqCst);
    let standing = h
        .accept(presented)
        .expect("the submission itself succeeded");

    match &standing {
        TermsAcceptance::Unknown { .. } => {}
        other => panic!("expected Unknown, got {other:?}"),
    }
    assert!(
        !standing.is_complete(),
        "an unestablished standing is not a complete one"
    );
    assert_eq!(h.recorded().len(), 1, "the acceptance itself did land");
}

#[test]
fn a_failed_submission_leaves_everything_required_outstanding() {
    // Nothing is known to have been recorded, so nothing is assumed: the
    // standing is read with an empty recorded set. It may over-list — which
    // is the safe direction, since re-accepting is a no-op and silence is a
    // purchase that fails later.
    let h = Harness::start(&[(TOS, 1)]);
    let presented = h.current_terms();
    h.now_requires(&[(TOS, 2), (PRIVACY, 1)]);

    let standing = h.create_account(presented);

    assert!(h.recorded().is_empty(), "the stale pair was refused");
    assert_eq!(
        outstanding(&standing),
        vec![TOS.to_string(), PRIVACY.to_string()],
    );
}
