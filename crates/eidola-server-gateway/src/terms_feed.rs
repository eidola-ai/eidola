//! Background resolver for the currently required legal-document versions.
//!
//! The terms of service and privacy policy govern billing, support, and
//! privacy practices that exist independently of any client/server release,
//! so their required versions must not be baked into the (measured, and
//! therefore per-release-immutable) server configuration. Instead, the
//! published website is the source of truth: this module polls each
//! document's exact source bytes (`/terms/source.md`, `/privacy/source.md`
//! — published by the site builder alongside the rendered page), computes
//! the SHA-256 of those bytes itself, parses the front-matter `version`,
//! and advances the shared `required_document` table.
//!
//! Ordering and failure semantics:
//! - Advancement is **monotonic** (`db::upsert_required_document` only
//!   accepts a strictly greater version), so a stale CDN edge, a lagging
//!   poll, or a website outage can never *regress* the requirement — it
//!   can only pause its advancement. CI enforces that a document's bytes
//!   never change without a version increment, which is what makes the
//!   integer ordering trustworthy.
//! - Because the table is shared by every server instance, one instance
//!   observing a new version converges the whole cluster's acceptance
//!   gates immediately; per-instance polling lag only delays *discovery*,
//!   never consistency.
//! - Poll failures are logged and retried on the next tick; the gate keeps
//!   enforcing the last known requirement. There is no fail-closed path —
//!   the gate is a legal formality, not a security boundary.

use std::time::Duration;

use deadpool_postgres::Pool;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::db::{self, RequiredDocumentRow};

/// The gated documents: (document name, source path, rendered-page path).
/// Paths are joined onto the feed base URL.
pub const DOCUMENTS: &[(&str, &str, &str)] = &[
    ("terms_of_service", "/terms/source.md", "/terms/"),
    ("privacy_policy", "/privacy/source.md", "/privacy/"),
];

/// Extract the front-matter `version = N` from a document's source bytes.
/// The front matter is the TOML block fenced by `+++` lines (the site
/// builder's format); this parser is deliberately narrow — we control both
/// producers of these files, and CI validates them at publish time.
pub fn parse_front_matter_version(src: &str) -> Option<i64> {
    let rest = src.strip_prefix("+++")?;
    let (raw, _) = rest.split_once("\n+++")?;
    for line in raw.lines() {
        if let Some(value) = line.trim().strip_prefix("version")
            && let Some(value) = value.trim_start().strip_prefix('=')
        {
            return value.trim().parse().ok();
        }
    }
    None
}

/// Fetch one document's source and turn it into a required-document row.
async fn resolve_document(
    client: &reqwest::Client,
    base_url: &str,
    document: &str,
    source_path: &str,
    page_path: &str,
) -> Result<RequiredDocumentRow, String> {
    let url = format!("{base_url}{source_path}");
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("fetch {url}: {e}"))?
        .error_for_status()
        .map_err(|e| format!("fetch {url}: {e}"))?;
    let bytes = resp.bytes().await.map_err(|e| format!("read {url}: {e}"))?;

    let text = std::str::from_utf8(&bytes).map_err(|e| format!("{url}: not utf-8: {e}"))?;
    let version = parse_front_matter_version(text)
        .ok_or_else(|| format!("{url}: no front-matter `version = N`"))?;
    let sha256 = hex::encode(Sha256::digest(&bytes));

    Ok(RequiredDocumentRow {
        document: document.to_string(),
        version,
        sha256,
        url: format!("{base_url}{page_path}"),
    })
}

/// One poll pass over every gated document. Failures are per-document and
/// logged; the shared requirement history only ever grows.
pub async fn poll_once(pool: &Pool, client: &reqwest::Client, base_url: &str) {
    use db::RecordRequiredOutcome;
    for (document, source_path, page_path) in DOCUMENTS {
        match resolve_document(client, base_url, document, source_path, page_path).await {
            Ok(row) => match db::record_required_document(pool, &row).await {
                Ok(RecordRequiredOutcome::Recorded) => info!(
                    "terms feed: recorded {} version {} ({})",
                    row.document, row.version, row.sha256
                ),
                Ok(RecordRequiredOutcome::AlreadyRecorded) => {}
                Ok(RecordRequiredOutcome::HashConflict { stored_sha256 }) => warn!(
                    "terms feed: published {} version {} has hash {} but version {} was \
                     recorded with hash {} — the document's bytes changed without a \
                     version increment (versioning-contract violation; CI should have \
                     caught this). Keeping the recorded hash; publish a new version to \
                     resolve.",
                    row.document, row.version, row.sha256, row.version, stored_sha256
                ),
                Err(e) => warn!("terms feed: recording {} failed: {}", row.document, e),
            },
            Err(e) => warn!("terms feed: {}", e),
        }
    }
}

/// Spawn the polling loop: one pass immediately, then every `refresh`.
pub fn spawn_terms_feed_task(pool: Pool, base_url: String, refresh: Duration) {
    let client = reqwest::Client::builder()
        .tls_backend_preconfigured(crate::tls_config())
        .build()
        .expect("failed to build terms feed HTTP client");

    tokio::spawn(async move {
        loop {
            poll_once(&pool, &client, &base_url).await;
            tokio::time::sleep(refresh).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_version_from_front_matter() {
        let src = "+++\ntitle = \"Terms\"\nversion = 3\n+++\n\n# Terms\n";
        assert_eq!(parse_front_matter_version(src), Some(3));
    }

    #[test]
    fn missing_version_or_front_matter_is_none() {
        assert_eq!(parse_front_matter_version("# No front matter\n"), None);
        assert_eq!(
            parse_front_matter_version("+++\ntitle = \"T\"\n+++\nbody"),
            None
        );
    }

    #[test]
    fn version_key_prefix_collisions_are_not_versions() {
        // `versionish = 9` must not parse as `version`.
        let src = "+++\nversionish = 9\nversion = 2\n+++\n";
        assert_eq!(parse_front_matter_version(src), Some(2));
    }

    #[test]
    fn unterminated_front_matter_is_none() {
        assert_eq!(parse_front_matter_version("+++\nversion = 1\n"), None);
    }
}
