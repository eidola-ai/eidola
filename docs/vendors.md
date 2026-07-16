# Infrastructure and vendors

This is the complete list of third parties that process data in the operation of Eidola's service, website, and business — what each one is for, and exactly what it can see. The [privacy policy](https://www.eidola.ai/privacy/) points here rather than naming vendors inline, so that this list can stay current (and its history auditable) in the repository.

The list is short because the architecture makes it short: inference content never reaches any of these parties in a form they can read, link, or keep, and most of them never see anything about you at all.

## Processors — parties handling data on our behalf

| Vendor | Role | What it sees |
| --- | --- | --- |
| **Stripe** | Payment processing | Your payment details, email, and transaction history — the full payment identity. Also acts as an *independent controller* under [its own policy](https://stripe.com/privacy) for network-wide fraud prevention and its legal compliance; that role is Stripe's own and not assignable by us. |
| **Crunchy Data** | Managed PostgreSQL hosting for the billing database | Pseudonymous billing records: account UUIDs, secret hashes, the credit ledger (amounts, timings, expiries), Stripe customer references, and encrypted credential-issuer keys. No names, emails, IP addresses, or content. |
| **Grafana Cloud** | Telemetry (OpenTelemetry traces, metrics, logs) | Operational telemetry only. Inference-path spans carry model name, token counts, status, and latency — never content, never an account. Account-layer spans may carry the pseudonymous account UUID. |
| **Google Workspace** | Company email | The contents of email you send us and we send you — the one store of ordinary personal information we hold. |
| **GitHub** | Repository, releases, CI, and website hosting (GitHub Pages) | Public repository activity under GitHub's own terms; standard short-lived web-server logs (visitor IPs) for the website and release downloads, which we do not receive. |

## Infrastructure inside the trust boundary

| Vendor | Role | What it sees |
| --- | --- | --- |
| **Tinfoil** | Confidential-compute hosting and inference upstream | Ciphertext, attested enclaves, and network metadata (connection IPs, timing, volume). Inference content is decrypted only inside enclaves whose measurements the client verifies on every connection — see [inference upstream](upstream.md) and the [threat model](threat-model.md). We treat Tinfoil's network layer as potentially logging transport metadata, like any network operator. |

## Independent platforms we participate in

| Platform | Role | Notes |
| --- | --- | --- |
| **Matrix.org Foundation** | Homeserver hosting our public community rooms | An independent controller under its own privacy policy. Room content is public and federates to servers nobody controls; see the privacy policy's community section. |

## What is deliberately absent

No advertising networks, no analytics providers, no data brokers, no "enrichment" services, no CRM holding profiles of users. Sub-processors of the vendors above (for example, a vendor's own cloud provider) are bound through our contracts with that vendor, per standard data-processing terms.

Changes to this list are visible in this file's git history.
