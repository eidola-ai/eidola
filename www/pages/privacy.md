+++
title = "Privacy policy"
description = "The formal privacy policy for Eidola, Inc. — the whole company, not just the service."
version = 1
+++

# Privacy policy

*The complete revision history of this document is public in [the repository](https://github.com/eidola-ai/eidola/commits/main/www/pages/privacy.md).*

## 1. Introduction

This privacy policy describes the information processing practices of Eidola, Inc. across its hosted service, website, community spaces, and support channels.

Eidola's privacy commitments are architectural before they are contractual, and our overarching approach is to minimize scenarios that require trust.

Most sections begin with a summary marked **In short**. Summaries are accurate restatements of the details, not limitations on them.

## 2. Paid service

> **In short:** We designed our systems so that your usage is impossible to tie to you. We can't tell which account made a request, and the content of requests is verifiably never logged or saved.

The service has two deliberately separated surfaces. The **linked** surface knows your account (balances, purchases). The **unlinked** surface handles all inference processing, is paid with anonymous credit tokens, and never receives or derives any identifier tying a request to the account that funded it, or to any other request.

With full access to our own database, we cannot answer "which account paid for this inference request" or "which requests came from the same person." There is no mechanism for inspecting, reviewing, or replaying anyone's interactions, and building one would violate the [privacy guarantees](/docs/privacy-guarantees/) attested to in each release.

Your prompts, attachments, and model outputs are never written to durable storage on infrastructure we control, and never appear in logs, telemetry, or error reports. Content is decrypted only inside attested confidential-compute enclaves — ours and our inference processor's — and you can verify that claim by reading [the source code](https://github.com/eidola-ai/eidola/); the client re-verifies the hardware attestation on every connection.

All telemetry we keep from the inference requests is generic and unidentifiable, including attributes like model name, token counts, status code, and latency, and the exact behavior is auditable in the source code. Nothing is ever tied to an account. Because we cannot identify who made an inference request and do not store its contents, there is nothing we can look up, hand over, correct, or delete.

## 3. Network metadata

> **In short:** The internet can see that your device talked to our service, when, and how much. We minimize what we record; others might not. If it matters to your threat model, use a VPN or Tor.

We treat transport metadata — IP addresses, connection timing, traffic volume — as observable by network operators, internet infrastructure, and our upstream providers regardless of anyone's stated policy, including ours.

Infrastructure we operate may record network metadata for the sole purpose of operating it, and any identifiable elements are purged as soon as they are no longer needed. Anonymous aggregated network metadata — for example, traffic volume by city over time — may be kept and used for broader business purposes, such as planning the location of future infrastructure.

Separately, our payment processor collects network information when you interact with its checkout and billing pages — see section 4.

In general, when a single IP address is shared by many users, profiling based on network metadata becomes less revealing. For this reason, we recommend connecting through a VPN provider you trust, or Tor.

## 4. Accounts and billing

> **In short:** An account is a random ID and a secret — no name, no email. Payments run through Stripe, which has its own legal obligations and its own policy. Transaction records are kept as the law requires.

Creating an account collects no personal information: no email, phone, name, address, or government identifier — we never ask. Our billing database holds the account's random identifier, a hash of its secret, an append-only ledger of credit grants and spends, and, once you make a purchase, a Stripe customer reference. That reference is the only path from an account to a human identity, which makes these records pseudonymous rather than anonymous.

Payments are processed by **Stripe**, configured to collect the minimum it permits. Stripe necessarily receives your payment details and, in addition to acting as our processor, operates as an independent controller under its own [privacy policy](https://stripe.com/privacy) for purposes such as fraud prevention across its network and its legal compliance obligations. We do not promise anonymity against Stripe with respect to payment metadata.

While you interact with Stripe's checkout and billing pages, Stripe also collects your IP address and related technical information under its own policy. Those records persist at the Stripe layer, linked to your payment identity, and are available to us.

Ledger and account records are financial records: we retain them as tax, payment-network, and anti-fraud law requires, and requests to delete them are subject to those legal retention obligations.

Converting credits to anonymous tokens crosses the boundary described in section 2 — from that moment, the spending side is unlinkable to you even by us.

## 5. Email

> **In short:** Email you send us is the one place we hold ordinary personal information. It lives in Google Workspace, we keep it as business records, and it's also how you exercise your privacy rights.

If you email us — support, questions, privacy requests — we receive whatever you send, associated with your email address, hosted in Google Workspace. We keep correspondence as ordinary business records, share it with no one beyond the service providers that host it, and retain records of privacy-rights requests as the law requires.

## 6. Community spaces

> **In short:** Our Matrix rooms and GitHub repository are public. What you post there is public, copied across servers we don't control, and effectively permanent. Speak accordingly.

Our Matrix rooms are public spaces on the matrix.org homeserver, operated by the Matrix.org Foundation under its own privacy policy; we moderate them. Matrix is a federated protocol: every participating server receives a copy, and deleting a message everywhere is not within anyone's power. On request (or by moderation) we will redact messages in the rooms we control, but that is best-effort, and federated copies may persist.

Development happens in public on GitHub, under GitHub's terms and privacy policy. Contributions — including the name, email, and timestamps in commits — become part of a permanent, distributed public record, as described plainly in [CONTRIBUTING](https://github.com/eidola-ai/eidola/blob/main/CONTRIBUTING.md) and acknowledged in the contributor license agreement. Erasure requests cannot rewrite that history. Private security reports are retained as security records.

## 7. The website

> **In short:** No cookies, no trackers, no analytics, no third-party resources. The absence of a consent banner is not an oversight.

Our website is static, uses no cookies or similar technology for tracking or analytics, and loads no resources from third-party websites. It is served by GitHub Pages, whose infrastructure maintains standard, short-lived server logs including visitor IP addresses; we do not receive or use them.

## 8. Who we share data with

> **In short:** Service providers who host things for us may receive information, under contracts that limit them to that role. We may have to provide information to other entities as part of valid legal process, but because we minimize the information we receive, there is little to provide.

We do not sell personal information or share it for advertising. Apart from Stripe's independent legal role described above, we do not grant anyone rights to use personal information for their own purposes. The service providers that process data on our behalf — payment, database hosting, telemetry, email, etc. — are few enough to name, and we maintain a list in [infrastructure and vendors](/docs/vendors/).

We comply with valid legal process under United States law, but we are only capable of producing the data we collect: pseudonymous billing records, Stripe-side payment records, and email — never inference content, never identity-linked usage. Network metadata (section 3) is generally obtainable by authorities directly from network and infrastructure operators, without our involvement. We will contest any demand to build interception, logging, or identification capabilities, and the per-release signed privacy guarantees make a compelled, quiet change of that stance detectable in the artifact itself.

## 9. Personal information at a glance

For readers who want the conventional inventory, this table covers every category of personal information we handle:

| Category | What we hold | Source | Purpose | Retention |
| --- | --- | --- | --- | --- |
| Contact information | Your email address and whatever your messages contain, if you email us | You | Support and privacy requests | Kept as business records; privacy-request records as the law requires |
| Commercial records | Account identifier, hashed secret, credit ledger, Stripe customer reference | You; Stripe | Billing, refunds, fraud prevention, legal compliance | As tax, payment-network, and anti-fraud law requires |
| Payment details | None — Stripe collects these directly, under [its policy](https://stripe.com/privacy) | — | — | — |
| Network activity | Transient network metadata (section 3) | Your device | Operating our infrastructure | Purged once no longer needed; anonymous aggregates may be kept |
| Inference content | None — prompts, attachments, and outputs are never stored (section 2) | — | — | — |
| Public contributions | What you post in our community spaces (section 6) | You | Community and development | Public and effectively permanent |

We collect no sensitive personal information as privacy laws define it — no government identifiers, precise geolocation, biometric or health data — and we obtain nothing from data brokers or other third-party sources. We do not sell or share personal information, and we do not use it for targeted advertising or profiling.

## 10. Your rights

> **In short:** Send an email to <privacy@eidola.ai>. If we can find data about you, we'll disclose or delete it as the law provides. For the anonymous layer, there is genuinely nothing to find.

Privacy laws in California and other states give you rights to know, correct, and delete personal information, and not to be discriminated against for exercising them. To exercise any of them, send an email to **<privacy@eidola.ai>** from the address your request concerns; a message passing standard email authentication (DKIM/SPF) is treated as verified for data keyed to that address. For account records, authenticate with the account itself or your payment receipt, since accounts carry no email. We acknowledge requests promptly and respond within the statutory deadlines (45 days under California law, extendable once).

We honor Global Privacy Control signals in the only coherent way available: we already don't sell or share personal information, so the signal finds nothing to switch off.

## 11. Children

The paid service is offered only to adults (18+, per the [terms of service](/terms/)). We do not knowingly collect personal information from children — or, for the most part, from anyone.

## 12. Changes to the privacy policy

Material changes to this policy are announced on this page, and the full revision history is public in the repository. The [privacy guarantees](/docs/privacy-guarantees/) document describes intrinsic attributes of the software, and a specific version is attested to during the signing of each release.

## 13. Contact

Send an email to **<privacy@eidola.ai>** for privacy matters or **<hello@eidola.ai>** for anything else. Eidola, Inc. is a Delaware public benefit corporation.
