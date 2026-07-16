+++
title = "Privacy policy"
description = "The formal privacy policy for Eidola, Inc. — the whole company, not just the service."
+++

# Privacy policy

*Draft — under review by counsel; effective at general availability. The complete revision history of this document is public in [the repository](https://github.com/eidola-ai/eidola).*

This is the privacy policy for everything Eidola, Inc. does — the hosted service, the website, our community spaces, and our inbox. Most privacy policies describe how a company handles your data. Most of this one describes how we arranged not to have it.

Eidola's privacy commitments are architectural before they are contractual. The enforceable technical contract is the [privacy guarantees](/docs/privacy-guarantees/) document: enumerated, durable commitments that a named release engineer signs against every release and that you can verify against the running code. This policy restates those guarantees in conventional legal form — it does not weaken them — and then covers the things that happen *outside* that verifiable environment, which is where a policy is actually load-bearing.

Each section begins with a summary marked **In short**. If a summary and its details ever appear to conflict, we intend the reading a reasonable person would take from the summary.

## 1. The short version

- When you run inference through the service, we never store your prompts or outputs, and we cannot link the request to your account — by cryptographic construction, not by promise.
- Our billing records contain no name, email, IP address, or other natural identifier. The only link between an account and a human runs through Stripe, our payment processor.
- The one place we hold ordinary, regulated personal information is email you send us.
- Our website sets no cookies, runs no analytics, and loads nothing from third parties. There is no banner because there is nothing to consent to.
- We do not sell or share personal information with anyone, for anything, ever. There is no "Do Not Sell" link because there is nothing the link would stop.

## 2. The service: inference

**In short:** We built the inference path so that your usage is not ours to know. We can't tell which account made a request, and the content is never written down.

The service has two deliberately separated surfaces. The **linked** surface knows your account (balances, purchases). The **unlinked** surface performs inference, paid with anonymous credit tokens, and never receives or derives any identifier tying a request to the account that funded it, or to any other request. With full access to our own database, we cannot answer "which account paid for this inference request" or "which requests came from the same person." There is no operator interface for inspecting, reviewing, or replaying anyone's traffic, and building one would violate the signed [privacy guarantees](/docs/privacy-guarantees/).

Your prompts, attachments, and model outputs are never written to durable storage on infrastructure we control, and never appear in logs, telemetry, or error reports. Content is decrypted only inside attested confidential-compute enclaves — ours and our inference upstream's — and you can verify that claim rather than take it on faith; the client re-verifies the hardware attestation on every connection.

The complete telemetry we keep from the inference path is: model name, token counts, status code, and latency. Nothing else, and never tied to an account.

Because we cannot identify who made an inference request, data-subject rights simply have no object on this surface: there is nothing we could look up, hand over, correct, or delete, and the law does not require us to build identification we deliberately do not have (see, e.g., GDPR Article 11 and the equivalent rules in US state privacy laws). This is the point of the design — rights you never need are better than rights you must exercise.

## 3. Network metadata — the honest boundary

**In short:** The internet can see that your device talked to our service, when, and how much. We don't record that; others might. If it matters to your threat model, use a VPN or Tor.

We treat transport metadata — IP addresses, connection timing, traffic volume — as observable by network operators, internet infrastructure, and our upstream providers regardless of anyone's stated policy, including ours. Our application handlers do not persist or emit client IP addresses or other network-layer identifiers, and our architecture ensures that even parties who do record network metadata cannot link it to your account through us. But we do not defend against traffic analysis, and we won't pretend otherwise (see the [threat model](/docs/threat-model/)).

If your threat model includes network observers, connect through a VPN provider you trust, or Tor. We are comfortable being a company whose privacy policy recommends countermeasures against layers we don't control.

## 4. Accounts and billing

**In short:** An account is a random ID and a secret — no name, no email. Payments run through Stripe, which has its own legal obligations and its own policy. Transaction records are kept as the law requires.

Creating an account collects no personal information: no email, phone, name, address, or government identifier — we never ask. Our billing database holds the account's random identifier, a hash of its secret, an append-only ledger of credit grants and spends, and, once you make a purchase, a Stripe customer reference. That reference is the only path from an account to a human identity, which makes these records pseudonymous rather than anonymous, and we treat them accordingly.

Payments are processed by **Stripe**, configured to collect the minimum it permits. Stripe necessarily receives your payment details and, in addition to acting as our processor, operates as an independent controller under its own [privacy policy](https://stripe.com/privacy) for purposes such as fraud prevention across its network and its legal compliance obligations — that is Stripe's doing, not a grant from us, and we can't turn it off. We do not promise anonymity against Stripe with respect to payment metadata.

Ledger and account records are financial records: we retain them as tax, payment-network, and anti-fraud law requires, and requests to delete them are subject to those legal retention obligations. Closing your account refunds your unexpired, unconverted balance, permanently disables the account's credential, and severs our link to Stripe's customer record; the anonymized ledger remains.

Converting credits to anonymous tokens crosses the boundary described in section 2 — from that moment, the spending side is unlinkable to you even by us.

## 5. Email

**In short:** Email you send us is the one place we hold ordinary personal information. It lives in Google Workspace, we keep it as business records, and it's also how you exercise your privacy rights.

If you email us — support, questions, privacy requests — we receive whatever you send, associated with your email address, hosted in Google Workspace. We keep correspondence as ordinary business records, share it with no one beyond the service providers that host it, and retain records of privacy-rights requests as the law requires (currently 24 months under California regulations).

## 6. Community spaces

**In short:** Our Matrix rooms and GitHub repository are public. What you post there is public, copied across servers we don't control, and effectively permanent. Speak accordingly.

Our Matrix rooms are public spaces on the matrix.org homeserver, operated by the Matrix.org Foundation under its own privacy policy; we moderate them. Messages federate: every participating server receives a copy, and deleting a message everywhere is not within anyone's power. On request (or by moderation) we will redact messages in the rooms we control — that is best-effort, and federated copies may persist.

Development happens in public on GitHub, under GitHub's terms and privacy policy. Contributions — including the name, email, and timestamps in commits — become part of a permanent, distributed public record, as described plainly in [CONTRIBUTING](https://github.com/eidola-ai/eidola/blob/main/CONTRIBUTING.md) and acknowledged in the contributor license agreement. Erasure requests cannot rewrite that history, and the law does not require the impossible. Private security reports are retained as security records.

## 7. The website

**In short:** No cookies, no trackers, no analytics, no third-party anything. The absence of a consent banner is not an oversight.

This website is static. It sets no cookies, stores nothing in your browser, includes no analytics, and loads no third-party resources — a content-security policy enforces that. It is served by GitHub Pages, whose infrastructure (like any host's) maintains standard, short-lived server logs including visitor IP addresses; we do not receive or use them. If we ever add usage measurement, it will be first-party and cookieless — designed so this section stays true and the banner stays absent.

## 8. Who we share data with

**In short:** Service providers who host things for us, under contracts that limit them to that. Nobody else, except valid legal process — which finds very little to take.

We do not sell personal information, share it for advertising, or grant anyone rights to use it for their own purposes (Stripe's independent legal role, described above, is the one structural exception). The service providers that process data on our behalf — payment, database hosting, telemetry, email — are few enough to name, and we do: see [infrastructure and vendors](/docs/vendors/), kept current in the repository.

We comply with valid legal process under United States law. What exists to produce is what this policy describes: pseudonymous billing records, Stripe-side payment records, and email — never inference content, never identity-linked usage. We will not build interception, logging, or identification capabilities in response to a demand; the signed privacy guarantees make a compelled, quiet change of that stance detectable in the artifact itself, which is exactly why they exist. If lawfully permitted to disclose legal process we receive, we will; where we are gagged, the verifiable build speaks instead.

## 9. Your rights

**In short:** Email <privacy@eidola.ai>. If we can find data about you, we'll disclose or delete it as the law provides. For the anonymous layer, there is genuinely nothing to find — that's the product working.

Privacy laws in California and other states (and, where it applies, the GDPR) give you rights to know, correct, and delete personal information, and not to be discriminated against for exercising them. To exercise any of them, email **<privacy@eidola.ai>** from the address your request concerns; a message passing standard email authentication (DKIM/SPF) is treated as verified for data keyed to that address, because controlling the mailbox is the identity in question. For account records, authenticate with the account itself or your payment receipt, since accounts carry no email. We acknowledge requests promptly and respond within the statutory deadlines (45 days under California law, extendable once).

What the answers will look like, honestly: email records — disclosed or deleted on request; billing records — disclosed on request, deleted except where financial-record retention laws require otherwise; public community content — redacted best-effort as described above; the anonymous service layer — nothing exists to disclose or delete. We honor Global Privacy Control signals in the only coherent way available: we already don't sell or share personal information, so the signal finds nothing to switch off.

## 10. Children

The paid service is offered only to adults (18+, per the [terms of service](/terms/)). We do not knowingly collect personal information from children — or, for the most part, from anyone.

## 11. Changes

Material changes to this policy are announced on this page, and the full revision history is public in the repository — you can diff any two versions, not just trust a "last updated" date. The privacy guarantees document, which binds the service itself, cannot be weakened silently: its evolution rules require a human attestant to be unable to sign falsely.

## 12. Contact

**<privacy@eidola.ai>** for privacy matters; **<hello@eidola.ai>** for anything else. Eidola, Inc. is a Delaware public benefit corporation.
