+++
title = "Terms of service"
description = "Terms of service for Eidola's paid service."
version = 1
+++

# Terms of service

*The complete revision history of this document is public in [the repository](https://github.com/eidola-ai/eidola/commits/main/www/pages/terms.md).*

## 1. Introduction

These terms are an agreement between you and Eidola, Inc., a Delaware public benefit corporation ("Eidola," "we"). They govern the paid, hosted service: creating an account, purchasing credits, and running inference through our infrastructure.

If you only download and run the Eidola software — without an account and without our service — these terms do not apply to you. The software is open source, and your use is governed by its licenses as stated in the repository.

Most sections begin with a summary marked **In short**. The summaries are part of these terms, not decoration: the detailed sections control, but they must be read consistently with their summaries, and we will not rely on any detail that a reasonable person would find contradicted by its summary. If you can't understand terms, you can't meaningfully agree to them — so we've written these to be understood.

## 2. Who can use the paid service

> **In short:** You must be at least 18 and in the United States.

The paid service is offered only to individuals who are at least 18 years old and located in the United States, and to entities organized in the United States. By creating an account you represent that this describes you. Our architecture does not let us verify age, identity, or location — deliberately — so we rely on this representation instead of collecting data to verify it.

You also represent that you are not prohibited from using the service under United States sanctions or export control laws, and that you will not use it from, or for the benefit of anyone in, an embargoed jurisdiction.

Anyone, anywhere, of any age may use the open-source software under its own license. This section limits only the paid service.

## 3. Accounts

> **In short:** Your account is a random ID and a secret, shown once. Whoever holds the secret is, to us, you. We cannot reset it.

An account consists of an identifier and a secret. The secret is shown to you exactly once, at creation, and we store only a hash of it. Accounts carry no name, email address, or other identity — this is a feature, and it has consequences you should understand:

- Possession of the secret is the only authentication. Anything done with your secret is attributed to your account. Keep it safe.
- We cannot recover or reset a lost secret. If you lose it, you lose access to the account.
- If you lose your secret but can demonstrate to us that you made a purchase (for example, with your payment receipt), we can locate the associated account through our payment processor, void its remaining unexpired, unconverted balance, and refund it as described in the refunds section. We cannot restore access to the account itself.

You are responsible for activity under your account. Don't share the secret; there is no mechanism for shared or delegated access today.

## 4. Credits

> **In short:** Credits are prepaid units of compute capacity, not money. They expire — subscription credits at the end of each billing period, one-time purchases one year after purchase — and the expiration is part of the price you pay.

Credits are units of service capacity: they entitle you to inference at the per-model prices in effect when you use them. They are not money, not a deposit, and not redeemable for cash except as the refunds section provides. Credits are bound to your account and are not transferable.

Credits expire:

- **Subscription credits** expire at the end of the billing period in which they were granted. A subscription is a recurring plan: each period's credits are for that period.
- **One-time purchases** expire one year after purchase.

Expiration is disclosed at the point of purchase, on your receipt, and alongside your balance. Plans whose credits expire sooner cost less per unit of capacity, the same trade found in committed-use and prepaid plans across the computing industry. When credits expire they are removed from your balance.

You may cancel a subscription at any time; already-granted credits remain usable until their period ends, and no further periods are charged.

Per-model prices may change over time. Current prices are always available in the app and through the API before you spend anything.

## 5. Refunds

> **In short:** Unused, unexpired credits are refundable on request. Expired credits are generally forfeited. Credits converted to anonymous tokens are beyond anyone's ability to refund — including ours.

To request a refund, email **<hello@eidola.ai>**. We will void the unexpired, unconverted credit balance of your account and return the corresponding amount to your original payment method through our payment processor, where it supports the refund. Refund requests are handled by a person, not an automated flow, so allow a few business days.

Expired credits are forfeited and we have no obligation to refund them, though we may make exceptions at our discretion; an exception is not a waiver of this section.

Credits that have been converted to anonymous tokens cannot be refunded, restored, or even audited: the conversion severs, cryptographically, the link between the tokens and your account. We cannot verify any claim about what happened to converted tokens — that inability is the product's core privacy guarantee, and it binds us as much as it protects you.

If you believe a charge is erroneous, contact us before disputing it with your card issuer — we can resolve genuine errors faster than a dispute can. Fraudulent chargebacks are reversed against the account's balance and may result in account closure.

## 6. Anonymous tokens — the privacy boundary

> **In short:** Converting credits to anonymous tokens is irreversible. The tokens expire within about two weeks, and we can never link them back to you — that is the entire point.

The service lets you convert account credits into anonymous credit tokens (ACTs). Conversion is final. From the moment of issuance, the tokens are unlinkable to your account by us or anyone else, by cryptographic construction rather than by policy.

Tokens expire on a short schedule set by cryptographic key rotation — currently no more than about two weeks from issuance. Expired tokens are unrecoverable and unspent value in them is forfeited; because tokens are unlinkable, no exception is possible even in principle. The app provisions tokens automatically in small amounts as you use the service, precisely so that little value sits at risk of lapsing; convert manually only what you expect to use.

## 7. Acceptable use

> **In short:** Don't use the service to break the law, violate others' rights, or disrupt the service itself.

You agree not to use the service:

- for anything unlawful, including generating child sexual abuse material or otherwise exploiting minors;
- to violate the rights of others; or
- to disrupt or degrade the service or its upstream infrastructure for other users.

Probing, verifying, and attempting to falsify our security and privacy claims is not a violation — it is encouraged, and the system is built to be interrogated. Good-faith security research is welcome; report findings through the repository's security policy.

We do not, and by design cannot, monitor the content of your use. We may refuse or terminate service where we lawfully learn of violations, and we will comply with valid legal process as described in our [privacy policy](/privacy/).

## 8. AI outputs

> **In short:** Models are probabilistic and fallible. Their output is not professional advice, and what you do with it is your responsibility.

The service runs third-party, openly available AI models. Their outputs are generated probabilistically: they can be inaccurate, incomplete, or unexpected, and no amount of testing can evaluate how a large model will behave in every scenario. Outputs are not medical, legal, financial, or other professional advice, and you should not rely on them as such.

You are responsible for evaluating outputs before acting on them and for the consequences of their use. As between you and Eidola, we claim no rights in your prompts or your outputs — we never see them, and they are yours.

## 9. Service changes, suspension, and termination

> **In short:** The service is young. Models and prices will change; we don't promise uninterrupted availability. If we ever terminate your account without cause, we will refund your unexpired, unconverted balance.

The service is provided as available. We do not guarantee uninterrupted operation, and the model catalog, features, and per-model prices may change.

We may suspend or terminate an account for material violation of these terms, for fraudulent payment activity, or where the law requires. If we terminate your account for any other reason, or discontinue the service, we will refund the account's unexpired, unconverted balance.

You may close your account at any time; closing it refunds your unexpired, unconverted balance as described in the refunds section.

## 10. Disclaimers

> **In short:** The service and software are provided as-is.

To the maximum extent permitted by law, the service and software are provided "as is" and "as available," without warranties of any kind, express or implied, including merchantability, fitness for a particular purpose, and non-infringement. Some jurisdictions do not allow the exclusion of implied warranties, so parts of this section may not apply to you.

## 11. Limit of liability

> **In short:** Our total liability is capped at what you paid us in the twelve months before your claim, minus what you converted to anonymous tokens.

To the maximum extent permitted by law, Eidola's aggregate liability arising out of or relating to the service is limited to the amounts you paid us in the twelve months preceding the event giving rise to the claim, less amounts converted to anonymous tokens during that period; and we are not liable for indirect, incidental, consequential, or punitive damages, or for lost profits or data.

Nothing in these terms limits liability for gross negligence, willful misconduct, or fraud, or restricts rights you hold under law that cannot be waived by contract.

## 12. Disputes

> **In short:** Talk to us first — we commit to sixty days of good-faith resolution. After that: small claims court where you live, or the courts in Delaware. We do not require arbitration and we do not take away your right to join a class action.

Before filing any claim, you agree to email us a description of the dispute and give us sixty days to resolve it in good faith; we commit to the same before filing any claim against you.

Either party may bring a qualifying claim in small claims court in the county where you live. All other disputes are resolved in the state or federal courts located in Delaware, and these terms are governed by Delaware law, excluding its conflict-of-laws rules, except where the law of your state of residence grants you consumer protections that cannot be varied by contract.

These terms do not require arbitration or waive your right to participate in a class action.

## 13. Changes to these terms

> **In short:** If these terms change materially, the app asks you to accept the new version before your next paid action. We don't pretend that silence is consent.

Each version of these terms carries a version number and is identified by the cryptographic hash of its exact text — both are shown at the top of this page when viewed on our website — and your acceptance of a specific version is recorded against your account. If we change these terms materially, you will be asked to review and accept the new version before making further purchases or conversions; declining means we refund your unexpired, unconverted balance and part ways. Already-issued anonymous tokens are honored under the terms in effect when they were issued, until they expire.

Notice of changes is posted here, and this document's full history is public in the repository.

## 14. Everything else

These terms, together with the [privacy policy](/privacy/) and the disclosures presented at purchase, are the entire agreement between you and Eidola about the paid service. If part of these terms is found unenforceable, the rest stands. A failure to enforce a term is not a waiver of it. You may not assign your account or these terms; we may assign them in connection with a merger, acquisition, or reorganization, and these terms bind our successors. Sections that by their nature should survive termination — including refunds, disclaimers, liability limits, and disputes — survive it.

Questions and legal notices: **<legal@eidola.ai>**. Eidola, Inc. is a Delaware public benefit corporation.
