# Anonymous credits

Eidola is paid for with **anonymous credit tokens**: the Privacy Pass ACT scheme ([draft-schlesinger-privacypass-act](https://datatracker.ietf.org/doc/draft-schlesinger-privacypass-act/), over the cryptographic construction in [draft-schlesinger-cfrg-act](https://datatracker.ietf.org/doc/draft-schlesinger-cfrg-act/), implemented by the `anonymous-credit-tokens` crate). An account buys credits; the client converts some of them into a token the gateway signs without seeing; and each inference request is paid for by spending part of that token, in a way the gateway cannot connect to the account or to the client's other requests.

The scheme does that only for the part of the system it describes. Everything around it decides whether it still holds: when keys rotate, how much is put in a token and when, what the database is allowed to remember, what a log line may say, how a refund gets back to a client whose connection dropped. A careless choice in any of them can undo the anonymity the cryptography provides, or turn the payment system into a way to take inference without paying for it, and neither failure shows up in a test that only checks the cryptography.

This page collects those choices in one place. It is written for people who are technical but are not cryptographers: users deciding what to trust, auditors, and contributors. The first part explains what the protocol itself gives. The second explains, one decision at a time, what the rest of the system has to do so that the protocol's promise survives contact with a real deployment. Each decision names where it is enforced and what would break it.

> [!IMPORTANT]
> **A change to any code this page names must be checked against it.** That covers the gateway's credential, issuance, spend, refund and recovery code, its billing schema, and the client's wallet, provisioning and charging code. If a change alters a decision recorded here, this page changes in the same pull request. The privacy promises themselves are in [privacy-guarantees.md](privacy-guarantees.md); this page is how the system keeps the ones about payment.

## What the protocol provides

### The parties and the token

There are two parties. The **issuer** is the gateway. It holds a private key and signs tokens. The **client** is the app on your device. It holds tokens and spends them.

A token is a small piece of data held only by the client. It carries a balance (`c`, in credits; one credit is one millionth of a US dollar), a secret serial number called the **nullifier**, a secret blinding value, the **request context** (more on that below), and the issuer's signature over all of those. The signature is what makes the token worth something; the secret values are what make it the client's alone.

### Issuance: blind signing

To get a token, the client makes up the secret values itself and sends the issuer a *commitment* to them: a value that binds the client to those secrets without revealing them. The issuer signs the commitment together with the balance and the request context. The client then turns that signature into a signature over its own secrets.

What the issuer sees at issuance:

- **who** is asking (issuance is on the account surface, authenticated with the account's credentials),
- **when**,
- **how much** (`c` is chosen by the issuer and sent back in the clear),
- **which key** it signed with, and
- **the request context** it signed into the token (also sent back in the clear).

What it does not see is the token's nullifier or its blinding value. Because of that, the issuer cannot later recognise the token when it is spent. This is the "blind" in blind signature, and it is the root of everything else on this page.

### Spending: a hidden balance and a public charge

To pay for a request, the client sends a **spend proof**. It is a zero-knowledge proof: it convinces the issuer of a statement without revealing anything beyond the statement. The statement is: "I hold a token signed by your key, under this request context, whose balance is at least `s`." Alongside it the client reveals three values in the clear:

- the **charge** `s`, the amount being taken from the token,
- the token's **nullifier**, and
- the **request context**.

The balance `c` stays hidden. So does everything about which issuance the token came from. Two spends of two different tokens look alike to the issuer apart from their charges, nullifiers and contexts.

### Refunds: change, under the same key

A spend uses up the whole token. The issuer answers with a **refund**, which the client combines with secret state it kept from the spend (the *pre-refund*) to build a **successor** token. The successor's balance is `c − s + t`:

- `c − s` is what was left in the old token. The issuer never learns it; the client proves it in zero knowledge.
- `t` is the amount the issuer hands back. It **is visible**: the refund message carries `t` in the clear.

So a spend can take a generous `s` up front, and the issuer returns the unused part as `t` once it knows what the request really cost. The successor has a fresh nullifier the issuer has never seen, is signed under **the same key** as the token it replaces, and carries **the same request context**.

### Nullifiers and double spending

Because the issuer cannot recognise a token, it needs another way to stop the same token being spent twice. That is the nullifier. Each token's nullifier is revealed exactly once, when it is spent. The issuer records every nullifier it has accepted and refuses any spend whose nullifier it has seen before. It must record the nullifier **before** it issues the refund. Otherwise a client could spend the same token twice concurrently, collect two refunds, and end up with two successors: a fork that duplicates value.

### Keys, epochs and the request context

The issuer's key pair is what the signature is checked against. Keys can be retired: a token signed under a retired key is no longer accepted. The Privacy Pass deployment draft uses this to bound how long a token lives, and lets the issuer forget the nullifiers of a retired key, since nothing under that key can be spent any more.

The **request context** is a value bound into every token at issuance and revealed at every spend. It lets an issuer scope tokens (to one origin, say). The deployment draft derives it from the issuer's name, the origin, an optional credential context, and the key id. Since it is revealed in the clear at every spend, the request context also determines **who looks alike**: tokens are indistinguishable only from other tokens with the same key and the same context.

A **domain separator** (`ACT-v1:{organization}:{service}:{deployment}:{version}`) is mixed into all of the scheme's parameters, so tokens from one deployment or protocol version are cryptographically useless in another.

### What the protocol leaves to the deployment

The scheme proves statements about tokens. It does not decide:

- how many people share a key, or for how long (the anonymity set),
- how much is put in a token, or when it is issued and first spent,
- what the issuer records, logs, or stores, and where,
- whether the request context and amount the issuer chooses at issuance are the ones every client gets,
- what happens when a response is lost, a process crashes, or a key lapses with value still in a token,
- what a request costs, or whether the charge depends on anything about the request beyond what the client can predict.

Each of those is a decision Eidola has made. The rest of this page goes through them.

## Why the rest of the system matters

### The price is the only defence

An account-based service that is abused can see who is abusing it. It can rate-limit that account, ban it, refund a complaint after looking at the history, or give out a free tier and watch for people farming it. Eidola's inference surface can do none of that. The gateway cannot tell which account sent a request, or whether two requests came from the same client. That is the point of the design, and it means there is **no identity to rate-limit, ban or investigate** on the inference path.

What is left is the price. Every request must be paid for in full, up front, by a token that was paid for in full, before any work is done, and the amount it may cost must be fixed by the request itself. If any path lets a request cost less than it should, or lets value be spent twice, or lets a token outlive what was paid for it, there is no second line of defence behind it.

### And there is no quick fix

The gateway's code is measured into its enclave and pinned by every client release; a release is reproducible, reviewed and signed by a human attestant ([releases.md](releases.md)). That is what lets users check what the gateway does. It also means a gateway change is a release, not a hotfix: the shortcuts an account-based SaaS takes when billing misbehaves (refund on complaint, a generous free tier, a server-side heuristic, "we'll patch it later") are either unavailable or would weaken a promise users are relying on. The decisions below have to be right when they ship.

### Two risks per decision

Each decision below protects one or both of:

- **Confidentiality and unlinkability**: whether the gateway, the database, the telemetry vendor or a network observer can connect a request to an account, or two requests to each other.
- **The business model**: whether someone can get inference without paying, keep value past what it was bought for, or lose value they paid for.

For each one: the **decision**, **why**, **where it is enforced**, and **what would break it**, so a reviewer knows what to look for.

## Keys, epochs and the anonymity set

**Decision.** One issuer key per epoch, never rotated within one, and the epochs are a **public calendar** rather than the gateway's choice. An epoch is seven days, and epoch `n` begins at Unix time `n × 604800` (a Thursday, 00:00 UTC). Epoch `n`'s key issues from the start of epoch `n` until the start of `n + 1` (`issue_from` to `issue_until`) and its tokens are accepted until the start of `n + 2` (`accept_until`): one epoch issuing, one more accepting. So a key's three instants are a function of its epoch alone, and anyone can check them with one division. The gateway keeps the current epoch's key and the next one's, whichever instance mints them and whenever. Keys are generated inside the enclave, and their private halves are stored in Postgres encrypted under `CREDENTIAL_MASTER_KEY`, which never leaves the enclave, with the key's id, public key, domain separator and all three instants bound into the encryption (below).

**Why.** Every token issued under one key looks like every other: the anonymity set is the accounts that received a token under that key ([privacy-guarantees.md §2.3](privacy-guarantees.md#2-unlinkability)). Long epochs make that set larger; a key that is retired bounds how long any token lives and lets its nullifiers be forgotten. The acceptance window beyond issuance means a token can be held and spent later, so the time it was issued and the time it is spent are not forced to be close (§2.4). One issuing key at a time means there is never a choice of key that could sort clients into groups. And a calendar fixed in advance means the windows themselves are not a choice either: nothing can start a key at a moment chosen around one client, stretch one, or bring an old one back, without the result being visibly off the calendar.

The durations are the ones the gateway used before the calendar (seven days issuing, seven more accepting); only the alignment is new. Before it, the first key started whenever the gateway first ran and each later key chained from its predecessor, so where the boundaries fell was a fact about the deployment's history that no client could check. Aligning to the Unix epoch is the simplest anchor anyone can reproduce; a boundary falls on a Thursday at midnight UTC, which is no worse a time to rotate than any other.

**Enforced.** `eidola_common::issuer_calendar` (`EPOCH_SECS`, `ANCHOR_UNIX_SECS`, `ISSUER_CALENDAR`: compile-time constants that the gateway and every client compute the same windows from); on the gateway `credentials::generate_key` (a key's windows are its epoch's), `ensure_keys` and `epoch_to_mint` (the current key, then the next; never a second key for an epoch this instance already holds one for; a serializable transaction so concurrent instances cannot mint two), `sole_key_for_epoch` (two keys for the current epoch issue under neither), `canonical_epoch` (a key off the calendar is never used, for issuance or for spends), `spawn_key_rotation_task`, and the `issuer_key` table's `one_key_per_epoch` constraint. Tests: `generated_keys_sit_exactly_on_the_calendar`, `a_gateway_mints_the_current_key_then_the_next_and_never_a_second`, `two_keys_for_one_epoch_issue_under_neither`, `a_key_off_the_calendar_is_refused_even_when_it_authenticates`, `keys_stay_on_the_calendar_across_restarts_and_an_edited_row_is_refused` (database-backed). The client picks the key whose issuing window covers now and refuses it unless it is on the calendar and is the key it already recorded for the epoch ([Key rotation and partitioning](#key-rotation-and-partitioning)).

**What would break it.**

- Shorter epochs, overlapping issuing windows, or more than one key issuing at once: each splits the anonymity set.
- Any per-account, per-region or per-instance choice of key.
- Windows computed from anything but the calendar (the time a gateway started, a predecessor's deadline, a configuration value), or a calendar constant changed without a release every client takes at once.
- An emergency path that mints a replacement key within an epoch: clients refuse it, by design (below).
- Tying acceptance to issuance more tightly, which forces spending soon after issuance.

## Issuer key records are bound to their windows

**Decision.** A key's private half is encrypted with AES-256-GCM under a key derived from `CREDENTIAL_MASTER_KEY` and the key id, with **associated data** covering everything else the gateway reads from the row: a fixed label (`eidola/issuer-key/v2`), the key id, the public key, the domain separator, and `issue_from`, `issue_until` and `accept_until`, each length-prefixed. A row whose windows, id, public key or separator were edited no longer decrypts, and the gateway refuses it: it is not cached, not issued under, and no spend is accepted under it. `GET /v1/keys` is answered from the gateway's own cache of keys that authenticated, never from the raw rows.

**Why.** Postgres is outside the enclave and treated as adversarial. The private key was always encrypted, but its windows were plain columns, and the gateway believed them. A database could have made an old key current again, so that everyone issued under it in that window formed a set of a few; moved a rotation to isolate whoever issued at a chosen moment; or extended an expired key's acceptance after its nullifiers were pruned, which allows a double spend. Binding the windows into the ciphertext makes them the enclave's statement rather than the database's.

**Enforced.** `credentials::issuer_key_aad`, `encrypt_private_key` / `decrypt_private_key`, `cache_key` (the one path into use, which also checks the calendar, the separator and that the id is its key's hash), `load_key` (a key read for a spend is authenticated before its deadline is read), `list_keys`. Tests: `an_issuer_key_decrypts_only_under_its_own_record` (each field), `an_edited_key_row_is_refused` (each edit through the row, including relabelling a key as another epoch's, which stays on the calendar and is caught only by the binding).

**What would break it.** Reading a key's windows from the row anywhere but through `cache_key`; serving `/v1/keys` from the database; a field the gateway acts on that is not in the associated data. The stored format changed with this (the wrapping key's HKDF info is `eidola-issuer-key-v2`); there is no reader for the old one, and a database holding old rows is recreated rather than migrated.

**What it does not stop.** A database can still delete rows, or hide them from a gateway that has not yet cached them. Deleting the current key just before a target issues, so that a gateway mints a fresh one for that epoch, is a partition the binding cannot see, since the new key is genuine. That is what the next section is for.

## Key rotation and partitioning

**Decision.** An honest gateway has exactly one key per epoch, a distinct key for every epoch, and never rotates within one, and **the client holds it to that for itself**. It checks the **whole** listing before it records or requests anything: every entry under its domain separator must sit on the calendar, be named by the hash of its public key, and be the only key for its epoch and the only epoch for its key. One entry that fails refuses the listing, whichever entry it is: an entry that escaped the check (next epoch's key with a fractional-second window, a mislabelled id, a key listed for two epochs) would escape being recorded, and the record is what holds the gateway to its keys later. Windows are compared **byte for byte** with the calendar's own spelling of the epoch's instants (`issuer_calendar::rfc3339_utc`, which the gateway also serves): a parse is used only to find which epoch to compare against, never to decide, so nothing a parser rounds away (a fraction of a second, an offset, a normalized date) can carry a key onto the calendar. Then every entry is recorded per epoch, first seen wins (`issuer_key_epoch`, scoped to the gateway's base URL, a year of epochs kept), including the next epoch's key, listed a whole epoch before it issues; and every entry must agree with the record. The client refuses (`AppError::IssuerKeyRefused`) a key off the calendar (`NotOnCalendar`), a key other than the one recorded for its epoch (`SecondKeyForEpoch`), and a key offered for a second epoch, in one listing or against the record (`OneKeyForSeveralEpochs`). Nothing is charged; credentials already held keep working; the GUI says so in the reader's language and the CLI prints it. The refusal lasts until the gateway's listing is consistent again, which for a second key means the next epoch's key.

**There is no emergency rotation, and that is a decision.** A client cannot tell a replacement key minted for a good reason from one minted to single it out: both are a second key for an epoch it already has one for, and both come from the same gateway. Any rule that let some second keys through would be a rule a partitioning server could follow. So the honest response to a key that must be withdrawn is to stop issuing until the next epoch, while tokens already issued keep spending under their own keys. If the reason for withdrawing the key reaches the next epoch's key too (it was already published, so clients have recorded it), issuance stops for two epochs. That is the cost of making the property checkable, and it is accepted.

**Why, and against whom.** This check does not depend on trusting the gateway, so it defends against the same partitioning whoever attempts it:

- **the database host**, who can delete or hide key rows so that a gateway mints a second key for an epoch;
- **the operator**, who runs the gateway and could ship a build that mints keys per client;
- **a compelled operator**, required under a gag order to single out a user.

Each of them ends at the same place: a client being offered a key other clients were not. A client that has seen the epoch's key refuses the other one, whatever produced it. The gateway's own checks (the binding above, and refusing to mint a second key for an epoch it holds) are the first line against the database; the client's is the line that holds against Eidola itself. It is a property the client verifies, not a promise about conduct.

**What it still cannot prevent.**

- **A client's first sight of an epoch's key.** A new install, or a client that has fetched no listing since the previous epoch began, records whatever it is shown first. Because each listing carries the next epoch's key, a client that issues regularly records each key a whole epoch before it issues; a server wanting to single it out has to start that early and keep every listing consistent. A client seeing a server for the first time has no history to compare with ([gaps.md § A client's first sight of an epoch's key](gaps.md#a-clients-first-sight-of-an-epochs-key)).
- **Partitioning by refusing service.** A gateway honest about its keys can still refuse issuance to some accounts, or delay it, so that the set issued under a key is smaller than it could have been. The client can see that it was refused, not who else was.
- **Sets split by when tokens are issued**, within an epoch. That is a matter of issuance timing ([Issuance sizing](#issuance-sizing)), not of keys.
- **Each device's record is its own.** Two devices on one account each record what they see; they do not compare notes.
- **A database reset looks like an attack.** A gateway brought up on an empty database mints new keys for the current and next epochs, and clients that recorded the old ones refuse them for up to two epochs. That is the property working; an operator must not reset the database mid-epoch expecting clients to follow.

**Enforced.** `account_allocate`, `checked_listing`, `listed_key_epoch` and `listed_key_is_its_hash` in app-core, `db::pin_issuer_key_epoch` and the `issuer_key_epoch` table, `AppError::IssuerKeyRefused`; the GUI's `space-error-issuer-key-*` messages. Tests: `a_window_that_only_rounds_onto_the_calendar_is_refused` (fractions, offsets, a missing `Z`, a normalized instant), `one_key_listed_for_two_epochs_is_refused`, `two_keys_for_any_one_epoch_are_refused`, `every_listed_key_is_held_to_the_checks`, `an_honest_listing_is_accepted_in_epoch_order`, `a_key_off_the_calendar_is_refused_before_issuance`, `a_second_key_for_a_seen_epoch_is_refused_until_the_next_epoch` (a real second key that would have signed, refused before any request; held credentials still spend; the next epoch's key accepted), `a_key_kept_across_an_epoch_boundary_is_refused`, `a_key_seen_for_one_epoch_is_refused_for_the_next` (the listing leaves out the entry it would collide with; the record catches it), `the_key_seen_for_an_epoch_survives_a_restart`, `the_first_key_seen_for_an_epoch_stands`, `a_refused_issuer_key_says_so_in_the_readers_language`.

**What would break it.** Letting a lenient parse decide whether a window is on the calendar; checking only the current key, or filtering any entry out of the checks or the record (by id, by epoch, by order); recording a key only once it is used (the next epoch's key would no longer be pinned in advance); replacing a recorded key; scoping the record to anything that changes with a gateway release (a measurement), which would reset it exactly when a new build arrives; letting any second key for an epoch, or any key for a second epoch, through.

## The domain separator and the request context

**Decision.** The domain separator is a fixed constant, `ACT-v1:eidola:inference:production:2026-03-05`, never rotated on a schedule. The request context is `SHA-256("eidola" ‖ "inference" ‖ issuer_key_id)`, with an empty credential context, so every token under one key carries the same one. The token challenge carries an empty redemption context, since nullifiers already make each spend single-use.

**Why.** The anonymity set is everyone sharing a key **and** a domain separator **and** a request context. Rotating the separator would split it without adding anything key rotation does not already give. The request context is revealed at every spend of a token and of all its successors, so a context that differed between clients would make every one of their spends recognisable: it would be an account id in all but name.

**Enforced.** `credentials::compute_request_context` and `domain_separator` on the gateway (`test_domain_separator_is_stable`); each spend's context is checked against the key's (`credentials::verify_presented_spend`). The issuer chooses the context and the amount at issuance and the issuance proof does not constrain either to what the client asked for, so **the client checks both** before it stores a new token: `check_issued_credential` in app-core refuses a token whose context is not the one every client under the key shares, or whose amount differs from the one requested (`an_issuance_that_could_mark_its_credential_is_refused`).

**What would break it.**

- Any per-account, per-session or per-request value in the request context or the credential context.
- A change to the separator outside a planned protocol version change (every outstanding token becomes unspendable).
- Removing the client's check, which leaves the context to whatever the gateway sends.

## Key consistency

**Decision.** A key's id is the SHA-256 of its public key, as the Privacy Pass draft specifies. The client checks that the key it was told is current has the id its public key implies, that the issuance response names that same key, and that every refund names the key of the token it replaces; it stores tokens only under that verified id.

**Why.** The key id is what each spend names, so it decides which tokens look alike. A client that stored a token under whatever id a response named could be steered into spending under a label the gateway can sort by, or into a token whose spends name a key the gateway cannot find (stranded value). A refund labelled with a newer key would also make the client believe a token lives longer than it does.

**Enforced.** `account_allocate` and `process_refund` in app-core, with `issuer_key_id_of`; tests `an_issuance_naming_another_key_is_refused` and `a_refund_naming_another_key_is_not_stored`. On the gateway the `issuer_key` table's `id_is_key_hash` constraint.

**What would break it.** Storing a token under a server-supplied id without recomputing it; accepting a refund under a key other than the spent token's.

**What the client cannot check.** Whether other clients were given the same key in an epoch it saw first. Within one client's history, it can: see [Key rotation and partitioning](#key-rotation-and-partitioning).

## Issuance sizing

**Decision.** When a request needs a token and none in the wallet covers it, the client converts `min(available, max(DEFAULT_ALLOCATION_CREDITS, charge))` from the account: 1,000,000 credits ($1.00) in the usual case, the remaining balance when less is left, and the charge itself when one request needs more than $1.00. Issuance ahead of need (below) always converts exactly `DEFAULT_ALLOCATION_CREDITS`, and never from a balance smaller than that. `eidola account allocate` converts any amount the user asks for.

**Why.** The amount converted is visible at issuance, and the charge `s` is visible at each spend. A fixed amount makes issuances look alike, and an allocation large enough for many requests means issuances are rare. A small one keeps the value parked in any one anonymous token, and so the value at risk if it lapses (below), small.

**Enforced.** `auto_allocation_amount`, `DEFAULT_ALLOCATION_CREDITS` and `Inner::provision_once` in app-core.

**What would break it.**

- Sizing issuance to the request about to be made. If the converted amount equals the next spend's charge, the two are linked by amount alone. This happens today only when a single request's worst-case charge exceeds $1.00, which no catalogued model's maximum completion reaches, but a large model or a long context could.
- Converting oddly specific amounts often (the remaining-balance case does this once per account as its balance runs out).

**Several tokens per request.** A request may be paid from two tokens at once, so a token too small for the next request is drained rather than stranded, and issuance never has to be sized to a request (below, [Spending several tokens in one request](#spending-several-tokens-in-one-request)). Whether to move from one standard allocation to a ladder of denominations is not settled.

## Draining a token

**Decision.** A spend may take a token's whole remaining balance, and a request paid from two tokens does exactly that with the first ([Spending several tokens in one request](#spending-several-tokens-in-one-request)). The charge `s` of such a spend is the token's exact remaining balance: no rounding.

**Why it is acceptable today.** A token's balance is hidden by the protocol. Someone who saw that one spend's charge equalled a token's remaining balance could, with the earlier charges of the same token and the amount it was issued with, try to reconstruct its history (which earlier charges sum to the issued amount minus this one). That needs the earlier charges, and the charges and refunds of a chain are not linked to each other: each spend reveals a fresh nullifier, and **no spend amount, refund amount, or anything derived from them is persisted or observable outside the enclave**. The refund is stored sealed under a key only the client holds (below), and neither the charge nor the refund is logged.

**Considered and not adopted.** The client could round the drained share down to a small public step (preferred over random jitter, because the loss is bounded and predictable), at the cost of stranding the remainder. It is not needed while the invariant above holds. Neither is a deferral rule (no drain in the first request after an issuance): issuance ahead of need already separates issuances from the requests that follow them, and two issuances per key make the second ordinary.

**What it gives up.** A drained token's balance is known exactly to the measured gateway code for the length of its request. Each drained input is also known to come from the same client as the other input of its request. The privacy promise for drained tokens therefore rests on the measured code retaining nothing ([gaps.md § A drained balance is visible to the measured code](gaps.md#a-drained-balance-is-visible-to-the-measured-code)), not on the cryptography.

**What would break it.** Persisting, logging or exporting a charge, a refund amount, or anything derived from them, anywhere outside the enclave. That would reopen this question.

## Issuance timing

**Decision.** The long-lived app keeps a small buffer of credit tokens, issued ahead of need at times unrelated to any request. At intervals drawn uniformly between 10 and 50 minutes (the first one interval after the app starts, never at launch), it looks at the wallet. When what the wallet can spend for at least the next 24 hours is below a quarter of the standard size (`DEFAULT_ALLOCATION_CREDITS`, $1.00), it issues tokens of that size: **two under a key the account has not yet been issued under** (one more where it already has an issuance under the key, such as a just-in-time one), and one at a time once it has two. It does not watch for requests, so an issuance happens at the first look after the buffer runs low: a random draw, not a reaction to a spend. Just-in-time conversion, immediately before the request it pays for, remains as the fallback when nothing held covers a request.

The buffer is kept only within limits that do not depend on the request that will follow:

- **Only for a wallet that has spent lately.** No buffer is kept unless a spend began within the last seven days. A token's value dies with its key at most two epochs after issuance, so a token issued for a profile nobody is using would most likely lapse unspent. This reads local history, but only to decide *whether* a buffer is kept; when and how much is issued are public constants either way, and an observer of the account surface learns from it only what the account's issuances already show, that the account is in use.
- **Never beside an unresolved issuance.** An issuance whose response was lost may already have been paid for, so it counts toward the buffer, and while one for the account is unresolved (its retry hold-off included) nothing new is issued. Otherwise each pass whose recovery failed would convert another allocation. The same holds at every door: just-in-time conversion converts nothing while an issuance for the account is pending, and `eidola account allocate` first asks again for the account's pending issuances and converts nothing while one stays unresolved ([Lost issuance responses](#lost-issuance-responses)).
- **Never for a backend the reader turned off.** With the Eidola backend disabled or removed, the provisioner sends nothing, not even the retry of a pending issuance.
- **Only the standard size, from a balance that covers it.** A balance below one standard allocation is left for the just-in-time path, which converts the remainder when a request needs it. Sizing is unchanged (see Issuance sizing).
- **Only under a key the client accepts.** Background issuance reads the current key through the same check every issuance does: the listing must be on the public issuer calendar, one key per epoch, agreeing with the per-epoch record, or the key is refused before anything is sent.
- **Only under a key that will carry it.** Nothing is issued under a key whose spend deadline falls within the 24-hour horizon, since the token would not count toward the buffer and could not carry the days ahead; issuance is refused outright under a key past its spend deadline.
- **Early in a key's life where it can wait.** Within 12 hours of a key's `issue_until`, while what is held still covers several worst-case requests, the provisioner waits for the next key: a token issued at the start of a window is spendable for up to two epochs, one issued at its end for one.
- **Two per key.** A request paid from two tokens under one key shows that its client was issued at least two under that key. Issuing two whenever the provisioner first issues under a key makes that every account that provisioned under it, not the heavy users. A pair left at one (the second never made, because the first response was lost and recovered only on a later pass) is finished on a later pass even when the buffer is covered, as long as one more token keeps what is held within its bound, so such a request is placed among the same accounts a single spend is. Both are the one public denomination, so no new amount appears on the issuance surface, and with at most two tokens per request nothing a request presents tells two issuances from more.
- **Not spent first.** Inputs are taken from the soonest-lapsing key, smallest first, and the buffer is replenished while the old token still holds a quarter of an allocation, so the next requests are paid by the old token and the new one's first spend comes when the reader's own use takes the old one below a request's hold (when it is drained, with the new one filling).

What is held at once, unresolved issuances included, is at most the threshold plus two allocations, so a buffer costs little more than just-in-time conversion would put at risk of lapsing anyway (`what_is_held_stays_within_the_threshold_plus_two_allocations` holds that while every retry fails).

**Bounded waits.** Every request an issuance makes (the key listing, the balance read, the issuance) has a 30-second deadline, connection to last byte, and a whole background pass is cut off at five minutes; a request given up on leaves its issuance pending, to be asked for again. Anything waiting for the issuance gate waits at most the provisioning queue's 30 seconds and then fails with a typed timeout. That matters because a turn waits for the gate while holding the spend gate: a server that accepted a request and stalled would otherwise stall every paid turn behind it.

**Where it runs.** In the process that holds the profile and answers for it over the control socket: the app (`AppCore::start_credential_provisioner`, started in `stores::open_app_core`, which the windowless service also goes through). A one-shot CLI invocation does not start it. The same loop asks again for spends left unsettled, at a random point in the middle half of each interval, so that unlinked request and the account-authenticated pass are never adjacent ([Refund recovery](#refund-recovery)). Every issuance in a process, background, just-in-time, explicit and recovered, is serialized by one gate, so two never convert for the same need. There is no setting: turning it off would only restore the linkable default, and `eidola account allocate` still converts any amount by hand.

**Enforced.** `Inner::provision_once` and the `PROVISION_*` constants in app-core (`PROVISION_CREDENTIALS_PER_KEY`); `db::spendable_credits_until`, `db::last_spend_at`, `db::issuances_under_key`. Tests in `chat_path.rs`: `the_provisioner_tops_up_a_wallet_running_low`, `the_provisioner_issues_two_credentials_under_a_key_it_has_not_issued_under`, `a_pair_interrupted_by_a_lost_response_is_finished_later`, `a_background_credential_is_not_spent_by_the_next_request`, `the_provisioner_keeps_no_buffer_for_a_wallet_that_has_not_spent`, `the_provisioner_never_issues_from_a_balance_below_a_standard_allocation`, `the_provisioner_never_issues_under_a_key_about_to_lapse`, `the_provisioner_prefers_a_keys_early_life`, `the_provisioner_issues_only_after_its_interval`, `the_just_in_time_path_remains_the_fallback`, `an_unresolved_issuance_holds_off_a_fresh_one`, `what_is_held_stays_within_the_threshold_plus_two_allocations`, `the_provisioner_leaves_a_disabled_backend_alone`, `a_stalled_issuance_gives_up_at_its_deadline`, `a_turn_waiting_on_a_stalled_issuance_times_out`.

**The cost that remains.** The fallback is the old behaviour: when nothing held covers a request (a fresh install, the first request after a quiet week, a CLI used without the app running, a request larger than what is held), the token is converted and spent at once, and that first spend can still be paired with its issuance ([gaps.md](gaps.md#issuance-to-first-spend-timing)). A background issuance is still timed: it follows, by one random interval, the spend that took the buffer below the threshold. Its timing therefore says that some spend in the preceding hour or so drained a token, which is weaker than a pairing and is shared by every spend in that window.

**What would break it.**

- Issuing in reaction to a request: on a spend, on a turn starting, on the app coming to the foreground.
- Sizing, timing or the threshold from anything about the requests themselves, or from usage history beyond whether a buffer is kept at all.
- Issuing smaller tokens more often, or issuing per request.
- Issuing fewer than two under a key the provisioner issues under, which leaves two-token requests to the heavy users' accounts.
- Spending the newest token first.

## Issuance concurrency

**Decision.** Debiting an account takes a lock on the account's row for the rest of the transaction, so two issuances against the same account run one after the other. An issuance takes the lock before it looks for a response already stored under its idempotency key, so a retry racing its own original waits for it and then finds what it stored.

**Why.** An issuance checks the balance and then debits it. Without the lock, two concurrent issuances each read the same balance, each pass the check, and both convert it: a client sending issuance requests in parallel could convert its balance once per request. The tokens cannot be traced or recalled once issued, so an overdraft here is pure loss.

**Enforced.** `debit_account` and `record_credential_issuance` in the gateway's `schema/schema.sql`; `concurrent_issuances_cannot_spend_one_balance_twice` and `an_issuance_racing_its_own_retry_is_debited_once` (`credentials.rs`, against Postgres).

**What would break it.** Any balance check and debit that are not serialized per account, including a refund, dispute or closure path that bypasses the account row lock. Paid balance changes belong to the shared billing library and preserve the purchase allocation that `debit_account` consumes.

## Lost issuance responses

**Decision.** Issuance is idempotent per account. Before it first sends an issuance request, the client writes down everything a retry needs: its secret issuance state, the request exactly as sent (its proof is randomized, so it could not be rebuilt), 32 random bytes naming the request to the gateway (the idempotency key), and the account it is for. The gateway signs, then, in one transaction under the account's row lock, looks for an issuance already stored under that account and key: if there is one it answers with the stored response and debits nothing; otherwise it debits and stores the response beside the debit. A retry with the same key but a different request (another commitment, amount or key) is refused with `409`. The request also names the issuer key the client expects, and the gateway never debits under a key other than the current issuing key, so a token is never paid for under a key the client would refuse. Whether the named key has stopped issuing is decided under the account's row lock, on the database's clock read after the lock is taken, both when a request names a key that is not current (refused with `409` once it has stopped, with "try again" (`503`) while the database still has it issuing) and again at the debit itself. An attempt that passed the current-key check and then stalled past its key's issue window therefore debits nothing, however late it reaches the database.

A retry is sent under the key the issuance was made under, read from the local record, never the key that is current when it is sent: the stored response answers it even after the epoch has moved. The client stops asking at that key's spend deadline, inside the hour the gateway keeps the stored response past `accept_until`.

The client asks again for any issuance whose response never arrived (no answer, or a server error) on each provisioner pass, before a conversion `eidola account allocate` asks for, and on `wallet credentials recover` in a process that runs no provisioner; only for the account it was made for, and only once it is ten minutes old, so a retry never races an original attempt the gateway may still be processing. Recovering a spend never sends one, and neither does a request being paid: a request may just have asked for an unsettled spend, and a retry is account-authenticated ([Refund recovery](#refund-recovery)).

**Nothing is converted beside an unresolved issuance.** While an issuance for the account is pending (too recent to ask for again, or asked for and still unanswered), nothing is converted. A request that needs a conversion sends nothing to the account surface: it can still be paid by a token another request or a recovery returns within its bounded wait, and otherwise fails with a typed "try again", until the provisioner's pass or an explicit door has resolved the issuance. `eidola account allocate` asks again for the pending issuance itself, fails the same way while it stays unanswered, and when it recovers one reports it and converts nothing more. Otherwise each attempt repeated after a lost answer would be a new request under a new idempotency key, the earlier ones would still be asked for again and debited if they never had been, and a run of failed attempts would convert once per attempt.

A refusal the gateway's issuance path itself gives a first attempt (`400`, `401`, `402`, `409`, `422`, `428`) means nothing was debited, since it is the only request under its key; any other status to a first attempt is not that path's verdict, and the issuance stays pending. A retry's refusal answers only the retry: the original attempt may have passed every check before it and still be on its way to the debit. So a retry is abandoned only on a refusal that no attempt still in flight can overturn: a malformed request (`400`, `422`), decided by bytes the first attempt shared, or `409`, which the gateway gives only when a response is stored under the key for a different request or when the named key has stopped issuing by the database clock under the account lock, the same clock and lock the debit checks. A refusal that turns on state that can change is not proof: the balance (`402`) may have grown by the time a stalled original debits, and the terms in force (`428`) may have changed after the original passed them, so these stay pending, like an authentication failure (`401`, `403`, decided before the stored lookup: a relinked account with a mistyped secret would otherwise abandon an issuance it had paid for), `429`, any other `4xx`, a `5xx` or no answer. A pending issuance still ends: the gateway checks the key before the terms, so once the key stops issuing the next retry is answered with the stored response or the final `409`, and the client stops asking at the key's spend deadline in any case. Stored responses are pruned with their key's nullifiers, an hour past `accept_until`; by then the key has long stopped issuing, so a retry arriving after the prune is refused rather than debited again.

**Why it costs no privacy.** Everything stored is on the account surface the issuance was already on, beside the ledger row that records the account, amount, key and time: the account, the idempotency key, a digest of the request, the key, the amount, the ledger entry, and the signed response. The response is what the issuer computed and could always have kept; issuance is blind, so it does not recognise the token when it is spent. The row carries no timestamp of its own. The idempotency key is random, is never sent with a spend, and is never derived from anything a spend reveals.

**Enforced.** `record_credential_issuance`, `resolve_issuance_off_current_key` and `prune_expired_issuances` in the gateway's `schema/schema.sql`; `credentials::issue_credentials`, `issuance_request_digest`; tests `a_retried_issuance_is_answered_once_and_debited_once`, `an_issuance_racing_its_own_retry_is_debited_once`, `an_attempt_stalled_past_its_keys_window_never_debits_after_a_retry_is_refused`, `stored_issuances_are_pruned_on_their_keys_deadline` (against Postgres). In app-core, `Inner::allocate_with_key`, `settle_issuance`, `first_refusal_is_final`, `retry_refusal_is_final`, `recover_pending_issuances_locked`, `resolve_pending_issuances_locked` and the `pre_credential` columns; tests `a_lost_issuance_response_is_recovered_without_a_second_debit` (across a restart), `a_turn_retried_after_a_lost_conversion_never_converts_again`, `a_request_never_retries_an_issuance_beside_a_spend_recovery`, `a_pending_issuance_does_not_stop_a_request_waiting_for_a_refund`, `an_explicit_conversion_repeated_after_a_lost_answer_recovers_it_instead`, `a_first_attempt_refused_by_something_else_stays_pending`, `the_provisioner_recovers_a_lost_issuance_instead_of_issuing_again`, `a_refused_issuance_is_not_asked_for_again`, `a_retry_refused_while_its_original_may_still_land_is_kept`, `a_pending_issuance_is_never_retried_for_another_account`.

**What would break it.**

- Debiting outside the transaction that stores the response, or looking for a stored response without the account lock.
- Answering a stored response for a request whose digest differs.
- Checking the terms gate or the current key before the stored lookup: a paid-for issuance would then be refused on retry, and abandoned.
- Refusing a named key as no longer issuing without the account lock, or on a clock other than the one the debit checks, or debiting without checking under the lock that the key is still issuing: an original stalled across the key's issue window could then debit after its retry was refused and abandoned.
- Treating a retry's refusal on mutable state (the balance, the terms) as proof that nothing will be debited.
- Converting at any door while an issuance for the account is unresolved.
- Abandoning a first attempt on a status the issuance path does not give.
- Generating the idempotency key after the request was sent, or reusing one across issuances.
- Pruning a stored response while its key could still issue.

**What it does not cover.** An original attempt that the gateway is still processing more than ten minutes after the client gave up on it could be refused on a retry, abandoned, and then debited. The gateway answers an issuance in well under a second, so this needs a server stalled for minutes mid-request.

## Spending and the nullifier

**Decision.** A spend's proof is verified first, under the key it names, which must still accept spends. Then its nullifier is inserted into Postgres, with the spend's provisional refund stored on the same row in the same statement (below). Only an insert that wins is served; a duplicate is refused with `409`. From that point every path out of the handler issues a refund.

**Why.** Recording before serving is what prevents a fork. The `(issuer_key_id, value)` primary key is enforced by the database every gateway instance shares, so a token spent concurrently against two instances is served once. Verifying before recording means an invalid proof never writes a row.

**The wire form.** A spend travels in the JSON request body and nowhere else, as its top-level `credentials` member: an array with one element per spend, each exactly `{ "token": "<base64url(Token)>", "refund_key": "<64 lowercase hex digits>" }`. The Token is the one `draft-schlesinger-privacypass-act` defines (`token_type`, `challenge_digest`, `issuer_key_id`, the spend proof), base64url *with* padding. A request carries one spend or two ([Spending several tokens in one request](#spending-several-tokens-in-one-request)); the member is an array so that every count has one encoding. There is one spelling: the client writes it and the gateway reads it through the same rule (`eidola_common::spend_credentials`), which refuses a Token that does not re-encode to itself (missing padding, non-zero pad bits, whitespace, another alphabet), a refund key that is not exactly lowercase hex, an unknown, missing or repeated member, a member that is not an array, no spend or more than two, and a spend proof with bytes after it. A credential in a header (the RFC 9577 `Authorization: PrivateToken` header, or a refund-key header) is refused, so there is one transport and nothing to reconcile between two. All of this is refused before anything is recorded. The gateway removes `credentials` before a request is forwarded to an engine. Every answer to a spend carries its refund in a `refunds` array, one element per spend in presentation order, in a blocking response, an error body, the streamed final metadata event and a recovery answer alike.

**Why the body.** A Token at `L = 128` was about 24 KB as base64url, beyond common header limits (8 to 16 KB in nginx, Apache, Node.js, and the HTTP/2 server on Eidola's own path); it worked only because TLS terminates inside the enclave, so every hop that parses the request is one Eidola configures. A request body has no such limit, the request is read whole before it is authorized anyway, and issuance already carries its messages as body content. This departs from RFC 9577, so a generic Privacy Pass origin would not read it; Eidola's client talks only to Eidola's server.

**The size of a balance, `L`.** A credential's balance lies in `[0, 2^L)` credits with `L = 32` (`eidola_common::spend_credentials::ACT_L`, the one constant client and server use): up to about $4,294 at a millionth of a dollar per credit, over four thousand times the standard $1.00 allocation. The ACT specification leaves `L` to the deployment and signals it nowhere, so both sides fix it at build time. A proof's size and cost are linear in `L`; it bounds only the largest representable balance and changes nothing a proof hides. An issuance or a spend of `2^32` credits or more cannot be made.

**The cost of verifying.** Verifying a proof is the expensive step (about 1.8 ms with its refund at `L = 32` with `anonymous-credit-tokens` 0.4.2, on an Apple-silicon laptop; enclave hardware will differ), and a chat spend pays it twice: once before the nullifier is recorded and once when its final refund is signed. The crate has no verify-only operation: it verifies a proof only as the first half of signing a refund for a fixed amount, and the refund commits to that amount, so a refund signed before the request runs cannot be adjusted to the amount the response settles at. The pre-flight verification signs a refund of nothing, which is kept as the spend's provisional refund (below). Removing the second verification needs the crate to split verification from signing. Recovery knows its amount (the whole charge) before it reads the database, so it verifies once and keeps the refund that verification signed (see Refund recovery below).

**Enforced.** `handlers::chat_completions_phases`, `handlers::verify_spend_proof`, `credentials::verify_presented_spend`, `credentials::record_spend`, `db::record_nullifier`; every call into the verifier is `credentials::verify_and_refund`'s (`the_verifier_is_called_only_through_its_counting_seam`), and `a_chat_spend_is_verified_at_preflight_and_once_to_sign_its_refund` counts the two. The wire form: `eidola_common::spend_credentials` (its tests cover every refused spelling and shape), `auth::presented_spend` and `auth::refuse_header_credentials` (`a_spend_token_is_accepted_only_as_padded_base64url`, `a_spend_without_its_refund_key_is_refused_before_it_is_taken`, `a_credential_in_a_header_is_refused`), `engine_request_body` (which drops the member), and the chat harness's mock gateway, which reads the member independently of the shared rule and refuses a credential in a header. On the client, the member is put onto a copy of the body as it is sent (`body_with_spend`), so the body the Record keeps never holds it (`a_spend_travels_in_the_body_and_the_record_never_keeps_it`).

**What would break it.**

- Serving before the insert, or keeping an in-memory set of spent nullifiers in place of the shared table. With more than one instance, the same token could then be served once per instance.
- A path that records a nullifier and returns without a refund. The client loses the token's whole remaining value, not just the charge.

## Holds, the completion ceiling and the pricing contract

**Decision.** The client presents a charge `s` that is exactly the request's worst case: the prompt priced by the shared contract and the completion priced at its ceiling. The gateway recomputes the same worst case from the same request and refuses a charge that differs from it, below (`402`) or above (`400`), before anything reaches an engine; the refusal comes after the nullifier is recorded, so it carries a full refund. When the request completes, the gateway charges what was used, clamped so it can never exceed the hold, and refunds the rest as `t`.

**Why exactly.** A hold below the worst case could be charged more than it holds. A hold above it is a number the client chose beyond what the request needs: a degree of freedom a client could be recognised by. And with the hold exactly the worst case, a request settled without reported usage, which is charged the whole hold, is never charged more than the worst case.

**Why.** The charge must be fixed by the request, before any work, with nothing left for later. The client and server compute it with the same function of the same bytes (`eidola_common::prompt_charge`, which counts content and tool bytes at a safe cost factor), so the client's hold always covers the gateway's charge and never has to be topped up. The completion side is the model's declared `max_output_tokens`: a request asking for more is refused (with a full refund) rather than quietly shortened, and a request naming no limit is given the declared one, so what the engine is asked for, what was priced and what can be charged are one number. `PRICING_MARKUP` may not go below the contract's cost factor (1.5), or prompt bytes would be sold below cost.

**Enforced.** `handlers::worst_case_cost`, `check_hold_equals_worst_case`, `check_completion_ceiling`, `actual_cost` and `settled_cost`; `backend::validate_pricing_markup`; the client's `estimate_charge_credits`, `completion_ceiling` and the local proxy's clamp. Tests `client_and_server_prompt_terms_agree`, `preflight_requires_the_hold_to_equal_the_worst_case`, `the_completion_side_is_held_to_the_declared_ceiling`, `token_dense_usage_is_clamped_to_the_contract`; and the chat harness's mock gateway, which computes the worst case from every body it receives and refuses any other hold as the gateway does, so every harness test of a paid request checks the client holds exactly (`MockServer::hold_mismatches`).

**What would break it.**

- Any input to the price the client cannot compute exactly in advance.
- A charge that depends on how the engine counts tokens without the contract's clamp.
- Charging after the fact, or for a completion that ran past what was priced.
- Accepting a hold other than the worst case, or a client holding more than it.
- A request field that makes the engine do more work than was priced (more output, more choices, a longer context).

## Spending several tokens in one request

**Decision.** A request presents one input or two (`MAX_CREDENTIALS_PER_REQUEST`, 2), each an unmodified ACT spend under its own key. The client chooses them by a rule with no thresholds: its tokens in order of their key's `accept_until` (soonest-lapsing first), then balance (smallest first), then nullifier; if the first covers the request's worst case `W` it pays alone with share `W`; otherwise the first token in the order for which some later token covers the rest is **drained** (its share is its whole balance `a`), and the first such later token **fills** with share `W − a`. That is normally the first token; a later one is drained only when the first has no fill but another pair covers `W`, so a wallet able to pay is never sent to convert more. Only when no pair covers `W` is a new token converted, to fill beside the first token's drain (or to pay alone, for an empty wallet), and the rule runs again. The shares sum to exactly `W`. Both inputs are marked spending in one local transaction, under the wallet's spend gate. The gateway checks every input before verifying any (a share of at least one credit, nullifiers distinct per key), verifies each, records every nullifier in one transaction or none, and treats the sum of the shares as the hold. When the request ends, its charge is consumed from the inputs **in the order presented** (`eidola_common::spend_credentials::partial_returns`): the first input pays first, and the change lands on later ones. That holds for every charge, a stream cut short included, which is charged what was generated and split by the same rule. Each input enters the gateway's live set before the request's nullifiers are recorded and leaves it after its own final refund's write, so recovery calls each input's refund provisional exactly while its request is served; each input's refund is signed under that input's own key, stored over its own provisional refund, recovered by its own nullifier, and settled by the client when its own refund is final. Inputs may be under two keys; the rule produces that only when an older key's residual has no fill under that key.

**Why.** A token whose balance is below the next request's hold would otherwise be stranded until its key lapses, and a hold above any one token's balance would need an issuance sized to it, which links the account to the request by amount. Draining recovers residuals without either. No value moves from one input to another, so none moves between keys or epochs: a refund may return at most its own input's share (ACT's `t ≤ s`), and every successor is under its own input's key. The allocation is fixed by the shares, their order and the charge, so the issuer has no choice of partial return with which to mark a successor, and the client checks it (`check_partial_returns`, a mismatch reported as the gateway's fault). Recording all or none means a request refused because one input was already recorded (a replay, or a recovery that raced it) leaves the other unrecorded, and its holder recovers it whole; it also means no input is ever half-spent by a crash.

**What it leaves alone.** ACT itself is unchanged. Every input is an ordinary spend proof, verified as one and answered by an ordinary refund whose partial return is within the bound ACT already enforces. What is added sits above the cryptography: how inputs are carried, the all-or-none recording, the in-order allocation, and per-input recovery.

**Considered and not adopted.**

- A merge endpoint, where a separate request presents several drained tokens under one key and the issuer mints one token of their total. Inference requests would stay single-input and merges could run at random times. But the issuer would know the minted token's value, which it never knows of a refund's successor, and that value would not be the public denomination, so the token would be recognisable. It also needs a minting path on the anonymous surface, which does not exist and which an ACT refund cannot provide.
- An aggregate proof over several tokens with hidden shares and one successor. It would remove the drain's disclosure, but it is new cryptography that needs its own security proof, and it would still need same-key inputs to keep value inside its epoch.
- Rolling a token's remainder into the current key when its key lapses, which moves value into a later epoch, and issuing tokens sized to the hold, which links the account to the request by amount.

**What it discloses.** Inputs presented together are known to come from one client: a request with two inputs under one key places its client among the accounts issued at least two tokens under that key, which two-per-key provisioning makes every provisioning account ([Issuance timing](#issuance-timing)). Two inputs under two keys place it among the accounts issued under both. A drained input's share is its token's exact balance, visible to the measured code for the length of the request and retained nowhere ([Draining a token](#draining-a-token)). The database sees the input count and keys of each request, since a request's nullifier rows commit together ([What Postgres may hold](#what-postgres-may-hold)), and a network observer sees the count from the request's size; neither sees an amount.

**Enforced.** In `eidola-common`, `spend_credentials::{MAX_CREDENTIALS_PER_REQUEST, hold, partial_returns}` (`the_charge_is_consumed_in_input_order`, `the_allocation_is_exact_bounded_and_a_prefix`). In the gateway, `handlers::check_presented_spends`, `PaidInputs`, `settle_inputs`, `credentials::record_spends`, `db::record_nullifiers`; tests `several_inputs_are_checked_before_any_is_verified`, `a_two_input_request_settles_each_input_under_its_own_key`, `each_input_of_a_two_input_request_has_exactly_one_successor_nullifier`, and against Postgres `a_requests_inputs_are_recorded_all_or_none`. In app-core, `choose_inputs`, `Inner::ensure_spend_inputs`, `Inner::acquire_spend`, `db::spendable_credentials`, `db::insert_spend_inputs` (the inputs of a request share a `spend_group`), `SpendPrep::settle`, `check_partial_returns`; tests `inputs_are_the_first_credential_or_a_drain_and_the_first_fill`, `allocation_happens_only_when_no_pair_covers_the_hold`, `partial_returns_are_checked_against_the_in_order_allocation`, `a_requests_inputs_are_taken_together_or_not_at_all`, and in `chat_path.rs` `a_credential_short_of_the_hold_is_drained_and_the_rest_filled`, `a_drain_with_change_is_drained_again_on_the_next_request`, `a_new_credential_fills_what_the_wallet_cannot`, `a_later_pair_pays_when_the_first_credential_has_no_fill`, `an_older_keys_residual_is_drained_into_a_newer_keys_fill`, `every_exit_of_a_two_input_round_settles_every_input`, `one_inputs_recovery_waits_while_the_others_is_final`, `a_settled_input_is_announced_while_the_other_still_waits`, `concurrent_turns_never_share_an_input`.

**What would break it.**

- More than two inputs without two-per-key provisioning rising to match, or a fill chosen by anything but the rule (the largest, the best fit, a random one), or a share sized from a balance other than a drain's.
- Recording some of a request's nullifiers without the others, or settling an input's refund under another input's key, or moving a partial return between inputs.
- An allocation other than in input order, or one the gateway chooses.
- Marking a request's inputs spending outside one local transaction or outside the spend gate, so two requests could take one token.
- Persisting which nullifiers were spent together, or any share, anywhere outside the enclave.

## Price independent of the prefix cache

**Decision.** A request costs the same whether or not an engine served part of its prompt from its prefix cache. The settled charge reads only the reported prompt and completion counts, never `cached_tokens`.

**Why.** The refund `t` is the hold minus the cost. With a cache discount, the cost would depend on whether this prompt's prefix had been served before, so the refund would say whether an earlier request shared it: a link between two requests, written into a value the gateway computes and the client receives. A discount for a declared, client-chosen cached prefix that the client can price in advance is planned; it is not offered today.

**Enforced.** `the_settled_charge_ignores_prefix_cache_hits` (`handlers.rs`).

**What would break it.** Reading `prompt_tokens_details.cached_tokens`, or any engine-reported value that depends on other requests, into the charge.

## Continuations

**Decision.** A request continuing a partial answer (`continue_final_message`, an assistant message with `reasoning_content`) is priced by the same contract: the partial answer and its reasoning are prompt bytes like any other.

**Why.** Otherwise continuing would be a way to buy completion at prompt prices, or to have the engine produce more than was priced.

**Enforced.** The strict request type in `types.rs` and `eidola_common::engine_protocol::check_continuation`; the pricing walk includes `reasoning_content`.

**What would break it.** A continuation field the pricing walk does not count.

## Refunds stay under the spent token's key

**Decision.** A refund is always signed under the key of the token it replaces, never under the current issuing key. This holds on every path: a blocking answer, a stream, an error after the nullifier, and recovery.

**Why.** A successor under the current key would let a token near the end of its life be renewed for a further epoch by spending one credit. Repeated, value bought once would live forever, past the expiry of the credits it was bought with, and the gateway would have to keep accepting it.

**Enforced.** `credentials::sign_refund` and recovery's `verify_and_sign_full_refund`, which take the key from the spend and from nowhere else; `a_refund_is_signed_under_the_spent_credentials_key`. On the client, `process_refund` verifies the refund against the spent token's public key and stores the successor under that key's id.

**What would break it.** A refund path that signs with "the current key", or a client that accepts a successor under another key.

## Expiry, on both sides

**Decision.** A key's tokens are accepted until `accept_until`, on every path, whether or not the key is cached. The gateway keeps the key for one more hour (`RETENTION_PAST_ACCEPT`) only to finish spends already accepted: signing their refunds and answering recovery. It never accepts a new spend in that hour. The client stops *starting* spends ten minutes before `accept_until` (`CREDENTIAL_SPEND_MARGIN`) and stops recovering at `accept_until`, so its whole window ends inside the gateway's.

**Why.** Too late: a key that is still accepted after its nullifiers are pruned allows a double spend; a key accepted past its deadline extends value. Too early: a client that treats a token as expired while the gateway would still take it strands paid-for value. The margins cover clock disagreement between the client and the gateway, between instances, and between instances and the database (each gateway refuses to rotate keys while its clock and the database's differ by more than ten seconds, `db::MAX_CLOCK_SKEW`).

**Enforced.** `credentials::key_usable`, `usable_key` and `load_key`; the compile-time assertion that `RETENTION_PAST_ACCEPT` exceeds twice `MAX_CLOCK_SKEW`; `a_cached_key_past_accept_until_refuses_a_spend`; the client's `issuer_key_deadlines` and the `credential_lifecycle` view, which the provisioning queue, recovery and every wallet listing read. Issuance refuses a key whose spend deadline has already passed, before any debit.

**What would break it.** Reading a key straight from the cache without its deadline; a client lifetime computed from `issue_until`; any path that accepts a fresh nullifier after `accept_until`.

## Value and the window it was bought for

**Decision.** A token lives until its key's `accept_until` and no longer, and its change stays under the same key. Account credit carries its own expiry (a subscription's credits expire at the end of the billing period, a purchase's after a year). Credit converted just before it expires becomes a token that is good until its key stops accepting: between one and two epochs (7 to 14 days by default) after the conversion. That bounded grace is the most value can outlive its window; nothing extends it further.

Value still in a token when its key lapses is lost. There is no rollover, because exchanging an old token for a new one without linking the two needs its own design. Small automatic allocations keep the amount at risk to one allocation's change per key; see [gaps.md § Credit left in a credential when its key lapses](gaps.md#credit-left-in-a-credential-when-its-key-lapses).

**What would break it.** Any rollover, renewal or refund path that carries value into a later key without its own unlinkability analysis; any way to convert credit whose expiry has already passed.

## Refund recovery

**Decision.** A client whose response was lost asks `POST /v1/credentials/refund` with the same spend, in the same body form (`{"credentials": [{"token", "refund_key"}]}`, exactly one element). Each input of a two-input request is recovered on its own, by its own nullifier and key, and stays pending until its own refund is final. The endpoint verifies the spend proof first, under a key that is still retained, and only once: that verification signs the full refund the never-arrived branch returns, which leaves the gateway only if that branch records the nullifier. Then:

- if the nullifier was recorded and a refund is stored, it returns that refund, with `"provisional": true` if it is still the provisional one and this gateway is still serving the request, and `false` if it is the final one or the provisional one is the spend's last word (below);
- if the nullifier was never recorded and the key still accepts spends, the request never arrived: it records the nullifier with the full refund stored beside it, in one statement, and returns that refund (`"provisional": false`);
- a stored refund that does not open for the presented key, or a nullifier recorded by the request a moment after the read, is answered `404`.

The client never asks for recovery of a spend it is still waiting on (`LiveSpends`), because recovery would record the nullifier and the real request would then be refused as a replay.

**Who asks again, and when.** A spend whose refund has not arrived, or was answered provisionally, stays `spending` until somebody asks for it. That is:

- the turn itself, once, as it ends without its refund;
- a later request that what the wallet holds cannot pay, before anything is converted for it and once more while it waits, with the last quarter of the wait kept for the answer, so a recoverable token is used before the account is asked for another;
- the long-lived app, once in each provisioner interval ([Issuance timing](#issuance-timing)), at a random point in the interval's middle half, and when it starts;
- `eidola wallet credentials recover`.

Recovering a spend is an unlinked request, named by the spend alone, and the client never sends the retry of a lost issuance beside it: that retry belongs to the provisioner's pass, which is at least a quarter of an interval away from the look for unsettled spends on either side, and a request being paid never sends one. What a request may send after asking for an unsettled spend that did not pay it is the just-in-time fallback (the balance read and the issuance), whose cost is the one [Issuance timing](#issuance-timing) already states, and only when no issuance is pending. The one exception is the reader's own `wallet credentials recover` in a CLI running without the app, which asks for unsettled spends and then for pending issuances, because nothing else in that process will; that places one account-authenticated request beside the recovery of a spend, at a time the reader chose. With the app running, the command recovers spends only and leaves issuances to the app. A one-shot CLI has no periodic look: its unsettled spend is asked for again when a later request needs it, or by that command.

**Why.** Recovery is what keeps a dropped connection from costing a token's whole balance. Verifying first means only the holder of a valid spend proof (the client that made it) learns anything, and an invalid proof writes nothing: without that, anyone could write nullifier rows under a public key id, or ask what became of a spend whose nullifier they knew.

**Enforced.** `credentials::recover_refund`, `credentials::verify_and_sign_full_refund`, `recovery_refuses_an_unverified_proof_before_the_database` (which also counts one verification per ask); the client's `Inner::recover_stranded_spends` (behind `recover_spending_credentials`, `ensure_spend_inputs` and the provisioner loop) and `try_refund_recovery`; tests `a_hold_whose_turn_ended_without_settling_it_is_recovered`, `a_stranded_hold_is_asked_for_again_before_anything_is_converted`, `the_long_lived_process_asks_again_for_a_stranded_hold`, `recovering_a_stranded_hold_sends_nothing_to_the_account_surface`.

**What would break it.** Reading or writing a nullifier row before verifying; refunding in the never-arrived branch after `accept_until`; a recovery that answers differently by anything other than the spend's own state; a client that retries an issuance, or sends anything else account-authenticated, in the same pass as the recovery of a spend.

**What it does not cover.** A gateway that crashes between recording a nullifier and settling it leaves the provisional refund as the spend's last: the client recovers the token's untouched balance `c − s`, and the hold `s` is lost (below, and [gaps.md](gaps.md#a-crash-mid-spend-costs-the-requests-hold)).

## Provisional refunds, the live set, and ending a request early

**Decision.** When the gateway records a spend's nullifier it stores, in the same statement, a sealed refund of nothing (`t = 0`, the refund its pre-flight verification signed anyway), marked provisional inside the seal. Before it records the nullifier, the gateway process puts the spend into an in-memory **live set** of the spends it is serving, and takes it out only after the request's final refund has been written, or its write attempted, whatever the outcome. Whenever the request ends (refused after recording, failed at the engine, completed, or cut short), the final refund overwrites the provisional one. Recovery of a provisional refund answers `"provisional": true` while this process's live set holds the spend; otherwise no request here can still settle it (a crash or a restart left it), and recovery declares it final at once by rewriting the row as final and answers `"provisional": false`. Both that declaration and settlement's final refund are compare-and-swap writes on the provisional row, so exactly one of them wins: a final refund written first is what recovery returns, and one written after a declaration changes nothing, so an answer cannot change once given. The client never spends a refund the gateway calls provisional: it keeps the spend `spending` and asks again ([Refund recovery](#refund-recovery) says when), settling on it only when the gateway calls it final or when the spent key's `spend_until` passes (when waiting longer would leave it worth nothing).

**Ending a request early.** A stream that ends before the engine finishes (its client went away, it reached the gateway's deadline, or the engine failed partway) is settled on **what was generated**: the prompt and the output the engine had produced by the last chunk the gateway received from it, priced and clamped by the shared contract, and the rest of the hold refunded. The engine reports its running count on every delta chunk, and on a chunk of its own when a step produced output it has not yet released as text, such as a tool call still being written (the gateway forwards only the final count to its client), and the gateway keeps the last one before it offers the chunk to the client, so output generated but not yet delivered is counted. Before any count has arrived, the prompt alone is charged (at the contract's cap), since the engine has begun on it. The stream then ends with its metadata event, carrying that refund, and **no `[DONE]`**, so the client reads it as the interrupted answer it is (writing no answer, and keeping any it would have replaced) and settles from the refund in band. The gateway watches for the client going away and for the deadline even while the engine sends nothing, and drops its upstream at once, which closes the connection to the node; the node cancels the request in its scheduler before its next step. Everything after a spend is recorded runs in a task of its own, for a stream as for a blocking request, so a client that goes away at any point does not take the settlement with it. A stream whose client left before it opened still runs to the engine's answer: a refusal refunds in full, and a stream that opens is ended at once and charged its prompt alone. A blocking request runs to its end and is billed by the usage the engine reports, since a blocking engine request reports nothing until it completes. Every paid request ends by the gateway's `MAX_REQUEST_DURATION` (30 minutes), a bound on the gateway's own resources rather than part of the payment contract; a blocking request cut there is charged its whole hold, since nothing trustworthy says what it generated.

**Why.** Without provisional refunds, a crash between recording and settlement left a row with no refund, recovery answered "not yet" forever, and the whole credential was lost, not just the request's hold. Two refunds of one spend cannot fork it: every refund of a proof signs the commitment inside that proof, which fixes the successor's nullifier, so the provisional and the final successors share one nullifier and spending either records it and refuses the other. That is why the two must be told apart (settling on the provisional successor would destroy the final one, losing the request's partial return), and why the compare-and-swap is what keeps an answer from changing: no clock or timer is needed for that. The live set only decides when a provisional refund is worth waiting for: while the request that will replace it is running, and not after. It lives in the enclave's memory, never in the database, which would otherwise learn when each request ran. Three gateway rules keep the "no fork" argument whole, and each holds: a refund is only ever issued for a proof whose nullifier is recorded; recovery of a recorded nullifier returns the stored refund and never signs a fresh full one; and a nullifier row outlives every successor under its key (pruning). Charging what was generated on an early ending is the only rule that is fair both ways: refunding everything would let a client generate freely and leave (a denial of service with no one to bill), and refunding nothing charged a client its whole hold for output that was never produced. The count is the engine's (the pinned node's), as for every completed request, and the gateway keeps the last one it received, so it is the gateway's own record of what it was sent. The stage lives inside the sealed plaintext, which only the client's key opens, so the database sees only the second write it already saw.

**Enforced.** `credentials::RefundStage`, `seal_refund_token` / `open_refund_token` (the stage is the plaintext's first byte; both stages are one length), `LiveSpends` / `LiveSpend` (counted, so a double spend refused at recording does not end the first request's entry; held on `ActSpend::live` by the handler and the stream task), `record_spend`, `store_sealed_refund` and `recover_presented_refund` (both through the compare-and-swap `db::replace_refund_token`, settlement against `ActSpend::provisional`, the token it recorded); `handlers::forward_stream` (the client, the deadline and the engine watched together; the upstream dropped on every early ending; the running count kept and stripped from what the client is sent), `early_cost`, `MAX_REQUEST_DURATION`, `DEADLINE_DELIVERY_GRACE` (every ending's terminal write abandoned past it); the relay's `tx.closed()` watch (`engine_router::relay_stream`); the node's running `usage` (`eidola-server-engine` `stream_response`). Tests: `a_refunds_stage_travels_inside_the_seal`, `a_spend_is_live_until_its_last_holder_lets_go`, `the_provisional_and_final_successors_of_one_spend_cannot_both_be_spent`, `an_early_ending_is_charged_for_what_was_generated`, `a_client_that_leaves_releases_the_engine_and_keeps_the_count`, `the_deadline_releases_the_engine_and_keeps_the_count`, `the_running_count_never_reaches_the_client`, `a_dropped_receiver_closes_the_engine_connection_without_waiting_for_output`, the node's `streaming_shapes_text_and_usage` and `client_disconnect_cancels_in_the_engine`, and against Postgres `recovery_stores_only_sealed_tokens_and_still_returns_the_refund` (provisional while served here, final once settled; a spend nothing serves declared final at once, and a final refund written after that loses the swap). Each was checked to fail with its rule removed (the live-set lookup, the disconnect watch, the relay's watch, the running count's stripping, the swap). On the client: `recover_refund`, `accepts_provisional_refund`; tests `a_provisional_refund_is_accepted_only_at_the_spend_deadline` and, in the chat harness, `a_provisional_refund_is_settled_on_only_when_the_gateway_calls_it_final` and `a_stream_cut_short_settles_on_what_was_generated`.

**What would break it.** Storing the provisional refund in a separate write after the nullifier (a crash between them is the old loss); a plain-text column for the stage; a live set in the database, or one a spend leaves before its final refund's write is attempted; an ending that does not overwrite the provisional refund; a final-refund write or a declaration that is not a compare-and-swap on the provisional row; refunding everything, or nothing, on an early ending; counting only what reached the client; a stream path that keeps the engine working after its client has gone; a client that settles on a refund the gateway still calls provisional before `spend_until`.

**What it does not cover.** A gateway crash still costs the request's hold, recovered as soon as the process is back (the restart time). A client that recovers a spend still running on *another* live gateway instance gets that instance's provisional refund declared final and loses the request's partial return (never more than the hold); production runs one instance, so this is at most a deploy overlap ([gaps.md](gaps.md#a-crash-mid-spend-costs-the-requests-hold)). Output the engine produced after its last chunk reached the gateway (the steps in flight before the cancel reaches it) is not charged. If the final refund's own write fails or loses the swap (it is best-effort, and logged), the provisional one stands and is declared final at the next recovery, losing the request's partial return unless the response reached the client.

## Sealed refund tokens

**Decision.** The stored copy of each refund is encrypted under a key derived from a **refund recovery key** the client sends with the spend (the `refund_key` beside its Token in the body's `credentials` element), which the client derives from the spend's secret pre-refund state and the gateway never stores or logs. It is bound to its own nullifier row and has the same length whatever the amount and whatever its stage (provisional or final). Recovery opens it only when the client presents the same key; any other key gets the same `404` as no refund stored. A spend without a well-formed key is refused before its nullifier is recorded.

**Why.** The refund message carries `t` in the clear, and `t` is the hold minus what the request cost: a function of how long the prompt and the answer were. Stored in plain beside the nullifier, every spend's refund amount would sit in a database outside the enclave. `CREDENTIAL_MASTER_KEY` is deliberately not part of the seal: the operator knows it, so it adds nothing against the operator or the database. It encrypts issuer private keys at rest.

**What would break it.** Storing a refund, or `t`, in plain; logging the recovery key, or a request body that carries it; a seal whose ciphertext length varies with the amount.

## Nullifier pruning

**Decision.** A key's nullifiers are deleted only once `accept_until + RETENTION_PAST_ACCEPT` has passed, the same instant the key stops being usable for anything.

**Why.** A nullifier pruned while any instance could still accept a spend under its key allows that token to be spent again.

**Enforced.** The SQL `prune_expired_nullifiers(INTERVAL)`, which refuses a non-positive margin and is called only with `RETENTION_PAST_ACCEPT`; `no_nullifier_is_pruned_while_its_key_could_accept_a_spend`, `the_schema_prunes_on_the_shared_deadline`, `pruning_and_lookup_honour_the_retention_margin`.

**What would break it.** Pruning on another schedule, by hand with a smaller interval, or keyed on anything other than the key's own deadline.

## What Postgres may hold

Postgres runs outside the enclave. It is treated as adversarial: its operator or hosting provider could be compelled, under a gag order, to hand over or watch everything in it.

**Decision.** For the account surface it holds what billing needs: accounts (a UUID and a secret hash), the credit ledger and its purchase allocations, actual purchase prices and currencies, durable refund and closure operations, Stripe observations and references, the database identity responsible for an operation, terms acceptances, one ledger row per issuance (account, amount, key, time), and beside it the stored response of that issuance (account, idempotency key, request digest, key, amount, ledger entry, and the signed response, with no timestamp of its own), kept until its key can be used for nothing. For the inference surface it holds, per spend, only the issuer key id, the nullifier, and the sealed refund; the inputs of one request are rows written in one transaction. **Nullifier rows carry no timestamp and no amount, and nothing written on the inference path names an account.** Issuer private keys are stored encrypted.

**Why.** The database is the one place both surfaces meet. A per-spend timestamp or amount beside per-account issuance timestamps and amounts would hand whoever reads it material to pair them.

**What the rows do not hide.** Postgres records when each row was written: a live observer sees each write as it happens, its write-ahead log, replicas and backups carry commit times, and its transaction ids (`xmin`) order nullifier rows against ledger rows. So the database can see when each spend began and finished, and therefore how long each request took; and, since the inputs of one request commit together, which nullifiers belong to one request, and so its input count and key set ([Spending several tokens in one request](#spending-several-tokens-in-one-request)). Never an amount. That is disclosed in [gaps.md § Spend timing at the database](gaps.md#spend-timing-at-the-database), and it is also why issuance timing (above) matters. If statement or parameter logging is enabled on the database, those logs also carry the values written, which is one more reason only ciphertexts and nullifiers may be written on the inference path.

**What would break it.** A timestamp, amount, account id, request id or content-derived value in any row the inference path writes; a new table written per spend; a join key between the two surfaces.

## Logs and telemetry

**Decision.** Nothing the gateway logs or exports names an amount, a nullifier, a key-derived secret or an account on the inference path. Issuance logs "credential issued" with no amount and no ledger id. Errors are logged through `ServerError`'s log-safe `Display`, which omits the message of `PaymentRequired` (it states the charge and the worst case). Telemetry is aggregate histograms and counters keyed by fixed labels; spans are exported only for a request that asks to be traced.

**Why.** Telemetry leaves the enclave for a vendor. A per-request amount or outcome with a timestamp is exactly the record the database is denied.

**Enforced.** `error.rs` tests on the redacted `Display`; `#[instrument(skip_all)]` throughout; the observability rules in the gateway's development guide and [gateway.md](gateway.md#telemetry-scope-and-boundary).

**What would break it.** Interpolating a charge, a refund amount, a nullifier or a ledger id into a log line; a new metric with a caller-derived label; tracing a credential handler's arguments.

**Residual.** Some refund paths log a fixed line per occurrence ("stream ended early (client_disconnect)", for instance). Each is a timestamped record that one request ended that way. It carries no amount, nullifier or account.

## Error responses

**Decision.** Spend errors depend only on the presented tokens and the request: an unknown or expired key, a bad challenge digest, a bad context or proof (`401`); a nullifier already spent, in which case none of the request's inputs is recorded (`409`); a malformed `credentials` member, more than two inputs, one token presented twice, a share of zero, or a credential in a header (`400`/`401`). A refusal after the nullifiers are recorded (an unknown model, a request over the ceiling, a hold below the worst case (`402`, stating both, which the client already knows) or above it (`400`), an engine that fails before answering) carries a full refund of every input; a stream that ends partway (its engine failed, its client left, or it reached the gateway's deadline) carries the refunds of what was not generated, the charge consumed from the inputs in order (above); a blocking request ended at the deadline (`504`) carries refunds of nothing, returning each token's untouched balance `c − s`. None depends on the hidden balance, which the gateway never learns.

**What would break it.** An error that reveals anything the presented spend does not already contain, or that differs by account or by gateway state in a way a client could probe.

## Payment reversals

**Decision.** `eidola-billing` owns the financial workflow shared by independent billing workers and the private support CLI. The CLI connects directly to the database and Stripe as its human or workload caller; it is not a privileged gateway endpoint. A separately runnable webhook receiver durably queues provider observations. The gateway authenticates an account-holder repair request and queues it in the database; it never executes that repair or runs an inbox worker. The account holder can read the completed generation and its report, with a newer request remaining pending even when it arrives during a repair. These linked-account operations can run outside the attested confidentiality boundary; ACT issuance and signing remain inside it.

Reconciliation fetches current provider facts under a per-account lock and applies purchases, refunds and disputes together, so a known chargeback cannot arrive one commit after spendable purchase credits. Every ledger change also takes the same account row lock as issuance. Durable work survives process loss, and database locks coordinate receiver/worker replicas and direct CLI execution. Supported producer and executor protocol versions are checked separately; an older producer can request work from a newer compatible executor, while unsupported work is retained rather than silently completed.

A purchase retains the credits granted, cash actually paid, currency, original expiry and provider payment identity. An ordinary refund first reserves the exact eligible credits in a durable operation, then requests the cash refund, then records its outcome. A separately chosen cash-only remedy changes no credits. Refund observations without such an intent record money movement and require review; they cannot invent a credit cancellation. A lost response leaves the operation pending, with its reservation intact. The repair command and webhook worker reconcile that same operation; neither can create a second entitlement from a duplicate or reordered event.

A chargeback deducts its cash value using the prices actually paid for affected credits, including other purchases. Any unrecovered value remains a deficit; it does not expire when the original credits do. Previous refunds and reversals prevent the same payment being recovered twice. A won dispute undoes the recorded deductions with their original expiries. Remaining positive credits stay usable while the account is open. Closure freezes new conversions and checkout before cancellation and refunds, and preserves the refund eligibility of credits that expire while closure is being processed.

**Privacy boundary.** None of these operations can identify or recall ACTs already issued. The account holder's repair endpoint asks for reconciliation of their own account only; it accepts no token, spend history, refund key or caller-chosen credit adjustment. A cash refund for a service failure or other individual remedy does not exchange ACTs back into account balance.

**What would break it.** A second balance writer that bypasses the shared transaction rules; guessing a purchase's value at par; letting a refund reservation be converted; treating a webhook delivery as an instruction to debit again; or any mechanism that identifies or revokes a particular customer's anonymous tokens.

## The local wallet

**Decision.** The client's database holds its tokens, the pre-issuance state of each issuance (with the request as sent, its idempotency key and the account it was for, so a lost response can be asked for again), the pre-refund state of each spend in flight, and the link between each spend and the request it paid for (the spend trail). It is the user's own record. One spend at a time is taken per wallet (`spend_gate`), every input of a request marked spending in one transaction, so two concurrent requests never pick the same token. Chat turns, utility chores and the local inference proxy all spend through the same path and settle through the same refund handling. Every issuance (ahead of need, just in time, explicit, recovered) goes through one gate, so two never convert for the same need.

**Why.** The device is the one place where the connection between account and requests is meant to exist. Its protection is the device's ([privacy-guarantees.md §8.1](privacy-guarantees.md#8-bounded-claims-what-this-document-does-not-promise)).

**What would break it.** Sending any of this to the gateway, or to telemetry.

## Review checklist

A change that touches credentials, issuance, spending, refunds, recovery, pricing or the billing schema should be able to answer yes to each of these.

- Is every spend still paid in full, up front, by an amount fixed by the request alone?
- Does every path after a nullifier is recorded still issue a refund, under the spent token's own key, and overwrite the provisional one, within the request deadline?
- Is the hold still exactly the worst case, on both sides, as the sum of a request's shares?
- Are a request's inputs still recorded all or none, taken by the client all or none under the spend gate, charged in input order, and refunded each under its own key?
- Does a credential still travel only in the request body, and never into a log, a Record row or an engine?
- Is every key use still checked against its deadline, and no nullifier pruned while its key could accept a spend?
- Is every balance check still serialized with its debit?
- Does nothing new persist, log or export a charge, a refund amount, a timestamp per spend, or a value derived from a request's content, outside the enclave?
- Does nothing new give two clients different keys, contexts or separators?
- Is every issuer key still on the public calendar, one per epoch, with its windows bound into its encryption, and does the client still refuse a key that is off the calendar or is not the one it recorded for the epoch?
- Does the client still check what the issuer chose (context, amount, key) before storing a token?
- Is no new value made to outlive the key it was issued under?
- Is every issuance still either ahead of need at a time no request decides, or the just-in-time fallback, and is its amount still the standard size rule?
- Does every debit for an issuance still happen in the transaction that stores its response, after the stored lookup, under the account lock?
- Does no idempotency key, request digest or stored response ever reach the inference surface or a log?

## Where to read the code

- Gateway: `crates/eidola-server-gateway/src/credentials.rs` (keys, issuance, recovery, sealed and provisional refunds, recording a request's inputs), `handlers.rs` (spend, settlement across inputs, refunds, the request deadline), `auth.rs` (the `credentials` member read as Tokens), `db.rs`, and `schema/schema.sql` (`issuer_key`, `credit_ledger`, `credential_issuance`, `nullifier`, `debit_account`, `record_credential_issuance`, `prune_expired_nullifiers`, `prune_expired_issuances`).
- Shared: `crates/eidola-common` (`prompt_charge`, the pricing contract; `issuer_calendar`, the public calendar key windows are checked against; `spend_credentials`, the `credentials` member, `ACT_L` and `partial_returns`; `refund_key`).
- Client: `crates/eidola-app-core/src/lib.rs` (`account_allocate`, `allocate_with_key`, `settle_issuance`, `recover_pending_issuances_locked`, `provision_once`, `start_credential_provisioner`, `check_issued_credential`, `ensure_spend_inputs`, `choose_inputs`, `acquire_spend`, `SpendPrep::settle`, `auto_allocation_amount`, `process_refund`, `recover_refund`, `body_with_spend`, `recover_stranded_spends`, `recover_spending_credentials`, `recover_pending_issuances`), `db.rs` and `schema/schema.sql` (the `credential_lifecycle` view), `proxy/` and `utility.rs` (the other spenders).
