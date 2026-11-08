# Anonymous credits

Eidola is paid for with **anonymous credit tokens**: the Privacy Pass ACT scheme ([draft-schlesinger-privacypass-act](https://datatracker.ietf.org/doc/draft-schlesinger-privacypass-act/), over the cryptographic construction in [draft-schlesinger-cfrg-act](https://datatracker.ietf.org/doc/draft-schlesinger-cfrg-act/), implemented by the `anonymous-credit-tokens` crate). An account buys credits; the client converts some of them into a token the server signs without seeing; and each inference request is paid for by spending part of that token, in a way the server cannot connect to the account or to the client's other requests.

The scheme does that only for the part of the system it describes. Everything around it decides whether it still holds: when keys rotate, how much is put in a token and when, what the database is allowed to remember, what a log line may say, how a refund gets back to a client whose connection dropped. A careless choice in any of them can undo the anonymity the cryptography provides, or turn the payment system into a way to take inference without paying for it, and neither failure shows up in a test that only checks the cryptography.

This page collects those choices in one place. It is written for people who are technical but are not cryptographers: users deciding what to trust, auditors, and contributors. The first part explains what the protocol itself gives. The second explains, one decision at a time, what the rest of the system has to do so that the protocol's promise survives contact with a real deployment. Each decision names where it is enforced and what would break it.

> [!IMPORTANT]
> **A change to any code this page names must be checked against it.** That covers the gateway's credential, issuance, spend, refund and recovery code, its billing schema, and the client's wallet, provisioning and charging code. If a change alters a decision recorded here, this page changes in the same pull request. The privacy promises themselves are in [privacy-guarantees.md](privacy-guarantees.md); this page is how the system keeps the ones about payment.

## What the protocol provides

### The parties and the token

There are two parties. The **issuer** is the Eidola server. It holds a private key and signs tokens. The **client** is the app on your device. It holds tokens and spends them.

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

An account-based service that is abused can see who is abusing it. It can rate-limit that account, ban it, refund a complaint after looking at the history, or give out a free tier and watch for people farming it. Eidola's inference surface can do none of that. The server cannot tell which account sent a request, or whether two requests came from the same client. That is the point of the design, and it means there is **no identity to rate-limit, ban or investigate** on the inference path.

What is left is the price. Every request must be paid for in full, up front, by a token that was paid for in full, before any work is done, and the amount it may cost must be fixed by the request itself. If any path lets a request cost less than it should, or lets value be spent twice, or lets a token outlive what was paid for it, there is no second line of defence behind it.

### And there is no quick fix

The server's code is measured into its enclave and pinned by every client release; a release is reproducible, reviewed and signed by a human attestant ([releases.md](releases.md)). That is what lets users check what the server does. It also means a server change is a release, not a hotfix: the shortcuts an account-based SaaS takes when billing misbehaves (refund on complaint, a generous free tier, a server-side heuristic, "we'll patch it later") are either unavailable or would weaken a promise users are relying on. The decisions below have to be right when they ship.

### Two risks per decision

Each decision below protects one or both of:

- **Confidentiality and unlinkability**: whether the server, the database, the telemetry vendor or a network observer can connect a request to an account, or two requests to each other.
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

**Decision.** An honest gateway has exactly one key per epoch and never rotates within one, and **the client holds it to that for itself**. Every listed key whose windows are on the calendar is recorded per epoch, first seen wins (`issuer_key_epoch`, scoped to the server's base URL), and that includes the next epoch's key, listed a whole epoch before it issues. Before it requests anything, the client refuses the current key if its windows are not exactly its epoch's (`IssuerKeyRefusal::NotOnCalendar`) or if it is not the key already recorded for its epoch (`IssuerKeyRefusal::SecondKeyForEpoch`). Nothing is charged; credentials already held keep working; the GUI says so in the reader's language and the CLI prints it. The refusal lasts until the next epoch's key is current.

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

**Enforced.** `account_allocate`, `listed_key_epoch` and `listed_key_is_its_hash` in app-core, `db::pin_issuer_key_epoch` and the `issuer_key_epoch` table, `AppError::IssuerKeyRefused`; the GUI's `space-error-issuer-key-*` messages. Tests: `a_key_off_the_calendar_is_refused_before_issuance`, `a_second_key_for_a_seen_epoch_is_refused_until_the_next_epoch` (a real second key that would have signed, refused before any request; held credentials still spend; the next epoch's key accepted), `the_key_seen_for_an_epoch_survives_a_restart`, `the_first_key_seen_for_an_epoch_stands`, `a_refused_issuer_key_says_so_in_the_readers_language`.

**What would break it.** Recording a key only once it is used (the next epoch's key would no longer be pinned in advance); replacing a recorded key; scoping the record to anything that changes with a gateway release (a measurement), which would reset it exactly when a new build arrives; letting any second key for an epoch through.

## The domain separator and the request context

**Decision.** The domain separator is a fixed constant, `ACT-v1:eidola:inference:production:2026-03-05`, never rotated on a schedule. The request context is `SHA-256("eidola" ‖ "inference" ‖ issuer_key_id)`, with an empty credential context, so every token under one key carries the same one. The token challenge carries an empty redemption context, since nullifiers already make each spend single-use.

**Why.** The anonymity set is everyone sharing a key **and** a domain separator **and** a request context. Rotating the separator would split it without adding anything key rotation does not already give. The request context is revealed at every spend of a token and of all its successors, so a context that differed between clients would make every one of their spends recognisable: it would be an account id in all but name.

**Enforced.** `credentials::compute_request_context` and `domain_separator` on the server (`test_domain_separator_is_stable`); each spend's context is checked against the key's (`credentials::verify_presented_spend`). The issuer chooses the context and the amount at issuance and the issuance proof does not constrain either to what the client asked for, so **the client checks both** before it stores a new token: `check_issued_credential` in app-core refuses a token whose context is not the one every client under the key shares, or whose amount differs from the one requested (`an_issuance_that_could_mark_its_credential_is_refused`).

**What would break it.**

- Any per-account, per-session or per-request value in the request context or the credential context.
- A change to the separator outside a planned protocol version change (every outstanding token becomes unspendable).
- Removing the client's check, which leaves the context to whatever the server sends.

## Key consistency

**Decision.** A key's id is the SHA-256 of its public key, as the Privacy Pass draft specifies. The client checks that the key it was told is current has the id its public key implies, that the issuance response names that same key, and that every refund names the key of the token it replaces; it stores tokens only under that verified id.

**Why.** The key id is what each spend names, so it decides which tokens look alike. A client that stored a token under whatever id a response named could be steered into spending under a label the server can sort by, or into a token whose spends name a key the server cannot find (stranded value). A refund labelled with a newer key would also make the client believe a token lives longer than it does.

**Enforced.** `account_allocate` and `process_refund` in app-core, with `issuer_key_id_of`; tests `an_issuance_naming_another_key_is_refused` and `a_refund_naming_another_key_is_not_stored`. On the server the `issuer_key` table's `id_is_key_hash` constraint.

**What would break it.** Storing a token under a server-supplied id without recomputing it; accepting a refund under a key other than the spent token's.

**What the client cannot check.** Whether other clients were given the same key in an epoch it saw first. Within one client's history, it can: see [Key rotation and partitioning](#key-rotation-and-partitioning).

## Issuance sizing

**Decision.** When a request needs a token and none in the wallet covers it, the client converts `min(available, max(DEFAULT_ALLOCATION_CREDITS, charge))` from the account: 1,000,000 credits ($1.00) in the usual case, the remaining balance when less is left, and the charge itself when one request needs more than $1.00. `eidola account allocate` converts any amount the user asks for.

**Why.** The amount converted is visible at issuance, and the charge `s` is visible at each spend. A fixed amount makes issuances look alike, and an allocation large enough for many requests means issuances are rare. A small one keeps the value parked in any one anonymous token, and so the value at risk if it lapses (below), small.

**Enforced.** `auto_allocation_amount` and `DEFAULT_ALLOCATION_CREDITS` in app-core.

**What would break it.**

- Sizing issuance to the request about to be made. If the converted amount equals the next spend's charge, the two are linked by amount alone. This happens today only when a single request's worst-case charge exceeds $1.00, which no catalogued model's maximum completion reaches, but a large model or a long context could.
- Converting oddly specific amounts often (the remaining-balance case does this once per account as its balance runs out).

**Open question.** Whether to keep one large allocation, move to a fixed denomination, use a small ladder of denominations, or allow a request to be paid from several tokens at once is not settled. A spend that can combine several tokens is planned as a protocol extension: several spend proofs in one request, the charge consumed in the order they are presented, and each input's change returned to its own successor under its own key. It is not designed here.

## Draining a token

**Decision.** A spend may take a token's whole remaining balance, and with several-token spends some spends will. The charge `s` of such a spend is the token's exact remaining balance.

**Why it is acceptable today.** A token's balance is hidden by the protocol. Someone who saw that one spend's charge equalled a token's remaining balance could, with the earlier charges of the same token and the amount it was issued with, try to reconstruct its history (which earlier charges sum to the issued amount minus this one). That needs the earlier charges, and the charges and refunds of a chain are not linked to each other: each spend reveals a fresh nullifier, and **no spend amount, refund amount, or anything derived from them is persisted or observable outside the enclave**. The refund is stored sealed under a key only the client holds (below), and neither the charge nor the refund is logged.

**Considered and not adopted.** The client could round the drained share down to a small public step (preferred over random jitter, because the loss is bounded and predictable), at the cost of stranding the remainder. It is not needed while the invariant above holds.

**What would break it.** Persisting, logging or exporting a charge, a refund amount, or anything derived from them, anywhere outside the enclave. That would reopen this question.

## Issuance timing

**Decision.** Automatic provisioning converts credits at the moment a request needs them, immediately before that request is spent.

**Why.** It keeps the client simple, and the account never has more converted than it uses.

**The cost.** Anyone who can see when issuances and spends happen (the database, which records the account and time of each issuance and sees each nullifier as it is written, or an observer of the server's traffic) can pair an issuance with the spend that follows it milliseconds later. That links the first spend of each token to the account, and a prompt-cache key on that request links the rest of its conversation lineage ([privacy-guarantees.md §2.7](privacy-guarantees.md#2-unlinkability)). The acceptance window allows tokens to be issued well before they are spent (§2.4); automatic provisioning does not use it. This is a disclosed residual: [gaps.md](gaps.md#issuance-to-first-spend-timing).

**What would break it further.** Issuing smaller tokens more often, or issuing per request.

## Issuance concurrency

**Decision.** Debiting an account takes a lock on the account's row for the rest of the transaction, so two issuances against the same account run one after the other.

**Why.** An issuance checks the balance and then debits it. Without the lock, two concurrent issuances each read the same balance, each pass the check, and both convert it: a client sending issuance requests in parallel could convert its balance once per request. The tokens cannot be traced or recalled once issued, so an overdraft here is pure loss.

**Enforced.** `debit_account` in the gateway's `schema/schema.sql`; `concurrent_issuances_cannot_spend_one_balance_twice` (`credentials.rs`, against Postgres).

**What would break it.** Any balance check and debit that are not serialized per account, including a second debit path that bypasses `debit_account`.

## Lost issuance responses

**Decision.** The server signs, then debits, then answers. Issuance is not idempotent: a client whose response was lost cannot ask for it again.

**The cost.** If the response is lost after the debit (a dropped connection, a client crash), the converted amount is gone: the account was charged and the client never obtained the token. It is at most one allocation per occurrence. Disclosed in [gaps.md](gaps.md#a-lost-issuance-response-strands-its-allocation).

## Spending and the nullifier

**Decision.** A spend's proof is verified first, under the key it names, which must still accept spends. Then its nullifier is inserted into Postgres. Only an insert that wins is served; a duplicate is refused with `409`. From that point every path out of the handler issues a refund.

**Why.** Recording before serving is what prevents a fork. The `(issuer_key_id, value)` primary key is enforced by the database every gateway instance shares, so a token spent concurrently against two instances is served once. Verifying before recording means an invalid proof never writes a row.

**Enforced.** `handlers::chat_completions_phases`, `handlers::verify_spend_proof`, `credentials::verify_presented_spend`, `db::record_nullifier`.

**What would break it.**

- Serving before the insert, or keeping an in-memory set of spent nullifiers in place of the shared table. With more than one instance, the same token could then be served once per instance.
- A path that records a nullifier and returns without a refund. The client loses the token's whole remaining value, not just the charge.

## Holds, the completion ceiling and the pricing contract

**Decision.** The client presents a charge `s` that covers the request's worst case: the prompt priced by the shared contract and the completion priced at its ceiling. The server recomputes the same worst case from the same request and refuses (`402`) a charge below it, before anything reaches an engine. When the request completes, the server charges what was used, clamped so it can never exceed the hold, and refunds the rest as `t`.

**Why.** The charge must be fixed by the request, before any work, with nothing left for later. The client and server compute it with the same function of the same bytes (`eidola_common::prompt_charge`, which counts content and tool bytes at a safe cost factor), so the client's hold always covers the server's charge and never has to be topped up. The completion side is the model's declared `max_output_tokens`: a request asking for more is refused (with a full refund) rather than quietly shortened, and a request naming no limit is given the declared one, so what the engine is asked for, what was priced and what can be charged are one number. `PRICING_MARKUP` may not go below the contract's cost factor (1.5), or prompt bytes would be sold below cost.

**Enforced.** `handlers::worst_case_cost`, `check_sufficient_charge`, `check_completion_ceiling`, `actual_cost` and `settled_cost`; `backend::validate_pricing_markup`; the client's `completion_ceiling` and the local proxy's clamp. Tests `client_and_server_prompt_terms_agree`, `the_completion_side_is_held_to_the_declared_ceiling`, `token_dense_usage_is_clamped_to_the_contract`.

**What would break it.**

- Any input to the price the client cannot compute exactly in advance.
- A charge that depends on how the upstream counts tokens without the contract's clamp.
- Charging after the fact, or for a completion that ran past what was priced.
- A request field that makes the engine do more work than was priced (more output, more choices, a longer context).

## Price independent of the prefix cache

**Decision.** A request costs the same whether or not an engine served part of its prompt from its prefix cache. The settled charge reads only the reported prompt and completion counts, never `cached_tokens`.

**Why.** The refund `t` is the hold minus the cost. With a cache discount, the cost would depend on whether this prompt's prefix had been served before, so the refund would say whether an earlier request shared it: a link between two requests, written into a value the server computes and the client receives. A discount for a declared, client-chosen cached prefix that the client can price in advance is planned; it is not offered today.

**Enforced.** `the_settled_charge_ignores_prefix_cache_hits` (`handlers.rs`).

**What would break it.** Reading `prompt_tokens_details.cached_tokens`, or any engine-reported value that depends on other requests, into the charge.

## Continuations

**Decision.** A request continuing a partial answer (`continue_final_message`, an assistant message with `reasoning_content`) is priced by the same contract: the partial answer and its reasoning are prompt bytes like any other.

**Why.** Otherwise continuing would be a way to buy completion at prompt prices, or to have the engine produce more than was priced.

**Enforced.** The strict request type in `types.rs` and `eidola_common::engine_protocol::check_continuation`; the pricing walk includes `reasoning_content`.

**What would break it.** A continuation field the pricing walk does not count.

## Refunds stay under the spent token's key

**Decision.** A refund is always signed under the key of the token it replaces, never under the current issuing key. This holds on every path: a blocking answer, a stream, an error after the nullifier, and recovery.

**Why.** A successor under the current key would let a token near the end of its life be renewed for a further epoch by spending one credit. Repeated, value bought once would live forever, past the expiry of the credits it was bought with, and the server would have to keep accepting it.

**Enforced.** `credentials::sign_refund`, which takes the key from the spend and from nowhere else; `a_refund_is_signed_under_the_spent_credentials_key`. On the client, `process_refund` verifies the refund against the spent token's public key and stores the successor under that key's id.

**What would break it.** A refund path that signs with "the current key", or a client that accepts a successor under another key.

## Expiry, on both sides

**Decision.** A key's tokens are accepted until `accept_until`, on every path, whether or not the key is cached. The server keeps the key for one more hour (`RETENTION_PAST_ACCEPT`) only to finish spends already accepted: signing their refunds and answering recovery. It never accepts a new spend in that hour. The client stops *starting* spends ten minutes before `accept_until` (`CREDENTIAL_SPEND_MARGIN`) and stops recovering at `accept_until`, so its whole window ends inside the server's.

**Why.** Too late: a key that is still accepted after its nullifiers are pruned allows a double spend; a key accepted past its deadline extends value. Too early: a client that treats a token as expired while the server would still take it strands paid-for value. The margins cover clock disagreement between the client and the server, between instances, and between instances and the database (each gateway refuses to rotate keys while its clock and the database's differ by more than ten seconds, `db::MAX_CLOCK_SKEW`).

**Enforced.** `credentials::key_usable`, `usable_key` and `load_key`; the compile-time assertion that `RETENTION_PAST_ACCEPT` exceeds twice `MAX_CLOCK_SKEW`; `a_cached_key_past_accept_until_refuses_a_spend`; the client's `issuer_key_deadlines` and the `credential_lifecycle` view, which the provisioning queue, recovery and every wallet listing read. Issuance refuses a key whose spend deadline has already passed, before any debit.

**What would break it.** Reading a key straight from the cache without its deadline; a client lifetime computed from `issue_until`; any path that accepts a fresh nullifier after `accept_until`.

## Value and the window it was bought for

**Decision.** A token lives until its key's `accept_until` and no longer, and its change stays under the same key. Account credit carries its own expiry (a subscription's credits expire at the end of the billing period, a purchase's after a year). Credit converted just before it expires becomes a token that is good until its key stops accepting: between one and two epochs (7 to 14 days by default) after the conversion. That bounded grace is the most value can outlive its window; nothing extends it further.

Value still in a token when its key lapses is lost. There is no rollover, because exchanging an old token for a new one without linking the two needs its own design. Small automatic allocations keep the amount at risk to one allocation's change per key; see [gaps.md § Credit left in a credential when its key lapses](gaps.md#credit-left-in-a-credential-when-its-key-lapses).

**What would break it.** Any rollover, renewal or refund path that carries value into a later key without its own unlinkability analysis; any way to convert credit whose expiry has already passed.

## Refund recovery

**Decision.** A client whose response was lost asks `POST /v1/credentials/refund` with the same spend. The endpoint verifies the spend proof first, under a key that is still retained. Then:

- if the nullifier was recorded and a refund was stored, it returns that refund;
- if the nullifier was recorded and no refund was stored, it answers `404`, because the request may still be in flight;
- if the nullifier was never recorded and the key still accepts spends, the request never arrived: it records the nullifier and refunds the whole charge.

The client never asks for recovery of a spend it is still waiting on (`LiveSpends`), because recovery would record the nullifier and the real request would then be refused as a replay.

**Why.** Recovery is what keeps a dropped connection from costing a token's whole balance. Verifying first means only the holder of a valid spend proof (the client that made it) learns anything, and an invalid proof writes nothing: without that, anyone could write nullifier rows under a public key id, or ask what became of a spend whose nullifier they knew.

**Enforced.** `credentials::recover_refund`, `recovery_refuses_an_unverified_proof_before_the_database`; the client's `recover_spending_credentials`, `try_refund_recovery`, and the startup sweep.

**What would break it.** Reading or writing a nullifier row before verifying; refunding in the never-arrived branch after `accept_until`; a recovery that answers differently by anything other than the spend's own state.

**What it does not cover.** A gateway that crashes after recording a nullifier and before storing its refund leaves a row with no refund: recovery answers `404` forever and the token's remaining value is lost. See [gaps.md](gaps.md#a-crash-mid-spend-strands-the-tokens-remaining-value).

## Sealed refund tokens

**Decision.** The stored copy of each refund is encrypted under a key derived from a **refund recovery key** the client sends with the spend (`x-eidola-refund-key`), which the client derives from the spend's secret pre-refund state and the server never stores or logs. It is bound to its own nullifier row and has the same length whatever the amount. Recovery opens it only when the client presents the same key; any other key gets the same `404` as no refund stored. A spend without a well-formed key is refused before its nullifier is recorded.

**Why.** The refund message carries `t` in the clear, and `t` is the hold minus what the request cost: a function of how long the prompt and the answer were. Stored in plain beside the nullifier, every spend's refund amount would sit in a database outside the enclave. `CREDENTIAL_MASTER_KEY` is deliberately not part of the seal: the operator knows it, so it adds nothing against the operator or the database. It encrypts issuer private keys at rest.

**What would break it.** Storing a refund, or `t`, in plain; logging the recovery key; a seal whose ciphertext length varies with the amount.

## Nullifier pruning

**Decision.** A key's nullifiers are deleted only once `accept_until + RETENTION_PAST_ACCEPT` has passed, the same instant the key stops being usable for anything.

**Why.** A nullifier pruned while any instance could still accept a spend under its key allows that token to be spent again.

**Enforced.** The SQL `prune_expired_nullifiers(INTERVAL)`, which refuses a non-positive margin and is called only with `RETENTION_PAST_ACCEPT`; `no_nullifier_is_pruned_while_its_key_could_accept_a_spend`, `the_schema_prunes_on_the_shared_deadline`, `pruning_and_lookup_honour_the_retention_margin`.

**What would break it.** Pruning on another schedule, by hand with a smaller interval, or keyed on anything other than the key's own deadline.

## What Postgres may hold

Postgres runs outside the enclave. It is treated as adversarial: its operator or hosting provider could be compelled, under a gag order, to hand over or watch everything in it.

**Decision.** For the account surface it holds what billing needs: accounts (a UUID and a secret hash), the credit ledger, Stripe references, terms acceptances, and one ledger row per issuance (account, amount, key, time). For the inference surface it holds, per spend, only the issuer key id, the nullifier, and the sealed refund. **Nullifier rows carry no timestamp and no amount, and nothing written on the inference path names an account.** Issuer private keys are stored encrypted.

**Why.** The database is the one place both surfaces meet. A per-spend timestamp or amount beside per-account issuance timestamps and amounts would hand whoever reads it material to pair them.

**What the rows do not hide.** Postgres records when each row was written: a live observer sees each write as it happens, its write-ahead log, replicas and backups carry commit times, and its transaction ids (`xmin`) order nullifier rows against ledger rows. So the database can see when each spend began and finished, and therefore how long each request took. That is disclosed in [gaps.md § Spend timing at the database](gaps.md#spend-timing-at-the-database), and it is also why issuance timing (above) matters. If statement or parameter logging is enabled on the database, those logs also carry the values written, which is one more reason only ciphertexts and nullifiers may be written on the inference path.

**What would break it.** A timestamp, amount, account id, request id or content-derived value in any row the inference path writes; a new table written per spend; a join key between the two surfaces.

## Logs and telemetry

**Decision.** Nothing the gateway logs or exports names an amount, a nullifier, a key-derived secret or an account on the inference path. Issuance logs "credential issued" with no amount and no ledger id. Errors are logged through `ServerError`'s log-safe `Display`, which omits the message of `PaymentRequired` (it states the charge and the worst case). Telemetry is aggregate histograms and counters keyed by fixed labels; spans are exported only for a request that asks to be traced.

**Why.** Telemetry leaves the enclave for a vendor. A per-request amount or outcome with a timestamp is exactly the record the database is denied.

**Enforced.** `error.rs` tests on the redacted `Display`; `#[instrument(skip_all)]` throughout; the observability rules in the gateway's development guide and [server.md](server.md#telemetry-scope-and-boundary).

**What would break it.** Interpolating a charge, a refund amount, a nullifier or a ledger id into a log line; a new metric with a caller-derived label; tracing a credential handler's arguments.

**Residual.** Some refund paths log a fixed line per occurrence ("issuing zero refund" when a client disconnects mid-stream, for instance). Each is a timestamped record that one request ended that way. It carries no amount, nullifier or account.

## Error responses

**Decision.** Spend errors depend only on the presented token and the request: an unknown or expired key, a bad challenge digest, a bad context or proof (`401`); a nullifier already spent (`409`); a charge below the worst case (`402`, stating both, which the client already knows). A refusal after the nullifier is recorded (an unknown model, a request over the ceiling, an engine that fails before answering) carries a full refund; a stream that fails partway carries a refund of nothing, returning the token's untouched balance `c − s`. None depends on the hidden balance, which the server never learns.

**What would break it.** An error that reveals anything the presented spend does not already contain, or that differs by account or by gateway state in a way a client could probe.

## Payment reversals

**Decision.** A Stripe refund or chargeback debits the account's ledger, netted against the credit pool it reverses. It cannot reach credits already converted: those are anonymous tokens that cannot be identified or recalled.

**Why.** That is the anonymity working as intended. It means buying credits, converting them, and then disputing the payment is a loss the business absorbs, bounded by how much was converted. Refunds to customers are possible only out of balance that has not been converted.

**What would break it.** Any mechanism that could identify or revoke a particular customer's tokens. That would be a tracing capability, and the guarantees forbid it.

## The local wallet

**Decision.** The client's database holds its tokens, the pre-refund state of each spend in flight, and the link between each spend and the request it paid for (the spend trail). It is the user's own record. One spend at a time runs per wallet (`spend_gate`), so two concurrent requests never pick the same token. Automatic provisioning, chat turns, utility chores and the local inference proxy all spend through the same path and settle through the same refund handling.

**Why.** The device is the one place where the connection between account and requests is meant to exist. Its protection is the device's ([privacy-guarantees.md §8.1](privacy-guarantees.md#8-bounded-claims-what-this-document-does-not-promise)).

**What would break it.** Sending any of this to the server, or to telemetry.

## Review checklist

A change that touches credentials, issuance, spending, refunds, recovery, pricing or the billing schema should be able to answer yes to each of these.

- Is every spend still paid in full, up front, by an amount fixed by the request alone?
- Does every path after a nullifier is recorded still issue a refund, under the spent token's own key?
- Is every key use still checked against its deadline, and no nullifier pruned while its key could accept a spend?
- Is every balance check still serialized with its debit?
- Does nothing new persist, log or export a charge, a refund amount, a timestamp per spend, or a value derived from a request's content, outside the enclave?
- Does nothing new give two clients different keys, contexts or separators?
- Is every issuer key still on the public calendar, one per epoch, with its windows bound into its encryption, and does the client still refuse a key that is off the calendar or is not the one it recorded for the epoch?
- Does the client still check what the issuer chose (context, amount, key) before storing a token?
- Is no new value made to outlive the key it was issued under?

## Where to read the code

- Gateway: `crates/eidola-server-gateway/src/credentials.rs` (keys, issuance, recovery), `handlers.rs` (spend, settlement, refunds), `auth.rs` (the token's wire form), `db.rs`, and `schema/schema.sql` (`issuer_key`, `credit_ledger`, `nullifier`, `debit_account`, `prune_expired_nullifiers`).
- Shared: `crates/eidola-common` (`prompt_charge`, the pricing contract; `issuer_calendar`, the public calendar key windows are checked against).
- Client: `crates/eidola-app-core/src/lib.rs` (`account_allocate`, `check_issued_credential`, `ensure_spendable_credential`, `auto_allocation_amount`, `process_refund`, `recover_spending_credentials`), `db.rs` and `schema/schema.sql` (the `credential_lifecycle` view), `proxy/` and `utility.rs` (the other spenders).
