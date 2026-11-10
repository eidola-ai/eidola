# Spending several anonymous credits in one request

**Status: design specification, implemented.** The body transport (§5.1, §5.4) with `credentials` and `refunds` always arrays, `L = 32` (§5.5), the hold equality `H = W` (§6.1 step 6, §7.4), provisional refunds, decided by what the gateway is serving, and early endings charged for what was generated (§6.7), recovery that verifies before it records (§6.5), multiple inputs with `N_max = 2` (atomic recording, in-order allocation, one refund per input under its own key, §6.1 to §6.4), the client's input rule and atomic taking of inputs (§7.1 to §7.6, with the §7.7 check of the partial returns), and two-credential provisioning (§9.1) are implemented. Not implemented: verifying each proof once rather than twice (§8.7, which needs a crate change), rounding drains to a public step (§7.5, deferred by design), and batching nullifier inserts (§9.6, noted only). Where this document and [anonymous-credits.md](../anonymous-credits.md) differ about what is implemented, that page describes the current behaviour. It specifies an extension to how Eidola presents Privacy Pass Anonymous Credit Tokens (ACT), written so that its normative parts can be adapted into a section of an Internet-Draft. It builds on mechanisms Eidola already has: sealed refund tokens with a client-held refund-recovery key, the client's spend deadline of `accept_until` minus a margin, and provisioning credentials of one public $1 denomination ahead of need.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in BCP 14 (RFC 2119, RFC 8174) when, and only when, they appear in all capitals.

## Abstract

Today an Eidola inference request carries exactly one ACT spend proof. This document lets a request carry an ordered list of up to `N_max` spend proofs, each an unmodified ACT spend under its own issuer key. The server verifies every proof, records every nullifier in one atomic step, and treats the request's hold as the sum of the presented charges. The actual charge is consumed from the inputs in the order the client presented them, and each input receives its own refund, under its own issuer key, into its own successor credential. No value ever moves from one input to another, so no value moves between issuer keys or epochs. The extension is pure composition of independent ACT spends: it needs no change to the ACT cryptography. It does create one new disclosure, which the analysis below quantifies: proofs presented together are known to come from one client at that instant.

## 1. Motivation

A request's hold `s` is a public number in the spend proof: the client proves its credential holds at least `s` credits, the server charges the actual cost `C ≤ s`, and issues a partial return `t = s − C` into the successor credential, which then holds `c − s + t = c − C` credits. Two problems follow from requiring one credential per request.

1. **Stranded residuals.** A credential whose remaining balance is smaller than the next request's hold cannot pay for it. Its balance is stranded: it can still serve smaller holds if any come along, but otherwise it lapses at its key's `accept_until`. With the proposed public issuance denomination `D` and every hold bounded by `D/k`, each credential strands up to `D/k` (6.25% at `k = 16`).
2. **Large holds need large credentials.** A hold of `s` needs one credential holding at least `s`. That pushes towards large or hold-sized issuances, and an issuance whose amount equals the next request's hold links the account that was issued to the request that spends (the issuance amount is recorded on the account-linked surface, the hold is public in the proof). It also limits concurrency: a client can have only as many requests in flight as it has credentials that each cover a whole hold.

Letting one request draw on several credentials solves both: a residual can be presented alongside another credential and drained, and a large hold can be covered by several smaller credentials.

## 2. Terminology

This document uses the terms of the ACT specification (`draft-schlesinger-cfrg-act`) and its Privacy Pass binding (`draft-schlesinger-privacypass-act`): *credential* (a client-held ACT credit token, holding a hidden balance `c`), *spend proof*, *Token* (the Privacy Pass structure carrying one spend proof), *nullifier* (`k`, revealed by a spend, unique per credential instance), *request context* (`ctx`), *issuer key* and `issuer_key_id`, *refund* (the issuer's response to a spend), *partial return* (`t`, the credits the issuer returns, `0 ≤ t ≤ s`), and *successor credential* (the credential the client constructs from a refund). In addition:

- **Input**: one Token presented in a request. A request has inputs `1..N`, in the order the client presented them.
- **Share** (`sᵢ`): the spend amount revealed by input `i`'s proof. ACT calls it the charge `s`; this document calls it a share to distinguish it from the request's charge.
- **Hold** (`H`): the sum of the shares, `H = Σ sᵢ`.
- **Worst-case cost** (`W`): the pre-flight minimum hold the server computes from the request under the pricing contract the client and server share (`eidola_common`): the prompt charge plus the completion ceiling at the completion rate, or the flat per-request price.
- **Charge** (`C`): the amount the server finally charges the request, `0 ≤ C ≤ H`.
- **Consumption** (`aᵢ`): the part of `C` allocated to input `i`. Its partial return is `tᵢ = sᵢ − aᵢ`.
- **Drain**: an input whose share equals the credential's entire remaining balance (`sᵢ = cᵢ`).
- **Fill**: the input whose share is derived from the hold rather than from a balance (`s_fill = W − Σ drains`).
- **Accepting key**: an issuer key `K` with `now < accept_until(K)`. Keys rotate weekly and accept for one further epoch, so at most two keys accept at once (three for an instant at a boundary).

Integer amounts are credits. All arithmetic on amounts is exact unsigned integer arithmetic; an implementation MUST use checked operations and treat overflow as a malformed request.

## 3. Overview

```text
Client                                                    Server (in enclave)
  | choose inputs 1..N and shares s1..sN (§7)                  |
  | ProveSpend(cred_i, s_i) for each i, mark all spending      |
  |-- POST /v1/chat/completions -----------------------------> |
  |   body: "credentials": [{token, refund_key}, ...]          | parse all, static checks (§6.1)
  |                                                            | verify every proof
  |                                                            | record all nullifiers atomically
  |                                                            | H = Σ s_i; check H = W
  |                                                            | run inference, settle C ≤ H
  |                                                            | allocate C in order (§6.3)
  |                                                            | refund_i = VerifyAndRefund(sk_i, proof_i, t_i)
  |<-- response, "refunds": [refund_1, ..., refund_N] -------- |
  | successor_i = ConstructRefundToken(pk_i, proof_i,          |
  |               refund_i, state_i)                           |
```

A request with one input is the same spend as today's, with the same proof, the same processing and the same refund. Only the transport changes, for every request alike (§11): the Token and its refund-recovery key move from request headers into the request body, as a one-element `credentials` array, and the response's refund moves into a one-element `refunds` array.

## 4. Relationship to the ACT cryptography

The extension changes nothing in ACT. Every input is produced by an unmodified `ProveSpend`, verified by an unmodified `VerifySpendProof`, and answered by an unmodified `IssueRefund` with a partial return `tᵢ` the issuer chooses within the bound ACT already enforces (`0 ≤ tᵢ ≤ sᵢ`). What the extension adds is protocol-layer composition: how inputs are carried, the all-or-nothing acceptance rule, the deterministic allocation that chooses each `tᵢ`, and recovery. §8.3 argues why that composition is sound and §8.4 why no per-request cryptographic binding is required for it.

## 5. Wire format

### 5.1 Request: the `credentials` member

Every request that spends carries its inputs in the request body, never in a header. The body of every endpoint that takes a spend is a JSON object (for inference, the OpenAI-compatible request), and the inputs are a top-level `credentials` member, an array with one element per input, in input order:

```json
"credentials": [
  { "token": "<base64url(Token)>", "refund_key": "<64 lowercase hex digits>" },
  { "token": "<base64url(Token)>", "refund_key": "<64 lowercase hex digits>" }
]
```

- **`token`** is one Token exactly as `draft-schlesinger-privacypass-act` defines it (Token Creation): `token_type`, `challenge_digest`, `issuer_key_id` and the encoded spend proof. Its bytes are unchanged; only where they travel differs.
- **`refund_key`** is that input's refund-recovery key: with sealed refund tokens, 32 bytes derived by the client from that input's own `PreRefund` state, as exactly 64 lowercase hexadecimal digits. Each input has its own `PreRefund`, and therefore its own key, so the key travels beside the Token it belongs to rather than in a parallel list.

The order of the array is the order of the inputs: element 1 is input 1. Rules:

- **Always an array.** `credentials` is an array whatever the input count, `1 ≤ N ≤ N_max`, so every count has one encoding and a single input is not a special case. A server MUST refuse a request with more than `N_max` elements before decoding any spend proof.
- **Body only.** A server MUST refuse a request that carries a credential anywhere else (an `Authorization: PrivateToken` header, or the legacy refund-key header), so there is one transport and nothing to reconcile between two.
- **Strict members.** Each element has exactly the members `token` and `refund_key`. A server MUST refuse an unknown member, a missing one, a duplicated member name (in an element or at the top level), and a `credentials` value that is not an array.
- **Encoding.** `token` is base64url (RFC 4648, Section 5) with padding, and a server MUST refuse a value that does not re-encode to itself (non-zero pad bits, missing padding, or whitespace). `refund_key` MUST be lowercase hex; uppercase digits, whitespace or a wrong length are refused. Each Token MUST decode with no trailing bytes.
- **Uniform type.** Every Token MUST carry the same `token_type` (`0xE5AD`) and the same `challenge_digest`. Tokens MAY name different `issuer_key_id` values (§7.6 says when a client should).
- **Not forwarded.** The server removes `credentials` before the request leaves it for the inference backend, as it removes every other field the backend does not need.

**Why the body.** One Token at Eidola's former `L = 128` was about 24 KB as base64url, which already exceeded common header limits of 8 to 16 KB: nginx (`large_client_header_buffers`, 8 KB per line), Apache (`LimitRequestFieldSize`, 8,190 bytes), Node.js (16 KB in total), and hyper's HTTP/2 server on Eidola's own path, which advertises a 16 KB `SETTINGS_MAX_HEADER_LIST_SIZE`. It worked only because TLS terminates inside the attested enclave, so every hop that parses the request is one Eidola configures. A list would have been larger still. Request bodies have no such limits along the path, the request is read whole before it is authorized in any case, and the ACT issuance protocol already carries its messages as request content. This departs from RFC 9577, whose `PrivateToken` scheme puts a Token in the `Authorization` header; §13.2 raises that with the working group.

### 5.2 (Removed)

The separate refund-recovery header is gone: each input's key travels in its `credentials` element (§5.1).

### 5.3 Response: refunds

Every response that follows the recording of the inputs' nullifiers (§6.6, classes 3 to 5) carries one refund per input, in input order, in a `refunds` array that replaces today's single `refund` object (an array even for one input, mirroring `credentials`), in the JSON body of a blocking response or error and in the final metadata event of a streamed response:

```json
"refunds": [
  { "refund": "<base64url(Refund)>", "issuer_key_id": "<hex>" },
  { "refund": "<base64url(Refund)>", "issuer_key_id": "<hex>" }
]
```

`refunds[i]` answers input `i`. The array MUST have exactly `N` elements. `issuer_key_id` repeats input `i`'s key; the client MUST check that it matches. A response to a request refused before its nullifiers were recorded carries no `refunds`.

Refunds travel in the body rather than in a response header, because a streamed response's settlement is known only when the stream ends, after its headers were sent. A generic Privacy Pass binding of this extension that carries refunds in the `PrivacyPass-Reverse` field (as the current ACT editor's draft does, following `draft-meunier-privacypass-reverse-flow-http`) would carry a list in input order, as an HTTP trailer when the response streams.

### 5.4 Refund recovery is per input

`POST /v1/credentials/refund` takes exactly one input, in the same body form as §5.1: `{"credentials": [{"token": …, "refund_key": …}]}`, an array of exactly one element. A recovery request with any other count MUST be refused. Its answer carries `refunds` as an array of one, with the `provisional` marker of §6.7. Each input of a multi-input request is recovered by its own nullifier and its own key, independently of the others (§6.5).

### 5.5 Sizes and limits

Eidola's ACT parameters use `L = 32`: a balance lies in `[0, 2^32)` credits, and since one credit is a millionth of a US dollar, one credential can hold up to about $4,294, over four thousand times the $1 issuance denomination. The ACT specification leaves `L` to the deployment (it must be at most 252) and does not signal it in keys, Tokens or challenges, so client and server fix it at build time. Measured with `anonymous-credit-tokens` 0.4.2 (release build, one thread, an Apple-silicon laptop; enclave hardware will differ):

| `L` | Spend proof (CBOR) | Token as base64url | Prove | Verify and refund |
|---|---|---|---|---|
| 32 | 4,919 B | about 6.6 KB | 1.4 ms | 1.8 ms |
| 64 | 9,303 B | about 12.5 KB | 2.5 ms | 3.1 ms |
| 128 | 18,071 B | about 24.2 KB | 4.9 ms | 6.1 ms |

Size and cost are linear in `L` (the ACT draft gives `32 × (14 + 4L)` bytes of raw proof). `L` bounds only the largest representable balance; it does not change what a proof hides or what the issuer learns. One refund is 176 bytes of CBOR whatever `L` is.

**`N_max` for Eidola is 2.** That is one drain and one fill (§7.2), under one key or, to drain an expiring residual, under two (§7.6). The protocol is written for a general `N_max`, and §9.1 is why Eidola's is the smallest useful one: every input beyond the first narrows the request's anonymity set, and at a small population that cost dominates. A deployment with a larger population could raise it.

## 6. Server processing

### 6.1 Order of operations

A server MUST process a request in this order, and MUST NOT record any nullifier before step 5.

1. **Parse.** Decode the `credentials` member per §5.1. Refuse on any error, on `N > N_max`, or on a credential carried outside the body.
2. **Static checks, every input.** `token_type` is `0xE5AD`. `challenge_digest` equals the expected digest. `issuer_key_id` names a key the server holds that is usable for a spend now (`now < accept_until`). The proof's `ctx` equals that key's request context. The share decodes to an integer (`ScalarToCredit`) and is at least 1 (zero shares are refused; see §9.4). The hold `H = Σ sᵢ` is computed with checked arithmetic.
3. **Duplicates.** The inputs' nullifiers MUST be pairwise distinct, compared per issuer key. A repeated nullifier refuses the whole request. Two inputs with the same nullifier are two spends of one credential instance; ACT permits a client to produce them (with different shares), and they are exactly the forking attempt nullifiers exist to stop.
4. **Verify every proof.** `VerifySpendProof(skᵢ, proofᵢ)` under each input's own key, with the key's spend deadline re-checked under the same lock or snapshot the verification uses. All proofs must verify. A server SHOULD run the checks of steps 1 to 3 for every input before verifying any proof, since they are cheap and the proofs are not (§8.7).
5. **Record atomically** (§6.2). If any nullifier is already recorded, record none and refuse the request.
6. **Validate the request** against the pricing contract: the model exists, the completion ceiling is within the model's declared ceiling, and `H = W` exactly. A failure here is after recording, so it MUST be answered with full refunds (§6.6, class 3).
7. **Serve.** Dispatch the request; on a dispatch failure, answer with full refunds (class 4).
8. **Settle.** Compute the charge `C` by the deployment's settlement rule with `H` in place of the single-spend charge, clamped so that `C ≤ H`. Allocate it (§6.3), issue one refund per input (§6.4), store each (§6.5), and respond (§5.3).

Steps 1 to 5 are the whole of today's pre-flight path, applied to every input. A one-input request runs exactly today's algorithm.

### 6.2 Atomic recording

All `N` nullifiers MUST be recorded in one serializable step: either every input's nullifier is recorded, or none is. In Eidola's PostgreSQL store this is one transaction inserting one `nullifier` row per input (`issuer_key_id`, `value`), and committing only if every insert succeeded (a single multi-row `INSERT … ON CONFLICT DO NOTHING` whose affected-row count is compared with `N`, or `N` inserts with a rollback on the first conflict).

The server SHOULD insert in a canonical order, sorted by (`issuer_key_id`, `value`), so that two concurrent requests sharing nullifiers cannot deadlock on the primary key. The outcome must not depend on that order: a conflict on any input rolls back every insert.

Atomicity is what makes partial failure safe. If a request is refused because one input's nullifier was already recorded (a replay, or a recovery that raced the request), then no other input of it was recorded, and each remains spendable by its holder: recoverable through `POST /v1/credentials/refund`, which records it and returns a full refund (§6.5).

### 6.3 Charge allocation

The charge is consumed from the inputs in the order the client presented them. Given shares `s₁..s_N` and a charge `C` with `0 ≤ C ≤ H`:

```text
remaining = C
for i in 1..N:
    a_i = min(s_i, remaining)
    remaining = remaining - a_i
    t_i = s_i - a_i
```

Properties, each of which an implementation MUST preserve:

- **Exact.** `Σ aᵢ = C` and `Σ tᵢ = H − C`, because `C ≤ H` leaves `remaining = 0` at the end.
- **Within ACT's bound.** `0 ≤ tᵢ ≤ sᵢ` for every input.
- **Per-input conservation.** Input `i`'s successor holds `cᵢ − sᵢ + tᵢ = cᵢ − aᵢ ≤ cᵢ`, under input `i`'s own key. No input's successor holds more than its predecessor did, and no value crosses from one input to another, so none crosses between keys.
- **Prefix structure.** For some index `j`, inputs before `j` are fully consumed (`tᵢ = 0`), input `j` is partly consumed, and inputs after `j` are fully returned (`tᵢ = sᵢ`). Early inputs are used up first and the change lands on later ones.
- **Deterministic and public.** The allocation is a function of the shares, their order and `C` only, so a client can predict and check it, and the server has no freedom in choosing `tᵢ` that could be used to mark a successor (§9.2).

`C = 0` (every failure after recording) returns every share in full: `tᵢ = sᵢ`.

Example, `W = 6,200`, a drain of 3,000 presented first and a fill of 3,200 second:

| Charge `C` | `a₁`, `a₂` | `t₁`, `t₂` | Successors |
|---|---|---|---|
| 0 (refused after recording) | 0, 0 | 3,000, 3,200 | both unchanged in value |
| 500 | 500, 0 | 2,500, 3,200 | drained input keeps 2,500; fill unchanged |
| 4,000 | 3,000, 1,000 | 0, 2,200 | drained input exhausted; fill pays 1,000 |
| 6,200 | 3,000, 3,200 | 0, 0 | both consumed in full |

### 6.4 Refund issuance, including zero partial returns

The server MUST issue a refund for every recorded input, `refundᵢ = IssueRefund(skᵢ, K′ᵢ, ctxᵢ, sᵢ, tᵢ)` under input `i`'s own key, including when `tᵢ = 0`.

A partial return of zero does not mean a worthless successor. The successor holds `cᵢ − sᵢ + tᵢ`, and the server never learns `cᵢ`: an input with `tᵢ = 0` may be a drain whose successor is empty or a partial spend whose successor still holds most of a credential. Declining to issue would destroy a balance the server cannot see, so the refund is not optional. Issuing every refund also keeps the response uniform: the number and size of refunds depend only on `N`, never on which inputs were exhausted. A client that knows a successor is empty (a drain with `tᵢ = 0`) discards it.

A refund is signed under the input's own key while that key is retained (`now < accept_until + retention`), exactly as a single-spend refund is today, so a request accepted just before one input's `accept_until` still settles every input.

If signing one input's refund fails, the server retries that input alone with `tᵢ = 0`, as it does for a single spend today (that still returns the blinded remainder `cᵢ − sᵢ`), and signs the others normally. It MUST NOT move that input's partial return to another input: that would cross keys.

### 6.5 Refund storage and recovery

Each input's refund is stored on its own nullifier row, sealed under its own refund-recovery key (sealed refund tokens), exactly as a single spend's is. Storing is best-effort, as today: a failure to store is logged and the refund is still returned in the response.

Recovery is per input and needs nothing from the other inputs of the same request. `POST /v1/credentials/refund` with input `i`'s Token and key:

- returns input `i`'s stored refund, if its nullifier is recorded and its refund stored;
- answers "recorded, refund not yet available" (`404`), if its nullifier is recorded and its request is still in flight;
- records the nullifier and returns a full refund (`tᵢ = sᵢ`), if its nullifier is not recorded. That is the case for every input of a request that never arrived, or that was refused before step 5 of §6.1.

The last case is safe for multi-input requests only because recording is atomic. If a client recovers input 1 of a request whose arrival it cannot confirm, and the request then arrives, the request's recording conflicts on input 1 and records nothing, so its other inputs are still unrecorded and are recovered the same way. Without atomicity, the late request could consume input 2 for a request the client already wrote off.

A server MUST verify the proof before recording a nullifier on the recovery path, as it does on the spend path. Eidola's recovery handler verifies before it reads or writes anything about the nullifier, and records a never-arrived input with its full refund stored beside it in one statement.

### 6.6 Failure handling

| Class | Where it fails | Nullifiers recorded | Refunds in the response | What the client does |
|---|---|---|---|---|
| 1 | Parsing, counting, static checks, duplicates, any proof's verification, any key unknown or past `accept_until` | none | none | Recover every input through `POST /v1/credentials/refund` (each gets a full refund into a fresh successor). Do not present the same proofs again (§7.7). |
| 2 | Recording: some input's nullifier already recorded | none (rolled back) | none (`409`) | Recover every input. The one already recorded returns its stored refund if it has one; the others are recorded now and fully refunded. |
| 3 | Request validation after recording (unknown model, ceiling, `H ≠ W`) | all | all, `tᵢ = sᵢ` | Construct every successor. |
| 4 | Dispatch or upstream failure | all | all, `tᵢ = sᵢ` | Construct every successor. |
| 5 | Settled (including a stream that ended early) | all | all, by §6.3 | Construct every successor; discard empty ones. |
| 6 | Response lost after any of 3 to 5 | all | stored, sealed per input | Recover each input by its own nullifier and key. |
| 7 | Process crash after recording, before refunds are stored | all | none stored | See §6.7. |

Every class leaves every input's value either unconsumed and recoverable (1, 2), or consumed only by `aᵢ` with the rest returned in a refund the client can obtain (3 to 6). Class 7 is the exception, and it exists for single spends today. One boundary applies to classes 1 and 2 as it does to single spends: recording a fresh nullifier, on the spend path or the recovery path, is a spend, allowed only while the input's key accepts. An unrecorded input whose key has passed `accept_until` has lapsed with its key; the client's spend margin (§7.2) exists so that no input it sends is that close to its deadline.

### 6.7 Crash safety

Today, a crash between recording a nullifier and storing its refund leaves the nullifier recorded and no refund to recover. The client's recovery is answered "not yet available" forever, and the whole credential (the blinded remainder `c − s` as well as the hold `s`) is lost. A multi-input request multiplies that exposure by `N`.

This specification closes it for single and multi-input requests alike with a **provisional refund**. In the same transaction that records the nullifiers, the server stores, for each input, a sealed refund signed with `tᵢ = 0` and marked provisional. At settlement it overwrites each with the final refund, marked final. A crash then loses at most each input's share, never its remainder.

**Why two refunds of one spend are safe for the issuer.** The successor's nullifier `k*` is fixed by the commitment `K′` inside the spend proof, and both refunds sign that same commitment; only `t` differs. A client holding both successors (the provisional `cᵢ − sᵢ` and the final `cᵢ − sᵢ + tᵢ`) can spend only one of them, because spending either reveals `k*`. This was checked against `anonymous-credit-tokens` 0.4.2: two refunds of one proof, with `t = 0` and `t = 100`, completed to two credentials with equal nullifiers and balances differing by 100.

**Why the two must be distinguishable.** Because the two successors share a nullifier, spending the provisional one destroys the final one. A recovery can return the provisional refund without any crash: the connection drops, or the client restarts and loses its record of which requests are in flight, while the server is still serving the request. If the client treated the provisional refund as a settlement and spent its successor, the later final refund would be unusable and the input's partial return `tᵢ` would be lost. So:

- The marker travels inside the sealed refund (so the database learns nothing from it beyond the overwrite it already sees), and recovery returns it: `POST /v1/credentials/refund` answers a provisional refund with `"provisional": true`, and a final one with `"provisional": false`.
- The server MUST overwrite every provisional refund of a request with a final one when the request ends for any reason (classes 3 to 5 of §6.6), before or as it responds.
- Every write that replaces a provisional refund is a compare-and-swap on the provisional row: the final refund at settlement, and recovery's declaration that a provisional refund is final. Exactly one wins, so an answer recovery gave never changes, and a final refund written late changes nothing. This alone keeps a client from losing value it was told it had; no timer is needed for soundness.
- The server answers a provisional refund with `"provisional": true` only while a request that will replace it is still running. Eidola's gateway keeps, in enclave memory, the set of spends each process is serving (entered before the nullifiers are recorded, left after the final refund's write is attempted); recovery of a provisional refund whose spend no process-local request is serving declares it final at once. A crash's provisional refunds therefore become final as soon as the gateway is back. A client that recovers a spend still running on a different gateway instance can lose that input's partial return, never more than its hold; Eidola runs one instance, so this is at most a deploy overlap.
- A stream that ends early (its client went away, the server's own request deadline, or an engine failure) is settled on what was generated up to the cut: the prompt and the output the engine had produced by the last count the server received from it, priced by the deployment's settlement rule and clamped to the hold. The server stops the engine when it cuts a stream. Refunding everything would let a client generate freely and leave; refunding nothing would charge it for output never produced.

On a provisional answer, the client MUST keep the input pending. It MAY construct and store the provisional successor, but MUST NOT spend it, and retries recovery later. It settles the input when recovery returns a final refund, which includes a provisional refund the server has declared final because no request is serving it any more. Pending state is durable (§7.1), so a restart preserves it. The deadline that bounds every input applies here too: the input's key retains refunds until `accept_until + retention`, and a successor is only worth anything if it can be spent before `accept_until`. A client that reaches its spend margin while an input is still provisional accepts the provisional successor at that point.

The cost is one extra signature per input on the pre-flight path, and recovery during flight returns a provisional refund instead of `404`.

## 7. Client behaviour

### 7.1 Taking several credentials at once

The client MUST take every input of a request in one atomic local step: under the same lock that serializes credential acquisition today, it produces every spend proof and durably marks every input `spending`, together with its `PreRefund` state, in one local transaction, before sending anything. The ACT specification requires the client's private state to be stored durably before a spend is sent; with several inputs, they must also be stored together, so a crash cannot leave some inputs marked spending and others not. The inputs of one request are grouped locally so that settlement processes them together, while recovery remains per input.

### 7.2 Choosing inputs

Inputs and shares are chosen so that the amounts reveal nothing that a single spend would not, except the balances of drained inputs (§9.2). The rule has no thresholds or tunable parameters. For a request with worst-case cost `W`:

1. **Order the wallet.** Take the credentials available for a spend now (not in flight, and before their spend deadline), sorted by key `accept_until` ascending, then balance ascending, then nullifier bytes. All credentials under one key share an `accept_until`, so this is: oldest key first, smallest first within a key.
2. **One input.** If the first credential covers `W`, spend it alone with share `W`.
3. **Two inputs.** Otherwise take the first credential in the order, with balance `a < W`, for which some later credential covers `W − a`. Drain it: present it first with share `a`, then the first such later credential, with share `W − a`. This is normally the first credential in the order; a later one is drained only when the first has no fill but another pair in the wallet covers `W`, so that a wallet able to pay is never sent to allocation.
4. **A new credential.** If no pair covers `W` (or the wallet is empty), drain the first credential and allocate a new credential by the client's ordinary allocation rule as its fill (or use the new credential alone, for an empty wallet).

The client MUST NOT present an input whose spend deadline (`accept_until` minus the client's margin) has passed. The request's own deadline is the earliest of its inputs'.

Consequences of the rule, all accepted:

- **Draining is eager.** A credential is drained as soon as it is the first in order and cannot cover the request at hand. Holds vary by request, so a credential that could have paid a cheaper request alone may be drained by a dearer one. No value is lost by that: the drain is consumed only by `a₁ ≤ C`, and its change comes back under its own key.
- **Runs of two-input requests.** A drained credential that is not fully consumed returns as a smaller successor (§6.3), which is again first in order and below `W`, so it is drained again on the next request, until a charge exhausts it. Because charges are usually far below `W`, one credential's tail can take several consecutive two-input requests. Each is placed in `A₂(K)`, which two-credential provisioning makes every provisioning account (§9.1), and in Eidola's deployment the drained balances are visible only to the measured code (§9.2).
- **Less concurrency.** During such a run each request holds two credentials, so a client with two credentials can have only one request in flight; a concurrent request finds nothing free and allocates a new credential (step 4). That costs no value, only more credits held ahead of need, which lapse with their key if unspent. The client's provisioning keeps enough credentials on hand for its usual concurrency.
- **Oldest value first.** Ordering by `accept_until` spends an older key's credentials before a newer key's, so value is consumed in the order it would lapse, and a residual under the older key is drained into a request whose fill is under the newer key when the older key has no fill (§7.6).

### 7.3 Canonical order

A client SHOULD order inputs as a deterministic function of public information only:

1. the drain before the fill;
2. with more than one drain (only where `N_max > 2`), by key `accept_until` ascending (soonest-expiring value is consumed first), then by share ascending, then by nullifier bytes ascending.

The order of §7.2 produces this. Every element of that order is already visible in the request, so the order adds nothing (the nullifier tie-break is uniformly random). An order that depends on anything else (a balance, a local id, a history) would leak it. Whether the server should enforce the canonical order rather than honour any order is an open question (§13.2).

### 7.4 Shares never come from balances, except drains

A share is either `W` (a single input), a drain's balance (§7.5), or a fill derived from `W` and the drains. In the last case, `W` is computed by the server from the request itself and the drain shares are already in the proofs, so the fill share discloses nothing new. A client MUST NOT size any other share from a balance. A partial share sized from a credential's balance would reveal the balance without draining it, and a sequence of such shares would let an issuer that remembers amounts reconstruct a credential's history.

The shares sum to exactly `W`, the same hold a single-input spend of the same request would show. A client MUST NOT over-hold, and the server refuses any `H ≠ W` (§6.1 step 6): an over-hold would be a degree of freedom a client could fingerprint itself with, and with `H = W` the charge for a request without reported usage, the whole hold, is never more than the worst case.

### 7.5 When to drain

A credential is drained exactly when §7.2 puts it first and it cannot cover the request (step 3). It is drained by presenting its whole balance as its share.

- **Exact drains.** Eidola's client drains exact balances, for the reason given in §9.2. A client MAY instead round a drain's share down to a multiple of a public step `Δ`, stranding less than `Δ` and revealing the balance only to precision `Δ`. A deployment in which the issuer is an adversary for amounts SHOULD do so, or SHOULD strand issuer-known successors (§9.2). How coarse `Δ` must be to matter depends on the population, so Eidola defers it.
- **No deferral.** A client may drain in any request, including the first after an issuance. §9.3 explains why a deferral rule was considered and not adopted.

### 7.6 Mixing issuer keys

Presenting inputs under two keys reveals that the client was issued credentials under both (§9.1). A client SHOULD present inputs under one key. It MAY mix keys only to drain a residual under the older key before that key stops accepting, when no fill under the older key exists. The fill under the newer key then pays whatever the residual does not; value is consumed from the older key first and never moves to the newer one. If the charge does not reach past the residual, the residual's change comes back under the older key, as a successor the client drains again or lets lapse, and the fill is returned whole under the newer key. Each input's refund is therefore always under its own key, even though, in the common case, only the last input's successor holds anything.

### 7.7 Settlement, verification and recovery

On a response that carries `refunds`, the client MUST, for every input `i`:

- check that `refunds[i].issuer_key_id` is input `i`'s key and that the refund verifies (`ConstructRefundToken(pkᵢ, proofᵢ, refundᵢ, stateᵢ)`). A refund that does not verify is treated as missing, and that input is recovered;
- store the successor (or discard it if its balance is zero), and settle the input.

The client SHOULD also check that the partial returns have the prefix structure of §6.3 and, where the response reports usage, that `Σ (sᵢ − tᵢ)` equals the charge the shared pricing contract computes from it. A mismatch is a server fault, not something to correct locally, and SHOULD be surfaced.

A client MUST NOT present a spend proof in a second request after it was sent once, even if the first request was refused before recording (class 1 or 2). It recovers the input instead. Re-presenting would show the same nullifier in two requests and link them, and the recovery path yields a fresh, unlinkable successor at no cost.

## 8. Security analysis

### 8.1 Adversaries and assumptions

The analysis distinguishes two adversaries.

- **Model A, the issuer as adversary.** The ACT and Privacy Pass threat model: the issuer–origin sees and may retain everything presented to it (nullifiers, shares, request contexts, keys, order, timing) and everything it computes (charges, partial returns), alongside its account-linked records of issuance (which account was issued how much, under which key, when). This is the model an Internet-Draft adaptation must assume.
- **Model B, Eidola's deployment.** The issuer runs as measured code inside a confidential-compute enclave, and that code retains no per-request amounts: shares, charges and partial returns live only in enclave memory for the request's duration, refunds are stored only sealed under a key only the client holds, and telemetry is aggregate. The adversaries are outside the enclave: the operator and its database (which sees nullifier rows, their keys, their commit timing, the sealed refunds, and the account-linked issuance ledger), and the network (which sees request and response sizes and timing). A model-A issuer here would have to run different code, which is visible as a release.

Assumptions:

1. ACT's unforgeability and unlinkability (q-SDH in ristretto255, BLAKE3 or the suite's hash modelled as a random oracle), as stated in the ACT specification.
2. The nullifier store is linearizable and supports atomic multi-row transactions.
3. TLS from the client terminates where the verifier runs (in Eidola, inside the attested enclave), so no third party sees spend proofs in transit.
4. The client stores its private spend state durably before sending, and its local store is not adversarial (Eidola's privacy guarantees exclude local adversaries).
5. Clocks agree within the bounded skew the gateway already assumes.

### 8.2 What the cryptography guarantees, and what the protocol must

Per proof, ACT guarantees:

- **Soundness:** the prover holds a credential issued under the named key with request context `ctx`, a balance of at least `s`, and the revealed nullifier `k`.
- **Unlinkability:** information-theoretically, the proof reveals nothing about the credential beyond `(k, s, ctx)` and the key it verifies under.
- **Refund correctness:** a refund completes to a successor holding `c − s + t`, with a fresh nullifier the issuer has not seen, and is unlinkable to the spend.
- **Refund integrity:** the client can verify a refund under the issuer's public key.

The protocol layer must guarantee everything else:

- each nullifier is accepted at most once, across inputs, requests and recovery (atomic recording, the duplicate check);
- a request is accepted for all its inputs or for none;
- `Σ aᵢ = C ≤ H` and `0 ≤ tᵢ ≤ sᵢ`;
- each refund is issued under its input's own key, and no partial return moves between inputs;
- each input's key is accepting when the request is accepted;
- every recorded input receives a refund, and each refund is recoverable per input;
- the wire encoding is canonical and the cost of verification is bounded;
- the client's choices of inputs, shares and order disclose no more than §9 states.

### 8.3 Soundness of the composition

**Value.** Fix an issuer key `K`. Every input under `K` that a server accepts is one standard ACT spend (an unmodified spend proof, verified by the unmodified verifier) answered by one standard refund with a partial return `tᵢ` in `[0, sᵢ]`. A partial return is a choice the honest issuer makes, within the bound ACT enforces, and it enters the credential only as credits the issuer itself signs. Unforgeability bounds what an adversary can spend by what the issuer has signed (issuances plus returns), whatever the issuer's policy for choosing returns, provided each is chosen within `[0, s]` after the proof is verified; the in-order allocation is such a policy, even though it depends on the other inputs. (This reading of the ACT security property should be confirmed against its formal statement when one is published; the specification currently states unforgeability informally.) So the inputs under `K`, across every request, are a sequence of standard ACT spends, and the credits they consume can never exceed what was signed under `K`. The multi-input layer adds three obligations, each discharged above: no nullifier is accepted twice (§6.1 step 3, §6.2), every partial return is within its own input's share (§6.3), and every refund is under its own input's key (§6.4). Because no refund ever carries value from one input to another, the per-key argument composes across keys without any cross-key accounting: the service a request receives is paid for by `C = Σ aᵢ`, and each `aᵢ` is consumed from the credential that presented it.

**Privacy.** Each proof uses fresh randomness, and ACT's unlinkability is perfect: given its public values, a proof's distribution is independent of which credential produced it. The proofs of one request are produced independently, so their joint distribution, given their public values, is independent of which credentials produced them. The issuer's view of a request is therefore simulatable from the tuple `((issuer_key_id, kᵢ, sᵢ, ctxᵢ) for i in 1..N)`, the order, and the refunds it computed itself. The nullifiers are uniformly random values the issuer has never seen, and the successors are unlinkable to the spends by ACT. So everything a multi-input request discloses beyond a single spend is in that tuple: the shares, the keys, their count and order, and the fact that they were presented together. §9 analyses exactly that.

**Interference between proofs.** Proofs of distinct credentials share nothing: each Fiat–Shamir challenge is computed over its own nullifier, request context and commitments, and in the current editor's draft each proof's session identifier includes its own `k` and `ctx`. A proof cannot be satisfied by material from another proof, since the verifier recomputes every proof's commitments from that proof alone. Two proofs with the same nullifier are the one case that matters, and the duplicate check refuses them.

**Keys.** A proof verifies only under the key that issued its credential (the verifier computes `Ā = A′ · x` with the named key's secret `x`), and `ctx`, which binds the key's identifier, is checked against the named key. An input cannot claim one key and be paid from another.

### 8.4 Binding proofs to a request

An ACT spend proof is not bound to anything outside itself. Its Fiat–Shamir transcript covers its nullifier, its request context and its own commitments, but not the HTTP request, the Token's `challenge_digest` or `issuer_key_id`, the refund-recovery key, or the other inputs. Anyone who holds a proof can present it in a different request, alone or with other proofs, and the first presentation to arrive wins. The question is whether multi-input requests need a per-request binding that single spends do not.

**Who could splice.** Only a party that sees proofs: the client (splicing its own proofs into another request is simply spending them there; nullifiers keep it single-use), the issuer (which verifies and charges and could serve or refuse a request regardless, so a binding constrains it in nothing), or a thief who obtained proofs from the network, a TLS terminator that is not the verifier, a log, or the client's store. In Eidola's deployment the first two are harmless and the third is excluded by assumption 3 (TLS terminates in the enclave) and assumption 4 (local adversaries are out of scope).

**What a thief gains, and whether multi-input changes it.** A thief who presents a stolen proof first obtains up to its share's worth of service. The refund is useless to the thief, since completing it needs the victim's `PreRefund` secrets. Worse, with sealed refund tokens the stored refund is sealed under the thief's recovery key, so the victim cannot recover even its remainder. Splicing proofs out of one multi-input request into others is the same theft, input by input. Single spends have the same exposure today, so multi-input requests add no new class of attack: stripping an input from a request leaves that request underfunded (class 3, full refunds of the rest) and the stripped input stolen.

**Conclusion.** No per-request binding is required for the soundness of the composition, and none can be built at the protocol layer: a binding needs a secret the server can check, and the only client secrets the server checks are inside the proofs. So the protocol layer cannot bind a proof to the request it travels in.

### 8.5 Double spends and replay

- **Within one request:** repeated nullifiers are refused before anything is recorded (§6.1 step 3); the atomic insert would refuse them anyway.
- **Across requests:** nullifiers are recorded atomically; a request that repeats any recorded nullifier records nothing (§6.2).
- **Against recovery:** recovery of an unrecorded input records it, so a later request containing it is refused whole, and the request's other inputs remain recoverable (§6.5).
- **After a crash mid-request:** the nullifiers are either all recorded or none are, so no input is half-spent, and the provisional refunds (§6.7) mean no remainder is lost.
- **Several refunds of one proof** (provisional, then final; or a recovery that raced the response): the successors share one nullifier, so at most one is ever spendable, and no value is created. The provisional marker keeps the client from spending the smaller one while the larger may still come, and every replacement of a provisional refund is a compare-and-swap, so a refund recovery declared final is never followed by a different final one (§6.7).
- **No forks.** A fork would be two successors of one spend that are both spendable. Every refund the server issues for a spend signs the commitment `K′` carried inside that spend's proof, and `K′` fixes the successor's nullifier `k*`, so every successor of one spend, however many refunds it receives and whatever their `t`, has the same `k*`. Spending any of them records `k*` and refuses the rest, and every later successor in the chain descends from the one spend that was accepted. Three server rules keep this airtight, and each is already required: a refund is only ever issued for a proof whose nullifier is recorded (§6.1, §6.5), recovery of a recorded nullifier returns the stored refund and never signs a fresh full refund, and a nullifier row outlives every successor that shares its key (§8.6). A refund cannot be issued against a different `K′`, because `K′` is inside the verified proof and not chosen by the server.
- **After a key's deadline:** an input under a key past `accept_until` refuses the request before recording (class 1). A nullifier is never pruned while its key could accept a spend, so a spend cannot be replayed after its nullifier was pruned.

### 8.6 Key rotation and epochs

The epoch rule is that no value extends into a later epoch beyond the existing grace of at most one epoch. A credential under key `K` is spendable until `accept_until(K)`, and every successor of it is a refund under `K`, lapsing at the same instant. Multi-input requests keep this exactly: each input's successor is under that input's own key and holds at most what the input held (§6.3), so no value moves from an older key to a newer one, and no credential is ever minted under a later key from value presented under an earlier one. A request mixing keys consumes the older key's value first when the client orders drains by expiry, which is the only effect mixing has on value. A request is accepted only if every input's key is accepting at the server's check (§6.1 steps 2 and 4), and each refund is signed during that key's retention margin, as for single spends.

### 8.7 Cost and denial of service

Verification is the expensive step, and it runs before anything is recorded, so an unauthenticated sender can make the server verify up to `N_max` invalid proofs per request instead of one. Verifying plus signing a refund takes about 1.8 ms per input at `L = 32` (§5.5; enclave hardware will differ). So `N_max = 2` bounds a bogus request to roughly 4 ms of verification. The bounds:

- the count is checked before any proof is decoded, and the cheap static checks for every input (key, context, digest, share, duplicates) run before any proof is verified (§6.1);
- verification stops at the first invalid proof; which index failed is visible only to the sender;
- the gateway currently verifies a proof once in pre-flight and again inside the refund call at settlement (`PrivateKey::refund` verifies before it signs). With several inputs, an implementation should verify once and sign refunds from the verified proofs, which needs either a crate change or accepting the doubled cost.

Rate limiting unauthenticated traffic is outside this specification, as it is for single spends.

## 9. Privacy analysis

### 9.1 Presenting inputs together

What is new is that the issuer learns that `N` credential instances, under the presented keys, belong to one client at one instant. Three consequences.

**The instances, not the chains, are linked.** ACT successors are unlinkable to their predecessors, so the link covers only the presented instances in this request. Later spends from their successors are as unlinkable to this request, and to each other, as any spends, except through the amounts discussed in §9.2. And since the inputs are all in one request, the link adds nothing at the level of requests: a request was already one client's.

**The anonymity set shrinks to accounts with enough issuances.** A chain of credentials starts at one issuance and has one live instance at a time, so `n` inputs under key `K` come from at least `n` issuances under `K` (to one account, or to accounts pooling credentials). Let `A_n(K)` be the accounts issued at least `n` credentials under `K`; the issuer (and, in model B, the database operator) can compute it from the issuance ledger. A single spend under `K` has anonymity set `A₁(K)`, as today. A request with `n_K` inputs under each key `K` in a set `S` has anonymity set `⋂_{K∈S} A_{n_K}(K)`. Concretely:

- a drain and a fill under one key: `A₂(K)`, the accounts issued at least twice under `K` that epoch;
- a residual under the older key and a fill under the newer one: `A₁(K_old) ∩ A₁(K_new)`, the accounts issued in both epochs;
- with only two keys accepting at any time, the possible key sets are `{K_new}`, `{K_old}` and `{K_old, K_new}`.

For a client that drains, the second issuance under `K` happens anyway: the drain exists because a credential ran low and another was issued. What multi-input adds is that one request is placed in `A₂(K)`. If, say, 30% of an epoch's accounts are issued twice, a drain request's anonymity set is 30% of a single spend's. In a small deployment, where `A₁(K)` is already small (see [Known gaps → Anonymity-set size](../gaps.md#anonymity-set-size)), this can be decisive.

**Making `A₂(K)` the common case.** The reduction is a function of how many accounts have two issuances under a key, and that is a client policy, not a property of usage. Eidola's client provisions credentials ahead of need. If it provisions **two** credentials at the start of each epoch it uses, each of one public denomination, then every provisioning account is in `A₂(K)` from the start, and `A₂(K) = A₁(K)` for those accounts: a drain-and-fill request is placed among every account that provisioned under `K`, not among the heavy users. A further issuance under `K` moves an account into `A₃(K)`, which matters only if `N_max ≥ 3`; with `N_max = 2` nothing a request presents can distinguish two issuances from more. This is why Eidola caps `N_max` at 2 and pairs it with two-credential provisioning. Both are credentials of the one public $1 denomination, so provisioning adds no new amount to the issuance surface. The cost is that a client holds $2 of credits per epoch rather than $1, and more value is exposed to lapse at a key's `accept_until`; the client's existing rules (spend the soonest-expiring credential that fits, and drain an expiring residual, §7.6) keep that small.

**Compared with the alternatives (§10).** A same-key merge endpoint links the same instances and places the merge in the same `A_n(K)`, and adds an extra request whose timing is separate from any inference request. Single spends link nothing, but strand residuals and, if holds can approach the denomination, push towards issuances sized to holds, which link an account to a request outright.

### 9.2 Draining reveals a balance

A drain's share is its credential's balance, so the issuer learns that balance exactly (or to precision `Δ`). ACT hides balances during spends; a drain gives up that property for the drained credential, and nothing else does.

**What a balance reveals.** With a public denomination `D`, a drained balance is `D` minus the charges the chain has paid. To an issuer that remembers every charge under `K` (model A), it is a subset-sum constraint on that epoch's charges. With `M` charges in the epoch, each of up to about `W_max` credits, the number of chains of length `ℓ` consistent with one balance is, heuristically (treating the charges as independent and spread over their range), about `C(M, ℓ) / (ℓ · W_max)`, which is enormous unless one client dominates the epoch (for example `M = 50`, `ℓ = 30`, `W_max = 62,500` gives about 2.5 × 10⁷ candidates). The constraint becomes identifying only when the epoch's traffic is so thin that the anonymity set is already near one. With a fixed denomination, a drain of a share exactly `D` reveals more: the credential was never spent, since no credential then exceeds `D`.

**Issuer-known successors.** A drained input whose share is not fully consumed (`C < Σ` of the shares up to and including it) receives a partial return, and its successor holds exactly `tᵢ`: the issuer computed `tᵢ`, and a drain leaves no hidden remainder. If that successor is later drained exactly, its share equals a partial return the issuer issued, which links the two requests (and a run of such drains links a run of requests). Charges are usually well below `W`, so this is common, not marginal. In model A a client SHOULD therefore do one of these:

- round drains down to a step `Δ`, which leaves a remainder below `Δ` that the issuer does not know, so a later drain of the successor matches only to precision `Δ`;
- or strand a successor whose exact value the issuer can compute, and never present it again. That loses at most `tᵢ` and still recovers `min(drain, C)` of the residual.

The allocation rule is what keeps this from being worse. Because `tᵢ` is determined by the shares, their order and `C`, the issuer cannot choose a partial return to mark a drained successor, as it could if it chose each `tᵢ` freely; and a client can check the rule (§7.7). The issuer could still mark one by misreporting `C`, which the client can detect only as far as it can check `C` against reported usage.

**Why exact drains are acceptable for Eidola (model B).** No adversary outside the enclave sees a share, a charge or a partial return: they are in the spend proofs and computations inside the enclave, and the stored refunds are sealed and fixed-length. The measured server code retains no per-request amount, so no party can run the subset-sum or matching attacks above. Eidola's client therefore drains exactly (`Δ = 1`) and spends issuer-known successors normally. The residual is that this rests on the measured code, not on the cryptography, for the drained credentials (§13.1 proposes how the privacy guarantees should say so).

### 9.3 Issuance-adjacent timing

The issuer sees issuance times on the account-linked surface and request times on the anonymous one. A client provisioning credentials on demand already sends its first spend from a new credential shortly after the issuance; that spend is one of many single spends under the key. A drain-and-fill request sent right after an issuance would be much rarer (about one per credential lifetime, which with typical charges far below the worst case spans tens to hundreds of requests), and would sit next to an account's second issuance under the key. That would correlate the account with the request far more sharply. A deferral rule would avoid that: no drain in the first request after an issuance, and drains at randomly chosen later requests. Eidola does not adopt one, for three reasons. Provisioning ahead of need, at randomized times, already separates issuances from the requests that follow them. Provisioning two credentials per epoch makes the second issuance under a key ordinary rather than a sign of a drain (§9.1). And a deferral rule needs wallet state ("issued since the last spend") written by the background provisioner and read by concurrent requests, which is the kind of race-prone coupling the rule of §7.2 is designed to avoid. A deferral would never make a request impossible (a fresh credential always covers a hold up to the denomination), so it remains available if the analysis changes.

### 9.4 Input count and order

**Order.** The canonical order (§7.3) is a function of the keys and shares already visible, plus a random tie-break, so it discloses nothing further. A client using another order would expose its implementation, and whatever its order depends on.

**Count.** `N` discloses how many drains are due, which is a function of the client's wallet state. Under §7, `N` is 1 for most requests and 2 when draining, and Eidola's `N_max` allows no more. Every client that uses its credentials down drains, in runs (§7.2), so `N = 2` is common to all such clients.

**Padding to a fixed count was considered and rejected.** Padding with zero-share inputs (ACT permits spending zero, which only refreshes a credential) would make every request present `N_max` proofs. That needs `N_max` live credentials per request (most clients hold one or two), multiplies proof size and verification by `N_max` on every request, and every padding input would still disclose its key and that its holder owns at least that many credentials, which is the §9.1 linkage again, so padding trades one fingerprint for another, larger one. For the same reason, servers refuse zero-share inputs: they pay for nothing and are a degree of freedom for fingerprinting. Whether the working group wants fixed or bucketed counts is an open question.

### 9.5 Network observer

Requests are not padded: their size already reveals the prompt's size. Each input adds about 6.7 KB, so a network observer learns `N` from a request's size, and also from the server's longer pre-flight (about 2 ms per input). That is wallet state, not content, and it is the same `N` the issuer sees. The response's `refunds` add about 330 bytes per input, which can move a padded response into the next size bucket or a streamed response's metadata into one more frame; that reveals the same `N`.

### 9.6 Database observer (model B)

The database sees what the issuer's store holds: one `nullifier` row per input, with its key, and the sealed refunds. Rows recorded in one transaction share a commit, so the database learns each request's `N` and key set, which, joined with the issuance ledger in the same database, gives it the anonymity-set reduction of §9.1. Amounts stay sealed. Batching nullifier inserts from different requests on a fixed interval would blur which rows belong together (the same mechanism proposed for the refund-write timing channel). It is noted here as a possible mitigation, not specified.

### 9.7 Interactions

- **Sealed refund tokens.** Each input carries its own recovery key derived from its own `PreRefund`, and its refund is sealed on its own row (§5.1, §6.5). Nothing new is persisted in the clear, and recovering one input never requires another's key.
- **Public denomination.** Provisioning issues credentials of one public $1 denomination. A hold up to $1 is always covered by a fresh credential, so multi-input requests serve mainly to drain residuals, recovering what would otherwise be stranded. With `N_max = 2`, two credentials can also cover a hold above what either holds alone. A hold no pair covers falls to step 4 of §7.2 and the client's ordinary allocation rule.
- **Holds and ceilings.** `W` is computed exactly as today, from the request and the model's declared completion ceiling. The server's check is `H = W`, with `H` the sum of the shares, for one input as for two. A request without reported usage is charged the whole hold, which is therefore exactly `W`.
- **Tool rounds and continuations.** Every round of a tool loop takes its own inputs, exactly as it takes its own credential today. A round may drain (§7.2), and a hold may not be abandoned unsettled to take another, for every input.
- **Prefix caching.** A `cache_key` links the requests of one lineage whatever their inputs, and multi-input requests change nothing about it. If charges ever depend on whether a prompt prefix was cached (a declared-prefix discount), `C` falls, so drains are less often fully consumed, making issuer-known successors more frequent in model A, and the hold `W` must still assume no discount, since the client cannot know the cache state.

## 10. Alternatives considered

| | Single spend (today) | Same-key merge endpoint | Split endpoint | Multi-input spend (this) | Aggregate proof (new ACT) |
|---|---|---|---|---|---|
| Mechanism | One proof per request | A separate request presents several drained credentials under one key; the issuer mints one credential of their total under that key | One spend; the issuer mints several credentials of public sizes | Several unmodified proofs per request; in-order allocation; one refund per input | One proof over several credentials with hidden shares and one successor |
| Stranded residual | Up to one hold per credential | Recovered at merge time | Not addressed | Recovered by drains | Recovered, with balances hidden |
| Large holds | Need one large credential | One merged credential | Concurrency, not size | Several credentials | Several credentials |
| What the issuer learns beyond today | — | Merged balances, linkage of the merged instances, a minted value the issuer knows, the merge's timing | Minted sizes and timing | Drained balances, linkage of co-presented instances (§9.1, §9.2) | Linkage only |
| Extra round trips | — | One per merge | One per split | None | None |
| ACT change | — | Minting from proofs needs a fresh issuance on the anonymous surface, or a refund with `t > s` (ACT forbids) | Fresh issuance on the anonymous surface | **None** | New proof system and a security proof |
| Epoch rule | Holds | Holds if same-key only | Holds if same-key only | Holds | Holds only for same-key inputs |
| Server complexity | — | New endpoint, minting path | New endpoint, minting path | Changes to the extractor, recording and settlement | New cryptography |

A merge endpoint has one real advantage: inference requests stay single-input, and merges can be scheduled at random times, independent of inference. But it mints a credential whose initial value the issuer knows, which a refund successor never has, and its value is not the public denomination, which makes it a rarer, more recognisable credential. It also needs a minting path on the anonymous surface, which does not exist and which ACT's refund cannot provide (a refund may return at most the spent share). Multi-input spending gets the same benefit without minting and without an extra request. An aggregate proof would hide shares and remove the drain disclosure, but it is new cryptography, it still needs same-key inputs to respect the epoch rule, and it belongs with the ACT authors.

Rejected outright: rolling a credential's remainder into the current key at expiry (moves value into a later epoch), and issuing credentials sized to the hold (links the account to the request).

## 11. Migration and compatibility

Eidola is pre-production and starts fresh: no installed client or server needs to interoperate across this change.

- **Single spends become the one-input case.** The Token's bytes, the refund-recovery key and the recovery endpoint keep their meaning; only their transport changes.
- **Transport.** The `Authorization: PrivateToken` header and the `x-eidola-refund-key` header are replaced by the body's `credentials` array (§5.1), on inference and recovery alike, and a credential in a header is refused.
- **Response shape.** `refund` (an object) becomes `refunds` (an array of one per input) in blocking responses, error bodies, recovery answers and the streamed metadata event; `openapi.json` is regenerated.
- **`L` from 128 to 32.** A wire change for proofs and issuance alike: every `::<128>` in client and server becomes the one shared constant, and existing credentials are not carried over.
- **Server store.** No schema change: one `nullifier` row per input, as today. Recording becomes a multi-row transaction. Provisional refunds (§6.7) write `refund_token` at insertion, in the same statement, with the stage (provisional or final) as the first byte of the sealed plaintext.
- **Client store.** In-flight inputs are grouped per request (a group identifier on the pending-refund rows), which is a schema change under the fresh-start rule (bump the local schema version; old development databases are refused).
- **Privacy guarantees.** The new disclosures (§9.1, §9.2) are reflected in `docs/privacy-guarantees.md` §2 and `docs/gaps.md` when this is implemented; proposed wording is an open question below.
- **Privacy Pass interoperability.** The body transport is specific to this extension: a generic RFC 9577 origin expects a Token in the `Authorization` header and would treat a request carrying only `credentials` as unauthenticated, recording nothing. A client can therefore use it only with servers it knows implement it; Eidola's client knows its server.

## 12. Test-vector and test plan

**Allocation vectors (normative).** Tables of `(shares in order, C) → (aᵢ, tᵢ)`, including: `N = 1` (today's single spend); `C = 0`; `C = H`; `C` at every prefix boundary (exactly `s₁`, `s₁ + s₂`, …); `C` one either side of each boundary; shares near `2^63` and checked overflow of `H`; and the example of §6.3. Property tests: `Σ aᵢ = C`, `0 ≤ tᵢ ≤ sᵢ`, the prefix structure, and determinism.

**Wire vectors.** Deterministic proofs from a seeded RNG (the crate accepts any `CryptoRngCore`; the ACT specification's appendix parameters and key), producing: the `credentials` member for `N = 1` and `N = 2`, and the expected refunds for given `tᵢ` (refund randomness seeded too). Negative vectors, each refused before anything is recorded: an empty array; `N > N_max`; `credentials` not an array; a Token or refund key in a header; an unknown, missing or duplicated member; an unpadded value; non-zero pad bits; a Token with trailing bytes; mixed `token_type`; mixed `challenge_digest`; a duplicated nullifier (same credential, two shares); a zero share; uppercase hex; whitespace or a wrong length in a refund key.

**Server processing, against a real Postgres.** One invalid proof among valid ones records nothing; one already-recorded nullifier rolls back the rest, and recovery then fully refunds each of the others; mixed keys settle each refund under its own key and each successor completes only under its own key; an input under a key past `accept_until` refuses the request; a request accepted just before one input's deadline settles every input after it; recovery of an unrecorded input followed by its request's late arrival records nothing for the request; concurrent requests sharing nullifiers in opposite orders do not deadlock; a crash injected after recording (with and without provisional refunds) leaves each input's remainder recoverable as specified; class 3 and class 4 failures return full refunds for every input.

**Client, in the chat harness.** Every exit point of a multi-input round settles every input; inputs are taken atomically (a crash between proofs leaves none marked spending); canonical order; no drain right after an issuance; drains defer and still happen before a credential's deadline; mixed keys only to drain an expiring residual; a refund that does not verify triggers recovery of that input alone; a response whose partial returns break the prefix structure is surfaced; and the bus exit-point table gains the new rows.

**Limits.** A request with `N_max` Tokens and a large prompt passes the attestation shim and the gateway's body limit, over HTTP/1.1 and HTTP/2.

**Teeth.** As the repository's practice: each property above, mutated (allocation from the end instead of the start, a non-atomic insert, a missing duplicate check, a partial return moved between inputs, a refund signed under the wrong key), makes its tests fail.

## 13. Open questions

### 13.1 For the maintainer

Decided:

- **`N_max = 2`** (§5.5), paired with provisioning two $1 credentials per epoch (§9.1), revisited when the population grows.
- **Mixed keys** are allowed, to drain an older key's residual before it stops accepting (§7.6). Each input's refund stays under its own key.
- **Provisional refunds** (§6.7) are adopted for single and multi-input requests alike, marked provisional, final as soon as no request is serving them (the gateway's in-memory live set), every replacement a compare-and-swap. A stream cut short is charged for what was generated.
- **No presentation binding.** A per-request binding of spend proofs is a change to ACT, not something this extension can build, and it is not needed for soundness (§8.4).
- **Body transport, exclusively** (§5.1), with `credentials` and `refunds` always arrays.
- **`L = 32`** (§5.5).
- **Input choice without thresholds** (§7.2): oldest key first, then smallest; drain the first credential when it cannot cover the hold. Exact drains; rounding deferred until the population makes it meaningful; no deferral rule (§9.3).
- **Hold equality:** `H = W` exactly, for every request (§7.4).

Open:

1. **Privacy-guarantee wording.** §2.2 says redemptions are cryptographically unlinkable. With exact drains, that remains true against everyone outside the enclave but rests on the measured code for drained credentials (§9.2). A proposed addition: *"A request may present up to two anonymous credentials. Credentials presented together are known to come from one client, which places the request among the accounts issued at least two credentials under each presented key; the client provisions two credentials per epoch so that this is every account that provisioned. A credential drained to zero discloses its balance to the server's measured code, which retains no per-request amount."*

### 13.2 For a working-group discussion

1. **Scope.** Is presenting several ACT Tokens in one HTTP request in scope for the Privacy Pass ACT binding, or a deployment matter?
2. **Transport.** One ACT Token is larger than many deployed header limits at practical `L` (§5.1), so Eidola carries Tokens in the request body. Should the Privacy Pass ACT binding define a body transport, for one Token or several, alongside RFC 9577's `Authorization` header, and how should an origin signal which it accepts?
3. **Allocation as origin policy or as specification.** The base specification lets the origin choose `t`. A multi-input binding needs a public, deterministic rule so clients can order inputs for privacy and detect deviations; is in-order consumption the right one to standardize?
4. **Advertising `N_max`.** A challenge attribute, alongside `cost`?
5. **Challenge `cost` with several Tokens.** Should `Σ sᵢ` equal `cost` exactly, or be at least `cost`?
6. **Count fingerprinting.** Fixed or bucketed counts (padding with zero-spend proofs), given §9.4's argument that padding leaks a larger fingerprint than it hides?
7. **Same-key aggregate spends.** Is a proof over several same-key credentials, with hidden shares and one successor, worth specifying in the ACT cryptography document, to remove the drain disclosure?
8. **Refund transport.** For a list of refunds in `PrivacyPass-Reverse`, the encoding, and trailers for streamed responses.

## References

- `draft-schlesinger-cfrg-act-01` and the current editor's draft: *Anonymous Credit Tokens*.
- `draft-schlesinger-privacypass-act-01` and the current editor's draft: *Privacy Pass Issuance Protocol for Anonymous Credit Tokens*.
- `draft-meunier-privacypass-reverse-flow` and `draft-meunier-privacypass-reverse-flow-http`.
- RFC 9576 (*The Privacy Pass Architecture*), RFC 9577 (*The Privacy Pass HTTP Authentication Scheme*), RFC 9578 (*Privacy Pass Issuance Protocols*).
- RFC 9110 (*HTTP Semantics*), RFC 4648 (*Base-N encodings*), RFC 8446 §3 (*presentation language*).
- `anonymous-credit-tokens` 0.4.2, the implementation Eidola uses.
- In this repository: [The server](../server.md) (anonymity set, key epochs), [Privacy guarantees](../privacy-guarantees.md) §2, [Known gaps](../gaps.md), and the gateway's credential code (`crates/eidola-server-gateway/src/credentials.rs`, `handlers.rs`, `auth.rs`).
