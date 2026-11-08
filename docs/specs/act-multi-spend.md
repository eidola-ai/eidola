# Spending several anonymous credits in one request

**Status: design specification, not implemented.** Nothing in this document describes the current behaviour of the Eidola server or client. It specifies an extension to how Eidola presents Privacy Pass Anonymous Credit Tokens (ACT), written so that its normative parts can be adapted into a section of an Internet-Draft. Where it depends on mechanisms that are proposed but not yet merged (sealed refund tokens and the `x-eidola-refund-key` header; the client's spend deadline of `accept_until` minus a margin; a fixed public issuance denomination), it says so and assumes the proposed behaviour.

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
- **Drain**: an input whose share equals the credential's entire remaining balance (`sᵢ = cᵢ`), or that balance rounded down to a public step (§7.5).
- **Fill**: the input whose share is derived from the hold rather than from a balance (`s_fill = W − Σ drains`).
- **Accepting key**: an issuer key `K` with `now < accept_until(K)`. Keys rotate weekly and accept for one further epoch, so at most two keys accept at once (three for an instant at a boundary).

Integer amounts are credits. All arithmetic on amounts is exact unsigned integer arithmetic; an implementation MUST use checked operations and treat overflow as a malformed request.

## 3. Overview

```text
Client                                                    Server (in enclave)
  | choose inputs 1..N and shares s1..sN (§7)                  |
  | ProveSpend(cred_i, s_i) for each i, mark all spending      |
  |-- POST /v1/chat/completions -----------------------------> |
  |   Authorization: PrivateToken tokens="<TokenList>"         | parse all, static checks (§6.1)
  |   x-eidola-refund-key: <key1>,<key2>,...                   | verify every proof
  |                                                            | record all nullifiers atomically
  |                                                            | H = Σ s_i; check H ≥ W
  |                                                            | run inference, settle C ≤ H
  |                                                            | allocate C in order (§6.3)
  |                                                            | refund_i = VerifyAndRefund(sk_i, proof_i, t_i)
  |<-- response, "refunds": [refund_1, ..., refund_N] -------- |
  | successor_i = ConstructRefundToken(pk_i, proof_i,          |
  |               refund_i, state_i)                           |
```

A request with one input is today's single spend: the same header parameter, the same proof, the same processing and the same refund. Only two spellings change, for every request alike (§11): the token value gains base64url padding, and the response's refund moves into a one-element `refunds` array.

## 4. Relationship to the ACT cryptography

The extension changes nothing in ACT. Every input is produced by an unmodified `ProveSpend`, verified by an unmodified `VerifySpendProof`, and answered by an unmodified `IssueRefund` with a partial return `tᵢ` the issuer chooses within the bound ACT already enforces (`0 ≤ tᵢ ≤ sᵢ`). What the extension adds is protocol-layer composition: how inputs are carried, the all-or-nothing acceptance rule, the deterministic allocation that chooses each `tᵢ`, and recovery. §8.3 argues why that composition is sound and §8.4 why no per-request cryptographic binding is required for it, while proposing one as an optional improvement to ACT itself.

## 5. Wire format

### 5.1 Request: the Authorization header

A request with **one** input is unchanged. It carries one Token (`draft-schlesinger-privacypass-act`, Token Creation) in the `token` parameter of the `PrivateToken` authentication scheme (RFC 9577, Section 2.2.2):

```text
Authorization: PrivateToken token="<base64url(Token)>"
```

A request with **two or more** inputs carries them in a new `tokens` parameter of the same scheme, whose value is the base64url encoding of a `TokenList`:

```text
Authorization: PrivateToken tokens="<base64url(TokenList)>"
```

```text
struct {
    opaque token<1..2^16-1>;   /* one Token, exactly as the base specification defines it */
} TokenEntry;

struct {
    TokenEntry entries<1..2^24-1>;
} TokenList;
```

The order of `entries` is the order of the inputs: entry 1 is input 1. Rules:

- **Exactly one form.** A request MUST carry exactly one of `token` and `tokens`, never both. `tokens` MUST carry at least two entries: a single input MUST use `token`. So every input count has one encoding, and a single input is byte-identical to the base specification.
- **Count.** `2 ≤ N ≤ N_max`. A server MUST refuse a request with more than `N_max` entries before decoding any spend proof.
- **Self-delimiting entries.** Each entry is length-prefixed, so the list parses whether the Token's spend-proof field is fixed-length (the `-01` CBOR encoding Eidola uses, where the proof runs to the end of the Token) or itself length-prefixed (the current editor's draft). Each entry MUST parse as exactly one Token with no trailing bytes, and the list MUST have no trailing bytes.
- **Uniform type.** Every entry MUST carry the same `token_type` (`0xE5AD`) and the same `challenge_digest`. Entries MAY name different `issuer_key_id` values (§7.6 says when a client should).
- **Encoding.** RFC 9577 requires the base64url value to include padding. Both parameters MUST use base64url (RFC 4648, Section 5) with padding, and a server MUST refuse a value that does not re-encode to itself (non-zero pad bits, missing padding, or whitespace). The value is sent as a quoted-string, because padding introduces `=`. The current Eidola implementation omits padding on the single-token path; §11 changes that, so there is one canonical spelling.
- **Unknown parameters.** RFC 9577 requires unknown `PrivateToken` parameters to be ignored, so a server that does not implement this extension sees a request with `tokens` and no `token`. It refuses that request as unauthenticated before recording anything (§6.6, class 1), so a client talking to an older server loses nothing.

The `Authorization` field carries one set of credentials and is not a list-based field (RFC 9110, Section 11.6.2), so the inputs cannot be spread across several `Authorization` lines. RFC 9110 (Section 11.2) lets a parameter name occur only once per challenge, and repeating `token` in credentials would invite the same ambiguity: an RFC 9577 origin reading the first or the last `token` would process one input of several. A new parameter of the existing scheme carrying one list is the least disruptive encoding that avoids both.

### 5.2 Request: refund-recovery keys

With sealed refund tokens (proposed alongside this specification), every spend carries `x-eidola-refund-key`, 32 bytes as exactly 64 lowercase hexadecimal digits, derived by the client from that spend's `PreRefund` state. Each input has its own `PreRefund`, and therefore its own key. A request with `N` inputs carries exactly `N` keys, in input order, joined by single commas with no whitespace:

```text
x-eidola-refund-key: 3f2a...c901,8be0...11d4
```

The field MUST appear on exactly one field line, with exactly `N` entries. A server MUST refuse any other spelling (uppercase digits, whitespace, an empty entry, a count that does not match the inputs) with `400` before recording any nullifier. With one input the field is byte-identical to the single-spend form.

### 5.3 Response: refunds

Every response that follows the recording of the inputs' nullifiers (§6.6, classes 3 to 5) carries one refund per input, in input order, in a `refunds` array that replaces today's single `refund` object, in the JSON body of a blocking response or error and in the final metadata event of a streamed response:

```json
"refunds": [
  { "refund": "<base64url(Refund)>", "issuer_key_id": "<hex>" },
  { "refund": "<base64url(Refund)>", "issuer_key_id": "<hex>" }
]
```

`refunds[i]` answers input `i`. The array MUST have exactly `N` elements. `issuer_key_id` repeats input `i`'s key; the client MUST check that it matches. A response to a request refused before its nullifiers were recorded carries no `refunds`.

Refunds travel in the body rather than in a response header, because a streamed response's settlement is known only when the stream ends, after its headers were sent. A generic Privacy Pass binding of this extension that carries refunds in the `PrivacyPass-Reverse` field (as the current ACT editor's draft does, following `draft-meunier-privacypass-reverse-flow-http`) would carry a list in input order, as an HTTP trailer when the response streams.

### 5.4 Refund recovery is per input

`POST /v1/credentials/refund` is unchanged and takes exactly one input in the `token` parameter, with that input's one refund-recovery key. A recovery request carrying `tokens` MUST be refused. Each input of a multi-input request is recovered by its own nullifier and its own key, independently of the others (§6.5).

### 5.5 Sizes and limits

Eidola's ACT parameters use `L = 128`. One spend proof is 18,071 bytes of CBOR, so one Token is 18,137 bytes and one `token` value is 24,184 base64url characters. One refund is 176 bytes of CBOR. With `N_max = 4` the `tokens` value is at most 96,748 characters.

Header-size limits along the request path MUST admit `N_max` Tokens. Today's single Token already depends on the path admitting about 24 KB in one header. Verifying a larger `N_max` against every hop (the attestation shim, the gateway's HTTP stack) is part of the test plan (§12). Proof size scales linearly with `L`, so a smaller `L` is the main lever on it (§13.1).

**`N_max` for Eidola is 4.** The client policy of §7 rarely needs more than 2 inputs and never more than 3 (one fill and at most one residual under each of the two accepting keys). 4 leaves headroom without letting a request cost more than a few proof verifications (§8.7).

## 6. Server processing

### 6.1 Order of operations

A server MUST process a request in this order, and MUST NOT record any nullifier before step 5.

1. **Parse.** Decode the `Authorization` value per §5.1 and the refund keys per §5.2. Refuse on any error, on `N > N_max`, or on a key count different from `N`.
2. **Static checks, every input.** `token_type` is `0xE5AD`. `challenge_digest` equals the expected digest. `issuer_key_id` names a key the server holds that is usable for a spend now (`now < accept_until`). The proof's `ctx` equals that key's request context. The share decodes to an integer (`ScalarToCredit`) and is at least 1 (zero shares are refused; see §9.4). The hold `H = Σ sᵢ` is computed with checked arithmetic.
3. **Duplicates.** The inputs' nullifiers MUST be pairwise distinct, compared per issuer key. A repeated nullifier refuses the whole request. Two inputs with the same nullifier are two spends of one credential instance; ACT permits a client to produce them (with different shares), and they are exactly the forking attempt nullifiers exist to stop.
4. **Verify every proof.** `VerifySpendProof(skᵢ, proofᵢ)` under each input's own key, with the key's spend deadline re-checked under the same lock or snapshot the verification uses. All proofs must verify. A server SHOULD run the checks of steps 1 to 3 for every input before verifying any proof, since they are cheap and the proofs are not (§8.7).
5. **Record atomically** (§6.2). If any nullifier is already recorded, record none and refuse the request.
6. **Validate the request** against the pricing contract: the model exists, the completion ceiling is within the model's declared ceiling, and `H ≥ W`. A failure here is after recording, so it MUST be answered with full refunds (§6.6, class 3).
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

A server MUST verify the proof before recording a nullifier on the recovery path, as it does on the spend path. (Today's recovery handler records first and verifies second. A bogus proof cannot exploit that, since it would need a credential's secret nullifier, but the specified order removes the question.)

### 6.6 Failure handling

| Class | Where it fails | Nullifiers recorded | Refunds in the response | What the client does |
|---|---|---|---|---|
| 1 | Parsing, counting, static checks, duplicates, any proof's verification, any key unknown or past `accept_until` | none | none | Recover every input through `POST /v1/credentials/refund` (each gets a full refund into a fresh successor). Do not present the same proofs again (§7.7). |
| 2 | Recording: some input's nullifier already recorded | none (rolled back) | none (`409`) | Recover every input. The one already recorded returns its stored refund if it has one; the others are recorded now and fully refunded. |
| 3 | Request validation after recording (unknown model, ceiling, `H < W`) | all | all, `tᵢ = sᵢ` | Construct every successor. |
| 4 | Dispatch or upstream failure | all | all, `tᵢ = sᵢ` | Construct every successor. |
| 5 | Settled (including a stream that ended early) | all | all, by §6.3 | Construct every successor; discard empty ones. |
| 6 | Response lost after any of 3 to 5 | all | stored, sealed per input | Recover each input by its own nullifier and key. |
| 7 | Process crash after recording, before refunds are stored | all | none stored | See §6.7. |

Every class leaves every input's value either unconsumed and recoverable (1, 2), or consumed only by `aᵢ` with the rest returned in a refund the client can obtain (3 to 6). Class 7 is the exception, and it exists for single spends today. One boundary applies to classes 1 and 2 as it does to single spends: recording a fresh nullifier, on the spend path or the recovery path, is a spend, allowed only while the input's key accepts. An unrecorded input whose key has passed `accept_until` has lapsed with its key; the client's spend margin (§7.2) exists so that no input it sends is that close to its deadline.

### 6.7 Crash safety

Today, a crash between recording a nullifier and storing its refund leaves the nullifier recorded and no refund to recover. The client's recovery is answered "not yet available" forever, and the whole credential (the blinded remainder `c − s` as well as the hold `s`) is lost. A multi-input request multiplies that exposure by `N`.

This specification RECOMMENDS closing it for single and multi-input requests alike: in the same transaction that records the nullifiers, store each input's sealed **provisional refund**, signed with `tᵢ = 0`, and overwrite it with the final refund at settlement. A crash then loses at most each input's share, never its remainder. This is safe for the issuer because two refunds of one spend proof yield successors with the same nullifier. The successor's nullifier `k*` is fixed by the commitment `K′` inside the spend proof, and both refunds sign that same commitment; only `t` differs. A client holding both successors (the provisional `cᵢ − sᵢ` and the final `cᵢ − sᵢ + tᵢ`) can spend only one of them, because spending either reveals `k*`. This was checked against `anonymous-credit-tokens` 0.4.2: two refunds of one proof, with `t = 0` and `t = 100`, completed to two credentials with equal nullifiers and balances differing by 100. A client SHOULD keep the successor with the larger balance and never spend the other.

The cost is one extra signature per input on the pre-flight path, and a recovery during flight returns the provisional refund instead of `404`. A client SHOULD NOT recover an input whose request may still be in flight, which Eidola's client already enforces for single spends. §13.1 lists this as a decision for the maintainer, since it changes recovery's in-flight answer.

## 7. Client behaviour

### 7.1 Taking several credentials at once

The client MUST take every input of a request in one atomic local step: under the same lock that serializes credential acquisition today, it produces every spend proof and durably marks every input `spending`, together with its `PreRefund` state, in one local transaction, before sending anything. The ACT specification requires the client's private state to be stored durably before a spend is sent; with several inputs, they must also be stored together, so a crash cannot leave some inputs marked spending and others not. The inputs of one request are grouped locally so that settlement processes them together, while recovery remains per input.

### 7.2 Choosing inputs

Inputs and shares are chosen so that the amounts reveal nothing that a single spend would not, except the balances of drained inputs (§9.2). For a request with worst-case cost `W`:

1. **No drain due:** if some credential covers `W`, spend the best-fitting one (the smallest that covers `W`) alone, with share `W`. That is today's rule, and the common case.
2. **A drain due** (§7.5): present the drains first, then a fill. The fill is the best-fitting credential covering `W − Σ drains`, preferring one under the same key as the drains (§7.6), with share `W − Σ drains`. If the drains alone cover `W`, the last drain becomes the fill instead: its share is `W` minus the shares before it.
3. **No credential covers `W`, and no combination of due drains and a fill does:** allocate a new credential (one public denomination, never an amount derived from `W`), and spend it alone. That request MUST NOT also drain (§9.3).
4. **A hold no single credential could ever cover** (only possible where holds may exceed the denomination): present drains of as many credentials as needed, soonest-expiring first, plus a fill. This is the case the fixed denomination is designed to make unnecessary; §9.7 discusses it.

The client MUST NOT present an input whose spend deadline (`accept_until` minus the client's margin) has passed. The request's own deadline is the earliest of its inputs'.

### 7.3 Canonical order

A client SHOULD order inputs as a deterministic function of public information only:

1. drains before the fill;
2. among drains, by key `accept_until` ascending (soonest-expiring value is consumed first), then by share ascending, then by nullifier bytes ascending.

Every element of that order is already visible in the request, so the order adds nothing (the nullifier tie-break is uniformly random). An order that depends on anything else (a balance, a local id, a history) would leak it. Whether the server should enforce the canonical order rather than honour any order is an open question (§13.2).

### 7.4 Shares never come from balances, except drains

A share is either `W` (a single input), a drain's balance (§7.5), or a fill derived from `W` and the drains. In the last case, `W` is computed by the server from the request itself and the drain shares are already in the proofs, so the fill share discloses nothing new. A client MUST NOT size any other share from a balance. A partial share sized from a credential's balance would reveal the balance without draining it, and a sequence of such shares would let an issuer that remembers amounts reconstruct a credential's history.

The shares sum to exactly `W`, the same hold a single-input spend of the same request would show. A client MAY over-hold (present `H > W`), but SHOULD NOT, since `H` is otherwise a constant of the request.

### 7.5 When to drain

A credential is a **drain candidate** when it is too small to be the best fit for the holds the client expects: its balance is below a client constant `θ` (RECOMMENDED: the smallest worst-case hold on any model the client offers), or its spend deadline is within a horizon (RECOMMENDED: one day) and it cannot cover the request at hand. A candidate is drained by presenting it with its whole balance as its share.

- **Rounding.** A client MAY round a drain's share down to a multiple of a public step `Δ`, stranding less than `Δ` and revealing the balance only to precision `Δ`. Eidola's client uses `Δ = 1` (exact drains), for the reason given in §9.2. A deployment in which the issuer is an adversary for amounts SHOULD use a larger step, or SHOULD strand issuer-known successors (§9.2).
- **Deferral.** A client MUST NOT drain in the first request it sends after an issuance it performed, and SHOULD drain at a request chosen at random among the following ones, before the candidate's deadline (§9.3).
- **Usually one drain per key** is due at a time under this policy: a drained input that is not fully consumed comes back as a smaller successor, which is drained again later, so a wallet rarely holds more than one small credential under a key.

### 7.6 Mixing issuer keys

Presenting inputs under two keys reveals that the client was issued credentials under both (§9.1). A client SHOULD present inputs under one key. It MAY mix keys only to drain a residual under the older key before that key stops accepting, when no fill under the older key exists. The fill under the newer key then pays whatever the residual does not; value is consumed from the older key first and never moves to the newer one.

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

**Proposed improvement to ACT (optional, both single and multi-input).** Add a *presentation context*, a byte string the prover chooses and the verifier supplies, absorbed into each spend proof's Fiat–Shamir session (for example `session_id = DS ‖ "spend" ‖ Encode(k) ‖ Encode(ctx) ‖ H(presentation_context)` in the current editor's draft). A multi-input client would set it to a digest of the full ordered input list (each `issuer_key_id` and nullifier, all known before any proof is produced), the request's `challenge_digest`, and a commitment to each refund-recovery key. Every proof would then verify only inside its own request, in its own position, with its own refund keys, which defeats theft, splicing, reordering and refund redirection by anyone without the credentials. This is a change to the ACT specification, and it would also close a gap single spends have today: unlike RFC 9577's token types, whose authenticator covers `challenge_digest`, an ACT Token's `challenge_digest` and `issuer_key_id` are not authenticated by its proof. It is raised as a question for the working group (§13.2), not adopted here.

### 8.5 Double spends and replay

- **Within one request:** repeated nullifiers are refused before anything is recorded (§6.1 step 3); the atomic insert would refuse them anyway.
- **Across requests:** nullifiers are recorded atomically; a request that repeats any recorded nullifier records nothing (§6.2).
- **Against recovery:** recovery of an unrecorded input records it, so a later request containing it is refused whole, and the request's other inputs remain recoverable (§6.5).
- **After a crash mid-request:** the nullifiers are either all recorded or none are, so no input is half-spent. With provisional refunds (§6.7) no remainder is lost; without them the residual is today's, multiplied by `N`.
- **Several refunds of one proof** (provisional, then final; or a recovery that raced the response): the successors share one nullifier, so at most one is ever spendable, and no value is created (§6.7).
- **After a key's deadline:** an input under a key past `accept_until` refuses the request before recording (class 1). A nullifier is never pruned while its key could accept a spend, so a spend cannot be replayed after its nullifier was pruned.

### 8.6 Key rotation and epochs

The epoch rule is that no value extends into a later epoch beyond the existing grace of at most one epoch. A credential under key `K` is spendable until `accept_until(K)`, and every successor of it is a refund under `K`, lapsing at the same instant. Multi-input requests keep this exactly: each input's successor is under that input's own key and holds at most what the input held (§6.3), so no value moves from an older key to a newer one, and no credential is ever minted under a later key from value presented under an earlier one. A request mixing keys consumes the older key's value first when the client orders drains by expiry, which is the only effect mixing has on value. A request is accepted only if every input's key is accepting at the server's check (§6.1 steps 2 and 4), and each refund is signed during that key's retention margin, as for single spends.

### 8.7 Cost and denial of service

Verification is the expensive step, and it runs before anything is recorded, so an unauthenticated sender can make the server verify up to `N_max` invalid proofs per request instead of one. Measured with `anonymous-credit-tokens` 0.4.2 at `L = 128` (release build, one thread, an Apple-silicon laptop; enclave hardware will differ): decoding a proof takes about 0.36 ms, proving about 5.4 ms, and verifying plus signing a refund about 6.0 ms. So `N_max = 4` bounds a bogus request to roughly 25 ms of verification. The bounds:

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

For a client that drains, the second issuance under `K` happens anyway: the drain exists because a credential ran low and another was issued. What multi-input adds is that one request is placed in `A₂(K)`. If, say, 30% of an epoch's accounts are issued twice, a drain request's anonymity set is 30% of a single spend's. In a small deployment, where `A₁(K)` is already small (see [Known gaps → Anonymity-set size](../gaps.md#anonymity-set-size)), this can be decisive, and a deployment SHOULD weigh it before enabling drains.

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

The issuer sees issuance times on the account-linked surface and request times on the anonymous one. A client provisioning credentials on demand already sends its first spend from a new credential shortly after the issuance; that spend is one of many single spends under the key. A drain-and-fill request sent right after an issuance would be much rarer (about one per credential lifetime, which with typical charges far below the worst case spans tens to hundreds of requests), and would sit next to an account's second issuance under the key. That would correlate the account with the request far more sharply. Hence §7.5: no drain in the first request after an issuance, and drains at randomly chosen later requests. Provisioning ahead of need, so that issuance is not tied to a particular request, helps further and is RECOMMENDED where the account balance allows.

### 9.4 Input count and order

**Order.** The canonical order (§7.3) is a function of the keys and shares already visible, plus a random tie-break, so it discloses nothing further. A client using another order would expose its implementation, and whatever its order depends on.

**Count.** `N` discloses how many drains are due, which is a function of the client's wallet state. Under §7, `N` is 1 for most requests, 2 when draining, and rarely 3. Every client that uses its credentials down drains at some point, so `N = 2` is common to all such clients, and because drains are deferred randomly, when one happens is not tied to the client's issuances.

**Padding to a fixed count was considered and rejected.** Padding with zero-share inputs (ACT permits spending zero, which only refreshes a credential) would make every request present `N_max` proofs. That needs `N_max` live credentials per request (most clients hold one or two), multiplies proof size and verification by `N_max` on every request, and every padding input would still disclose its key and that its holder owns at least that many credentials, which is the §9.1 linkage again, so padding trades one fingerprint for another, larger one. For the same reason, servers refuse zero-share inputs: they pay for nothing and are a degree of freedom for fingerprinting. Whether the working group wants fixed or bucketed counts is an open question.

### 9.5 Network observer

Requests are not padded: their size already reveals the prompt's size. Each input adds about 24 KB, so a network observer learns `N` from a request's size, and also from the server's longer pre-flight (about 6 ms per input). That is wallet state, not content, and it is the same `N` the issuer sees. The response's `refunds` add about 330 bytes per input, which can move a padded response into the next size bucket or a streamed response's metadata into one more frame; that reveals the same `N`.

### 9.6 Database observer (model B)

The database sees what the issuer's store holds: one `nullifier` row per input, with its key, and the sealed refunds. Rows recorded in one transaction share a commit, so the database learns each request's `N` and key set, which, joined with the issuance ledger in the same database, gives it the anonymity-set reduction of §9.1. Amounts stay sealed. Batching nullifier inserts from different requests on a fixed interval would blur which rows belong together (the same mechanism proposed for the refund-write timing channel). It is noted here as a possible mitigation, not specified.

### 9.7 Interactions

- **Sealed refund tokens.** Each input carries its own recovery key derived from its own `PreRefund`, and its refund is sealed on its own row (§5.2, §6.5). Nothing new is persisted in the clear, and recovering one input never requires another's key.
- **Fixed public denomination.** With every issuance of `D` and every hold bounded by `D/k`, a fresh credential always covers a hold, so problem 2 of §1 does not arise and multi-input requests serve mainly to drain residuals, recovering what would otherwise be stranded (up to `D/k` per credential). Multi-input requests would also let `D` be lowered, or holds be allowed above `D` (covered by several credentials, rule 4 of §7.2), at the costs of §9.1 and §9.2. Lowering `D` splits the issuance anonymity set by client version and is out of scope.
- **Holds and ceilings.** `W` is computed exactly as today, from the request and the model's declared completion ceiling. The server's check is `H ≥ W`, with `H` the sum of the shares. A server that charges the whole hold when the upstream reports no usage would charge `H`; with canonical shares `H = W`, as for a single spend (§13.1 asks whether to cap it at `W`).
- **Tool rounds and continuations.** Every round of a tool loop takes its own inputs, exactly as it takes its own credential today. A round may drain, subject to §7.5, and a hold may not be abandoned unsettled to take another, for every input.
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

- **Single spends become the one-input case.** The `token` parameter, the single refund-recovery key and the recovery endpoint keep their form (apart from the padding change below). The server's processing of a one-input request is today's.
- **Response shape.** `refund` (an object) becomes `refunds` (an array of one per input) in blocking responses, error bodies and the streamed metadata event; `openapi.json` is regenerated.
- **Encoding.** The `token` value becomes padded base64url (RFC 9577), and the server refuses the unpadded spelling. The current implementation is unpadded.
- **Server store.** No schema change: one `nullifier` row per input, as today. Recording becomes a multi-row transaction. Provisional refunds (§6.7), if adopted, write `refund_token` at insertion.
- **Client store.** In-flight inputs are grouped per request (a group identifier on the pending-refund rows), which is a schema change under the fresh-start rule (bump the local schema version; old development databases are refused).
- **Privacy guarantees.** The new disclosures (§9.1, §9.2) are reflected in `docs/privacy-guarantees.md` §2 and `docs/gaps.md` when this is implemented; proposed wording is an open question below.
- **Privacy Pass interoperability.** A server that does not implement `tokens` refuses a multi-input request without recording anything (§5.1), and the client recovers its inputs. A client can therefore send multi-input requests only to servers it knows implement them; Eidola's client knows its server.

## 12. Test-vector and test plan

**Allocation vectors (normative).** Tables of `(shares in order, C) → (aᵢ, tᵢ)`, including: `N = 1` (today's single spend); `C = 0`; `C = H`; `C` at every prefix boundary (exactly `s₁`, `s₁ + s₂`, …); `C` one either side of each boundary; shares near `2^63` and checked overflow of `H`; and the example of §6.3. Property tests: `Σ aᵢ = C`, `0 ≤ tᵢ ≤ sᵢ`, the prefix structure, and determinism.

**Wire vectors.** Deterministic proofs from a seeded RNG (the crate accepts any `CryptoRngCore`; the ACT specification's appendix parameters and key), producing: the `TokenList` bytes and the `tokens` header value for `N = 2, 3, 4`; the refund-key header; and the expected refunds for given `tᵢ` (refund randomness seeded too). Negative vectors, each refused before anything is recorded: `N = 1` in `tokens`; both `token` and `tokens`; `N > N_max`; an unpadded value; non-zero pad bits; an entry with trailing bytes; a list with trailing bytes; mixed `token_type`; mixed `challenge_digest`; a duplicated nullifier (same credential, two shares); a zero share; a key count that does not match; uppercase hex; whitespace in the key list.

**Server processing, against a real Postgres.** One invalid proof among valid ones records nothing; one already-recorded nullifier rolls back the rest, and recovery then fully refunds each of the others; mixed keys settle each refund under its own key and each successor completes only under its own key; an input under a key past `accept_until` refuses the request; a request accepted just before one input's deadline settles every input after it; recovery of an unrecorded input followed by its request's late arrival records nothing for the request; concurrent requests sharing nullifiers in opposite orders do not deadlock; a crash injected after recording (with and without provisional refunds) leaves each input's remainder recoverable as specified; class 3 and class 4 failures return full refunds for every input.

**Client, in the chat harness.** Every exit point of a multi-input round settles every input; inputs are taken atomically (a crash between proofs leaves none marked spending); canonical order; no drain right after an issuance; drains defer and still happen before a credential's deadline; mixed keys only to drain an expiring residual; a refund that does not verify triggers recovery of that input alone; a response whose partial returns break the prefix structure is surfaced; and the bus exit-point table gains the new rows.

**Limits.** A request with `N_max` Tokens passes the attestation shim and the gateway's HTTP stack, over HTTP/1.1 and HTTP/2.

**Teeth.** As the repository's practice: each property above, mutated (allocation from the end instead of the start, a non-atomic insert, a missing duplicate check, a partial return moved between inputs, a refund signed under the wrong key), makes its tests fail.

## 13. Open questions

### 13.1 For the maintainer

1. **`N_max`.** 4 is proposed (§5.5). Lower bounds the cost and the fingerprint; higher serves rule 4 of §7.2 should holds ever exceed the denomination.
2. **Drains enabled at all, given §9.1 at today's population?** In a small deployment `A₂(K)` may be very small. An alternative is to ship the mechanism with drains limited to expiring residuals.
3. **Mixed keys.** Allow them only for an expiring older-key residual (as proposed), or refuse mixed-key requests on the server altogether, giving up that residual?
4. **Drain parameters.** `θ`, the deadline horizon, `Δ = 1`, and the deferral distribution.
5. **Provisional refunds (§6.7).** Adopt them for single and multi-input requests alike, changing recovery's in-flight answer from `404` to a provisional refund?
6. **Settlement without usage.** Charge the hold `H` (today's rule) or `min(H, W)`?
7. **Encoding.** Padded base64url per RFC 9577 (proposed), or keep the current unpadded spelling for both parameters?
8. **Response shape.** A `refunds` array always (proposed), or keep `refund` for one input?
9. **Smaller `L`.** Every proof is about 18 KB at `L = 128`; `L = 64` would roughly halve it and the verification cost, at a much smaller representable balance.
10. **Privacy-guarantee wording.** §2.2 says redemptions are cryptographically unlinkable. With exact drains, that remains true against everyone outside the enclave but rests on the measured code for drained credentials (§9.2). A proposed addition: *"A request may present several anonymous credentials. Credentials presented together are known to come from one client, which places the request among the accounts issued at least as many credentials under each presented key. A credential drained to zero discloses its balance to the server's measured code, which retains no per-request amount."*

### 13.2 For a working-group discussion

1. **Scope.** Is presenting several ACT Tokens in one HTTP request in scope for the Privacy Pass ACT binding, or a deployment matter?
2. **Encoding.** A new `tokens` parameter on `PrivateToken` (proposed), a list-valued dedicated field, or the request body? How should a multi-input presentation signal itself to an origin that ignores unknown parameters?
3. **Presentation binding in ACT (§8.4).** Should the ACT spend proof absorb a presentation context, binding a Token to its challenge, its request, its position among co-presented Tokens, and its refund channel? That would also authenticate `challenge_digest`, which today no ACT proof covers.
4. **Allocation as origin policy or as specification.** The base specification lets the origin choose `t`. A multi-input binding needs a public, deterministic rule so clients can order inputs for privacy and detect deviations; is in-order consumption the right one to standardize?
5. **Advertising `N_max`.** A challenge attribute, alongside `cost`?
6. **Challenge `cost` with several Tokens.** Should `Σ sᵢ` equal `cost` exactly, or be at least `cost`?
7. **Count fingerprinting.** Fixed or bucketed counts (padding with zero-spend proofs), given §9.4's argument that padding leaks a larger fingerprint than it hides?
8. **Same-key aggregate spends.** Is a proof over several same-key credentials, with hidden shares and one successor, worth specifying in the ACT cryptography document, to remove the drain disclosure?
9. **Refund transport.** For a list of refunds in `PrivacyPass-Reverse`, the encoding, and trailers for streamed responses.

## References

- `draft-schlesinger-cfrg-act-01` and the current editor's draft: *Anonymous Credit Tokens*.
- `draft-schlesinger-privacypass-act-01` and the current editor's draft: *Privacy Pass Issuance Protocol for Anonymous Credit Tokens*.
- `draft-meunier-privacypass-reverse-flow` and `draft-meunier-privacypass-reverse-flow-http`.
- RFC 9576 (*The Privacy Pass Architecture*), RFC 9577 (*The Privacy Pass HTTP Authentication Scheme*), RFC 9578 (*Privacy Pass Issuance Protocols*).
- RFC 9110 (*HTTP Semantics*), RFC 4648 (*Base-N encodings*), RFC 8446 §3 (*presentation language*).
- `anonymous-credit-tokens` 0.4.2, the implementation Eidola uses.
- In this repository: [The server](../server.md) (anonymity set, key epochs), [Privacy guarantees](../privacy-guarantees.md) §2, [Known gaps](../gaps.md), and the gateway's credential code (`crates/eidola-server-gateway/src/credentials.rs`, `handlers.rs`, `auth.rs`).
