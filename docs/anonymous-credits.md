# Anonymous credits

Paying for hosted compute privately is a harder problem than it looks. Buying credits touches a payment processor like Stripe, which knows who you are. Running a model requires proving to the server that you have paid enough to cover the work. Yet Eidola's core privacy promise is that nobody — not even Eidola, with full access to our own database — can tell which account paid for a given prompt, or whether two prompts came from the same person.

To bridge that gap, Eidola uses **Anonymous Credit Tokens (ACT)**, an emerging IETF draft specification ([Privacy Pass ACT](https://datatracker.ietf.org/doc/draft-schlesinger-privacypass-act/)). The specification supplies the cryptography, which is essential — but cryptography only protects the small slice of the system it describes. Everything around it decides whether privacy survives contact with a real deployment: how keys rotate, how much money goes into a token and when, what happens when a token has a few cents left over, how a dropped connection recovers its change, and what the database is allowed to remember.

Because ACT is still a draft standard, very little has been written about these surrounding choices — and because Eidola's threat model includes *ourselves* and *our own database*, we have to solve failure modes that many other deployments can safely ignore.

This page walks through those choices in order, starting from how the tokens work and then tackling each real-world problem that breaks a naive implementation.

## How anonymous credit tokens work

### Tokens, blind signing, and spends

Think of an anonymous credit token as a piece of digital cash stored only in your app's local wallet. It holds a credit balance (roughly equivalent to millionths of a dollar, ignoring discounts), a hidden random serial number called a **nullifier**, and the gateway's cryptographic signature vouching for its value.

A token moves through four steps:

1. **Blind issuance.** On the [linked account surface](architecture.md#linked-vs-unlinked-surfaces), your app asks to convert part of your account balance into a token. Your app invents the token's secret serial number locally, wraps it in a cryptographic blindfold, and sends the blinded commitment to the gateway. The gateway debits your account and signs the blindfolded token. It learns *who* converted credit, *when*, and *how much*, but never the serial number inside. Once your app removes the blindfold, it holds a genuinely signed token the gateway cannot recognize.
2. **Zero-knowledge spending.** When you send a prompt on the [unlinked inference surface](architecture.md#linked-vs-unlinked-surfaces), your app presents a **spend proof**. Without revealing the token's total balance or where it came from, the proof convinces the gateway of one statement: *"I hold a valid token signed by your key whose balance is at least this request's charge."* Alongside the proof, your app reveals only the amount it's attempting to spend and the token's serial number (nullifier).
3. **Double-spend prevention.** Because the gateway cannot recognize tokens, it stops you from spending the same token twice by recording every revealed nullifier in its database *before* serving the request. A second attempt with the same nullifier is rejected.
4. **Change as a fresh token.** Spending a token consumes it completely. When your request finishes, the gateway returns a **refund** message carrying any unused portion of the charge. Your app combines that refund with secret state it kept from the spend to construct a brand-new successor token — with a fresh serial number and balance the gateway has never seen — holding whatever balance remained plus the unused change.

### Why price is the only defense

An ordinary AI service can see which account is making every request. If someone abuses it, the service can rate-limit that account, ban it, investigate its history, or issue manual refunds after reading the logs.

Eidola's inference surface has none of those tools by design. The gateway does not know which account sent a request, and cannot tell if two requests came from the same user. There is **no identity to rate-limit, ban, or investigate** on the inference path.

That leaves one defense against abuse: **price**. Every request must be paid for in full, up front, before the engine does work, using a cost fixed solely by the request itself. And because the gateway's code is measured into its enclave and pinned by every client release ([Architecture](architecture.md#fail-safe-verification)), there are no silent hotfixes if a billing edge case goes wrong. Every decision below therefore guards two things at once: your **unlinkability** and the system's **solvency**.

## Keeping tokens indistinguishable

Even with perfect blind signatures, a token is only as anonymous as the crowd it blends into — its **anonymity set**. A careless server or client can shrink that crowd down to one person without ever breaking the math.

### Key epochs and the public calendar

**The problem.** Every token is signed by an issuer key, and every spend names which key signed it. That means a token is only indistinguishable from other tokens issued under the *same* key (and the same fixed protocol context).

If a server — or a compromised database, or a legally compelled operator — wanted to single out one user, it could mint a special key just for that account, rotate keys right before and after that user converted credits, or ask the database to extend an old expired key whose nullifiers had already been deleted.

**How Eidola solves it.** Issuer keys follow a strict, **public weekly calendar** that anyone can verify with simple arithmetic:

- Every epoch is exactly seven days, anchored to a fixed weekly boundary (Thursdays at 00:00 UTC).
- There is **one key per epoch** — never two keys issuing at once, and never a per-user or per-region key.
- Each key issues new tokens during its seven-day epoch, and remains spendable for one additional seven-day grace epoch after issuance ends. That overlap ensures a token minted near the end of a week still has a full week to be spent, and decouples *when* you convert credits from *when* you spend them.
- Before being sent to the database, each issuer private key is encrypted with its exact validity dates authenticated into the ciphertext, so the database cannot silently edit a key's window or resurrect an expired key.
- **Your app enforces the calendar itself.** Every time it fetches the gateway's key list, it checks that every key matches the public calendar to the exact second, that each epoch has at most one key, and that the key matches the one it first saw for that epoch (including the upcoming epoch's key, published a full week in advance). If a gateway ever offers an off-calendar key or a second key for a seen epoch, your app refuses it — existing tokens keep working, and no new credit is converted until the listing is honest again.

**The trade-off.** There is **no emergency key rotation**. Your app cannot tell a replacement key minted for an honest operational emergency from one minted to single you out. If an issuer key ever has to be withdrawn, the honest response is to halt new conversions until the next epoch begins while already-issued tokens keep spending. Two remaining edges are disclosed in [Known gaps](gaps.md): a brand-new install has no prior history to check its [first sight of a weekly key](gaps.md#first-sight-of-a-weekly-key), and during quiet weeks with few active users the [anonymity set itself is smaller](gaps.md#anonymity-set-size).

### Fixed denominations and buffering ahead

**The problem.** Even when everyone shares one weekly key, two easy mistakes can still link an account to a prompt:

- **Amount correlation.** When you convert credits to anonymous tokens, the gateway sees the exact amount. If your app converted $0.374 right before sending a prompt whose upfront hold was $0.374, matching the conversion to the spend would be trivial.
- **Timing correlation.** Conversions happen on the linked account surface; spends happen on the unlinked inference surface. If your app always converted a token one second before spending it, anyone watching the gateway or database could pair the two events by timestamp.

**How Eidola solves it.** Your app breaks both links with two simple rules:

- **One standard denomination.** Automatic conversions always convert the same public amount — **$1.00** (or your entire remaining balance if you have less than $1.00 left). A prompt's cost never decides how much gets converted; it only decides whether a conversion is needed. If a single request's worst-case hold is larger than what the wallet plus one standard denomination can cover, the request is refused rather than minting a custom-sized token.
- **Background buffering ahead of need.** While the desktop app is running, a background loop wakes at random intervals (between 10 and 50 minutes, unrelated to when you chat) and checks whether your wallet has enough spendable credit left for the next day. If it is running low — and you have actually used the app within the past week, so idle accounts don't mint tokens that will expire unused — it converts tokens ahead of time. Under a new weekly key, it converts **two** standard tokens so that spending two tokens together ([Draining small leftovers](#draining-small-leftovers)) looks completely ordinary for every active user, not just heavy ones. Oldest tokens are always spent first, so freshly buffered tokens sit untouched until your normal usage naturally reaches them.

**What remains.** If your wallet cannot cover a request — on a fresh install, after a quiet week, or from a one-shot CLI run without the desktop app open — your app falls back to converting a standard $1.00 token just in time and spending it right away. That first spend after a just-in-time conversion can be paired with its issuance by timing ([Known gaps](gaps.md#just-in-time-issuance-timing)). Subsequent spends from that token's change are unlinkable again.

### Draining small leftovers

**The problem.** Every token starts at $1.00 and shrinks as you use it. Eventually you end up with a token holding, say, $0.04 — too small to cover the upfront hold for your next prompt. If a request can only take one token, that $0.04 is stranded until the key expires. Yet merging small tokens into a fresh token on a special server endpoint would reveal the exact combined amount (a unique, recognizable value) and require minting outside the standard protocol.

**How Eidola solves it.** A single request can present **up to two tokens** at once:

- Your app sorts spendable tokens oldest-key-first, smallest-balance-first. If the first token covers the request's hold, it pays alone.
- If the first token is too small, your app **drains** it completely (offering its entire remaining balance) and **fills** the remainder of the hold from the next token that can cover it.
- The gateway verifies both tokens, records both nullifiers together, and consumes the final charge in order (from the first token first, returning unused change to the second). Each token's refund stays strictly under its own issuer key, so no value ever leaks into a different epoch.

**The trade-off.** Presenting two tokens together tells the gateway they came from the same client (which is why background buffering issues two tokens per key, keeping two-token spends inside the same anonymity set). Draining a token also reveals that token's exact remaining balance to the gateway's measured enclave code for the duration of that request. Because the gateway never logs or stores spend amounts or refund amounts outside enclave memory, nothing persists that could reconstruct a token's history — meaning privacy for drained balances rests on the measured enclave code rather than the cryptography alone ([Known gaps](gaps.md#drained-balances-visible-to-measured-code)).

## Pricing without revealing history

An account-based service can run your prompt first and bill your account after it sees how long the answer was. An anonymous service cannot — there is no account to bill afterward. Every cost has to be paid up front, without leaking anything about your history or anyone else's.

### Upfront holds and the request fee

**The problem.** Before a model runs, nobody knows how many words it will write — or even the exact token count of your prompt until the inference engine tokenizes it. Meanwhile, because the server cannot ban abusive users, any error or refusal that costs the client *nothing* becomes a free denial-of-service weapon: an attacker could replay valid proofs against invalid requests endlessly, forcing the gateway to run expensive zero-knowledge verifications and database writes for free.

**How Eidola solves it.**

- **Shared pricing contract.** Both your app and the gateway compute the worst-case cost of a request using the same deterministic formula over the request bytes and its output limit. Your app presents a hold equal to that exact worst-case cost, so work can never go unpaid and a client cannot choose an arbitrary hold amount that might distinguish it.
- **Settle on actual usage.** Once the engine answers, the gateway charges only for the actual tokens used (clamped to the hold) and returns the rest as change inside the refund token.
- **Fixed request fee.** Every request whose credentials are accepted pays a tiny fixed fee (**$0.001**), even if the request is later refused or fails before the engine produces text (an unknown model, a cache miss, no engine available). That fee makes spam economically costly without ever needing an account identity to rate-limit.

### Declared prefix caching

**The problem.** In a multi-turn conversation, each new turn re-sends the whole conversation so far. Reusing the inference engine's cached computation from the previous turn makes long conversations dramatically faster and cheaper.

But ordinary "automatic cache discounts" break anonymous billing in two ways:

1. **Information leak.** If the gateway discounts your bill whenever the engine happens to find a cache hit, the refund amount reveals whether that prompt prefix was already in memory — leaking state about previous requests (or, in a shared cache, other users' requests).
2. **Broken upfront holds.** A post-hoc discount doesn't lower the upfront hold required to send the request, so long conversations would still hit the single-token hold ceiling.

**How Eidola solves it.** A request's price **never** depends on what an engine happens to have in its cache — only on what your app **declares** up front:

- Because your app tracks its own conversation turns under its rotated [prefix-cache key](architecture.md#what-stays-on-your-device), it knows what the previous turn left cached. On the next turn, it explicitly declares how many leading tokens are already cached and a tight token budget for the new text appended after them.
- Both your app and the gateway price the hold and final bill using that declaration — cached tokens at the lower cached rate, new tokens at the regular prompt rate.
- Before computing anything, the inference engine checks whether both statements are true (the prefix is actually cached under that key, and the remaining prompt fits the declared budget). If either fails — say, because the engine restarted or evicted the entry under memory pressure — the engine refuses the request immediately before doing compute. The gateway refunds everything except the $0.001 request fee.
- Your app settles the refund, clears its memory of what was cached, and automatically resends the request once without a cache declaration. Crucially, **that uncached retry never triggers a new account conversion** — it pays only from tokens already in your wallet, or stops and asks you to retry. Otherwise, someone who could restart an engine could force users with long conversations to convert account credits on cue, creating a timing correlation.

### Output ceilings and automatic continuation

**The problem.** Modern reasoning models can output tens of thousands of tokens in a single response. Holding the model's entire maximum output limit on every prompt would lock up a huge chunk of a $1.00 token even for a two-sentence reply, causing unnecessary multi-token drains and early refusals. Yet letting your app pick a smaller limit based on how much money is left in your wallet would leak your wallet balance to the server.

**How Eidola solves it.** Your app chooses an output ceiling from a small public ladder (starting at 8,192 tokens and doubling up to the model's maximum) computed strictly from the request itself and published model prices — never from your wallet balance or user settings. The rung scales up with your prompt length (and drops a rung if needed so a single standard $1.00 token can cover the hold).

If the model is still writing when it hits that ceiling, your app automatically sends a follow-up continuation request under the same prefix-cache key — declaring the original prompt as cached so it is cheap to resume — and stitches the answer together seamlessly. If a continuation fails or runs out of wallet funds, everything generated so far is kept in your conversation with a button to continue later.

*(Trade-off: a very long answer becomes a short sequence of back-to-back requests, which is visible as sequential database writes and growing request sizes; see [Known gaps](gaps.md#database-spend-timing).)*

## Surviving crashes and dropped connections

Real networks drop packets, laptops go to sleep mid-stream, and servers occasionally restart. With ordinary accounts, retrying is trivial. With anonymous cash, a naive retry either charges you twice or destroys the entire balance left on your token.

### Idempotent issuance

**The problem.** Suppose your app asks the gateway to convert $1.00 from your account into a blind-signed token, and your Wi-Fi drops before the response arrives. Did the gateway debit your account or not? If your app simply sends a fresh request, you could be debited twice. If it gives up, your $1.00 vanishes.

**How Eidola solves it.** Before sending an issuance request, your app saves its blind secrets, the exact request payload, and a random **idempotency key** to disk. On the gateway, debiting your account and storing the signed blind response happen inside one locked database transaction keyed by that idempotency key.

If the response is lost, your app waits a short interval and asks again with the exact same request and idempotency key — never converting a new token while an earlier conversion is unresolved. If the gateway already debited you, it simply hands back the stored blind response without debiting again. This costs zero privacy: issuance already happens on the linked account surface, and the stored response is blind-signed, so keeping it until the key expires reveals nothing about when or where the token is later spent.

### Sealed refunds and crash recovery

**The problem.** Now imagine the much scarier failure: your app spends a $1.00 token on a $0.05 prompt, the gateway records the nullifier (so the old token is permanently spent), and then your connection drops — or the gateway crashes mid-answer — before you receive the $0.95 change token.

Without recovery, a single dropped connection would wipe out the token's *entire* remaining balance, not just the prompt's cost. Yet if the gateway simply saved the plain refund in the database so you could come back and fetch it later, anyone with read access to the database would see the exact refund amount for every spend — leaking how long every prompt and answer was and making it possible to reconstruct the chain of spends.

**How Eidola solves it.** Three mechanisms work together so a dropped connection loses nothing and a gateway crash loses at most the single request's hold:

1. **Client-keyed sealed refunds.** With every spend, your app sends a one-time **refund recovery key** derived from secret state it already holds locally. Inside the enclave, the gateway encrypts the stored refund under that key and never logs or persists the key itself. The database stores only a fixed-length ciphertext that neither PlanetScale nor Eidola can open — even with all server secrets. When your app comes back to recover a lost refund, it presents the original spend proof and recovery key to unlock it.
2. **Provisional refunds written atomically with the nullifier.** When the gateway records a spend's nullifier before serving the request, it writes a *provisional* sealed refund (returning your untouched remainder, minus the hold) in the exact same database write, while tracking the active request in enclave memory. When the request finishes, the final sealed refund (returning the unused part of the hold too) atomically overwrites the provisional one. If your connection drops while the request is still running, recovery tells your app the refund is still provisional so it waits; once the request finishes — or immediately if the gateway crashed and restarted — recovery hands back the final word. Because both provisional and final refunds map to the same successor serial number, a client can never double-spend by redeeming both.
3. **Always-streamed settlement.** Every chat answer is streamed from the engine to the gateway and from the gateway to your app. If you close the app, lose your connection, or hit an engine failure partway through an answer, the gateway cancels the engine immediately and settles your bill on what was actually generated up to that point — never charging the whole hold for ungenerated text, and never letting an anonymous caller generate text for free by disconnecting early.

*(What still cannot be recovered: if the gateway process itself crashes between recording a nullifier and settling the request, the provisional refund becomes final and you recover the rest of your token minus that one request's hold; see [Known gaps](gaps.md#expiring-tokens-and-mid-spend-crashes).)*

### When a key expires

**The problem.** Nullifiers cannot be kept in the database forever, and purchased credits carry their own legal expiry (a billing period for subscriptions, one year for one-time purchases). If change from a spend were signed under *today's* newest key instead of the spent token's key, someone could keep value alive indefinitely by spending one tiny fraction of a cent every week.

**How Eidola solves it.** Every refund is always signed under the **same weekly key** as the token it replaced. Every token therefore lives for at least one full week (and up to two weeks if minted early in its epoch) and stops being accepted at its key's hard deadline. The gateway keeps each key for one extra hour solely to finish in-flight settlements and refund recoveries, then prunes all of that key's nullifiers once nothing under it can ever be spent again.

**The trade-off.** Any unspent change still sitting inside a token when its key expires is lost. Unlinkable rollover — exchanging expiring value for a token under a newer key without linking the two — requires its own privacy design and is not implemented. Small $1.00 allocations, oldest-first spending, two-token draining, and pausing background buffering for inactive accounts keep the amount at risk small ([Known gaps](gaps.md#expiring-tokens-and-mid-spend-crashes)).

## What lives outside the enclave

Even when the gateway runs inside a hardware-attested enclave, it has to coordinate with pieces outside the enclave: the shared database, billing processes, telemetry, and your own device.

### What the database may hold

Our Postgres database runs outside the enclave (hosted on PlanetScale) and is treated as adversarial — we assume its host or anyone holding its credentials could inspect live queries, backups, or replication logs.

Because Postgres is the one place both sides of Eidola touch, the schema keeps them strictly apart:

| Surface | What Postgres stores | What it never stores |
| --- | --- | --- |
| **Account & billing** | Random account ID, hashed secret, Stripe customer reference, terms acceptances, credit ledger (purchases, conversions with amounts and times), queued billing jobs, encrypted issuer keys | Prompts, answers, models used, or which spend came from which conversion |
| **Inference spends** | Issuer key ID, spent nullifier, fixed-length client-encrypted refund | Account ID, cost, refund amount, model, prompt content, or timestamp column |

**What the database can still observe.** Even without a timestamp column, a live database observer (or replication log) sees *when* each spend row is inserted (request start) and updated (request end), and sees when two nullifier rows commit in the same transaction for a two-token spend. That exposes request durations and bursts of follow-up turns, never amounts or content ([Known gaps](gaps.md#database-spend-timing)).

**Separation from billing operations.** Separate billing processes share the database to handle Stripe purchases, refunds, chargebacks, and account closures ([Architecture](architecture.md#payments-and-billing)). They touch only the account side and never read or write anonymous spend records — once account balance has been converted into anonymous tokens, nothing in billing, support, or Stripe can identify, trace, or recall those tokens.

### Your local wallet

Your device is the one place where your account, your anonymous tokens, your in-flight spend recovery keys, and your conversation history are meant to live together. Your app takes spends one at a time inside a local lock so concurrent requests never accidentally spend the same token twice, and persists every pre-issuance and pre-spend recovery record to durable storage *before* a single byte leaves your device.

That local database is yours alone — it is never uploaded or included in telemetry, and its protection is the security of your own device ([Privacy guarantees §8.1](privacy-guarantees.md#8-bounded-claims-what-this-document-does-not-promise)).
