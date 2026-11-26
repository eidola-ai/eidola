# Known gaps

No privacy system is finished on day one. This page catalogues every known gap or deferred defense in Eidola's trust chain — why it matters in practice, what constrains it today, and where we want to take it.

## Hardware and hosting

### Physical access to server hardware

**Why it matters.** Recent research (including attacks such as [tee.fail](https://tee.fail/)) demonstrated that an attacker with physical access to a confidential-computing server can interpose on the memory bus and extract secrets from an enclave. In response, both AMD and Intel explicitly defined physical hardware attacks as outside the threat model of their current confidential-computing implementations. That means hardware attestation alone proves *what code is running on genuine silicon*, not *whether someone with a screwdriver can touch the machine*.

**Where things stand today.** Our answer here is weak today. For inference engines running on Intel TDX, the gateway restricts connections to an allowlist of specific machines endorsed by our hosting provider (Tinfoil), rather than accepting any Intel chip anywhere. For the gateway itself (AMD SEV-SNP), the client checks that the processor is genuine AMD silicon meeting firmware floors, but cannot yet distinguish a server in Tinfoil's data center from an identical AMD machine an attacker physically controls. Even on endorsed machines, physical security still rests on trusting the data center operator.

**Direction.** We are actively working on adding stronger transparency and independent auditability around the physical machines hosting enclaves, and are in close communication with Tinfoil.

### Trust in confidential-compute vendors

**Why it matters.** Every hardware attestation ultimately chains back to signing keys controlled by AMD, Intel, and NVIDIA. If a hardware vendor issued fraudulent attestation chains — vouching for a non-confidential machine as an enclave — that layer of protection would collapse.

**Where things stand today.** WebPKI provides a modest extra layer against an external attacker (who would also need a valid TLS certificate for our domain), but does little against a compromised vendor or insider. We accept hardware-vendor trust as residual today.

**Direction.** Open-source hardware roots of trust (such as OpenTitan) are the long-term way to reduce vendor opacity — an industry-wide shift we track rather than something Eidola can solve alone.

### GPU attestation and host endorsements

**Why it matters.** GPU-hosted models run on confidential VMs paired with confidential-computing GPUs. Ideally, the gateway would independently verify every GPU's cryptographic attestation chain on every connection, and host-machine policies would not depend on a single provider's endorsements.

**Where things stand today.** Three nuances remain at the engine layer:

- **GPUs are appraised at boot inside the measured VM, not re-appraised per handshake.** When a GPU engine boots, its measured boot image runs NVIDIA's local attestation verifier against every attached GPU and refuses to open them unless all pass; our engine then checks before use that confidential-computing mode is active and developer tools are disabled. On each request connection, the gateway verifies the VM's hardware attestation and checks the nonce-labelled GPU evidence count, rather than independently validating NVIDIA's certificate chain and reference measurements itself.
- **Host policies come from Tinfoil's signed endorsements.** For TDX engine deployments, the allowlist of permitted machines and firmware floors is derived from Tinfoil's Sigstore-signed platform endorsements committed into the gateway build (verified offline at build time). Eidola verifies that Tinfoil signed those endorsements, not independently that every machine in the list is sound, and adopting a newer endorsement release requires a reviewed gateway update. Some host hardware flags — including simultaneous multithreading (SMT), which allows cross-thread side channels — are accepted on multi-socket hosts today.
- **Offline revocation windows.** Because attestation checks run offline using vendor-signed certificates and revocation lists carried with each report, an attacker can replay older vendor-signed revocation data until its signed validity window expires, delaying recognition of a brand-new revocation by that window.

**Direction.** Our goal is for the gateway verifier to appraise NVIDIA GPU evidence independently on every handshake, and to extend endorsed-machine pinning to SEV-SNP gateway deployments as Tinfoil's platform releases mature.

### TLS channel binding

**Why it matters.** On every new connection, the hardware attestation binds the enclave's TLS *public key* (together with a fresh random nonce), not the individual TLS *session*. If a bug in the measured boot image ever leaked the enclave's TLS private key, an active network attacker could terminate your TLS connection outside the enclave and relay fresh, nonce-bound attestations fetched from the real enclave — passing every client check.

**Where things stand today.** The TLS private key is generated inside enclave memory by the measured boot image's attestation shim and has no export interface, so exfiltrating it requires exploiting a vulnerability in that pinned, inspectable image rather than any operator action.

**Direction.** Stronger channel binding — committing a per-session TLS exporter value directly into the hardware attestation report — would limit any hypothetical key leak to a single session. This requires upstream changes in Tinfoil's attestation shim.

## Network and metadata

### Passive traffic analysis

**Why it matters.** Even with strong encryption, network observers (ISPs, Wi-Fi operators, transit providers, or the hosting network) can infer a lot from packet sizes and timing — when you are active, roughly how long conversations run, and how much text you send.

**Where things stand today.** Eidola's [response shaping](architecture.md#response-shaping) hides individual token sizes and inter-token rhythm on both network hops by sending fixed-size frames on a fixed clock tick and padding non-streaming responses to size buckets. However, connection patterns, total response duration, and **outgoing request sizes** remain visible. Request sizes are especially revealing: each turn sends the conversation so far, and automatic continuations send only the text just generated — exposing exact byte growth between turns. Users who need metadata protection today can route Eidola through Tor.

**Direction.** Padding outgoing request bodies to fixed size buckets (tracked in issue [#620](https://github.com/eidola-ai/eidola/issues/620)) is the most straightforward step toward closing the request-size leak. Over the longer term, we are considering transport options such as a Tor hidden service or independent oblivious HTTP (oHTTP) relays to obscure broader traffic patterns.

### Network identity

**Why it matters.** Your IP address is itself an identifier. Even if Eidola's servers never log IP addresses, a network observer watching traffic into our infrastructure could see the same IP address authenticate to the linked account surface (or visit Stripe checkout) and shortly afterward make anonymous inference requests — undermining application-layer unlinkability at the network layer.

**Where things stand today.** Eidola never persists or logs client IP addresses in application code, but cannot hide your IP from the internet path between you and our servers. Users who want network-identity separation today should connect through Tor or a reputable, high-traffic VPN.

**Direction.** We are interested in partnering with independent organizations to operate network-terminating relays — such as oHTTP or MASQUE — so that no single party, Eidola included, sees both your network identity and your requests.

## Anonymous credits

### Anonymity-set size

**Why it matters.** An anonymous credit token is unlinkable only *within the crowd of accounts* that converted tokens under the same weekly issuer key ([Key epochs and the public calendar](anonymous-credits.md#key-epochs-and-the-public-calendar)). Early in Eidola's life — or during a quiet week — that crowd may be small. In the extreme case where only one account converted tokens in a given week, an operator could attribute spends by elimination even though the cryptography held. Presenting two tokens in one request narrows the set to accounts that received at least two tokens under those keys (which background buffering mitigates by minting two tokens per key for every active user).

**Where things stand today.** Weekly epochs, a fixed non-rotating protocol context, and a full-week grace period keep partitions as coarse as billing hygiene allows, and two-per-key buffering keeps multi-token spends ordinary. None of those increases a small user population on its own.

**Direction.** The anonymity set grows naturally with adoption; if early traffic is sparse, epoch lengths can be widened in a visible release.

### Database spend timing

**Why it matters.** Even though anonymous spend rows in Postgres carry no account ID, no amount, and no timestamp column, Postgres runs outside the enclave. Anyone watching live database writes — or reading replication logs and backups — can see *when* a spend row is inserted (request start) and updated (request end), revealing how long each paid request took. Back-to-back requests — such as automatic continuations of a long answer, multi-step tool calls, or an uncached retry after a prefix-cache miss — also appear as tight clusters of database writes that can look linked during quiet periods.

**Where things stand today.** Refunds are stored encrypted under a client-supplied key at a fixed length, nothing written names an account or cost, and duration is the same coarse signal a network observer already sees from response length. Still, because we treat the database as adversarial, this is a real gap.

**Direction.** We currently accept this as a tolerable residual risk.

### Just-in-time issuance timing

**Why it matters.** If your app converts account credit into a token and spends that token one second later, anyone watching the gateway or database can pair the linked conversion with the unlinked spend that immediately follows — linking that token's first request (and any subsequent turns sharing its prefix-cache key) back to your account.

**Where things stand today.** While the desktop app is running, [background buffering](anonymous-credits.md#fixed-denominations-and-buffering-ahead) converts $1.00 tokens ahead of need at random 10–50 minute intervals unrelated to your chat activity. However, your app still falls back to just-in-time conversion when nothing held can pay for a request — on a fresh install, after a week of inactivity, from a standalone CLI run without the desktop app open, or when your remaining account balance is under $1.00. Standard $1.00 allocations keep this fallback infrequent, and CLI users can pre-allocate tokens manually with `eidola account allocate`.

**Direction.** A lightweight background buffering option could close this gap for CLI-only users who do not keep the desktop app running, and it is an approach we are considering.

### First sight of a weekly key

**Why it matters.** Your app prevents a malicious gateway or database from singling you out with a private issuer key by pinning the first key it sees for each weekly epoch — including next week's key, published a full week early ([Key epochs and the public calendar](anonymous-credits.md#key-epochs-and-the-public-calendar)). However, a brand-new install (or a device that hasn't checked in for over a week) has no prior record for the current epoch and must accept the first valid key it is shown.

**Where things stand today.** Active apps record every key a week before it starts issuing, forcing an attacker to target a user well in advance and stay consistent across every check. Still, first sight remains unverified against what other users saw.

**Direction.** Making every epoch's issuer key verifiable on first contact — either by publishing each key to an independent public transparency log or by deriving keys deterministically from measured enclave state — would let even a brand-new install confirm it received the same key as everyone else.

### Drained balances visible to measured code

**Why it matters.** Ordinary anonymous spends hide your token's total balance behind a zero-knowledge proof. When your app pays a request using two tokens — draining a small leftover token and filling the rest from a second token ([Draining small leftovers](anonymous-credits.md#draining-small-leftovers)) — the drained token's exact remaining balance is briefly visible to the gateway's measured code for the duration of that request.

**Where things stand today.** Because the gateway never logs, exports, or stores spend amounts or refund amounts outside enclave memory, nothing persists that could link a drained balance back to earlier spends. Privacy for drained tokens therefore relies on the gateway's measured enclave code rather than the cryptography alone.

**Direction.** Rounding drained shares down to a coarse public step would hide exact remainders at the cost of stranding tiny amounts; we have deferred that trade-off until real usage shows whether it is warranted.

### Expiring tokens and mid-spend crashes

**Why it matters.** Two small edge cases can cost a user a limited amount of prepaid credit:

- **Credit left in a token when its weekly key expires.** Because refund change stays under the spent token's original weekly key so expired credits cannot live forever, any leftover change still unspent when a key's two-week lifetime ends is lost ([When a key expires](anonymous-credits.md#when-a-key-expires)).
- **A gateway crash mid-request.** If the gateway crashes between recording a spend's nullifier and writing its final refund, crash recovery returns the rest of your token immediately via the provisional refund, but the single request's upfront hold is lost ([Sealed refunds and crash recovery](anonymous-credits.md#sealed-refunds-and-crash-recovery)).

**Where things stand today.** Small $1.00 allocations, oldest-first spending, two-token draining, and pausing background buffering for idle wallets keep expiring leftovers small. Gateway crashes are rare, and the provisional refund ensures a crash can never wipe out a token's whole balance — only the hold of the one interrupted request.

**Direction.** A cleaner long-term answer would be an unlinkable rollover scheme that migrates expiring change into a new epoch without linking old and new tokens, which we are interested in exploring as the protocol matures.

## Releases and build chain

### Single attestant and jurisdiction diversity

**Why it matters.** Eidola's release-attestation defense against legal compulsion and hardware-key compromise gets dramatically stronger when multiple independent people — holding separate hardware keys in different legal jurisdictions — must all sign off on every release ([Human attestations](releases.md#human-attestations)).

**Where things stand today.** Although the client verifier already supports an arbitrary M-of-N attestant threshold pinned in the prior release, **only one human attestant is required today**. A single person is a single point of legal compulsion or operational risk.

**Direction.** As the trusted circle around Eidola grows, we intend to onboard additional independent attestants holding hardware-backed keys across multiple jurisdictions, then raise the pinned threshold through a normal attested release.

### First-install downgrade

**Why it matters.** Once Eidola has installed an update, it enforces continuity — refusing any future update that doesn't explicitly descend from the exact commit you are running, which blocks stale-release substitution and rollback attacks ([How your app verifies an update](releases.md#how-your-app-verifies-an-update)). A brand-new installation, however, has no prior recorded commit to check against (and no build can embed its own release commit hash without creating a circular dependency). An attacker controlling your download mirror could therefore serve a genuinely signed, internally consistent *older* release to a fresh install.

**Where things stand today.** Stepping through updates on an existing install checks version ordering and continuity once recorded; on desktop, full in-place atomic replacement of a running app is also still being wired on top of the verified staging pipeline. First-install freshness remains outside what the client can prove by itself.

**Direction.** To let even a first-time installer check that a download is genuinely current, we are considering embedding a public *freshness anchor* — such as a recent timestamped artifact from a public append-only system — into each release, alongside completing safe in-place update promotion on desktop.

### Non-source-bootstrapped build inputs

**Why it matters.** Reproducible builds prove that a binary came from specific committed inputs, not that every compiler or prebuilt binary inside those inputs was free of hidden tampering. Fully source-bootstrapped builds — where even the compiler itself is compiled from auditable source — are the gold standard.

**Where things stand today.** Eidola's gateway, static Linux CLI, and mobile WebAssembly bundle are built with a source-bootstrapped toolchain (StageX). Four parts of the stack still rely on hash-pinned, reproducible — but not fully source-bootstrapped — inputs:

- **macOS desktop app:** built hermetically with Nix, relying on Apple's closed SDK/Xcode toolchain (and wrapped with Apple's non-reproducible Developer ID signature, bound via human attestation).
- **Linux desktop app:** built reproducibly with Nix (using the open nixpkgs binary toolchain rather than StageX, because desktop GPU/Wayland integration requires glibc).
- **Inference engine GPU kernels:** the engine binary itself is StageX-built, while its ahead-of-time GPU kernels are compiled with NVIDIA's closed CUDA toolkit and checked against a committed hash manifest on every build.
- **Confidential-VM boot images (`cvmimage`):** built reproducibly with Nix by Tinfoil from pinned inputs that include binary Ubuntu packages, Docker runtime binaries, and NVIDIA's driver and firmware. We independently rebuild every pinned `cvmimage` release byte for byte from Tinfoil's public build definition ([rebuild log](contributing/rebuilding-platform-images.md)), confirming it matches its inputs — not what every binary input contains.

**Direction.** Every non-bootstrapped input is pinned by exact hash at the import boundary so silent substitution is impossible. Moving further toward full source-bootstrapping on macOS, GPUs, and confidential-VM images depends on broader ecosystem progress.

### Transparency-log hardening

**Why it matters.** Sigstore's public transparency log (Rekor) and certificate authority (Fulcio) are independent layers backing CI provenance and human release attestations ([Transparency logs and CI provenance](releases.md#transparency-logs-and-ci-provenance)). Two deeper verification checks are not yet implemented in the client verifier:

- **Fulcio Signed Certificate Timestamps (SCTs):** confirming that a CI signing certificate was logged in a public Certificate Transparency log (defense in depth on top of OIDC identity and chain verification).
- **Rekor witnessed checkpoint signatures:** confirming that a verified Rekor inclusion proof roots to the globally witnessed public log head rather than a hypothetical private fork served only to Eidola.

**Where things stand today.** Both CI signatures and human attestations are verified offline against pinned Rekor public keys and Signed Entry Timestamps, and human hardware-key signatures remain mandatory regardless of Fulcio.

**Direction.** Fulcio SCT verification and witnessed Rekor checkpoint validation are both straightforward layers of defense in depth that we would like to add to the client verifier.
