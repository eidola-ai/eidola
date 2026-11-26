# Threat model

No privacy claim is meaningful without stating *who* you are defending against — and *who* you are not. This page maps each adversary against Eidola's [privacy guarantees](privacy-guarantees.md), explaining what stops them, what remains exposed, and where residual trust sits.

## Adversaries we defend against

### A curious or malicious Eidola operator

**Capability.** Operates Eidola's infrastructure and can read its logs, its database, its telemetry, and its network traffic.

**What stops them.** The gateway enforces a structural split between the linked account surface and the unlinked inference surface, using [anonymous credit tokens](anonymous-credits.md) so the operator can see *that* a token was spent, never *which account* funded it. Prompts and answers exist in plaintext only inside hardware-attested enclaves ([Architecture](architecture.md#the-gateway)), responses are shaped on both network hops to hide token sizes and rhythm, and telemetry is aggregated inside the enclaves with no per-request records. Deploying a modified gateway or engine to bypass these rules fails immediately: every client pins one exact gateway measurement, and the gateway pins its engines — so weakening server behavior requires shipping a new, attested client release ([Verifiable releases](releases.md)).

**Residual exposure.** Because the database runs outside the enclave, a live observer of database writes can see when each spend starts and ends — revealing rough request durations and bursts of follow-up turns, never content or amounts ([Database spend timing](gaps.md#database-spend-timing)).

### A passive network observer

**Capability.** Watches all encrypted packets between your device and Eidola's servers (your ISP, Wi-Fi operator, or transit carriers).

**What stops them.** TLS terminated directly inside the gateway enclave protects message confidentiality and integrity. Response shaping — fixed-size frames on a fixed clock tick for streamed answers, and size-bucketed non-streaming responses — hides individual token sizes and timing ([Response shaping](architecture.md#response-shaping)).

**Residual exposure.** Network metadata remains visible: your IP address, when connections happen, the size of each outgoing request (requests are not padded yet), and how long each response runs. Users who need network-layer anonymity should connect through Tor or a trusted VPN with a large userbase ([Network and metadata](gaps.md#network-and-metadata)).

### An active network attacker (MITM)

**Capability.** Intercepts, modifies, or redirects traffic between your device and Eidola — even using a fraudulently issued WebPKI TLS certificate for our domain.

**What stops them.** On every new TLS connection, your app sends a fresh random nonce and verifies a hardware attestation report over that same connection binding both the nonce and the server's TLS public key to the exact enclave measurement compiled into the app ([Fail-safe verification](architecture.md#fail-safe-verification)). A forged certificate without a matching hardware attestation — or a replayed attestation from an earlier connection — is rejected before any prompt or credential is sent.

**Residual exposure.** Because the attestation binds the enclave's long-lived TLS key rather than each individual TLS session, an attacker who somehow exfiltrated the private key from inside the enclave could terminate TLS outside it and relay fresh attestations ([TLS channel binding](gaps.md#tls-channel-binding)).

### A rogue deployer or server impersonator

**Capability.** Either an insider with deployment access to Eidola's hosting platform attempting to swap binaries, alter configuration, point the gateway at an unverified model server, or roll back to an older build — or an external attacker hijacking DNS/BGP to impersonate Eidola from outside our infrastructure.

**What stops them.** Every enclave's launch identity covers both its boot image and the hash of its committed deployment configuration. Any change to the running image, command arguments, or measured environment produces a different launch identity that the client refuses. Because each client release pins one exact gateway measurement — and that gateway pins its inference engines and model weights — rollback to a genuinely attested *older* build also fails. Database engine-placement records control availability only, never trust ([Pinning and attestation](architecture.md#pinning-and-attestation)).

**Residual exposure.** Runtime secrets injected for external operator services (database, Stripe, telemetry destination) are not part of the enclave measurement; substituting them redirects traffic only to services already treated as untrusted, while measured code still fixes everything sent to them. Platform boot images are pinned by hash rather than built from source by Eidola ([Non-source-bootstrapped build inputs](gaps.md#non-source-bootstrapped-build-inputs)).

### A compromised release pipeline

**Capability.** Gains write access to our GitHub repository, CI runners, or the OIDC identity that signs release manifests.

**What stops them.** Your app never accepts a release signed only by CI. Every update also requires [human attestations](releases.md#human-attestations) signed from hardware-held keys (requiring physical presence) whose fingerprints — along with the minimum signer count and exact claim templates — are compiled into the currently installed app. Controlling CI cannot forge a hardware-bound human signature.

**Residual exposure.** Simultaneously compromising every pinned attestant's hardware key defeats this. Today the required threshold is one attestant while the verifier supports multiple independent signers ([Single attestant and jurisdiction diversity](gaps.md#single-attestant-and-jurisdiction-diversity)).

### A tampered build toolchain or binary

**Capability.** Subverts a prebuilt compiler, container base image, or binary package — or silently swaps an upstream dependency after review — to slip malicious code into Eidola's binaries without touching public source.

**What stops them.** Every dependency is pinned by exact version and cryptographic hash, preventing silent substitution. Server, CLI, and mobile-bundle builds use a source-bootstrapped reproducible toolchain (StageX), desktop builds use hermetic, reproducible Nix inputs, and GPU kernels are compiled ahead of time and checked against a committed hash manifest — narrowing the risk from opaque, arbitrarily produced binaries to public, widely scrutinized source code. CI enforces bit-for-bit manifest reproducibility before any release reaches `main`, and release attestants independently reproduce the manifest and inspect the full diff before signing, catching tampering local to any single build machine.

**Residual exposure.** Source bootstrapping and hash pinning do not stop a vulnerability or backdoor introduced directly into public upstream source — whether a library Eidola depends on or a compiler in the bootstrap chain — if it escapes community scrutiny and our review. Several build inputs also remain opaque binaries rather than source-bootstrapped, including Apple's SDK, NVIDIA's CUDA compiler, and binary packages in the platform boot image ([Non-source-bootstrapped build inputs](gaps.md#non-source-bootstrapped-build-inputs)).

### A legally compelled engineer or officer

**Capability.** A court order, technical capability notice, or secret directive attempts to force an Eidola engineer or officer to introduce a backdoor or weaken privacy guarantees.

**What stops them.** Every release attestation requires a positive, word-for-word statement signed under the attestant's legal identity affirming that they are *not* subject to legal compulsion or a gag order weakening the release ([What an attestant affirms](releases.md#what-an-attestant-affirms)). A compelled, gagged signer must either refuse to sign — halting releases and creating a public signal — or commit an explicit falsehood on a public transparency log. As more attestants across jurisdictions are pinned, an adversary must secretly compel multiple independent people at once.

**Residual exposure.** Coordinated secret compulsion of *every* pinned attestant defeats this silently — a heightened risk while only one attestant is required ([Single attestant and jurisdiction diversity](gaps.md#single-attestant-and-jurisdiction-diversity)).

## Adversaries we do not defend against

### A compromised local device

If your computer or phone is compromised — malware, keyloggers, OS-level surveillance, or another privileged process — your local conversations, account secret, and wallet are exposed just like any other file on your device ([Privacy guarantees §8.1](privacy-guarantees.md#8-bounded-claims-what-this-document-does-not-promise)). On iOS, this trusted local boundary includes the native shell around the mobile app, which cannot be verified against public source ([Verifying on Android vs. iOS](apps.md#verifying-on-android-vs-ios)).

### Physical access to server hardware

Current generations of confidential-computing CPUs from AMD and Intel explicitly treat physical attacks on the host machine — such as interposing on the memory bus ([tee.fail](https://tee.fail/)) — as out of scope. Hardware attestation proves what code is running on genuine silicon, not who has physical access to the server rack. This is an important real-world limitation that we discuss in [Physical access to server hardware](gaps.md#physical-access-to-server-hardware).

### Hardware-vendor forgery

If AMD, Intel, or NVIDIA issued fraudulent attestation certificates — vouching for an enclave that does not actually provide confidential execution — that layer of the verification chain would break. Open hardware roots of trust are the long-term industry direction, not something Eidola can build alone ([Trust in confidential-compute vendors](gaps.md#trust-in-confidential-compute-vendors)).

### Microarchitectural side channels

Confidential-compute hardware isolates enclave memory from the host OS, but does not guarantee immunity from microarchitectural side channels (speculative execution, cache timing, power analysis, or fault injection). Attestation enforces minimum firmware floors to require known patches, and engine hosts may run with simultaneous multithreading (SMT) enabled, which admits cross-thread side channels. Eidola does not promise that no new hardware vulnerabilities will be discovered.

### Traffic analysis

Beyond hiding individual token sizes and timing via response shaping, Eidola does not defend against network traffic analysis. Connection timing, IP addresses, request sizes, and total response durations remain visible to network observers; users needing metadata anonymity should layer Eidola over Tor or a trusted VPN with a large userbase ([Network and metadata](gaps.md#network-and-metadata)).

### Voluntary user disclosure

If you paste identifying details into a prompt, copy model outputs into another service, or share screenshots of your chats, you have stepped outside Eidola's privacy boundary by your own action.

## Residual trust

Every system rests on layers it cannot verify from inside itself. Here is every anchor Eidola still relies on, so you can decide whether it fits your threat model:

| Trust anchor | What we rely on it for | What limits the exposure |
| --- | --- | --- |
| **Confidential-compute vendors** (AMD, Intel, NVIDIA) | Genuine hardware isolation and honest attestation chains | Inference depends on the enclave CPU and GPU vendors. Long-term mitigation is open hardware roots ([Trust in confidential-compute vendors](gaps.md#trust-in-confidential-compute-vendors)). |
| **Physical data-center security & hosting platform** (Tinfoil) | Physical protection against memory-bus attacks, platform boot images, and signed host endorsements | TDX engine hosts are pinned to signed endorsements. Physical-machine transparency is an active gap ([Physical access to server hardware](gaps.md#physical-access-to-server-hardware)). |
| **WebPKI** (Let's Encrypt) | Standard TLS certificate issuance for our domain | Every connection also requires a fresh hardware attestation bound to the certificate's public key — a forged WebPKI certificate alone fails. |
| **Sigstore Rekor** (Linux Foundation) | Public, append-only logging of release signatures | Clients verify Rekor inclusion proofs offline against pinned public keys; Rekor is externally mirrored. Witnessed checkpoint checks remain a gap ([Transparency-log hardening](gaps.md#transparency-log-hardening)). |
| **GitHub OIDC** | Tying CI manifest signatures to our release workflow | Defense in depth only — human attestations from hardware-held keys are required on every release. |
| **Your first installed app** | Starting from an honest binary with an intact compiled-in trust root | Public reproducible builds and platform verification ([Apps & platform verification](apps.md)), plus release continuity on subsequent updates. |
| **Mobile native shell (iOS)** | Serving the verified bundle and storing local files honestly | On Android, the shell reproduces from source. On iOS, Apple encrypts App Store binaries so the shell must be trusted alongside iOS itself ([Verifying on Android vs. iOS](apps.md#verifying-on-android-vs-ios)). |
| **Your hardware and OS** | Local process isolation, storage, and input privacy | Outside Eidola's control ([A compromised local device](#a-compromised-local-device)). |

Where mitigations are still in progress or deferred, they are tracked in [Known gaps](gaps.md).
