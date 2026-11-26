# Verifiable releases

Even a perfectly audited app today can be undone tomorrow by a silent update. If a company's build server is breached, if a malicious or coerced engineer slips a backdoor into the next version, or if the company's commitments change under new leadership or priorities — an ordinary auto-updater will quietly deliver compromised software to every user overnight.

In Eidola, a **release** is the complete unit of trust: every client app, the paired gateway, the inference engines that gateway accepts, the reproducible build hashes, and the cryptographic signatures vouching for them. Nothing updates silently, and no single compromised machine or account can authorize a release on its own.

Each layer below solves a problem that the layer before it leaves open.

## Reproducible builds and the artifact manifest

**The problem.** Public source code only proves what a developer published in a repository — it does not prove that the binary you downloaded was actually compiled from that code. A compromised build server could slip surveillance code into a release binary while leaving the public repository untouched.

**How Eidola solves it.** Every release is built reproducibly from one public git commit so anyone can rebuild the release artifacts and arrive at the exact same cryptographic hashes. Those hashes are recorded in a single file — `artifact-manifest.json` — committed alongside the source and re-derived in CI on every release.

Crucially, **everything in the manifest is a deterministic function of the repository's committed inputs**: it records hashes for the server and engine container images, the desktop packages, the static CLI binaries, the Android APK, the mobile app bundle, and the paired gateway's enclave measurement. Platform signatures that depend on private keys (such as Apple's macOS notarization or Android APK signing) are structurally barred from ever entering the manifest, so anyone can reproduce it without holding a single secret key.

**What this still leaves open.** Reproducibility proves *what binaries a given commit produces*, not *which commit you should trust* — and it does not stop an attacker from secretly building a backdoored branch for a single targeted user.

## Transparency logs and CI provenance

**The problem.** Even with reproducible builds, a compromised operator could privately compile a modified release from a hidden branch and serve it only to one victim, while everyone else receives the clean release. How do you tie a signed release back to the public repository and leave a permanent public record of everything that was ever signed?

**How Eidola solves it.** Eidola uses **Sigstore** and its public, append-only transparency log (**Rekor**), operated independently by the Linux Foundation and mirrored and checkpointed by outside witnesses (independence we would not get by hosting a log ourselves):

- **CI provenance as defense in depth.** When a release tag is cut, our automated GitHub release workflow signs `artifact-manifest.json` using Sigstore's keyless OIDC identity and logs it to Rekor — adding a signal that the manifest came from the official workflow running on that tag. Because some release steps use self-hosted runners, CI provenance is a relatively weak signal on its own, much like WebPKI beneath hardware attestation — useful as an extra layer, never sufficient by itself.
- **An independent public record.** Both CI's manifest signature and every human attestation (below) are recorded in Rekor, producing tamper-evident proof that a given release was signed.

**What this still leaves open.** First, anyone who compromises GitHub, the workflow, or a build runner can forge a CI signature — which is why your app never accepts a release signed only by CI. Second, the public Rekor log has a practical limit today: because it does not make it trivial to query *"show every release Eidola has ever signed,"* it does not yet provide a live canary that automatically alerts the world if a hidden release appears. Instead, it serves primarily as **post-hoc evidence**: if a targeted, compromised release ever surfaces, its logged signature is permanent, undeniable proof.

## Human attestations

**The problem.** Automated build pipelines can be breached, and build software has neither judgment nor legal accountability. Even a genuinely reproducible build from a public commit could still contain a subtle backdoor pushed by a compromised maintainer, a coerced engineer, or a company whose priorities shifted.

**How Eidola solves it.** To become trustable, every release must also carry signed **human attestations** from named officers, created using hardware-held keys (such as a YubiKey requiring a PIN and physical touch for every signature) and logged to Rekor. Where CI proves *which workflow ran*, a human attestation puts a real person's name, review, and legal accountability behind the release.

### What an attestant affirms

Each human attestation is a structured document containing specific, positive claims rendered from [pinned templates](https://github.com/eidola-ai/eidola/blob/main/releases/schema/attestation-templates.json). When your app verifies an update, it re-renders every claim from its own compiled-in templates and rejects any attestation that differs by a single character. The claims cover three areas:

- **Build and code integrity.** The attestant personally reproduced `artifact-manifest.json` from the release commit on hardware under their exclusive physical control, confirmed that platform-signed installables (like the notarized macOS app) reconstruct cleanly from the reproducible unsigned build plus detached signatures, and personally inspected the full source diff since the previous release.
- **Privacy commitments.** Based on that diff review, the release delivers the commitments in [Privacy guarantees](privacy-guarantees.md), does not weaken, narrow, or remove any guarantee that was in effect in the previous release, and contains no known backdoor, covert surveillance mechanism, or undisclosed data path.
- **Freedom from compulsion.** The attestant is signing freely and of their own volition with a hardware key under their exclusive physical control — not under coercion, a gag order, or legal compulsion to weaken the release or misrepresent its properties.

If a release ever needed to weaken a guarantee, an honest attestant could not sign truthfully; verification would break loudly and surface the discrepancy rather than slipping through silently.

**What a human attestation does not promise.** A signed attestation is not a guarantee that a release is free of security bugs, that every consequence of a change was anticipated, or that the signer audited every upstream dependency and compiler down to machine code — no human reviewer can do that. Mistakes, oversights, and vulnerabilities will happen, which is precisely why Eidola's source code and reproducible builds are public. Human attestation serves a narrower, still vital purpose: holding named people accountable for their own intent, review, and claims, and making silent release-pipeline compromise vastly harder.

**What this still leaves open.** Who decides *which* humans are authorized, *how many* must sign, and *what exact claims* they must affirm? If the incoming update could supply those rules, a single attacker could ship an update that lowered the bar to one forged key and rewrote the claims.

## The compiled-in trust root

**The problem.** Any verification rule fetched over the network — or carried inside an incoming update — can be rewritten by the very attacker trying to trick your app.

**How Eidola solves it.** Every Eidola client — desktop, CLI, and the mobile app's WebAssembly bundle — compiles a fixed **trust root** into itself at build time. While advanced users can explicitly override which server endpoint, enclave measurement, or hardware certificates the client connects to (for development or self-hosting), nothing — neither a server response nor any runtime setting — can alter the rules that govern release verification. That compiled-in trust root pins:

- **The paired gateway identity.** Exactly one enclave measurement for the gateway build that belongs to this release (which in turn pins the inference engines it accepts; see [Architecture](architecture.md#fail-safe-verification)).
- **Authorized human attestants and threshold.** The public-key fingerprints of every person authorized to attest releases, plus the minimum number of independent human attestations required. Because that minimum is pinned in the *currently installed* app — never read from the incoming release — a single coerced signer cannot lower the bar by shipping a release that asks for fewer signatures. *(Today the threshold is 1 while the verifier already supports M-of-N across multiple attestants; see [Known gaps](gaps.md#single-attestant-and-jurisdiction-diversity).)*
- **The authorized CI workflow identity** and **Sigstore transparency-log roots**, so neither a forged CI job nor a spoofed transparency log can pass verification.
- **Exact claim templates and strict schema versions.** Every release document carries an integer schema version with no "backwards-compatible minor bump" tolerance. If a future release introduces fields or a schema version your installed app does not know how to enforce, the app refuses to accept it silently — showing you plainly that the release's claims differ from what it expects.

## How your app verifies an update

**The problem.** Even with signed manifests, human attestations, and a compiled-in trust root, an attacker controlling your network or the release mirror could still try to serve a genuine *older* release with a known vulnerability (a downgrade or rollback attack), or skip past an intermediate release that rotated keys or claim templates.

**How Eidola solves it.** Whenever Eidola checks for updates — and again before installing one — the entire verification runs locally on your device in five steps:

1. **Step one release at a time.** Rather than jumping straight to the latest version, updates step through releases in sequence so every transition (including key or schema rotations) is verified by a client holding the exact pins it was signed under. The discovery listing (`release.json`) is unsigned and serves purely as a map of URLs — it carries no hashes and no trust policy.
2. **Verify CI's manifest signature.** Your app downloads `artifact-manifest.json` and checks CI's Sigstore bundle against the compiled-in workflow identity, release tag, and Rekor inclusion proof.
3. **Verify the human attestations.** It checks each attestant's hardware-key signature against the compiled-in fingerprints, verifies each entry's inclusion in Rekor, confirms the required minimum number of independent signers, and re-renders every claim word for word against its pinned templates.
4. **Check continuity.** Once an installed commit is recorded, your app verifies that the incoming release explicitly names that exact commit as its predecessor — blocking both stale-release substitution and rollback to an older, vulnerable version. *(A fresh install has no prior recorded commit to anchor against; see [Known gaps](gaps.md#first-install-downgrade).)*
5. **Present the verified claims to you.** Only after every check passes does your app show you the verified attestation summary and offer the update. If any step fails — including a release signed by CI with no human attestation — the app refuses it as unverifiable, with no override prompt.

How those verified release hashes map onto the actual file or app running on your specific operating system — macOS, Linux, Android, or iOS — is covered in [Apps & platform verification](apps.md).
