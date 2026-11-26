# How to think about Eidola

Today, most AI products demand enormous trust: when you send your thoughts to someone else's server, the software and models behind it can change at any moment without telling you, and your only assurance of privacy is a company's policy page.

Eidola takes a different approach — one that used to be ordinary. When software came on a CD and ran on your own computer, its behavior couldn't silently change, and your files, drafts, and habits stayed yours. That wasn't because the software vendor was unusually virtuous; it was because the structure of the technology made anything else impossible.

Modern AI models are often too large for a laptop or phone, so running them means reaching a data center. **Eidola is designed to extend the sovereignty of a local tool to remote compute** — so that using a hosted machine doesn't mean surrendering control of your work, your privacy, or the software you rely on.

## Compute, not proprietary AI

Eidola's paid service sells **private compute**, not proprietary AI.

Most AI products sell "intelligence as a service" — an arrangement where another company sits inside your conversations as a party to your work. Eidola provides "compute as a service," operating more like a utility supplying electricity or fuel: it delivers the raw power your AI workloads need, with effectively no visibility into what you do with it.

The models you run through Eidola are open-weight models — publicly released files that anyone can download and inspect. The engine that runs them is open source too. If you own or rent the hardware, you can run the exact same models on the exact same engine without involving Eidola at all. And inside the Eidola app, you can run smaller open models directly on your own computer, or point the app at any server you already control, with no Eidola account required.

What our paid service does is make large models practical and affordable for ordinary people who don't have a rack of GPUs or the time to operate one — without asking them to trade away privacy to get there.

## Three principles

Everything in these docs follows from three design principles.

### 1. Your device is sovereign

Eidola is an installed application on hardware you control. Your conversations, drafts, settings, and wallet live locally on your device — never in a cloud account we can browse or take away.

Every trust decision happens on your device, against values compiled into the app before it shipped:

- **One client pairs with one exact server build.** Your app doesn't just check a domain name or a TLS certificate; on every new connection, it verifies a hardware proof that the server is running the exact measured build paired with the release installed on your machine — and that server, in turn, verifies the exact inference engines it talks to.
- **Updates require you.** Neither the client nor the server changes underneath you. A new server build reaches you only when you review and install a new client release that pins it.
- **Failure is refusal.** If anything in the verification chain cannot be confirmed, the app refuses to connect. There is no degraded mode and no "proceed anyway" button.

### 2. Structure over promises

Privacy claims are easy to write and hard to audit. In Eidola, the guarantees that matter are properties of the *running code* — enforced by cryptography, hardware isolation, and reproducible builds rather than by operator goodwill:

- **Public source and reproducible builds.** Anyone can read the code, rebuild the released binaries from source, and verify that they get the same bytes.
- **Verified confidential compute.** Both the gateway and the inference engines run inside hardware-attested confidential enclaves, with TLS terminated inside each enclave. No hosting provider, network observer, or Eidola operator can read your prompts in transit or in memory, and nothing writes them to storage or logs.
- **Payment separated from use.** Buying credits touches a payment processor; spending them uses anonymous cryptographic tokens that cannot be linked back to your account or to each other — even with full access to our database.
- **Accountable human attestations.** Every release requires signed attestations from named humans — using hardware-held keys and recorded on a public transparency log — affirming that they reproduced the build, reviewed the diff against our [privacy guarantees](privacy-guarantees.md), and are signing freely, without coercion.

No single person can audit every line of code, every cryptographic detail, and every build pipeline on their own — and we don't expect you to. Making the entire chain verifiable means *anyone* you choose to trust — a technical friend, an independent researcher, a security auditor — can inspect it on your behalf, rather than forcing you to take our word for it.

### 3. Limits are stated plainly

No system eliminates trust completely. Every architecture rests on layers it cannot verify from inside itself — from the hardware vendors who manufacture confidential-compute chips, to the network metadata visible to your ISP, to the security of your own device.

Rather than glossing over those edges, we document them explicitly in our [threat model](threat-model.md) and [known gaps](gaps.md). Where a defense is incomplete or depends on ongoing ecosystem work, we say so plainly so you can decide whether Eidola fits your own threat model.

## Where to go next

- **Want to try Eidola?** Start with [Getting started](guide/getting-started.md).
- **Want to see how the pieces fit together?** Read [Architecture](architecture.md) for the interactive system map and end-to-end walkthrough.
- **Want to inspect what we commit to — and what we don't?** Read [Privacy guarantees](privacy-guarantees.md), [Threat model](threat-model.md), and [Known gaps](gaps.md).
