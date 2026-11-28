# Getting started

Eidola is a private AI app that lives on your device. You can use it to run open-weight models locally on your own computer or on Eidola's confidential compute, or connect it to a compatible server you already run — and you can switch or combine them whenever you like.

This guide walks you through installing the app, deciding whether you need an account, sending your first message, and keeping Eidola up to date.

## Installing Eidola

- **macOS:** Download the signed `.zip` archive from the [GitHub Releases](https://github.com/eidola-ai/eidola/releases) page, unzip it, and drag **Eidola.app** into your Applications folder. The desktop bundle also includes the `eidola` command-line tool.
- **Linux:** Install the `.deb` package on Debian or Ubuntu, install through Nix, or download a standalone static CLI binary (`amd64` or `arm64`) from [GitHub Releases](https://github.com/eidola-ai/eidola/releases).
- **iOS and Android:** Install Eidola from the Apple App Store or Google Play (Android users can also download the signed APK directly from GitHub Releases).

If you want to verify that the copy on your device matches Eidola's public source code before using it, see [Apps & platform verification](../apps.md).

## Account or no account

You do **not** need an account to use Eidola.

- **Local or external models only:** If you plan to run models directly on your computer or connect to your own server, you can skip account creation completely. During setup — or anytime later under **Settings → Backends → Eidola** — toggle the Eidola backend off to run entirely on-device without contacting Eidola's servers. See [Local & external models](local-models.md).
- **Hosted private compute:** If you want to use larger open-weight models running on Eidola's confidential servers, create an account during setup (or later in **Settings → Account**).

When you create an account, Eidola creates a random **account ID** and **account secret**.

> [!IMPORTANT]
> **Save your account ID and account secret in a password manager right away.** Because Eidola has no email address or personal details on file for you, there is no "forgot password" link.

You can enter the same account ID and secret on a second device to share your credit balance across devices, but your conversations themselves stay on the device where you had them.

## Adding credits

Hosted models run on a simple prepaid credit system — either a monthly subscription or one-time credit top-ups — managed in **Settings → Account**. Payments are processed through Stripe.

**Paying through Stripe does not connect your billing details to your conversations.** Behind the scenes, your app automatically converts small chunks of your account balance into **anonymous credit tokens** before you chat, so the servers that run your prompts never know which account paid for them. You do not need to manage these tokens by hand; if you are curious how the unlinkability works, read [Anonymous credits](../anonymous-credits.md).

## Your first conversation

Open a new conversation and send a message.

When you send a prompt to a hosted model, Eidola quietly runs a few checks before your message ever leaves your device:

1. **It checks the server's hardware proof.** On every new connection, your app verifies cryptographic proof from the server's processor that it is running inside a genuine confidential enclave with the exact code pinned into your copy of Eidola. If anything looks wrong, the connection is refused rather than silently falling back.
2. **It pays anonymously.** Your app attaches an anonymous credit token from your local wallet — never your account ID or payment details.
3. **It streams the answer straight to your device.** Once your reply finishes, Eidola's servers keep no copy of your prompt or the response.

**Everything personal stays on your device:** your conversations, your account secret, your wallet of anonymous tokens, and your local **Record** — a private log on your machine where you can inspect past hardware attestations, requests, and spending anytime you like.

## Keeping Eidola up to date

Because your app checks that Eidola's servers match the exact version pinned inside it, updates matter — and Eidola never installs one silently.

When a new release is available, your app downloads the release's signed manifest and **verifies it locally on your device**: confirming that named humans signed off on the release with physical hardware keys, that the signatures were recorded on a public transparency log, and that the release follows cleanly from the version you already have. Only after every check passes does Eidola show you the verified summary and offer the update.

Our approach to updates and verification is particularly thorough. To learn how this works under the hood, see [Verifiable releases](../releases.md).

## Where to go next

- **[Running hosted models](hosted-models.md)** — How private compute, your anonymous wallet, and model versions work day to day.
- **[Local & external models](local-models.md)** — Running models offline on your own hardware or connecting Ollama, vLLM, or another server.
- **[Command line & other tools](tools-and-cli.md)** — Connecting code editors and scripts through the local proxy, or using `eidola` from your terminal.
