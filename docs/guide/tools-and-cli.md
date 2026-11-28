# Command line & other tools

Eidola is not limited to its own chat window. You can connect coding editors, writing apps, and scripts to Eidola's attested hosted models (and local models) through the **local inference proxy**, or work directly from your terminal with the `eidola` command-line tool.

## The local inference proxy

Many third-party apps — code editors, terminal assistants, browser extensions, and custom scripts — let you plug in an "OpenAI-compatible API" by entering a base URL and an API key.

Eidola's **local inference proxy** gives you exactly that endpoint right on your own computer (`http://127.0.0.1:11437` by default). Requests sent through the proxy go through Eidola's normal privacy pipeline:

- Hosted requests still verify hardware attestation on every connection and pay with your anonymous credit tokens.
- Requests are logged in your local **Record** so you can audit attestations and spending.
- Nothing sent through the proxy clutters your Eidola conversations or Spaces.

### Setting up the proxy

Open **Settings → Proxy** in the desktop app:

1. **Create an API key.** Give the key a memorable label (such as the name of the editor or tool) and generate it. Every key starts with `eid-` and is **shown only once** — Eidola stores only a one-way digest, so copy it into your tool right away. You can create separate keys per tool and revoke any key individually anytime.
2. **Choose which backends to expose.** Every backend starts **unchecked** on purpose, so no external tool can spend your hosted credits until you explicitly check the box for that backend.
3. **Choose on-device model availability (optional).** If you expose your local models, decide whether external tools may use only **currently loaded models**, or **all downloaded models** (which lets a request load a model into memory automatically).
4. **Turn on "Serve requests."** Once active, point your tool at the base URL `http://127.0.0.1:11437/v1` and paste your `eid-...` API key.

### Keep the proxy on loopback (`127.0.0.1`)

By default, the proxy listens only on `127.0.0.1` — meaning only programs running on the **same computer** can reach it.

You can edit the listening address and port in **Settings → Proxy** if you have an advanced setup, but keep one rule in mind: **the local proxy speaks plain HTTP with no TLS encryption**. If you bind it to a network address reachable by other machines, prompts and API keys travel unencrypted across that network. Eidola displays a clear warning whenever a non-loopback address is configured.

## The `eidola` command line

Every Eidola desktop installation bundles the `eidola` command-line tool built from the exact same release, so your CLI and desktop app never drift out of sync.

### Getting `eidola` on your `PATH`

- **macOS:** Open **Settings → General**, find the **Command line** row, and click to install the `eidola` link onto your `PATH`.
- **Linux:** The `.deb` package installs `/usr/bin/eidola` automatically, and a Nix installation provides `eidola` in `bin/`. For servers or terminal-only machines, standalone static Linux binaries (`amd64` and `arm64`) are also published on [GitHub Releases](https://github.com/eidola-ai/eidola/releases).

### Sharing your profile with the desktop app

Eidola's local database allows only one writer at a time, so the CLI adapts automatically:

- **When the desktop app is running,** `eidola` connects to it over a local socket. Both share the exact same wallet, buffered anonymous tokens, loaded local models, and conversation history with no conflicts.
- **When the desktop app is closed,** `eidola` opens your profile directly. If a script launches two commands at once, the second waits politely for the first to finish instead of failing.

A few identity-sensitive setup commands — such as creating or resetting an account, or adding and removing backends — require direct access to the profile and will ask you to quit the desktop app first.

### Common workflows

**Send a quick prompt:**

```sh
eidola chat "Summarize the difference between symmetric and asymmetric encryption."
```

By default, `eidola chat` writes **only the model's answer to standard output**, while loading notices and usage summaries go to standard error — so you can pipe answers cleanly into files or other commands:

```sh
eidola chat -m qwen3.8-27b "Draft three bullet points for our release notes" > notes.md
```

To target a specific backend (for instance, a local model or an external server), append `@<backend>`:

```sh
eidola chat -m gemma-4-e4b@local "Explain this regex in plain English"
```

If an answer stops at its length limit, `eidola` prints a warning with the exact command to continue it seamlessly:

```sh
eidola chat --continue <post-id>
```

**Inspect models and local downloads:**

```sh
# Inspect a hosted model's exact weights hash and upstream files
eidola model show qwen3.8-27b

# List, download, or manage local models
eidola model list
eidola model download <model-or-url>
eidola model load <model>
eidola model unload <model>
```

**Check your account, credits, and anonymous wallet:**

```sh
eidola account                        # Show your account ID
eidola account balances               # Check credit balances
eidola account prices                 # List available plans and top-ups
eidola account checkout <price-id>    # Open Stripe checkout to add credits
eidola wallet credentials list        # Inspect buffered anonymous tokens
eidola wallet credentials recover     # Recover any stranded refunds
```

**Manage backends (including offline mode):**

```sh
eidola backend list
eidola backend disable eidola         # Switch to local-only, no-account mode
eidola backend enable eidola
```

**Check for verified updates:**

```sh
eidola update check
```

## Running Eidola in the background

Keeping Eidola running in the background lets the desktop app stay ready to answer CLI commands, keeps the local inference proxy available to your editors, and allows background buffering of anonymous credit tokens to run smoothly:

- **macOS:** Turn on **Open at login** in **Settings → General**.
- **Linux:** Install and manage Eidola's `systemd` user service (which runs the app windowless in the background):

```sh
eidola service install
eidola service start
eidola service stop
```
