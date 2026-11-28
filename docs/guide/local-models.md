# Local & external models

You do not need to send your prompts to Eidola's servers — or to any server at all. Eidola can run open-weight models directly on your own computer, connect to models you already host elsewhere, or combine local and hosted models side by side.

Everything is managed from **Settings → Backends**, organized into three tabs: **Eidola**, **Local**, and **External**.

## Running models on your own device

Eidola bundles a built-in local inference runner (`llama.cpp`), so once you download a model file your prompts never leave your computer — even if you unplug from the internet.

### Downloading and using local models

1. Open **Settings → Backends → Local** and make sure the **Local** backend is enabled.
2. Choose a model from the built-in curated catalog — such as the **Gemma 4** family, ranging from compact models that run comfortably on everyday laptops to larger models for machines with plenty of memory — **or** paste a direct download link to any compatible `.gguf` model file (for example, from Hugging Face).
3. Start the download. Downloads are managed in the background by Eidola itself, so closing the Settings window will not interrupt them.

Once a model finishes downloading, it appears in your model picker alongside any other enabled backends. When you send a message to a local model, Eidola loads it into memory automatically (the first reply takes a few extra seconds while the model loads). You can also load, unload, or delete models anytime from **Settings → Backends → Local** to free up memory or disk space.

> [!TIP]  
> **Picking the right size for your computer:** Larger models generally write better answers, but they also require more memory (RAM or unified GPU memory) and run slower. If your computer feels sluggish or responses crawl, try a smaller model from the catalog.

### Fully offline, account-free mode

If you want Eidola to work strictly on your own device with no account and no connection to Eidola's hosted service:

1. Open **Settings → Backends → Eidola**.
2. Toggle the **Eidola** backend **off**.

When the Eidola backend is disabled, the **Account** and **Wallet** panes disappear from Settings, and Eidola will never contact Eidola's inference servers. You can turn it back on anytime if you later decide to try hosted models.

## Connecting your own server

If you already run models on a desktop workstation, a home server, a local tool like Ollama or LM Studio, or another provider, open **Settings → Backends → External** to connect them. Eidola supports two kinds of external backends:

- **OpenAI-compatible server:** Works with Ollama, vLLM, LM Studio, `llama-server`, or any self-hosted or third-party service that speaks the standard OpenAI chat API. Enter the server's base URL, add an API key if the server requires one, and choose or pin the models you want to use.
- **System `llama.cpp`:** Point Eidola at a folder of `.gguf` model files you already keep on your machine (and optionally at your own `llama-server` binary). Eidola scans the folder and lets you load and unload those models — with an optional **auto-start** toggle so prompts can start the model automatically — while never downloading into or deleting from your directory.

## What changes across backends

All three options work in the same Eidola interface, so it helps to keep the privacy and billing differences clear:

|  | Eidola hosted | On-device (Local) | External server |
| --- | --- | --- | --- |
| **Where prompts go** | Eidola's confidential enclaves | Never leave your device | Wherever that server runs |
| **Hardware attestation** | Verified on every connection | Not needed (your own hardware) | None — outside Eidola's trust chain |
| **Cost** | Prepaid anonymous credits | Free (uses your hardware) | Set by whoever runs the server |
| **Account needed** | Anonymous random-ID account | None | Depends on the server |
| **Best for** | Large, capable models without needing big local hardware | Offline work, zero marginal cost, complete local control | Existing Ollama/vLLM setups or custom servers |

> [!NOTE]  
> Eidola's [privacy guarantees](../privacy-guarantees.md) apply to Eidola's attested hosted service and local app behavior. When you send prompts to an external server you configured, privacy depends on who operates that server and how you connect to it.
