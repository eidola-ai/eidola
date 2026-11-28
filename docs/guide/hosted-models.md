# Running hosted models

When you choose a hosted model in Eidola, your prompts run on confidential hardware operated by Eidola — verified on every connection and paid for anonymously. This guide covers how hosted compute works day to day: what you are paying for, how your credits and anonymous wallet work, and how to choose between model tags and pinned weights.

## Private compute, not proprietary AI

When you pay for hosted models in Eidola, you are paying for **private, verifiable compute** to run the same models and software that you could run on your own hardware.

Every hosted model Eidola serves is an open-weight model running on Eidola's open-source inference engine inside hardware-protected enclaves. Much like your electric company, we provide the power for your work without becoming a participant in it. Even if we wanted to, we cannot read your prompts or responses, and we see very little that could be used to profile you.

Finally, you are never locked in: if you ever decide to run the same open-weight model on your own hardware or another server, you can. For the full picture behind this approach, see [How to think about Eidola](../paradigm.md).

## Credits and your wallet

Eidola separates **buying credits** from **spending them**, so your billing history can never be tied to your conversations. You will see two related views in Settings:

- **Settings → Account** shows your credit balance (both subscription credits and one-time top-ups) and lets you add credits or manage your subscription through Stripe.
- **Settings → Wallet** shows the **anonymous credit tokens** your app has already converted from your account balance and is holding ready to spend.

### Automatic background buffering

You almost never need to convert credits manually. While Eidola is open, it quietly converts small portions of your account balance into anonymous credit tokens ahead of time, spaced out at random intervals unrelated to when you chat. When you send a message, your app simply pulls an already-buffered token from your wallet.

If you open Eidola for the first time, return after a quiet week, or run a one-off terminal command with the desktop app closed, your wallet may not have a token ready yet. In that case your app converts a token right before sending your message — it works seamlessly, though that single first spend happens close in time to the conversion (subsequent spends from its change are fully separated again).

To learn more about how tokens, refunds, and weekly key windows work under the hood, see [Anonymous credits](../anonymous-credits.md).

## Choosing a model: tags vs. pinned weights

On many AI services, a model name is just a moving label — the provider can silently swap the underlying model, quantize it to save money, or change its behavior overnight without telling you.

In Eidola, every hosted model is identified by the **cryptographic fingerprint (hash) of its exact weight files**, and the server refuses to answer unless its loaded weights match what your app asked for. That gives you two ways to pick a model:

- **Follow a model tag** (for example, `qwen3.8-27b`) when you want convenient access to the latest published version under that name. Unlike other platforms, Eidola **never swaps silently** — if a tag you follow moves to new weights, the app calls out the change.
- **Pin exact weights** (using the model's `sha256:...` fingerprint) when consistency matters — for ongoing research, careful writing, or any workflow where a surprise change in model behavior would get in your way. Eidola guarantees your requests only ever run on those exact weights, and notifies you if newer weights are available under the tag or if your pinned weights are scheduled to retire.

You can inspect any model's exact weights hash and upstream source right in the app (or via `eidola model show` in the terminal). To learn more, see [Model identity](../model-identity.md).
