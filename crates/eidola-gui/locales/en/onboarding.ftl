# The onboarding window (src/onboarding/) — the first-run flow, and the one
# surface where a reader agrees to anything.
#
# English is the source locale: every message must exist here, and the build
# script generates one typed accessor per message from this file. A message
# missing from another locale falls back to English at runtime; a message that
# exists only in another locale is a build error.
#
# TRANSLATORS, read this first. The slide bodies are **Markdown**, rendered by
# the app's prose editor: `##` opens a heading, `*word*` is emphasis, `**word**`
# is strong, a line beginning `- ` is a bullet, `[text](url)` is a link, and a
# blank line separates paragraphs. Translate the words; keep the markers and the
# paragraph breaks where they are. There are no `{` or `}` characters in any
# body except the placeables shown — a literal brace would need escaping.

# The two legal documents are named by their **published titles**, which stay
# English in every locale because that is the text the reader is agreeing to and
# the name they will find at the other end of the link. The sentences around
# them are chrome and do localize.
#
# The `-fixed-` prefix is what makes that a rule rather than a hope: a locale
# resource defining a term whose id begins `-fixed-` is a **build error**
# (codegen rule 14). Without it a translation could define its own
# `-terms-of-service`, override the source's through `add_resource_overriding`,
# and rename the document inside the sentence a reader affirms — with nothing
# refusing it.
-fixed-terms-of-service = Terms of Service
-fixed-privacy-policy = Privacy Policy

# The window's OS-level name. It paints nothing (the title bar is transparent)
# but it names the window in the macOS Window menu, the window switcher and
# VoiceOver's window chooser, and labels the accessibility tree's root node —
# so it is part of the window's accessible surface. Keep it short: it is read
# in a list beside every other open window.
onboarding-window-title = Get Started

# -- Slide bodies ------------------------------------------------------------

# "Pause here" — the first thing a reader sees. The point is the contrast, so
# keep the emphasis on *not*.
onboarding-pause-body =
    ## *Pause here*

    Eidola is *not* the same as ChatGPT, Claude or Gemini.

# The CD-era sovereignty analogy. "Delivered on a CD" is doing real work — it
# names a thing a reader already understands about software they own.
onboarding-tool-body =
    ## Eidola is *your* tool

    In years past, an application was delivered to your computer on a CD:

    - Its behavior *couldn't* spontaneously change without your involvement.
    - Your files, plans, usage patterns, and insights were *yours alone*, undiscoverable by any third party.
    - The **structure** of the technology — *not* some company's promises — enforced these properties.

    Eidola approximates this approach as closely as possible, structurally maximizing end-user sovereignty even for workloads that are best run in a data center.

# What the architecture guarantees, and the invitation to check it rather than
# believe it. "Don't blindly trust our claims; verify them" is the sentence the
# whole product rests on — translate it plainly, never softer.
onboarding-control-body =
    ## *Your* control

    You and only you are in control — not us, not the operators who run the hardware:

    - **Only you can read, retain, or profile your interactions.** Your data is decrypted only inside sealed, hardware-attested enclaves that keep nothing, and the side of Eidola that handles payment is cryptographically separated from the side that serves your requests.
    - **Only you can update Eidola — on your device and the server.** Nothing changes until your client verifies a new version and you decide to trust it.

    Don't blindly trust our claims; verify them. If you don't know how to evaluate our code and architecture, **request the opinion of the most technical person you already trust**.

# Models are fallible and the effects are the reader's. This is a disclosure:
# keep it direct, and do not soften "your responsibility".
onboarding-responsibility-body =
    ## *Your* responsibility

    Eidola is a tool that makes it easier for *you* to run AI models without highly-specialized technical skills or expensive hardware. Nobody is looking over your shoulder, so it's critical that you understand:

    - You will be running AI models, which are at their core collections of probabilities. The models that run in Eidola are freely available to download and use with or without Eidola.
    - Even the very best models are fallible. They can be amazingly useful, but do make mistakes and can exhibit unexpected behavior. It is mathematically impossible to evaluate how a large model will behave in every possible scenario.
    - The models themselves have no intrinsic memory or ability to cause external effects; they can only evaluate data and take actions that you make available to them. Eidola makes it easy to understand and configure access, but the results — both good and bad — are ultimately your responsibility.

# The branch point. `$unlinkability` is the documentation URL, supplied by the
# app rather than written here, so no translation can send a reader somewhere
# else. Keep it as the link target exactly as it stands.
onboarding-get-started-body =
    ## Get started

    You'll need some credits to run models.

    Your account is just a random id — buying credit is the only step that touches a payment method, and even then [we are structurally unable to link your requests back to it]({ $unlinkability }).

onboarding-create-account-body =
    ## Create an account

    Please read and understand our { -fixed-terms-of-service } and { -fixed-privacy-policy }.

onboarding-new-account-body =
    ## Your new account

    Your new account has been created:

onboarding-existing-account-body =
    ## Your existing account

    Enter your account details:

onboarding-purchase-body =
    ## Add credit

    Choose a plan or purchase credits directly. We use Stripe to process payments.

    Subscription credits last their billing period; one-time purchases last a year. Unused, unexpired credits are refundable on request.

# -- Moving through the flow -------------------------------------------------

# The up-chevron on every slide past the first. It has no visible text, so this
# is the whole of what a screen reader hears: say what the click does.
onboarding-back = Go to the previous slide

onboarding-cta-pause = OK, you have my attention.
# The three narrative slides end in the same acknowledgement, so they share one
# message — three ids saying one thing would be three chances to drift.
onboarding-cta-understood = I understand.
onboarding-cta-new-account = I need a new account.
onboarding-cta-existing-account = I already have an account.
# The quiet third way. Deliberately a full sentence and deliberately calm: it is
# a real choice (on-device models keep working), not a refusal or a downgrade.
# It is the visible text **and** the accessible name — one sentence, said once.
onboarding-cta-skip-account = Continue without an account — on-device models only.

# -- Consent -----------------------------------------------------------------

# The agreement checkbox: the visible label and the accessible name, which are
# the same sentence. This is the one sentence in the app a reader is asked to
# affirm, so translate it plainly and completely — no abbreviation, no
# rewording of what is being agreed to. The two document names are terms and
# stay in English (see the note at the top).
onboarding-consent-agree = I agree to the { -fixed-terms-of-service } and { -fixed-privacy-policy }.
# Shown while the current documents are being fetched. Their versions are what
# acceptance is recorded against, so the slide waits rather than guessing.
onboarding-terms-loading = Checking the current documents…
# Re-runs that fetch. Both the visible text and the accessible name.
onboarding-terms-retry = Try again

# A link out of the app. The arrow is what marks it as leaving; the label is
# whatever the row names.
onboarding-link-external = { $label } ↗
onboarding-link-repository = The Eidola code repository
# The two published policies, linked when the server names no documents of its
# own. Their titles do not translate; these exist so the link rows and the
# consent sentence read one name from one place.
onboarding-link-terms-of-service = { -fixed-terms-of-service }
onboarding-link-privacy-policy = { -fixed-privacy-policy }
# A required document's link label: its published name plus the version
# acceptance will be recorded for. One message rather than a name with a
# version appended, so a locale can put the two in its own order.
# TRANSLATORS: $name is the document's published title and does not translate.
onboarding-document-version = { $name } (version { $version })

onboarding-cta-create = Create a new account.
# The same button while the request is out. It says what is happening, not that
# anything is nearly done.
onboarding-cta-create-pending = Creating your anonymous account…

# -- The new account's credentials -------------------------------------------

onboarding-account-id = Account ID
onboarding-account-secret = Account Secret
# The id field's placeholder: the *shape* of an account id, not words.
# TRANSLATORS: leave this exactly as it is.
onboarding-account-id-placeholder = 00000000-0000-0000-0000-…
onboarding-account-secret-placeholder = account secret
# The copy verb beside a credential, and its accessible name — the same verb
# with the subject its row supplies, because a screen reader hears two
# identical "Copy"s without it.
onboarding-copy = Copy
onboarding-copy-label = Copy { $label }
# What the secret is for, and what losing it costs. The second half is the part
# that matters: say it without hedging.
onboarding-new-account-note = This is used only to add and consume credits. If the account secret is lost, it cannot be recovered, and you will need to create a new account.
onboarding-cta-saved = I've saved this somewhere.

# -- Checking an existing account --------------------------------------------

onboarding-cta-verify = Check account balance.
onboarding-cta-verify-pending = Checking…
# Refused before any request, because one field alone cannot check anything.
onboarding-verify-missing-credentials = Enter both an account ID and secret.
# The server answers "no such account" and "wrong secret" with one status, so
# that it cannot be used to discover which account ids exist. The copy honours
# that: it names both possibilities and offers the way forward.
onboarding-verify-unverifiable = We couldn't verify that account. Check the ID and secret, or create a new account instead.
# The balance a checked account holds. $credits is already grouped for reading
# ("1,250"); $count is the same number, unformatted, and is here only so the
# noun agrees with it.
onboarding-verified-balance =
    { $count ->
        [one] This account is valid and has a balance of { $credits } credit.
       *[other] This account is valid and has a balance of { $credits } credits.
    }
onboarding-cta-existing-purchase = I want to purchase more credits.
onboarding-cta-existing-done = This looks good.

# -- Adding credit -----------------------------------------------------------

# Where the checkout happens and where the credit lands — both worth saying
# before a browser window opens on its own.
onboarding-purchase-checkout-note = Checkout opens in your browser; credit lands on this account.
# The server refuses a second subscription, so the recurring plans are absent
# from the list. This says why, and where the existing one is managed.
# TRANSLATORS: "Settings ▸ Account" is a menu path, and those two labels are
# still English on screen — leave the path as it stands so it matches what the
# reader will actually look for, and translate it only in the same change that
# translates the Settings navigation itself.
onboarding-purchase-subscribed-note = This account already has a subscription — manage it in Settings ▸ Account. You can still add one-time credit here.
onboarding-purchase-loading = Loading plans…
onboarding-purchase-none-topups = No one-time top-ups are available right now.
onboarding-purchase-none-plans = No plans are available right now.
# The link came back for an account that is no longer the configured one, so it
# was discarded rather than opened — funding an account the reader has walked
# away from is the outcome this prevents. Saying so beats a button that quietly
# did nothing.
onboarding-checkout-stale-mint = The account changed while that was being prepared, so nothing was opened. Try again.
onboarding-cta-purchase-later = I will purchase credits later.
