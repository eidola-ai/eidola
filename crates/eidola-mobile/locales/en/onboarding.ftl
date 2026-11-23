# The first run on a phone (src/ui/onboarding.rs). Most messages are the
# desktop app's onboarding messages, word for word (crates/eidola-gui/locales/),
# so the two apps say the same thing; the rest are the phone's own.
#
# TRANSLATORS: the page bodies are **Markdown** (`##` a heading, `*word*`
# emphasis, `**word**` strong, `- ` a bullet, a blank line between paragraphs).
# Translate the words; keep the markers.

# The two legal documents are named by their **published titles**, which stay
# English in every locale: the `-fixed-` prefix makes a locale that defines its
# own a build error.
-fixed-terms-of-service = Terms of Service
-fixed-privacy-policy = Privacy Policy

# The screen's title.
onboarding-title = Welcome

# "Pause here" — the first thing a reader sees. The point is the contrast, so
# keep the emphasis on *not*.
onboarding-pause-body =
    ## *Pause here*

    Eidola is *not* the same as ChatGPT, Claude or Gemini.

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

# The three narrative slides end in the same acknowledgement, so they share one
# message — three ids saying one thing would be three chances to drift.
onboarding-cta-understood = I understand.

onboarding-cta-new-account = I need a new account.

onboarding-cta-existing-account = I already have an account.

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

# What the secret is for, and what losing it costs. The second half is the part
# that matters: say it without hedging.
onboarding-new-account-note = This is used only to add and consume credits. If the account secret is lost, it cannot be recovered, and you will need to create a new account.

onboarding-cta-saved = I've saved this somewhere.

# Account creation succeeded on the server but this profile could not keep the
# credential. The two rows below this line are the only copy of the secret.
# $reason is the cause as the app reports it (English).
onboarding-create-not-saved = Your account was created, but it could not be saved here ({ $reason }). This is the only copy of its secret: copy both now, then link them with “I already have an account.” once that is fixed.

# Beneath the credential rows when the credential was also written to a file
# beside the profile. $path is that file's location; nothing removes it but the
# reader, so the sentence says when to.
onboarding-create-recovery-file = A copy was also saved to { $path }. Delete that file once the account is linked.

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

# Where the checkout happens and where the credit lands — both worth saying
# before a browser window opens on its own.
onboarding-purchase-checkout-note = Checkout opens in your browser; credit lands on this account.

onboarding-cta-purchase-later = I will purchase credits later.

# The branch point. The desktop's words, with its link moved out of the
# sentence into a row of its own (onboarding-link-unlinkability).
onboarding-get-started-body =
    ## Get started

    You'll need some credits to run models.

    Your account is just a random id — buying credit is the only step that touches a payment method, and even then we are structurally unable to link your requests back to it.

# Opens the documentation of that claim in the browser.
onboarding-link-unlinkability = How your requests stay unlinked from your payment
# The quiet third way. On a phone there are no on-device models, so it says
# only what it does; an account can be set up later from Settings.
onboarding-cta-skip-account = Continue without an account.
# On the add-credit page: opens the plans.
onboarding-cta-plans = Choose a plan…
# A refused creation or check, other than the ones with words of their own;
# $reason is the error's own text (English).
onboarding-create-failed = Couldn’t create the account: { $reason }
onboarding-verify-failed = Couldn’t check that account: { $reason }
