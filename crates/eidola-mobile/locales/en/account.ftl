# The account and the wallet (src/ui/account_view.rs).

account-title = Account
# No account on this device: the screen offers the first run's account steps.
account-none = There’s no account on this device. An account holds the credit you buy; conversations are paid for with anonymous credits drawn from it.
account-set-up = Set up an account…
account-section-credentials = Credentials
account-id = Account ID
account-secret = Account Secret
account-show-secret = Show the account secret
account-hide-secret = Hide the account secret
account-secret-note = Keep these somewhere safe. If the secret is lost it cannot be recovered, and the credit on the account cannot be used.
account-section-balance = Balance
account-balance = Available
# A number of credits. $credits is already grouped for reading ("1,250"); $count
# is the same number, unformatted, so the noun agrees with it.
account-credits =
    { $count ->
        [one] { $credits } credit
       *[other] { $credits } credits
    }
# One pool of credit within the balance, and when it expires ($when).
account-pool =
    { $count ->
        [one] { $credits } credit
       *[other] { $credits } credits
    }
account-pool-expiring =
    { $count ->
        [one] { $credits } credit, { $when }
       *[other] { $credits } credits, { $when }
    }
account-expired = expired
account-expires-today = expires today
account-expires-tomorrow = expires tomorrow
account-expires-days =
    { $count ->
        [one] expires in { $count } day
       *[other] expires in { $count } days
    }
account-expires-weeks =
    { $count ->
        [one] expires in { $count } week
       *[other] expires in { $count } weeks
    }
account-expires-months =
    { $count ->
        [one] expires in { $count } month
       *[other] expires in { $count } months
    }
account-expires-years =
    { $count ->
        [one] expires in { $count } year
       *[other] expires in { $count } years
    }
account-balance-loading = Loading…
# $reason is the error's own text (English).
account-balance-failed = Couldn’t read the balance: { $reason }
account-balance-stale = Couldn’t re-read the balance — showing the last one. { $reason }
account-refresh = Refresh
account-retry = Try again
account-add-credit = Add credit
account-section-subscription = Subscription
account-subscription-checking = Checking your subscription…
account-subscription-failed = Couldn’t check your subscription.
account-subscription-stale = Couldn’t re-check your subscription — showing the last answer.
account-subscription-active = Your subscription is active.
account-subscription-trialing = Your subscription is in its trial period.
account-subscription-past-due = Your subscription is active, but a payment did not go through. Update your payment method to keep it.
# $status is the payment processor's own word for it.
account-subscription-other = Your subscription is in force, reported as “{ $status }”.
account-subscription-in-force = Your subscription is in force.
account-subscription-inactive = You don’t have a subscription right now.
account-subscription-none = You don’t have a subscription on this account.
# Opens the payment processor's billing portal in the browser.
account-manage = Manage subscription
account-billing = Billing and receipts
account-check-again = Check again
# While a billing link is being prepared.
account-opening = Opening…
# The link came back for an account that is no longer the configured one, so
# it was not opened.
account-stale-mint = The account changed while that was being prepared, so nothing was opened. Try again.
# $reason is the error's own text (English).
account-door-failed = Couldn’t open that: { $reason }
account-section-reset = Reset
account-reset = Reset account…
# The armed warning of the two-step reset.
account-reset-warning = This forgets the account keys on this device. Remaining balance becomes unreachable; the local record of what was spent stays on this device.
account-reset-confirm = Reset account
account-reset-keep = Keep account
account-op-failed = Couldn’t change the account: { $reason }

# The wallet (read-only).
wallet-title = Wallet
wallet-note = Anonymous spend credentials provision themselves from your balance — the server cannot link them back to your account.
wallet-empty = No credentials yet — they appear when you start chatting.
# $generation is the credential's generation number.
wallet-active = active · generation { $generation }
wallet-spending =
    { $count ->
        [one] in flight — { $credits } credit held
       *[other] in flight — { $credits } credits held
    }
wallet-spent =
    { $count ->
        [one] spent — { $credits } credit charged
       *[other] spent — { $credits } credits charged
    }
wallet-expired = expired
wallet-failed = Couldn’t read the wallet: { $reason }
