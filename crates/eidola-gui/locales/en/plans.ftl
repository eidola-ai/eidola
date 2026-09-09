# The shared plans rows (src/plans.rs) — the purchase choices the onboarding
# window and Settings ▸ Account both render.
#
# **The component holds no strings; each caller brings its own** (the
# `load_error_panel` rule). These messages are what a caller hands it, chosen in
# whichever locale that caller names — so the onboarding slide reads them in the
# reader's language while the Account pane, still English around them, pins them
# to the source locale until its own extraction. One definition either way.
#
# English is the source locale: every message must exist here, and the build
# script generates one typed accessor per message from this file.

# The rows are a single-select list; this is what names it to a screen reader.
plans-list = Available plans
# Replaces a plan's price line while its checkout request is in flight. A real
# request is out — this is not a fake state, and it ends when the link opens or
# the failure is reported.
plans-opening-checkout = Opening checkout…

# A price that costs nothing. A word, so it is said in the reader's language;
# the amounts beside it are numbers, and stay as the server states them.
plans-free = Free
# What a recurring price is charged with: the amount, then how often. The
# upstream names the interval, so this selects on its own vocabulary and keeps
# a fourth answer for one this build has never seen — a cadence the reader is
# owed even when the app cannot spell it.
# TRANSLATORS: $count is how many intervals make one billing period (usually 1);
# $interval is the upstream's own word and is never shown, only selected on.
# The single-interval English is byte-for-byte what this app rendered before the
# cadence was localized, so the surfaces still English around these rows did not
# move; the multi-interval arm says it in words instead of the old "/2xmonth".
plans-price-cadence =
    { $count ->
        [1]
            { $interval ->
                [day] { $amount }/day
                [week] { $amount }/week
                [month] { $amount }/month
                [year] { $amount }/year
               *[other] { $amount }/{ $interval }
            }
       *[other]
            { $interval ->
                [day] { $amount } every { $count } days
                [week] { $amount } every { $count } weeks
                [month] { $amount } every { $count } months
                [year] { $amount } every { $count } years
               *[other] { $amount } every { $count } { $interval }
            }
    }

# **The conspicuous expiry disclosure, at the point of purchase.** It must stay
# consistent with the published terms and with the server's own webhook expiry
# logic (period end vs. one year) — so translate what it says exactly, and never
# soften "expire".
#
# $credits is already grouped for reading ("5,000,000"); $count is the same
# number unformatted, and is here only so the noun and its verb agree with it.
# The whole line is one sentence per plan kind rather than a noun with a clause
# appended, because which half of it inflects differs by language.
plans-credits-one-time =
    { $count ->
        [one] { $credits } credit, expires one year after purchase
       *[other] { $credits } credits, expire one year after purchase
    }
plans-credits-recurring =
    { $count ->
        [one] { $credits } credit, expires at the end of each billing period
       *[other] { $credits } credits, expire at the end of each billing period
    }
# A plan whose product carries a description of its own: the credits line, then
# what the seller calls it.
# TRANSLATORS: $description is the product's own text, supplied by the server
# and not translated here.
plans-credits-described = { $line } — { $description }
