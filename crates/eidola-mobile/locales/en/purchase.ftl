# Adding credit (src/ui/purchase.rs). The plan rows' words are the desktop
# app's (its plans.ftl), so the two say the same thing.

purchase-title = Add credit
purchase-intro = Choose a plan or buy credit once. Subscription credits last their billing period; one-time purchases last a year. Unused, unexpired credits are refundable on request.
# Where the checkout happens — worth saying before the browser opens.
purchase-checkout-note = Checkout opens in your browser, outside this app, and the credit lands on this account. Come back here when you’re done.
purchase-subscribed-note = This account already has a subscription — manage it from Account. You can still add one-time credit here.
purchase-loading = Loading plans…
purchase-failed = Couldn’t load the plans: { $reason }
purchase-none = No plans are available right now.
purchase-none-topups = No one-time top-ups are available right now.
purchase-opened = Checkout is open in your browser. Your balance is read again when you come back.

# The plan rows (the desktop's plans.ftl).
plans-list = Available plans
plans-opening-checkout = Opening checkout…
plans-free = Free
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
plans-credits-described = { $line } — { $description }

# A checkout the server refused because the account has not agreed to the current terms.
purchase-terms-required = Checkout was refused: this account hasn’t agreed to the current terms. Review and agree to them, then try again.
purchase-review-terms = Review the terms…
