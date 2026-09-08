# The shared plans rows (src/plans.rs), French.

plans-list = Formules disponibles
plans-opening-checkout = Ouverture du paiement…

plans-credits-one-time =
    { $count ->
        [one] { $credits } crédit, expire un an après l'achat
       *[other] { $credits } crédits, expirent un an après l'achat
    }
plans-credits-recurring =
    { $count ->
        [one] { $credits } crédit, expire à la fin de chaque période de facturation
       *[other] { $credits } crédits, expirent à la fin de chaque période de facturation
    }
plans-credits-described = { $line } — { $description }
