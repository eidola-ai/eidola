# The shared plans rows (src/plans.rs), French.

plans-list = Formules disponibles
plans-opening-checkout = Ouverture du paiement…

plans-free = Gratuit
plans-price-cadence =
    { $count ->
        [1]
            { $interval ->
                [day] { $amount }/jour
                [week] { $amount }/semaine
                [month] { $amount }/mois
                [year] { $amount }/an
               *[other] { $amount }/{ $interval }
            }
       *[other]
            { $interval ->
                [day] { $amount } tous les { $count } jours
                [week] { $amount } toutes les { $count } semaines
                [month] { $amount } tous les { $count } mois
                [year] { $amount } tous les { $count } ans
               *[other] { $amount } tous les { $count } { $interval }
            }
    }

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
