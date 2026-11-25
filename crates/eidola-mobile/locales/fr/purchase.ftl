# Purchase (French). See the English source for what each message is.
purchase-title = Ajouter du crédit
purchase-intro = Choisissez une formule ou achetez du crédit en une fois. Les crédits d'abonnement durent leur période de facturation ; les achats ponctuels durent un an. Les crédits inutilisés et non expirés sont remboursables sur demande.
purchase-checkout-note = Le paiement s'ouvre dans votre navigateur, hors de cette application, et le crédit est versé sur ce compte. Revenez ici une fois terminé.
purchase-subscribed-note = Ce compte a déjà un abonnement — gérez-le depuis Compte. Vous pouvez encore ajouter du crédit ponctuel ici.
purchase-loading = Chargement des formules…
purchase-failed = Impossible de charger les formules : { $reason }
purchase-none = Aucune formule n'est disponible pour le moment.
purchase-none-topups = Aucun achat ponctuel n'est disponible pour le moment.
purchase-opened = Le paiement est ouvert dans votre navigateur. Votre solde est relu à votre retour.

# The plan rows: the desktop app's plans.ftl translations.
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

purchase-terms-required = Le paiement a été refusé : ce compte n'a pas accepté les conditions en vigueur. Lisez-les et acceptez-les, puis réessayez.
purchase-review-terms = Lire les conditions…
