# Account (French). See the English source for what each message is.
account-title = Compte
account-none = Il n'y a aucun compte sur cet appareil. Un compte contient le crédit que vous achetez ; les conversations sont payées avec des crédits anonymes tirés de lui.
account-set-up = Configurer un compte…
account-section-credentials = Identifiants
account-show-secret = Afficher le secret du compte
account-hide-secret = Masquer le secret du compte
account-secret-note = Conservez-les en lieu sûr. Si le secret est perdu, il ne peut pas être récupéré, et le crédit du compte ne pourra pas être utilisé.
account-section-balance = Solde
account-balance = Disponible
account-credits =
    { $count ->
        [one] { $credits } crédit
       *[other] { $credits } crédits
    }
account-pool =
    { $count ->
        [one] { $credits } crédit
       *[other] { $credits } crédits
    }
account-pool-expiring =
    { $count ->
        [one] { $credits } crédit, { $when }
       *[other] { $credits } crédits, { $when }
    }
account-expired = expiré
account-expires-today = expire aujourd'hui
account-expires-tomorrow = expire demain
account-expires-days =
    { $count ->
        [one] expire dans { $count } jour
       *[other] expire dans { $count } jours
    }
account-expires-weeks =
    { $count ->
        [one] expire dans { $count } semaine
       *[other] expire dans { $count } semaines
    }
account-expires-months =
    { $count ->
        [one] expire dans { $count } mois
       *[other] expire dans { $count } mois
    }
account-expires-years =
    { $count ->
        [one] expire dans { $count } an
       *[other] expire dans { $count } ans
    }
account-balance-loading = Chargement…
account-balance-failed = Impossible de lire le solde : { $reason }
account-balance-stale = Impossible de relire le solde — le dernier est affiché. { $reason }
account-refresh = Actualiser
account-retry = Réessayer
account-add-credit = Ajouter du crédit
account-section-subscription = Abonnement
account-subscription-checking = Vérification de votre abonnement…
account-subscription-failed = Impossible de vérifier votre abonnement.
account-subscription-stale = Impossible de revérifier votre abonnement — la dernière réponse est affichée.
account-subscription-active = Votre abonnement est actif.
account-subscription-trialing = Votre abonnement est en période d'essai.
account-subscription-past-due = Votre abonnement est actif, mais un paiement n'a pas abouti. Mettez à jour votre moyen de paiement pour le conserver.
account-subscription-other = Votre abonnement est en vigueur, avec l'état « { $status } ».
account-subscription-in-force = Votre abonnement est en vigueur.
account-subscription-inactive = Vous n'avez pas d'abonnement en ce moment.
account-subscription-none = Vous n'avez pas d'abonnement sur ce compte.
account-manage = Gérer l'abonnement
account-billing = Facturation et reçus
account-check-again = Vérifier à nouveau
account-opening = Ouverture…
account-stale-mint = Le compte a changé pendant la préparation, rien n'a donc été ouvert. Réessayez.
account-door-failed = Impossible d'ouvrir : { $reason }
account-section-reset = Réinitialiser
account-reset = Réinitialiser le compte…
account-reset-warning = Cela oublie les clés du compte sur cet appareil. Le solde restant devient inaccessible ; l'historique local de ce qui a été dépensé reste sur cet appareil.
account-reset-confirm = Réinitialiser le compte
account-reset-keep = Garder le compte
account-op-failed = Impossible de modifier le compte : { $reason }
wallet-title = Portefeuille
wallet-note = Les identifiants de dépense anonymes se constituent d'eux-mêmes à partir de votre solde — le serveur ne peut pas les relier à votre compte.
wallet-empty = Aucun identifiant pour l'instant — ils apparaissent quand vous commencez à discuter.
wallet-active = actif · génération { $generation }
wallet-spending =
    { $count ->
        [one] en cours — { $credits } crédit retenu
       *[other] en cours — { $credits } crédits retenus
    }
wallet-spent =
    { $count ->
        [one] dépensé — { $credits } crédit facturé
       *[other] dépensé — { $credits } crédits facturés
    }
wallet-expired = expiré
wallet-failed = Impossible de lire le portefeuille : { $reason }
account-id = Identifiant du compte
account-secret = Secret du compte
