# The onboarding window (src/onboarding/), French.
#
# `-terms-of-service`, `-privacy-policy`, `onboarding-link-terms-of-service`,
# `onboarding-link-privacy-policy` and `onboarding-account-id-placeholder` are
# deliberately absent: the two published document titles stay English in every
# locale, and the id placeholder is a shape rather than words. They fall back to
# the English source.

onboarding-pause-body =
    ## *Faites une pause*

    Eidola n'est *pas* la même chose que ChatGPT, Claude ou Gemini.

onboarding-tool-body =
    ## Eidola est *votre* outil

    Autrefois, une application vous était livrée sur un CD :

    - Son comportement ne *pouvait pas* changer spontanément sans votre intervention.
    - Vos fichiers, vos projets, vos habitudes d'utilisation et vos idées n'appartenaient qu'à *vous seul*, hors de portée de tout tiers.
    - C'était la **structure** de la technologie — et *non* les promesses d'une entreprise — qui garantissait ces propriétés.

    Eidola se rapproche autant que possible de cette approche, en maximisant structurellement la souveraineté de l'utilisateur, même pour des traitements qu'il vaut mieux exécuter dans un centre de données.

onboarding-control-body =
    ## *Votre* contrôle

    Vous, et vous seul, avez le contrôle — ni nous, ni les opérateurs qui font tourner le matériel :

    - **Vous seul pouvez lire, conserver ou profiler vos interactions.** Vos données ne sont déchiffrées qu'à l'intérieur d'enclaves scellées et attestées par le matériel, qui ne conservent rien, et la partie d'Eidola qui gère le paiement est cryptographiquement séparée de celle qui traite vos requêtes.
    - **Vous seul pouvez mettre Eidola à jour — sur votre appareil comme sur le serveur.** Rien ne change tant que votre client n'a pas vérifié une nouvelle version et que vous n'avez pas décidé de lui faire confiance.

    Ne croyez pas aveuglément nos affirmations ; vérifiez-les. Si vous ne savez pas comment évaluer notre code et notre architecture, **demandez l'avis de la personne la plus technique en qui vous avez déjà confiance**.

onboarding-responsibility-body =
    ## *Votre* responsabilité

    Eidola est un outil qui vous permet d'exécuter plus facilement des modèles d'IA sans compétences techniques très spécialisées ni matériel coûteux. Personne ne regarde par-dessus votre épaule, il est donc essentiel que vous compreniez ceci :

    - Vous allez exécuter des modèles d'IA, qui sont au fond des ensembles de probabilités. Les modèles qu'Eidola exécute sont librement téléchargeables et utilisables, avec ou sans Eidola.
    - Même les meilleurs modèles sont faillibles. Ils peuvent être extraordinairement utiles, mais ils commettent des erreurs et peuvent avoir un comportement inattendu. Il est mathématiquement impossible d'évaluer comment un grand modèle se comportera dans tous les cas de figure.
    - Les modèles n'ont en eux-mêmes ni mémoire intrinsèque ni capacité à produire des effets externes ; ils ne peuvent qu'évaluer les données et effectuer les actions que vous mettez à leur disposition. Eidola facilite la compréhension et la configuration de ces accès, mais les résultats — bons comme mauvais — relèvent en dernier ressort de votre responsabilité.

onboarding-get-started-body =
    ## Commencer

    Il vous faudra des crédits pour exécuter des modèles.

    Votre compte n'est qu'un identifiant aléatoire — l'achat de crédit est la seule étape qui touche à un moyen de paiement, et même alors [il nous est structurellement impossible de relier vos requêtes à celui-ci]({ $unlinkability }).

onboarding-create-account-body =
    ## Créer un compte

    Veuillez lire et comprendre nos { -terms-of-service } et { -privacy-policy }.

onboarding-new-account-body =
    ## Votre nouveau compte

    Votre nouveau compte a été créé :

onboarding-existing-account-body =
    ## Votre compte existant

    Saisissez les informations de votre compte :

onboarding-purchase-body =
    ## Ajouter du crédit

    Choisissez une formule ou achetez directement des crédits. Nous utilisons Stripe pour traiter les paiements.

    Les crédits d'abonnement durent le temps de leur période de facturation ; les achats ponctuels durent un an. Les crédits inutilisés et non expirés sont remboursables sur demande.

onboarding-back = Aller à la diapositive précédente

onboarding-cta-pause = D'accord, vous avez mon attention.
onboarding-cta-understood = Je comprends.
onboarding-cta-new-account = J'ai besoin d'un nouveau compte.
onboarding-cta-existing-account = J'ai déjà un compte.
onboarding-cta-skip-account = Continuer sans compte — modèles sur l'appareil uniquement.

onboarding-consent-agree = J'accepte les { -terms-of-service } et la { -privacy-policy }.
onboarding-terms-loading = Vérification des documents actuels…
onboarding-terms-retry = Réessayer

onboarding-link-external = { $label } ↗
onboarding-link-repository = Le dépôt de code d'Eidola
onboarding-document-version = { $name } (version { $version })

onboarding-cta-create = Créer un nouveau compte.
onboarding-cta-create-pending = Création de votre compte anonyme…

onboarding-account-id = Identifiant du compte
onboarding-account-secret = Secret du compte
onboarding-account-secret-placeholder = secret du compte
onboarding-copy = Copier
onboarding-copy-label = Copier { $label }
onboarding-new-account-note = Il sert uniquement à ajouter et à consommer des crédits. Si le secret du compte est perdu, il ne peut pas être récupéré et vous devrez créer un nouveau compte.
onboarding-cta-saved = Je l'ai enregistré quelque part.

onboarding-cta-verify = Vérifier le solde du compte.
onboarding-cta-verify-pending = Vérification…
onboarding-verify-missing-credentials = Saisissez à la fois un identifiant et un secret de compte.
onboarding-verify-unverifiable = Nous n'avons pas pu vérifier ce compte. Vérifiez l'identifiant et le secret, ou créez plutôt un nouveau compte.
onboarding-verified-balance =
    { $count ->
        [one] Ce compte est valide et dispose d'un solde de { $credits } crédit.
       *[other] Ce compte est valide et dispose d'un solde de { $credits } crédits.
    }
onboarding-cta-existing-purchase = Je veux acheter plus de crédits.
onboarding-cta-existing-done = Cela me convient.

onboarding-purchase-checkout-note = Le paiement s'ouvre dans votre navigateur ; le crédit est versé sur ce compte.
onboarding-purchase-subscribed-note = Ce compte a déjà un abonnement — gérez-le dans Settings ▸ Account. Vous pouvez tout de même ajouter du crédit ponctuel ici.
onboarding-purchase-loading = Chargement des formules…
onboarding-purchase-none-topups = Aucun achat ponctuel n'est disponible pour le moment.
onboarding-purchase-none-plans = Aucune formule n'est disponible pour le moment.
onboarding-checkout-stale-mint = Le compte a changé pendant la préparation, rien n'a donc été ouvert. Réessayez.
onboarding-cta-purchase-later = J'achèterai des crédits plus tard.
