# The first run on a phone (src/ui/onboarding.rs), French: the
# desktop app's onboarding translations where the words are the same.

# The screen's title.
onboarding-title = Bienvenue

# The two published document titles and the id placeholder are deliberately
# absent, as on the desktop: they fall back to the English source.

onboarding-pause-body =
    ## *Faites une pause*

    Eidola n'est *pas* la même chose que ChatGPT, Claude ou Gemini.

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

onboarding-create-account-body =
    ## Créer un compte

    Veuillez lire et comprendre nos { -fixed-terms-of-service } et { -fixed-privacy-policy }.

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

onboarding-cta-understood = Je comprends.

onboarding-cta-new-account = J'ai besoin d'un nouveau compte.

onboarding-cta-existing-account = J'ai déjà un compte.

onboarding-consent-agree = J'accepte les { -fixed-terms-of-service } et la { -fixed-privacy-policy }.

onboarding-terms-loading = Vérification des documents actuels…

onboarding-terms-retry = Réessayer

onboarding-link-external = { $label } ↗

onboarding-document-version = { $name } (version { $version })

onboarding-cta-create = Créer un nouveau compte.

onboarding-cta-create-pending = Création de votre compte anonyme…

onboarding-account-id = Identifiant du compte

onboarding-account-secret = Secret du compte

onboarding-account-secret-placeholder = secret du compte

onboarding-new-account-note = Il sert uniquement à ajouter et à consommer des crédits. Si le secret du compte est perdu, il ne peut pas être récupéré et vous devrez créer un nouveau compte.

onboarding-cta-saved = Je l'ai enregistré quelque part.

onboarding-create-not-saved = Votre compte a été créé, mais il n'a pas pu être enregistré ici ({ $reason }). C'est la seule copie de son secret : copiez les deux maintenant, puis associez-les avec « J'ai déjà un compte. » une fois le problème résolu.

onboarding-create-recovery-file = Une copie a aussi été enregistrée dans { $path }. Supprimez ce fichier une fois le compte associé.

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

onboarding-cta-purchase-later = J'achèterai des crédits plus tard.

onboarding-get-started-body =
    ## Commencer

    Il vous faudra des crédits pour exécuter des modèles.

    Votre compte n'est qu'un identifiant aléatoire — l'achat de crédit est la seule étape qui touche à un moyen de paiement, et même alors il nous est structurellement impossible de relier vos requêtes à celui-ci.

onboarding-link-unlinkability = Comment vos requêtes restent dissociées de votre paiement
onboarding-cta-skip-account = Continuer sans compte.
onboarding-cta-plans = Choisir une formule…
onboarding-create-failed = Impossible de créer le compte : { $reason }
onboarding-verify-failed = Impossible de vérifier ce compte : { $reason }
