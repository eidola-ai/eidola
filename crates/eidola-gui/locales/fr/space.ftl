# The conversation window (src/space_view/mod.rs), French.

space-error-archived = Cette conversation est archivée : elle ne peut plus recevoir de réponses.
space-error-not-joined = Vos agents ont ouvert cette conversation entre eux. Vous pouvez la lire en entier ; y participer suppose de la rejoindre, ce que cette version ne sait pas encore faire.

space-footnote-delegation-concluded = est arrivée à son terme
space-footnote-delegation-paused = a atteint sa limite de réponses ({ $depth } sur { $limit })
space-footnote-delegation-budget = a épuisé ses { $limit } tours
space-footnote-delegation-failed = s’est arrêtée : { $reason ->
        [upstream] le modèle était injoignable
        [funding] le tour n’a pas pu être payé
        [configuration] quelque chose dans sa configuration
       *[other] un tour n’a pas pu aboutir
    }

space-regenerating = Régénération…
space-error-response-truncated = Le modèle a consacré toute sa longueur autorisée à réfléchir sans jamais commencer de réponse. Rien n’a été modifié : réessayez, ou posez une question plus courte.
space-error-regeneration-in-flight = Cette réponse est déjà en cours de régénération.
space-answer-cut-off = Cette réponse a atteint sa limite de longueur et s’interrompt en pleine pensée.
