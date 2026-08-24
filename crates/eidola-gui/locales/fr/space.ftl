# The conversation window (src/space_view/mod.rs), French.

space-error-archived = Cette conversation est archivée : elle ne peut plus recevoir de réponses.
space-error-not-joined = Vos agents ont ouvert cette conversation entre eux. Vous pouvez la lire en entier ; pour modifier ou redemander quelque chose ici, participez d’abord en publiant un message.

space-footnote-delegation-concluded = est arrivée à son terme
space-footnote-delegation-concluded-truncated = est arrivée à son terme, sur une réponse coupée à sa limite de longueur
space-footnote-delegation-paused = a atteint sa limite de réponses ({ $depth } sur { $limit })
space-footnote-delegation-paused-truncated = a atteint sa limite de réponses ({ $depth } sur { $limit }), sur une réponse coupée à sa limite de longueur
space-footnote-delegation-budget = a épuisé ses { $limit } tours
space-footnote-delegation-budget-truncated = a épuisé ses { $limit } tours, sur une réponse coupée à sa limite de longueur
space-footnote-delegation-failed = s’est arrêtée : { $reason ->
        [upstream] le modèle était injoignable
        [funding] le tour n’a pas pu être payé
        [configuration] quelque chose dans sa configuration
       *[other] un tour n’a pas pu aboutir
    }

space-regenerating = Régénération…
space-error-response-truncated = Le modèle a consacré toute sa longueur autorisée à réfléchir sans jamais commencer de réponse. Aucune réponse n’a été ajoutée ni remplacée : réessayez, ou posez une question plus courte.
space-regenerating-elsewhere = Cette réponse est déjà en cours de régénération.
space-answer-cut-off = Cette réponse a atteint sa limite de longueur et s’interrompt en pleine pensée.
