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
space-error-model-unavailable = { $model } n’est plus proposé par son backend, et aucun autre modèle ne répondra à sa place. Rien n’a été envoyé ni facturé : choisissez un autre modèle, puis réessayez.
space-regenerating-elsewhere = Cette réponse est déjà en cours de régénération.
space-answer-cut-off = Cette réponse s’est arrêtée à sa limite de longueur.
space-continue-answer = Continuer
space-continue-answer-label = Poursuivre cette réponse là où elle s’est arrêtée
space-continuing = Poursuite de la réponse…
space-error-issuer-key-off-calendar = Le serveur d’Eidola a proposé une clé de paiement qui ne suit pas le calendrier public des clés ; cette application ne l’a donc pas utilisée. Rien n’a été envoyé ni facturé, et les crédits que vous détenez déjà fonctionnent toujours.
space-error-issuer-key-second-for-epoch = Le serveur d’Eidola a proposé cette semaine une clé de paiement différente de celle que cette application connaît déjà. Un serveur honnête n’utilise qu’une clé par semaine : cette application n’acceptera donc pas de nouveaux crédits sous cette clé avant celle de la semaine prochaine. Rien n’a été envoyé ni facturé, et les crédits que vous détenez déjà fonctionnent toujours.
space-error-issuer-key-reused = Le serveur d’Eidola a proposé la même clé de paiement pour deux semaines différentes, ce qu’un serveur honnête ne fait jamais ; cette application ne l’a donc pas utilisée. Rien n’a été envoyé ni facturé, et les crédits que vous détenez déjà fonctionnent toujours.
