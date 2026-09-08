# Réglages ▸ Proxy — le proxy d'inférence local (src/proxy_settings.rs), français.

proxy-lead = Laissez d'autres outils de cet ordinateur utiliser les modèles que vous atteignez via Eidola. Ils parlent l'API OpenAI ; chaque requête passe toujours par Eidola — attestée, payée depuis votre portefeuille et inscrite au Registre.

proxy-serve = Répondre aux requêtes
proxy-serve-name = Répondre aux requêtes locales via Eidola

proxy-listening = À l'écoute sur { $address }
proxy-stopped = Pas à l'écoute
proxy-listen-failed = Impossible d'écouter sur cette adresse — { $reason }

proxy-exposed-warning = Cette adresse est joignable depuis votre réseau, et le proxy ne chiffre encore rien. Tout ce qui y passe — vos invites et les réponses — circule en clair.

proxy-address = Adresse
proxy-address-name = L'adresse IP sur laquelle le proxy écoute
proxy-port-name = Le port sur lequel le proxy écoute
proxy-binding-change = Modifier…
proxy-binding-save = Enregistrer
proxy-binding-cancel = Annuler

proxy-backends = Fournisseurs
proxy-backends-note = Seul ce que vous cochez ici est joignable. Un outil qui nomme autre chose s'entend répondre que le modèle n'existe pas.
proxy-backend-name = Proposer { $backend } via le proxy
proxy-backends-empty = Aucun fournisseur n'est encore configuré.

proxy-exposure = Modèles sur l'appareil
proxy-exposure-loaded = Seulement chargés
proxy-exposure-downloaded = Tous les téléchargés
proxy-exposure-note = « Tous les téléchargés » démarre un moteur à la première requête qui nomme un modèle, ce qui prend du temps et occupe de la mémoire. « Seulement chargés » propose ce qui tourne déjà.

proxy-keys = Clés d'API
proxy-keys-note = Un outil envoie sa clé comme jeton bearer. Eidola n'en conserve qu'une empreinte : une clé s'affiche une fois et ne peut plus être affichée.
proxy-keys-empty = Aucune clé pour l'instant — rien ne peut atteindre le proxy tant que vous n'en créez pas une.
proxy-key-label-placeholder = Qu'est-ce qui utilisera cette clé ?
proxy-key-create = Générer une clé
proxy-key-revoked = révoquée
proxy-key-unused = jamais utilisée
proxy-key-used = utilisée
proxy-key-revoke = Révoquer
proxy-key-revoke-name = Révoquer la clé nommée { $label }

proxy-key-minted = Copiez-la maintenant. Eidola n'a gardé que son empreinte et ne peut plus l'afficher.
proxy-key-copy = Copier
proxy-key-done = Terminé

proxy-failed = Impossible de lire les réglages du proxy.
proxy-retry = Réessayer
