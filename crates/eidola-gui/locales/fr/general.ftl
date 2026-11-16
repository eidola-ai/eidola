# Réglages ▸ Général — apparence, taille du texte, démarrage et ligne de
# commande (src/general.rs), français.

general-appearance = Apparence

general-day-night = Jour et nuit
general-day-night-auto = Auto
general-day-night-system = Système
general-day-night-day = Jour
general-day-night-night = Nuit
general-day-night-choice = Jour et nuit : { $choice }
general-day-night-note = Système suit macOS. Auto suit le soleil — le jour entre le lever et le coucher.

general-time-of-day = Heure du jour
general-on = Activé
general-off = Désactivé
general-time-of-day-choice = Heure du jour : { $choice }
general-time-of-day-note = Suit la lumière du soleil — fraîche vers le lever, neutre à midi et à minuit, chaude vers le coucher. Le lever et le coucher sont estimés d'après votre fuseau horaire.

general-light = Lumière
general-light-cool = Froide
general-light-neutral = Neutre
general-light-warm = Chaude
general-light-choice = Lumière : { $choice }
general-light-note = Un caractère de lumière fixe pour la palette.

general-text-size = Taille du texte
general-text-size-smaller-name = Texte plus petit, actuellement { $percent }
general-text-size-larger-name = Texte plus grand, actuellement { $percent }
general-text-size-reset = Réinitialiser
general-text-size-reset-name = Rétablir la taille du texte à 100 %
general-text-size-note = Agrandit ou réduit tout le texte. Aussi dans le menu View — Actual Size, Zoom In, Zoom Out.

general-startup = Démarrage
general-open-at-login = Ouvrir à la connexion
general-login-item-switch = Lancer Eidola à l'ouverture de session
general-login-item-name = Ouvrir Eidola à la connexion
general-login-item-off = Eidola reste discret jusqu'à ce que vous l'ouvriez.
general-login-item-on = Eidola s'ouvre avec votre session.
general-login-item-needs-approval = Activé, mais macOS attend votre accord — autorisez Eidola dans Réglages Système → Général → Ouverture.
general-login-item-unsupported = Indisponible — il faut macOS 13 ou ultérieur, et une app installée et signée.
general-login-item-unsupported-linux = Indisponible ici — sous Linux, `eidola service install` active le service utilisateur.
general-login-item-too-old = L'ouverture à la connexion demande macOS 13 ou ultérieur — le système de ce Mac est plus ancien.
general-login-item-not-here = L'ouverture à la connexion n'est pas disponible sur cette plateforme.

general-command-line = Ligne de commande
general-cli-field = Commande Terminal
general-cli-not-installed = Non installée. L'installation place une commande `eidola` dans { $dir }, liée à cette copie d'Eidola, pour la lancer depuis le Terminal.
general-cli-needs-admin = macOS demandera le mot de passe d'un administrateur.
general-cli-installed = Installée — `eidola` dans le Terminal lance cette copie d'Eidola, via { $path }.
general-cli-dangling = { $path } pointe vers une copie d'Eidola qui n'existe plus ({ $target }) — l'app a sans doute été déplacée. Mettez à jour le lien pour utiliser cette copie.
general-cli-other-copy = { $path } lance une autre copie d'Eidola, située dans { $target }. Mettez à jour le lien pour utiliser cette copie.
general-cli-occupied-file = { $path } est déjà un fichier qu'Eidola n'a pas créé ; Eidola ne le remplacera pas. Si vous n'en voulez plus, déplacez-le ou supprimez-le vous-même, puis installez.
general-cli-occupied-directory = { $path } est un dossier ; Eidola n'y placera pas de commande. Si vous n'en voulez plus, déplacez-le ou supprimez-le vous-même, puis installez.
general-cli-occupied-link = { $path } est un lien vers { $target }, qui n'est pas Eidola ; Eidola ne le remplacera pas. Si vous n'en voulez plus, déplacez-le ou supprimez-le vous-même, puis installez.
general-cli-occupied-other = { $path } est déjà quelque chose qu'Eidola n'a pas créé ; Eidola ne le remplacera pas.
general-cli-unreadable = Impossible de vérifier { $path } — { $reason }
general-cli-not-bundled = Disponible uniquement quand Eidola tourne en tant qu'app installée.
general-cli-translocated = macOS exécute cette copie d'Eidola depuis un emplacement temporaire, et un lien vers elle cesserait de fonctionner. Placez Eidola dans votre dossier Applications, ouvrez-le depuis là, puis installez.
general-cli-missing = Cette copie d'Eidola ne contient pas d'outil en ligne de commande.
general-cli-install = Installer
general-cli-install-admin = Installer…
general-cli-install-name = Installer la commande eidola dans { $dir }
general-cli-update = Mettre à jour le lien
general-cli-update-admin = Mettre à jour le lien…
general-cli-update-name = Faire pointer { $path } vers cette copie d'Eidola
general-cli-installing = Installation…
general-cli-waiting = En attente du mot de passe d'un administrateur…
general-cli-declined = Rien n'a été modifié. Pour l'installer vous-même, lancez ceci dans le Terminal :
general-cli-failed = Impossible d'installer la commande — { $reason }
general-cli-manual = Pour l'installer vous-même, lancez ceci dans le Terminal :
general-cli-copy = Copier
general-cli-copy-name = Copier la commande d'installation
