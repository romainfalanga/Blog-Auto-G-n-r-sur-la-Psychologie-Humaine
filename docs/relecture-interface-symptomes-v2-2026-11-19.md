# Relecture indépendante de l’interface symptômes v2 — 19 novembre 2026

Périmètre : commits `70334ef` et `bc74473`, layout Hugo, scripts/CSS communs, chargement public différé et bootstrap Edge. Relecteur distinct de l'auteur UI.

## Vérifications

- API, clé du brouillon et page de bootstrap sont séparées pour Mon enfance et Mes symptômes ; la clé locale est suffixée par identifiant du compte. Un brouillon anonyme n'est importé qu'après confirmation d'une carte vide.
- Les données publiques à empreinte ne contiennent aucune réponse utilisateur. La banque est préchargée puis réutilisée ; ses détails sont chargés seulement à la demande et une erreur laisse possible une nouvelle tentative explicite. Texte rendu par nœuds DOM et non HTML provenant des notes.
- Edge : seules les quatre routes exactes GET sont traitées, destination fournisseur fixe, seul le cookie est relayé, absence de cache partagé, `Vary: Cookie`, validateurs supprimés et JSON échappé. Après 300 ms sans réponse préparée, le navigateur utilise ses lectures ordinaires ; aucune résolution tardive n'est injectée dans la page. Le délai limite l'attente des données préparées, pas la durée du chargement Hugo lui-même.
- Les lignes symptômes ne sont créées qu'à l'ouverture des thèmes ; Mon enfance conserve son rendu immédiat et ses anciennes réponses. La grille de choix, les fenêtres de précision et la sauvegarde sérialisée sont communes.
- Le moteur Mon enfance est comparé à l'implémentation publiée sur 200 profils mixtes ; le moteur symptômes navigateur/Worker est comparé sur 60 profils de la banque réelle. Les liens limités et l'incertitude n'augmentent pas le classement.

## Point relevé puis corrigé

Le suffixe du brouillon local ne suffit pas si une autre fenêtre remplace la session A par B : une fenêtre encore ouverte sur A peut envoyer un PUT avec le cookie B et une version compatible, ou charger B après un échec réseau puis sauvegarder sous la portée locale A. Une liaison explicite à l'identifiant attendu doit donc accompagner les lectures et mutations de carte et être vérifiée par le serveur. Correction partagée appliquée : `X-Expected-User-Id` sur GET/PUT/DELETE, vérifié avant accès à la carte par le serveur. Le corps PUT accepte également une identité attendue. Une ancienne fenêtre reçoit `409 session_changed`, garde son brouillon et ne peut plus lire, remplacer ou effacer la carte de l'autre compte.

Les captures et scénarios navigateur de l'auteur sont dans `work/reprise-nov19/` ; ce rapport distingue ces preuves de la relecture indépendante. Les contrôles Edge et parité ont été relancés indépendamment (`ui-independent-tests.log`).

## Contre-vérifications indépendantes

- Contrôles Edge et parité : **7/7 réussis**.
- `journal/tests/symptom-map-session-review.test.mjs` : vrai Worker, cookie B et identité attendue A ; GET, PUT et DELETE refusés, aucune donnée B dans la réponse et carte B inchangée. **1/1 réussi** (`ui-session-review-independent.log`).
- Suite Worker existante relancée : **1/1 réussi**, requête bornée, authentification, export et effacement (`ui-session-review.log`).
- Recette navigateur relancée indépendamment après correction : **390, 1280 et 1440 px réussis**, sauvegarde/rechargement, thèmes/détails différés, erreur de détails suivie d'une reprise, précision et ligne de sécurité, blocage d'une session remplacée. Journal : `ui-browser-review.log`. Capture mobile du formulaire également inspectée.

**Avis favorable après correction du changement de compte.** Le premier passage du nouveau scénario de session avait révélé que le message 409 était remplacé par un conflit générique. Corrigé : code `session_changed` conservé, sauvegarde désactivée, brouillon de A conservé et action explicite Recharger la page. La contre-recette passe sans erreur JavaScript ni débordement. La vérification distante reste à faire après déploiement.
