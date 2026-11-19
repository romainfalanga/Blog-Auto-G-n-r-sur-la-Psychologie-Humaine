# Interface partagée Mon enfance / Mes symptômes

La page Hugo `mes-symptomes.md` sélectionne le jeu `symptomes` dans le layout et les deux scripts de Mon enfance. L'API, le brouillon, les textes, le compteur et le classement restent propres au jeu. Aucune demande d'accord ni questionnaire v1. Les lignes sont créées à l'ouverture d'un thème ; la première rubrique incomplète est ouverte après chargement de la session et des réponses. Les précisions gardent la fenêtre, la sauvegarde explicite et les conflits existants.

Le jeu public est transformé au build Hugo en JSON à empreinte SHA-256, sans aucune réponse utilisateur. Les résumés explicatifs des liens sont dans un second JSON à empreinte chargé à l'ouverture d'un détail. Le préchargement et la lecture de la banque partagent le même mode de requête pour réutiliser la réponse publique. Les réponses privées suivent le bootstrap de session existant et restent `private, no-store` ; la fonction Edge donne au fournisseur 300 ms avant de laisser le navigateur charger les API. Un bootstrap tardif n'est jamais réinjecté : aucune réponse tardive ne remplace les saisies du navigateur.

Mon enfance conserve son JSON incorporé et son rendu initial pour préserver le premier affichage déjà livré. Son classement est comparé avec l'instantané publié avant v2 sur 200 profils mixtes (Oui/Non/Souvent historique/Je ne sais pas et retours). Le retrait des anciennes pages Worker/proxy de Mes symptômes ne touche pas le menu Ambre actuel de la PR 104.

Le moteur symptômes est la même fonction que côté Worker : familles essentielles, seuil d'affichage, ordre score/c/m/identifiant, conditions d'usage des produits toutes à Oui. Les liens limités et réponses incertaines sont dans « À préciser », sans classement par ceux-ci. Un Oui à un symptôme de sécurité affiche une ligne 3114 sous cette question.

Tests : `tests/personal-page-data.test.cjs`, `tests/symptom-map-ranking.test.cjs`, `tests/symptom-map-browser.cjs`, et suites historiques de Mon enfance. Les fixtures légères des anciennes suites choisissent désormais explicitement la branche enfance du template Hugo ; leurs assertions de comportement sont conservées. Les journaux de recette locale sont sous `work/reprise-nov19/ui-*`.

## Vérifications terminées

- 33 scénarios historiques Mon enfance réussis dans `ui-enfance-final.log` (une première exécution sous forte charge avait expiré ; le scénario isolé puis la suite complète passent).
- 3 parcours symptômes complets à 390, 1280 et 1440 px : sauvegarde/rechargement, notes, thèmes et détails différés, sécurité, absence de débordement et d'erreur JS. Le parcours mobile vérifie aussi l'échec 503 puis la reprise du chargement des détails.
- 3 tests de calcul : comparaison de 200 profils Mon enfance avec l'instantané publié, cas de familles/produits/incertitude, parité navigateur/Worker sur 60 profils de la banque réelle.
- 4 tests de la fonction Edge, dont séparation des routes et repli après 300 ms, et les 5 tests de navigation concernés.

La relecture indépendante a relevé un risque interonglets hérité : changer de compte pendant qu'une carte reste ouverte. Toutes les lectures et mutations de cartes portent maintenant `X-Expected-User-Id`, contrôlé par le Worker. Un refus `session_changed` préserve le brouillon du compte initial, suspend les sauvegardes et propose de recharger la page au lieu de traiter cela comme un conflit de réponses. Le navigateur vérifie ce refus sans écriture ; la relecture backend vérifie les trois méthodes avec une véritable session d'un autre compte.

Les captures `symptoms-form-*` et `symptoms-results-*` sous `work/reprise-nov19/` sont rendues par Chromium avec les CSS du site et un jeu synthétique de 20 signes, destiné à contrôler le comportement ; elles ne constituent pas une recette de production ni une revue clinique de la banque.
