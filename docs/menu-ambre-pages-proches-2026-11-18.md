# Ambre dans le menu, pages des proches, état vide des bilans — 18 novembre 2026

**État : développé et vérifié localement. Non commité, non poussé, non déployé.** Aucune migration, aucune nouvelle règle Netlify, aucun appel vers la production. Le chantier éditorial reste en pause.

## Demande du propriétaire

1. Retirer la bulle Ambre et remettre « Ambre » en tête du menu, vers `/mon-accompagnement/`. L’accueil reste public.
2. Mes proches : une sous-page par proche, liste simplifiée, retrait visuel de « Repères sur cette relation ».
3. Mes bilans : plus de lien vers la conversation quand aucun bilan n’existe.

## Ce qui change

### Bulle Ambre retirée

- Supprimés : `static/js/ambre-widget.js`, `static/css/ambre-widget.css`, leur chargement dans `layouts/_default/baseof.html`, leur injection dans les pages du Worker, les deux exports `AMBRE_WIDGET_*` de `scripts/build_navigation.mjs`, la suite `tests/ambre-widget-browser.cjs`.
- Mode embarqué supprimé, car il ne servait qu’à la bulle : paramètre `?embedded=1`, classe `ambre-embedded` et ses styles, pont `postMessage` (messages `open`, `close`, `ready`, `navigate`, `focus-*`, `session-changed`), vérifications d’origine et de fenêtre, relais de navigation, boucle de focus, événements `dte:session-changed`, `dte:session-invalidated` et `dte:page-changed`. `?embedded=1` est désormais ignoré : la page de conversation s’affiche normalement.
- Toutes les réponses du Worker portent de nouveau `X-Frame-Options: DENY` et `frame-ancestors 'none'`, sans exception.
- Conservé : la remise à zéro de l’état privé quand la session change de compte (`clearSessionState`, appelée par `refresh`), le brouillon de conversation lors des navigations internes, la dictée.
- Menu : `data/navigation.json` commence par `Ambre → /mon-accompagnement/` ; `journal/src/navigation.generated.mjs` régénéré ; commentaire de `netlify.toml` mis à jour. Le menu du Worker et celui du site restent identiques (test).
- Conséquence assumée : la synchronisation de session entre onglets (`BroadcastChannel`) vivait dans le script de la bulle et disparaît avec elle, comme avant le 17 novembre. Un onglet resté ouvert après une déconnexion ailleurs reçoit toujours les refus 401 du serveur.

### Une page par proche

- Adresse : `/mes-proches/<identifiant>/` (avec ou sans barre finale), servie par le Worker. La règle Netlify existante `/mes-proches/*` → `…/mes-proches/:splat` (200, forcée) et son en-tête `private, no-store` couvrent déjà ces adresses : rien à ajouter.
- Premier affichage rempli : le Worker insère dans la page les réponses de `/api/me`, `/api/people/<id>` et `/api/people/<id>/timeline` (les lectures authentifiées existantes, sans écriture ni appel de modèle). Vérifié dans Chromium : contenu complet dès le premier rendu, aucune requête API, au chargement direct comme au rechargement.
- Contenu : retour « ← Mes proches », prénom en titre, rôle, informations (relation, anciens repères de période, champs facultatifs), Modifier, Supprimer, puis « Votre histoire avec … » : rencontre, événements liés et leurs notes, notes personnelles, repères proposés par Ambre et validés (toujours retirables), ajout de note, lien vers les événements liés.
- Proche inconnu ou d’un autre compte : mêmes réponses 404 des deux lectures, message « Ce proche est introuvable » et lien « Revenir à Mes proches ». Aucun nom d’un autre compte n’atteint la page. Visiteur non connecté : écran de connexion habituel.
- Liste `/mes-proches/` : carte courte (prénom, rôle, première ligne de la relation, lien « Notre histoire »). Modifier et Supprimer ne sont plus sur la carte mais sur la page du proche ; l’ajout d’un proche reste sur la liste.
- Compatibilité : `/mes-proches/?person=<id>` redirige (302) vers la page du proche ; `/mes-proches/#person-<id>` cible toujours la carte ; les liens « Avec … » de Ma ligne de vie ouvrent la page du proche.
- Modification depuis la page : après enregistrement, la page et sa chronologie sont rechargées. Une note ajoutée met à jour la révision du proche et celle d’un brouillon de modification ouvert, pour éviter un faux conflit. Après suppression, retour à la liste avec « Proche supprimé. » et l’adresse supprimée est remplacée dans l’historique.
- « Repères sur cette relation » n’est plus affiché ni embarqué dans la page. L’API des observations, les données et le module `journal/src/people-observations-ui.mjs` (conservé, non chargé, toujours testé) restent. Les libellés de rubriques nécessaires au formulaire de proposition de la conversation sont embarqués séparément (`PEOPLE_AXES_UI`).
- API : `GET /api/people/<id>` et `GET /api/life-events/<id>` ajoutent `fields_ready` (déjà présent sur les listes), nécessaire au formulaire de modification ouvert directement sur la page.

### Mes bilans

- État vide : titre et explication conservés, lien « Ouvrir la conversation avec Ambre » retiré.

## Points d’attention

- Les anciens repères automatiques (origine `auto`, non validés individuellement) et les propositions en attente n’ont plus d’affichage dans Mes proches. Les propositions se valident toujours dans la conversation ; l’ensemble reste lisible dans Paramètres et dans l’export. Le retrait d’un ancien repère automatique n’a plus de bouton dans l’interface (l’API `DELETE /api/people/<id>/observations/<id>` existe toujours). À décider par le propriétaire si un retrait doit rester possible.
- La liste des proches charge toujours les 50 premiers proches (comportement antérieur inchangé).
- La page d’un proche répond 200 même quand le proche est introuvable : c’est l’application qui affiche le message, comme pour les autres pages privées.

## Vérifications locales

- `npm --prefix journal test` : 847 réussis (840 avant ; 1 test de la bulle remplacé, 8 ajoutés).
- `node --test tests/*.test.cjs` : 48 réussis, 1 ignoré sans `HUGO_BIN` (nouveau fichier `tests/menu-ambre.test.cjs`, 3 tests).
- Contrôles `--check` et tests Python de `.github/workflows/verify.yml` : voir le rapport de livraison du lot.
- Suites navigateur existantes relancées : `tests/v5-A5-P2-browser.cjs` (adaptée à la page du proche et au menu) et `tests/personal-loading-browser.cjs`.
- Recette Chromium à 390 et 1440 px avec le vrai Worker et SQLite en mémoire (scripts hors dépôt) : menu, absence de bulle, liste, page du proche au premier rendu, rechargement, retour, historique, modification, note, lien depuis la ligne de vie, proche inconnu, autre compte, visiteur non connecté, état vide des bilans, conversation avec et sans `?embedded=1`, suppression. Site public construit avec Hugo 0.140.0 hors dépôt : Ambre en tête du menu, aucune bulle, accueil public.

## Déploiement à prévoir

1. Worker complet (`journal/src/`), sans migration (26 migrations inchangées).
2. Netlify : construction habituelle ; aucune nouvelle redirection ni en-tête. Les anciens fichiers `/js/ambre-widget.js` et `/css/ambre-widget.css` disparaissent du site.
3. Ordre : aucun ordre ne casse les données. Si Netlify est publié d’abord, les pages privées gardent brièvement l’ancienne bulle, qui fonctionne encore. Si le Worker est publié d’abord, les pages publiques encore servies avec l’ancienne bulle verraient son iframe refusée (`frame-ancestors 'none'`) jusqu’à la publication Netlify : garder l’intervalle court.
4. Recette distante : menu sur une page publique et une page privée, `/mes-proches/<id>/` en chargement direct, identifiant d’un autre compte, `/mes-bilans/` sans bilan, en-têtes d’encadrement de `/mon-accompagnement/?embedded=1`.
