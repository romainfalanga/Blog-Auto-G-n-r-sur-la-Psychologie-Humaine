# Refonte « carnet clinique » — journal du 18 novembre 2026

Application de `docs/audit-design-2026-11-18.md`. **Non commité, non poussé, non déployé.** Aucun changement d'API, de données, de routes, de consentements ni de quotas. Aucune migration, aucune règle Netlify nouvelle.

## Fait

### Un seul système
- `static/css/site-system.css` (ancien `site-header.css`, renommé) : jetons du § 2.1 de l'audit, cadre de page (1040 px, colonne de 720 px alignée à gauche, gouttière 16 px puis 24 px), en-tête (nom du site visible sur mobile, page courante soulignée ou marquée d'un trait vertical dans le menu mobile), pied de page (nom du site seul), titre de page unique, boutons (primaire, secondaire, texte, rouge pour la suppression, 44 px), champ unique (48 px, 16 px), fenêtre unique (centrée, feuille basse sous 600 px), focus, `prefers-reduced-motion`.
- Ce fichier est chargé par Hugo (`layouts/partials/head.html`, avant `style.css`) et recopié dans le Worker par `scripts/build_navigation.mjs` (`SITE_SYSTEM_CSS`, `SITE_FOOTER_HTML`) : une seule définition, identique des deux côtés.
- `static/css/style.css` réécrit (pages publiques) ; `static/css/cartographie.css` réécrit (Mon enfance) ; anciens styles intégrés de `journal/src/ui.mjs` supprimés ; `journal/src/workspace-style.mjs` et le style de `change-interface.mjs` réécrits. Plus d'ombre (sauf fenêtre), plus de dégradé, plus d'orange, deux rayons (6 et 10 px), sept tailles de texte.

### Par page
- **Accueil, /personnages/** : carte entièrement cliquable (lien « Découvrir ses histoires » retiré), thèmes en texte, « 123 personnages », état vide de recherche en texte simple.
- **Fiche personnage** : portrait sans encadré, thèmes et concepts en une ligne de texte, histoires en lignes séparées par des filets, ligne entière cliquable.
- **Article** : plus de carte blanche, titre 30→40 px sans lettres qui se touchent, colonne de 720 px, « Pour poursuivre votre lecture » en liste de liens.
- **404** : `layouts/404.html`, en français, avec l'en-tête.
- **Mon enfance** : thèmes en lignes avec chevron, thème ouvert marqué d'un filet vert ; trois réponses sur une ligne à toutes les largeurs ; « Préciser » et « Liens étudiés » sur une ligne à gauche ; résultats en lignes ; onglets soulignés ; fenêtres centrées sur ordinateur, feuille basse sur mobile (la feuille de précision reste au-dessus du clavier) ; associations repliées sous leur titre dans « Liens étudiés » ; confirmations dans une fenêtre du site. Compteur « n / 7 renseignés » conservé.
- **Mes proches** : plus de carte enveloppante ni de sous-titre ; bouton d'ajout à côté du titre (ordinateur) ou pleine largeur (mobile) ; carte de proche entièrement cliquable (« Hélène · Mère ») ; formulaire à trois champs visibles, le reste sous « Ajouter des détails » ; barre d'action collante sur mobile.
- **Page d'un proche** : contenu sur le fond, filets, actions en boutons texte.
- **Ma ligne de vie** : repère d'année, titre, puis une ligne « septembre · Déclic · avec Julien, Sofia » sans répéter l'année ; une seule ligne d'actions ; « Sans date ».
- **Mes bilans** : introduction retirée, état vide raccourci.
- **Mon compte** : une colonne (Identité, Connexion, Accès à Ambre, Mes données, suppression) ; quota en deux lignes ; fenêtre de suppression avec bouton rouge.
- **Ambre** : conversation en pleine page (un seul défilement, saisie en bas, pied de page masqué), sans cadre ; écran « Avant de commencer » raccourci, lignes avec leur décompte (« 6 réponses sur 60 ») ; page déconnectée simple. Compteur « n / 100 messages » conservé.
- **Paramètres, Mon suivi** : style harmonisé (colonne, filets, chevrons, onglets sur une ligne), introductions retirées, anciens noms remplacés dans les libellés visibles. Contenu non réécrit.
- **Connexion** : aide du mot de passe à l'inscription seulement ; « Je suis psychologue » à la place de la liste « Type de compte » (même valeur envoyée).

### Fenêtres de confirmation
Tous les `confirm()` et l'`alert()` du Worker et de Mon enfance passent par une fenêtre du site (`confirmDialog`) : suppression d'un proche, d'un moment, d'une note, d'un bilan, de la mémoire, remplacement d'un brouillon, remplacement proposé par Ambre, déconnexion partout, abandon d'une saisie du plan, conflit et effacement dans Mon enfance.

### Textes modifiés notables
- « Enregistrer » / « Annuler » partout (au lieu de « Enregistrer mes informations », « Enregistrer ce proche », « Fermer le formulaire »…).
- Formulaire de proche : « Lien », « Qu’est-ce qui vous aide dans cette relation ? », « Qu’est-ce qui vous pèse ? », « Vous sentez-vous en sécurité avec cette personne ? ». Moment : « Titre », « Type », « Date », « Avec ».
- Plus de « (facultatif) », « — facultatif » ; « Non renseigné » devient « — ».
- Compte : « Accès gratuit. » ; « Il vous reste 100 messages et 1 bilan ce mois-ci, jusqu’au 1ᵉʳ décembre. » ; « 100 messages et 1 bilan par mois, bilan initial inclus. » ; « Se déconnecter partout » ; « Exporter » ; « Supprimer mon compte ».
- Ambre : « Avant de commencer », bouton « Commencer ». La phrase « assistante psychologue IA… ne pose pas de diagnostic et ne remplace pas un suivi professionnel » est inchangée, placée sous le bouton. Les textes de consentement (OpenRouter) sont inchangés.
- Mon enfance : « cartographie » remplacé par « vos réponses » dans les messages ; Mon suivi : « Mon enfance », « Mon compte », « Mon plan » ; Paramètres : « Mon compte ».

## Corrections après relecture indépendante

- Partage : la rubrique `portrait` s'appelle « Mon compte et la synthèse rédigée par Ambre » (Mon suivi et Paramètres), car elle partage le profil déclaré et la synthèse.
- « Effacer la conversation ? » : bouton rouge, « Annuler » à gauche. Annuler « Effacer cette mémoire » n'affiche plus « Enregistré. ».
- Contour des champs `--line-strong` : `#748693` (3,44:1 sur le fond `#F2F5F7`, 3,77:1 sur blanc).
- Cibles de 44 px : liens de la fenêtre de connexion, « Retrouver ce bilan… », lien du pied de page, tout résumé de dépliant.
- Onglets de Paramètres : passage à la ligne sur petit écran, plus d'onglet coupé. Conversation : plus de hauteur minimale, un seul défilement clavier ouvert.
- Après suppression d'un moment, le focus revient sur « Ajouter un moment ». Fenêtre de connexion : premier focus sur l'adresse e-mail ; « Je suis psychologue » ne signale son état que par son libellé.
- « Moments sans date précise » rétabli ; titre d'article en graisse 400 ; « 1ᵉʳ » ; dépliant « Comment mes choix sont-ils utilisés ? » sans lectures aimées, « J'aime » ni portrait (la phrase sur OpenRouter est inchangée).
- `/work/` (captures des suites navigateur) ajouté à `.gitignore`.

## Non fait, et pourquoi

- **M5** (formulaire de connexion directement dans la page) : la fenêtre de connexion est partagée par toutes les pages ; changement de structure à risque pour les parcours de connexion. La page déconnectée est simplifiée, la fenêtre harmonisée.
- **P3 complet** (formulaire de proche sur une page dédiée) : remplacé par le repli « Ajouter des détails » et la barre collante, sans changer le cycle des brouillons.
- **P5/L3** (Supprimer déplacé dans l'écran de modification) : laissé à sa place, en bouton texte rouge.
- **L4** (« Type » sans présélection) : l'API exige un type ; la présélection est gardée pour ne changer aucun comportement.
- **L5** (validation maison des champs) : bulles natives conservées.
- **A4** (24 personnages puis « Afficher plus ») : touche la recherche ; non fait.
- **S1/U2/U4** (réécriture de Paramètres et Mon suivi : étapes repliées, rythme des séances en trois choix, rubriques vides masquées, consentement raccourci) : hors demande (« sans réécrire leur contenu ») et texte de consentement à faire valider.
- **R4/C7** : les deux compteurs sont conservés à la demande du propriétaire.
- Code mort de l'ancien portrait et des articles aimés dans `ui.mjs` : non retiré (hors périmètre), ses styles ne sont plus fournis.

## Tests adaptés

- `journal/tests/ui-v2.test.mjs` : libellés, pied de page (désormais : un seul, réduit au nom du site), fenêtre de confirmation, carte de proche, quota.
- `tests/cartographie-*.cjs`, `tests/v4-C-*`, `tests/v5-C-*` : feuille commune chargée et cadre `#main-content` comme sur la vraie page ; trois réponses sur une ligne (non rognées, 44 px, distinctes) ; alignement à gauche commun à la question, aux réponses et au bouton de précision ; confirmations dans la fenêtre du site, tout dialogue natif devient une erreur ; associations repliées.
- `tests/v5-A5-P2-browser.cjs`, `tests/personal-loading-browser.cjs`, `tests/change-plan-browser.cjs` : libellés, « Ajouter des détails », « Je suis psychologue ». `change-plan-browser.cjs` visait `/mon-portrait/`, adresse que son banc ne sert pas (échec antérieur à ce lot) : il vise `/mon-compte/`.
- `tests/menu-ambre.test.cjs`, `tests/character-search-ui.test.cjs` : noms des exports partagés, « 2 personnages ».

## Vérifications

`npm --prefix journal test` 847/847 ; `node --test tests/*.test.cjs` 49/49 avec `HUGO_BIN` (48 + 1 ignoré sans) ; les sept suites navigateur de `tests/` ; tous les `--check` et tests Python de `.github/workflows/verify.yml`. Recette visuelle Chromium à 390×844 et 1440×900 (vrai Worker, SQLite en mémoire ; site Hugo 0.140.0 construit hors dépôt) : aucun débordement horizontal, aucune erreur JavaScript.

## Déploiement à prévoir

Worker complet et Netlify ensemble : les deux embarquent la même feuille commune. Pas d'ordre imposé pour les données.

## 18 novembre 2026 — Corrections après retour du propriétaire (« c'était presque mieux avant »)

Sobriété conservée (ni ombre décorative, ni dégradé, ni orange) ; la structure revient là où elle aide à lire. Aucun changement d'API, de données, de routes, de consentements ni de quotas.

1. **Ambre avant le lancement** : la hauteur fixe et le défilement interne ne s'appliquent plus qu'à une conversation commencée (`.conversation-page:has(.chat-thread:not([hidden]))`). « Avant de commencer » défile avec la page, pied de page visible, sans conteneur à défilement interne (vérifié à 390 tactile, 1280×700 et 1440×900). Conversation lancée : une seule zone qui défile (la page ne défile pas), barre fine et discrète (`scrollbar-width: thin`, 6 px sous WebKit), écartée du texte.
2. **Mon enfance, « Ce qui en ressort »** : chaque résultat est une carte (fond blanc, filet, coins de 10 px, marge intérieure, 12 px entre cartes). Titre, phrase, « Relié à : … », puis une rangée séparée par un filet : « Je me reconnais » / « Pas moi » à gauche, « Voir le détail » à droite (mobile : les deux réponses à parts égales, le détail dessous). Réponse choisie : bouton plein avec ✓, et la carte reconnue prend une bordure verte. Détail séparé par un filet, associations sous un filet gauche. « Écartés » : même forme de carte repliable avec chevron, sujets en lignes avec chevron ; le titre n'est plus répété quand une ligne est ouverte. « Décrire comment cela se manifeste aujourd'hui » aligné à gauche. Colonne de résultats sur ordinateur : barre de défilement fine.
3. **Mes proches** : `-webkit-tap-highlight-color: transparent` sur tout le site (`html` et tous les éléments, feuille commune). La bordure verte des cartes (proches, personnages, articles) ne suit plus `:focus-within` mais `:has(:focus-visible)`, donc jamais au toucher ni au clic ; le survol est réservé aux appareils qui survolent (`@media (hover: hover)`), y compris pour les boutons, le menu et la croix des fenêtres (fin des survols « collés » après un toucher, qui coloraient notamment « Modifier » en vert foncé). Les titres focalisés par script (`tabindex="-1"`) ne dessinent pas d'anneau.
4. **Page d'un proche** : en-tête inchangé (retour, prénom, rôle) ; « Informations » dans une carte, Modifier / Supprimer sous un filet, le formulaire de modification s'ouvre dans la carte (barre d'action collante blanche sur mobile). « Votre histoire avec … » devient une frise verticale (trait, point par étape, date en petit au-dessus) : la rencontre d'abord (« Date non précisée » quand elle manque), chaque événement lié (titre vers l'événement, texte, ses notes sous un filet, petit lien « Ajouter une note à ce moment »), les notes (« Votre note · 18 nov. 2026 », point creux) et les repères (« Repère validé · 17 nov. 2026 », point creux). Un seul bouton « + Ajouter une note », bordé, sous la frise. « Voir les événements liés » retiré (les titres mènent déjà à l'événement). Ordre des éléments : celui du serveur, inchangé.
5. **Focus à l'ouverture** : nouvel utilitaire `focusEntry(champ, titre)` dans `ui.mjs` : un champ ne reçoit le focus que si `matchMedia('(hover: hover) and (pointer: fine)')` ; sinon le titre (`tabindex="-1"`, `preventScroll`). Appliqué à l'ajout et à la modification d'un moment et d'un proche (y compris sur la page d'un proche), à la fenêtre de connexion, à la suppression du compte (le champ mot de passe n'est plus focalisé par `showModal` sur mobile), aux fenêtres du plan (Mon suivi, Ambre). Mon enfance : la fenêtre de précision donne le focus au champ sur ordinateur, au titre sur écran tactile. Les dépliants de note restent sans focus automatique.
6. **Pied de page** : filet `1px solid var(--line)` en haut, marge intérieure de 16 px, espacement du contenu au-dessus inchangé (48/64 px). Même feuille commune côté Hugo et Worker ; il reste en bas des pages courtes (404, Mes proches, Mes bilans vérifiés).

### Autres régressions corrigées (point 7)
- **Ma ligne de vie** : chaque moment redevient une carte posée sur la frise des années ; notes sous un filet gauche à l'intérieur de la carte ; actions séparées du récit par un filet ; repère d'année moins collé à sa carte.
- **Mon suivi** : « Dernière séance… » n'est plus collé au bouton « Enregistrer mon suivi ».
- **Paramètres** : les phrases qui suivent un dépliant ou un bouton (« Les consignes sont rapportées par vous… », « Les invitations restent limitées… ») ne sont plus collées.
- Revues sans changement : accueil, fiche, article, 404, Mes bilans, Mon compte, fenêtres de confirmation, menu mobile, onglet « Mon enfance » (réponses).

### Tests adaptés
- `journal/tests/ui-v2.test.mjs` : chargement de `finePointer`, `focusEntry`, `noteDay` ; libellés de la frise (« Repère validé », « Votre note », « Ajouter une note à ce moment »).
- `journal/tests/change-interface.test.mjs` : `focusEntry` fourni au contexte.
- `tests/cartographie-browser.cjs` : sur ordinateur le champ de précision reçoit le focus ; nouveau cas tactile (390, `hasTouch`) où le titre reçoit le focus ; l'espace de lecture se mesure jusqu'aux boutons (filet ajouté).

### Vérifications
`npm --prefix journal test` 847/847 ; `node --test tests/*.test.cjs` 49/49 avec `HUGO_BIN` ; les sept suites navigateur de `tests/` ; tous les `--check` et tests Python de `.github/workflows/verify.yml`. Bancs hors dépôt (vrai Worker + SQLite, Hugo 0.140.0) à 390×844 tactile, 1280×700 et 1440×900 avec 3 proches, 3 moments notés, 2 notes et 1 repère validé, réponses dans Mon enfance : aucun conteneur à défilement interne avant lancement, aucune `activeElement` de type champ après ouverture en tactile, `border-top` du pied de page, surbrillance au toucher transparente, aucun débordement horizontal ni erreur JavaScript. Remarque de banc : une capture pleine page de Playwright désactive l'émulation tactile jusqu'au rechargement ; le banc recharge la page après chaque capture pleine page.

## 19 novembre 2026 — Arrivée sur Ambre, prérequis en étapes, Mon enfance allégée, retour tactile, « Ajouter un moment »

Non commité par ce lot, non déployé. Aucun changement d'API, de données, de modèle des moments, de consentements, de textes de sécurité ni de quotas ; `editorial/symptomes/` non touché.

1. **Arriver sur Ambre** : `netlify.toml` redirige `/` vers `/mon-accompagnement/` (302, `force = true`, sinon Netlify sert l'`index.html` d'Hugo). Le catalogue reste à `/personnages/` (aucune règle ne le touche). Le logo de l'en-tête et le nom du site en pied de page mènent directement à `/mon-accompagnement/` (pas de saut par la redirection) ; le pied de page ajoute un lien « Personnages ». Le fil d'Ariane menait déjà à `/personnages/` ; « Explorer les lectures » (articles aimés, liste vide) mène désormais à `/personnages/` au lieu de `/`. Aucun lien interne ne pointe plus vers `/`.
2. **Ambre avant le lancement** : titre « Ambre » agrandi avec son rôle, « Avant de commencer » et la phrase de présentation, puis une carte « Vos informations · n sur 4 complétées » (petite jauge de quatre traits sur ordinateur) avec quatre étapes numérotées dans une liste ordonnée : pastille numérotée (à compléter) ou cochée (fait), nom, détail, état « À compléter » / « Renseigné », accès direct (« Compléter » bordé ou « Modifier »), toute la ligne cliquable avec retour au toucher. « Commencer » plus grand (pleine largeur sur mobile), avec la raison juste dessous tant qu'il est désactivé : « Il reste à compléter : … » (plus la phrase sur « Je ne sais pas » si Mon enfance manque). La mention « assistante psychologue IA… ne pose pas de diagnostic… » est inchangée, en petit sous un filet. Conversation lancée et consentement inchangés.
3. **Mon enfance** : questions sans cadre (plus de filet gauche vert du thème ouvert ; filets très clairs entre questions), réponses Non / Oui / Je ne sais pas en contrôle segmenté fin (un seul contour, séparateurs d'un pixel, choix : teinte claire, ✓, graisse et anneau intérieur ; cibles de 44 px conservées), thèmes repliables plus fins (titre en vert et chevron tourné quand ouvert, description et compteur « n / 7 renseignés » plus petits sur mobile). « Ce qui en ressort » garde ses cartes, plus légères sur mobile (marges réduites, titre 17 px, boutons « Je me reconnais » / « Pas moi » à filet clair, choix en teinte claire avec ✓ au lieu du bouton plein). Comportements, trois réponses, absence de niveaux de preuve et de références conservés.
4. **Mes proches sur ordinateur** : à partir de 900 px, la liste occupe toute la largeur de contenu (1040 px) en grille `auto-fill, minmax(300px, 1fr)` : trois colonnes à 1280 et 1440 px, « Ajouter un proche » aligné sur le bord droit de la grille. Chaque carte porte l'initiale du proche dans une pastille, la relation sur deux lignes au plus. Le formulaire d'ajout ouvert reste à la largeur de lecture.
5. **Retour tactile** : `:active` sur les boutons (léger assombrissement ou fond clair, réduction à 98 %), liens textuels (opacité), cartes des proches, des personnages et des articles, étapes d'Ambre, thèmes et repliables de Mon enfance, liens du menu. Safari n'applique `:active` qu'avec un écouteur `touchstart` : ajouté (passif) dans `static/js/site-navigation.js`. Navigation interne de l'espace personnel : la carte touchée garde un fond clair (`data-pending`) et, au-delà de 180 ms d'attente, une barre fine sous l'en-tête et une petite roue à la place du chevron (`html.is-navigating`), retirées à l'arrivée. Surbrillance bleue toujours transparente.
6. **« Ajouter un moment »** sur la page d'un proche, à côté de « Ajouter une note » (pleine largeur l'un sous l'autre sur mobile). Il ouvre `/ma-ligne-de-vie/?ajouter-avec=<id>` : formulaire de moment ouvert, ce proche coché (fusionné dans un brouillon de nouveau moment existant ; un moment en cours de modification n'est jamais remplacé), lien « Retour à … ». L'enregistrement passe par `POST /api/life-events` existant puis revient sur la page du proche, où la frise recharge et montre le moment. « Annuler » revient aussi sur la page du proche en gardant le texte saisi en brouillon.

### Tests
- Adaptés : `journal/tests/ui-v2.test.mjs` (pied de page avec lien Personnages, fonctions `navigationPending`, `prepareMomentForPerson`, `momentReturn` chargées), `tests/menu-ambre.test.cjs` et `tests/legacy-private-pages.test.cjs` (la page d'accueil publique est remplacée par la redirection 302 voulue), `tests/v5-A5-P2-browser.cjs` (le formulaire de note générale est dans `.person-timeline-adds`).
- Ajoutés : redirection `/` → Ambre (302, forcée, unique, catalogue non redirigé, logo et pied de page, plus aucun lien vers `/`) ; bouton « Ajouter un moment » sur la page d'un proche ; ouverture avec le proche coché, enregistrement par l'API et retour ; annulation, fusion de brouillon, moment en cours de modification et proche inconnu laissés intacts, aucun retour sans le paramètre.

### Vérifications
`npm --prefix journal test` 850/850 ; `node --test tests/*.test.cjs` 50/50 avec `HUGO_BIN` ; les sept suites navigateur de `tests/` ; tous les `--check` et tests Python de `.github/workflows/verify.yml`. Bancs hors dépôt (vrai Worker + SQLite, Hugo 0.140.0) à 390×844 tactile, 1280×700 et 1440×900 : 60 contrôles Worker et 15 publics réussis, aucune erreur JavaScript ni débordement horizontal. Limite de banc : Chromium sans écran n'applique pas `:active` aux touchers émulés ; l'état pressé est mesuré avec un appui de pointeur sur la page mobile, le toucher lui-même (pas de surbrillance, pas d'anneau) étant vérifié séparément.
