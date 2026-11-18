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
- Compte : « Accès gratuit. » ; « Il vous reste 100 messages et 1 bilan ce mois-ci, jusqu’au 1er décembre. » ; « 100 messages et 1 bilan par mois, bilan initial inclus. » ; « Se déconnecter partout » ; « Exporter » ; « Supprimer mon compte ».
- Ambre : « Avant de commencer », bouton « Commencer ». La phrase « assistante psychologue IA… ne pose pas de diagnostic et ne remplace pas un suivi professionnel » est inchangée, placée sous le bouton. Les textes de consentement (OpenRouter) sont inchangés.
- Mon enfance : « cartographie » remplacé par « vos réponses » dans les messages ; Mon suivi : « Mon enfance », « Mon compte », « Mon plan » ; Paramètres : « Mon compte ».

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
