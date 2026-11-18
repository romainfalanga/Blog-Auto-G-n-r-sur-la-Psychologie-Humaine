# Audit design et expérience — Décode ton esprit (18 novembre 2026)

Lecture seule : aucun fichier du dépôt modifié, rien de commité.
Captures : `shots/` (138 fichiers, préfixe `m-` = 390×844, `d-` = 1440×900, `-out-` = déconnecté, `-full` = page entière, `-fold` = premier écran). Montages de relecture : `crops/`. Mesures calculées (tailles, rayons, largeurs) : `metrics.log`. Tous ces chemins sont relatifs à
`/tmp/claude-0/-home-user-Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/0d730b71-6a7b-5067-a403-c056872fe890/scratchpad/design-audit/`.

## 0. Méthode et limites

- Compte synthétique `test-design-…@example.com` créé en production, rempli par l'API (identité, 3 proches, 3 moments) et par l'interface (6 réponses de Mon enfance). Aucun message envoyé à Ambre, aucun bilan lancé ; les écritures vers `/api/chat`, `/api/reviews`, `/api/plan`, `/api/care` étaient bloquées dans le navigateur de test (aucune n'a été tentée).
- Compte supprimé à la fin : `DELETE /api/account` → 200 `{"ok":true}` ; ancienne session → `user:null` ; nouvelle connexion → 401. Identifiants et session locaux effacés.
- Pages capturées aux deux tailles : accueil, `/personnages/`, fiche personnage, article, Mon enfance (vide, thème ouvert, répondu, résultats, fenêtre de précision, fenêtre « Liens étudiés »), Mes proches (liste, détails ouverts, formulaire, modification, suppression), Ma ligne de vie (liste, note, formulaire, erreur), Mes bilans (vide), Mon compte (connecté, détails, suppression, déconnecté, connexion, erreur, inscription), Ambre (prérequis, déconnecté), Paramètres (3 onglets), Mon suivi (page, fenêtre d'objectif), menu mobile, recherche vide et avec résultats, 404, focus clavier.
- **Non vu** : conversation Ambre avec des messages, bilan rempli, espace psychologue, paiement. Les recommandations sur la conversation portent donc sur l'enveloppe, pas sur les bulles de message.
- **Artefacts du navigateur de test à ne pas prendre pour des défauts** : dates affichées au format américain (`04/12/1990`) et bulle de validation en anglais (« Please fill out this field ») viennent de Chromium sans langue système ; la police `system-ui` est rendue par une police Linux. Ce que cela révèle reste vrai : le site dépend de contrôles natifs (date, select, validation, `confirm()`) dont l'aspect change d'un appareil à l'autre.
- Hors périmètre (autre agent) : bulle flottante Ambre, sous-page par proche, état vide des bilans. Les corrections ci-dessous supposent qu'Ambre revient dans le menu (6 entrées) et que la liste des proches devient une liste de liens.

---

## 1. Diagnostic : ce qui fait « généré par IA », concrètement

Le site n'est pas laid, il est **homogène par défaut et incohérent dans le détail** : chaque écran a été produit séparément avec les mêmes réflexes (une carte blanche arrondie, une pastille, un texte d'explication, un bouton vert), sans système commun. C'est exactement la signature qu'on reconnaît.

### 1.1 Chiffres relevés dans les sources

| Mesure | Valeur | Où |
|---|---|---|
| Tailles de police distinctes | **99** | `static/css/style.css` (47), `journal/src/workspace-style.mjs` (25), `journal/src/ui.mjs` (22), `change-interface.mjs` (5) |
| Rayons d'angle distincts | **37** (0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 18, 20, 24, 999 px…) | mêmes fichiers |
| Couleurs hexadécimales distinctes | 67 dans `style.css`, 68 dans `ui.mjs` | — |
| Jeux de variables parallèles | 2 : `--color-text/--color-accent/…` (Hugo) et `--ink/--accent/--wash/…` (Worker), mêmes teintes, noms différents | `style.css:13`, `workspace-style.mjs:2` |
| Largeurs de page | 1160 (accueil, Mon enfance), 1040 (pages Worker), 900 (fiche), 880 (article), 860 (cartes de Mon suivi) | `metrics.log` |
| Taille du titre de page (ordinateur) | 29,6 px gras (accueil) · 58,4 px maigre (pages privées) · 48,8 px gras (fiche) · 54,4 px gras (article) · 17 px sans empattement (Ambre) · 32 px sans empattement gras (Mon compte déconnecté) | `metrics.log` |
| Texte courant | 17,6 px/30,8 (Hugo) · 16 px/24,8 (Worker) · 18,4 px/33 (article) | `metrics.log` |
| Marge latérale mobile | 16 px (Hugo) contre 15 px (Worker) | `metrics.log` |

### 1.2 Les signes visibles, avec capture et sélecteur

1. **Des cartes dans des cartes dans des cartes.** Mes proches : `section.panel` (rayon 16, ombre) → `.source-item` (rayon 12) → formulaire `.source-editor` (fond gris, rayon 12). Mon enfance : `.carto-form-body` (rayon 14) → `.carto-theme-panel` (rayon 12) → `.carto-choices` (rayon 15) → `.carto-see` (pilule 999). Sur mobile il reste ~296 px de texte utile sur 390. — `m-proches-full.png`, `m-enfance-full.png`, `crops/m-forms.png`.
2. **Titre de page géant, maigre, à l'interlettrage très serré**, suivi d'un sous-titre gras sans empattement qui répète le titre : « Mes proches » → « Votre répertoire », « Ma ligne de vie » → « Vos moments ». `h1` Worker : 58,4 px, graisse 500 (Georgia n'a pas de 500 : rendu en 400), `letter-spacing:-.045em` (`workspace-style.mjs:25`), `.carto-intro h1` (`style.css:702`). L'accueil, lui, a un `h1` de 29,6 px gras : deux sites différents. — `d-proches-full.png`, `d-home-fold.png`.
3. **Pastilles partout** : rôle du proche (`.source-tag` « Mère », « Meilleur ami ou meilleure amie »), type de moment (« Déclic », « Moment positif »), proches liés (pastilles soulignées), état des prérequis (« Renseigné », « À compléter »), thèmes des fiches, « Liens étudiés (40) », « Je me reconnais / Pas moi ». Six formes de pastille différentes (rayon 6, 7, 999 ; fonds gris, vert pâle, beige, blanc bordé). — `d-ligne-full.png`, `d-ambre-full.png`, `crops/m-rest.png`.
4. **Trois façons d'ouvrir/fermer** sur une même carte : triangle natif `▶` de `<details>` (« Votre histoire avec Hélène », « Ajouter une note »), `+`/`−` textuel (« Repères sur cette relation », thèmes de Mon enfance, Paramètres), `×` (thème ouvert). Et quatre « plus » différents : `+` du bouton vert, `+` de « Repères », `+ Ajouter une précision », `+` à droite des thèmes. — `m-proches-details.png`, `crops/d-rest2.png`.
5. **Au moins sept styles de bouton** : plein vert rayon 8 ; secondaire blanc bordé rayon 8 (51 px de haut contre 49 pour le primaire) ; texte vert gras ; texte rouge gras ; lien souligné vert ; pilule blanche (`.carto-see`, `.carto-btn`) ; pilule grise (`.carto-fb`) ; segment (`.carto-choice`) ; onglet pilule (`.carto-tabs button`) ; onglet souligné (`button.settings-tab`). « Enregistrer » est vert plein dans Mon compte, **pilule blanche** dans la fenêtre de précision de Mon enfance, blanc bordé pour « Enregistrer la note ». — `m-enfance-note-dialog.png`, `crops/d-rest.png`, `d-compte-full.png`.
6. **Trois fenêtres modales différentes** : centrée (connexion, suppression de compte, objectif), feuille basse (Liens étudiés sur mobile), feuille haute (précision sur mobile). **Sur ordinateur, les deux fenêtres de Mon enfance s'ouvrent collées en haut à gauche de l'écran** : la remise à zéro `*{margin:0}` (`style.css:6`) annule le `margin:auto` natif de `<dialog>` et `.carto-dialog` (`style.css:811`) ne le rétablit pas. — `d-enfance-note-dialog.png`, `d-enfance-liens.png`, `crops/d-dialogs.png`.
7. **Textes d'aide et précautions à chaque écran**, ton de notice : « Accès gratuit pendant la phase de découverte. 100 messages et 1 bilan par mois civil, selon l'heure de Paris. Le bilan initial est inclus. Il vous reste… Renouvellement le 1 décembre 2026 (heure de Paris). » (4 paragraphes pour une information) ; « Proches associés (facultatif, 10 maximum) » ; « (facultatif) » répété 8 fois dans un seul formulaire ; « Non renseigné » comme valeur visible ; « Intervalle habituel entre les séances, en jours » ; « 37 source(s) chargée(s) sur 37. » — `d-compte-full.png`, `crops/m-forms.png`, `crops/m-suivi-param.png`.
8. **Voix instable** : « Mes proches » / « Votre répertoire » / « Comment vivez-vous cette relation ? » / « Ce qui m'aide dans cette relation » / « Je me sens en sécurité… » dans le même formulaire ; nom du site au tutoiement. Vocabulaire d'un autre âge du produit encore visible : « Cartographie », « Portrait », « Plan de changement » (cases de partage de Mon suivi), « Modifier ma présentation », « Mon portrait disponible pour Ambre » (Paramètres). — `crops/m-suivi-param.png`.
9. **Pages système non habillées** : la 404 est la page par défaut de Netlify, en anglais, sans en-tête (« Page not found… If this is your site… ») ; la suppression d'un proche ou d'un moment passe par un `confirm()` du navigateur alors que la suppression du compte a une vraie fenêtre ; déconnecté, `/mon-compte/` et `/mon-accompagnement/` affichent une carte presque vide. — `d-out-404.png`, `d-out-ambre.png`, `m-out-login.png`.
10. **Mobile traité comme un ordinateur rétréci** : en-tête avec le logo seul, flottant au centre gauche, sans nom du site (`.site-wordmark` masqué) ; conversation Ambre enfermée dans une carte à défilement interne (`.conversation-scroll` 680 px visibles pour 992) ; formulaire de proche de 11 champs déroulé dans la liste (2 400 px) avec le bouton « Enregistrer » tout en bas ; 60 questions empilant chacune 5 blocs (titre, exemple, segment sur 2 lignes, pilule, lien). — `m-ambre-fold.png`, `m-proches-form.png`, `crops/m-enfance-a.png`.
11. **Hiérarchie plate à l'intérieur des cartes** : dans une carte de moment, l'année apparaît deux fois (repère « 2008 » puis « 2008 » dans la carte), la date, la pastille, le titre, « Avec », « Ajouter une note », « Modifier », « Supprimer » ont tous à peu près le même poids ; 60 à 90 px de vide autour de « Ajouter une note ». — `d-ligne-full.png`.
12. **Mélange de polices sans règle** : dans Mon enfance, les questions sont en sans empattement gras et les résultats en Georgia gras (`.carto-card-title`), dans la même page ; le titre de la fenêtre est en Georgia, son contenu en sans. — `crops/d-rest2.png`, `m-enfance-liens.png`.
13. **Pied de page incohérent** : présent sur les pages Hugo (bande grise « Décode ton esprit »), absent sur toutes les pages Worker ; sur les pages courtes il remonte au milieu de l'écran, suivi d'une bande vide (`main{min-height:60vh}`, `style.css:560`, pas de colonne flexible). — `d-enfance-full.png`, `crops/d-misc.png` (recherche vide).
14. **Alignements qui sautent** : passer de Mon enfance (contenu à x=140, largeur 1160) à Mes proches (x=200, 1040) décale le titre de 60 px ; dans Mon suivi, trois largeurs sur la même page (titre 1040, cartes 860 centrées, « Mon plan » plus large que les autres). — `d-suivi-full.png`.
15. **Restes de gabarit** : `--gradient-accent`, `--color-cta` orange, `--shadow-md/lg`, `@keyframes slideDown`, `transition: all .25s` dans `style.css` ; ombre `0 2px 4px #172d3c07` sur les seules cartes Worker. Peu visibles aujourd'hui, mais c'est la réserve d'où ressortent les effets « gabarit ».
16. **Paramètres et Mon suivi** : pages hors menu, très longues (3 100 à 5 300 px sur mobile), jargon technique en clair (« openai/gpt-5-nano », « 6 entrées disponibles dans l'index », « OpenRouter », « provenance vérifiable »), accordéons dans des accordéons. — `d-parametres-full.png`, `crops/m-suivi-param.png`.

### 1.3 Ce qui est déjà bien (à garder)

Palette de base sobre (encre, gris-bleu, vert profond) ; contrastes du texte corrects ; focus clavier visible et cohérent (`d-focus-*.png`) ; cibles principales à 44 px ; pas de débordement horizontal à 390 px ; menu mobile simple et lisible (`m-menu-open.png`) ; texte des articles confortable ; aucune animation gênante.

---

## 2. Direction de design proposée : « carnet clinique »

Une seule idée : **du papier, pas un tableau de bord.** Une page = un titre, un contenu aligné à gauche sur une seule colonne, des filets fins pour séparer, très peu de boîtes. Le vert ne sert qu'à l'action. Tout ce qui n'aide pas à écrire ou à relire disparaît.

### 2.1 Jetons (une seule source, à partager entre Hugo et le Worker)

```css
:root{
  /* Couleurs : 9, pas une de plus */
  --ink:#172D3C;        /* texte */
  --muted:#526575;      /* texte secondaire (≥ 5:1 sur --bg et sur blanc) */
  --bg:#F2F5F7;         /* fond de page */
  --surface:#FFFFFF;    /* champs, fenêtres, cartes cliquables */
  --line:#D6E0E6;       /* filets de séparation */
  --line-strong:#7F919E;/* contour des champs et boutons secondaires (≥ 3:1) */
  --accent:#205E54;     /* action, lien, sélection */
  --accent-wash:#E8F2EF;/* sélection douce, survol */
  --danger:#962B40;     /* suppression, erreur */
  /* Typographie */
  --font-title:Georgia,"Times New Roman",serif;
  --font-ui:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --t-xs:13px; --t-sm:14px; --t-md:16px; --t-lg:18px; --t-xl:22px; --t-2xl:28px; --t-3xl:36px;
  /* Espacements */
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:24px; --s6:32px; --s7:48px; --s8:64px;
  /* Formes */
  --r-control:6px; --r-card:10px;
  --page:720px;         /* colonne de lecture et de saisie */
  --page-wide:1040px;   /* listes en grille (accueil, Mon enfance à deux colonnes) */
  --gutter:16px;        /* 24px à partir de 600px */
}
```

À supprimer : `--color-cta`, `--gradient-accent`, `--shadow-sm/md/lg`, `--color-exercise`, `--color-pro-reminder`, et l'un des deux jeux de noms (garder les noms courts du Worker, faire pointer les anciens `--color-*` dessus le temps de la migration).

### 2.2 Échelle typographique

| Rôle | Police | Taille mobile → ordinateur | Graisse | Interligne | Interlettrage |
|---|---|---|---|---|---|
| Titre de page (`h1`, partout) | Georgia | 28 → 36 px | 400 | 1,15 | −0,01em |
| Titre d'article | Georgia | 30 → 40 px | 700 | 1,15 | −0,01em |
| Titre de section (`h2`) | système | 18 → 20 px | 600 | 1,3 | 0 |
| Titre d'élément (`h3`, nom d'un proche, d'un moment, question) | système | 16 px | 600 | 1,35 | 0 |
| Texte courant, champs | système | 16 px | 400 | 1,6 | 0 |
| Texte d'article | système | 17 → 18 px | 400 | 1,7 | 0 |
| Texte secondaire, libellés de champ | système | 14 px | 400 (libellé : 600) | 1,45 | 0 |
| Mention (date, compteur de caractères) | système | 13 px | 400 | 1,4 | 0 |

Sept tailles au lieu de 99. Georgia uniquement pour le titre de page et les titres d'article/de fiche : plus de Georgia dans les cartes de résultats ni dans les fenêtres. Aucune taille sous 13 px. Longueur de ligne : 60 à 72 caractères (`max-width:68ch` sur les paragraphes).

### 2.3 Surfaces, bordures, rayons

- **Pas d'ombre**, sauf la fenêtre modale (`0 8px 24px rgb(23 45 60 / .12)`).
- **Un seul niveau de boîte.** Le contenu d'une page repose directement sur le fond ; une liste est une suite de lignes séparées par `border-top:1px solid var(--line)`. Une carte blanche (`--surface`, 1 px `--line`, rayon 10) n'existe que si elle est **entièrement cliquable** (personnage, proche) ou si c'est une fenêtre.
- Deux rayons : 6 px (boutons, champs, pastilles) et 10 px (cartes, fenêtres). Plus de pilules 999.
- Pastilles : une seule forme (13 px, `--muted`, sans fond, précédée d'un point médian) — en pratique du texte secondaire : « Hélène · Mère », « septembre 2020 · Déclic ». Fond coloré réservé à un seul cas : l'état « À compléter ».

### 2.4 Boutons, liens, champs

| Élément | Style |
|---|---|
| Bouton primaire (un seul par écran) | fond `--accent`, texte blanc, 16 px/600, hauteur 44 px, `padding:0 18px`, rayon 6 |
| Bouton secondaire | fond `--surface`, 1 px `--line-strong`, texte `--ink`, mêmes dimensions |
| Bouton texte | texte `--accent` 600, sans fond, `min-height:44px`, `padding:0 4px`, souligné au survol |
| Action destructrice | bouton texte en `--danger` ; devient primaire rouge uniquement dans la fenêtre de confirmation |
| Lien dans un texte | `--accent`, souligné (`text-underline-offset:3px`) |
| Champ, zone de texte, liste déroulante | fond `--surface`, 1 px `--line-strong`, rayon 6, hauteur 48 px, 16 px (évite le zoom iOS), `padding:0 12px` ; libellé 14 px/600 au-dessus à 6 px ; aide 13 px `--muted` en dessous ; erreur 13 px `--danger` en dessous + contour `--danger` |
| Choix exclusif court (Non / Oui / Je ne sais pas) | trois boutons secondaires égaux sur une ligne, hauteur 44 px ; choisi = fond `--accent`, texte blanc |
| Dépliant | un seul motif : chevron `›` à droite qui pivote, jamais `▶`, `+`, `−`, `×` |
| Focus | `outline:2px solid var(--accent); outline-offset:2px` partout (déjà presque le cas) |

### 2.5 Ton des libellés

1. Une seule voix : les titres de navigation restent à la première personne (« Mes proches ») ; **tout le reste vouvoie** (« Comment vivez-vous cette relation ? », « Qu'est-ce qui vous aide ? »). Plus de « je » dans les libellés de champ.
2. Un verbe, deux mots : « Enregistrer », « Annuler », « Modifier », « Supprimer », « Ajouter ». Pas de « Enregistrer mes informations », « Fermer le formulaire », « Supprimer définitivement mon compte et mes données ».
3. Un champ facultatif ne dit rien ; seul un champ obligatoire le signale. Supprimer tous les « (facultatif) », « — facultatif », « 10 maximum ».
4. Aucune valeur « Non renseigné » visible : option vide « — ».
5. Aucun terme technique hors Paramètres (modèle, index, source chargée, heure de Paris, mois civil, traces).
6. Un état vide = une phrase + une action. Une précaution clinique = dite une fois, là où elle sert.
7. Un seul nom par chose : « Mon enfance » (plus « Cartographie »), « Mon compte » (plus « Portrait »), « Mon plan » (plus « Plan de changement »).

### 2.6 Mobile

- En-tête 56 px : logo **et** nom du site à gauche, bouton menu 44×44 à droite. Menu : 6 entrées de 48 px (Ambre incluse), page courante marquée par un trait vertical vert, pas par un pavé gris.
- Marge latérale 16 px identique partout ; contenu directement sur le fond (pas de carte enveloppante qui mange 2×18 px).
- Fenêtres : toujours une **feuille basse** pleine largeur (rayon 10 en haut, `max-height:88dvh`, zone de boutons collante en bas avec `padding-bottom:env(safe-area-inset-bottom)`) ; centrées sur ordinateur (`max-width:520px`).
- Formulaires longs : barre d'action collante en bas (`position:sticky; bottom:0`, fond `--bg`, filet supérieur) avec Enregistrer + Annuler.
- Hauteurs en `dvh`, jamais `vh` ; `scroll-margin-top:72px` sur les ancres ; `inputmode`/`autocomplete` renseignés ; pas de défilement dans un défilement.
- Conversation : occupe tout l'écran sous l'en-tête, la zone de saisie est le seul élément fixe.

### 2.7 Mouvement et accessibilité

- Transitions : couleur et fond 120 ms ; ouverture de fenêtre/feuille 160 ms (opacité + 8 px). Rien d'autre. `scroll-behavior:smooth` retiré.
- `@media (prefers-reduced-motion:reduce){*,*::before,*::after{transition:none!important;animation:none!important}}` dans **les deux** feuilles (aujourd'hui seulement partiel dans `style.css:516`).
- Contrastes AA : texte ≥ 4,5:1 (les deux gris actuels passent) ; contours de champ ≥ 3:1 (le `#D6E0E6` actuel sur blanc est autour de 1,3:1 → `--line-strong`) ; bouton désactivé lisible (`Lancer la discussion` grisé : garder un contour et expliquer pourquoi au-dessus).
- Un seul `h1` par page, identique en forme ; ordre des titres sans saut ; `aria-current="page"` dans le menu ; erreurs reliées au champ par `aria-describedby`, en français, sans bulle native.

---

## 3. Changements concrets, par page et par priorité

Effort : S (< 1 h), M (demi-journée), L (un jour ou plus). Risque : faible / moyen / élevé (régression fonctionnelle ou tests à adapter). Priorité : P1 = à faire d'abord, le gain « ne fait plus IA » est le plus fort.

### 3.0 Transversal

**Visuel (CSS)**

| # | P | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|
| T1 | P1 | `d-proches-full`, `d-home-fold`, `d-out-article-fold` | Titres de page de 5 formes différentes | Une règle commune : `h1{font:400 clamp(28px,4vw,36px)/1.15 var(--font-title);letter-spacing:-.01em;margin:0 0 24px}` dans `workspace-style.mjs:22-26`, `.carto-intro h1` (`style.css:702`), `.character-directory-heading :is(h1,h2)` (`style.css:575`), `.perso-profile-name` (`:620`) ; article : même règle en 700, 30→40 px (`:657`) | S | faible |
| T2 | P1 | `d-enfance-full` → `d-proches-full` | Saut de 60 px entre pages (1160 / 1040 / 860 / 880 / 900) | `--app-width` et `--max-width-wide` → 1040 px ; pages de saisie (Proches, Ligne de vie, Compte, Bilans, Suivi, Paramètres) en colonne de 720 px **alignée à gauche de la grille de 1040**, pas centrée ; `.care-section` (860 centré) supprimé | M | faible |
| T3 | P1 | toutes | 99 tailles, 37 rayons, 2 jeux de variables | Poser les jetons du § 2.1 en tête de `style.css` et de `WORKSPACE_STYLE`, faire pointer `--color-*` vers les nouveaux noms, puis remplacer taille par taille (viser les 7 tailles, 2 rayons) | L | moyen (visuel seulement, mais large) |
| T4 | P1 | `d-proches-full`, `d-ambre-full` | Ombre sur les seules cartes Worker ; rayons 12/14/16 | `.panel,.conversation{box-shadow:none;border-radius:var(--r-card)}` (`workspace-style.mjs:28,185`) | S | faible |
| T5 | P2 | `d-compte-full` | Bouton secondaire plus haut (51) que le primaire (49) ; 7 styles | Table du § 2.4 : `button,.button{min-height:44px;padding:0 18px;border-radius:6px;font:600 16px/1 var(--font-ui)}` ; `.secondary` bordure `--line-strong` ; retirer `.carto-btn`, `.carto-see`, `.carto-fb`, `.carto-tabs button` en pilule | M | faible |
| T6 | P2 | `crops/d-misc` (recherche vide), `d-enfance-full` | Pied de page au milieu de l'écran, bande vide dessous | `body{min-height:100dvh;display:flex;flex-direction:column} main{flex:1}` ; retirer `min-height:60vh` (`style.css:560`) | S | faible |
| T7 | P2 | `m-home-fold` et toutes les `m-*` | Logo seul, décentré, sur mobile | Afficher `.site-wordmark` sous 600 px (17 px Georgia 700), `.header-inner{justify-content:space-between}`, logo 28 px | S | faible |
| T8 | P3 | — | Réserve d'effets de gabarit | Supprimer `--gradient-accent`, `--color-cta*`, `--shadow-*`, `@keyframes slideDown/fadeIn`, `transition:all` (`style.css:20-21,38-41,44-53,217,375,412`) après vérification qu'aucune page active ne les utilise | S | faible |
| T9 | P3 | — | `prefers-reduced-motion` partiel | Règle globale du § 2.7 dans les deux feuilles ; retirer `scroll-behavior:smooth` (`style.css:57`) | S | faible |

**Structure et textes**

| # | P | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|
| T10 | P1 | `d-out-404`, `m-out-404` | 404 Netlify en anglais, sans en-tête | Créer `layouts/404.html` (gabarit `baseof`) : titre « Page introuvable », une phrase, lien « Retour à l'accueil » | S | faible |
| T11 | P1 | `d-enfance-note-dialog`, `d-enfance-liens` | Fenêtres de Mon enfance collées en haut à gauche sur ordinateur | `@media(min-width:641px){.carto-dialog{margin:auto}}` (`style.css:811`) ; vérifier la même chose pour tout `<dialog>` Hugo | S | faible |
| T12 | P2 | `crops/d-dialogs`, `m-enfance-note-dialog`, `m-compte-delete` | 3 modèles de fenêtre | Un composant : feuille basse sur mobile, centrée 520 px sur ordinateur, titre 20 px système 600, croix 44×44, boutons à droite (primaire à droite). L'appliquer à `.carto-dialog`, `.carto-note-dialog`, fenêtres de `ui.mjs` et `change-interface.mjs` | M | moyen (tests navigateur de focus à relancer) |
| T13 | P2 | `pass3.log` (« NATIVE DIALOG confirm ») | Suppression d'un proche/moment par `confirm()` | Réutiliser la fenêtre de T12 : « Supprimer Hélène ? » + une phrase + [Annuler] [Supprimer] | M | moyen |
| T14 | P2 | `crops/m-forms`, `crops/m-suivi-param` | Voix et vocabulaire instables | Passe de textes selon § 2.5 (liste précise dans chaque page ci-dessous) | M | faible (tests de libellés à ajuster) |
| T15 | P3 | `d-enfance-full` contre `d-proches-full` | Pied de page présent sur Hugo, absent sur Worker | Choisir : le même pied (nom du site seul, choix du propriétaire) sur les deux, ou aucun. Recommandation : le même partout, 14 px `--muted`, sans bande de fond | S | faible |

### 3.1 Accueil et `/personnages/`

| # | P | Type | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|---|
| A1 | P2 | CSS | `d-home-fold`, `m-home-fold` | 123 cartes identiques (nom, âge, 5 thèmes, filet, « Découvrir ses histoires ») : le mur de cartes type gabarit | Rendre la carte entièrement cliquable (`a` englobant ou `::after` étendu), supprimer le filet et le lien « Découvrir ses histoires » (`.perso-btn`), garder nom (Georgia 18 px) + âge + thèmes sur 2 lignes ; `padding:16px`, rayon 10, survol : bordure `--accent` | M | faible |
| A2 | P3 | Texte | `d-home-fold` | « 123 personnages à explorer » : formule de catalogue | « 123 personnages » (ou rien) | S | faible |
| A3 | P3 | CSS | `m-home-search-empty` | État vide dans une carte blanche de plus | Texte simple `--muted` sous le champ, sans carte : « Aucun personnage pour « zzzzqq ». » | S | faible |
| A4 | P3 | Structure | `m-home-full` (27 444 px) | 123 cartes d'un coup sur mobile | Afficher 24, puis « Afficher plus » ; ou liste compacte sur mobile (nom · âge, thèmes sur une ligne) | M | moyen |

### 3.2 Fiche personnage et article

| # | P | Type | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|---|
| B1 | P2 | CSS | `d-out-article-fold`, `m-out-article-fold` | Titre 54 px, interlettrage −2,4 px : les lettres se touchent (« reconnaître ») | Règle T1 article : 40 px max, `letter-spacing:-.01em`, `max-width:24ch` | S | faible |
| B2 | P2 | CSS | `d-out-article-fold` | Article dans une grande carte blanche (rayon 16, marge interne 68 px) posée sur fond gris | Retirer la carte : `.single-article{background:none;border:0;padding:0;max-width:720px}` ; texte directement sur le fond | S | faible |
| B3 | P3 | CSS | `m-out-personnage-fold`, `crops/m-rest` | Fiche : encadré gris + 5 pastilles bordées + listes d'histoires avec 3 pastilles vertes chacune | Portrait sans encadré ; thèmes en une ligne de texte séparée par des points médians ; pastilles d'histoire → une ligne 13 px `--muted` | M | faible |
| B4 | P3 | CSS | `crops/m-rest` (bas d'article) | Lectures liées : 3 cartes avec pastilles, filet, « Baptiste … Lire » | Liste de liens simples : titre (16 px 600) + une ligne d'accroche | S | faible |

### 3.3 Mon enfance (`/cartographie/`)

| # | P | Type | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|---|
| C1 | P1 | CSS | `m-enfance-full`, `d-enfance-full` | Carte (`.carto-form-body`) contenant 6 cartes grises (`.carto-theme-panel`) | `.carto-form-body{background:none;border:0;padding:0}` ; chaque thème = ligne de 64 px séparée par un filet, titre 16 px 600, chevron à droite ; thème ouvert : fond inchangé, filet vert à gauche de 2 px | M | faible |
| C2 | P1 | CSS | `crops/m-enfance-a`, `crops/d-rest2` | Par question : segment sur 2 lignes (Non/Oui puis « Je ne sais pas » dessous), pilule centrée, lien centré → 5 blocs, 3 styles | Trois boutons égaux sur **une** ligne à 390 px (`grid-template-columns:1fr 1fr 1.4fr`, 44 px, 14 px, « Je ne sais pas » tient en 130 px) ; dessous, une seule ligne alignée à gauche de deux boutons texte : « Préciser » et « Liens étudiés » | M | moyen (tests navigateur Mon enfance : sélecteurs `.carto-choice` conservés) |
| C3 | P2 | CSS | `d-enfance-answered` | Réponse « Oui » : toute la question passe sur fond vert pâle + bouton vert plein | Garder seulement le bouton choisi en vert ; pas de fond de ligne (`.carto-factor-row.is-checked{background:none}`) | S | faible |
| C4 | P2 | CSS | `crops/d-rest2`, `m-enfance-resultats-fold` | Résultats : titres en Georgia gras, cartes blanches, deux pilules grises + lien souligné | Titre 16 px système 600 ; ligne « Relié à » en 13 px `--muted` ; « Je me reconnais » / « Pas moi » en boutons secondaires 44 px ; « Voir le détail » en bouton texte ; cartes → lignes séparées par filets | M | faible |
| C5 | P2 | CSS | `m-enfance-fold` | Onglets en pilules (39 px de haut, sous les 44) | Onglets soulignés pleine largeur, 48 px, trait de 2 px `--accent` sous l'onglet actif (même motif que Paramètres) | S | faible |
| C6 | P2 | Texte | `m-enfance-liens` | Fenêtre « Liens étudiés » : 37 paragraphes de méta-analyses, titre « Associations étudiées (37) » en Georgia | Titres système ; garder la phrase de précaution une fois ; replier chaque association (titre seul, texte au toucher) | M | faible |
| C7 | P3 | À confirmer | `m-enfance-full` | « 0 / 7 renseignés » sous chaque thème : le propriétaire a fait retirer les compteurs ailleurs, mais celui-ci sert le prérequis des 60 réponses | Ne pas retirer sans accord ; si conservé : 13 px `--muted`, à droite du titre, « 2 sur 7 » | S | — |
| C8 | P3 | Texte | `m-ambre-scrolled` | Dans les prérequis d'Ambre : « Mon enfance — 5 éléments cochés » alors que 6 réponses sur 60 sont données | Même unité que la règle : « 6 réponses sur 60 » | S | faible |

### 3.4 Mes proches

(La sous-page par proche est en cours ailleurs : ci-dessous, ce qui reste vrai pour la liste et le formulaire.)

| # | P | Type | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|---|
| P1 | P1 | CSS | `d-proches-full`, `m-proches-full` | Grande carte « Votre répertoire » contenant des cartes | `.source-panel{background:none;border:0;box-shadow:none;padding:0}` ; chaque proche = carte cliquable unique (nom 16 px 600 + « Mère » en `--muted`, première ligne du texte, chevron) | S (CSS) | faible |
| P2 | P1 | Structure | `d-proches-full` | Titre « Mes proches » puis sous-titre « Votre répertoire » qui le répète | Supprimer le `h2` ; « Ajouter un proche » à droite du `h1` (ordinateur) ou pleine largeur dessous (mobile) | S | faible (test de libellé) |
| P3 | P1 | Structure | `m-proches-form` (3 153 px), `crops/d-forms` | Formulaire de 11 champs déroulé dans la liste, 3 niveaux de boîtes, bouton en bas | Formulaire en page ou feuille dédiée ; 3 champs visibles (Prénom, Lien, « Comment vivez-vous cette relation ? ») ; les 8 autres derrière « Ajouter des détails » ; barre collante Enregistrer/Annuler | L | moyen (tests de formulaire A5-P2) |
| P4 | P2 | Texte | `crops/m-forms` | « Cette personne est… », « (facultatif) » ×8, « Non renseigné », mélange je/vous, « Fermer le formulaire », « Enregistrer ce proche » | « Lien » ; retirer tous les « (facultatif) » ; option vide « — » ; « Qu'est-ce qui vous aide dans cette relation ? », « Qu'est-ce qui vous pèse ? », « Vous sentez-vous en sécurité avec cette personne ? » ; « Annuler » ; « Enregistrer » | S | faible |
| P5 | P2 | CSS | `m-proches-full` | Pastille « Meilleur ami ou meilleure amie » ; `▶` et `+` sur la même carte ; « Modifier » vert + « Supprimer » rouge en permanence | Rôle en texte `--muted` après le nom ; un seul motif de dépliant ; « Supprimer » déplacé dans l'écran de modification | S | faible |
| P6 | P3 | Structure | `m-proches-details` | « Votre histoire avec Hélène » s'ouvre sur du vide ; « Chargement des repères… » | À vérifier après la livraison des sous-pages : un dépliant vide ne doit pas s'afficher ; pas de texte « Chargement… », réserver la hauteur | S | faible |

### 3.5 Ma ligne de vie

| # | P | Type | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|---|
| L1 | P1 | CSS | `d-ligne-full`, `m-ligne-full` | Carte « Vos moments » → cartes de moment ; même défaut que P1/P2 | Même traitement : pas de carte enveloppante, pas de `h2` « Vos moments » ; chaque moment = bloc sans bordure accroché au trait de la frise, séparé par 24 px | S | faible |
| L2 | P1 | Structure | `d-ligne-full` | Année en double (« 2008 » repère + « 2008 » dans la carte) ; date + pastille + titre de même poids | Repère d'année à gauche (Georgia 18 px) ; dans le bloc : titre 16 px 600, puis une ligne `--muted` « septembre · Déclic · avec Julien, Sofia » (liens soulignés, plus de pastilles) ; ne pas répéter l'année | M | faible |
| L3 | P2 | CSS | `d-ligne-full`, `crops/d-rest` | 60 à 90 px de vide autour de « Ajouter une note » ; filet au-dessus ; « Enregistrer la note » en bouton blanc ; « Modifier / Supprimer » toujours visibles | Une ligne d'actions en boutons texte : « Ajouter une note · Modifier » (Supprimer dans la modification) ; marges 12 px ; champ de note sur une ligne qui grandit, bouton primaire « Enregistrer » | M | faible |
| L4 | P2 | Texte | `crops/m-forms` | « Ce moment, en quelques mots », « Comment le décririez-vous ? » (préréglé sur « Moment positif »), « Repère dans le temps », « Proches associés (facultatif, 10 maximum) », « Gérer mes proches » | « Titre », « Type » (option vide par défaut, pas de présélection), « Date », « Avec » ; limite de 10 dite seulement si elle est atteinte | S | faible |
| L5 | P3 | Structure | `crops/m-forms` (erreur) | Erreur de champ vide par bulle native du navigateur | Validation maison : message 13 px `--danger` sous le champ, focus sur le premier champ en erreur (`novalidate` + `aria-describedby`) | M | moyen |
| L6 | P3 | Texte | `d-ligne-full` | « Moments sans date précise » | « Sans date » | S | faible |

### 3.6 Mes bilans

État vide en cours de simplification ailleurs. Un seul point à transmettre : la phrase sous le titre (« Relisez l'analyse initiale d'Ambre… », `d-bilans-full.png`) contredit le choix « pas d'introduction sous les titres » appliqué aux autres pages.

### 3.7 Mon compte

| # | P | Type | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|---|
| M1 | P1 | Texte | `d-compte-full`, `m-compte-full` | « Mon accès à Ambre » : 4 paragraphes (« mois civil », « selon l'heure de Paris », « phase de découverte ») | Deux lignes : « Accès gratuit. » / « Ce mois-ci : 100 messages et 1 bilan restants, jusqu'au 1er décembre. » | S | faible (test de libellé) |
| M2 | P1 | CSS | `d-compte-full` | Grille 2×3 de blocs séparés par des filets, hauteurs inégales (grand vide sous « Adresse e-mail ») | Une colonne de 720 px, sections empilées : Identité · Connexion · Accès à Ambre · Mes données · Supprimer ; `h2` 18 px, 32 px entre sections, un seul filet entre elles | M | faible |
| M3 | P2 | Texte | `d-compte-full` | « Enregistrer mes informations », « Déconnecter tous les appareils », « Conserver une copie », « Effacer les messages, bilans, traces et la mémoire de conversation. Votre plan est conservé sauf si vous choisissez de l'effacer aussi. », « Comment mes choix sont-ils utilisés ? » | « Enregistrer » ; « Se déconnecter partout » ; section « Mes données » avec « Exporter » et « Effacer la conversation » (détail dans la fenêtre de confirmation, pas sur la page) ; le dépliant renvoie vers Paramètres | S | faible |
| M4 | P2 | CSS | `m-compte-full` | Liens « Se déconnecter » / « Déconnecter tous les appareils » en 16 px gras soulignés, empilés avec 58 px d'écart ; « Supprimer mon compte et mes données » en rouge gras souligné | Boutons texte du § 2.4 sur une ligne ; suppression en bouton texte `--danger` non souligné, 14 px, en fin de page | S | faible |
| M5 | P2 | CSS | `m-out-login`, `d-out-login` | Déconnecté : carte avec un `h1` sans empattement gras (autre gabarit), deux boutons, grand vide | Même `h1` que partout ; afficher directement le formulaire de connexion dans la page (pas de bouton qui ouvre une fenêtre) ; lien « Créer un compte » dessous | M | moyen (tests de connexion) |
| M6 | P3 | Texte | `m-out-register` | « Type de compte : Personne accompagnée » ; « Au moins 12 caractères. » aussi sur l'écran de connexion | Remplacer la liste par un lien discret « Je suis psychologue » ; l'aide de mot de passe seulement à l'inscription | S | faible |
| M7 | P3 | CSS | `m-compte-delete` | Fenêtre : titre 32 px gras sur deux lignes, bouton « Supprimer définitivement » en vert | Titre 20 px ; bouton primaire en `--danger` ; « Annuler » à sa gauche | S | faible |

### 3.8 Conversation avec Ambre (`/mon-accompagnement/`)

| # | P | Type | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|---|
| R1 | P1 | CSS | `m-ambre-fold`, `m-ambre-scrolled` | Sur mobile : carte bordée dans la page, avec son propre défilement ; la 4e ligne des prérequis est coupée | Sous 600 px : `.conversation{border:0;border-radius:0;margin:0 -16px;box-shadow:none}`, hauteur `calc(100dvh - 56px)`, un seul défilement ; l'écran de prérequis défile avec la page (pas de `.conversation-scroll` à hauteur fixe tant qu'il n'y a pas de messages) | M | moyen (tests widget/iframe) |
| R2 | P1 | Texte | `d-ambre-full` | Trois paragraphes avant la liste, puis un quatrième après qui redit la règle | Titre « Avant de commencer » ; une phrase ; la liste ; la précaution en 13 px sous le bouton. Supprimer le paragraphe « Renseignez votre prénom… » (la liste le dit) | S | faible |
| R3 | P2 | CSS | `d-ambre-full`, `m-ambre-scrolled` | Pastilles « Renseigné » ×3 + « À compléter » ; liens « Modifier »/« Compléter » de 20 px de haut | Ligne complète : coche discrète `--accent` et pas de lien ; ligne incomplète : « 6 réponses sur 60 » + bouton secondaire « Compléter » (44 px) ; ligne entière cliquable | S | faible |
| R4 | P2 | À confirmer | `d-ambre-full` | « 0 / 100 messages » en gras dans la barre de titre : compteur permanent | À confirmer avec le propriétaire (les compteurs ont été retirés ailleurs) ; proposition : ne l'afficher qu'à partir de 80 %, sinon seulement dans Mon compte | S | — |
| R5 | P2 | Structure | `d-out-ambre` | Déconnecté : grande boîte vide, zone de saisie grisée « Connectez-vous pour écrire… », micro et « Envoyer » désactivés | Remplacer par une page simple : titre, une phrase, [Se connecter] [Créer un compte] | S | faible |
| R6 | P3 | Texte | `d-ambre-full` | En-tête « Ambre · assistante psychologue IA » ; bouton « Lancer la discussion avec Ambre » | Garder la mention IA (question juridique ouverte, voir § 4) ; bouton : « Commencer » | S | faible |

### 3.9 Paramètres (`/parametres/`) — hors menu

| # | P | Type | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|---|
| S1 | P2 | Texte | `d-parametres-full` | Introduction sous le titre ; étapes numérotées avec « Modèle openai/gpt-5-nano », « 6 entrées disponibles dans l'index », « 37 source(s) chargée(s) sur 37 » | Retirer l'introduction (cohérence avec le choix du propriétaire) ; replier les 5 étapes sous « Comment une réponse est préparée » (fermé par défaut) ; nom du modèle en 13 px `--muted` ; « 37 éléments » | M | faible |
| S2 | P2 | CSS | `d-parametres-full`, `crops/m-suivi-param` | Accordéons dans des accordéons dans une carte ; 3 onglets qui passent sur 2 lignes à 390 px | Un seul niveau de dépliant ; onglets défilables sur une ligne (`overflow-x:auto;white-space:nowrap`) ; pas de carte enveloppante | M | faible |
| S3 | P3 | Texte | `crops/m-suivi-param` | « Modifier ma présentation », « Mon portrait disponible pour Ambre », « Consulter mon portrait » : anciens noms | Aligner sur « Mon compte » ; retirer les liens morts vers une présentation masquée | S | faible |

### 3.10 Mon suivi (`/mon-suivi/`) — hors menu

| # | P | Type | Capture | Problème | Correction | Effort | Risque |
|---|---|---|---|---|---|---|---|
| U1 | P2 | CSS | `d-suivi-full` | Trois largeurs sur la page ; 6 cartes blanches empilées ; boutons pleine largeur de 800 px | Colonne unique 720 px alignée à gauche ; sections séparées par des filets ; boutons à largeur de contenu | M | faible |
| U2 | P2 | Texte | `d-suivi-full`, `crops/m-suivi-param` | Introduction sous le titre ; « Intervalle habituel entre les séances, en jours — Par exemple : 14, 30 ou 60 » ; « Dernière séance : Non renseignée · Prochaine séance : Non renseignée » ; cases « Cartographie », « Portrait », « Plan de changement » ; consentement de 5 lignes citant OpenRouter ; « Points partagés avant séance » et « Journal des accès » vides mais affichés | Retirer l'introduction ; « Rythme des séances » avec 3 choix (toutes les 2 semaines / chaque mois / autre) ; masquer les lignes vides ; « Mon enfance », « Mon compte », « Mon plan » ; consentement en une phrase + « En savoir plus » ; masquer les sections vides | M | moyen (libellés testés ; texte de consentement à faire valider) |
| U3 | P2 | CSS | `crops/m-misc` (fenêtre « Votre objectif ») | Le libellé « Ce que vous souhaitez changer » est barré par le contour de focus du champ ; « — facultatif » | 6 px entre libellé et champ dans cette fenêtre (`change-interface.mjs`) ; retirer « — facultatif » ; « État » et « Priorité choisie » masqués à la création | S | faible |
| U4 | P3 | Structure | `d-suivi-full` | 7 dépliants `▶` dans « Mon plan », la plupart vides (« Aucun élément dans cette rubrique pour le moment. ») | N'afficher que les rubriques non vides ; état vide global : « Aucun objectif pour l'instant. » + [Ajouter un objectif] | M | moyen |

### 3.11 Ordre de réalisation conseillé

1. **Une demi-journée, sans risque** : T10 (404), T11 (fenêtres en haut à gauche), T6 (pied de page), T7 (nom du site sur mobile), T4 (ombres, rayons), T1 (titres), M1, R2, P2 et L1 (sous-titres en double).
2. **Un à deux jours, CSS seul** : T2, T3, T5, C1, C3, C5, P1, L3, M2, M4, B1, B2, R1, R3.
3. **Ensuite, structure** : T12 et T13 (fenêtres), C2 et C4 (questions et résultats), P3 (formulaire de proche), L2, L5, M5, A1, puis Paramètres et Mon suivi.

Après chaque lot : relancer les parcours navigateur existants (`tests/*-browser.cjs`), car plusieurs s'appuient sur des libellés (« Enregistrer ce proche », « Fermer le formulaire ») et sur la présence de `.carto-tabs`.

---

## 4. Ce qu'il ne faut pas changer

Choix explicites du propriétaire, déjà publiés (`docs/reprise-2026-11-16.md`, `docs/livraison-compte-timelines-2026-11-17.md`). Les corrections ci-dessus les respectent ; à ne pas « réparer » par inadvertance lors d'une refonte visuelle.

- **Pas d'introduction sous les titres** de Mon enfance, Mes proches, Ma ligne de vie, Mon compte. (Bilans, Paramètres et Mon suivi en ont encore une : les retirer va dans le sens du choix.)
- **Pas de compteurs** à côté de « Votre répertoire » et « Vos moments », pas de compteur dans l'onglet de Mon enfance, pas de barre fixe en bas sur mobile dans Mon enfance. Les deux compteurs restants (« 0 / 7 renseignés », « 0 / 100 messages ») sont signalés « à confirmer », pas à retirer d'office.
- **Mon enfance** : trois réponses seulement (Non / Oui / Je ne sais pas), « Je ne sais pas » distinct de Non et de l'absence de réponse ; **aucun niveau de preuve, aucune référence bibliographique, aucun code CIM/DSM** à l'écran ; pas de textes d'aide du questionnaire ni des résultats au-delà de la phrase de classement ; « Ce qui en ressort » = une seule liste triée par nombre de vécus associés, sans catégories ni vue d'ensemble ; « Pas moi » réversible dans un repli ; aucun rang présenté comme un risque ou un diagnostic ; description visible carte repliée ; précisions dans une fenêtre titrée avec Enregistrer/Annuler, Échap et retour du focus ; pas de défilement automatique ; encadré 3114 conservé quand il s'affiche ; « Supprimer ma cartographie » discret en bas ; pas de retour de « Ce qui m'a aidé » ; adresse `/cartographie/` conservée.
- **Mon compte** : identité réduite à prénom, nom, date de naissance, genre (+ précision libre) ; présentation et activité masquées, données conservées ; abonnement, export, déconnexion et suppression réunis ici ; Mon compte en dernière position du menu.
- **Menu** : Mon enfance, Mes proches, Ma ligne de vie, Mes bilans, Mon compte (Ambre y revient, travail en cours) ; **Paramètres et Mon suivi hors menu**, accessibles par adresse.
- **Pied de page réduit au nom du site.**
- **Ma ligne de vie** : formulaire réduit (pas de ressentis/soutiens), dates adaptées à la précision sans jour inventé, notes datées multiples.
- **Mes proches** : chronologie commençant par la rencontre, notes communes avec la ligne de vie, observations d'Ambre seulement après validation individuelle.
- **Conversation** : pas de menu « Options de conversation » ; prérequis contrôlés par le serveur (identité, 3 proches, 3 moments, 60 réponses) ; mention « assistante psychologue IA » et phrase « ne pose pas de diagnostic et ne remplace pas un suivi professionnel » (avis juridique en attente) ; accès gratuit, 100 messages et 1 bilan par mois.
- **Pages publiques** : pas de bibliographie ni d'appels de notes dans les articles ; l'accueil reste public et affiche les personnages.
- **Premier affichage** : données prêtes à l'ouverture, pas d'écran de chargement intermédiaire, pas de bandeau « Vous êtes connecté… ». Toute refonte doit garder cette absence de saut à l'ouverture (ne pas réintroduire de squelettes ni de « Chargement… »).
- **Acquis d'accessibilité** : cibles de 44 px sur les réponses, focus visible par contour, boucle de focus des fenêtres.
