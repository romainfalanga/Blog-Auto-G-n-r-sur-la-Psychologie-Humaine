# Plan Cartographie V3 — 16 novembre 2026

Retour de l’utilisateur sur la V2 : descriptions tronquées, affichage cassé à l’ouverture d’une conséquence sur mobile, navigation « Mon enfance » peu pratique sur mobile, blocs du bas à retirer, et surtout un outil à penser sur le long terme : la personne y revient, écrit ses propres précisions, et Ambre peut y écrire. La cartographie devient la carte de travail commune de la personne et d’Ambre.

## 1. Interface

- Descriptions des causes affichées en entier, partout.
- Correction du bug mobile : le texte des liens d’une carte reprenait le style de l’encadré « Vue d’ensemble » (classe CSS en double).
- « Mon enfance » : un sommaire des thèmes en cartes (titre, description, avancement « 3 réponses sur 12 ») ; on entre dans un thème, on répond, on revient au sommaire ou on passe au suivant. Sur ordinateur, le sommaire reste visible à gauche du thème ouvert.
- Blocs du bas retirés (ce qui protège, notes, aide, sources globales). Les sources restent accessibles dans chaque lien, et l’encadré 3114 s’affiche toujours quand les idées suicidaires ressortent.
- Nouveau thème « Ce qui m’a aidé » : les facteurs de protection deviennent des réponses (ils étaient un simple texte), transmis à Ambre.

## 2. Un outil sur la durée

- Texte libre sur chaque cause (« Ce que j’en retiens ») et sur chaque conséquence (« Comment cela se manifeste aujourd’hui »).
- Plus de verrouillage : après un accord donné une fois, l’enregistrement est automatique à chaque modification ; la personne peut tout modifier ou supprimer.
- Les notes d’Ambre apparaissent sous la cause ou la conséquence concernée, signées « Ambre », avec un bouton pour les retirer.

## 3. Ambre

- Elle lit : réponses, intensités, retours « je me reconnais », textes de la personne, ressources, ses propres notes.
- Elle écrit : quand la personne décrit dans la conversation un vécu d’enfance ou une manifestation actuelle rattachable à une cause ou une conséquence de la carte, elle y ajoute une note courte, fidèle aux mots de la personne. Elle n’invente rien, ne coche pas de cause à la place de la personne, et l’annonce dans sa réponse.
- Consignes : croiser cartographie, ligne de vie, proches et portrait ; aider à se souvenir par des questions ouvertes, sans jamais suggérer qu’un événement a eu lieu (risque de faux souvenirs) ; relier un vécu d’enfance à un fonctionnement présent comme hypothèse à vérifier.

## 4. Technique

- Migration `0017` : colonnes `notes`, `resources`, `ambre_notes` sur `cartographies`.
- `PUT /api/cartographie` accepte `notes` et `resources` ; `DELETE /api/cartographie/ambre-notes/:id` ; les écritures d’Ambre ne changent pas la version utilisée par l’enregistrement automatique.
- Tests, déploiement D1 puis Worker, vérification en ligne sur ordinateur et mobile.
