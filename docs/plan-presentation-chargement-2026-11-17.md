# Présentation unique et chargement stable — 17 novembre 2026

## Demande et périmètre

Plan confirmé par « go » : Mon portrait ne contient qu'un champ de présentation directement modifiable et son enregistrement ; retirer catégories, autres champs et articles aimés. Dans Mon enfance, séparer les réponses par une croix sur mobile et des traits verticaux sur ordinateur. Éliminer les contenus transitoires trompeurs sur Mon portrait, Mon enfance, Mes proches et Ma ligne de vie. Conserver toutes les données historiques.

| Exigence | Réalisation | Vérification |
| --- | --- | --- |
| R1 — Un seul champ dans Mon portrait | `self_description`, 10 000 caractères, enregistrement explicite ; anciens groupes et fenêtre d'édition retirés | Tests UI et navigateur ; sauvegarde, rechargement, édition pendant PUT, conflit 409 avec comparaison explicite |
| R2 — Données historiques conservées | PUT limité à `self_description` + `expected_revision` ; API parcellaire existante inchangée, aucune migration | Vrai Worker et SQLite : identité, émotions et anciens textes identiques après sauvegarde |
| R3 — Séparateurs Mon enfance | Croix 2×2 sur mobile, traits verticaux sur ordinateur ; événements uniques à trois réponses | Captures et 7 parcours choix ; cibles 48 px, labels non coupés, focus et sélection conservés |
| R4 — Chargement stable | Titre et attente dès le démarrage ; session inconnue distincte d'anonyme ; aucune collection avant présentation ; cartographie attend session puis carte | 18 parcours personnels avec délais contrôlés à 320/390/768/1440 ; 20 parcours cartographie (latence, stockage local, conflits, panne et reprise) |
| R5 — Navigation et erreurs | Réponses tardives ignorées ; brouillons conservés ; réessai après erreur ; logout anonyme navigable | Relecture indépendante et scénarios navigateur |

## Diagnostic vérifié

Les styles du Worker sont inclus avant le contenu, sans chargement différé. Le flash de Mon portrait venait de `renderLikes()` immédiat alors que `renderSelfProfile()` attendait l'API : les articles apparaissaient avant le formulaire inséré au-dessus. Un clic de navigation pendant `/api/me` passait aussi par `landing()` comme si la session était anonyme. Mon enfance dessinait d'abord un brouillon anonyme ou une carte vide avant de connaître le compte et la carte distante. Les correctifs traitent ces séquences, sans cache de données personnelles ajouté.

Les anciens liens d'onglets du portrait reviennent au champ unique. Le lien « Mes articles aimés » depuis les articles est retiré ; les boutons J'aime et leurs explications restent utilisables sur les articles. Paramètres conserve la consultation des données de lecture sans lien vers la rubrique supprimée.

## Contrôles locaux

- 789 tests journal réussis, dont 99 tests UI. Les anciens tests du formulaire multi-rubriques sont remplacés par les contrats du champ unique.
- 43 tests site réussis, dont les fixtures Hugo ; contrôles de `.github/workflows/verify.yml` réussis.
- Navigateurs : 18 chargement/présentation, 6 formulaires P2 avec Worker + SQLite, 20 cartographie, 7 choix ; pas d'erreur JavaScript ni de débordement observé.
- Relecture indépendante : code, tests personnels/UI, navigation après déconnexion et cartographie.
- Aucune migration, aucun changement du catalogue scientifique, aucune génération IA demandée.

## Livraison

- PR [95](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/pull/95), source `987c9f2ecdbb8dc057ff003cbc45759aeb4614d8`, fusion `4e2ab45a2e5e73f979b662833276d4d3f4ae01c3`, arbre `e4ce506a054a761e925c4613e4c292a770629983`.
- CI `44117038532` réussie ; relecture indépendante favorable (99 UI, 18 personnels, 27 cartographie et complément déconnexion/retour).
- Worker `a1d26191d90a4cda920233278861bc5b`, 17 novembre à 16:41:11 Paris : 44 modules transférés puis relus SHA-256, zéro différence, 7 liaisons et cron conservés, aucun changement de schéma.
- Netlify `6afc75f61cb72d01449c6e84`, `ready`, même commit de fusion, publié le 17 novembre à 16:43:23 Paris.

Recette distante terminée : 16 parcours (les quatre pages à 320/390/768/1440 px), 32 captures, délais réels de session et de données contrôlés, sauvegarde/rechargement de présentation et historique préservé. Séparateurs, réponse inconnue et ancienne réponse à `controle` vérifiés. Aucune erreur JavaScript, console ou interception et aucun appel de génération.

Le premier essai a été interrompu par une erreur de l'interception réseau du test (retrait d'un gestionnaire avant la poursuite de sa requête), sans défaut applicatif établi. Le compte a été supprimé avant la relance avec un gestionnaire stable. Le second parcours complet a réussi. Les deux comptes synthétiques ont été supprimés (API 200 et session vide), les fichiers d'identifiants effacés ; contrôle D1 sur 43 tables : zéro ligne résiduelle, aucune violation de clé étrangère et 24 migrations inchangées. Les sorties locales et captures sont sous `work/chargement-pages/`, `work/cartographie-chargement/` et `work/cartographie-choix/` ; aucun identifiant de compte ni secret n'est versionné.
