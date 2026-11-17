# Mon enfance, navigation et affichage simplifié — 17 novembre 2026

Base : main `424c5745cd8ee2b809634cf5679e3b8ebdb144b4`. Cette demande remplace le classement force × spécificité et les anciens sous-onglets du portrait. Les données personnelles, le graphe scientifique et ses sources internes restent conservés.

| Exigence | Résultat observable | Responsable | État |
|---|---|---|---|
| R1 | Menu et titre « Mon enfance » ; liens historiques /cartographie/ fonctionnels | Principal + interface | Vérifié et publié |
| R2 | Résultats dans une liste classée par nombre de vécus associés décroissant, sans priorité clinique cachée ; Pas moi reste réversible | Interface | Vérifié et publié |
| R3 | Aucun badge de catégorie sur les cartes ; aucune citation bibliographique dans les détails | Interface | Vérifié et publié |
| R4 | Références scientifiques retirées de l’affichage partout sur le site ; données de recherche conservées | Sources publiques | Vérifié et publié |
| R5 | Choix lisibles à 320/390/768/1440 px, « Je ne sais pas » sur une ligne, cibles tactiles ≥44 px, clavier/focus ; sauvegarde et conflits conservés | Interface | Vérifié et publié |
| R6 | Paramètres hors navigation principale, toujours accessible à /parametres/ pour gestion du compte et envers du décor | Principal | Vérifié et publié |
| R7 | Mon portrait avant Mon enfance ; portrait sans sous-menu ni vue de synthèse Ambre ; formulaires et liens de lecture conservés | Principal | Vérifié et publié |
| R8 | Relecture indépendante, tests adaptés, Worker complet puis site, recette réelle et nettoyage | Principal + relecture | Vérifié et publié ; recette et nettoyage terminés |

## Choix de conception

Conserver l’identité existante : fond #F2F5F7, papier #FFFFFF, texte #172D3C, texte secondaire #526575, accent #205E54 et séparateurs #D6E0E6. Garder Georgia pour la marque et les titres existants, la police système pour les formulaires. Questions alignées à gauche ; choix en grille stable à deux colonnes sur petit écran, en rangée sur espace suffisant. Les réponses longues ne sont ni tronquées ni découpées en mots isolés. État choisi identifiable autrement que par couleur ; focus clavier visible.

La simplification porte sur le rendu, pas sur la suppression de preuves internes ni de données. Les citations personnelles nécessaires à la traçabilité d’Ambre et les liens d’aide ou de navigation ne sont pas des bibliographies. Il n’existe actuellement aucune page intitulée « L’envers du décor » : Paramètres reste disponible à son adresse directe, hors menu.

## Séquence

1. Modifier en parallèle les surfaces indépendantes (cartographie ; rendu des références ; navigation/portrait).
2. Intégrer les changements et aligner les métadonnées d’ordre transmises à Ambre.
3. Relecture indépendante des contrats, données préservées et rendus ; corriger les constats.
4. Tests de calcul/API, formulaires et navigation ; construction Hugo et tests navigateur aux tailles prévues, puis inspection des captures.
5. Sauvegarder sur GitHub, publier le Worker complet sans nouvelle migration, fusionner et vérifier Netlify ; recette avec compte synthétique supprimé ensuite.

## Validation intégrée

Relecture indépendante favorable des trois lots : 70 tests de calcul/API, 103 de navigation/portrait, filtre de bibliographie des Paramètres et 8 parcours Mon enfance à 320/390/768/1440 px. Corrections retenues : sélection avec contour et graisse en plus de la couleur ; consignes d’Ambre cohérentes avec le tri ; références filtrées aussi dans les traces JSON préfixées.

- Formulaires P2 : 6 parcours réels Worker/SQLite, sans erreur JS ni débordement.
- Cartographie : 16 parcours de sauvegarde, 7 de compatibilité historique/incertitude, 8 d’interface et 6 de détail des associations ; sources internes et noyau du graphe inchangés (`d9ab1373af12b3bdc9d3634d64115bc258d430b6f4b0aa7cd9effa4c80292308`).
- Site : 43 tests, y compris 2 tests de rendu Hugo ; compilation Hugo 0.140.0 réussie, 7 931 pages. Toutes les commandes de verify.yml exécutées. La suite journal compte 794 tests, tous validés par la CI finale ; les assertions du classement reflètent désormais le tri demandé.
- Comparaison complète des rendus publics : 151 HTML modifiés, texte inchangé hors espaces, aucun lien interne/ancre supprimé ; 201 occurrences de liens de référence déliées (157 URL). Les références intégrées à une phrase gardent leurs mots ; les bibliographies, appels de notes et citations pures sont retirés du rendu. Les manuscrits, preuves et fichiers d’audit restent intacts.
- Paramètres reste accessible directement, compte/export/suppression et transparence personnelle préservés. Le JSON affiché y est explicitement indiqué sans bibliographie ; les données stockées et exports ne sont pas modifiés.

Aucune migration nécessaire ; les 24 migrations existantes sont conservées.

## Publication et recette finale

- PR 94 : source `82a2a4fe5e0c8bd5cbe4178c0172a67b9f89b088`, fusion `e626c32993cebd65d06e5b67b5d0c14c4be97464`, arbre `c187d3ee1ce05b46823e33c197ec2d65d0475a34` identique au travail local. CI `44102059437` réussie.
- Worker complet `229ad678358140e9b94f282b9d7daa7c`, 17 novembre 2026 à 14:22 UTC : 44 modules relus par SHA-256, zéro écart, 7 liaisons et cron minute conservés.
- Netlify `6afc635ce18129fb2516d764`, ready, même commit de fusion, publié à 14:24:03 UTC (15:24 Paris).
- Recette réelle aux quatre tailles : menu/titre, formulaire sans sous-onglet, Paramètres hors menu mais accessible directement, choix binaires et incertitude, sauvegarde/rechargement/export, anciennes réponses et notes conservées, tri décroissant et détails sans catégories/bibliographies. Trois articles publics contrôlés. 20 captures, aucune erreur JS/console ni débordement, aucun appel de génération.
- L’assertion finale du script comptait initialement les quatre lectures GET `/api/chat/insight` de Paramètres comme des appels IA. Les parcours et l’export étaient déjà terminés : ces lectures ne déclenchent pas de génération. Filtre de test corrigé, rapport brut conservé et validation des résultats enregistrée séparément.
- Compte synthétique supprimé (DELETE 200 puis session vide), credentials locaux effacés ; contrôle D1 sur 43 tables : zéro reliquat, aucune violation de clé étrangère, 24 migrations. Rapports et captures dans `work/enfance-ux/production/`, sans credentials versionnés.
