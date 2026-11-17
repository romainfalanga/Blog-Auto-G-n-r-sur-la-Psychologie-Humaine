# Mon enfance, navigation et affichage simplifié — 17 novembre 2026

Base : main `424c5745cd8ee2b809634cf5679e3b8ebdb144b4`. Cette demande remplace le classement force × spécificité et les anciens sous-onglets du portrait. Les données personnelles, le graphe scientifique et ses sources internes restent conservés.

| Exigence | Résultat observable | Responsable | État |
|---|---|---|---|
| R1 | Menu et titre « Mon enfance » ; liens historiques /cartographie/ fonctionnels | Principal + interface | Vérifié localement |
| R2 | Résultats dans une liste classée par nombre de vécus associés décroissant, sans priorité clinique cachée ; Pas moi reste réversible | Interface | Vérifié localement |
| R3 | Aucun badge de catégorie sur les cartes ; aucune citation bibliographique dans les détails | Interface | Vérifié localement |
| R4 | Références scientifiques retirées de l’affichage partout sur le site ; données de recherche conservées | Sources publiques | Vérifié localement |
| R5 | Choix lisibles à 320/390/768/1440 px, « Je ne sais pas » sur une ligne, cibles tactiles ≥44 px, clavier/focus ; sauvegarde et conflits conservés | Interface | Vérifié localement |
| R6 | Paramètres hors navigation principale, toujours accessible à /parametres/ pour gestion du compte et envers du décor | Principal | Vérifié localement |
| R7 | Mon portrait avant Mon enfance ; portrait sans sous-menu ni vue de synthèse Ambre ; formulaires et liens de lecture conservés | Principal | Vérifié localement |
| R8 | Relecture indépendante, tests adaptés, Worker complet puis site, recette réelle et nettoyage | Principal + relecture | Tests et relecture terminés ; publication à faire |

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
- Site : 43 tests, y compris 2 tests de rendu Hugo ; compilation Hugo 0.140.0 réussie, 7 931 pages. Toutes les commandes de verify.yml exécutées. La suite journal compte 794 tests : une assertion de l’ancien libellé de classement a été adaptée après le premier passage (les 793 autres tests réussissaient).
- Comparaison complète des rendus publics : 151 HTML modifiés, texte inchangé hors espaces, aucun lien interne/ancre supprimé ; 201 occurrences de liens de référence déliées (157 URL). Les références intégrées à une phrase gardent leurs mots ; les bibliographies, appels de notes et citations pures sont retirés du rendu. Les manuscrits, preuves et fichiers d’audit restent intacts.
- Paramètres reste accessible directement, compte/export/suppression et transparence personnelle préservés. Le JSON affiché y est explicitement indiqué sans bibliographie ; les données stockées et exports ne sont pas modifiés.

Aucune migration nécessaire. Publication et recette distante à consigner ici et dans la note de reprise.
