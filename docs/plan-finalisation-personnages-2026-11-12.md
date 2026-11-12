# Plan de finalisation de la page Personnages — 12 novembre 2026

État de départ vérifié (commit `21c18cf`) : 1419 articles, 119 personnages (33 parcours « psychologie » d’origine, 86 parcours cliniques publiés), 68 cas du registre ouverts dont 4 en cours (Sylvain en relecture, Loubna 10/12, Serge 8/12, Florence 6/12). Cinq angles conceptuels restent sans récit direct (APA, RDoC, CIF).

## Deux trames à respecter

| Type | Rôle des douze épisodes | Trame de chaque article |
| --- | --- | --- |
| Personnage « psychologie » (d’origine) | Douze mécanismes, biais, émotions ou schémas différents vécus par la même personne, avec une vie cohérente (métier, proches, projet). | Scène d’ouverture → explication du concept → manifestations → 3 techniques → retour à la scène (décision et conséquence) → conclusion → « Du côté des proches ». |
| Personnage clinique (format 2) | Douze facettes distinctes d’un même trouble, avec chronologie diagnostique cohérente. | Même trame ; l’explication porte sur la facette du trouble, sources primaires dans les champs éditoriaux. |

## Phases

1. **Page d’accueil renommée « Personnages »** (navigation, titre et intitulé de la page).
2. **Achever le lot en cours** : relecture de Sylvain, fin de Loubna, Serge et Florence, puis relectures indépendantes et intégration.
3. **Audit de cohérence de tout le corpus** (lecture seule) : un agent auditeur par groupe de parcours vérifie pour chaque article la présence et l’ordre des parties, la compréhension de l’histoire (qui, où, quoi, pourquoi), les raccords avec les épisodes voisins, la distinction des aspects et l’utilité des conseils aux proches. Rapports : `docs/audit-coherence-2026-11-12/<id>.json`.
4. **Corrections** des défauts bloquants ou majeurs relevés, par des agents correcteurs distincts des auditeurs ; preuves de lecture et empreintes mises à jour ; parcours d’origine incomplets (11 épisodes) complétés à douze, épisode surnuméraire de Camille examiné.
5. **Nouveaux parcours cliniques** des 64 cas restants, par lots parallèles : créateur (dossier + plan) → rédacteur (12 épisodes dans l’ordre) → relecteur indépendant → intégration.
6. **Nouveaux parcours « psychologie »** pour les angles conceptuels sans récit direct.
7. **Intégration et vérifications** : régénération des index, tests, contrôles éditoriaux, rendu Hugo ; sauvegarde Git régulière sur la branche de travail et demande de fusion. Le déploiement public suit la fusion dans `main`.

Les lacunes non traitées restent consignées dans `docs/etat-couverture.md`.
