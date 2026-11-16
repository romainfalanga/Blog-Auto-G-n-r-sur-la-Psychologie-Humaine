# Exécution — Ambre, stratégies de changement et couverture

Base vérifiée : `1b86169` (fusion PR 83). Branche de travail : `codex/ambre-changement-couverture-20261117`. Le chantier éditorial des personnages reste en pause. Migrations 0019/0020 déjà appliquées ; aucune migration nouvelle appliquée à ce stade.

## Décisions du propriétaire

- Retirer l’encadré 3114 de « Ce qui en ressort ».
- Retirer le bouton « Supprimer ma cartographie » de cette page.
- Corriger les défauts de l’audit §14 sauf le code mort n°15. Le point19 est une information de coordination, traitée en lisant le plan de couverture.
- Conserver les dernières simplifications publiées : onglets du portrait Mon portrait/Ambre, liste des résultats triée par nombre de vécus, introductions retirées.
- Publier les chantiers complets après tests et relecture ; migration manquante seulement, Worker complet, puis site, recette synthétique et effacement.

## Suivi

| ID | Résultat observable | Responsable | Dépendance | État |
|---|---|---|---|---|
| R1 | Défauts1–14,17–18 corrigés avec cas profil rempli et maximal ; coût maîtrisé | engine17 + principal | Audit17 | En cours |
| R2 | Recherche25–40stratégies sourcées, arbre de décision, suivi, relecture | strategies17 | Sources consultées | En cours |
| R3 | Plan critiqué indépendamment et révisé à plusieurs reprises avant implémentation | strategies17 + principal + relecteur | R2 | À faire |
| R4 | Catalogue sélectionné par tour et plan persistant corrigible ; retours planifiés respectueux | Principal + intégration moteur | R1,R3 | À faire |
| R5 | Suivi psychologue, consignes déclarées, mémoire et plan dans les points autorisés, préparation personnelle | Principal | R3,R4 | À faire |
| R6 | Paramètres affiche sources/stratégies/plan/critères/limites réels | Principal | R1,R4,R5 | À faire |
| R7 | Neuf fragments, sources réellement consultées, décisions pour chaque périmètre | Neuf missions séparées | Plan couverture17 | Lots1,2,8,9 en cours ; autres à faire |
| R8 | Intégration contrôlée, registre hors périmètre, correspondances161repères et classifications, générateurs | Principal | R7 | À faire |
| R9 | Plafonds questionnaire, contexte non suggéré, résumés et interface bornés | Principal | R8 | À faire |
| R10 | Encadré3114 et bouton de suppression retirés de la page | Principal | Réponses utilisateur | Codé, vérification en cours |
| R11 | Relectures indépendantes, tests complets, publications prouvées, recette et nettoyage | Principal + relecteurs | Tous | À faire |

## Organisation et contrats

Un seul auteur par fichier. engine17 possède provisoirement ambre-engine, chat-backend, chat-safety, chat-text, ambre-insight et leurs tests. strategies17 possède la recherche et le plan de changement. Chaque lot clinique ne modifie que son fragment et son rapport de sources. Le principal possède l’intégration des données, le générateur, l’interface Cartographie, les fichiers communs et la livraison. Les auteurs de recherche ne déploient pas et ne sous-délèguent pas.

Les nouvelles stratégies et le plan ne sont implémentés qu’après les cycles de critique/révision demandés. Les corrections indépendantes du moteur et les recherches de couverture avancent pendant cette préparation. Les sources non accessibles restent signalées ; aucun lien n’est ajouté pour atteindre un quota.

## Vérification prévue

Tests moteur sur profils réels synthétiques bornés et chemins de reprise/réécriture ; tests d’isolation, révision, export, suppression et révocation ; égalité calcul navigateur/Worker ; contrôles des générateurs ; suite de verify.yml ; Playwright1440/390/320 ; recette distante avec comptes synthétiques supprimés ensuite. Les preuves de publication et les limites scientifiques sont consignées à la livraison.
