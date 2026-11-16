# Plan « Ambre au centre » — 16 novembre 2026

## Constat

- **Cartographie « effacée »** : la base ne contient aucune cartographie. Les réponses n’étaient gardées que dans la mémoire de la page tant que l’accord et « Enregistrer » (placés dans l’onglet des résultats) n’avaient pas été utilisés ; chaque rechargement les perdait. Défaut de conception à corriger, pas une suppression.
- Espace vide en bas des pages de la cartographie : marge prévue pour la barre mobile, appliquée partout.
- Le site doit s’ouvrir sur Ambre ; les personnages sortent du menu et du fonctionnement d’Ambre.

## 1. Ne plus rien perdre

- Brouillon local automatique (stockage du navigateur) : réponses, précisions, ressources et retours survivent aux rechargements et aux mises à jour, connecté ou non.
- Personne connectée : bandeau visible en haut de la cartographie tant que rien n’est enregistré (« Vos réponses ne sont pas encore enregistrées dans votre profil »), avec l’accord et le bouton au même endroit ; ensuite enregistrement automatique. Le brouillon local est repris à la connexion et effacé après enregistrement réussi.
- Marge basse supprimée hors barre mobile.

## 2. Navigation

Menu : **Ambre · Cartographie · Mon portrait · Mes proches · Ma ligne de vie · Mes bilans · Mon compte**. La racine du site redirige vers Ambre. Le catalogue des personnages reste consultable à `/personnages/` (articles et liens existants conservés) mais n’est plus dans le menu, et la cartographie ne renvoie plus vers les articles.

## 3. Parcours avec Ambre

1. **Fenêtre d’accueil** sur la page d’Ambre tant que la conversation n’a pas été lancée : ce que fait Ambre (analyse approfondie à partir de la cartographie, du portrait, des proches et de la ligne de vie, puis conversation centrée sur ce que la personne veut comprendre et changer), et l’état des quatre sources avec un lien vers chacune.
2. **« Lancer la discussion avec Ambre »** puis confirmation (« Êtes-vous certain ? Cela va initier la conversation à partir des informations que vous avez renseignées. »), accord explicite pour l’IA inclus. Désactivé si aucune source n’est renseignée.
3. **Analyse initiale** : premier bilan, rédigé à partir de toutes les sources ; il explique ce qui semble ressortir (hypothèses, jamais diagnostic), ce qui reste à éclaircir, et demande ce que la personne veut travailler.
4. **Conversation** : centrée sur les besoins et les objectifs de la personne ; Ambre pose les questions qui aident à creuser (ouvertes, une à la fois, sans suggérer de souvenirs), relie présent et histoire, propose des pistes concrètes de changement, écrit ses notes dans la cartographie.
5. **« Faire le point »** dans la conversation : nouveau bilan complet à partir des quatre sources, de la conversation depuis le dernier bilan et du bilan précédent (ce qui a évolué, ce qui se confirme, objectifs, prochaines étapes).
6. **« Mes bilans »** (`/mes-bilans/`) : historique daté de l’analyse initiale et des points, relisible et supprimable, inclus dans l’export.

## 4. Ambre sans les personnages

Suggestions de lecture, candidats et mentions d’articles retirés des consignes, du contexte, de l’accueil et de l’interface. En échange, Ambre reçoit pour les conséquences reconnues ou convergentes le résumé des liens de la cartographie (niveau de preuve et formulation issue des sources), en plus du catalogue de repères psychologiques existant.

## 5. Tenue en charge (1 ou 10 000 personnes)

- Bilans traités de façon asynchrone et reprenable (réponse immédiate, suivi par interrogation, reprise par la tâche planifiée si l’exécution est interrompue), idempotents, un seul en cours par personne.
- Limites : un point par 24 h et par personne, comptabilisé dans les budgets existants ; contexte borné (sources résumées, conversation tronquée par ancienneté) ; table `ambre_reviews` indexée par personne et date ; aucune requête non bornée.
- Brouillon local : aucune charge serveur ; enregistrement automatique regroupé (une requête au plus toutes les 1,2 s).

## 6. Vérifications

Tests unitaires et d’API, essai réel de bout en bout avec un compte temporaire supprimé ensuite, contrôle visuel ordinateur et mobile, vérification en ligne après fusion.
