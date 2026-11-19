# Bilans historiques, symptômes passés et suivi — 19 novembre 2026

Base : e106deb, PR 105/106. Lot publié par la PR 107. Voir [la livraison et le blocage fournisseur](livraison-bilans-historiques-2026-11-19.md). Branche : codex/bilans-historique-nov19.

| Exigence | Critère observable | Responsable | État |
|---|---|---|---|
| R1 | Tous les thèmes Mes symptômes fermés à l’arrivée | Interface | Réalisé, tests locaux réussis |
| R2 | Non / Oui / Je ne sais pas / Dans le passé ; précision exigée avant validation du passé ; annulation sans mutation | Interface | Réalisé, tests locaux réussis |
| R3 | Grille lisible à 390/1440 px, Mon enfance préservée ; passé distinct dans stockage, classement, export et contexte | Interface | Réalisé, tests locaux réussis |
| R4 | Chaque nouveau bilan possède un instantané privé immuable et consultable ; pas de reconstruction rétroactive | Backend | Réalisé, tests locaux réussis |
| R5 | Analyse de toutes les sections par blocs, couverture traçable et reprise ; aucun bilan complet annoncé si traitement partiel | Backend | Réalisé, tests locaux réussis |
| R6 | Mes diagnostics : évaluations datées, méthode et manques visibles ; hypothèses automatiques distinguées des conclusions professionnelles | Clinique + UI | Réalisé, tests locaux réussis |
| R7 | Recherche OMS/CDDR, HAS, NICE ; couverture explicite des sujets et absence de probabilité diagnostique inventée | Clinique | Réalisé, tests locaux réussis |
| R8 | Prochain rendez-vous saisissable/modifiable/supprimable depuis Mon compte, contexte Ambre actualisé | Principal + UI | Réalisé, tests locaux réussis |
| R9 | Dossier psychologue enrichi, accès révocable limité aux rubriques autorisées, provenance de la mémoire et des conclusions conservée | Principal | Réalisé, tests locaux réussis |
| R10 | Export, effacement, sécurité entre comptes, relecture indépendante, tests, migrations et déploiement vérifiés | Principal | Publié, migrations et recette fonctionnelle vérifiées ; génération IA réelle bloquée par OpenRouter HTTP 402 |

## Décisions d’architecture

- `answers[id] = "past"` signifie avant les douze derniers mois. Une précision non vide est obligatoire. Cette valeur ne devient ni un oui récent ni une alerte actuelle.
- Les instantanés conservent les données personnelles utiles, pas une copie du catalogue public, des secrets ou des journaux techniques. Les anciens bilans sont identifiés sans archive disponible.
- Les données longues sont analysées par blocs avec reprise et couverture conservée. Les limites de taille et de budget sont explicites ; aucune troncature silencieuse ne vaut analyse complète.
- Les référentiels ne sont pas un score de rapprochement : durée, retentissement, contexte, exclusions et évaluation différentielle peuvent manquer. Les hypothèses automatiques ne sont pas présentées comme des diagnostics professionnels. Pas de renvoi générique répétitif ; sécurité actuelle conservée.
- L’espace psychologue et les séances existent déjà. Réutiliser ces mécanismes ; aucun partage automatique supplémentaire. Les informations professionnelles sont déclarées, non vérifiées par la plateforme.
- Les réglages d’accès gratuit, limites de dépenses et l’éditorial public restent ceux de la livraison précédente.
