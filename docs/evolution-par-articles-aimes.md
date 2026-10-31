# Évolution par les articles aimés

Demande du 31 octobre 2026. Le site existant, ses 231 articles, ses 20 personnages et leurs adresses sont conservés.

| Exigence | Résultat observable | Responsable | État |
|---|---|---|---|
| R1 Découverte | Personnages, cartes d'articles et lectures montrent des mots-clés issus des sources existantes | découverte | en cours |
| R2 Collection | Un article se marque en un clic; les explications sont facultatives; collection sans plafond de trois | serveur + interfaces | en cours |
| R3 Préparation | Progression vers dix articles aimés; aucun prénom, âge ou questionnaire biographique demandé | serveur + journal | en cours |
| R4 Portrait | Portrait hypothétique issu des affinités et commentaires, avec axes et sources | serveur + génération | en cours |
| R5 Révision | Chez les abonnés, ajout/retrait/commentaire et retour négatif ou positif invalident le portrait antérieur | serveur | en cours |
| R6 Récits | Personnage fictif stable choisi par l'IA; récits privés cohérents et une lecture quotidienne | génération | en cours |
| R7 Automatisation | Générateur Python historique réellement importé/réutilisé dans GitHub Actions, profils et récits protégés | génération + principal | en cours |
| R8 Publication | Migrations sans perte, secret OpenRouter conservé dans Cloudflare, domaine et redirection corrigée conservés | principal | en cours |
| R9 Vérification | Tests de seuil, isolation, versions concurrentes, refus de profil obsolète, réutilisation des fonctions historiques et recette du domaine | revue + principal | à faire |

## Décisions

- Dix articles aimés sont nécessaires pour l'abonnement. Un commentaire enrichit le portrait, mais n'est jamais obligatoire pour aimer un article.
- Une affinité avec une fiction ne prouve pas que l'utilisateur a vécu les événements du personnage. Les déclarations explicites et corrections ont priorité.
- Une révision du portrait est déclenchée par chaque changement de source chez l'abonné. Les changements proches peuvent être traités ensemble au prochain passage de l'automatisation.
- La clé OpenRouter reste un secret Cloudflare. GitHub Actions utilise son identité OIDC et une passerelle serveur limitée aux travaux autorisés; aucune clé permanente supplémentaire ni texte personnel dans les logs.
- L'automatisation quotidienne d'origine redevient le moteur, avec son cadrage, sa rédaction, sa relecture, sa cohérence et ses résumés. Les données privées ne sont jamais écrites dans le dépôt public.
- Le prix demandé reste 10 EUR par mois; la configuration Stripe constitue un prérequis distinct de l'implémentation.
