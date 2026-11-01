# Évolution par les articles aimés

Demande du 31 octobre 2026. Le site existant, ses 231 articles, ses 20 personnages et leurs adresses sont conservés.

| Exigence | Résultat observable | Responsable | État |
|---|---|---|---|
| R1 Découverte | Personnages, cartes d'articles et lectures montrent des mots-clés issus des sources existantes | découverte | implémenté et vérifié |
| R2 Collection | Un article se marque en un clic; les explications sont facultatives; collection sans plafond de trois | serveur + interfaces | implémenté et vérifié |
| R3 Préparation | Progression vers dix articles aimés; aucun prénom, âge ou questionnaire biographique demandé | serveur + journal | implémenté et vérifié |
| R4 Portrait | Portrait hypothétique issu des affinités et commentaires, avec axes et sources | serveur + génération | implémenté et vérifié |
| R5 Révision | Chez les abonnés, ajout/retrait/commentaire et retour négatif ou positif invalident le portrait antérieur | serveur | implémenté et vérifié |
| R6 Récits | Personnage fictif stable choisi par l'IA; récits privés cohérents et une lecture quotidienne | génération | implémenté et vérifié |
| R7 Automatisation | Générateur Python historique réellement importé/réutilisé dans GitHub Actions, profils et récits protégés | génération + principal | implémenté et vérifié |
| R8 Publication | Migrations sans perte, secret OpenRouter conservé dans Cloudflare, domaine et redirection corrigée conservés | principal | implémenté et vérifié |
| R9 Vérification | Tests de seuil, isolation, versions concurrentes, refus de profil obsolète, réutilisation des fonctions historiques et recette du domaine | revue + principal | vérifié |

## Décisions

- Dix articles aimés sont nécessaires pour l'abonnement. Un commentaire enrichit le portrait, mais n'est jamais obligatoire pour aimer un article.
- Une affinité avec une fiction ne prouve pas que l'utilisateur a vécu les événements du personnage. Les déclarations explicites et corrections ont priorité.
- Une révision du portrait est déclenchée par chaque changement de source chez l'abonné. Les changements proches peuvent être traités ensemble au prochain passage de l'automatisation.
- La clé OpenRouter reste un secret Cloudflare. GitHub Actions utilise son identité OIDC et une passerelle serveur limitée aux travaux autorisés; aucune clé permanente supplémentaire ni texte personnel dans les logs.
- L'automatisation quotidienne d'origine redevient le moteur, avec son cadrage, sa rédaction, sa relecture, sa cohérence et ses résumés. Les données privées ne sont jamais écrites dans le dépôt public.
- Le prix demandé reste 10 EUR par mois; la configuration Stripe constitue un prérequis distinct de l'implémentation.

## Validation et prérequis restant

- Hugo compilé; les 231 articles et les 20 personnages sont conservés, avec mots-clés et navigation contrôlés sur le domaine.
- 19 tests backend et 7 tests Python passent, dont concurrence, isolation, consentement, seuil et transport OIDC.
- Recette réelle : 11 articles aimés, deux explications facultatives, aucun questionnaire; portrait IA et premier récit privé publiés via le workflow GitHub original. Le retour négatif a été pris en compte par une nouvelle analyse IA, publiée à la version courante des sources.
- La migration D1 est appliquée, le Worker publié et le Cron Cloudflare retiré. OpenRouter reste un secret Cloudflare.
- L’ouverture commerciale attend la configuration Stripe du prix à 10 EUR/mois et des secrets de paiement. La récupération du mot de passe par email reste à ajouter avant cette ouverture.

## Ajustements de présentation du 1er novembre 2026

- R10 : même header et menu mobile repliable pour le blog et le journal; ancienne navigation secondaire retirée.
- R11 : cartes unifiées sur accueil, catégories et suggestions : personnage, titre, résumé bref, trois thèmes au maximum et durée de lecture.
- R12 : « Ça me parle » réservé à la fin des articles lus; retiré des listes publiques et des articles non aimés dans l’explorateur. Les explications et retraits des articles déjà aimés restent disponibles.
- Vérification : compilation Hugo, contrôle des pages générées, syntaxe JavaScript et recette responsive à 390 pixels et sur ordinateur.

### Simplification demandée le 1er novembre

Le bouton mobile affiche uniquement les trois traits (nom accessible conservé). « Accueil » est le premier lien des menus public et journal. Les cartes commencent par leur titre, présentent le prénom seul en pied de carte à la place de la durée et proposent uniquement « Lire », sans flèche.
