# Plan et suivi — Mon compte, timelines et Ambre

Base : main 5c1a234 ; branche codex/compte-timelines-ambre.
L’accès gratuit reste actif jusqu’à configuration du paiement (choix explicite de Romain).

| Exigence | Résultat observable | Responsable | État |
|---|---|---|---|
| R1 | Mon compte dernier menu, quatre champs identité, abonnement et actions compte préservés | UI + intégration | Réalisé, vérifié localement |
| R2 | 100 messages et 1 bilan / mois civil Paris, réservation atomique et compteurs serveur | Accès | Réalisé, vérifié localement |
| R3 | Événement simplifié, contrôles temporels adaptés et notes datées multiples | Données + UI | Réalisé, vérifié localement |
| R4 | Timeline proche : rencontre, événements communs, notes personnelles et Ambre attribuées | Données + UI | Réalisé, vérifié localement |
| R5 | Bulle Ambre globale, panneau mobile/ordinateur, navigation/session cohérentes | Intégration + UI | Réalisé, vérifié localement |
| R6 | Identité complète, 3 proches, 3 événements et toutes les questions enfance avant génération | Accès | Réalisé, vérifié localement |
| R7 | Description résultat immédiate, détails utiles et espacement mobile | Enfance | Réalisé, vérifié localement |
| R8 | Fenêtres de précision/manifestation préremplies avec sauvegarde explicite | Enfance | Réalisé, vérifié localement |
| R9 | Retrait des mentions répétées demandées, anciennes données conservées | Enfance + UI | Réalisé, vérifié localement |
| R10 | Affichage prêt, navigation sans ancien contenu, vérifications mobile/ordinateur | Tous + intégration | Recettes locales réussies ; vérification distante à faire |

Ordre : contrats et migrations additives → implémentations séparées → intégration → tests du projet et recette navigateur → relecture indépendante → sauvegarde Git et déploiement vérifié selon autorisation existante.

Décisions : quota commun au mois civil Europe/Paris, bilan initial inclus dans le bilan mensuel mais jamais dans les messages ; budgets techniques hebdomadaires séparés. Pas d’activation Stripe sans configuration. Rencontres sans date restent sans date inventée. Notes Ambre existantes et proposées gardent attribution, citation et validation individuelle. Aucune suppression des anciens paragraphes stockés. Ancienne URL portrait redirigée vers compte.

## Vérification avant publication

Relectures indépendantes : [données et accès](relecture-compte-timelines-donnees-2026-11-17.md), [conversation embarquée](relecture-compte-timelines-widget-2026-11-17.md). Les courses avant fournisseur et les défauts de focus/session relevés sont corrigés et contre-vérifiés.

Recettes locales : 113 tests UI, 19 parcours affichage/navigation, 12 parcours formulaires et timelines aux quatre largeurs, 55 parcours Mon enfance et 4 largeurs pour la bulle. Les scénarios couvrent aussi dates partielles, CAS des notes, brouillons, conservation des champs historiques et premier affichage prêt. La suite complète serveur et la livraison distante sont consignées dans la note de livraison finale.

Validation finale locale : `npm --prefix journal test` : **838/838** ; `node --test tests/*.test.cjs` : **46/46** ; toutes les commandes de `.github/workflows/verify.yml` exécutées, dont **43 + 13 + 21 tests Python**, vertes. Génération de navigation cohérente, vérification de syntaxe et `git diff --check` sans erreur. Aucun appel modèle distant pour ces tests.
