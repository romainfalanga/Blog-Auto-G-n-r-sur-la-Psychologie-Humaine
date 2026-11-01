# Recommandations et portrait dès la première lecture aimée

## Objectif et parcours

Une lecture aimée suffit à produire immédiatement des repères de lecture et une sélection d'articles. Le portrait IA commence également dès ce premier choix, sans abonnement ni case d'accord séparée. Les dix articles restent uniquement le seuil d'accès à l'abonnement et aux histoires personnelles. Une explication visible près du bouton de lecture informe que ce choix lance la personnalisation et que les analyses IA utilisent OpenRouter; aucune case supplémentaire.

Un choix de fiction est un indice d'affinité, jamais une preuve de biographie, de maladie ou de trait certain. Une explication explicite et une correction comptent davantage qu'un simple rapprochement de thèmes. Une similarité calculée n'est pas une probabilité que l'article plaise.

## Mécanisme retenu

1. Préparer une représentation des 231 articles : sujet, thèmes, titre, description et repères du personnage. Les thèmes rares comptent davantage que les termes présents partout. Accents, pluriels et variantes françaises courantes sont rapprochés.
2. À chaque demande, reconstruire les affinités à partir de la collection actuelle, des explications, des corrections et des retours sur les récits. Un article retiré disparaît des sources. Les phrases exprimant un désaccord ne sont pas traitées comme des préférences positives.
3. Combiner un socle durable et des signaux récents. Une lecture ancienne garde du poids; une précision récente aide à adapter les prochaines propositions. Le portrait IA n'entre dans le classement que s'il correspond à la version actuelle des sources.
4. Classer les articles non aimés puis diversifier la sélection : éviter une série de quasi-doublons, limiter la répétition d'un même personnage, et réserver une place à un autre personnage ayant des thèmes proches. Un prénom identique ne suffit pas à placer une histoire en tête : la continuité doit être étayée par des points communs ou des explications.
5. Montrer une raison courte et vérifiable pour chaque suggestion. Ne pas exposer de commentaires privés dans les cartes publiques. Les recommandations personnelles sont servies par une API authentifiée et non mise en cache publiquement.
6. Afficher tout de suite des repères provisoires sourcés. L'IA approfondit ensuite les axes, les préférences de lecture, l'état actuel explicitement évoqué et l'évolution. Elle doit réviser ses hypothèses anciennes, pas les accumuler comme des faits.
7. Réutiliser le workflow GitHub et le générateur historique pour les portraits puis les récits. Les nouvelles sources invalident une analyse ou une génération en cours. Les récits restent réservés aux abonnés, avec personnage fictif stable et un maximum quotidien.

## Coût, cohérence et disponibilité

Le classement immédiat ne fait aucun appel fournisseur. Le catalogue et ses statistiques sont préparés une fois; aucune donnée d'un utilisateur n'est conservée dans un état global partagé. Les changements rapprochés sont regroupés avant les analyses IA. Les comptes gratuits disposent de trois démarrages d'analyse IA par 24 heures et d'au moins dix minutes entre deux analyses; leurs repères et recommandations restent recalculés à chaque changement. Les abonnés conservent leurs actualisations et récits avec les limites de tâches existantes. Une alternance évite que les analyses gratuites bloquent les histoires des abonnés.

Les instantanés SQLite associent sources et version. Le serveur rejette les résultats obsolètes. L'interface ignore les réponses plus anciennes et garde les explications non enregistrées. Les suppressions de compte doivent aussi supprimer les données de suivi et quotas.

## Critères de validation

| Exigence | Preuve attendue |
|---|---|
| Retrait du dernier like | Aucune suggestion personnelle ni ancien portrait présenté comme actuel, tâches invalidées |
| Premier like | Suggestions non vides et repères provisoires sans paiement ni accord séparé |
| Pertinence | Un article thématiquement proche passe devant une histoire sans rapport du même personnage |
| Explication | Une précision sur un sous-thème fait remonter des articles correspondants, y compris d'autres personnages |
| Désaccord | Une négation ou correction n'est pas renforcée comme intérêt positif |
| Évolution | Une précision récente peut déplacer le classement sans effacer toutes les sources anciennes |
| Diversité | La première sélection évite les doublons et la monopolisation par un personnage |
| Isolation | Aucune source ou suggestion personnalisée d'un autre compte n'est accessible |
| Concurrence | Une analyse commencée avant une nouvelle source ne peut être publiée comme actuelle |
| Abonnement | Premier portrait possible gratuitement; récits privés et paiement restent soumis aux règles de souscription |
| Automatisation | Exécution réelle GitHub avec portrait issu d'un seul article sur un compte fictif |

## Suivi

Plan relu avant intégration; implémentation et vérifications en cours. Les tests permettent de vérifier les invariants et cas de pertinence connus. La qualité des recommandations devra ensuite se mesurer avec les retours réels; aucune méthode ne permet de certifier à l'avance qu'une sélection sera la meilleure pour chaque personne.
