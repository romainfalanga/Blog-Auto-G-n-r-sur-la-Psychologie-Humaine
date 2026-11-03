# Reprise de l'implémentation V2 — 3 novembre 2026

Autorisation : Romain a validé le [plan](plan-portrait-et-trois-personnages.md) par « go ». Base Git : `cc6ce8668e026157d744b93392d3d2140d6007f0`.

## Suivi

| Exigences | Travail | État |
|---|---|---|
| R1–R2 | Navigation des cartes et barre de recherche épurée | Implémenté ; recherche, Hugo et navigation réelle validés. |
| R3–R5 | Proches, ligne de vie, sources et portrait | Implémenté et testé ; recette IA réelle en cours. |
| R6–R8 | Trois personnages, continuités et sept crédits hebdomadaires | Implémenté et testé ; recette IA réelle en cours. |
| R9 | Retours, classement, génération et provenance | Implémenté et testé ; recette IA réelle en cours. |
| R10 | Contrats, migration, revue et preuves | Contrats, revue indépendante et migration de recette validés ; publication en cours. |

## Contrats et organisation

Le [contrat V2](contrats-implementation-v2.md) fixe les noms de champs partagés entre Worker, interface et runner. Backend/migrations, interface et génération/recommandations ont des responsables d'écriture distincts. L'intégration et le déploiement restent coordonnés centralement.

La recette utilise une base Cloudflare distincte avec des comptes et histoires synthétiques. Aucune donnée personnelle de production n'y est copiée. Les contrôles porteront aussi sur une base contenant d'anciennes histoires et leurs retours avant migration.

L'accès Stripe est actuellement non configuré dans les bindings de production ; le chantier conserve les droits d'abonnement existants mais n'effectue aucun achat et n'affirme pas que l'encaissement est activé.

Les résultats et versions de publication seront ajoutés après validation effective.

## Vérifications avant publication

- 93 tests JS du journal, 16 tests de recherche, 35 tests Python : tous passent. Hugo compile 1 076 pages ; les sources des articles publics restent intactes.
- Dix contrôles API passent sur une D1 isolée avec uniquement des données synthétiques : authentification, migration des archives, crédits historiques, CRUD, conflits concurrents, dates partielles, liens privés, suppression d’un proche sans perte d’événement, export et recommandations sans like.
- Navigateur réel : recherche « rumination », fermeture par Échap sans effacer la requête ni déplacer la carte (1 062,09375 px avant/après), ouverture de Sophie ; recherche « peur de décevoir » puis clic direct avec suggestions ouvertes : page Thomas ouverte.
- Revue indépendante : échec tardif d’un ancien bail et facettes périmées corrigés ; reproductions concurrentes repassées.
- Production : point de restauration D1 enregistré avant opération ; migrations 0005/0006 appliquées sans erreur (50 instructions). Contrôle des clés étrangères vide et compte préexistant conservé. Génération temporairement suspendue pendant le changement de protocole.

Les formulaires privés sont vérifiés par tests d’interface et appels authentifiés au serveur ; leur parcours connecté n’a pas encore été exercé dans le navigateur de cette recette. L’évaluation du classement est synthétique et ne démontre pas de gain général sur la cohorte ayant des likes. Voir la note algorithmique.
