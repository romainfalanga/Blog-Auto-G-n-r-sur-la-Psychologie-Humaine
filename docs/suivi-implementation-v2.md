# Reprise de l'implémentation V2 — 3 novembre 2026

Autorisation : Romain a validé le [plan](plan-portrait-et-trois-personnages.md) par « go ». Base Git : `cc6ce8668e026157d744b93392d3d2140d6007f0`.

## Suivi

| Exigences | Travail | État |
|---|---|---|
| R1–R2 | Navigation des cartes et barre de recherche épurée | Implémenté ; recherche, Hugo et navigation réelle validés. |
| R3–R5 | Proches, ligne de vie, sources et portrait | Implémenté et testé ; portrait et préparation IA validés, récit réel encore à vérifier. |
| R6–R8 | Trois personnages, continuités et sept crédits hebdomadaires | Implémenté et testé ; portrait et préparation IA validés, récit réel encore à vérifier. |
| R9 | Retours, classement, génération et provenance | Implémenté et testé ; portrait et préparation IA validés, récit réel encore à vérifier. |
| R10 | Contrats, migration, revue et preuves | Publié ; migration, revue indépendante, CI et contrôles de production validés. |

## Contrats et organisation

Le [contrat V2](contrats-implementation-v2.md) fixe les noms de champs partagés entre Worker, interface et runner. Backend/migrations, interface et génération/recommandations ont des responsables d'écriture distincts. L'intégration et le déploiement restent coordonnés centralement.

La recette utilise une base Cloudflare distincte avec des comptes et histoires synthétiques. Aucune donnée personnelle de production n'y est copiée. Les contrôles incluent une base contenant d’anciennes histoires et leurs retours avant migration.

L'accès Stripe est actuellement non configuré dans les bindings de production ; le chantier conserve les droits d'abonnement existants mais n'effectue aucun achat et n'affirme pas que l'encaissement est activé.

Les versions et limites de recette sont consignées ci-dessous.

## Vérifications avant publication

- 93 tests JS du journal, 16 tests de recherche, 35 tests Python : tous passent. Hugo compile 1 076 pages ; les sources des articles publics restent intactes.
- Dix contrôles API passent sur une D1 isolée avec uniquement des données synthétiques : authentification, migration des archives, crédits historiques, CRUD, conflits concurrents, dates partielles, liens privés, suppression d’un proche sans perte d’événement, export et recommandations sans like.
- Navigateur réel : recherche « rumination », fermeture par Échap sans effacer la requête ni déplacer la carte (1 062,09375 px avant/après), ouverture de Sophie ; recherche « peur de décevoir » puis clic direct avec suggestions ouvertes : page Thomas ouverte.
- Revue indépendante : échec tardif d’un ancien bail et facettes périmées corrigés ; reproductions concurrentes repassées.
- Production : point de restauration D1 enregistré avant opération ; migrations 0005/0006 appliquées sans erreur (50 instructions). Contrôle des clés étrangères vide et compte préexistant conservé. Génération temporairement suspendue pendant le changement de protocole.

Les formulaires privés sont vérifiés par tests d’interface et appels authentifiés au serveur ; leur parcours connecté n’a pas encore été exercé dans le navigateur de cette recette. L’évaluation du classement est synthétique et ne démontre pas de gain général sur la cohorte ayant des likes. Voir la note algorithmique.

## Publication et contrôles réels

- Implémentation Git : `095d15093b77d3b304a3ee66d13a55e8acf08ed5` ; dépendance de validation YAML ajoutée dans `72e10286582ae30c4ea93df26c4abf7da244870a` après le premier passage CI. L’environnement local possédait déjà YAML ; le runner GitHub neuf a révélé l’absence de déclaration de cette dépendance.
- Worker actif : déploiement `2feeb057aa614e7d8174380e76117543`, modules V2 et bindings existants conservés. Génération réactivée.
- Netlify : déploiement initial `6ae9e7ae797054c68ea33f48`, puis `6ae9e83e2f34a036b8ba4b80` sur le commit `72e1028`, tous deux `ready`, sur le domaine existant.
- [CI finale](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/41648990943) : succès, 144 tests au total. La première CI a échoué sur l’import YAML ; la correction est vérifiée sur un runner neuf.
- [Génération V2 réelle](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/41648627110) : succès. Un compte fictif avec une relation décrite, un événement positif et zéro like a obtenu un portrait `ready` puis trois personnages. Trois appels pour le portrait, un pour les fiches, sans erreur ni nouvelle tentative. Compteurs fournisseur du seul compte de recette : 8 762 tokens d’entrée, 4 167 de sortie ; coût enregistré `0.0105245` par la passerelle. La préparation conserve les sept crédits (zéro consommé/réservé).
- Navigateur en production : « rumination » → suggestion « rumination mentale » → « Découvrir ses histoires » ouvre `/personnages/sophie/`. Le menu et la page `/mes-personnages/` affichent le nouveau nom et l’offre hebdomadaire.
- Catalogue D1 : 231 articles publics resynchronisés depuis le même constructeur que les métadonnées. Contrôle des clés étrangères sans anomalie après nettoyage.
- Compte fictif de production et son droit temporaire supprimés. Retour à un seul compte, celui qui préexistait. Aucun achat effectué. Aucun récit de test demandé ni crédit consommé.

## Vérifications restant à effectuer

Le contrôle automatique d’approbation a rejeté la poursuite du test qui devait demander une histoire réelle : il peut engager un coût fournisseur et créer un récit en production. Le processus de test a été arrêté ; la base confirme l’absence de demande. Un accord explicite pour ce test payant est nécessaire avant reprise. Il ne faut pas présenter la qualité narrative ou la publication complète d’un récit V2 comme déjà vérifiées chez le fournisseur. Elles sont couvertes localement par les scénarios simulés, y compris continuité, correction, quotas et publication concurrente.

L’encaissement reste désactivé (`billing:false`) faute de bindings Stripe. Le test complet des formulaires privés dans un navigateur connecté et une évaluation éditoriale indépendante restent aussi à réaliser. Le portrait et les trois fiches ont, eux, été vérifiés par les API authentifiées et la chaîne Actions/OpenRouter réelle.
