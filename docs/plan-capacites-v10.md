# Ambre V10 — capacités vérifiables et remise à zéro contrôlée

Base auditée : `5eb09cf`. L’audit initial n’a lu aucune donnée privée réelle et n’a exécuté aucune mutation externe. Les compteurs de production ci-dessous ont été fournis par root, sans texte privé. Ce document distingue le code préparé des opérations de production restant à effectuer.

## Défauts démontrés

1. `chat-backend.mjs`, `makeProposal` : `story_request` choisissait le personnage principal, même absent, sans vérifier dix likes, portrait complet ou quota. Le prompt ne recevait pas ces conditions courantes. Le refus intervenait seulement dans `continueCharacter` à la confirmation. Une réponse pouvait donc annoncer son impossibilité tout en affichant une proposition impossible.
2. `reading-intent.mjs` et `recommendations.mjs` : « syndrome du sauveur » n’était pas rapproché de l’abnégation ou du sacrifice de soi. Le mot générique « syndrome » favorisait les articles sur l’imposture ou Stockholm. Repro sans profil : les premiers résultats étaient ces thèmes ; « surresponsabilité » ne donnait aucun résultat. Le catalogue public contient pourtant Thomas/Karpman et trois lectures d’abnégation avec Thomas, Camille ou Sophie.
3. Le modèle connaissait seulement six suggestions personnalisées. Leur absence n’était pas une recherche exhaustive de la bibliothèque, ni la preuve d’absence d’un personnage public. Le personnage personnel à préparer et les personnages des articles publics pouvaient être confondus.
4. Le fil neuf ne contenait aucun accueil déterministe décrivant les capacités et le profil préexistant. Demander cet accueil au fournisseur coûterait inutilement un appel et un message.
5. Les tours du fil n’exposaient pas leur clé d’idempotence au client : après une réponse POST perdue, un message optimiste pouvait être difficile à réconcilier avec le tour déjà enregistré.

## Contrat de capacités

`GET /api/chat` et les réponses POST gardent leurs champs existants et ajoutent `capabilities`. Le modèle reçoit le même contrat vérifié côté serveur. Les valeurs ne proviennent jamais du client ou d’un texte de l’utilisateur.

```json
{
  "source_version": 12,
  "profile": {"has_information": true, "has_description": true, "first_name": "Prénom courant"},
  "self_profile_patch": {"allowed": true},
  "memory_append": {"allowed": true, "remaining_characters": 8000, "max_characters": 2000},
  "person_create": {"allowed": true, "count": 5, "limit": 50},
  "person_update": {"allowed": true, "count": 5},
  "life_event_create": {"allowed": true, "count": 0, "limit": 500},
  "life_event_update": {"allowed": false, "count": 0},
  "reading_lookup": {"allowed": true},
  "story_request": {
    "allowed": false,
    "reasons": ["likes_required", "portrait_required"],
    "likes_count": 1,
    "likes_target": 10,
    "portrait_ready": false,
    "character_ready": true,
    "quota": {"limit": 7, "used": 0, "reserved": 0, "remaining": 7, "renews_at": "ISO"},
    "active_request": null
  }
}
```

Les nombres sont illustratifs. Les compteurs de mémoire suivent les caractères JavaScript comme les validateurs existants, y compris les caractères représentés par deux unités UTF-16. La présence d’informations n’expose aucun inventaire privé dans l’accueil.

| Capacité | Conditions et limites effectives |
|---|---|
| Réfléchir avec Ambre | Consentement conversation, accès actif ou aperçu gratuit, fournisseur configuré, génération non suspendue, cooldown respecté, 100 réponses réussies par semaine ; échecs libérés |
| Lire un article public | Aucun prérequis de dix likes ou de personnage personnel ; IDs, titres, personnages et liens fournis par le catalogue vérifié |
| Modifier sa présentation | Champs autorisés, citation courante, date complète explicite pour une naissance, aperçu modifiable, confirmation et contrôle de révision |
| Préciser une synthèse | Ajout mémoire cité, maximum 2000 caractères, mémoire finale maximum 10000 ; aucune réécriture directe d’un portrait généré |
| Ajouter/modifier un proche | Maximum 50 proches ; modification d’une cible appartenant au compte ; ajouts narratifs assemblés au serveur et limites finales conservées |
| Ajouter/modifier un événement | Maximum 500 événements ; dates valides, dix proches associés au plus, confidentialité et faits non modifiés conservés |
| Demander une histoire personnelle | Dix lectures aimées, présentation personnelle, portrait actuel complet, personnage principal prêt, accès disponible, service disponible, quota de sept, aucune demande active, limite de trente demandes quotidiennes |
| Supprimer un proche, réécrire toute la mémoire ou forcer une synthèse | Aucune opération de ce type exposée par les propositions d’Ambre ; ne pas promettre une exécution par le chat |

Le contrat de capacités n’est pas une réservation. Une limite de caractères dépend aussi du fragment proposé ; les validateurs et les gardes transactionnelles restent la frontière finale. Les contraintes de versions, quotas, confidentialité, idempotence et fournisseurs existantes restent appliquées.

## Application en trois étapes

1. **Avant le modèle** : transmettre les capacités et retirer du schéma les types indisponibles, notamment `story_request`. Le contexte explique les raisons réellement présentes et distingue les personnages publics du personnage personnel.
2. **Après le modèle** : recharger les capacités ; supprimer une opération devenue impossible avant toute persistance. Si une histoire interdite est malgré tout renvoyée, remplacer aussi la prose par une explication déterministe, sans faux succès, avec lectures compatibles disponibles. Conserver les autres propositions personnelles valides.
3. **À la confirmation** : vérifier à nouveau les conditions avant de créer une demande ; `continueCharacter` conserve ses gardes atomiques, parent attendu, quota et clé durable `chat-{proposal_id}`. Un rejeu d’une demande déjà créée réconcilie son résultat sans nouvelle réservation.

Un appel fournisseur ne prouve jamais qu’une écriture a réussi. Une proposition n’est pas une commande, une commande en file n’est pas une histoire publiée, une synthèse en cours n’est pas une synthèse actuelle.

## Recherche du catalogue

`readingLookup(query)` parcourt les métadonnées publiques de tout le catalogue, indépendamment du profil et des évitements anciens. La requête « syndrome du sauveur », « rôle de sauveur » ou « surresponsabilité » utilise un rapprochement éditorial borné vers abnégation, sacrifice de soi et triangle de Karpman.

Le résultat contient `scope: complete_public_catalogue`, `searched_all`, `catalogue_size`, `total_matches`, une liste bornée de correspondances vérifiées et `match_kind`. Les rapprochements sont explicitement `related` et ne diagnostiquent personne. `absence_proven` reste faux : un échec de correspondance lexicale ne démontre pas l’absence sémantique de tout récit possible.

Le contexte du modèle sépare ce résultat global de `reading_candidates`. Seules les suggestions compatibles avec le contexte et les préférences peuvent devenir des cartes `reading_suggestions`. La recherche globale peut établir l’existence d’un récit malgré un évitement enregistré, sans annuler cet évitement pour le recommander.

## Accueil sans coût

Un fil vide expose `welcome: {id: "ambre-welcome-v1", role: "assistant", content}` séparément de `messages`. Le texte est déterministe, visible avant consentement et adapté au prénom courant ainsi qu’à la présence d’informations déjà saisies. Il explique réflexion contextualisée, lectures publiques, propositions confirmables et conditions d’une histoire, puis pose une seule question.

L’accueil n’insère aucun `chat_turn`, ne consomme aucun quota et n’appelle aucun fournisseur. Une réouverture d’un fil existant ne l’ajoute pas à son historique. Les informations préexistantes ne sont ni supprimées ni à saisir une seconde fois.

Seuls les messages utilisateur exposent `client_key = idempotency_key`. `pending_request` expose également cette clé. Les réponses assistant n’en disposent pas. Cela permet de remplacer un affichage optimiste par le tour serveur après une réponse réseau perdue.

## Réinitialisation du seul compte Romain

### État communiqué par root

Un compte cible identifié : quatre tours de conversation, aucune proposition acceptée, une proposition d’histoire en attente, aucune histoire, aucun retour d’histoire, un personnage personnel, un like, cinq proches et aucun événement. Aucune valeur privée n’a été lue pour préparer ce plan.

Cette absence de propositions acceptées permet de conserver intégralement les collections personnelles actuelles sans reconstituer l’origine de leurs champs. Le reset ne doit pas s’appuyer sur le seul prénom pour identifier le compte : root doit utiliser l’ID vérifié du compte concerné et lier toutes les requêtes à cet ID.

### Données à préserver exactement

- `profiles.self_data`, `profiles.data`, consentements métier, `self_revision` ; prénom, genre, naissance et tous les textes manuels.
- `personal_people`, `life_events`, `life_event_people`, avec leurs IDs, champs, révisions, dates, limites de fiction et relations.
- `article_likes` et ses raisons/différences ; `inspiration_selections` historiques.
- `memories.text` et `memories.version`, y compris les précisions explicitement confirmées ou écrites ; seul `memories.proposal`, brouillon généré, peut être vidé.
- Tout retour manuel éventuel et ses métadonnées ; identité de connexion, abonnement, facturation, sessions et budgets observés.

Une proposition acceptée ne rend pas ses données supprimables comme une simple sortie IA : son aperçu a été validé, parfois édité, et un champ peut ensuite avoir été modifié manuellement. `creation_id = chat-{id}` identifie certaines créations ; il ne constitue pas un journal complet de tous les remplacements ou ajouts futurs. Sans preuve de provenance et de révision suffisante, préserver la source canonique entière. Ne pas déduire la provenance depuis les mots du champ.

### Barrières et transaction

1. Relever uniquement les compteurs, IDs techniques, statuts, versions et empreintes des collections du compte cible. Produire côté opérateur une sauvegarde de restauration protégée, sans publier les textes ni les imprimer dans des logs. Vérifier de nouveau zéro proposition acceptée, zéro histoire et zéro retour juste avant le reset.
2. Suspendre la génération et les entrées d’écriture pendant l’opération. Pour ce compte unique, root peut utiliser la pause de génération existante, mais doit aussi empêcher les écritures concurrentes et les tâches déjà parties de republier. Révoquer les baux et utiliser la version globale du profil comme frontière d’invalidation ; les anciennes tâches doivent échouer à la publication.
3. Dans une transaction liée au seul `user_id`, révoquer/annuler les demandes réservées et leurs jobs, puis retirer `journal_jobs`, `generation_jobs`, `context_source_derivatives`, `user_portraits`, `recommendation_exposures`, les propositions de chat et le personnage. Effacer les contenus de conversation.
4. Conserver les tours de chat purgés sous forme de métadonnées comptables (`content=''`, `response=''`, `purged=1`) si leur consommation doit rester dans le quota hebdomadaire. Le fil et l’export seront vides. Une suppression physique des tours sans registre compensateur remettrait implicitement du quota à disposition ; ne pas confondre nettoyage de texte et effacement des coûts.
5. Conserver `provider_budgets`, qui reflète des appels facturés. Avant suppression du personnage, vérifier les réservations/consommations des `story_requests` : sa suppression cascade leurs lignes. Pour ce compte sans histoires, toute consommation non nulle impose une vérification de comptabilité avant suppression ; ne pas réinitialiser silencieusement un quota consommé.
6. Vider `memories.proposal`, sans modifier `memories.text`. Ne supprimer de `context_mutations` que les entrées démontrées propres à un artefact supprimé ; les clés de corrections manuelles conservées empêchent des doubles ajouts et doivent rester.
7. Une éventuelle augmentation de `profiles.version` est une invalidation technique, sans modification des données manuelles ni de leurs révisions propres. Supprimer les jobs après cette invalidation : ses triggers peuvent modifier des jobs existants. La pause doit rester active jusqu’à la fin des vérifications ; aucun appel à un endpoint qui prépare un portrait pendant cette phase.
8. Désactiver la conversation si nécessaire pour redemander un consentement explicite au nouveau fil. Ne pas réactiver un consentement automatiquement. Après commit et vérification, lever la pause ; les futures analyses devront repartir uniquement des sources canoniques préservées et du consentement courant.

Ne pas utiliser `/api/account DELETE`. Ne pas effacer un profil ou une mémoire pour empêcher la recréation d’un job. La pause doit bloquer `sourcesChanged`, les claims natifs/Python et les cron ; les anciens baux invalidés ne doivent plus pouvoir publier.

### Contrainte générale des retours manuels

Le schéma actuel lie `feedback` à `stories` avec `ON DELETE CASCADE`. Un reset général supprimant des histoires ferait donc disparaître les retours manuels. Le compte communiqué a zéro retour, ce qui évite une migration pour cette opération ponctuelle. Si le précontrôle trouve un retour, arrêter le reset prévu et préparer un archivage exact, relié au compte et chargé comme source canonique indépendante avant suppression de l’histoire. Ne pas créer une histoire fictive factice pour satisfaire la clé étrangère.

### Validation avant reprise

- Comparaison exacte des champs manuels et des empreintes avant/après, avec nombre et IDs des lignes ; aucun texte dans le compte rendu public.
- `PRAGMA foreign_key_check` sans anomalie ; zéro proposition, zéro personnage, zéro portrait et zéro dérivé actif ; tous les baux antérieurs inutilisables.
- Fil et export de conversation vides, accueil déterministe présent, prénom courant utilisé sans inventaire privé ; quota et coûts conservés selon la stratégie comptable choisie.
- Un like reste un like et les cinq proches restent identiques ; aucune création de job pendant la pause et aucun succès tardif de l’ancienne génération.
- Après reprise, le seul contexte de personnalisation vient des sources conservées ; aucune ancienne synthèse, histoire ou réponse d’Ambre n’est une preuve.

## Régressions attendues

Tests synthétiques : accueil sans consentement/fournisseur/quota ; profil déjà présent ; 0/1/9/10 likes ; portrait incomplet ou périmé ; personnage absent ; quota épuisé et demande active ; changement de quota durant l’appel ; sortie modèle fautive avec faux succès et autre proposition personnelle valide ; confirmation d’une vieille proposition devenue impossible ; rejeu après réponse POST perdue ; recherche globale sauveur avec catalogue réel et évitement ancien ; absence de correspondance sans fausse preuve d’absence ; mémoire pleine avec caractères UTF-16.

Le reset doit d’abord être répété sur une base synthétique comprenant un autre compte et des jobs en cours. La validation finale n’exige aucune lecture du texte privé de Romain, seulement des comparaisons et métadonnées effectuées côté opérateur.
