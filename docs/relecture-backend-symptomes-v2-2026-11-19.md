# Relecture indépendante du backend Mes symptômes v2 — 19 novembre 2026

Périmètre : `cartographie.mjs`, intégration Worker, migration 0029 et retrait différé 0030, partage professionnel, export/suppression et articulation de la sécurité avec Ambre. Relecteur distinct de l'auteur du backend.

## Vérifications favorables

- Table limitée aux deux noms autorisés par construction ; aucune table issue du corps HTTP. Identifiants de personne liés à la session et paramètres SQL liés, y compris DELETE. L'export utilise uniquement l'identifiant du compte authentifié.
- Sauvegarde entière validée : identifiants connus, types de réponses, notes bornées, version attendue entière. `2` est refusé pour symptômes et reste accepté pour l'historique Mon enfance. Absence d'accord explicite conforme à la demande ; les dates de sécurité soumises par le client sont ignorées.
- Première création protégée par conflit unique ; mises à jour protégées par version dans le WHERE. Les horodatages et la version proviennent du serveur. Les notes d'Ambre de Mon enfance ne sont pas modifiées par ce backend.
- Migration 0029 additive, clé étrangère avec cascade, objets JSON vérifiés, révisions de partage incrémentées à insertion/modification/suppression. Migration 0030 est séparée et documentée après vérification du Worker sans v1.
- `symptoms` est un périmètre de partage explicite. Il n'est pas ajouté aux partages historiques. Le dossier passe par autorisation, journalisation et seconde vérification après lecture ; un retrait de périmètre concurrent invalide la réponse.
- Le classement respecte les familles, le prérequis produit, l'incertitude et le rangement des éléments écartés. La logique Mon enfance reste une branche distincte.

## Constats transmis et suivi

1. **Sécurité persistante** : le premier état expirait définitivement 14 jours après `safety_at`, même avec un Oui persistant. Corrigé par l'auteur Ambre : un Oui actif conserve le drapeau ; `safety_asked_at` espace les questions. Test ajouté et réussi.
2. **Horodatage avec notes** : l'écriture de `safety_asked_at` devait accepter la réponse avec annonces de notes autant que l'enveloppe simple. Corrigé : `response IN (envelope, noted)` dans la même transaction.
3. **Course publication** : le hash de contexte inclut la révision symptômes, mais une modification entre la lecture du snapshot et la transaction de publication doit également être refusée par une garde SQL sur la version/absence attendue. Corrigé et contre-vérifié : garde SQL owner/version/dates/révision monotone des sources, avec absence attendue si nécessaire. La révision monotone empêche aussi DELETE/recréation dans la même seconde de réutiliser un contexte périmé. Le trigger exclut `safety_asked_at` pour que la publication de sa propre question ne périme pas ses traces.

## Preuves

`journal/tests/symptom-map-review.test.mjs` ajoute deux tests indépendants : partage explicite, isolation entre comptes, révocation de périmètre pendant la lecture et rejet des noms de table injectés. Exécution conjointe avec `symptom-map-v2.test.mjs` et `symptomes-ambre.test.mjs` : **15/15 réussis**. Journal local : `work/reprise-nov19/backend-review-tests.log`.

Les tests du véritable Worker sur la limite de corps, l'absence de session et l'export/suppression complets ont été demandés à l'auteur backend. Ce rapport ne remplace pas la validation finale intégrée ni la recette distante.

## Contre-vérification finale

Avis favorable sur le périmètre relu. `symptom-map-publication-review.test.mjs` provoque trois courses immédiatement avant la transaction réelle de publication : modification, suppression/recréation à version et seconde identiques, création après snapshot absent. Dans les trois cas, le travail échoue sans publier de réponse périmée. Le partage professionnel concurrent et les gardes de sécurité restent verts. Exécution finale groupée : **20/20 réussis**, `work/reprise-nov19/backend-review-tests-final.log`. Aucun appel fournisseur réel. Les contrôles globaux et la recette distante restent à exécuter par l’intégrateur.
