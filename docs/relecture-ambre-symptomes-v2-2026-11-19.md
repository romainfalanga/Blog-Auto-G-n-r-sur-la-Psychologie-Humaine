# Relecture indépendante — Ambre et Mes symptômes v2

Relecteur : agent backend_v2, distinct de l'auteur ambre_v2. Périmètre : symptom-map-context, intégration moteur conversation, bilans, consignes, sécurité et budgets. Relecture du code et tests locaux ; pas une validation clinique humaine ni un essai fournisseur distant.

## Constats et corrections vérifiées

- Une précision libre sur une question sans réponse disparaissait de l'index. Elle est désormais présente, explicitement « Non renseigné », sans être assimilée à Non. Test dédié réussi.
- Une question de sécurité déjà rédigée pouvait empêcher l'ajout des ressources. La garde conserve la question et ajoute les ressources vérifiées si le 3114 manque. Test dédié réussi. La phrase déterministe n'invente plus une déclaration « récente » à partir d'une période de douze mois.
- Les sujets indexés contiennent désormais également les liens aux réponses Non, incertaines et absentes, et leurs libellés ; les informations manquantes ne sont pas déduites d'une simple liste de Oui.
- En coordination avec la relecture backend indépendante : contrôle de révision des symptômes dans le hash et les gardes SQL de publication conversation/bilan. L'époque monotone `care_source_revisions` couvre effacement puis recréation avec version/date identiques. La modification de `safety_asked_at` seule n'invalide pas les sources.

## Contrats relus

Les consignes de rédaction et de relecture autorisent une hypothèse nommée avec ses appuis et ce qui manque, interdisent un diagnostic affirmé, un score ou une probabilité chiffrée. L'orientation systématique a été retirée ; danger, forte gêne et demande de la personne restent des motifs d'orientation. Les contrôles de danger actuels restent prioritaires.

Les deux cartes disposent de blocs fixes distincts (3 500 caractères symptômes, 3 000 enfance) et d'un index pour les détails. Les bilans ont une part symptômes dédiée, réduite de 6 000 à 3 000 si nécessaire ; le contexte reste borné à 40 000 caractères. Les plafonds de budget sont vérifiés avec les nouvelles décisions documentées, sans activation de dépense quotidienne.

Le premier signal de sécurité est fixé par le serveur. Le délai de quatorze jours part d'une question publiée (`safety_asked_at`), pas d'une lecture ou d'un appel fournisseur. La mise à jour est dans le batch de publication, conditionnée à une réponse effectivement enregistrée. Les gardes existantes de consentement, bail, prérequis et réservation mensuelle sont conservées. Les erreurs ou annulations sans réponse publiée ne passent pas cette condition. Une réponse de secours effectivement publiée avec la question peut légitimement démarrer le délai.

## Vérification et avis

Exécution indépendante : `symptomes-ambre.test.mjs`, `symptom-map-v2.test.mjs`, `symptom-map-worker.test.mjs` : **16/16 réussis**, journal `work/reprise-nov19/review-s3.log`. Cela comprend les deux contextes, notes libres, ressources, question et horodatage publiés, délai, budget sur profil rempli, authentification, limite du corps et export.

La relecture performance/backend distincte ajoute trois courses réelles juste avant le batch : modification, effacement/recréation à date identique, création après état absent. Les trois empêchent la publication obsolète ; voir `symptom-map-publication-review.test.mjs` et le rapport backend.

Avis favorable sur ce périmètre après les corrections ci-dessus. La validation globale d'intégration, des profils maximaux et du déploiement reste celle de l'agent intégrateur ; aucun test local ne prouve à lui seul l'état du fournisseur en production.
