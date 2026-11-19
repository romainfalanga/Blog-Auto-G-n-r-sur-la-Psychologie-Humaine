# Relecture indépendante des instantanés de bilans — 19 novembre 2026

## Périmètre et verdict initial

Audit de `ef58d1a` et de l’état de travail visible : `review-snapshot.mjs`, intégration dans `chat-backend.mjs`, migration 0031, onze tests dédiés et `contrat-instantanes-bilans-2026-11-19.md`. Aucun code de production modifié. Les tests dédiés existants passent (11/11). Un test supplémentaire temporaire a reproduit une course de retrait du consentement avant l’envoi fournisseur ; il a été supprimé après exécution et transmis à l’agent chargé de la correction.

**Réserve bloquante avant acceptation : revalider l’autorisation après la dernière lecture des sources et avant l’appel fournisseur.** Une deuxième réserve concerne l’accès aux évaluations des bilans non validés. Les autres axes examinés ont des protections cohérentes ; les limites de portée et de fidélité décrites plus bas ne doivent pas être présentées comme des garanties absolues.

## Résultats positifs vérifiés

- Capture des rubriques en une requête SELECT : l’état de données n’est pas assemblé par des lectures successives indépendantes. Tri déterministe des lignes et comparaison de la représentation canonique avant les étapes importantes.
- Capture, écritures d’étapes et publication conditionnées à l’empreinte des sources. Une modification après un bloc empêche les appels suivants et le texte final ; scénario exécuté par les tests.
- Archive immuable via trigger UPDATE, mais supprimable : cascade de suppression du bilan et du compte ; suppression lors du retrait du consentement ; suppression à la nouvelle génération du plan ; invalidation `sources_changed`/`consent_withdrawn`. Le test de frontière d’oubli a été exécuté avec succès.
- Conversation limitée aux messages terminés, non purgés, de génération courante ; le tour généré par le bilan en cours est exclu. Les dépendances héritées du plan, des tours et des bilans sont enregistrées avant transmission.
- Relecture de propriété/présence/consentement après chargement de l’archive dans l’API ; refus de la copie périmée lors d’une suppression/retrait intercalé. Isolation entre comptes vérifiée.
- Export complet des archives avec vérification de l’énumération après lecture ; un retrait intercalé impose de recommencer. Plafond 16 Mio explicite, pas d’omission silencieuse annoncée comme export complet.
- Limites de taille avant fournisseur : 512 Kio sources puis payload final avec libellés ; 10 000 lignes par rubrique avec lecture d’une ligne supplémentaire pour détecter le dépassement. Le refus trop volumineux n’appelle pas le fournisseur.
- Tous les fragments textuels sont inclus dans les blocs avec chemin et offsets. Le dernier élément au-delà de 65 000 caractères est effectivement envoyé. Les limites de synthèse produisent une erreur explicite.
- Réservations hebdomadaires/journalières, contingent de bilan et contrôle d’accès réutilisés pour chaque appel de bloc, rédaction et relecture. Les refus explicitement non consommés remboursent les réservations ; les appels effectivement envoyés restent comptabilisés.
- Reprise après cooldown vérifiée : les blocs sauvegardés ne sont pas réanalysés. Une progression sauvegardée peut reprendre sans perdre le travail après échéance de la fenêtre d’exécution.
- Publication texte/message en batch, sous lease et conditions de sources. Brouillon effacé à l’échec définitif. Les tests de sécurité datée et de citations avec échappement passent.

## R1 — course de consentement avant fournisseur (priorité haute)

Dans `reviewCall`, le dernier contrôle `providerGuard` précède `await readSnapshotSource`. Cette dernière lecture peut rendre la main alors qu’une autre requête retire le consentement et invalide le bail. Si les sources personnelles de la requête canonique restent identiques, la comparaison réussit puis `fetch` transmet tout de même le bloc. `chat_settings` ne fait pas partie de cette empreinte.

Reproduction exécutée avec le harness réel SQLite et fournisseur mock :

1. Utilisateur prêt, consentement actif, POST de bilan.
2. Wrapper de `env.DB.prepare` sur le SELECT `WITH owner AS…` ; après la lecture, uniquement une fois un appel réservé dans `provider_budgets`, retirer le consentement et passer le bilan courant à `failed/consent_withdrawn`, bail nul.
3. Retourner le payload lu précédemment au code appelant.
4. Observer les appels mock.

Résultat : `{raced:true,calls:1,consent:0}`. L’assertion `calls === 0` échoue. Le texte final n’est pas publié grâce aux gardes suivantes, mais cette protection arrive trop tard pour empêcher la transmission privée.

Correction recommandée et transmise : après comparaison des sources, exécuter de nouveau la vérification d’accès et la garde de lease/consentement/prérequis/réservation, immédiatement avant `fetch`, dans le même bloc de gestion qui rembourse les réservations si aucun appel n’est envoyé. Ajouter le test de course permanent. Cela ferme la fenêtre supplémentaire introduite par la lecture des sources ; cela ne doit pas être formulé comme une atomicité transactionnelle impossible entre D1 et un fournisseur externe.

## R2 — évaluation accessible avant validation du bilan (priorité moyenne)

L’API snapshot vérifie le propriétaire et le consentement, mais ne requiert pas `ambre_reviews.status='ready'`. Elle renvoie donc le payload et l’évaluation déterministe pendant un bilan `running`, et après certains échecs (notamment `context_changed`, qui ne supprime pas l’archive via le trigger actuel). Le filtrage des bilans prêts dans l’interface ne protège pas l’API directe.

Ce n’est pas une fuite intercomptes : les données restent celles du propriétaire. Cela contredit toutefois le contrat de recherche qui prévoit la restitution de l’évaluation avec le bilan validé. Recommandation : maintenir la capture en interne pour la reprise, mais renvoyer un état explicite non disponible/409 tant que le bilan n’est pas prêt, y compris au recontrôle final. Si l’export doit inclure les captures de bilans échoués, ajouter leur statut et les identifier comme telles ; sinon limiter explicitement l’export publié aux bilans prêts. Ne pas supprimer une archive utile à la reprise simplement pour masquer le défaut d’API.

## Limites et améliorations non bloquantes

### Exhaustivité technique versus fidélité de l’analyse

`complete:true` atteste la couverture de la liste de tables et l’envoi de chaque fragment. Un résumé de bloc limité à 800 caractères ne peut garantir que chaque nuance de milliers de caractères subsiste dans la synthèse finale. Les citations sont vérifiables, mais leur exactitude n’atteste pas la complétude sémantique du résumé. Le contrat et les textes utilisateur doivent parler de « toutes les rubriques parcourues », pas promettre « aucun fait omis dans l’interprétation ». L’évaluation déterministe utilise bien les réponses originales plutôt que ces résumés.

La liste de tables est fermée mais les colonnes sont découvertes dynamiquement et filtrées par liste d’exclusion. Les colonnes sensibles actuelles sont exclues ; une future migration pourrait ajouter une colonne privée nouvelle à une table autorisée sans être ajoutée à `HIDDEN`. Une liste explicite de colonnes par table, ou un test de contrat de colonnes, réduirait ce risque d’évolution.

### Budgets et reprises

Le plafond de 40 000 caractères du synopsis est vérifié après l’analyse de tous les blocs. C’est honnête et n’entraîne pas de publication partielle, mais peut faire échouer un dossier après plusieurs appels payés. Une borne préalable conservative fondée sur le nombre de blocs et les références permettrait de signaler plus tôt les dossiers forcément trop volumineux, sans tronquer les données.

La reprise fait confiance aux indices continus des résultats stockés et au snapshot immuable ; c’est suffisant contre une interruption ordinaire. Le hash/version de l’algorithme de découpage n’est pas stocké avec chaque progression. Si cette logique change lors d’un déploiement alors qu’un bilan est en cours, une reprise devrait vérifier une version de découpage ou les références/longueurs contre les blocs recalculés, plutôt que se fier uniquement aux indices.

### Portée de l’oubli

Les triggers testés couvrent le retrait du consentement et la frontière explicite de génération. Une édition ordinaire laisse volontairement les archives historiques intactes. Les parcours qui promettent « oublier » doivent continuer à utiliser cette frontière, pas seulement modifier/supprimer une ligne source. Cette revue vérifie les mécanismes et les parcours testés ; elle ne prétend pas avoir simulé toutes les mutations personnelles possibles.

## Conditions de clôture

- Test R1 permanent vert après correction ; aucun envoi fournisseur dans la course reproduite.
- Décision explicite et tests R2 sur l’accès direct aux captures running/failed, cohérente avec la publication des évaluations.
- Les onze tests existants restent verts ; contrôle d’intégration et suites générales par le contrôleur.

Pas de déploiement ni de mutation de production effectué dans cette revue.

## Corrections contre-testées

Commit 947c186 : garde accès/consentement/lease vérifiée après la lecture des sources et avant la transmission, remboursement si annulée ; archives publiques réservées aux bilans ready ; export des autres états explicitement étiqueté sans évaluation. Quatorze tests dédiés passent, dont les courses de retrait de consentement et de perte de lease. Suite intégrée : 948 tests serveur et 62 tests site verts. Les réserves de fidélité d’une synthèse de blocs restent des limites : couverture des données ne signifie pas validation clinique complète.
