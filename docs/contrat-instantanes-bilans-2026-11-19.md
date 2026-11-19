# Contrat des instantanés de bilans

Migration 0031 additive : un instantané par nouveau bilan, propriétaire et bilan en clés étrangères avec suppression en cascade. Aucun instantané rétroactif n'est créé pour les bilans déjà publiés. `payload` et `assessment` sont immuables ; les retraits explicites et les frontières d'oubli peuvent les supprimer.

## Capture et analyse

La capture est une requête SQL unique sur une liste fermée de tables personnelles. Aucun secret, session, paiement, journal fournisseur ou catalogue public n'entre dans cette liste. Les colonnes JSON deviennent des objets dans `payload.sections`. Les libellés des deux cartes sont figés dans `question_labels`. Les créations personnelles de fiction restent identifiées comme fiction, pas comme témoignages.

Le contexte conversationnel respecte la génération courante et exclut les messages purgés. Les dépendances de tous les objets du plan et des messages/bilans hérités sont inscrites avant l'appel fournisseur. Un retrait ou une correction qui impose l'oubli purge également l'archive ; « immuable » ne signifie pas « indestructible ».

La capture complète est limitée à 512 Kio de sources et 10 000 lignes par rubrique. Un dépassement refuse explicitement le bilan (`snapshot_too_large`), sans masquer des données. Chaque item est analysé dans des blocs d'au plus environ 20 000 caractères ; un texte long est fragmenté avec chemin, offset et longueur. Les résultats intermédiaires gardent leurs références, résumés et citations vérifiées dans `ambre_reviews.draft.snapshot_chunks`. Les reprises ne réanalysent pas les blocs terminés. La synthèse finale reprend tous les résultats, avec une limite explicite de 40 000 caractères (`snapshot_analysis_limit` si dépassée), sans suppression silencieuse de blocs.

Tous les appels passent par les réservations et budgets existants. La comparaison de l'état source canonique est faite avant la capture, avant chaque appel fournisseur, dans les écritures des étapes et dans la publication. Une modification intercalée fait échouer le bilan avec `context_changed`, sans mélanger plusieurs dates. Le texte final n'est publié qu'après relecture.

Le moteur déterministe `buildClinicalAssessment(payload)` est appelé sur l'intégralité du snapshot, indépendamment des résumés du fournisseur. Il ne produit pas un diagnostic professionnel.

## API propriétaire

`GET /api/reviews/:id/snapshot` :

- `available`, `review_id`, `captured_at` (secondes Unix), `schema_version`, `complete`, `review_status`, `assessment` ;
- `sections: [{id,count}]` ;
- absence historique : `{available:false, reason:'legacy_review'}` ; archive retirée : `snapshot_removed`.

`?section=people&cursor=0&limit=50` ajoute `items`, `total`, `next_cursor` (nombre ou null). Limite de page 1 à 100. Les cartes rendent des réponses et retours avec libellés figés et notes associées ; les notes sans réponse restent présentes. Les relations d'événements, notes et observations contiennent les noms/titres capturés. Les références techniques restent dans le stockage, sans constituer des noms d'affichage.

La réponse privée est recontrôlée après lecture : propriétaire, bilan, présence de l'archive et consentement actuel. Un retrait concurrent renvoie 409, pas une copie périmée.

Export : `review_snapshots` inclut payload et assessment complets. L'énumération est vérifiée de nouveau après lecture ; une suppression, un ajout ou un retrait de consentement intercalé impose de réessayer. Au-delà de 16 Mio, l'export refuse explicitement et les archives restent consultables par bilan ; aucun instantané n'est omis en prétendant fournir un export complet.

## Vérifications dédiées

`journal/tests/review-snapshot.test.mjs` : dernier élément au-delà de 65 000 caractères effectivement transmis, capture immuable, provenance des blocs, reprise après refus fournisseur sans répétition, refus sans appel au-delà du plafond, source modifiée entre appels, propriétaire et pagination, export/suppression, absence de secrets, sécurité datée, citation avec échappement, retrait pendant lecture/export et frontière d'oubli. Les tests historiques de la période pré-0031 gardent explicitement leurs anciennes enveloppes.
