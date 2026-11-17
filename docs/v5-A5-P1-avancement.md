# Lot A5-P1 — avancement (bilans)

## Fait
- Migration `journal/migrations/0023_review_digest.sql` (additive) : `digest`, `feedback`, `stage`, `draft` sur `ambre_reviews`. Sonde `reviewColumnsReady` (`review-digest.mjs`) : avant la migration, bilans relus et publiés sans `digest`, fil = extrait daté, retour → 503 `feature_unavailable`.
- Nouveau module `journal/src/review-digest.mjs` : schémas, validation du `digest` (citation exacte via `quote-guard`, sinon élément retiré et compté), prose lisible, fil des bilans (1 200 / 3 000 car.), contrôles déterministes, verdict de relecture, suggestion de point.
- Bilan structuré + relecture séparée (`chat-backend.mjs` `runReview`) : écriture (4 800 jetons max) → relecture `ambre_review_check` (modèle critique, 1 500 jetons) → une réécriture au plus → sinon `failed/review_not_approved`, rien publié ; brouillon conservé entre étapes (`stage`, `draft`) sous la même garde que la publication ; bail en ligne +30 s, cron 180 s.
- Contexte du point : `review_thread` (3 000), `previous_feedback`, `memory` (objectifs, essais, alliance, séance, sujets fermés, sécurité), `change_plan`, `care`.
- Tour : `review_thread` remplace `last_review`, part fixe de 1 200 car. hors détails ; empreinte et clé d'index incluent `digest` et `feedback` (12 bilans lus en une requête bornée) ; `review_suggestion` (séance dans 48 h) ; consigne mise à jour.
- Retour de la personne : `PUT /api/reviews/:id/feedback` ; page Mes bilans (« Ce qu'Ambre retient de ce bilan », trois boutons, texte facultatif 600) ; Paramètres (« Ce qu'Ambre retient de vos bilans ») ; export (`digest`, `digest_text`, `feedback`, `ambre_review_thread`) ; effacement de la conversation vide `digest`/`feedback`/`draft`.
- Suggestion de point dans `journey.review.suggestion` et sous « Faire le point ».
- Plafond supplémentaire : 8 bilans créés par jour (un bilan refusé par la relecture n'est pas publié, donc pas compté dans les 4 publiés).
- Tests : `v5-A5-P1-reviews.test.mjs` (11) ; `v5-A5-budget` (+10 : tour avec fil complet, coût du bilan) ; `v4-R-export` (+1) ; `v4-R-migrations` (+1, attentes 0022+0023) ; adaptés : `journey` (relecture répondue à part, `max_tokens` 4 800, `sources.check`), `ambre-engine` et `v5-A5-sources` (`review_thread` au lieu de `last_review`), `ui-v2` (liste de fonctions), `v5-A-helper` (digests, relecture mesurée).
- README (migration, bilans structurés, relecture, fil, retour, suggestion).

## Mesures (profil rempli de l'audit v5, jetons ≈ car./3)
- Tour, rédaction : P0 6 397 → P1 6 487 (+1,4 % ; +9,0 % contre la référence figée) ; tour total ≤ +5,1 % ; fil avec 3 bilans structurés : 1 081 car.
- Bilan : écriture initial 13 047 / point 13 622 (≤ +8,8 %) + relecture ≈ 2 630 → +26 % / +30 % au total ; sortie max 3 200 → 4 800 + 1 500 ; réécriture au plus : ×2.

## Reste / hors lot
- Non fait : 2 lignes du fil dans l'entrée de l'analyse, objectifs en cours pour la relecture du tour, fil dans le partage « reviews » au psychologue (`care-backend.mjs`), correction d'un état d'objectif par la personne.
- `v4-R-cron` F8 instable (échoue aussi sur `baa46aa`, 1 fois sur 6).
