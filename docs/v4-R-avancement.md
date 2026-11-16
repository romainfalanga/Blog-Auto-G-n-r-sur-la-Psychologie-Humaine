# Lot R (robustesse et tenue en charge) — avancement

Base : plan v4.1 (`docs/plan-ambre-v4-2026-11-18.md`), audit `docs/audit-robustesse-ambre-2026-11-18.md`.
Suite de départ : `npm --prefix journal test` 594/594.

## Fait
- F10 (+ base F12/F21) : module commun `journal/src/provider-access.mjs` ; garde consultée avant réservation et trace dans `generateText`, les bilans (`runReview`) et le portrait (`proxyJobLLM`) ; refus 401/402/403/429/503 remboursés (`provider_budgets`, `chat_engine_jobs.reserved_tokens`, plafond quotidien), étape de trace `refused` sans réservation, non comptée dans `attempt` ni `attempts` ; `Retry-After` respecté ; table `provider_state` (portée clé ou modèle) ; bloc d'erreur de `work()` : refus = report jusqu'à `retry_at` (au plus 600 s après la mise en file), sinon échec typé non décompté ; message 429 dédié de nouveau atteignable. Tests `journal/tests/v4-R-provider.test.mjs` (4, échouaient avant). Cinq tests existants de `ambre-engine.test.mjs` simulaient l'indisponibilité par 503/429 : ils utilisent désormais 502 (erreur consommée, toujours comptée), ce qu'ils protègent est inchangé.
- F1 : compaction SQL en place (`COMPACT_TRACE_SQL`, `ambre-insight.mjs`) des traces de tours terminés après 30 jours, et au-delà de 200 traces détaillées par personne (à chaque trace enregistrée) ; passage borné à 100 traces par minute dans le cron ; libellés fixes des critères restaurés à la lecture ; Paramètres signalent « Résumé (détail conservé 30 jours) » et la politique d'usage expose `trace_detail_days`/`trace_detail_max_turns`. Mesure (3 000 tours, J+31) : 72,2 Mo → 7,89 Mo de traces pour 4,00 Mo de conversation (≤ 2×).
- F2 : index partiel `chat_engine_live`, requête du cron sans `SCAN`, suppression des tâches `done`/`failed` de plus de 7 jours (500 par passage). Tests `v4-R-retention.test.mjs` (3).
- Migration `journal/migrations/0022_robustness.sql` (additive).
- F11 : `clientAddress()` (worker.mjs) lit `x-nf-client-connection-ip`, sinon `CF-Connecting-IP`, IPv6 par /64 ; limiteur par adresse e-mail conservé ; `workers_dev` inchangé, pas de 403 ; purge bornée (500/minute) de `rate_limits` et `sessions` expirées dans le cron.
- F12 : `DAILY_PROVIDER_TOKEN_CAP` (défaut 0 = désactivé) commun à tous les usages (conversation 100 %, tâches de fond 80 %) ; au-delà, état `daily_cap`, tour échoué non décompté avec message clair (+ ressources si signal de danger) ; `ACCOUNT_CREATIONS_PER_IP_DAY` (défaut 20).
- F13 (avancé) : chaînes du cron isolées (plus de `throw portraits.reason`), compteurs `service_metrics` sans contenu, route `POST /api/internal/metrics` (OIDC comme les autres routes internes).
- F21 : `OPENROUTER_ANALYSIS_FALLBACK_MODEL` pour analyse, contrôle de risque, relecture et mémoire ; bascule dès qu'un 5xx ou un refus met le modèle principal en refroidissement.
- Tests : `v4-R-provider.test.mjs` (6), `v4-R-worker.test.mjs` (5). Suite : 614/614.

## En cours
- F3, F8/F9.

## Reste
- F22, F4, F16, F23, F14 partiel.
