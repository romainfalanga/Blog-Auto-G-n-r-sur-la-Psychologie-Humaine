# Audit de robustesse d’Ambre — 18 novembre 2026

Audit en lecture seule (aucune modification de code, aucun commit, aucun déploiement, aucun appel réseau vers la production). Les démonstrations rejouent le vrai moteur (`journal/src/*.mjs`) sur SQLite en mémoire avec toutes les migrations du dépôt et un fournisseur simulé, comme `journal/tests/*.test.mjs`. Scripts jetables hors dépôt (scratchpad de session, dossier `robustesse/`).

Base auditée : `dd23271` (branche courante), après les correctifs et intégrations du 17 novembre. L’ancien audit `audit-ambre-2026-11-17.md` n’est pas refait : seules les failles **encore présentes** ou **nouvelles** sont décrites.

**Suite de tests** : `npm --prefix journal test`, lancée une fois au début : **594 tests, 594 réussis, 0 échec** (14,7 s).

> Document enrichi au fil de l’audit. Les sections finales (liste priorisée, limites de charge) sont rédigées en dernier.

Méthode des mesures : harnais `harness.mjs` (copie conforme du montage de `journal/tests/ambre-engine.test.mjs`), profil « rempli » des tests (6 proches, 20 moments, cartographie, présentation), instrumentation du nombre d’instructions D1 par requête et `EXPLAIN QUERY PLAN` sur la vraie base migrée. Les durées sont celles de SQLite local (ordre de grandeur relatif, pas une mesure D1). Les « lignes lues » sont déduites du plan : `SEARCH … (user_id=?) | USE TEMP B-TREE` lit toutes les lignes de la personne.

## 1. Conversations très longues

### 1.1 Mesures (démo `d1-long.mjs`)

Un tour réel est d’abord exécuté pour obtenir une trace réaliste (24 098 octets pour un tour simple analyse/rédaction/relecture sur profil rempli). Cette trace est ensuite recopiée sur N échanges anciens, puis un nouveau tour réel est exécuté.

| N échanges | Instructions D1 par tour complet | Entrée analyse / rédaction / relecture (caractères envoyés) | Sondage `GET /api/chat` : instructions / octets | `GET /api/chat/insight` : octets | Export chat (octets) | Traces en base | Lignes `chat_engine_jobs` |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 168 | 6 012 / 15 509 / 6 742 | 10 / 3 260 | 165 929 | 26 496 | 0 | 1 |
| 300 | 168 | 7 793 / 19 700 / 8 399 | 10 / 94 651 | 212 936 | 8,2 Mo | 7,3 Mo | 301 |
| 1 000 | 168 | 7 793 / 19 700 / 8 399 | 10 / 94 651 | 212 955 | 27,3 Mo | 24,1 Mo | 1 001 |
| 3 000 | 168 | 7 795 / 19 707 / 8 402 | 10 / 94 828 | 212 961 | 30,4 Mo (tronqué) | 72,3 Mo | 3 001 |

Ce qui tient : **le coût d’entrée par tour est borné** (il ne croît pas avec l’historique : six échanges, mémoire bornée, index présélectionné) ; le nombre d’instructions D1 par tour est constant (168, sous la limite de 1 000 requêtes par invocation du forfait payant, même si un tour se répartit en pratique sur plusieurs invocations). Ce qui croît sans borne : traces, lignes de tâches terminées, export, lignes lues à chaque sondage.

### F1 — Traces privées sans rétention : la base D1 sature (critique, probabilité haute)

- **Démonstration** : 24 Ko de trace pour un tour simple ; l’ancien audit mesurait 41 à 69 Ko pour les profils type/maximal et la limite applicative est de 128 Ko par tour (`ambre-insight.mjs:4`). 3 000 échanges d’une seule personne = 72 Mo de traces, contre 4 Mo de texte de conversation : **les traces pèsent 18 fois la conversation**.
- Aucune purge : la seule suppression de `chat_turn_traces` est l’effacement volontaire (`chat-backend.mjs:746`) ou les références du plan (`change-plan.mjs:108`). Aucune durée de conservation, aucune compaction des `stages[].input` après publication.
- **Projection** (D1 : 10 Go par base au forfait payant) : 1 000 personnes × 20 tours/semaine × 40 Ko ≈ 0,8 Go/semaine → **saturation en ~12 semaines** ; 10 000 personnes × 20 tours ≈ 8 Go/semaine → **saturation en ~9 jours**. À saturation, toutes les écritures D1 échouent : inscriptions, tours, mémoire, suivi.
- Fichiers : `ambre-engine.mjs:186-198` (`startTrace` réécrit toute la trace à chaque étape), `ambre-insight.mjs:4,23`.
- **Correction** : conserver les entrées exactes (`details.stages[].input`, `system`) 30 jours au plus puis les remplacer par leur empreinte et leurs tailles ; ne stocker la consigne système qu’une fois par version (table de consignes adressées par hash) ; purge planifiée bornée dans le cron ; alerte sur `PRAGMA page_count`.
- **Test** : après 31 jours simulés, la purge vide `stages[].input` mais garde verdicts, modèles, jetons ; taille totale des traces de 3 000 tours < 5 Mo.

### F2 — Lignes `chat_engine_jobs` jamais supprimées et balayage complet chaque minute (haute, probabilité haute)

- **Démonstration** : `EXPLAIN QUERY PLAN` de la requête du cron (`ambre-engine.mjs:406`) : `SCAN chat_engine_jobs | USE TEMP B-TREE FOR ORDER BY`. Le prédicat `stage NOT IN ('done','failed')` ne peut pas utiliser l’index `chat_engine_due(stage,…)` (`0019_ambre_engine.sql:19`). Une ligne par tour publié subsiste à l’état `done` (3 001 lignes après 3 000 échanges).
- Conséquence : à 10 000 personnes × 300 tours, **3 millions de lignes lues chaque minute** par ce seul `SELECT` (≈ 130 milliards par mois, au-delà des 25 milliards inclus), et une requête de plusieurs centaines de millisecondes qui monopolise la base D1 (exécution séquentielle) toutes les minutes.
- **Correction** : index partiel `CREATE INDEX chat_engine_live ON chat_engine_jobs(lease_until,updated_at) WHERE stage NOT IN ('done','failed')` et requête identique ; supprimer (ou archiver dans la trace) la ligne de tâche à `done`/`failed` après 24 h.
- **Test** : plan de la requête de reprise = `SEARCH … USING INDEX chat_engine_live` ; aucune ligne `done` de plus de 24 h après un passage du cron.

### F3 — Chaque sondage relit tout l’historique de la personne (moyenne, probabilité haute)

- **Démonstration** : `SELECT … FROM chat_turns WHERE user_id=? AND purged=0 ORDER BY rowid DESC LIMIT 60` (`chat-backend.mjs:195`) → `SEARCH … USING INDEX sqlite_autoindex_chat_turns_2 (user_id=?) | USE TEMP B-TREE FOR ORDER BY` : les N lignes (avec leurs textes) sont lues puis triées pour en garder 60. Même plan pour la dernière réponse (`ambre-engine.mjs:441,448`). Le client sonde **toutes les 1,5 s** pendant une réponse (`chat-interface.mjs:212`) : un tour de 30 à 60 s = 20 à 40 sondages = 60 000 à 120 000 lignes lues à 3 000 échanges. Chaque sondage renvoie aussi les 60 derniers échanges complets (95 Ko ici, jusqu’à ~1 Mo avec des messages de 6 000 caractères).
- Les comptages par heure (`created_at>datetime('now','-1 hour')`, `chat-backend.mjs:504`) et les bilans du jour (`:553`) parcourent aussi toutes les lignes de la personne (index `chat_turns_generation` sans `created_at` en tête).
- **Correction** : index `(user_id, purged, rowid)` ou `ORDER BY created_at` sur un index `(user_id, created_at)` ; sondage léger `GET /api/chat/requests/:id` (statut seul) pendant l’attente, la liste complète seulement à la fin ; pagination du fil.
- **Test** : plan sans `TEMP B-TREE` ; taille de réponse du sondage d’attente < 2 Ko.

### F4 — Export incomplet et non borné en mémoire (moyenne, probabilité moyenne)

- **Démonstration** : à 3 000 échanges l’export du chat pèse 30 Mo alors que les traces seules font 72 Mo : `exportData` coupe **en silence** à 1 000 traces et 1 000 bilans (`chat-backend.mjs:519`), sans indicateur de couverture (contrairement à l’export du suivi, qui le documente). À l’inverse, tous les messages sont chargés d’un bloc puis sérialisés : un compte très ancien approche la limite mémoire de 128 Mo d’un Worker (traces 128 Ko × 1 000 = 128 Mo à elles seules) et la taille maximale de résultat D1.
- **Correction** : export paginé en flux (comme `care-backend.mjs` `exportPages`) avec curseur, `chat_export_coverage`, ou traces exportées sans les entrées exactes au-delà de la rétention.
- **Test** : 1 500 traces → export complet ou couverture explicite ; mémoire maximale bornée.

### 1.2 Mémoire après de nombreux lots (démo `d2-memory.mjs`)

Vrai `updateMemory` sur 1 000 échanges (125 lots de 8), avec un fournisseur simulé raisonnable : une citation exacte par message, les appuis rangés en « Sécurité et appuis ». Messages spéciaux : n° 3 tentative de suicide en 2019, n° 5 allergie et traitement, n° 7 objectif principal, n° 9 « Je vis avec Paul », n° 400 « séparés ».

| Fait ancien important | Encore en mémoire après 1 000 échanges ? |
|---|---|
| Tentative de suicide de 2019 (rubrique sécurité) | **Non** — seul le booléen `safety_seen` subsiste |
| Allergie à la pénicilline, traitement | **Non** |
| « Je vis avec Paul » / « séparés » | Non / non (12 faits au plus, FIFO) |
| Objectif principal « reprendre contact avec Léo » | Oui (rubrique peu alimentée) |

### F5 — Mémoire FIFO : les faits de sécurité et médicaux anciens disparaissent (haute, probabilité haute)

- **Démonstration** : la rubrique sécurité contient finalement 12 fois « Ma voisine Anne est un appui précieux, semaine N » ; la tentative de 2019 a été évincée par `slice(-12)` (`ambre-engine.mjs:395-396`). La déduplication est exacte (`unique`), donc des quasi-doublons saturent une rubrique. Aucune priorité, aucun épinglage, aucune résolution de contradiction : deux faits contradictoires coexistent tant qu’ils tiennent dans les 12 derniers, puis disparaissent ensemble.
- Ce que le rédacteur reçoit (`memoryPrompt`, `ambre-engine.mjs:25-38`) : 380 caractères par rubrique hors sécurité (850) et sujets fermés (1 300), soit 1 à 6 citations récentes. Après quelques centaines d’échanges, la continuité repose sur une poignée de citations récentes et le plan.
- **Correction** : séparer « antécédents de sécurité » (jamais évincés, épinglés, éditables par la personne) et « appuis » ; éviction par importance et non par ancienneté ; dédoublonnage approximatif ; marquage `superseded_by` quand une citation plus récente contredit (au minimum, garder les deux avec dates) ; dates des citations.
- **Test** : 1 000 échanges, la citation de tentative reste dans `safety_history` ; 12 appuis quasi identiques n’évincent rien.

### F6 — Une seule citation inexacte bloque la mémoire indéfiniment (haute, probabilité moyenne)

- **Démonstration** : message « Je n’arrive plus à dormir depuis l’accident. » ; le modèle renvoie la citation avec une apostrophe droite. La validation rejette **tout le lot** si un seul élément est inexact (`ambre-engine.mjs:393`, `items.some(…)` → `throw`), `through_turn` n’avance pas. Après 10 essais : aucune mémoire, `attempts=10`, attente 3 600 s, **10 appels facturés**, puis 24 appels par jour sans fin, sur le même lot.
- **Correction** : filtrer les éléments invalides au lieu de rejeter le lot (le second passage, ligne 395, le fait déjà) ; normaliser apostrophes/espaces avant comparaison en conservant le texte original ; après 3 échecs, avancer en marquant le lot « non résumé » dans la couverture.
- **Test** : lot avec 1 citation sur 8 normalisée → 7 citations retenues, `through_turn` avancé.

### F7 — Citation de sécurité tronquée au début du message (moyenne, probabilité moyenne)

- **Démonstration** : message de 360 caractères dont la fin est « je veux mourir » : la citation ajoutée par règle est `clip(turn.content,240)` (`ambre-engine.mjs:396`) et **ne contient pas le signal**. Même défaut pour les sujets fermés (`ambre-engine.mjs:73`, `clip(message,240)`).
- **Correction** : extraire la phrase qui a déclenché la règle (position du motif ± 120 caractères) ; idem pour `closed_topics`.
- **Test** : la citation stockée contient le motif détecté.
