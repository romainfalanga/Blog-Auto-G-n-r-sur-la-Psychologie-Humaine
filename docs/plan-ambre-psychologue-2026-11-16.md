# Plan « Ambre, assistante d’un suivi psychologique » — 16 novembre 2026

Sources de ce plan : `docs/recherche-pratique-clinique-2026-11-16.md` (pratiques professionnelles, 40 règles, grilles 7.b, 7.c, 7.d) et `docs/audit-ambre-2026-11-16.md` (fonctionnement actuel, mesures, points d’insertion).

## Principe

Ambre est une **assistante psychologue IA**. Elle ne remplace pas un professionnel : elle prépare, prolonge et documente le travail fait avec lui. La meilleure manière de l’utiliser est d’avoir un professionnel de santé à côté ; elle reste utilisable sans, en le disant clairement. Trois chantiers, menés en parallèle par trois agents, avec des contrats fixés ici.

## A. Moteur d’Ambre (chaque message)

Chaîne d’un tour, reprenable et bornée :

1. **Filet de sécurité déterministe** (avant tout appel) : détection élargie des idées suicidaires et du danger ; si le fournisseur échoue, la personne reçoit quand même un message de sécurité (3114, 15/112). Le 3114 est toujours connu d’Ambre.
2. **Analyse du tour** (petit modèle, sortie JSON courte, repli déterministe) : risque, état de la personne (table 7.d), ce qu’elle demande (être écoutée, comprendre, avis, agir), rupture d’alliance, sujets fermés, et **identifiants à ouvrir** dans l’index des sources.
3. **Matrice de contexte** : posture imposée et interdits du tour (« aucune question », « aucun conseil », question sensible autorisée ou non selon la grille 7.b) ; **contexte par niveaux** : niveau 0 = index complet d’une ligne par élément (proches, moments, causes, hypothèses, notes) + mémoire de conversation ; niveau 1 = détail des seuls éléments retenus. Consignes fixes en tête, mises en cache chez le fournisseur.
4. **Rédaction** (modèle principal).
5. **Relecture** : contrôles déterministes sur chaque réponse (une question au plus, pas de diagnostic, numéros vérifiés, risque traité, notes annoncées = notes écrites) puis relecture par modèle selon la grille 7.c ; critère bloquant en échec → une réécriture, sinon réponse de sécurité. Le filtre qui coupait des phrases est supprimé.
6. **Mémoire de conversation** : note évolutive structurée (demande et objectifs, faits rapportés, hypothèses marquées comme telles, ce qui a été essayé, alliance, sécurité, sujets fermés, à apporter en séance), mise à jour tous les quelques tours hors du tour ; l’historique envoyé = cette note + les derniers tours contigus. Coût stable quelle que soit la longueur de la conversation.
7. **Trace** de chaque tour (analyse, blocs de contexte et tailles, verdict de relecture, jetons, durée) pour la page Paramètres.

Ambre connaît le **cadre de suivi** (psychologue ou non, fréquence, dernière et prochaine séance) et s’y ajuste : préparer la séance, noter ce qui est à y apporter, ne pas se substituer ; sans psychologue, encourager un accompagnement sans insister.

## B. Suivi avec un psychologue

- Deux types de comptes : **personne** et **psychologue** (choisi à l’inscription ; nom, titre et numéro professionnel déclarés, affichés comme « déclarés, non vérifiés »).
- **Lier son psychologue** : la personne saisit l’adresse du professionnel et choisit ce qu’elle partage (conversation, cartographie, portrait, proches, ligne de vie, bilans). Elle reçoit un **code à transmettre** au professionnel ; il le saisit dans son espace pour activer le partage. Aucune indication n’est donnée sur l’existence d’un compte ; aucun courriel n’est envoyé ; révocable à tout moment ; chaque consultation est journalisée et visible par la personne.
- **Mon suivi** (personne) : psychologue ou non, fréquence (toutes les 2 semaines, mensuelle, tous les 2 mois, autre — modifiable), dates des séances passées et à venir, partages en cours, comptes rendus préparés.
- **Espace psychologue** : demandes en attente, patients, dossier partagé en lecture seule, séances, et **point avant séance** : tout ce qui s’est passé depuis la dernière séance (thèmes, faits rapportés avec citations, évolution, ce qui a été essayé, points de vigilance, questions ouvertes), préparé automatiquement avant la date de séance ou à la demande. La personne peut lire ce qui est préparé pour son psychologue.

## C. Paramètres et interface

- Menu : « Mon compte » devient **Paramètres** (`/parametres/`), avec un bouton vers « Mon compte » (`/mon-compte/`, inchangé) et un accès à « Mon suivi ».
- **L’envers du décor d’Ambre** : les étapes d’un message ; ce qu’Ambre sait de vous (sources, index, mémoire de conversation, notes, cadre de suivi) ; comment les dernières réponses ont été construites (trace) ; les règles qu’elle suit et leurs sources ; ses limites.
- La fenêtre d’accueil devient un **panneau dans la page** (le menu reste accessible) et présente Ambre comme assistante, à utiliser de préférence avec un professionnel.

## Contrats entre chantiers

### Tables (migration `0020_care.sql`, chantier B)

```sql
ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'member' CHECK(role IN ('member','psychologist'));
CREATE TABLE care_settings (user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 has_psychologist INTEGER CHECK(has_psychologist IN (0,1)), frequency_days INTEGER CHECK(frequency_days BETWEEN 1 AND 365),
 updated_at INTEGER NOT NULL);
CREATE TABLE care_sessions (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 session_at INTEGER NOT NULL, created_by TEXT NOT NULL, created_at INTEGER NOT NULL);
```
`has_psychologist` NULL = non renseigné. Les autres tables du chantier B (profil professionnel, liens, comptes rendus, journal d’accès) sont libres.

### `journal/src/care-context.mjs` (déjà écrit, ne pas modifier sa signature)

`loadCareContext(env,userId,now)` → `{has_psychologist:true|false|null, frequency_days:number|null, last_session_at:number|null, next_session_at:number|null}` ; ne lève jamais (tables absentes → tout à `null`).

### Chantier A → autres

- `chat.generateText(env,{userId,system,user,maxTokens,timeoutMs,purpose})` → `{text,input_tokens,output_tokens}` ou lève une erreur portant `code` ; réserve le budget de jetons de `userId`. Utilisé par B pour le point avant séance.
- Table `chat_memory(user_id PRIMARY KEY, summary TEXT (JSON), through_turn TEXT, turns_covered INTEGER, updated_at INTEGER)` (migration `0019_ambre_engine.sql`). B peut la lire pour le point avant séance.
- `GET /api/chat/insight` (personne connectée) →
```json
{"care":{"has_psychologist":null,"frequency_days":null,"last_session_at":null,"next_session_at":null},
 "knowledge":{"sources":[{"key":"cartography|portrait|people|life_events","count":0,"updated_at":null}],
   "index":[{"kind":"person|event|cause|hypothesis|note","label":"…"}],"index_total":0},
 "memory":{"updated_at":null,"turns_covered":0,"sections":[{"key":"…","title":"…","items":["…"]}]},
 "turns":[{"id":"…","created_at":"…","analysis":{"need":"…","states":["…"],"risk":"none|possible|explicit","posture":"…","constraints":["…"]},
   "context":{"blocks":[{"key":"…","title":"…","chars":0,"items":0}],"input_tokens":0,"output_tokens":0,"cached_tokens":0},
   "review":{"checked":true,"passed":true,"failed":["…"],"rewritten":false},"duration_ms":0}],
 "models":{"analysis":"…","writing":"…","review":"…"}}
```
Champs absents tolérés par l’interface. Aucun texte privé de la conversation dans `turns` (seulement des étiquettes).

### Chantier B → autres

- `POST /api/auth/register` accepte `role` (`member` par défaut, ou `psychologist` avec `professional:{name,title,registration_number}`) ; `/api/me` renvoie `user.role`.
- API du suivi sous `/api/care/…` (décrite par B dans `journal/README.md`).
- B fournit `journal/src/care-interface.mjs` exportant `CARE_INTERFACE_SCRIPT` (chaîne de script, même convention que `chat-interface.mjs`) qui définit `renderCareMember(view,current)` et `renderCarePsychologist(view,current)`.

### Chantier C → autres

- C possède `ui.mjs`, `chat-interface.mjs`, `workspace-style.mjs`, la navigation et `netlify.toml` ; il enregistre les pages `/parametres/`, `/mon-suivi/` (appelle `renderCareMember`) et `/espace-psychologue/` (appelle `renderCarePsychologist`), inclut `CARE_INTERFACE_SCRIPT` s’il existe, et adapte le menu au rôle.

## Tenue en charge et coût

Analyse et relecture sur un petit modèle (entrées de l’ordre de 1 à 2 milliers de jetons) ; consignes fixes en cache ; contexte détaillé limité aux éléments retenus ; mémoire mise à jour par lots ; tout traitement long sous bail, repris par la tâche planifiée ; aucune requête non bornée ; budgets par personne conservés.

## Vérifications

Tests unitaires et d’API par chantier, jeux d’essai sur les cas indirects de risque, essais réels de bout en bout avec comptes temporaires (personne et psychologue) supprimés ensuite, contrôle visuel ordinateur et mobile, vérification en ligne.
