# Contrat serveur : premier message d'Ambre et bilan lancé depuis la conversation (19/11)

Ce document met en œuvre les lots 1 et 2 de la [version 1.1 du plan](plan-ambre-premier-message-2026-11-19.md) et la partie serveur du contrat d'API du lot 3. Il n'y a eu aucun déploiement, aucun appel payant au fournisseur ni aucun commit par l'agent serveur. Toutes les vérifications passent par un fournisseur simulé (`journal/tests/ambre-premier-message.test.mjs`).

## `GET /api/chat`

### `readiness.pages`

Il y a une entrée par page du premier message, dans l'ordre suivant :

| `id` | `label` | `href` | `done`/`answered` | `total`/`required` |
|---|---|---|---|---|
| `account` | Mon compte | `/mon-compte/` | champs d'identité valides | 4 |
| `people` | Mes proches | `/mes-proches/` | min(nombre, 3) | 3 |
| `timeline` | Ma ligne de vie | `/ma-ligne-de-vie/` | min(nombre, 3) | 3 |
| `childhood` | Mon enfance | `/cartographie/` | réponses actives | N (questions actives) |
| `symptoms` | Mes symptômes | `/mes-symptomes/` | réponses | 516 |

- `saved` : pour Mon enfance et Mes symptômes, il faut `consent = 1` **et** `locked_at` renseigné. Pour les autres pages, la valeur est toujours `true`.
- `complete` : toutes les réponses sont données **et** la page est enregistrée. La règle est la même que celle de `readiness.complete` et de la garde SQL de création du bilan.
- `done = total` avec `saved = false` signifie que la page doit afficher « enregistrez ».
- `answered` et `required` répètent `done` et `total`, comme le demande l'interface.

### `journey.review`

```
{ running_id, stage, blocks_done, blocks_total, started_at,
  estimate: {p50_seconds, p90_seconds, basis[, sample]} | null,
  last_failure: {id, kind, error_code, retryable, retry_at, daily_limit, daily_remaining} | null,
  daily: {created, limit, resets_at},
  last_at, next_point_at[, suggestion] }
```

- `stage` vaut `null` quand aucun bilan ne tourne. Sinon, il prend l'une des valeurs suivantes :
  - `capture` : l'instantané n'est pas encore pris. Cette phase comprend l'attente d'un portrait, au plus 3 minutes.
  - `reading` : les blocs sont en cours d'analyse (`blocks_done` < `blocks_total`).
  - `write` : rédaction du document et du message de conversation.
  - `check` : relecture.
  - `rewrite` : réécriture unique.

  L'interface traite toute valeur autre que `write`, `check` et `rewrite` comme une phase de lecture, ce qui convient à `capture` et à `reading`.
- `blocks_total` est stocké à la capture (`ambre_reviews.sources.blocks_total`). Il vaut `null` avant la capture et `0` pour un bilan ancien sans instantané. `blocks_done` est le nombre de blocs analysés et sauvegardés.
- `started_at` correspond à `created_at` du bilan en cours, en secondes Unix.
- `estimate` est présent quand un bilan tourne ou peut être lancé. Sinon, il vaut `null`.
  - `basis: 'observed'` : médiane et percentile 90 au rang le plus proche de `completed_at − created_at`, calculés sur les 50 derniers bilans `ready` avec instantané, tous comptes confondus. Il en faut au moins 3. Le résultat est mis en cache 5 minutes par base.
  - Sinon, des valeurs initiales sont calculées (`REVIEW_ESTIMATE` dans `chat-backend.mjs`) :
    - appel immédiat i = 25 s, cron toutes les c = 60 s, bail L = 180 s ;
    - durée d'un appel : bloc b = 15 s, rédaction w = 75 s, relecture k = 20 s ;
    - `travail = blocs × b + w + k` et `reste = max(0, travail − i)` ;
    - `p50 = i + ⌈reste / L⌉ × c/2 + reste` ;
    - `p90 = i + ⌈reste90 / L⌉ × c + reste90`, avec `reste90 = reste + w + k` (une réécriture) ;
    - les deux valeurs sont arrondies aux 5 s supérieures.

    Avant la capture, le calcul prend 8 blocs (`basis: 'initial_default'`). Une fois le nombre de blocs connu, `basis` vaut `'initial'`. Par exemple, 8 blocs donnent 275 à 430 s. **Les durées par appel sont des valeurs de départ. Elles restent à remplacer par la mesure de la recette payante (lot 4).**
- `last_failure` contient le dernier bilan s'il est `failed`. Il repasse à `null` dans trois cas :
  - un nouveau bilan démarre, car le plus récent est alors `running` ;
  - un bilan aboutit ;
  - la cause est levée : l'accord est redonné pour `consent_withdrawn`, l'accès est rétabli pour `access_required`, les prérequis sont complets pour `prerequisites_required`.

  Il survit au rechargement de la page.
  - `kind` (`initial` ou `point`) indique la route à appeler : `POST /api/chat/start` ou `POST /api/chat/review`.
  - `retryable` vaut `false` pour les causes `prerequisites_required`, `consent_withdrawn`, `access_required`, `snapshot_too_large`, `snapshot_analysis_limit` et `nothing_to_analyze`. Il vaut aussi `false` sans accès ou fournisseur disponible, et quand le crédit mensuel est épuisé. Dans ce dernier cas, `retry_at` donne la date de renouvellement.
  - Quand une relance est possible, `retry_at` vaut `null` (immédiatement) ou la plus tardive de ces échéances : pause du fournisseur, plafond quotidien du service (+1 h), limite de 8 créations par 24 h (`daily.resets_at`), limite de 4 bilans publiés par 24 h (+24 h).
- `daily` est la limite de 8 créations de bilan par 24 h glissantes. Elle n'est pas réduite par la suppression d'un bilan.

### Message d'Ambre après un bilan (`messages[]`)

`{id, role:'assistant', review_id, review_kind, chat_message, content, links:[{label:'Lire le bilan complet',href:'/mes-bilans/'},{label:'Voir Mes diagnostics',href:'/mes-diagnostics/'}], created_at}`

Dans les nouvelles enveloppes, `content` vaut `chat_message`. Les anciennes enveloppes n'ont ni `chat_message` ni `links`, et le client affiche `content`. Aucun faux message n'est créé pour les anciens bilans. Le document complet reste dans `ambre_reviews.content`, consultable par `GET /api/reviews/:id` (Mes bilans). Le chemin `generate()` hérité relit ce document pour `last_review`.

## Lancement : `POST /api/chat/start {idempotency_key, accepted:true}`

L'accord n'est plus enregistré avant la création du bilan. Il est écrit dans la **même transaction** que l'insertion du bilan, et seulement si celle-ci a lieu. Une création refusée (prérequis, quota, limite, bilan en cours) n'enregistre donc aucun accord. Le premier message sert d'accord unique : la carte `chat-gate` et `PUT /api/chat/consent` ne sont plus nécessaires pour lancer, mais la route est conservée pour le retrait. Le point mensuel passe par `POST /api/chat/review {idempotency_key}`, inchangé.

## Bilan (lot 1)

1. **Instantané figé.** Après la capture, le bilan ne travaille plus que sur l'instantané. La comparaison d'empreinte des sources vivantes (`snapshotGuard`) a été retirée des appels, de la sauvegarde des étapes et de la publication. Une modification ou un ajout sera pris en compte au bilan suivant. Les gardes suivantes restent en place : plan de changement (`changeGuard`), bail, accord, accès, prérequis, réservation mensuelle, budgets hebdomadaire et quotidien.
2. **Garde de suppression.** Elle s'applique avant chaque appel, à chaque sauvegarde d'étape et à la publication (`snapshotPresenceGuard`). Les éléments saisis par la personne et présents dans l'instantané doivent encore exister :
   - proches, moments, notes, observations, messages non purgés, bilans prêts, séances, personnages, histoires, conclusions professionnelles, lectures, inspirations et retours ;
   - Mon enfance et Mes symptômes toujours consentis, ainsi que le suivi.

   Si un élément a disparu, le bilan se termine par `sources_changed` (code final). L'instantané est alors effacé par le trigger 0031, et le crédit mensuel est rendu.
3. **Sécurité tardive.** À la publication, le serveur recalcule le signal de sécurité à partir de l'état vivant : alerte de Mes symptômes, reconnaissance « idées suicidaires » dans Mon enfance, mémoire. Un signal absent à la capture impose deux choses :
   - les ressources de confiance en tête du message : 3114 et 15/112, ou 17/112 en cas de violence ;
   - les mêmes ressources à la fin du document, si elles en sont absentes.

   Cette correction est inscrite dans `sources.check.safety_added`. La publication est aussi gardée par l'empreinte de ce signal : si elle change entre la lecture et l'écriture, le code non final `safety_changed` déclenche une nouvelle tentative.
4. **Capture tolérante.** En cas d'écriture concurrente pendant la capture, la lecture est refaite jusqu'à 2 fois. Au-delà, le bilan se termine par `context_changed`.
5. **Portrait en cours.** Si une tâche de portrait `pending` ou `running` existe pour la version courante, la capture attend, sans compter de tentative (`error_code = portrait_pending`, nouvel essai par le cron). Au-delà de 180 s après la demande, la capture est prise sans portrait : `payload.omitted = [{section:'portrait',reason:'refresh_pending'}]`, signalé au rédacteur et dans `sources.omitted`. Le cron du portrait n'est pas suspendu.
6. **Citations en trois classes** (`classifyQuotes`) :
   - les mots de la personne sont vérifiés dans son seul corpus (`snapshotPersonCorpus`), jamais dans un texte produit par l'IA (portrait, mémoire, bilans précédents, réponses, résumés de blocs) ;
   - les libellés de l'interface (`INTERFACE_LABELS` : questions, réponses, thèmes, sujets, période) sont admis comme libellés, mais refusés s'ils sont précédés de « vous écrivez / dites / vos mots » ;
   - toute autre citation est une invention.

   Le rédacteur suit la même règle (`QUOTE_RULE`). L'analyse des blocs connaît la correspondance entre valeurs et libellés (`past = Dans le passé`). La relecture reçoit `interface_labels_quoted`.
7. **Une seule règle d'hypothèse** (`HYPOTHESIS_RULE`) : « hypothèse : X, appuyée par A et B ; il manque C », jamais « vous avez / souffrez de / êtes [trouble] ». Elle est partagée par le rédacteur (commun et initial), le critère `no_diagnosis`, `hypotheses_open` et `MAP_HYPOTHESIS_RULES`. Aucun critère n'est désactivé : un « no » du relecteur fait toujours échouer le bilan.
8. **`review_snapshots`.** Un bilan échoué ne garde aucun instantané : suppression dans `runReview`, dans le balayage du cron et par la migration additive `0033_review_failed_snapshots.sql` (trigger et nettoyage unique). Les bilans prêts ne sont jamais touchés.

## Sortie (lot 2)

- `REVIEW_OUTPUT_SCHEMA` vaut `{content, chat_message, digest}`. Il y a **un seul appel de rédaction**. `chat_message` fait au plus 2 500 caractères et présente ce qui ressort, puis 2 à 3 questions.
- `REVIEW_WRITE_TOKENS` passe de 4 800 à 5 600. La réservation suit, puisqu'elle est calculée à partir de ce plafond.
- `chat_message` passe par les mêmes contrôles déterministes que le document (diagnostic, citations, ressources, numéros, enregistrement annoncé), avec en plus le nombre de questions (`chat_questions`, non bloquant). Il est aussi soumis à la relecture du modèle.
- Si seule la longueur pose problème (moins de 200 ou plus de 2 500 caractères), un message de repli déterministe est construit à partir du digest vérifié, ou des premiers paragraphes du document. En cas de signal, ce repli commence par les ressources. Il est ensuite contrôlé comme le message du modèle (`sources.check.chat_fallback`).
- Le contenu du bilan n'est pas réduit : l'instantané reste complet, l'évaluation clinique déterministe, l'analyse de tous les blocs et la relecture sont conservées.

## Codes d'échec (`error_code`) pour la table de messages de l'interface

- **Finaux** : `sources_changed`, `context_changed`, `review_not_approved`, `consent_withdrawn`, `prerequisites_required`, `access_required`, `snapshot_too_large`, `snapshot_analysis_limit`, `nothing_to_analyze`, `generation_budget`, `quota_reservation_missing`.
- **Nouvelle tentative par le cron jusqu'à 3** : `provider_timeout`, `provider_http_*`, `invalid_response`, `invalid_check`, `safety_changed`, `provider_error`.
- **Après épuisement** : `attempts_exhausted`.
- **Liés au fournisseur** : `provider_cooldown`, `daily_cap`.

## Points ouverts

- Les durées d'appel de l'estimation initiale ne sont pas mesurées : la recette payante du lot 4 doit les fixer.
- La requête d'estimation observée parcourt les bilans `ready` (au plus une fois toutes les 5 minutes par isolat) et n'utilise pas d'index.
- `safety_changed`, s'il se répète, consomme une tentative et une relecture.
- Une ancienne étape `check` en cours au déploiement, sans `chat_message`, publie le repli déterministe sans relecture du modèle de ce message.
- La relecture de code indépendante reste à faire : l'agent serveur ne disposait pas d'outil de sous-agent.
