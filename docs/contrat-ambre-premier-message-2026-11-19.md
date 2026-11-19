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
  - `publish` : bilan approuvé, publication en cours ou reprise après un `safety_changed`. Le verdict validé est enregistré : la reprise publie directement, sans nouvelle relecture.

  L'interface traite toute valeur autre que `write`, `check`, `rewrite`, `publish` et `done` comme une phase de lecture, ce qui convient à `capture` et à `reading`.
- `blocks_total` est stocké à la capture (`ambre_reviews.sources.blocks_total`). Il vaut `null` avant la capture et `0` pour un bilan ancien sans instantané. `blocks_done` est le nombre de blocs analysés et sauvegardés.
- `started_at` correspond à `created_at` du bilan en cours, en secondes Unix.
- `estimate` est présent quand un bilan tourne ou peut être lancé. Sinon, il vaut `null`.
  - `basis: 'observed'` : médiane et percentile 90 au rang le plus proche de `completed_at − created_at`, calculés sur les 50 derniers bilans `ready` avec instantané, tous comptes confondus. Il en faut au moins 3. Le résultat est mis en cache 5 minutes par base.
  - Sinon, des valeurs initiales sont calculées (`REVIEW_ESTIMATE` dans `chat-backend.mjs`) :
    - appel immédiat i = 25 s, cron toutes les c = 60 s, bail L = 180 s ;
    - durée d'un appel : bloc b = 12 s, rédaction w = 30 s, relecture k = 10 s (recalibrées le 19/11, voir plus bas) ;
    - `travail = blocs × b + w + k` et `reste = max(0, travail − i)` ;
    - `p50 = i + ⌈reste / L⌉ × c/2 + reste` ;
    - `p90 = i + ⌈reste90 / L⌉ × c + reste90`, avec `reste90 = reste + w + k` (une réécriture) ;
    - les deux valeurs sont arrondies aux 5 s supérieures.

    Avant la capture, le calcul prend 8 blocs (`basis: 'initial_default'`). Une fois le nombre de blocs connu, `basis` vaut `'initial'`.

    **Calibrage sur la recette réelle du 19/11** (compte synthétique, 6 blocs) : bilan `ready` 122 s après sa création ; lecture des 6 blocs en parallèle ≈ 70 s (≈ 12 s par bloc) ; rédaction, relecture et réécriture ≈ 50 s. L'ancienne estimation (215–400 s pour 6 blocs) était trop haute. Valeurs obtenues :

    | blocs | p50 | p90 | affichage |
    |---|---|---|---|
    | 6 | 145 s | 215 s | entre 2 et 4 minutes |
    | 8 (défaut avant capture) | 170 s | 240 s | entre 3 et 4 minutes |
    | 12 | 215 s | 345 s | entre 4 et 6 minutes |

    La mesure (122 s) est sous le p50, ce qui laisse la marge du cron (attente moyenne c/2) ; le p90 compte une réécriture et une période complète du cron. Une seule mesure ne fait pas une distribution : dès 3 bilans `ready`, la base `observed` remplace ces valeurs.
- `last_failure` contient le dernier bilan s'il est `failed`. Il repasse à `null` dans trois cas :
  - un nouveau bilan démarre, car le plus récent est alors `running` ;
  - un bilan aboutit ;
  - la cause est levée : l'accord est redonné pour `consent_withdrawn`, l'accès est rétabli pour `access_required`, les prérequis sont complets pour `prerequisites_required`.

  Il survit au rechargement de la page.
  - `kind` (`initial` ou `point`) indique la route à appeler : `POST /api/chat/start` ou `POST /api/chat/review`.
  - `retryable` vaut `false` pour les causes `prerequisites_required`, `consent_withdrawn`, `access_required`, `snapshot_too_large`, `snapshot_analysis_limit` et `nothing_to_analyze`. Il vaut aussi `false` sans accès ou fournisseur disponible, et quand le crédit mensuel est épuisé. Dans ce dernier cas, `retry_at` donne la date de renouvellement.
  - Quand une relance est possible, `retry_at` vaut `null` (immédiatement) ou la plus tardive de ces échéances : pause du fournisseur, plafond quotidien du service (+1 h), limite de 8 créations par 24 h (`daily.resets_at`), limite de 4 bilans publiés par 24 h (plus ancien de ces bilans + 24 h).
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

   Si un élément a disparu, le bilan se termine par `sources_changed` (code final). Quand une garde échoue, le serveur cherche la vraie cause dans cet ordre : suppression, accord (`consent_withdrawn`), accès (`access_required`), prérequis (`prerequisites_required`), signal de sécurité. `context_changed` n'est renvoyé qu'en dernier recours. L'instantané est alors effacé par le trigger 0031, et le crédit mensuel est rendu.
3. **Sécurité tardive.** À la publication, le serveur recalcule le signal de sécurité à partir de l'état vivant : alerte de Mes symptômes, reconnaissance « idées suicidaires » dans Mon enfance, mémoire. Un signal absent à la capture impose deux choses :
   - les ressources de confiance en tête du message : 3114 et 15/112, ou 17/112 en cas de violence ;
   - les mêmes ressources en tête du document, si elles en sont absentes.

   Cette correction est inscrite dans `sources.check.safety_added`. La publication est aussi gardée par l'empreinte de ce signal : si elle change entre la lecture et l'écriture, le code `safety_changed` reprend directement à la publication au passage suivant du cron. Cette reprise ne consomme pas de tentative, dans la limite des 10 minutes d'un bilan en cours.
4. **Capture tolérante.** En cas d'écriture concurrente pendant la capture, la lecture est refaite jusqu'à 2 fois. Au-delà, le bilan se termine par `context_changed`.
5. **Portrait en cours.** Si une tâche de portrait réellement exécutable existe pour la version courante (`pending` ou `running`, moins de 3 essais et de 20 appels, ni `batch_required` ni `generation_budget`, comme dans `personal-backend.mjs`), la capture attend, sans compter de tentative (`error_code = portrait_pending`, nouvel essai par le cron). Au-delà de 180 s après la demande, la capture est prise sans portrait : `payload.omitted = [{section:'portrait',reason:'refresh_pending'}]`, signalé au rédacteur et dans `sources.omitted`. L'état du portrait est recalculé à chaque essai de capture. Le cron du portrait n'est pas suspendu.
6. **Citations en trois classes** (`classifyQuotes`) :
   - les mots de la personne sont vérifiés dans son seul corpus (`snapshotPersonCorpus`, y compris Mes précisions `memories.text`), jamais dans un texte produit par l'IA (portrait, mémoire, bilans précédents, réponses, résumés de blocs) ;
   - les libellés de l'interface (`INTERFACE_LABELS` : questions, réponses, thèmes, sujets, période) sont admis comme libellés, mais refusés s'ils sont précédés de « vous écrivez / dites / vos mots » ;
   - toute autre citation est une invention.

   Le rédacteur suit la même règle (`QUOTE_RULE`). L'analyse des blocs connaît la correspondance entre valeurs et libellés (`past = Dans le passé`). La relecture reçoit `interface_labels_quoted`.
7. **Une seule règle d'hypothèse** (`HYPOTHESIS_RULE`) : « hypothèse : X, appuyée par A et B ; il manque C », jamais « vous avez / souffrez de / êtes [trouble] ». Elle est partagée par le rédacteur (commun et initial), le critère `no_diagnosis`, `hypotheses_open` et `MAP_HYPOTHESIS_RULES`. Aucun critère n'est désactivé : un « no » du relecteur fait toujours échouer le bilan.
8. **`review_snapshots`.** Un bilan échoué ne garde aucun instantané : suppression dans `runReview`, dans le balayage du cron et par la migration additive `0033_review_failed_snapshots.sql` (trigger et nettoyage unique). Les bilans prêts ne sont jamais touchés.

## Sortie (lot 2)

- `REVIEW_OUTPUT_SCHEMA` vaut `{content, chat_message, digest}`. Il y a **un seul appel de rédaction**. `chat_message` fait au plus 2 500 caractères et présente ce qui ressort, puis 2 à 3 questions.
- `REVIEW_WRITE_TOKENS` passe de 4 800 à 5 600. La réservation suit, puisqu'elle est calculée à partir de ce plafond.
- `chat_message` passe par les mêmes contrôles déterministes que le document (diagnostic, citations, ressources, numéros, enregistrement annoncé), avec en plus le nombre de questions (`chat_questions`, non bloquant). Il est aussi soumis à la relecture du modèle.
- Si seule la longueur pose problème (moins de 200 ou plus de 2 500 caractères), un message de repli déterministe est construit à partir du digest vérifié, ou des premiers paragraphes du document. Il ne reprend aucune phrase interrogative du document : ses 2 ou 3 questions viennent des questions ouvertes du digest, une seule question chacune, ou des questions fixes. En cas de signal, ce repli commence par les ressources. Il est ensuite contrôlé comme le message du modèle (`sources.check.chat_fallback`).
- Le contenu du bilan n'est pas réduit : l'instantané reste complet, l'évaluation clinique déterministe, l'analyse de tous les blocs et la relecture sont conservées.

## Codes d'échec (`error_code`) pour la table de messages de l'interface

- **Finaux** : `sources_changed`, `context_changed`, `review_not_approved`, `consent_withdrawn`, `prerequisites_required`, `access_required`, `snapshot_too_large`, `snapshot_analysis_limit`, `nothing_to_analyze`, `generation_budget`, `quota_reservation_missing`.
- **Nouvelle tentative par le cron jusqu'à 3** : `provider_timeout`, `provider_http_*`, `invalid_response`, `invalid_check`, `provider_error`. `safety_changed` est repris sans tentative comptée pendant 10 minutes.
- Chaque code a son propre message dans `AMBRE_FAILURES`. `snapshot_too_large` et `snapshot_analysis_limit` sont présentés comme des limites fixes, sans relance.
- **Après épuisement** : `attempts_exhausted`.
- **Liés au fournisseur** : `provider_cooldown`, `daily_cap`.

## Corrections après la recette réelle du 19/11

1. **Liens du message de synthèse.** Cause : l'interface n'affichait « Lire le bilan complet » et « Voir Mes diagnostics » que si `content` différait de `chat_message`. Or, dans les nouvelles enveloppes, les deux sont égaux : seul le lien ancien « Retrouver ce bilan dans Mes bilans » apparaissait. L'interface lit désormais `message.links` (libellé non vide, chemin interne commençant par `/`). À défaut, elle affiche les deux liens par défaut si `chat_message` est présent, et le lien ancien sinon.
2. **Paragraphes.** `chat_message` est demandé en 2 à 4 paragraphes courts (moins de 350 caractères chacun), suivis des 2 ou 3 questions, une par ligne. Un exemple de forme est fourni au rédacteur. `structureChatMessage` remet en forme, sans changer un mot, un message écrit d'un seul bloc : les questions finales passent sur leurs propres lignes et les paragraphes trop longs sont coupés en fin de phrase. Le repli déterministe suit la même forme. L'interface affiche les paragraphes (`.chat-paragraph`) et les questions dans un bloc distinct (`ul.chat-questions`), y compris pour les anciens messages d'un seul bloc.
3. **Fonctionnement interne.** La consigne du rédacteur (« Fonctionnement interne ») et celle de la charge utile des blocs interdisent de parler à la personne des blocs, de l'instantané, des résumés, de la vérification ou de la rareté des citations, du modèle ou de la relecture. Deux contrôles s'y ajoutent :
   - un contrôle déterministe, `internal_process`, sur le document et sur le message ;
   - un critère de relecture, `no_internal`.

   Une première détection impose l'unique réécriture. Si la mention persiste après la réécriture, les phrases concernées sont retirées à la publication (`sources.check.internal_removed`) au lieu de faire échouer tout le bilan.
4. **Première réponse de conversation.** Le « critère 13 » de la trace est numéroté à partir de 1 dans `REVIEW_CRITERIA` : c'est « Posture adaptée au besoin », et non « Mots simples et longueur proportionnée » (n° 14). C'est aussi le code du contrôle déterministe bloquant `missingReferral`, qui exige la ressource d'un signal actif. Hypothèse retenue : la ligne 3919 du repli prouve qu'un signal `ipv` était actif, alors que le message n'en contient aucun indice (le détecteur lexical renvoie `none`). Ce signal venait donc du modèle d'analyse, nourri par le contexte riche d'après bilan. Faute de 3919, les deux brouillons échouaient au contrôle 13. Ce contrôle ne pouvait pas être corrigé par une réécriture pertinente. Cette hypothèse a été rejouée localement avec un fournisseur simulé, sans appel payant. Corrections :
   - un signal `ipv` de contrôle ou physique proposé par le modèle n'est gardé que si le message ou l'un des deux messages précédents de la personne parle d'un partenaire ou de violence ;
   - un signal `ipv` grave du modèle est toujours gardé ;
   - le signal écarté est tracé dans `analysis.unanchored_signals`.

   Le repli ne contient donc plus la ligne 3919 sans signal ancré. Il répond au message : il reprend les mots exacts de la personne s'ils tiennent en 220 caractères, puis pose une question ouverte. En posture de calme (stabilisation, réparation, détresse, `no_questions`, `no_exploration`), il ne pose pas de question. Le texte générique reste réservé aux messages longs. Les consignes de longueur sont aussi explicites et cohérentes avec le critère 14 :
   - rédacteur : 60 à 180 mots, jamais plus de 220 ; 40 à 100 mots pour un message court ou en posture de stabilisation ; une seule piste ; un exemple de réponse ;
   - relecteur : le critère 14 est en échec seulement au-delà de 220 mots (100 pour un message court), en cas de jargon ou au-delà de deux pistes.

   Aucun critère n'est désactivé. Cette hypothèse n'a pas été vérifiée sur la trace réelle complète (`analysis.signals`) : la vérifier à la prochaine recette.

## Points ouverts

- Les durées d'appel de l'estimation initiale reposent sur une seule recette réelle (6 blocs, 122 s). La base `observed` les remplacera dès 3 bilans prêts.
- La requête d'estimation observée parcourt les bilans `ready` (au plus une fois toutes les 5 minutes par isolat) et n'utilise pas d'index.
- Une ancienne étape `check` en cours au déploiement, sans `chat_message`, publie le repli déterministe sans relecture du modèle de ce message.
- La relecture indépendante a été faite par le coordinateur. Ses corrections ont été intégrées le 19/11 (relecture du document complet hors de la fenêtre, Mes précisions, étape `publish`, vraie cause d'échec, portrait exécutable, messages d'échec, ressources en tête, rafraîchissement au retour sur l'onglet, case d'accord sans `journey`).
