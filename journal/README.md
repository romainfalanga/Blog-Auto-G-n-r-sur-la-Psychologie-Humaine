# Espace personnel et Ambre

Pages de l’interface : `/mon-accompagnement/`, `/mon-portrait/`, `/mes-proches/`, `/ma-ligne-de-vie/`, `/mes-bilans/`, `/mon-compte/`. Les données sont privées et authentifiées. Les anciens liens conservent les paramètres liés aux articles publics, à la navigation personnelle ou au paiement lorsque la destination les utilise.

## Base et migration

Appliquer les migrations `0001` à `0020` dans l'ordre. Sur une base existante, créer d'abord un point de restauration D1 et suspendre les générations. La migration `0005_personal_context_characters.sql` conserve IDs, corps, retours et ordre des histoires ; elle les rattache au premier personnage et inscrit leurs crédits dans leur semaine d'origine. Les portraits V1 sont invalidés pour être reconstruits à partir des sources originales. `0006_recommendation_exposures.sql` stocke uniquement des IDs d'articles publics récemment proposés.

Ne pas rejouer aveuglément une migration appliquée. Le déploiement de `worker.mjs` exige tous les modules importés de `src/`, notamment `personal-backend.mjs`, `private-context.mjs`, `recommendations.mjs`, sa baseline et le catalogue. `GENERATION_PAUSED=true` suspend la prise en charge et la publication des analyses ; lecture et édition personnelle restent disponibles.

Les proches, événements, associations, révisions, manifestes et demandes sont décrits dans [le contrat partagé](../docs/contrats-implementation-v2.md). Les créations utilisent une clé idempotente ; les modifications et suppressions vérifient la révision. Supprimer un proche conserve les événements liés. Export et suppression du compte couvrent les nouvelles tables.

## Portrait et recommandations

Un like, une relation décrite, un événement ou une mémoire personnelle peuvent démarrer le portrait. Un nom et rôle seuls ne servent pas de preuve. Les dérivés réutilisables sont attachés à une source, sa révision, son hash et la version de l'analyse. Toute assertion publiée exige une citation exacte d'une source autorisée. Un like sans commentaire reste un intérêt de faible confiance. Les très grands contextes sont traités par tranches avec couverture explicite, jamais silencieusement tronqués.

En mode `FREE_ACCESS_ENABLED=true`, la plateforme est accessible sans paiement et ces anciens plafonds gratuits ne bloquent plus les révisions de portrait. Le quota de conversation et les budgets techniques restent actifs. Les sources modifiées invalident l'analyse en cours ; les recommandations provisoires restent disponibles. Les préférences de style ou d'évitement ne deviennent pas des faits autobiographiques.

Le classement V2 est déterministe, sans appel IA par requête, et ne remplit pas la liste avec des sujets sans rapport. Les expositions sont bornées à cent IDs sur sept jours et ne modifient pas le portrait. Les résultats synthétiques ne démontrent pas une amélioration générale chez les lecteurs ayant déjà des likes : consulter [l'évaluation](../docs/implementation-algorithmique-v2.md).

## Accès et analyse du portrait

Le tarif configuré est de 29 EUR/mois après sept jours d’essai. La conversation permet 100 réponses réussies par semaine. Ambre sélectionne des lectures dans le catalogue public et peut proposer des mises à jour des données personnelles à vérifier avant enregistrement.

Le Worker analyse le portrait. Pour les contextes longs, le workflow `.github/workflows/generate-articles.yml` exécute `scripts/run_journals.py` toutes les quinze minutes, manuellement ou après modification du pipeline. OIDC vérifie le dépôt, la branche, le workflow et l’événement. La clé OpenRouter reste dans Cloudflare.

Les anciennes adresses du personnage et de ses histoires redirigent vers Ambre. La migration `0014_retire_personal_stories.sql` ferme les demandes de récit ouvertes et préserve les données historiques dans l’export du compte.

## Vérification

```sh
npm test
node --test ../tests/*.test.cjs
python -m unittest discover -s tests -p 'test_personal*.py'
node tests/evaluate-recommendations.mjs
```

La recette et les versions réellement publiées figurent dans [le suivi V2](../docs/suivi-implementation-v2.md). Les preuves historiques V1 sont archivées dans les documents précédents ; elles ne remplacent pas une vérification du pipeline V2.

## Accompagnement V4

`0007` ajoute le portrait déclaré et le personnage unique ; `0008` les métadonnées d’essai Stripe ; `0009` conversations, consentement et propositions ; `0010` l’angle approuvé d’une histoire. Les migrations sont additives.

`OPENROUTER_CHAT_MODEL` configure séparément le modèle de conversation (défaut `anthropic/claude-sonnet-5.5`). La clé reste `OPENROUTER_API_KEY`. Les messages utilisent un retour 202 et `waitUntil`, puis un polling authentifié ; le délai fournisseur est borné à 25 secondes. Les réponses échouées ne consomment pas un message, mais restent dans les compteurs de tentatives et de budget fournisseur.

Routes : `GET/DELETE /api/chat`, `PUT /api/chat/consent`, `POST /api/chat/messages`, `GET /api/chat/requests/:id`, `POST /api/chat/proposals/:id/accept|reject`. Les corrections des pages personnelles utilisent leurs versions et droits existants. `POST /api/billing/cancel` programme la fin de l’abonnement ou de l’essai sans dépendre du fournisseur IA.

Configurer Stripe avec un prix mensuel EUR 2900 centimes, un webhook signé et les trois bindings serveur requis. Le succès des tests simulés ne remplace pas une recette Stripe en mode test puis le contrôle de production. Ne pas déclencher un paiement pour une simple vérification technique.


## Cartographie et révisions V6

Catalogue partagé : `data/psychology_catalog.json`, génération par `node scripts/build_psychology_catalog.mjs` (contrôle `--check` en CI). Les 53 repères restent des processus/ressources éditoriaux, pas une classification diagnostique. La carte conserve observations, fonctions possibles, alternatives, citations personnelles et contre-indices. Les copies de la même preuve ne font pas converger une hypothèse.

Les enregistrements relancent l’analyse native avec debounce court et contrôle sémantique séparé. Un cron Worker chaque minute récupère les tâches interrompues. Pour les contextes dépassant 130 000 caractères, la tâche est laissée au pipeline Python par lots (workflow toutes les 15 minutes) ; la couverture reste explicite. Aucun secret GitHub supplémentaire n’est nécessaire.

Le statut Stripe n’est pas modifié par le mode de test. Pour rétablir le paiement requis, modifier explicitement `FREE_ACCESS_ENABLED`; ne pas fabriquer d’abonnement dans la base. La clé OpenRouter reste un secret du Worker. Les analyses coûtent des crédits au propriétaire même si l’interface est ouverte gratuitement.

`0011_native_portrait_candidate.sql` ajoute un candidat non publié, lié au job et invalidé à chaque changement de source. La reprise vérifie ce candidat sans refaire la synthèse ; publication uniquement après contrôle sémantique. `OPENROUTER_PORTRAIT_MODEL` vaut actuellement `anthropic/claude-sonnet-5.5` ; le pipeline Python conserve le modèle analytique général.

## Cartographie de l’enfance enregistrée

`0016_cartographies.sql` crée `cartographies` (une ligne par compte : `answers` `{facteur: 0|1|2}` (non explicite, oui, souvent), `feedback` `{conséquence: "oui"|"non"}`, `consent`, `locked_at`, `version`, `updated_at`). `0017_cartographie_notes.sql` ajoute `notes` `{"factors":{id:"texte"},"outcomes":{id:"texte"}}`, `resources` `{protection: 1}` et `ambre_notes` `[{id,target,target_id,text,created_at}]`. Les deux migrations sont additives. Suppression en cascade avec le compte ; `/api/export` contient toute la ligne sous `cartographie`.

Routes authentifiées :

- `GET /api/cartographie` → `{cartographie, summary}` (les deux `null` sans enregistrement). `cartographie` : `answers`, `feedback`, `notes`, `resources`, `ambre_notes`, `consent`, `locked`, `locked_at`, `version`, `updated_at`.
- `PUT /api/cartographie` avec l’état complet `{answers, feedback?, notes?, resources?, consent: true, locked, expected_version}` (alias `version`, `null` ou `0` pour une première sauvegarde ; un bloc absent vaut un bloc vide). 80 000 octets au plus, 1 000 caractères par note, 80 notes. `400 consent_required` sans accord explicite, `400 invalid_cartography`, `409 revision_conflict`. `locked: true` exige au moins une réponse, une note ou une ressource. Un non explicite compte comme réponse mais jamais comme vécu dans le calcul. Le PUT ne modifie jamais `ambre_notes`.
- `DELETE /api/cartographie/ambre-notes/:id` retire une note d’Ambre sans changer `version` ; `DELETE /api/cartographie` supprime tout.

La cartographie n’est pas une source du portrait : l’enregistrer ne change pas `profiles.version` et ne relance pas d’analyse.

`src/cartographie.generated.mjs` (causes, conséquences, liens, poids, ressources) provient de `data/cartographie.json` via `node scripts/build_cartographie_module.mjs` (`--check` en CI et dans la construction Netlify). `src/cartographie.mjs` reprend le classement de `static/js/cartographie.js` ; un test compare les deux.

Ambre lit une cartographie enregistrée (`locked_at` renseigné) avec accord : le bloc `childhood_cartography` du contexte contient causes, convergences, pistes, retours, notes de la personne, ressources et ses propres notes, et les consignes `CHILDHOOD_CARTOGRAPHY_GUIDANCE` s’ajoutent au système. Elle écrit par le champ de sortie `cartography_notes` (3 par tour, 400 caractères, cible `factor`, `outcome` ou `general`) : le serveur valide les identifiants, écarte les doublons, plafonne à 60 notes en retirant les plus anciennes, et n’écrit rien en cas de danger immédiat, de demande de ne pas conserver ou de cadre fictif. Ces écritures ne changent ni les réponses ni `version`. Le message du tour porte `cartography_notes` quand des notes ont été ajoutées.

Déploiement : migration D1 `0017` puis Worker (`cartographie.mjs`, `cartographie.generated.mjs`, `chat-backend.mjs`, `people-observations.mjs`, `psychology-framework.mjs`).

## Parcours avec Ambre et bilans

`0018_ambre_reviews.sql` (additive) crée `ambre_reviews` : `kind` (`initial`, `point`), `status` (`running`, `ready`, `failed`), `content`, `error_code`, `idempotency_key` (unique par compte), `sources` (ce qui était disponible), `attempts`, `lease_token`, `lease_until`, `created_at`, `completed_at`. Un seul bilan `running` par compte ; suppression en cascade ; `/api/export` contient `ambre_reviews`.

La conversation ne commence qu’après une analyse initiale. `GET /api/chat` renvoie `journey` : `{started, sources:{cartography:{done,count}, portrait:{done}, people:{done,count}, life_events:{done,count}}, can_start, review:{running_id, last_at, next_point_at}}` (dates en secondes). `started` vaut vrai dès qu’une analyse initiale est `running` ou `ready`. Tant qu’il est faux, `POST /api/chat/messages` répond `403 journey_not_started` ; pendant la préparation d’un bilan, `409 review_in_progress`.

- `POST /api/chat/start` `{idempotency_key, accepted: true}` : `400 nothing_to_analyze` si aucune source, enregistre l’accord de conversation, crée l’analyse initiale, répond `202` avec l’état de la conversation et `review_id` (`200` si la même clé a déjà abouti, `409 journey_started` ou `review_in_progress` sinon).
- `POST /api/chat/review` `{idempotency_key}` : nouveau point ; `403 journey_not_started`, `409 review_in_progress`, `429 review_too_soon` avec `retry_at` si le dernier bilan abouti date de moins de 24 h.
- `GET /api/reviews?limit=&cursor=` (20 au plus par page, sans le texte : `excerpt` et `length`), `GET /api/reviews/:id` (texte complet, statut), `DELETE /api/reviews/:id` (retire aussi le message correspondant de la conversation ; supprimer l’analyse initiale demande de relancer la discussion).

Exécution : la demande fait une tentative immédiate bornée (`REVIEW_INLINE_TIMEOUT_MS`, 25 s par défaut, `0` pour la désactiver) sous un bail. Le cron minute (`scheduled`) reprend les bails expirés par lots (`REVIEW_CRON_BATCH`, 5 par défaut, 20 au plus) avec un délai de 120 s ; trois tentatives au plus, puis `failed`. Chaque tentative réserve des jetons dans `provider_budgets` (`WEEKLY_TOKEN_BUDGET`). Un bilan terminé est publié avec son message dans la conversation (`review_id`, `review_kind`), sans compter dans les 100 messages hebdomadaires. `OPENROUTER_REVIEW_MODEL` permet un modèle distinct (défaut : celui de la conversation).

Ambre ne propose plus d’articles ni de personnages : plus de `reading_candidates`, de `catalogue_lookup` ni de `reading_suggestions`. Le bloc `childhood_cartography` reçoit `evidence` : pour les conséquences reconnues ou en convergence, les liens des causes cochées (cause, niveau de preuve, résumé issu des sources), 6 000 caractères au plus, reconnues d’abord. Le dernier bilan accompagne chaque tour (`last_review`, borné).

## Correctifs accueil et sécurité du 16 novembre 2026

L’accueil, la confirmation de lancement, l’attente de l’analyse et la confirmation de bilan sont des panneaux intégrés à la conversation. Le menu reste utilisable, y compris sur téléphone. L’accueil présente Ambre comme une assistante psychologue IA à utiliser de préférence avec un professionnel de santé, sans diagnostic.

`chat-text.mjs` distingue les questions adressées à la personne des points d’interrogation dans les citations. Les citations et les incises restent dans leur phrase complète ; les deux questions directes au maximum sont conservées pour cette version. Le passage à une question et la relecture complète relèvent du chantier A.

`chat-safety.mjs` fournit un filet lexical conservateur pour les idées suicidaires directes ou indirectes, certaines demandes sur des moyens et les menaces de violence. Ce n’est pas une évaluation clinique exhaustive. Pour un signal détecté, une réponse déterministe est enregistrée dans le même INSERT que le tour, avant tout appel fournisseur. Elle est donc immédiatement visible et reste disponible après erreur HTTP, JSON invalide, délai dépassé, budget épuisé ou bail expiré. Elle est supprimée lors de l’effacement de la conversation ou du compte. Une réponse générée avec succès conserve les ressources de sécurité après les filtres ; aucune proposition, observation sur un proche ou note de cartographie n’est écrite pour ce tour. Les consignes générales connaissent le 3114, le 15 et le 112 même sans cartographie (et le 17 pour une violence immédiate). Aucun appel IA supplémentaire, aucune migration.

Validation locale : 374 tests du journal, 29 tests du site, 43 tests Python personnels, tous les contrôles de `.github/workflows/verify.yml`. Recette Chromium à 1440, 390 et 320 pixels : menu utilisable pendant l’accueil et la confirmation, aucun dialogue modal, aucun débordement horizontal ni erreur JavaScript. Les erreurs fournisseur et les courses avec expiration, retrait du consentement et purge sont simulées dans les tests ; les données de production ne servent pas à ces simulations.

## Moteur Ambre et transparence (0019)

La production active `createChatBackend({engineEnabled:true})`. Le moteur persiste les étapes `analysis`, `writing`, `review`, `rewrite`, `publish` dans `chat_engine_jobs`, sous bail. L'enregistrement précède la réponse HTTP 202 ; le polling authentifié et le cron reprennent les étapes restantes sous le même bail. Le secours déterministe est toujours enregistré avant un appel réseau. Les données privées sont relues et leur version contrôlée avant publication ; un retrait du consentement, une purge ou une modification des sources invalide le travail.

`ambre-engine.mjs` construit un index borné des sources, puis ouvre au plus huit détails pertinents. Le modèle principal reçoit les six derniers échanges et une mémoire structurée bornée. Analyse et relecture utilisent `OPENROUTER_ANALYSIS_MODEL` et `OPENROUTER_CRITIC_MODEL` (défaut `openai/gpt-5-nano`), la rédaction `OPENROUTER_CHAT_MODEL` (défaut `anthropic/claude-sonnet-5.5`). Les petits appels utilisent un effort de raisonnement minimal et une sortie réservée de 1 500 jetons (analyse/relecture) ou 2 400 (mémoire). Une relecture interrompue ou tronquée reprend seule, trois essais au plus ; les traces distinguent `output_limit`, délai dépassé et format invalide. Les consignes fixes demandent le cache fournisseur. La relecture applique les quinze critères cliniques, des contrôles déterministes et au plus une réécriture. Un résultat non validé est remplacé par une réponse prudente, sans modification des sources personnelles.

La mémoire évolue par lots d'au moins quatre, au plus huit échanges. Elle conserve des citations vérifiables des messages de la personne, pas des inventions de l'assistant. Une correction ou un effacement avance le repère de lecture : les anciens échanges ne recréent pas la note effacée. Les sujets fermés et les appuis de sécurité sont prioritaires dans le budget de contexte. Une limite conservatrice empêche l'exploration si toutes les fermetures ne tiennent plus dans le résumé.

- `GET /api/chat/insight?offset=0&turns_offset=0` : contexte actuel réel, sources consultables par pages de 100 (`current.sources.next_offset`), mémoire, résumés de traces par pages de 20 (`turns_next_offset`), modèles configurés et règles. Les données enregistrées hors usage direct de la conversation sont distinguées.
- `GET /api/chat/insight?turn_id=ID` : uniquement `selected_turn`, vérifiée contre le propriétaire du compte. Les nouvelles traces V2 conservent les entrées et consignes effectivement transmises à chaque appel, son modèle demandé, le modèle indiqué par le fournisseur lorsqu’il est disponible, son statut et sa consommation. `ambre-insight.mjs` déduplique les valeurs et limite le détail à 128 Ko ; toute omission est explicite. Aucun raisonnement interne retourné par le fournisseur ni secret n’est conservé. Les anciennes traces n’ont pas de détails reconstructibles. Les entrées sont des instantanés historiques : modifier une source ne réécrit pas un appel passé.
- `PUT /api/chat/memory` : `{revision,sections:{goals,facts,hypotheses,tried,alliance,safety,closed,session}}`, tableaux de textes (300 caractères chacun ; 12 par rubrique, 30 sujets fermés). `{sections:{}}` efface la note. Une révision obsolète renvoie 409.
- `DELETE /api/chat` purge aussi mémoire, tâches et traces. L'export et la suppression du compte couvrent ces tables.

## Suivi psychologue (0020)

Les comptes existants restent `member`. L'inscription accepte `role:"psychologist"` et `professional:{name,title,registration_number}`. Ces informations sont **déclarées, non vérifiées** ; elles ne constituent pas une vérification d'identité ou de titre. L'adresse électronique n'est pas vérifiée. Un compte professionnel seul ne donne aucun accès à un dossier.

La personne crée un partage avec une adresse, des rubriques et un accord explicite. Un code aléatoire de 128 bits, conservé seulement sous forme d'empreinte, doit être transmis directement au professionnel. Il expire après sept jours et ne peut être activé que par le compte professionnel existant au moment de la demande. La réponse de création ne révèle pas l'existence du destinataire. Aucun courriel n'est envoyé. Le professionnel consulte uniquement les rubriques autorisées, en lecture seule. Les accès, refus et changements de partage sont journalisés ; le partage est revérifié avant de renvoyer les données.

API authentifiée, mutations avec contrôle d'origine :

- `GET /api/care` : cadre, première page de partages, séances et points ; les champs `*_next_offset` signalent la suite.
- `PUT /api/care/settings` : `{has_psychologist:true|false|null,frequency_days:1..365|null}`.
- `POST /api/care/links` : `{email,scopes,accepted:true}` ; retourne le code une seule fois. Rubriques : `conversation`, `cartography`, `portrait`, `people`, `life_events`, `reviews`.
- `PUT /api/care/links/:id` : `{revision,scopes,accepted:true}` ; `DELETE` avec `{revision}` révoque. Changer le périmètre invalide aussi les points existants et en cours.
- `POST /api/care/activate` : `{code}`, compte psychologue uniquement.
- `POST /api/care/sessions` : `{session_at}` (secondes Unix) ; `PUT /:id` ajoute `revision` ; `DELETE /:id` exige `revision`.
- `GET /api/care/links`, `/sessions`, `/briefs`, `/access-log` : pagination `?offset=…` et `next_offset`.
- `GET /api/care/patients` : liste paginée du professionnel. `GET /api/care/dossiers/:linkId` : métadonnées, séances et points. `GET /:scope` ouvre une rubrique autorisée ; `/sessions` et `/briefs` donnent les pages suivantes.
- `POST /api/care/briefs` : `{link_id,session_id}` ; préparation asynchrone, `GET /api/care/briefs/:id` pour lire son état et son contenu. La personne et son professionnel autorisé lisent le même point.

Le point avant séance couvre la période suivant la séance précédente, à partir des seules rubriques partagées. Le cron le prépare dans les 24 heures précédant une séance et reprend les lots sous bail. Les citations doivent figurer exactement dans l'extrait envoyé. Les synthèses IA sont attribuées comme telles ; l'absence de données et la couverture partielle restent visibles. Les versions du partage, de la séance et des sources sont contrôlées avant publication. Révoquer le partage efface le contenu des points correspondants et empêche une réponse tardive de le rétablir. Les limites journalières de préparation et le budget de jetons par personne sont partagés entre demandes manuelles et automatiques.

## Pages et déploiement A/B/C

`/parametres/` remplace Mon compte dans le menu et conserve un bouton vers `/mon-compte/`. `/mon-suivi/` contient les réglages, séances, partages, points et accès. `/espace-psychologue/` permet d'activer un code et de consulter les dossiers. Les nouvelles pages disposent de routes Worker et de proxys Netlify ; les réponses privées restent `no-store`.

Appliquer **0019 puis 0020**, après point de restauration D1 et suspension temporaire des générations ; ne pas rejouer une migration déjà appliquée. Déployer ensuite tous les modules Worker, conserver les secrets existants, retirer la suspension, puis fusionner les routes/menu dans `main`. Vérifier les deux rôles, le moteur réel, les points, la révocation et la suppression des comptes temporaires en production.

## Interface et pertinence de la cartographie — novembre 2026

Six thèmes en accordéons fermés au chargement, sans parcours imposé ni septième thème. Les anciennes ressources sont préservées en base. Pour un membre connecté, les nouvelles saisies déclenchent la sauvegarde avec `consent: true`, sans case supplémentaire ; une notice explique enregistrement et usage par Ambre. Sans connexion, seul le brouillon local est écrit. La file de sauvegarde conserve les saisies pendant un PUT, affiche les conflits 409 sans les écraser et attend un PUT en cours avant DELETE.

Le classement organise des pistes de lecture : aucun score de risque individuel ni multiplicateur d’intensité. Les familles de vécus évitent certains recouvrements, sans prétendre indépendance statistique. Les modèles cliniques, définitions et contextes de classification ne deviennent pas des preuves empiriques par addition. Les restrictions du jeune enfant et les réponses négatives restent explicites, y compris dans le contexte d’Ambre. Le résumé contient les références disponibles et leurs limites ; le moteur sélectionne des extraits bornés, en conservant les métadonnées prudentes.

Le périmètre audité est décrit dans `docs/audit-cartographie-transparence-2026-11-16.md`. Les 18 schémas de Young sont référencés ; ce fait n’implique pas une couverture exhaustive du DSM-5-TR ni une validation psychométrique. Le menu du portrait contient uniquement À propos de moi et Mon portrait ; les anciens liens directs vers les lectures restent utilisables.
