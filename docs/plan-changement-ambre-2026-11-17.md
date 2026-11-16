# Plan de changement d'Ambre — 17 novembre 2026

**Version 3 — plan retenu pour implémentation après deux cycles de critique et vérification indépendante finale.** Ce document ne constitue ni une livraison ni une validation clinique. Le catalogue machine est préparé dans la livraison documentaire décrite en fin de document ; le moteur, les migrations et l'interface sont des chantiers séparés.

Recherche associée : [36 stratégies et sources réellement consultées](recherche-strategies-changement-2026-11-17.md). Fondations : [audit moteur, §13–14](audit-ambre-2026-11-17.md), [pratique clinique, §6–8](recherche-pratique-clinique-2026-11-16.md), [reprise](reprise-2026-11-16.md).

## 1. Résultat attendu et périmètre

Ambre doit pouvoir aider une personne qui le souhaite à choisir un petit changement, garder une trace fidèle de ce qu'elle accepte d'essayer, revenir doucement sur les résultats et ajuster la suite. Elle continue à écouter lorsqu'un plan ne convient pas. Les méthodes disponibles sont choisies en fonction de la demande et du contexte, sans attribuer de diagnostic ni transformer un repère de cartographie en prescription.

L'outil reste utilisable sans psychologue. Avec un professionnel, il prépare et prolonge le travail déclaré par la personne ; il ne se substitue pas à son jugement. Le professionnel conserve un accès en lecture seule aux seuls périmètres partagés volontairement.

**Demandes déjà établies dans la reprise** : catalogue sélectionné par tour ; objectifs/étapes/essais/résultats persistants ; retour sans harcèlement ; articulation avec le suivi ; critères de relecture ; transparence Paramètres ; critiques et révisions multiples avant implémentation. **Contraintes de ce chantier** : pas de nouveau modèle à chaque tour, environ 12 000 jetons d'entrée cumulés pour un tour habituel, au plus deux relances dans la conversation et aucune notification externe.

**Choix retenus pour la proposition v3 après deux critiques** : adoption par carte explicite, trois objectifs actifs, deux relances au plus espacées de sept jours, préparation personnelle déterministe, agrégat JSON versionné sous CAS et politique de suppression ci-dessous. Les plafonds de contexte/stockage restent des paramètres produit à mesurer. Ce ne sont ni des normes cliniques ni des décisions prises à la place de la personne qui utilisera Ambre.

Hors périmètre : thérapie autonome, traitement de trauma, EMDR/rescriptage, changements médicamenteux, sevrage, exercices physiologiques intenses, série d'expositions spécialisées, notification push/courriel/SMS, écriture du professionnel dans le dossier, diagnostic par questionnaire, gamification de l'observance.

**Rôles et défauts explicites** : toutes les routes privées `/api/plan` sont réservées au compte membre propriétaire authentifié ; le rôle professionnel n'y accède pas en invoquant un lien de suivi. Une vue professionnelle passe exclusivement par les routes de partage et leur liste blanche. Les propositions dans le chat utilisent le consentement de conversation déjà existant, sans en déduire un engagement ; `proposals_enabled:true` peut être désactivé dans Paramètres. Désactiver les propositions suspend seulement les nouvelles suggestions et conserve les objectifs acceptés accessibles. Les retours sont inactifs par défaut (`followup_opt_in:false`) ; le périmètre `plan` est absent des liaisons existantes ; les consignes rapportées du professionnel sont privées et séparées de `care`.


## 2. Préconditions moteur et point de coordination

Les corrections de l'audit sont un préalable à l'ajout de contexte : réserver une réponse et sa relecture sur profil rempli/maximal ; respecter les limites de budget sans faux message hebdomadaire ; dates lisibles ; historique gardant la question finale ; mémoire récente ; règles de sécurité tenant compte du contexte ; cadre du suivi transmis au relecteur ; distinction faits/hypothèses ; suppression des faux positifs connus. Le chantier moteur peut les corriger indépendamment du présent plan.

**Contrat proposé au chantier moteur** : prévoir des extensions optionnelles et bornées, absentes par défaut ; ne pas activer le catalogue ni créer des engagements tant que le plan n'a pas été révisé. Le numéro `0021` cité dans l'audit n'est pas réservé : choisir la prochaine migration libre au moment d'intégrer.

```text
loadChangeContext(env, userId, now)
  -> {available, revision, epoch, context_generation, preferences, goals[], active_trial?, due_trial?, context_hash}
selectChangeCandidates({message, plan, allowed_topics, care})
  -> [{id, label, intent_tags, restriction}]  // 0..6, pas de détail biographique
loadStrategyDetails(ids)
  -> 0..2 fiches validées, au plus une action développée dans la réponse
validateChangeDraft({analysis, writing, snapshot, message})
  -> propositions vérifiées / opérations de relance / refus explicites
```

Les fonctions ci-dessus sont des **interfaces proposées**, indépendantes du stockage par agrégat. `loadCareContext(env,userId,now)` garde sa signature et sa vue publique existantes. Les nouvelles consignes privées passent dans un chargeur distinct `loadPrivateCareInstructions(env,userId,now)`, jamais dans le sérialiseur du dossier professionnel. Une absence de migration ou de catalogue donne `available:false` et désactive le changement, sans faire croire à un plan vide que l'on pourrait recréer.

## 3. Catalogue et sélection sans appel supplémentaire

Après validation documentaire, `data/change_strategies.json` sera la source unique, compilée vers un module Worker séparé du catalogue de psychologie. Chaque fiche contient : `id`, `version`, famille, libellé simple, intentions compatibles, indications, exclusions, type de permission, un petit pas, mesures facultatives, seuil d'orientation en mots, sources avec type/lecture/passage, et statut documentaire `draft|reviewed|disabled`. Les identifiants stables sont ceux du dossier : C01–C09, D01–D06, A01–A05, SCH01–SCH04, K01–K03, M01–M04, G01–G05 ; ils ne seront pas renommés en identifiants anglais à la compilation. Les sources gardent S01–S18, espace de noms distinct.

Un lien vers un repère clinique est facultatif, sourcé et ne signifie jamais « ce repère prescrit cette technique ». Chaque fiche possède aussi `restriction:self_help_support|prepare_session_only|disabled`, indépendante du statut documentaire, ainsi que `source_refs` (ID, URL, type, lecture, passage) et `research_ref` vers son entrée. Toutes les entrées proposées sont `self_help_support`, sauf C09 qui commence `prepare_session_only`, jusqu'à relecture clinique spécifique. `self_help_support` décrit un soutien limité et ne vaut pas validation clinique. Un champ `requires_reported_professional_exercise` peut restreindre une variante de continuité, sans autoriser une nouvelle technique intense. Les techniques plus intensives exclues du dossier ne sont pas sérialisées comme options cachées.

Le format compact conserve systématiquement `id`, `restriction`, `permission`, `blocked_contexts` et les préconditions critiques (par exemple `requires_safe_interpersonal_context` pour C07/C08/D06). C0 est une règle commune immuable côté serveur ; les exclusions particulières restent structurées. Les explications, exemples et sources longues sont les seuls champs compressibles. Si une fiche ne tient pas avec ses restrictions intactes, **la fiche entière est omise**. Aucun découpage à 1 600 caractères ne peut retirer une exclusion ou transformer C09 en exercice d'exposition.

Cas de contrôle : violence/coercition rapportée et demande de phrase pour dire non → C08 n'est pas admissible à une mise en situation ou confrontation. Ambre répond au besoin de sécurité et d'appui, sans pousser à tester la réaction de l'auteur des violences. Le simple fait que la personne demande une formule ne prouve pas un contexte interpersonnel sûr.


Flux proposé :

1. **Présélection déterministe** de zéro à six fiches : demande actuelle, essai en cours, préférences et restrictions ; bonus de pertinence pour une stratégie déjà acceptée, jamais remplissage artificiel. Pas de sélection sur un mot isolé ni sur un diagnostic supposé. Si aucune correspondance fiable, liste vide.
2. **Analyse existante** : reçoit seulement les candidats courts et l'état compact du plan. Elle sélectionne zéro à deux identifiants dans cette liste. Elle peut aussi repérer un objectif explicitement formulé, un choix ou un retour d'essai, avec citation du message actuel. Pas de nouvelle interprétation biographique.
3. **Validation serveur** : liste blanche et restrictions prévalent sur le modèle. Les identifiants sont vidés en sécurité, stabilisation aiguë, réparation, écoute seule, `no_advice`, sujet fermé ou préférence désactivée. Un retour spontané peut être reconnu sans proposer de nouvel exercice.
4. **Rédaction existante** : reçoit au plus deux fiches détaillées, un objectif courant et un essai pertinent. Elle développe au plus une action. Une demande d'options peut recevoir deux options brèves, sans créer deux tâches.
5. **Relecture existante** : dispose du même cadre compact, des restrictions, du suivi et des critères de changement. Contrôles déterministes avant les critères sémantiques. Pas de quatrième modèle systématique.
6. **Publication atomique** : seules les propositions effectivement publiées sont sauvegardées ; une carte ne vaut pas acceptation. Toute invalidation de révision annule les écritures et relances associées.

Le détail de tout le catalogue reste hors prompt. Les réponses n'affichent pas une liste de techniques ou d'étiquettes ; Paramètres peut expliquer la méthode et ses sources à la demande.

### Extension compacte de l'analyse

```json
{
  "strategy_ids": ["C04"],
  "change_signal": {
    "kind": "none|goal_stated|trial_requested|trial_reported|obstacle|stop_requested",
    "target_id": null,
    "quote": "citation exacte du message actuel"
  }
}
```

`kind:none` se sérialise sans `quote` ni `target_id`. Une citation doit être un segment exact du message, borné à 240 caractères ; ce plafond est un choix de budget. Un identifiant doit appartenir au plan de ce compte et à l'instantané du tour. Une ambiguïté (« oui », deux essais possibles, passé/futur incertain) ne déclenche aucune transition d'acceptation ou de résultat.

L'analyse détecte ; elle ne modifie pas le dossier. La rédaction peut produire un brouillon structuré limité, associé à l'action réellement proposée. Le serveur vérifie les données factuelles et l'absence d'instructions de contrôle dans les champs. Les consignes rapportées par la personne sont des données non fiables pour le système : elles peuvent guider une réponse compatible, jamais neutraliser sécurité ou règles d'accès.

## 4. Le plan persistant : ce qui est proposé, choisi, essayé et observé

Quatre objets sont distingués pour éviter les faux engagements :

| Objet | Contenu | Qui en décide ? |
|---|---|---|
| Proposition | Hypothèse de travail ou petit pas suggéré, origine Ambre/personne, citation, date | Ambre peut la rédiger ; elle reste proposée |
| Objectif | Changement souhaité dans les mots de la personne, raison/valeur facultative, priorité, état | Personne ; création directe ou adoption explicite d'une proposition |
| Étape | Action concrète rattachée à l'objectif, stratégie éventuelle, limites, mesure choisie | Personne ; éditable avant adoption |
| Essai | Une occurrence datée d'une étape : intention, résultat rapporté, effets, suite | Personne ; Ambre propose une transcription corrigeable |

Un objectif n'est pas une hypothèse clinique. Une étape n'est pas un résultat. Répéter une étape crée un nouvel essai et préserve l'observation précédente ; modifier un texte n'écrase pas rétroactivement les conditions d'un ancien essai. Les données d'essai gardent un instantané minimal de l'action et de la mesure acceptées.

Schéma logique : propositions, objectifs, étapes et essais gardent des identifiants stables, une révision propre et les champs décrits ci-dessous. **Choix physique retenu pour v3 : un agrégat JSON versionné par propriétaire**, plutôt que quatre pipelines. Les objets et les points personnels sont des collections de cet agrégat ; leurs états ne se confondent pas.

```text
change_plans(user_id PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  revision, epoch, document_json, bytes, updated_at)
document_json = {
  schema_version, proposals[], goals[], steps[], trials[], preparations[],
  private_care_instructions[], private_care_revision
}

proposition : id, revision, kind, target_id?, payload,
  status, source_turn_id?, source_quote?, created_at, resolved_at?
objectif : id, revision, title, why?, priority, status,
  origin, accepted_at, created_at, updated_at
étape : id, revision, goal_id, text, strategy_id?, strategy_version?,
  measure?, professional_context?, status, accepted_at, dates
essai : id, revision, step_id, step_snapshot, status,
  planned_for?, occurred_at?, occurred_on?, timing_note?, result_text?,
  attempt?, outcome?, adverse_effect_text?, followup_opt_in,
  review_mode, review_on?, reminder_count, last_reminder_at?, dates

change_request_keys(user_id, request_key, epoch, request_hash,
  entity_type, entity_id, state, created_at)
change_artifact_refs(user_id, artifact_type, artifact_id,
  entity_type, entity_id, revision, context_generation)
```

Le propriétaire, le plan revision/epoch et les indicateurs de contrôle durables existent même avec un document vide ; une purge les fait avancer sans les réinitialiser. `context_generation`, les choix de propositions/relances et les indicateurs de refus/vigilance restent dans l'état de contrôle du compte, séparés du document effaçable. Le champ de génération est aussi enregistré sur chaque tour.

Une mutation charge l'agrégat borné, valide les références internes et les états, puis remplace le document par `UPDATE ... WHERE user_id=? AND revision=? AND epoch=?`, avec révision augmentée. Les compteurs et dates restent du ressort du serveur. Une mutation administrative concurrente provoque le même 409 qu'une modification de texte ; aucun dernier écrivain gagnant silencieux. La publication conversationnelle et la mutation de l'agrégat restent atomiques sous bail et gardes : l'agrégat ne dispense pas de ces contrôles. Les dépendances d'artefacts et marqueurs d'idempotence restent des métadonnées séparées pour pouvoir supprimer le document sans perdre les protections de rejeu.

Bornes initiales proposées : 20 objectifs conservés dont 3 actifs, 60 étapes, 200 essais, 40 propositions contenant encore un texte, 20 points personnels ; document total **256 Kio UTF-8 maximum**. La première borne atteinte arrête seulement la nouvelle écriture concernée, avec explication et possibilité d'exporter puis de supprimer volontairement des éléments. Aucune rotation silencieuse, aucun essai ancien effacé pour faire de la place. Les cartes résolues peuvent n'avoir plus qu'une métadonnée de résolution, sans doublon du texte adopté. Le registre des requêtes est soumis au quota de mutations et ne conserve aucun payload ; son stockage est mesuré séparément, sans purger les protections puis autoriser un ancien rejeu.

Textes bornés avant sauvegarde, jamais tronqués après acceptation : objectif 240 caractères, raison/étape 800, résultat/effet indésirable 1 200 chacun, citation 240, instruction professionnelle 1 200 (au plus 10), point personnel 6 000. Une valeur trop longue est refusée avec conservation de la saisie locale ; elle n'est pas amputée. Ces limites sont affichées dans l'éditeur et l'export restitue **tous** les objets effectivement conservés, indépendamment des pages d'affichage. À confirmer par essais de charge avant activation.

Les collections internes sont exposées par pages de 50 maximum ; pas d'envoi automatique du document complet au navigateur ou aux modèles. Contrôles explicites de rôle/propriété sur toutes les routes, références internes et IDs imbriqués. Pas d'ancien texte dans un journal de révision : un journal d'opérations éventuel conserve seulement des métadonnées, jamais un payload supprimé.

États proposés :

- Proposition : `proposed|accepted|dismissed|expired`. Pas de relance d'une proposition non adoptée ; expiration technique ne signifie pas refus de la personne.
- Objectif : `active|paused|completed|abandoned`. Seule la personne conclut qu'il est atteint. Pas de fermeture automatique par score.
- Étape : `active|paused|completed|abandoned`, uniquement après adoption ; les brouillons restent dans les propositions.
- Essai : `planned|reported|cancelled`. `reported` contient `attempt:done|partial|not_tried|stopped|uncertain` et `outcome:helpful|neutral|worse|uncertain|null`. Ainsi « non essayé » ne clôt pas automatiquement l'objectif et « fait » ne signifie pas « utile ».

Dates : les instants techniques sont stockés UTC ; **Europe/Paris est le fuseau fixe retenu pour cette version**, mentionné à côté des dates dans l'interface, puisque le compte n'a pas encore de préférence de fuseau. Un jour sans heure est conservé comme `YYYY-MM-DD`, avec `date_precision:day`, et comparé au jour courant obtenu par `Intl.DateTimeFormat` en Europe/Paris ; aucune conversion implicite à minuit UTC. Un fait sans date a `occurred_at:null` et `occurred_on:null` : la date de saisie ne devient pas sa date. « Cette semaine » reste un `timing_note` libre, sans échéance automatique. `created_at`/`updated_at` sont séparés de ces dates déclarées.

Plafonds de départ proposés : trois objectifs actifs, une étape mise en avant par objectif, un seul essai à discuter dans un tour. Les autres objets restent accessibles et paginés ; si une limite d'activation est atteinte, l'interface l'explique et propose de choisir une priorité, sans effacer le travail. Ces nombres sont à valider à l'usage.

## 5. Adoption, corrections et révisions

Une carte présente « Ce qu'Ambre propose » avec **Choisir cet essai**, **Modifier**, **Pas maintenant**. L'adoption enregistre le contenu réellement affiché et accepté, la révision et l'heure. L'utilisateur peut aussi créer un objectif ou saisir directement son résultat depuis Mon suivi, sans modèle.

Dans la conversation, « je vais le faire » peut préparer une carte ciblée ; il ne justifie pas de déduire l'accord à une date, mesure chiffrée, relance ou partage. Un retour « j'ai essayé » prépare une proposition de résultat reliée au bon essai ; si le lien est ambigu, la carte demande de choisir l'essai. Les acceptations conversationnelles automatiques sont laissées hors de cette version : le bénéfice de fluidité devra être comparé au risque d'engagement inventé après tests.

Correction d'un résultat : la valeur corrigée devient la source courante, `revision` augmente, synthèses préparées invalidées. Les métadonnées peuvent indiquer « corrigé le… », sans retenir indéfiniment l'ancien texte. Une suppression retire l'objet des prompts, recherches, exports futurs et comptes rendus accessibles après invalidation. Un objet abandonné reste visible dans l'historique tant qu'il n'est pas supprimé.

## 6. Relances douces, dans une conversation ouverte par la personne

La personne choisit séparément « Ambre peut revenir sur cet essai ». Par défaut, `followup_opt_in:false` et `review_mode:none`. Elle peut choisir `on_date` avec un jour précis ou `next_conversation`. Ce dernier devient admissible à partir du **prochain message utilisateur envoyé après l'adoption**, hors du tour qui a proposé/adopté l'essai ; il ne déclenche rien à l'ouverture de la page. Dans les deux modes, la pertinence du sujet et les autres exclusions restent nécessaires. Sans choix de mode, aucune échéance n'est déduite.

Éligibilité calculée côté serveur :

1. Essai `planned`, objectif/étape actifs, retour autorisé, date atteinte si renseignée.
2. Aucune suspension/refus de sujet, de conseil ou de suivi, aucune voie sécurité/stabilisation/réparation/écoute seule.
3. Le sujet de l'essai est pertinent dans la conversation courante. Une échéance n'est pas une raison de détourner une demande urgente ou indépendante.
4. Moins de deux relances **sur toute la durée de cet essai**, et espacement d'au moins 604 800 secondes UTC (sept jours complets) entre relances ; pas deux relances sur deux tours consécutifs même si le calendrier le permet.
5. Un seul essai admissible retenu, par date puis priorité ; pas de file de rappels qui finit par être vidée en rafale.

Une relance est une invitation facultative, pas un reproche ni la présomption que l'exercice aurait dû être fait. Après deux invitations sans retour, aucune nouvelle relance de cet essai. L'interface peut afficher « Ambre n'y reviendra plus spontanément » ; un nouveau cycle nécessite un choix explicite de la personne, sans remise à zéro silencieuse du compteur par édition de date.

La rédaction renvoie, si pertinent, `followup:{trial_id, quote}`. `quote` doit être une citation exacte et non vide de l'invitation dans le **texte final publié**, bornée à 300 caractères et désignant le seul essai admissible dans ce tour. Une réécriture doit restituer son propre objet `followup` ; l'ancien ne se transmet pas automatiquement. Le serveur vérifie présence exacte dans la réponse finale, ID admissible, permission, compteur et temporalité ; la relecture vérifie que le passage est réellement une invitation facultative portant sur cet essai. Un booléen ou la présence du nom de l'essai ne suffisent pas. Si la citation manque ou échoue, aucune relance n'est comptée et la réponse ne doit pas contenir une invitation non traçable.

Compteur incrémenté **dans la même transaction que la publication** validée. Aucun compteur consommé par échec fournisseur, relecture rejetée, réponse annulée, reprise de tâche ou conflit de révision. Un identifiant de publication unique interdit un double comptage. Une réécriture sans invitation ne la compte pas.

« Je n'ai pas eu le temps », « j'ai honte de ne pas l'avoir fait », silence et abandon sont distingués. Un résultat rapporté retire l'essai de la file ; l'arrêt demandé désactive le retour immédiatement et n'attend pas un prochain appel modèle. Aucune tâche planifiée ne contacte la personne hors conversation.


### Arrêter sans attendre un modèle

Un bouton **Suspendre les retours d'Ambre** reste accessible dans la conversation et Mon suivi ; un bouton par essai désactive son retour. `PUT /api/plan/followups` avec `enabled:false` et cible `all` ou un ID du compte est traité immédiatement, sous révision, sans modèle, même quand un tour est en cours. Ce changement invalide son instantané de publication. Il ne supprime ni objectif ni résultat. Pour `enabled:false`, une révision client ancienne ne doit pas perdre l'arrêt : le serveur applique cette restriction seule à l'état courant, avec reprise CAS bornée ou mise à jour atomique dédiée, sans écraser les autres champs. Réactiver exige au contraire une révision exacte et le clic explicite. Si une panne empêche de sauvegarder l'arrêt, le tour en cours ne doit pas publier de relance et l'interface signale l'erreur.

Avant toute analyse, un parseur déterministe accepte une liste étroite de commandes directes couvrant tout le message normalisé (espaces/ponctuation finale, casse et apostrophes), par exemple « Ne me relance plus », « Arrête les relances », « Ne me rappelle plus les exercices ». Il suspend globalement les retours et accuse réception factuellement. Les citations rapportées (« elle m'a dit… »), négations de l'arrêt (« n'arrête pas… »), conditionnels et mot isolé « arrête » ne correspondent pas. Une demande ciblée sans ID certain provoque au plus une clarification, sans suppression ni réactivation. Le bouton offre toujours un arrêt immédiat non ambigu.

Un arrêt est durable : seul un clic explicite **Réactiver les retours**, montrant le périmètre, peut le lever. Le modèle, une modification d'objectif/date et un nouveau message ne le peuvent pas. Une réactivation globale ne remet aucun compteur d'essai à zéro et ne réactive pas une autorisation d'essai retirée séparément.

## 7. Articulation avec le psychologue et préparation de séance

Dans Mon suivi : champ privé « Ce que mon professionnel m'a proposé de travailler », daté et modifiable, avec mention **rapporté par vous**. Une instruction peut être liée à une étape ; la personne peut préciser qu'elle est actuelle, à vérifier ou terminée. Aucun titre, diagnostic ou intention du professionnel n'est inventé. Aucune validation professionnelle implicite à partir de la seule présence d'un compte lié.

`loadPrivateCareInstructions` est réservé au moteur du propriétaire et à son écran personnel. `serializeProfessionalCare` reste une liste blanche des champs publics historiques ; ni la route dossier existante ni `GET /api/care/patients/:id` ne reçoit ces consignes. `serializeSharedPlan` est distinct et retire aussi références/citations privées. Le moteur reçoit un extrait seulement s'il concerne l'essai en cours. En cas de contradiction ou d'incertitude, Ambre peut préparer la question de séance ; elle ne recommande pas de contredire une consigne de soin. Une consigne rapportée dangereuse ne neutralise pas la voie sécurité.

**Point personnel avant séance**, disponible sans compte professionnel lié et sans partage : dans Mon suivi, choisir une séance existante ou « Préparer un prochain rendez-vous » sans date, puis **Préparer mon point personnel**. Une composition déterministe, sans appel modèle ni nouvelle file, affiche les objectifs acceptés, essais/résultats depuis la précédente séance (ou une période choisie), effets pénibles rapportés et une zone « Ma question ». Les faits sans date apparaissent dans une section « Date non précisée », jamais inclus artificiellement dans une évolution temporelle.

La personne peut consulter à côté les seules rubriques privées de mémoire `goals`, `tried` et `session`, dans une section **Notes de conversation à vérifier**, en précisant leur absence de provenance persistante ; elles ne sont pas ajoutées au point par défaut. Un ajout volontaire conserve `origin:memory_unverified` et ne devient ni fait confirmé ni contenu partageable automatiquement. Hypothèses, sécurité, sujets fermés et alliance ne sont pas importés par défaut. La personne peut saisir sa propre reformulation en note personnelle.

Prévisualiser, modifier chaque rubrique, retirer un élément, puis **Enregistrer mon point personnel** sauvegarde un document privé avec révision et références aux objets utilisés. Le brouillon n'est pas conservé côté serveur avant ce clic ; annuler l'abandonne. Le document enregistré reste privé, consultable, corrigeable et supprimable jusqu'à suppression par la personne ou invalidation de ses sources. Il n'expire pas silencieusement et n'est pas automatiquement partagé à une prochaine liaison professionnelle. Un éventuel partage du point exige une nouvelle prévisualisation à partir des seules sources autorisées ; les notes sans provenance en sont exclues. Le point n'attribue pas « amélioration » ou « aggravation clinique » à une mesure courte.

**Partage opt-in** : nouveau périmètre `plan`, absent des liens existants et non activé par migration. Le choix montre les rubriques exposées et un aperçu. Le partage du plan n'accorde pas celui de la conversation ou du champ privé de consigne professionnelle. La lecture partagée du plan omet citations brutes et identifiants de sources non partagées ; les formulations d'objectifs/essais choisies sont les données partagées, ce que l'aperçu doit rendre clair.

Un point destiné au professionnel n'utilise que les périmètres actuellement autorisés. Une mémoire globale construite à partir de sources privées ne doit pas être injectée au motif qu'elle « résume » des données : soit chaque item a une provenance autorisée, soit elle est exclue du point partagé. Le point personnel peut utiliser la mémoire du compte dans ses limites habituelles, et le résumé partagé est reconstruit depuis les sources autorisées.

Révocation ou changement de périmètre bloque les nouvelles lectures/publications et invalide les copies de comptes rendus servies par l'application ; contrôle d'autorisation avant lecture et avant publication, réponses privées sans cache partagé. L'interface ferme ou retire les copies encore contrôlées par l'application lorsqu'elle apprend la révocation. **Elle ne peut pas retirer un document déjà téléchargé, copié ou capturé par son destinataire**, ni annuler une lecture réseau déjà livrée. Cette limite est indiquée avant le partage ; aucune suppression à distance n'est promise. Le serveur ne peut reprendre une donnée depuis le cache d'un point révoqué. Le professionnel reste en lecture seule. Une personne ne reçoit aucune notification externe de ce chantier.

## 8. API proposée et sécurité de concurrence

API sous `/api/plan`, en module séparé, avec les conventions d'authentification et de réponses JSON du projet. Les noms restent à valider avec l'intégration.

| Route | Fonction et protections |
|---|---|
| `GET /api/plan?cursor=…` | État courant, préférences, propositions et objets paginés ; limites explicites, `revision`, `epoch`, `next_cursor` |
| `PUT /api/plan/preferences` | Activation, préférences de suivi ; compare `revision` et `epoch` |
| `POST /api/plan/goals` | Création volontaire ; clé d'idempotence, longueur et propriété vérifiées |
| `PATCH /api/plan/goals/:id` | Texte, priorité ou état ; compare la révision de l'objet et l'epoch |
| `POST /api/plan/proposals/:id/accept` | Adoption du contenu affiché ou édité, référence/révision exactes ; transaction unique |
| `POST /api/plan/proposals/:id/dismiss` | Refus sans relance ; opération idempotente |
| `POST /api/plan/goals/:id/steps` | Étape manuelle ou issue d'une proposition ; stratégie optionnelle |
| `PATCH /api/plan/steps/:id` | Texte/mesure/état corrigés sous révision ; anciens essais gardent leur instantané accepté |
| `DELETE /api/plan/steps/:id` | Retire étape et essais/propositions dépendants, purge des copies dérivées |
| `DELETE /api/plan/proposals/:id` | Supprime le contenu de la proposition et les cartes liées ; ne supprime pas un objectif déjà adopté séparément |
| `PUT /api/plan/followups` | Arrêt/réactivation explicite, cible `all` ou essai ; aucune réactivation par modèle |
| `POST /api/plan/steps/:id/trials` | Nouvel essai d'une étape acceptée ; instantané minimal |
| `PATCH /api/plan/trials/:id` | Résultat/date/retour souhaité ; `occurred_at:null` et `occurred_on:null` admis, jamais transition de propriété |
| `POST /api/plan/steps/:id/results` | Création volontaire d'un essai déjà rapporté, même sans essai prévu et sans date connue ; déduplication explicite, pas de date inventée |
| `DELETE /api/plan/trials/:id` | Retire essai, propositions dépendantes et copies générées associées |
| `GET /api/plan/preparation` | Composition personnelle déterministe non sauvegardée, pas de lien professionnel requis |
| `POST /api/plan/preparations` | Enregistre le point personnel prévisualisé, avec références de source, révision et statut privé |
| `PATCH /api/plan/preparations/:id` | Édition propriétaire, sous révision ; jamais publication implicite |
| `DELETE /api/plan/preparations/:id` | Supprime point et copies générées ; invalide un accès partagé éventuel |
| `DELETE /api/plan/goals/:id` | Supprime étapes/essais liés ; invalidation des synthèses et révisions |
| `DELETE /api/plan` | Effacement de tout le plan, hausse de l'epoch, annulation des tâches en cours liées |

Réponse de conflit : `409 revision_conflict`, avec instruction de recharger et version courante autorisée ; aucune fusion automatique de textes sensibles. La fenêtre conserve la saisie locale pour que la personne puisse la reprendre. Erreur d'appartenance : `404`, sans révéler l'existence d'un objet d'un autre compte. Corps et sorties bornés ; pagination de toute collection et couverture d'export explicite.

**Verrou de publication** : l'instantané du tour comprend `plan_revision`, `plan_epoch`, `context_generation`, révisions des sources, `private_care_revision`, choix de partage pertinents et version du catalogue. Tout changement de préférences ou de consigne professionnelle fait avancer `plan_revision` et la révision spécifique concernée dans la même transaction ; une garde ne peut donc omettre une interdiction récente. La transaction de publication vérifie ces valeurs, la validité du bail et l'état du tour. Si l'une a changé, aucune écriture de proposition ni relance ne passe. Une reprise se fait sur une source courante, dans le budget existant ; pas de boucle de modèles indéfinie.

**Suppression pendant génération** : l'epoch est un marqueur monotone, distinct de la présence de lignes. Une purge ne le remet jamais à zéro ; un job ancien ne peut reconstruire le plan après suppression. L'effacement du compte invalide définitivement le propriétaire ; les cascades couvrent toutes les nouvelles tables. Une suppression de ligne fait aussi avancer la révision globale. Les caches/embeddings éventuels n'existent pas dans cette version ; s'ils sont ajoutés, ils devront participer à ce contrat.

**Idempotence** : un même clic, une reprise réseau ou un lot rejoué ne crée qu'un objet. Registre sans texte sensible : `(user_id, request_key, epoch, request_hash, entity_type, entity_id, state, created_at)`, unicité de `(user_id, request_key)` ; clé de publication dérivée de `(turn_id, proposal_slot)`. Le hachage porte sur le corps canonique, sans garder celui-ci dans le registre. Chaque mutation porte l'epoch courant et la révision attendue. Rejeu d'un ancien epoch → `409 stale_epoch`, aucune création.

Même clé/epoch et même requête : retour du pointeur et de la révision **actuels**, seulement si l'objet est encore autorisé. Même clé et contenu différent → `409 idempotency_conflict`. Objet supprimé → entrée réduite `deleted`, `410 resource_deleted`, aucune recréation ; ces marqueurs restent jusqu'à suppression du compte, sans payload, et sont inclus dans le cadrage de conservation. Une nouvelle intention après purge exige epoch courant **et nouvelle clé**. La durée de vie du registre n'est pas un motif de conservation de données supprimées.

Carte déjà acceptée : le même rejeu retourne le lien vers l'objet encore présent, sans réappliquer l'ancien texte ; une autre demande d'adoption répond `409 proposal_resolved`. Carte refusée/expirée/supprimée : aucune adoption possible ; une nouvelle proposition exige une nouvelle demande explicite. Suppression d'un objet adopté met aussi à jour l'état de résolution, sans laisser un pointeur qui le recrée. L'idempotence n'autorise jamais à restituer un contenu inaccessible après révocation ou suppression.

## 9. Export, suppression et mémoire dérivée

Export du compte : objectifs, étapes, essais, résultats, préférences, dates, sources encore disponibles, statut d'adoption, versions et métadonnées utiles ; pas de secrets ni de données d'autres comptes. Export complet paginé ou assemblé sous tâche bornée : ne pas appliquer silencieusement une limite fixe en laissant croire que tout a été exporté. Les liens source supprimés sont signalés comme indisponibles, sans texte privé retenu ailleurs.

**Choix courant pour les suppressions, à relire après le cycle 1** :

- Supprimer un essai/objectif retire son contenu, ses citations copiées et ses propositions dépendantes ; les synthèses/points qui le reprennent deviennent inaccessibles puis sont purgés ou régénérés. L'interface dit ce qui est retiré.
- « Effacer mon plan » supprime les objets, citations et consignes privées associés, incrémente l'epoch et retire les résumés dérivés. Les arrêts explicites de relance et indicateurs de refus/vigilance restent dans l'état de contrôle séparé ; aucun retour ni exploration n'est réactivé par cet effacement. Pas de reconstitution depuis l'historique existant sans nouveau choix explicite.
- « Effacer ma conversation » supprime conversation, mémoire et propositions non adoptées issues du chat. Les objectifs/essais **déjà explicitement adoptés** restent dans leur espace dédié, mais leurs citations conversationnelles sont retirées et leur provenance affiche « message source supprimé ». L'écran de suppression l'annonce clairement et propose aussi d'effacer le plan ; l'API expose un choix explicite `include_plan`. Ce comportement doit être validé en critique pour éviter toute attente trompée de suppression complète.
- « Supprimer mon compte » supprime toutes les tables, tâches, points et accès concernés selon le contrat du projet ; aucune option de conservation du plan.

**Mémoire existante sans provenance : invalidation conservatrice mais proportionnée.** `chat_memory.summary` contient aujourd'hui des listes de textes ; retirer sélectivement un fait n'est pas garanti. La coupure de reconstruction s'applique aux suppressions et aux corrections **de contenus déjà transmis aux modèles ou présents dans des artefacts dérivés**. Un objectif saisi puis corrigé avant tout envoi n'exige pas cette coupure. Changer priorité, date prévue, retour souhaité ou statut administratif n'efface pas la continuité : seule la révision augmente, avec invalidation des vues/snapshots devenus périmés. La dépendance est établie par les sources/transmissions, pas devinée par un modèle.

Lorsqu'une coupure est nécessaire, la mémoire textuelle entière, ses jobs et les synthèses/points sans provenance suffisante sont invalidés. Les copies à provenance fiable ne sont retirées que si elles dépendent du contenu concerné. Les textes supprimés ne sont pas gardés sous prétexte de vigilance. À leur place, l'état de contrôle peut garder seulement `reminders_paused`, `exploration_reconsent_required` et `past_safety_signal` : indicateurs booléens, sans texte ni étiquette clinique. Ils sont initialisés d'après les contrôles explicites déjà présents, jamais créés sur simple inférence lors d'une purge.

`exploration_reconsent_required` interdit les explorations personnelles spontanées tant que la personne n'a pas rouvert elle-même le sujet courant ou demandé explicitement cette exploration ; perdre le libellé d'un sujet fermé n'autorise pas à le rouvrir. `past_safety_signal` impose seulement prudence et vérification contextuelle si un signal actuel le justifie ; **il ne prouve aucun danger actuel et ne déclenche pas à lui seul une réponse de crise**. L'arrêt de relance reste levable uniquement par l'action explicite prévue au §6.

Une action distincte **Effacer toute ma mémoire de conversation**, accompagnée d'un aperçu de ce qui est effacé et conservé, permet à la personne de supprimer aussi ces indicateurs de mémoire/refus ; les préférences de notification/relance choisies séparément restent arrêtées par défaut tant qu'elle ne les réactive pas. La détection du danger du message courant continue à fonctionner. Cette action fait avancer génération/révisions et ne restaure pas les anciens tours dans les prompts.

Chaque nouveau tour reçoit à sa création `context_generation` du compte, valeur conservée dans `chat_turns` par migration ; les tours existants sont affectés à la génération initiale 0. Ce champ ne doit pas être réaffecté lors d'une reprise de job. La coupure de reconstruction décrite ci-dessus incrémente cette génération dans la même transaction, annule les baux et vide les instantanés anciens ; **historique envoyé aux modèles et futurs lots de mémoire sont limités aux tours de la génération courante**. Les anciens tours peuvent rester consultables par leur propriétaire, mais ne sont plus réinjectés automatiquement. Un ancien tour encore en cours garde sa génération d'origine et ne peut publier après la purge. Le test couvre exactement un ancien tour rejoué après l'effacement. Ce choix sacrifie une partie de continuité ; l'écran le dit plutôt que promettre une suppression fine inexistante.

Pour les copies créées par la nouvelle fonctionnalité, toute carte, réponse, trace brute, brouillon, bilan et point conserve des dépendances d'objets dans `change_artifact_refs` (propriétaire, type/id d'artefact, type/id d'objet, révision/génération). Les dépendances sont ajoutées à partir des **entrées réellement envoyées** au générateur, pas seulement de ce que le modèle affirme avoir utilisé ; la réponse d'origine de la proposition est incluse. Les cartes renvoient à l'état courant et ne conservent pas une deuxième copie de citation dans le JSON conversationnel.

Une suppression purge le contenu et les citations des artefacts dépendants ou invalide intégralement l'artefact si sa granularité ne permet pas mieux. Leur lecture et leur export les filtrent dès la transaction d'invalidation ; une purge matérielle différée, si nécessaire, ne laisse pas le contenu accessible. Les artefacts anciens susceptibles de reprendre le contenu corrigé/supprimé mais sans dépendances fiables sont invalidés conservativement, jamais annoncés comme nettoyés sélectivement. Une réponse de suppression expose `removed` et `retained` en langage clair : **les messages originaux de la personne, s'ils restent dans son historique, sont une source distincte**, avec lien pour effacer aussi la conversation. Aucune citation générée dans une carte, un bilan, un point ou un export ne reste accessible sous un message « supprimé ».

Une réintroduction ultérieure de l'information ne peut venir que d'un nouveau message ou d'une saisie explicite ; le modèle n'a aucun droit de restaurer depuis une génération antérieure. Les points personnels `memory_unverified` sont invalidés en entier lors d'une purge de mémoire.

## 10. Budget et performance

**Cible habituelle : environ 12 000 jetons d'entrée cumulés sur analyse + rédaction + relecture**, mesurés chez le fournisseur, sans compter les tokens cachés comme « gratuits » dans la trace. Ce n'est ni 12 000 par étape ni une égalité garantie pour chaque tokeniseur. Tolérance proposée pour le jeu habituel : borne haute 13 800 (+15 %), les cas moins coûteux étant naturellement acceptés ; moyenne et médiane sont publiées. Réécritures et traitements hors tour sont rapportés séparément. Le profil maximal doit également terminer toutes les étapes dans son enveloppe réservée, même s'il sort de cette cible habituelle. Aucune requête de modèle nouvelle par tour normal. Mémoire et point de séance restent des traitements séparés, déjà budgétés, déclenchés par besoin.

Enveloppe indicative de travail : analyse ~1 700, rédaction ~7 900, relecture ~2 400 = ~12 000 d'entrée. Le moteur corrigé a déjà mesuré, avec fournisseur simulé, environ 11 588 entrées sur profil rempli et 12 395 sur profil maximal (retour du relecteur du cycle 1) ; ces chiffres sont des estimations instrumentées, pas une recette fournisseur réelle. L'enveloppe ci-dessus devra être ajustée après mesure des prompts corrigés, **avant** activation du catalogue. Les descriptions des étapes ne doivent pas recopier tout le plan ou tout le profil.

Caps de départ proposés pour les ajouts : index de six stratégies ≤900 caractères ; deux détails ≤1 600 caractères au total ; plan courant ≤1 800 caractères ; suivi pertinent ≤500 caractères. Total maximum additionnel brut ≤4 800 caractères, réparti entre les étapes sans duplication systématique. Le constructeur calcule le coût estimé des blocs ajoutés **pour chaque étape** puis retire au moins la même réserve aux détails de sources de moindre priorité, en enlevant des items entiers. L'index des stratégies à l'analyse est lui aussi compensé dans son enveloppe, et la relecture ne reçoit pas une seconde copie du plan complet. Une marge de 15 % couvre l'écart caractères/tokeniseur avant pré-réservation. Si la compensation retire une information indispensable ou empêche la réserve de réécriture, aucun nouvel exercice n'est proposé dans ce tour ; un retour simple sur essai reste possible. Cet espace n'est donc pas ajouté à un prompt déjà plein. Conserver les derniers faits, la permission, l'état et les limites même si les explications de méthode sont retirées.

Ordre de priorité sous pression : sécurité et demande actuelle → limites/permissions → dernier essai pertinent et dates → cadre professionnel pertinent → détail de la seule stratégie retenue → autres éléments. Pas de troncature au milieu d'une citation ou d'une exclusion. Si le plan ne tient pas, produire une réponse d'écoute ou un retour simple, sans inventer que l'action a été enregistrée.

Réservation préalable : vérifier la capacité pour analyse, rédaction, relecture et la possibilité d'une réécriture nécessaire selon le contrat corrigé du moteur. Un profil maximal doit avoir une voie complète bornée ; relever un plafond seul ne constitue pas une optimisation. Le journal distingue budget du tour, budget hebdomadaire, fournisseur indisponible et conflit de source.

Mesures à joindre au rapport : profils léger/habituel/rempli/maximal, long message et plan chargé, zéro/deux stratégies, résultat d'essai et première/deuxième relance, réécriture, échec fournisseur. Entrées, sorties, cache, coût et durée réels ; simulation séparée. Un objectif de coût non atteint bloque l'activation générale, pas l'écoute de la personne.

## 11. Mon suivi et Paramètres

**Mon suivi** accueille une rubrique « Mon plan », avec objectifs actifs, prochain petit pas choisi et historique d'essais. Chaque carte affiche un état simple : proposé, choisi, en pause, retour renseigné, arrêté. Les résultats montrent leurs mots, date réelle si connue et suite choisie. Modification dans une fenêtre avec Enregistrer/Annuler et erreurs/conflits dans la fenêtre, cohérente avec le portrait existant.

Pas de compteur de performance, pourcentage de guérison, diagnostic, score de motivation ou badge de retard. Aucun bandeau permanent supplémentaire. Le menu reste accessible. Mobile : cartes en une colonne, contrôles clavier/tactile, aucun débordement à 320 px.

**Paramètres** explique :

- quand Ambre propose une piste, les familles disponibles et les limites ;
- « ce qu'Ambre propose » versus « ce que vous avez choisi », avec accès au contenu retenu ;
- les préférences de plan et de retours, désactivables sans perdre l'accès à l'historique ;
- les dernières stratégies considérées/retenues et la raison courte, uniquement données du compte ;
- les critères de relecture applicables et les cas sans exercice, les tailles/jetons et dates ;
- les sources documentaires avec niveau de lecture et caractère adapté, sans revendiquer un traitement validé ;
- export/suppression et périmètres de partage avec aperçu et révocation.

Les traces générales ne recopient pas les citations ou résultats privés. Une vue propriétaire de détail peut montrer les éléments retenus avec autorisation explicite de la route, sans les exposer au professionnel via le simple partage du plan. Les champs absents sur d'anciennes traces sont tolérés.

## 12. Applicabilité des critères et recette nécessaire

Chaque critère de changement est transmis avec un identifiant et une applicabilité calculée : C-R1/C-R3 dès qu'un contenu de changement est présent ; C-R2 pour une proposition ou transition ; C-R4/C-R6 pour un exercice développé ; C-R5 pour un essai rapporté ; C-R7 si une consigne professionnelle pertinente ou un partage est évoqué ; C-R8 uniquement si une invitation de retour est proposée. Le serveur détermine ces conditions depuis les objets et les sections de réponse ; le relecteur peut signaler un exercice implicite manqué, ce qui rend le critère applicable. Une réponse d'écoute sans changement n'a pas huit « oui » artificiels.

Verdict par critère : `pass|fail|not_applicable|not_checked`. `not_applicable` n'entre pas dans le dénominateur des critères évalués ; `not_checked` n'est jamais un succès. Si un critère bloquant applicable n'a pu être évalué, aucun exercice ne peut être publié. La trace/Paramètres distingue clairement ces quatre états.


| Domaine | Scénarios minimaux et résultat attendu |
|---|---|
| Sélection | Demande d'écoute → zéro stratégie ; demande d'agir → candidates pertinentes ; repère clinique seul → aucun exercice présumé ; ID inventé filtré ; pas de remplissage artificiel |
| Sécurité clinique | « Faire disparaître cette habitude » sans crise artificielle ; danger réel prioritaire ; « honte de l'essai non fait » accueillie sans reproche ; violence + demande explicite de phrase pour dire non → aucune mise en situation via C08 ; C09 reste préparation de séance même sous budget court ; restriction jamais tronquée |
| Arrêt | Bouton pendant génération ; commande exacte, propos rapporté, négation et phrase ambiguë ; jamais de réactivation par modèle |
| Adoption | Proposition ≠ objectif accepté ; « peut-être »/silence ≠ accord ; carte éditée accepte exactement le texte affiché ; date/relance/partage restent distincts |
| Résultat | « Fait mais pire », « pas eu le temps », date inconnue, deux essais possibles ; rien inféré ni écrasé ; répétition produit un nouvel essai |
| Relance | zéro sans opt-in ; une pertinente ; deuxième espacée ; jamais troisième ; modification de date ne remet pas le compteur ; refus immédiat ; pas de notification |
| Relecture | contexte plan/care présent ; critères avec IDs ; bon traitement des sans objet ; rejet ou panne ne publie pas un exercice non relu ; pas de formule « enregistré » sans écriture |
| Concurrence | Deux onglets éditent ; suppression pendant appel ; révocation pendant point ; bail expiré/reprise ; clé rejouée après correction/purge, clé avec autre contenu, carte résolue ; ancien tour et ancienne mémoire ne franchissent pas la nouvelle génération |
| Isolation | toutes routes sur objet d'un autre compte, IDs imbriqués, rôle professionnel en écriture, citations d'un périmètre privé ; aucune fuite |
| Vie privée | Export au-delà d'une page et à 256 Kio ; suppression/correction d'un contenu transmis coupe la reconstruction, modification de priorité/date/statut ne la coupe pas ; refus/arrêts préservés sans ancien texte ; vigilance passée sans crise automatique ; copies générées et exports nettoyés ; consigne privée absente de toutes vues professionnelles ; révocation bloque nouvelles lectures sans promesse sur fichier téléchargé |
| Coût | trois étapes normales proches de la cible ; profil maximal complet ; relecture et réécriture effectivement possibles ; erreurs distinctes et mesurées |
| Interface | 1440/390/320 px, clavier, lecteurs d'écran pour états et erreurs, édition longue, conflit non destructif, partage absent par défaut |

Les cas cliniques doivent faire l'objet d'une relecture indépendante sur réponses concrètes. Les tests de forme ne prouvent pas la sécurité ni l'utilité clinique. La recette réelle utilisera des comptes synthétiques supprimés après contrôle, avec preuve des migrations et de l'effacement si ce chantier est ensuite autorisé à être livré.

## 13. Relectures prévues avant toute implémentation

1. **Critique indépendante n°1** : sources/limites cliniques et architecture/consentement/coût. Examiner notamment C09, schémas, acceptation explicite, budget et suppression de conversation.
2. **Révision v2** : répondre point par point, conserver les désaccords non résolus et modifier le document, sans passer directement au code.
3. **Critique indépendante n°2** : reprendre la version révisée avec scénarios adverses concrets ; vérifier que les réponses aux premières critiques sont effectives.
4. **Révision v3** : stabiliser les contrats/API et critères de recette. Nouvelle critique si un défaut important persiste. L'absence de commentaire n'est pas une validation.
5. Seulement après cette boucle : catalogue machine, backend/persistance et moteur, puis interface/partage, puis vérifications et revue finale selon les règles du dépôt. Les changements demandés après critique sont intégrés avant publication.

## 14. Décisions encore ouvertes et historique

| Sujet | Proposition courante | Condition de stabilisation |
|---|---|---|
| Exposition graduée | Restreinte à préparation de séance par défaut | Relecture clinique des critères d'activation |
| Adoption conversationnelle | Carte explicite pour toute adoption/résultat | Tester compréhension et fluidité, sans consentement inféré |
| Nombre d'objectifs/espacement | Trois actifs et sept jours, plafonds produit | Critique UX et coût ; aucun fondement clinique revendiqué |
| Suppression du chat | Plan adopté séparé, citations retirées, option explicite de tout effacer | Critique confidentialité et attente utilisateur |
| Consigne professionnelle | Rapportée par la personne, privée par défaut | Contrats de contexte/partage et provenance |
| Budget | Répartition indicative 1 700/7 900/2 400 | Mesures du moteur corrigé, avec réécriture |

- **v1** : recherche et proposition structurée ; aucun catalogue machine ni code implémenté.
- **v2** : réponses au [cycle 1 de critique indépendante](relecture-plan-changement-2026-11-17.md), sans modification de ce rapport ; aucun code nouveau.
- **v3** : cinq ajustements du cycle 2 intégrés ; agrégat JSON CAS retenu, limites de purge et de révocation précisées. Validation finale du plan consignée dans le rapport indépendant ; implémentation autorisée.

### Réponses au cycle 1

| Point | Réponse v2 | Section |
|---|---|---|
| 1 Arrêt immédiat | Route/boutons sans modèle, commandes entières strictes, réactivation uniquement explicite | §6, §8 |
| 2 Mémoire sans provenance | Invalidation globale, génération de contexte et coupure de reconstruction, anciens tours exclus des modèles | §9 |
| 3 Idempotence | Clé globale par compte + epoch + hash, marqueur deleted, cartes résolues et préférences/consignes dans révision | §8 |
| 4 Relance réelle | Citation exacte de l'invitation finale, seule cible admissible, relecture sémantique et débit transactionnel | §6 |
| 5 Dates | Europe/Paris fixe, jour local sans heure, prochain message post-adoption, 604 800 secondes, date inconnue explicite | §4, §6 |
| 6 Budget | Compensation de chaque bloc par retrait de détail, marge et tolérance mesurées, simulation distinguée du réel | §10 |
| 7 Point personnel | Parcours déterministe, sections mémoire privées à vérifier, édition/enregistrement/suppression sans pipeline IA | §7, §8 |
| 8 Accès | Chargeur privé séparé de care partagé ; sérialiseurs explicites ; scope plan jamais ajouté aux liens anciens | §2, §7 |
| 9 Stratégies/revue | IDs identiques au JSON, SCH distinct des sources, restriction machine/référence, sans objet distinct du succès | §3, §12 et dossier recherche |
| 10 Corrections | Routes étapes/essais/propositions complètes, résultat sans date, dépendances/purge des copies et export | §8, §9 |

Les réponses v2 ont été examinées au cycle 2. Les ajustements supplémentaires sont tracés ci-dessous ; l'auteur ne déclare pas lui-même la validation indépendante.

### Réponses au cycle 2

| Point | Réponse v3 | Section |
|---|---|---|
| 1 Purge proportionnée/refus | Coupure seulement pour suppression/correction de contenu déjà transmis ; mutations administratives sans remise à zéro ; booléens de refus/vigilance sans texte et action d'effacement complet | §4, §9, §12 |
| 2 Révocation réaliste | Blocage des nouvelles lectures et des copies contrôlées par l'application ; aucune promesse de retrait d'un fichier déjà téléchargé | §7, §12 |
| 3 Rôles/défauts | Routes privées membre propriétaire ; consentement conversation pour propositions, désactivation sans effacer les choix ; retours off, scope plan absent, consignes séparées | §1, §7, §8 |
| 4 Restrictions compactes | Champs critiques systématiques, omission entière si trop longue, cas violence + C08 et C09 restreinte | §3, §10, §12 |
| 5 Stockage physique | Agrégat JSON versionné CAS, IDs/révisions par objet, quotas explicites, export complet ; garde de publication et tests inchangés | §4, §8, §12 |

**État v3 : plan retenu pour implémentation par la vérification finale indépendante.** Le coût réel, les purges effectives et la recette des scénarios restent à démontrer par l'implémentation ; aucune nouvelle preuve clinique n'est revendiquée. C09 demeure `prepare_session_only`.


## 15. Livraison documentaire et catalogue après feu vert v3

Le [rapport indépendant](relecture-plan-changement-2026-11-17.md) a retenu v3 pour implémentation après les deux cycles. Dans le périmètre confié à l'auteur du dossier, seuls les livrables suivants ont été ajoutés :

- `data/change_strategies.json` : version de données 1, 36 stratégies avec identifiants stables, 18 sources et accès réellement obtenus, règles communes intégrales, statut documentaire explicite, restrictions machine et références de recherche par ancre.
- `scripts/build_change_strategies.mjs` : validation déterministe et génération ; `--check` refuse un module périmé. Les restrictions C09, les contextes bloqués et les préconditions interpersonnelles requises ne sont pas optionnels. Les niveaux de lecture de cette version restent ceux effectivement documentés.
- `journal/src/change-strategies.generated.mjs` : export `CHANGE_STRATEGIES = {version, common_rules, sources, strategies}`. Aucune sélection de stratégie, route de plan ou mutation moteur n'a été écrite par ce sous-chantier.

Vérifications exécutées : génération et `--check` réussis ; syntaxe Node du générateur et du module valide ; import généré identique au JSON source ; 12 mutations invalides rejetées (dont C09 autonome, C08 sans contexte sûr, retrait de `no_advice`, source inconnue, faux niveau intégral, date invalide, doublon et restriction compacte amputée). Une copie temporaire du module a été volontairement modifiée : `--check` a bien refusé la dérive, sans toucher au module du dépôt. Les 36 ancres de recherche sont présentes.

Ces vérifications garantissent la cohérence du catalogue, pas une validation clinique ni la recette du moteur. Restent aux chantiers d'intégration : sélection/contextes, persistance/CAS, relances, partage/préparation, interface, mesures réelles et tests de bout en bout. Aucun commit individuel supplémentaire ni déploiement effectué par ce sous-chantier.
