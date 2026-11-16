# Plan de changement d'Ambre — 17 novembre 2026

**Version 1 — proposition à critiquer et réviser avant implémentation.** Ce document ne constitue ni une livraison ni une validation clinique. Le catalogue machine, les migrations et l'interface ne sont pas encore écrits par ce chantier.

Recherche associée : [36 stratégies et sources réellement consultées](recherche-strategies-changement-2026-11-17.md). Fondations : [audit moteur, §13–14](audit-ambre-2026-11-17.md), [pratique clinique, §6–8](recherche-pratique-clinique-2026-11-16.md), [reprise](reprise-2026-11-16.md).

## 1. Résultat attendu et périmètre

Ambre doit pouvoir aider une personne qui le souhaite à choisir un petit changement, garder une trace fidèle de ce qu'elle accepte d'essayer, revenir doucement sur les résultats et ajuster la suite. Elle continue à écouter lorsqu'un plan ne convient pas. Les méthodes disponibles sont choisies en fonction de la demande et du contexte, sans attribuer de diagnostic ni transformer un repère de cartographie en prescription.

L'outil reste utilisable sans psychologue. Avec un professionnel, il prépare et prolonge le travail déclaré par la personne ; il ne se substitue pas à son jugement. Le professionnel conserve un accès en lecture seule aux seuls périmètres partagés volontairement.

**Demandes déjà établies dans la reprise** : catalogue sélectionné par tour ; objectifs/étapes/essais/résultats persistants ; retour sans harcèlement ; articulation avec le suivi ; critères de relecture ; transparence Paramètres ; critiques et révisions multiples avant implémentation. **Contraintes de ce chantier** : pas de nouveau modèle à chaque tour, environ 12 000 jetons d'entrée cumulés pour un tour habituel, au plus deux relances dans la conversation et aucune notification externe.

**Choix proposés, encore discutables** : mode d'adoption par carte explicite, plafonds de contexte, statuts détaillés, espacement de sept jours, limites d'objectifs actifs et politique de suppression ci-dessous. Ces choix ne sont ni des normes cliniques ni des décisions déjà prises par la personne qui utilisera Ambre.

Hors périmètre : thérapie autonome, traitement de trauma, EMDR/rescriptage, changements médicamenteux, sevrage, exercices physiologiques intenses, série d'expositions spécialisées, notification push/courriel/SMS, écriture du professionnel dans le dossier, diagnostic par questionnaire, gamification de l'observance.

## 2. Préconditions moteur et point de coordination

Les corrections de l'audit sont un préalable à l'ajout de contexte : réserver une réponse et sa relecture sur profil rempli/maximal ; respecter les limites de budget sans faux message hebdomadaire ; dates lisibles ; historique gardant la question finale ; mémoire récente ; règles de sécurité tenant compte du contexte ; cadre du suivi transmis au relecteur ; distinction faits/hypothèses ; suppression des faux positifs connus. Le chantier moteur peut les corriger indépendamment du présent plan.

**Contrat proposé au chantier moteur** : prévoir des extensions optionnelles et bornées, absentes par défaut ; ne pas activer le catalogue ni créer des engagements tant que le plan n'a pas été révisé. Le numéro `0021` cité dans l'audit n'est pas réservé : choisir la prochaine migration libre au moment d'intégrer.

```text
loadChangeContext(env, userId, now)
  -> {available, revision, epoch, preferences, goals[], active_trial?, due_trial?, context_hash}
selectChangeCandidates({message, plan, allowed_topics, care})
  -> [{id, label, intent_tags, restriction}]  // 0..6, pas de détail biographique
loadStrategyDetails(ids)
  -> 0..2 fiches validées, au plus une action développée dans la réponse
validateChangeDraft({analysis, writing, snapshot, message})
  -> propositions vérifiées / opérations de relance / refus explicites
```

Les fonctions ci-dessus sont des **interfaces proposées**. `loadCareContext(env,userId,now)` garde sa signature existante ; des champs optionnels peuvent être ajoutés en accord avec les consommateurs. Une absence de migration ou de catalogue donne `available:false` et désactive le changement, sans faire croire à un plan vide que l'on pourrait recréer.

## 3. Catalogue et sélection sans appel supplémentaire

Après validation documentaire, `data/change_strategies.json` sera la source unique, compilée vers un module Worker séparé du catalogue de psychologie. Chaque fiche contient : `id`, `version`, famille, libellé simple, intentions compatibles, indications, exclusions, type de permission, un petit pas, mesures facultatives, seuil d'orientation en mots, sources avec type/lecture/passage, et statut `draft|reviewed|restricted|disabled`.

Un lien vers un repère clinique est facultatif, sourcé et ne signifie jamais « ce repère prescrit cette technique ». L'exposition C09 commence `restricted` : préparation de séance, jusqu'à relecture clinique spécifique. Les techniques plus intensives exclues du dossier ne sont pas sérialisées comme options cachées.

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
  "strategy_ids": ["cbt_activity_pleasure"],
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

Schéma logique proposé (noms à stabiliser à l'intégration) :

```text
change_state(user_id, revision, epoch, preferences_json, updated_at)
change_proposals(id, user_id, kind, target_id?, payload_json,
  status, source_turn_id?, source_quote?, created_at, resolved_at?, revision)
change_goals(id, user_id, title, why?, priority, status,
  origin, accepted_at, created_at, updated_at, revision)
change_steps(id, user_id, goal_id, text, strategy_id?, strategy_version?,
  measure_json?, professional_context?, status, accepted_at, revision, dates)
change_trials(id, user_id, step_id, step_snapshot_json, status,
  planned_for?, occurred_at?, result_text?, outcome?, adverse_effect_text?,
  followup_opt_in, review_on?, reminder_count, last_reminder_at?,
  created_at, updated_at, revision)
change_events(id, user_id, entity_type, entity_id, operation,
  from_revision, to_revision, actor, created_at)
```

Clés étrangères sur le propriétaire et la hiérarchie ; contrôles explicites d'appartenance sur toutes les routes, y compris identifiants imbriqués. Pas de texte supprimé conservé dans `change_events` ; journal d'opérations réduit à métadonnées. La durée de conservation des métadonnées reste à cadrer, sans prétendre que le droit impose ici une durée particulière.

États proposés :

- Proposition : `proposed|accepted|dismissed|expired`. Pas de relance d'une proposition non adoptée ; expiration technique ne signifie pas refus de la personne.
- Objectif : `active|paused|completed|abandoned`. Seule la personne conclut qu'il est atteint. Pas de fermeture automatique par score.
- Étape : `active|paused|completed|abandoned`, uniquement après adoption ; les brouillons restent dans les propositions.
- Essai : `planned|reported|cancelled`. `reported` contient `attempt:done|partial|not_tried|stopped|uncertain` et `outcome:helpful|neutral|worse|uncertain|null`. Ainsi « non essayé » ne clôt pas automatiquement l'objectif et « fait » ne signifie pas « utile ».

Dates : stockage UTC ; affichage dans le fuseau déclaré par la personne. `planned_for` peut être une date locale sans heure, conservée avec le fuseau ; ne pas transformer « cette semaine » en date certaine sans le montrer. `occurred_at` n'est rempli que si rapporté, sinon `null`. `created_at` et `updated_at` sont des dates techniques, séparées de la date du fait.

Plafonds de départ proposés : trois objectifs actifs, une étape mise en avant par objectif, un seul essai à discuter dans un tour. Les autres objets restent accessibles et paginés ; si une limite d'activation est atteinte, l'interface l'explique et propose de choisir une priorité, sans effacer le travail. Ces nombres sont à valider à l'usage.

## 5. Adoption, corrections et révisions

Une carte présente « Ce qu'Ambre propose » avec **Choisir cet essai**, **Modifier**, **Pas maintenant**. L'adoption enregistre le contenu réellement affiché et accepté, la révision et l'heure. L'utilisateur peut aussi créer un objectif ou saisir directement son résultat depuis Mon suivi, sans modèle.

Dans la conversation, « je vais le faire » peut préparer une carte ciblée ; il ne justifie pas de déduire l'accord à une date, mesure chiffrée, relance ou partage. Un retour « j'ai essayé » prépare une proposition de résultat reliée au bon essai ; si le lien est ambigu, la carte demande de choisir l'essai. Les acceptations conversationnelles automatiques sont laissées hors v1 : le bénéfice de fluidité devra être comparé au risque d'engagement inventé après tests.

Correction d'un résultat : la valeur corrigée devient la source courante, `revision` augmente, synthèses préparées invalidées. Les métadonnées peuvent indiquer « corrigé le… », sans retenir indéfiniment l'ancien texte. Une suppression retire l'objet des prompts, recherches, exports futurs et comptes rendus accessibles après invalidation. Un objet abandonné reste visible dans l'historique tant qu'il n'est pas supprimé.

## 6. Relances douces, dans une conversation ouverte par la personne

La personne choisit séparément « Ambre peut revenir sur cet essai ». Par défaut, aucun retour planifié sur un essai nouvellement adopté. Elle peut choisir une date ou « lors d'un prochain échange » ; l'absence de date n'entraîne jamais un rendez-vous inventé.

Éligibilité calculée côté serveur :

1. Essai `planned`, objectif/étape actifs, retour autorisé, date atteinte si renseignée.
2. Aucune suspension/refus de sujet, de conseil ou de suivi, aucune voie sécurité/stabilisation/réparation/écoute seule.
3. Le sujet de l'essai est pertinent dans la conversation courante. Une échéance n'est pas une raison de détourner une demande urgente ou indépendante.
4. Moins de deux relances **sur toute la durée de cet essai**, et espacement proposé d'au moins sept jours entre relances ; pas deux relances sur deux tours consécutifs même si le calendrier le permet.
5. Un seul essai admissible retenu, par date puis priorité ; pas de file de rappels qui finit par être vidée en rafale.

Une relance est une invitation facultative, pas un reproche ni la présomption que l'exercice aurait dû être fait. Après deux invitations sans retour, aucune nouvelle relance de cet essai. L'interface peut afficher « Ambre n'y reviendra plus spontanément » ; un nouveau cycle nécessite un choix explicite de la personne, sans remise à zéro silencieuse du compteur par édition de date.

Compteur incrémenté **dans la même transaction que la publication de la réponse qui contient réellement la relance**. Aucun compteur consommé par échec fournisseur, relecture rejetée, réponse annulée, reprise de tâche ou conflit de révision. Un identifiant de publication unique interdit un double comptage. Une réécriture sans la relance ne la compte pas.

« Je n'ai pas eu le temps », « j'ai honte de ne pas l'avoir fait », silence et abandon sont distingués. Un résultat rapporté retire l'essai de la file ; l'arrêt demandé désactive le retour immédiatement et n'attend pas un prochain appel modèle. Aucune tâche planifiée ne contacte la personne hors conversation.

## 7. Articulation avec le psychologue et préparation de séance

Dans Mon suivi : champ privé « Ce que mon professionnel m'a proposé de travailler », daté et modifiable, avec mention **rapporté par vous**. Une instruction peut être liée à une étape ; la personne peut préciser qu'elle est actuelle, à vérifier ou terminée. Aucun titre, diagnostic ou intention du professionnel n'est inventé. Aucune validation professionnelle implicite à partir de la seule présence d'un compte lié.

Le moteur reçoit un extrait seulement s'il concerne l'essai en cours. En cas de contradiction ou d'incertitude, Ambre peut préparer la question de séance ; elle ne recommande pas de contredire une consigne de soin. Une consigne rapportée dangereuse ne neutralise pas la voie sécurité.

**Point personnel avant séance**, disponible sans compte professionnel lié et sans partage : objectifs choisis, essais depuis la dernière séance, résultats et obstacles dans les mots de la personne, effets pénibles, une question qu'elle souhaite apporter. L'utilisateur peut prévisualiser, corriger et retirer des éléments. Le point n'attribue pas « amélioration » ou « aggravation clinique » à une mesure courte.

**Partage opt-in** : nouveau périmètre `plan`, absent des liens existants et non activé par migration. Le choix montre les rubriques exposées et un aperçu. Le partage du plan n'accorde pas celui de la conversation ou du champ privé de consigne professionnelle. La lecture partagée du plan omet citations brutes et identifiants de sources non partagées ; les formulations d'objectifs/essais choisies sont les données partagées, ce que l'aperçu doit rendre clair.

Un point destiné au professionnel n'utilise que les périmètres actuellement autorisés. Une mémoire globale construite à partir de sources privées ne doit pas être injectée au motif qu'elle « résume » des données : soit chaque item a une provenance autorisée, soit elle est exclue du point partagé. Le point personnel peut utiliser la mémoire du compte dans ses limites habituelles, et le résumé partagé est reconstruit depuis les sources autorisées.

Révocation ou changement de périmètre invalide les comptes rendus concernés, même prêts et déjà ouverts dans une autre requête ; contrôle d'autorisation avant lecture et avant publication. Le serveur ne peut reprendre une donnée depuis le cache d'un point révoqué. Le professionnel reste en lecture seule. Une personne ne reçoit aucune notification externe de ce chantier.

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
| `POST /api/plan/steps/:id/trials` | Nouvel essai d'une étape acceptée ; instantané minimal |
| `PATCH /api/plan/trials/:id` | Résultat/date/retour souhaité ; jamais transition de propriété |
| `DELETE /api/plan/goals/:id` | Supprime étapes/essais liés ; invalidation des synthèses et révisions |
| `DELETE /api/plan` | Effacement de tout le plan, hausse de l'epoch, annulation des tâches en cours liées |

Réponse de conflit : `409 revision_conflict`, avec instruction de recharger et version courante autorisée ; aucune fusion automatique de textes sensibles. La fenêtre conserve la saisie locale pour que la personne puisse la reprendre. Erreur d'appartenance : `404`, sans révéler l'existence d'un objet d'un autre compte. Corps et sorties bornés ; pagination de toute collection et couverture d'export explicite.

**Verrou de publication** : l'instantané du tour comprend `plan_revision`, `plan_epoch`, révisions des sources, choix de partage pertinents et version du catalogue. La transaction de publication vérifie ces valeurs, la validité du bail et l'état du tour. Si l'une a changé, aucune écriture de proposition ni relance ne passe. Une reprise se fait sur une source courante, dans le budget existant ; pas de boucle de modèles indéfinie.

**Suppression pendant génération** : l'epoch est un marqueur monotone, distinct de la présence de lignes. Une purge ne le remet jamais à zéro ; un job ancien ne peut reconstruire le plan après suppression. L'effacement du compte invalide définitivement le propriétaire ; les cascades couvrent toutes les nouvelles tables. Une suppression de ligne fait aussi avancer la révision globale. Les caches/embeddings éventuels n'existent pas dans v1 ; s'ils sont ajoutés, ils devront participer à ce contrat.

**Idempotence** : un même clic, une reprise réseau ou un lot rejoué ne crée qu'un objet. Clé unique portée par la requête utilisateur ou par `(turn_id, proposal_slot)` pour une proposition ; résolution d'une proposition effectuée une seule fois. L'idempotence n'autorise pas à réutiliser une réponse devenue inaccessible après révocation ou suppression.

## 9. Export, suppression et mémoire dérivée

Export du compte : objectifs, étapes, essais, résultats, préférences, dates, sources encore disponibles, statut d'adoption, versions et métadonnées utiles ; pas de secrets ni de données d'autres comptes. Export complet paginé ou assemblé sous tâche bornée : ne pas appliquer silencieusement une limite fixe en laissant croire que tout a été exporté. Les liens source supprimés sont signalés comme indisponibles, sans texte privé retenu ailleurs.

**Choix proposé v1 pour les suppressions, à critiquer** :

- Supprimer un essai/objectif retire son contenu, ses citations copiées et ses propositions dépendantes ; les synthèses/points qui le reprennent deviennent inaccessibles puis sont purgés ou régénérés. L'interface dit ce qui est retiré.
- « Effacer mon plan » supprime tous les objets et préférences de suivi associées, incrémente l'epoch et retire les résumés de plan de la mémoire. Pas de reconstitution depuis l'historique existant sans nouveau choix explicite.
- « Effacer ma conversation » supprime conversation, mémoire et propositions non adoptées issues du chat. Les objectifs/essais **déjà explicitement adoptés** restent dans leur espace dédié, mais leurs citations conversationnelles sont retirées et leur provenance affiche « message source supprimé ». L'écran de suppression l'annonce clairement et propose aussi d'effacer le plan ; l'API expose un choix explicite `include_plan`. Ce comportement doit être validé en critique pour éviter toute attente trompée de suppression complète.
- « Supprimer mon compte » supprime toutes les tables, tâches, points et accès concernés selon le contrat du projet ; aucune option de conservation du plan.

La mémoire de conversation n'est pas une copie indépendante du plan : elle référence les identifiants encore valides ou reçoit le contexte structuré courant. Les prochains lots de résumé ne doivent pas ressusciter un essai supprimé en relisant de vieux tours. Un marqueur d'effacement/d'exclusion doit être pris en compte par le résumeur sans conserver le contenu supprimé.

## 10. Budget et performance

**Cible habituelle : environ 12 000 jetons d'entrée cumulés sur analyse + rédaction + relecture**, mesurés chez le fournisseur, sans compter les tokens cachés comme « gratuits » dans la trace. Ce n'est ni 12 000 par étape ni une promesse pour un profil maximal. Aucune requête de modèle nouvelle par tour normal. Mémoire et point de séance restent des traitements séparés, déjà budgétés, déclenchés par besoin.

Enveloppe indicative de travail : analyse ~1 700, rédaction ~7 900, relecture ~2 400 = ~12 000 d'entrée. Ces nombres devront être ajustés après mesure des prompts corrigés, **avant** activation du catalogue. Les descriptions des étapes ne doivent pas recopier tout le plan ou tout le profil.

Caps de départ proposés pour les ajouts : index de six stratégies ≤900 caractères ; deux détails ≤1 600 caractères au total ; plan courant ≤1 800 caractères ; suivi pertinent ≤500 caractères. Total maximum additionnel brut ≤4 800 caractères, réparti entre les étapes sans duplication systématique. Cet espace doit remplacer du contexte moins pertinent ; il n'est pas ajouté à un prompt déjà plein. Conserver les derniers faits, la permission, l'état et les limites même si les explications de méthode sont retirées.

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

## 12. Critères de recette et scénarios nécessaires

| Domaine | Scénarios minimaux et résultat attendu |
|---|---|
| Sélection | Demande d'écoute → zéro stratégie ; demande d'agir → candidates pertinentes ; repère clinique seul → aucun exercice présumé ; ID inventé filtré ; pas de remplissage artificiel |
| Sécurité clinique | « Faire disparaître cette habitude » sans crise artificielle ; danger réel prioritaire ; « honte de l'essai non fait » accueillie sans reproche ; violence → aucune exposition/assertivité à risque ; technique traumatique refusée |
| Adoption | Proposition ≠ objectif accepté ; « peut-être »/silence ≠ accord ; carte éditée accepte exactement le texte affiché ; date/relance/partage restent distincts |
| Résultat | « Fait mais pire », « pas eu le temps », date inconnue, deux essais possibles ; rien inféré ni écrasé ; répétition produit un nouvel essai |
| Relance | zéro sans opt-in ; une pertinente ; deuxième espacée ; jamais troisième ; modification de date ne remet pas le compteur ; refus immédiat ; pas de notification |
| Relecture | contexte plan/care présent ; critères avec IDs ; bon traitement des sans objet ; rejet ou panne ne publie pas un exercice non relu ; pas de formule « enregistré » sans écriture |
| Concurrence | deux onglets éditent ; suppression pendant appel ; révocation pendant point ; bail expiré/reprise ; idempotence et refus de la résurrection |
| Isolation | toutes routes sur objet d'un autre compte, IDs imbriqués, rôle professionnel en écriture, citations d'un périmètre privé ; aucune fuite |
| Vie privée | export au-delà d'une page, effacement objet/plan/chat/compte, invalidation de mémoire et points, aucune donnée supprimée dans métadonnées/trace/copied quote |
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

| Sujet | Proposition v1 | Condition de stabilisation |
|---|---|---|
| Exposition graduée | Restreinte à préparation de séance par défaut | Relecture clinique des critères d'activation |
| Adoption conversationnelle | Carte explicite pour toute adoption/résultat en v1 | Tester compréhension et fluidité, sans consentement inféré |
| Nombre d'objectifs/espacement | Trois actifs et sept jours, plafonds produit | Critique UX et coût ; aucun fondement clinique revendiqué |
| Suppression du chat | Plan adopté séparé, citations retirées, option explicite de tout effacer | Critique confidentialité et attente utilisateur |
| Consigne professionnelle | Rapportée par la personne, privée par défaut | Contrats de contexte/partage et provenance |
| Budget | Répartition indicative 1 700/7 900/2 400 | Mesures du moteur corrigé, avec réécriture |

- **v1** : recherche et proposition structurée ; aucun catalogue machine ni code implémenté. Critiques indépendantes et révisions à venir.
