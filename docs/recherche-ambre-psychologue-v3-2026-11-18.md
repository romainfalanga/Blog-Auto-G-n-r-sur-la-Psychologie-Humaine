# Ce qui manque encore à Ambre — recherche comparative v3 (18 novembre 2026)

Recherche documentaire et analyse comparative, **sans modification du code**. Complète, sans les refaire, `recherche-pratique-clinique-2026-11-16.md` (dite « R1 ») et `recherche-strategies-changement-2026-11-17.md` (dite « R2 »). Rédigé au fil de l’eau : une section terminée est une section sauvegardée.

## 0. Méthode

- **Niveaux de vérification** (mêmes conventions que R1) : **[V]** texte ouvert et lu pendant cette recherche ; **[R]** résumé officiel ou page de présentation lus, texte intégral non lu ; **[S]** source secondaire, extrait de moteur de recherche ou connaissance générale non relue — à recontrôler avant toute citation publique.
- **Comparaison au code** : chaque apport est confronté à ce qu’Ambre fait réellement, en citant `fichier:fonction` ou la constante concernée. Verdicts : **présent**, **partiel**, **absent**.
- **Propositions** : consigne (texte court à insérer), stratégie au format de `data/change_strategies.json`, critère de relecture vérifiable, signal pour l’analyse du tour, élément de mémoire ou de formulation. Les seuils chiffrés proposés sont des choix de conception, pas des normes publiées.
- **Hors périmètre** : tout ce que R1 et R2 traitent déjà (alliance générale, entretien motivationnel, trauma et fenêtre de tolérance, risque suicidaire de base, 3114, Stanley-Brown, erreurs d’Iftikhar, Moore et al., MITI/CTS-R/EPITOME, 36 stratégies TCC/TCD/ACT/schémas/compassion/EM/objectifs). Ils ne sont rappelés que lorsqu’un écart d’implémentation subsiste.
- **Budget** : les correctifs du 17 novembre placent un tour nominal à ~11 700 jetons d’entrée et ~29 000 réservés sous un plafond de 30 000 (`ambre-engine.mjs:TURN_BUDGET`). Toute proposition est donc chiffrée en coût de jetons ; les ajouts constants au prompt de rédaction se paient à chaque tour et à chaque réécriture.

## 1. Ce qu’Ambre fait réellement aujourd’hui (base de comparaison)

Lecture du code au commit `dd23271`.

### 1.1 Chaîne d’un tour (`journal/src/ambre-engine.mjs:work`)

1. **Analyse** : `deterministicAnalysis` (expressions régulières : détresse, rupture, honte, dissociation, ambivalence, dépendance, écoute seule, refus de questions, sujet fermé) puis appel `analysis` avec `ANALYSIS_SYSTEM` et `ANALYSIS_SCHEMA` : `need` ∈ {listen, understand, advice, act}, `states` ∈ `STATES` = {distress, rupture, shame, ambivalence, dissociation, dependency, unusual_belief}, `risk` ∈ {none, possible, explicit}, `posture` ∈ {listen, explore, stabilize, repair, safety, options}, `constraints` ∈ {no_advice, no_questions, no_exploration, respect_closed_topics}, `selected_ids`, `closed_topics`, champs de changement. Le déterministe prime toujours sur le modèle.
2. **Vérification du risque** (`RISK_CHECK_SYSTEM`) seulement si le modèle propose un risque que les règles n’ont pas vu ; catégories {self_harm, harm_to_others, violence, indirect_warning, current_confirmation} ; citation exacte exigée (`checkedRisk`).
3. **Rédaction** : prompt interne à `work` (« Tu es Ambre, assistante IA, jamais une professionnelle… 80 à 180 mots : reflet précis puis au plus une question… ») + `engineContract` (`chat-backend.mjs`) + `CHANGE_SYSTEM_RULES`/`CHANGE_DRAFT_RULES` si un changement est actif. Voie sécurité distincte : `SAFETY_WRITING_SYSTEM`, réponse de secours `chat-safety.mjs:safetyReply`.
4. **Relecture** : `REVIEW_CRITERIA` (15 critères oui/non/sans objet) + `CHANGE_REVIEW_CRITERIA` C-R1…C-R8 (`change-strategies.mjs`) + contrôles déterministes `deterministicReview` (questions, diagnostic lexical, promesse d’enregistrement, numéros inconnus, absence de numéro en cas de risque, conseil malgré `no_advice`). Une réécriture au plus, puis `FALLBACK` prudent.
5. **Publication** puis **mémoire** (`updateMemory`) : tous les 4 à 8 tours, un appel range des **citations exactes de la personne** dans huit rubriques `MEMORY_SECTIONS` = goals, facts, hypotheses, tried, alliance, safety, closed, session ; 12 éléments par rubrique (30 pour `closed`).

### 1.2 Ce que le prompt du moteur ne contient pas

Le prompt riche `psychology-framework.mjs:buildCompanionSystemPrompt` (explications alternatives, situation→interprétation→émotion→action, exceptions, réciprocité, emprise sans confrontation symétrique, croyances inhabituelles) et `JOURNEY_GUIDANCE` sont utilisés par l’**ancienne voie** (`chat-backend.mjs`, ligne ~396). Le moteur actif (`createAmbreEngine`, ligne ~797) reçoit un prompt de rédaction plus court écrit dans `work` et le contrat `engineContract`. Vérifié : `worker.mjs` crée le service avec `engineEnabled:true`, et `send` appelle alors `engine.work` au lieu de `generate` ; le système de rédaction du moteur est le texte écrit dans `work` suivi de `engineContract` (champs JSON, propositions, notes), sans appel à `buildCompanionSystemPrompt`. Plusieurs consignes de fond de l’ancienne voie (explications alternatives, vu/ressenti/imaginé pour un proche, exceptions et ressources, emprise sans confrontation symétrique, croyances de persécution, dates approximatives) ne sont donc **pas** transmises au rédacteur actif. Les bilans, eux, utilisent toujours `buildReviewSystemPrompt`. Cette observation conditionne plusieurs comparaisons ci-dessous (« présent dans l’ancienne voie seulement »).

### 1.3 Cartographie, catalogue et plan

- Index de connaissances (`knowledgeIndex`) : proches, ligne de vie, sources déclarées, portrait, cartographie de l’enfance (causes, conséquences, preuves, ressources, notes), hypothèses du portrait, quatre derniers bilans. Sélection lexicale (`rankIndex`, `analysisIndex`, `selectDetails`) puis choix par le modèle d’analyse.
- Catalogue de 60 repères (`data/psychology_catalog.json`) : pensée, émotions, soi, relations, famille, attachement, action, contexte, besoins, ressources… dont `grief_change`, `identity_transition`, `overwork`, `social_belonging`, `values_alignment`.
- 36 stratégies de changement (`data/change_strategies.json`) en 7 familles : TCC (C01–C09), TCD (D01–D06), ACT (A01–A05), schémas (SCH01–SCH04), compassion (K01–K03), entretien motivationnel (M01–M04), objectifs et suivi (G01–G05).
- Bilans écrits (`buildReviewSystemPrompt`) : analyse initiale et points d’étape, 600 à 900 mots.

### 1.4 Repérages bruts dans le code (grep du 18 novembre)

| Thème | Présence dans les prompts/consignes du moteur actif |
|---|---|
| 3919 (violences conjugales), 119 (enfance en danger) | Seulement dans la liste blanche de numéros de `deterministicReview` ; jamais proposés dans une consigne |
| Mineurs, âge minimal | Aucune règle conversationnelle (le prompt ancien parle seulement de l’âge comme donnée d’adresse) |
| Troubles alimentaires | Seulement comme exclusion de restriction alimentaire dans les stratégies |
| Psychose, manie, consommation (alcool, drogues) | Absents ; seul l’état `unusual_belief` existe |
| Contrôle coercitif, emprise | Exclusion « Violence/coercition : aucune confrontation » dans `CHANGE_SYSTEM_RULES` ; aucune évaluation ni plan de sécurité |
| Deuil, rupture amoureuse, burn-out, solitude | Repères de catalogue (`grief_change`, `overwork`, `social_belonging`), aucune consigne propre |
| Retour de la personne sur l’échange (SRS) | Rubrique mémoire `alliance` alimentée seulement si la personne commente spontanément |
| Humour, ajustement de style | Absents |
| Pauses, fin d’accompagnement, dépendance | État `dependency` détecté par deux expressions ; consigne « ne favorise ni dépendance affective » ; aucun rappel de pause ni de fin |

## 2. Compétences de compréhension

### 2.1 Formulation de cas évolutive et partagée (au-delà des 5P de R1)

**Sources**
- Kuyken, Padesky & Dudley, *Collaborative Case Conceptualization* (Guilford, 2009) et article de 2008 — https://www.guilford.com/books/Collaborative-Case-Conceptualization/Kuyken-Padesky-Dudley/9781462504480/contents **[S]** (sommaire et extraits de recherche ; l’extrait PDF de l’éditeur a renvoyé 503). Ce qui est établi par les extraits : métaphore du « creuset » ; trois principes — empirisme collaboratif (« la chaleur » du processus), formulation qui **progresse par niveaux** (descriptive → explicative transversale « pourquoi cela m’arrive-t-il encore ? » → longitudinale « mon avenir ressemble-t-il à mon passé ? », seulement si nécessaire), **forces intégrées à chaque étape**. Une échelle de compétence (CCC-RS, Kuyken, Beshai & Dudley) note le niveau de formulation, l’empirisme collaboratif et les forces — https://eprints.ncl.ac.uk/163378 **[S]**.
- Elliott & Greenberg (2016), *Humanistic-Experiential Psychotherapy in Practice: Emotion-Focused Therapy*, version auteur — https://strathprints.strath.ac.uk/57807/1/Elliott_Greenberg_OUP2016_Humanistic_experiential_psychotherapy_in_practice.pdf **[V]**, tableau 1 : formulation en trois temps ; étape 11 « Co-construct the formulation narrative linking presenting relational and behavioral difficulties to triggering events and core emotion schemes » ; étape 10, thèmes « self-self relations, self-other relations, existential issues/interrupted life projects ».
- R1 §1.6 (Macneil et al. 2012, 5P ; mise en garde : une formulation peut aussi avoir des effets négatifs) reste la base.

**Comparaison au code**
- `ambre-engine.mjs:MEMORY_SECTIONS` : la rubrique `hypotheses` ne contient que des **citations de la personne** (`updateMemory` refuse toute phrase qui n’est pas une sous-chaîne exacte d’un message). Il n’existe **aucun objet de formulation** : ni déclencheurs, ni boucle de maintien, ni forces reliées, ni statut « proposée / reconnue / corrigée / écartée ». → **absent** pour la formulation propre à la conversation.
- La cartographie (`knowledgeIndex`, `CHILDHOOD_CARTOGRAPHY_GUIDANCE`) apporte un équivalent longitudinal (causes→conséquences, retours oui/non) et le portrait des hypothèses (`psychological_map`). Le statut oui/non de la cartographie est un bon modèle à reprendre. → **partiel** (niveau longitudinal présent, niveaux descriptif et transversal absents).
- `buildReviewSystemPrompt` (bilans) relie histoire et présent « comme une piste à vérifier », mais le bilan est un texte, non une structure révisable tour après tour. → **partiel**.
- Relecture : aucun critère ne vérifie qu’une explication proposée reste au niveau justifié par les données (on peut passer d’un fait unique à une explication longitudinale). → **absent**.

**Propositions**
1. **Objet `formulation` dans la mémoire** (proposé, pas validé) : au plus 3 « boucles » `{declencheur, lecture, emotion_ou_besoin, action, effet_court, effet_long, forces, statut: proposee|reconnue|corrigee|ecartee, citations:[turn_id+citation]}`. Seule la personne fait passer une boucle à « reconnue » (même mécanique que les cartes de changement : proposition puis choix explicite). Coût : ~250 caractères par boucle en analyse/rédaction lorsque pertinente ; à substituer à une partie des `details`, pas à ajouter.
2. **Règle de niveau** (consigne de rédaction, ~200 caractères) : « Reste au niveau que les faits permettent : décrire une situation ; relier plusieurs situations seulement si la personne en a rapporté au moins deux ; relier au passé seulement si elle l’a fait elle-même ou l’accepte, et toujours comme hypothèse avec ce qui la contredirait. »
3. **Critère de relecture R16 « Niveau d’explication proportionné »** : non si la réponse attribue une cause biographique ou un mécanisme récurrent à partir d’un seul épisode non relié par la personne. Vérifiable sur les citations transmises.
4. **Stratégie** (format catalogue) :
```json
{"id":"F01","label":"Dessiner ensemble une boucle actuelle","family":"formulation","status":"proposed",
 "intent_tags":["understand","patterns"],
 "indications":["La personne demande pourquoi une réaction revient","Au moins deux situations comparables rapportées"],
 "exclusions":["Pas de lien avec l'enfance non apporté par la personne","Pas en débordement ni rupture"],
 "permission":"P1","small_step":"Relire une situation récente : déclencheur, ce que je me suis dit, ce que j'ai fait, ce que cela a produit tout de suite et ensuite",
 "measure":"La personne dit si la boucle lui ressemble, en partie ou pas",
 "restriction":"self_help_support","blocked_contexts":["safety","stabilize","repair","listen","closed_topic"],
 "requires_safe_interpersonal_context":false,"source_refs":["à créer : Kuyken 2009 ; Elliott & Greenberg 2016"],
 "research_ref":"docs/recherche-ambre-psychologue-v3-2026-11-18.md#21"}
```

### 2.2 Thèmes relationnels récurrents (psychodynamique brève : CCRT)

**Sources**
- Grenyer & Marceau (2022), « Helping patients master core conflictual relationship themes in psychotherapy », *J Clin Psychol* 78(3):386–395 — https://pmc.ncbi.nlm.nih.gov/articles/PMC9306742 **[V, par extraction]**. Le thème central se repère dans des **épisodes relationnels** racontés : souhait (W, « to obtain love and nurturing »), réponse de l’autre (RO, perçue ou attendue), réponse de soi (RS, « withdrawing and becoming depressed »). La **récurrence** (« how often the same theme appears again and again in a person’s narratives ») est liée à la sévérité ; après thérapie, « a person’s main CCRT pattern does not disappear », mais des réponses plus adaptées gagnent du terrain.
- Fidélité inter-juges modeste des composantes (kappas ~0,3–0,44 dans une étude de cotation simplifiée, BMC Psychiatry 2020) — https://bmcpsychiatry.biomedcentral.com/articles/10.1186/s12888-020-02558-4/tables/2 **[S]**. Conséquence : un modèle qui « code » un CCRT sera peu fiable ; le présenter comme une hypothèse soumise à la personne.

**Comparaison au code** : `psychology-framework.mjs` demande de « pars[ir] d’un échange concret » et de distinguer vu/ressenti/imaginé (ancienne voie), le catalogue contient `reciprocity`, `abandonment_fear`, `closeness_distance` ; `people-observations.mjs` conserve des citations par proche. Mais rien ne **compare des épisodes entre proches différents** ni ne repère un souhait commun. → **partiel**.

**Propositions**
- **Signal d’analyse** `relationship_episode` (booléen + `person_id` si identifié) quand le message raconte une interaction précise ; permet de joindre les observations du même proche et d’un autre proche déjà décrit.
- **Consigne** (~180 caractères) : « Quand plusieurs épisodes avec des proches différents se ressemblent, tu peux proposer la ressemblance (ce que vous espériez, comment l’autre a répondu ou comment vous l’avez perçu, ce que vous avez fait) comme une question, jamais comme un trait. »
- **Mémoire** : ajouter dans `formulation` un champ `theme_relationnel` à trois cases, rempli seulement après accord.
- **Stratégie** `F02 « Comparer deux épisodes relationnels »` (P1, bloquée en contexte de violence : `requires_safe_interpersonal_context: true`).

### 2.3 Mentalisation (Bateman & Fonagy)

**Sources** (aucun texte primaire ouvert : PDF du manuel qualité MBT en 404, article de l’*American Journal of Psychotherapy* 2021 en 403)
- Volkert et al. / article *Am J Psychother* 2021, doi:10.1176/appi.psychotherapy.20210020 **[S]** : sous forte activation émotionnelle, la pensée bascule vers des modes « prémentalisants » (équivalence psychique : la pensée vaut réalité ; mode téléologique : seuls les actes comptent ; faire-semblant : discours sur les états mentaux déconnecté du vécu). Ordre d’intervention : **validation empathique d’abord**, puis, quand l’activation baisse, exploration, questions qui interrogent, élaboration de l’affect, travail relationnel.
- Posture « **not-knowing** » : curiosité, ouverture, humilité, questionnement actif plutôt qu’explication — manuels MBT cités par les extraits **[S]** ; même idée en psychothérapie interpersonnelle, https://psychotherapy.psychiatryonline.org/doi/10.1176/appi.psychotherapy.20190021 **[S]**.

**Comparaison au code**
- Gradation selon l’activation : `work` force `stabilize` + `no_exploration` + `no_questions` en `distress`/`dissociation` → **présent** (fondé sur R1, fenêtre de tolérance).
- Posture de non-savoir : la rédaction impose « hypothèses incertaines et révisables, invitation à corriger » → **présent** sur le fond.
- **Absent** : (a) repérage de l’**équivalence psychique** (« il me déteste, c’est évident ») sans détresse aiguë : aujourd’hui seul `unusual_belief` existe, réservé aux croyances inhabituelles ; (b) repérage du **pseudo-mentalisme** (longues analyses intellectuelles sans affect, proche de `intellectualization` du catalogue, mais sans posture associée) ; (c) **questions de perspective** sur l’autre (« qu’est-ce qui, selon vous, se passait pour lui à ce moment ? ») : présentes dans l’ancien prompt seulement.

**Propositions**
- **États d’analyse** `certainty_about_others` et `intellectualizing` (ajout à `STATES`, sans effet de blocage) ; posture associée « ralentir et explorer » : une seule question de perspective, curiosité explicite, pas de contre-argument.
- **Consigne** (~220 caractères) : « Si la personne présente l’intention d’un autre comme une évidence, ne la confirme ni ne la contredis : montre ta curiosité, demande ce qui l’a amenée à cette lecture et ce que l’autre pourrait en dire. Si elle analyse longuement sans émotion, invite une seule fois à dire ce qu’elle ressent maintenant. »
- **Critère de relecture** (fusionnable avec n°11 « pas de complaisance ») : « La réponse ne confirme pas comme un fait une intention attribuée à un tiers. » Déterministe partiel : motifs « il/elle (veut|cherche à) vous » dans la réponse.

### 2.4 Thérapie centrée sur les émotions (EFT de Greenberg) : marqueurs et tâches

**Source** : Elliott & Greenberg (2016), même URL **[V]**. « A defining feature of the EFT approach is that intervention is marker guided » ; les marqueurs indiquent aussi la **disponibilité** de la personne. Marqueurs → tâches : *réaction problématique* (« je me suis senti si triste et je ne sais pas pourquoi ») → déploiement évocateur ; *ressenti flou* (« I just can’t put my finger on it ») → focusing ; *conflit interne autocritique* → dialogue entre deux parts ; *auto-interruption* (« je retiens mes larmes ») ; *affaire inachevée* envers un proche → chaise vide ; *détresse bloquée* (« No one will ever understand me ») → apaisement compassionnel de soi ; plus *rupture d’alliance*, *confusion / faire de l’espace*, et marqueurs narratifs (« same old story, empty story, untold story, healing story », Angus & Greenberg 2011). Quatre types d’émotions : primaire adaptée, primaire inadaptée (apprise, souvent après des expériences traumatiques), **secondaire réactive** (réaction à l’émotion primaire, ex. colère contre sa peur), instrumentale.

**Comparaison au code**
- Les marqueurs correspondent à des besoins déjà partiellement servis : autocritique → K01–K03 (compassion) ; ressenti flou → A02, D01 ; réaction problématique → C01 (situation-pensée-émotion-action) mais sans l’objectif EFT de **trouver le sens implicite**. → **partiel**.
- Les techniques de chaise et le travail sur « l’affaire inachevée » sont **explicitement exclus** de `CHANGE_SYSTEM_RULES` (« ni … chaises, reparentage, rescriptage »). Cette exclusion doit **rester** pour un outil écrit sans clinicien (risque d’activation émotionnelle sans contenant).
- La distinction **émotion secondaire / primaire** n’existe nulle part (ni `ANALYSIS_SCHEMA`, ni catalogue). → **absent**. C’est la contribution EFT la plus transposable par écrit et la moins risquée.
- Les marqueurs narratifs « même vieille histoire » / « histoire vide » rejoignent la récurrence (2.2) et l’alexithymie (`emotional_identification`). → **absent** comme signal.

**Propositions**
- **Signal d’analyse** `emotion_marker` ∈ {none, problematic_reaction, unclear_felt_sense, self_critical_split, self_interruption, unfinished_business, stuck_anguish, same_old_story}. Effets : `unfinished_business` et `stuck_anguish` → aucune tâche de changement, accueil et, si récurrent, préparation de séance ; `problematic_reaction` → C01 étendu ou F03 ; `unclear_felt_sense` → F04.
- **Stratégie F03 « Retrouver ce qu’une réaction voulait dire »** : petit pas « décrire le moment exact où la réaction est montée et ce que ce moment signifiait pour vous » ; exclusions : pas de trauma, pas en débordement ; P1.
- **Stratégie F04 « Mettre un mot sur un ressenti flou »** (focusing allégé, écrit) : « Remarquer où cela se situe dans le corps, essayer un mot, vérifier s’il convient ; “je ne sais pas” est une réponse complète » ; bloquée en `dissociation`.
- **Consigne** (~160 caractères) : « Une colère, une honte ou un découragement peut recouvrir une autre émotion. Tu peux le proposer comme hypothèse, une fois, sans insister. »

### 2.5 Thérapie interpersonnelle (TIP / IPT) : quatre domaines

**Source** : Markowitz & Weissman (2004), « Interpersonal psychotherapy: principles and applications », *World Psychiatry* 3(3):136–139 — https://pmc.ncbi.nlm.nih.gov/articles/PMC1414693 **[V, par extraction]** : « Mood and life situation are related » ; quatre foyers : deuil compliqué (« someone important may have died »), conflit de rôle (« a struggle with a significant other »), transition de rôle (« some other important life change »), déficits interpersonnels (« the default focus […] a confusing term ») ; **inventaire interpersonnel** (« a review of the patient’s patterns in relationships, capacity for intimacy ») ; rôle de malade temporaire ; traitement limité dans le temps (12–16 semaines), « The acute time limit pressures the patient to take action ». Miller et al. (1998), personnes âgées — https://pmc.ncbi.nlm.nih.gov/articles/PMC3330480 **[V]** : transition 41 %, conflits 34,5 %, deuil 23 % dans leur échantillon.

**Comparaison au code**
- « Mes proches » et la ligne de vie fournissent de fait un **inventaire interpersonnel** déclaré (`knowledgeIndex` : `person`, `event`). → **présent** comme données.
- Aucune lecture « **quel événement de vie récent accompagne la difficulté actuelle ?** » : l’analyse ne relie pas l’humeur à un deuil, un conflit ou une transition récents de la ligne de vie. Le repère `identity_transition` et `grief_change` existent dans le catalogue mais ne déclenchent aucune posture. → **partiel**.
- Conflit de rôle : la TIP distingue trois stades (négociation, impasse, dissolution) **[S, connaissance générale]** ; Ambre n’a pas ce repère, et les stratégies C07/C08/D06 (demande, limite, enjeu de conversation) ne demandent pas d’abord **ce que chacun attend de l’autre**. → **partiel**.

**Propositions**
- **Signal d’analyse** `life_event_focus` ∈ {none, loss, dispute, transition, isolation} avec `event_id` si la ligne de vie contient l’événement daté des 12 derniers mois. Coût ~100 caractères en sortie d’analyse.
- **Stratégie I01 « Attentes réciproques dans un désaccord »** : « Écrire ce que vous attendez de l’autre et ce que vous pensez qu’il attend de vous ; repérer un point négociable » ; `requires_safe_interpersonal_context: true` ; bloquée en cas de violence ou de contrôle (cf. §5).
- **Stratégie I02 « Ce que la transition a fait perdre et apporté »** : lister une perte et un gain (ou une possibilité) du nouveau rôle ; indiquée pour déménagement, retraite, parentalité, rupture, diagnostic.
- **Mémoire** : rubrique ou étiquette `focus` (foyer de travail choisi par la personne) pour éviter de disperser l’accompagnement d’un tour à l’autre.

### 2.6 Thérapie métacognitive (Wells)

**Source** : Normann & Morina (2018), « The Efficacy of Metacognitive Therapy: A Systematic Review and Meta-Analysis », *Front Psychol* 9:2211 — https://www.frontiersin.org/articles/10.3389/fpsyg.2018.02211 **[V, par extraction, 100 000 premiers caractères]**. Le syndrome cognitivo-attentionnel (CAS : inquiétude/rumination persistantes, surveillance de la menace, stratégies d’adaptation contre-productives) est maintenu par des croyances métacognitives **positives** (« Worry helps me stay prepared ») et **négatives** (« I have no control over my worry/rumination »). Effets : g = 2,06 contre liste d’attente (k = 9) ; g = 0,69 puis 0,37 au suivi contre interventions cognitivo-comportementales, avec appel à davantage d’essais. Techniques **[S]** : pleine conscience détachée, **report de l’inquiétude** (créneau différé), entraînement attentionnel.

**Comparaison au code** : repère `rumination` du catalogue (« Cette réflexion débouche-t-elle sur une action ou revient-elle au même point ? ») ; A01 (prendre du recul sur une pensée) ; C06 (problème concret). Aucune question sur les **croyances au sujet de la rumination** (« ruminer vous aide-t-il ? pouvez-vous arrêter ? »), aucun **report** de l’inquiétude. → **partiel**. Risque spécifique non couvert : Ambre elle-même peut **devenir l’outil de rumination** (analyses répétées du même événement), ce que l’APA 2025 signale pour la recherche de réassurance (R1 §5.1).

**Propositions**
- **Signal d’analyse** `repetitive_thinking` : même situation analysée au moins 3 tours sur les 6 transmis (calculable à partir de `history` + mots-clés, ou estimé par le modèle d’analyse).
- **Consigne** : « Si la même question revient sans élément nouveau, dis-le avec douceur et propose de regarder ce que la réflexion apporte, plutôt qu’une nouvelle analyse. »
- **Stratégie MC01 « Reporter l’inquiétude à un créneau choisi »** : petit pas « noter la pensée, la reporter à un moment de 15 minutes choisi, voir si elle est encore importante » ; exclusions : idées suicidaires, TOC suivi (risque de ritualisation), obsession avec rituels ; `measure` : « l’inquiétude était-elle encore là au créneau ? ». Familles : `metacognitive`.
- **Stratégie MC02 « Ce que je crois sur ma rumination »** : deux questions écrites (« À quoi cela me sert ? Est-ce que je peux m’arrêter ? »), sans contester.

### 2.7 Approches narrative et orientée solutions, sens (logothérapie)

**Sources**
- Narrative (White & Epston) : externalisation, exceptions (« unique outcomes »), réécriture — revue Ghavibazou et al. (2022) et revue de 20 ECR (Permatasari et al., 2026) citées par extraits **[S]** ; Hilker note que la valeur de l’externalisation repose surtout sur des descriptions de praticiens **[S]**. Preuve faible ; technique d’écriture peu risquée.
- Orientée solutions (SFBT) : méta-analyse 2023 (*Research on Social Work Practice*, doi:10.1177/10497315231162611) **[S, extrait]** : questions miracle et d’exception plus efficaces que des questions centrées sur le problème pour diminuer l’affect négatif ; questions d’échelle associées à davantage d’actions ; Franklin et al. (2016), revue des recherches de processus — https://meadowscenter.org/wp-content/uploads/2022/04/Franklin_et_al-2016-Journal_of_Marital_and_Family_Therapy1.pdf **[S]**.
- Sens : psychothérapie centrée sur le sens (Breitbart), essais chez des personnes atteintes de cancer avancé ; quatre sources de sens (créative, expérientielle, attitudinale, historique) ; méta-analyse 2021 jugeant les preuves « émergentes » — https://pmc.ncbi.nlm.nih.gov/articles/PMC5501751 **[S]**.

**Comparaison au code**
- Exceptions : `INTERVIEW_GUIDE.stages` contient `seek_counterexample`, le prompt dit « Appuie-toi sur les réussites, les exceptions » → **présent** (ancienne voie surtout ; rédacteur actif : « hypothèses… invitation à corriger » seulement).
- Questions d’échelle : G01/G03 demandent une mesure facultative ; pas de question d’échelle de progrès (« de 0 à 10, où en êtes-vous ? qu’est-ce qui ferait +1 ? »). → **partiel**.
- Externalisation (« la peur », « l’autocritique » comme quelque chose qui agit sur la personne) : A01 (recul sur une pensée) s’en approche. → **partiel**.
- Sens et valeurs : A03 (valeur personnelle), repère `values_alignment`. Rien sur l’**attitude face à une souffrance inévitable** (maladie, deuil, perte irréversible), où l’objectif n’est pas de changer la situation. → **partiel**.

**Propositions**
- **Stratégie N01 « Nommer le problème comme un visiteur »** (externalisation légère) : « Donner un nom à ce qui vous gêne et repérer un moment où il a eu moins d’emprise » ; exclusions : psychose, dissociation (risque de renforcer une expérience d’influence extérieure), violence (ne pas externaliser la responsabilité d’un agresseur).
- **Stratégie S01 « Le plus petit signe que cela va un peu mieux »** (SFBT) : échelle 0–10 choisie par la personne et question « qu’est-ce qui ferait un point de plus ? » ; exclusions : deuil récent, crise.
- **Stratégie L01 « Ce qui compte encore quand on ne peut pas changer la situation »** : sources de sens (ce que je fais, ce que je vis, la manière dont je fais face, ce que je transmets) ; indications : maladie chronique, deuil, perte irréversible ; jamais comme réponse à des idées suicidaires (voie sécurité prioritaire).

## 3. Facteurs communs et compétences relationnelles fines

### 3.1 Réactance et ajustement du degré de directivité

**Source** : Beutler, Harwood, Michelson, Song & Holman (2011), « Resistance/Reactance Level », *J Clin Psychol: In Session* 67:133–142 — https://clinica.ispa.pt/sites/default/files/68._reactance_level.pdf **[V]**. Méta-analyse de 12 études (N = 1 102) : faible réactance → meilleure réponse aux approches directives ; forte réactance → meilleure réponse aux approches non directives (d = 0,82). Pratiques recommandées, citées : signes d’état « expressed anger at the treatment or therapist, ranging from simple dissatisfaction with therapeutic progress to overt expressions of resentment » ; réponse : reconnaître et refléter, parler de la relation, « renegotiation of the therapeutic contract regarding goals and therapeutic roles » ; en forte réactance, « de-emphasize the use of rigid homework assignments », tâches présentées comme des expériences « that require minimal overt action », « fewer instructions should be used » ; « Avoid stimulating the patient’s level of resistance » ; voir la résistance comme « a signal that ineffective methods are being used ». Mise à jour 2018 (13 études, N = 1 208, d = 0,79) **[S]**.
Préférences : Swift, Callahan, Cooper & Parkin (2018), *J Clin Psychol* 74(11):1924–1937 **[S]** — tenir compte des préférences réduit l’abandon (OR rapporté ~1,79 par une revue secondaire) et améliore modestement les résultats (d ≈ 0,28).

**Comparaison au code**
- Rupture repérée par six expressions (`deterministicAnalysis` : « tu ne comprends », « réponse de robot »…) et par le modèle (`states: rupture`) → posture `repair` → **présent** pour la confrontation explicite.
- **Absent** : (a) une **mémoire du style préféré** (plus ou moins directif, réponses courtes ou développées, exemples concrets ou non, tutoiement refusé, humour accepté ou non) ; la rubrique `alliance` ne garde que des citations éparses et `memoryPrompt` la limite à 380 caractères ; (b) la **réactance comme trait** : une personne qui refuse plusieurs propositions de suite reçoit encore des propositions tant que `no_advice` n’est pas exprimé dans le message courant ; (c) les **ruptures de retrait** (réponses de plus en plus courtes, « oui », « je sais pas », déférence) — R1 les décrit, aucune règle ne les détecte.
- Le plan de changement exige déjà permission (P1) et présente les essais comme facultatifs (C-R6) → **présent** pour les devoirs non rigides.

**Propositions**
- **Signal calculé (sans appel modèle)** `declined_streak` : nombre de propositions de changement refusées ou ignorées consécutivement (déjà disponibles dans le plan : propositions `proposed` non adoptées, `rejected`). À partir de 2 : `analysis.constraints += no_advice` sauf demande explicite, et consigne « reviens à l’écoute et demande ce qui serait utile ». Coût nul en jetons d’entrée.
- **Signal calculé** `withdrawal_markers` : les trois derniers messages de la personne raccourcissent fortement (ex. < 40 % de sa médiane) ou se réduisent à des acquiescements. Effet : une question de méta-communication légère (« J’ai l’impression d’avoir peut-être pris trop de place ; qu’est-ce qui vous serait le plus utile maintenant ? »), jamais deux tours de suite.
- **Mémoire `style`** (rubrique 9, ≤ 5 citations exactes, 300 caractères) : préférences exprimées sur la forme. Transmise à la rédaction à chaque tour (coût ~100 jetons). Visible et effaçable comme les autres rubriques.
- **Critère de relecture R17 « Directivité ajustée »** : non si la réponse propose un exercice ou un conseil alors que la personne a refusé une proposition dans l’un des deux tours précédents, sans nouvelle demande.

### 3.2 Retour de la personne à chaque échange (SRS/ORS) et suivi des progrès

**Sources**
- de Jong et al. (2021), « Using progress feedback to improve outcomes and reduce drop-out, treatment duration, and deterioration: A multilevel meta-analysis », *Clin Psychol Rev* 85:102002 — https://www.pubmed.gov/33721605/ **[R, résumé via extraits]** : 58 études, 21 699 patients ; effet **faible** sur les symptômes (d = 0,15), similaire pour les cas « not-on-track » (d = 0,17) ; légère réduction de l’abandon (OR = 1,19). Effets plus nets quand des outils d’aide clinique accompagnent le retour **[S]**. Nuance donc R1 §1.8 (Lambert 2018) : le bénéfice existe mais il est modeste.
- SRS (Duncan et al., 2003, *J Brief Therapy* 3(1):3–12) : quatre items visuels analogiques remplis en fin de séance ; taux d’utilisation 96 % contre 29 % pour la WAI courte ; revue psychométrique Murphy et al. (2020) — https://pubmed.ncbi.nlm.nih.gov/32420833/ **[S]**. Libellés exacts des items non vérifiés (les mêmes réserves que R1). Instrument protégé par des droits : ne pas le reproduire, s’en inspirer.

**Comparaison au code**
- Rubrique mémoire `alliance` (« Votre retour sur les échanges ») : ne capte que les retours **spontanés**. → **partiel**.
- Bilans (`REVIEW_POINT`) : comparent avant/après sur le fond, pas sur la relation. → **absent** pour le retour relationnel.
- Interface : aucun bouton de retour rapide sur une réponse (le grep « feedback » de `chat-interface.mjs` ne concerne que l’état d’erreur). → **absent**.
- Mise en garde déjà notée par R1 (Sharma et al. 2023) : un pouce levé récompense la complaisance ; ne jamais l’utiliser comme mesure de qualité clinique ni pour ajuster le style vers plus d’accord.

**Propositions**
- **Interface (sans jeton)** : à la fin d’une réponse, trois boutons facultatifs « Cela m’aide / Pas vraiment / On s’est mal compris » + champ libre ; enregistrés dans une table, injectés dans la mémoire `alliance` sous forme de **fait déclaré daté** (pas de citation inventée). « On s’est mal compris » déclenche `rupture` au tour suivant (posture `repair`). Ne jamais en tirer un score affiché.
- **Point relationnel périodique** : dans `REVIEW_POINT`, ajouter une phrase « Demande ce qui, dans la manière d’échanger, aide ou gêne (rythme, longueur, questions, conseils) » ; coût ~40 jetons par bilan, pas par tour.
- **Échelle de progrès choisie** (rejoint S01, §2.7) : si la personne l’accepte, une question d’échelle sur son objectif tous les ~10 échanges, stockée dans le plan (`measure`), jamais imposée.

### 3.3 Réparation d’une rupture (écart d’implémentation de R1 §4.2)

R1 a sourcé la réparation (Eubanks, Muran & Safran 2018). Écarts constatés dans le code :
- La consigne `repair` du rédacteur dit « reconnaître la rupture et sa part sans se justifier » → **présent**.
- **Absent** : (a) **changer la tâche** après une rupture (le plan peut encore proposer un essai au tour suivant ; seul le tour de rupture est bloqué par C0) ; (b) **expliquer la raison** de ce qu’Ambre proposait ; (c) **vérifier au tour suivant** que la réparation a été reçue. Proposition : drapeau `last_turn_repair=true` (calculé à partir de la trace du tour précédent, coût nul) → au tour suivant, aucune nouvelle méthode et une phrase de vérification légère si la personne n’a pas repris d’elle-même.
- Critère de relecture : en posture `repair`, non si la réponse contient une justification défensive (« je voulais seulement », « comme je l’ai dit ») — motif déterministe simple, à compléter par le relecteur.

### 3.4 Humour

**Sources** : Sarink & García-Montes (2023), revue systématique PRISMA de l’humour comme intervention en psychothérapie pour la dépression et l’anxiété — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9845902/ **[S, extrait]** : première revue sur le sujet ; risque signalé que l’humour fasse douter la personne du sérieux du thérapeute. Corrélation humour–alliance dans une étude isolée (Ventis et al., rapportée en 2019) **[S]** ; absence de lien dans une autre thèse **[S]**. **Preuve faible**.

**Comparaison** : aucune consigne. → **absent**, mais **faible priorité**.

**Proposition** : consigne minimale (~120 caractères) « N’initie pas d’humour. Si la personne en utilise, tu peux l’accueillir légèrement, jamais sur sa souffrance ni pour minimiser. » Critère de relecture : l’humour initié par Ambre en posture `stabilize`, `repair` ou `safety` = non.

### 3.5 Pauses, fin et dépendance à l’IA

**Sources**
- Norcross, Zimmerman, Greenberg & Swift (2017), *Psychotherapy* 54:66–75 ; résumé des auteurs (Greenberg, Norcross & Zimmerman, 2017) — https://societyforpsychotherapy.org/expert-pantheoretical-advice-psychotherapy-termination/ **[V]**. Huit conduites de fin consensuelles : « Prepare Explicitly for Termination », « Process Feelings of Patient and Therapist », « Discuss Patient’s Future Functioning and Coping », « Help Patient to Use New Skills Beyond Therapy », « Frame Personal Development as Unfinished », « Anticipate Post-Therapy Growth and Generalization », « Reflect on Patient Gains », « Express Pride in Patient’s Progress and Mutual Relationship » (la dernière est à transposer avec prudence : une IA n’a pas de « relation mutuelle »).
- OpenAI, « Helping people when they need it most » (26 août 2025) — https://openai.com/index/helping-people-when-they-need-it-most/ **[S, page en 403 ; citations concordantes de plusieurs relais]** : « these safeguards can sometimes be less reliable in long interactions » ; incitations à faire une pause lors de très longues sessions.
- APA 2025 (R1 §5.1) : « nudges that encourage users to take breaks », pas de persuasion contre les conversations réelles ; Fang et al. 2025 (R1) : usage volontaire intensif associé à davantage de solitude et de dépendance.
- IPT (§2.5) : la limite de temps « pressures the patient to take action ».

**Comparaison au code**
- État `dependency` : deux expressions (« seule à me comprendre », « personne d’autre que toi/vous ») et posture « préserver les liens humains ». Critère 15 « Autonomie et liens humains préservés ». → **partiel**.
- Quota de 100 réponses par semaine (README) : limite commerciale, pas clinique ; aucune indication horaire (conversations nocturnes prolongées), aucune durée de session. → **absent** comme mesure de bien-être.
- Aucune notion de **pause choisie**, de **fin d’accompagnement** ou de **bilan de clôture** : les bilans sont des « points d’étape » sans fin prévue. → **absent**.
- Aucune comparaison de **l’usage** dans le temps (fréquence, heures). → **absent**.

**Propositions**
- **Signaux calculés (sans appel modèle)** : `session_turns_today`, `late_night` (00 h–05 h Europe/Paris, `current_time` existe déjà), `days_active_last_14`. Seuils de conception : > 25 tours dans la journée ou ≥ 3 nuits tardives sur 7 → consigne d’une phrase « Propose doucement une pause ou un appui humain si cela a du sens, sans clore l’échange » une fois par jour au plus. Jamais en voie sécurité (l’échange ne doit pas être interrompu en crise).
- **Élargir `dependency`** au-delà des deux expressions : « je ne parle qu’à toi », « tu es mon/ma seul·e ami·e », « je préfère te parler qu’à », « tu me comprends mieux que mon psy », « sans toi je ». Coût nul (expressions) + le modèle d’analyse les repère déjà sémantiquement.
- **Bilan de clôture** (`REVIEW_CLOSING`, sur demande ou proposé quand un objectif est marqué atteint) : ce qui a changé selon la personne, ce qu’elle a appris à faire seule, les situations à venir et comment y faire face, ce qui reste ouvert (« inachevé » au sens de Norcross), qui peut l’aider hors d’Ambre. Sans formule de fierté affective.
- **Mode pause** dans les paramètres : la personne suspend les relances et peut indiquer une date de retour ; `isStopFollowups` existe déjà pour les relances (`change-plan.mjs`), à étendre.
- **Critère de relecture** : non si la réponse se présente comme irremplaçable, invite à revenir sans raison (« revenez quand vous voulez, je serai toujours là »), ou dévalorise un proche ou un soignant par comparaison (déjà partiellement dans le prompt : « exclusivité affective ou supériorité sur un proche ou soignant ») → transformer en **contrôle déterministe** sur « toujours là », « je serai là pour vous à tout moment », « mieux que ».

## 4. Situations particulières

Constat transversal : le moteur actif n’a **aucune consigne propre à une situation de vie**. Le catalogue de repères contient `grief_change`, `identity_transition`, `overwork`, `social_belonging`, `stress_resources`, mais un repère ne déclenche ni posture, ni ressource, ni exclusion. La proposition commune est donc un **petit registre de « fiches situation »** (`data/situation_guides.json`, généré comme `change_strategies`), sélectionné par un champ d’analyse `situation` (0 à 2 valeurs) et injecté **seulement quand il est sélectionné** (≤ 450 caractères par fiche, substitué à des `details`). Coût nul les autres tours. Chaque fiche : `{id, signaux, a_faire, a_eviter, ressources_verifiees, strategies_compatibles, strategies_exclues, orientation}`.

### 4.1 Deuil

**Sources** : Stroebe & Schut, modèle du double processus (1999 ; « a decade on », *Omega* 2010) — https://pubmed.ncbi.nlm.nih.gov/21058610/ **[S]** : alternance entre activités tournées vers la perte (manque, souvenirs) et vers la restauration (nouvelles routines, nouveaux rôles), l’oscillation étant elle-même adaptative. Trouble de deuil prolongé — Cn2r (centre national de ressources et de résilience) — https://cn2r.fr/en/le-trouble-de-deuil-prolonge/ **[V, par extraction]** : CIM-11 « at least 6 months, all age categories combined » ; DSM-5-TR « 6 months […] in adolescents and children, compared with 12 months for adults » ; mise en garde contre la pathologisation de réactions « perfectly normal in another » culture et débat sur la médicalisation du deuil.
**Comparaison** : `ANALYSIS_SYSTEM` cite le deuil comme sujet sensible (question sensible sous conditions) → **présent** pour la prudence ; aucune normalisation de l’oscillation, aucun repère de durée, aucune distinction deuil / dépression, aucune interdiction de stratégies inadaptées (C04 « activité plaisir » ou S01 échelle juste après un décès peuvent sembler déplacés). → **partiel**.
**Fiche `grief`** : à faire — accueillir sans calendrier, normaliser l’alternance « penser à la personne / s’occuper du reste », demander comment on parle de la personne disparue chez elle (culture, rites) ; à éviter — « il faut faire son deuil », étapes imposées, recherche active d’activités plaisir dans les premières semaines sauf demande, qualifier le deuil de trouble ; orientation — si une douleur intense et envahissante dure au-delà de 6 à 12 mois et empêche de fonctionner, dire qu’un professionnel peut évaluer un deuil prolongé ; idées de rejoindre la personne disparue → voie sécurité (signal à ajouter à `crisisSignal` : « rejoindre (ma mère|mon père|lui|elle) » en contexte de deuil, sous vérification sémantique).

### 4.2 Rupture amoureuse et conflit de couple

**Sources** : TIP (§2.5, conflit de rôle et transition) **[V]** ; pas de source spécifique ouverte sur la rupture amoureuse dans cette recherche (connaissance générale **[S]** : rumination, contact répété avec l’ex-partenaire, risque suicidaire accru lors de ruptures chez les jeunes adultes — non vérifié ici).
**Comparaison** : le prompt ancien (`buildCompanionSystemPrompt`) traite la compréhension d’un proche (vu / ressenti / imaginé, vérifier auprès du proche) et interdit la confrontation symétrique en cas de violence ; le moteur actif ne transmet pas ces consignes (§1.2). Aucune fiche « rupture ». → **partiel**.
**Fiche `breakup_or_couple_conflict`** : à faire — distinguer conflit entre égaux et rapport de contrôle (voir §5.1) **avant** toute aide à la communication ; I01 (attentes réciproques) seulement si aucun signe de peur, contrôle ou violence ; à éviter — juger l’autre absent (« il est toxique »), conseiller de rompre ou de rester, interpréter un partenaire en « pervers narcissique » (déjà interdit par `deterministicReview` n°3) ; signaux de vigilance — idées suicidaires après rupture, surveillance de l’ex-partenaire (orienter sans complaisance), menace de l’ex-partenaire.

### 4.3 Parentalité et période périnatale

**Sources** : aucune source périnatale ouverte ici. Le 119 (enfance en danger) est vérifié par R1 **[V]**.
**Comparaison** : rien. → **absent**.
**Fiche `parenting`** (minimale, à sourcer avant activation) : déculpabiliser sans minimiser ; distinguer fatigue ordinaire et signes d’alerte périnataux (tristesse durable, pensées de faire du mal au bébé ou à soi, perte de contact avec la réalité) → orientation médicale rapide (médecin, sage-femme, PMI) et 15 si danger ; pensées intrusives non désirées de faire du mal au bébé : ne pas dramatiser, orienter vers un professionnel (fréquentes et souvent liées à l’anxiété — **[S] à vérifier dans une source périnatale** avant toute consigne) ; tout danger pour un enfant → 119.

### 4.4 Épuisement professionnel et travail

**Source** : OMS, « Burn-out an “occupational phenomenon” » (28 mai 2019) — https://www.who.int/news/item/28-05-2019-burn-out-an-occupational-phenomenon-international-classification-of-diseases **[V]** : « a syndrome conceptualized as resulting from chronic workplace stress that has not been successfully managed » ; trois dimensions (« energy depletion or exhaustion », « increased mental distance from one’s job, or feelings of negativism or cynicism », « reduced professional efficacy ») ; « should not be applied to describe experiences in other areas of life » ; « not classified as a medical condition ».
**Comparaison** : repères `overwork`, `perfectionism`, `overcontrol` ; aucune règle sur le vocabulaire (« vous faites un burn-out » serait un quasi-diagnostic non détecté par `deterministicReview`) ; aucune orientation vers médecin du travail / médecin traitant. → **partiel**.
**Fiche `work_exhaustion`** : à faire — décrire avec les trois dimensions sans conclure ; séparer ce qui dépend de la personne et de l’organisation du travail (éviter de tout ramener à un défaut individuel) ; rappeler que le médecin traitant et le médecin du travail peuvent évaluer et agir (arrêt, aménagement) ; stratégies compatibles C06, G01, D04 ; à éviter — augmenter la charge par des exercices quotidiens ; « burn-out » appliqué à la vie familiale (l’OMS le réserve au travail ; parler d’épuisement). **Critère de relecture** : ajouter « burn-out » aux étiquettes diagnostiques de `deterministicReview` n°3 sous la forme « vous (faites|êtes en) (un )?burn.?out ».

### 4.5 Solitude

**Source** : OMS, Commission on Social Connection, rapport du 30 juin 2025 — https://www.who.int/news/item/30-06-2025-social-connection-linked-to-improved-heath-and-reduced-risk-of-early-death **[R, communiqué via extraits]** : une personne sur six concernée par la solitude ; taux les plus élevés chez les 13–29 ans ; ~871 000 décès par an associés (estimation sur 2014–2019).
**Comparaison** : repères `withdrawal`, `social_belonging` ; critère 15 (liens humains) ; état `dependency`. Le risque spécifique — **la personne seule remplace les liens par Ambre** — est couvert par une consigne générale, sans action concrète. → **partiel**.
**Fiche `loneliness`** : à faire — distinguer isolement (peu de contacts) et sentiment de solitude ; un seul lien accessible plutôt qu’un programme ; ressources de proximité non numériques (associations, activités) sans les imposer ; à éviter — se présenter comme compagnie suffisante. Lien avec §3.5 (signaux d’usage).

### 4.6 Identité, orientation sexuelle et identité de genre

**Source** : loi n° 2022-92 du 31 janvier 2022 interdisant les pratiques visant à modifier l’orientation sexuelle ou l’identité de genre ; article 225-4-13 du code pénal — https://www.vie-publique.fr/loi/281790-loi-interdisant-les-therapies-de-conversion-lgbt **[S, synthèse officielle non ouverte ; texte Légifrance non lu]**.
**Comparaison** : `buildCompanionSystemPrompt` respecte le genre déclaré et `gender_description` et interdit toute inférence à partir du genre → **présent** (ancienne voie ; le moteur actif transmet `identity`). Rien n’interdit explicitement d’**accompagner une demande de changer son orientation ou son identité** (risque de pratique de conversion), et le schéma `SCH03` « assouplir une règle personnelle » ou C02 pourraient être détournés. → **partiel**.
**Consigne** (~200 caractères) : « Orientation sexuelle et identité de genre ne sont jamais des problèmes à corriger. Si la personne souhaite les changer, accueille la souffrance (rejet, honte, peur) sans proposer de méthode de changement ; ces pratiques sont interdites en France. » **Critère** : exclusion explicite dans `common_rules` et C-R1.

### 4.7 Maladie chronique

**Comparaison** : C04/C05 et l’activation peuvent heurter une limite physique ; aucune règle « adapter au niveau d’énergie et aux contraintes médicales » ; interdiction déjà présente de modifier un traitement → **partiel**.
**Fiche `chronic_illness`** : rythme et énergie (« un pas adapté à votre énergie du jour »), acceptation et sens (L01, A02), ne jamais attribuer un symptôme physique à une cause psychologique, renvoyer les questions médicales au soignant ; transition de rôle (I02).

### 4.8 Neurodivergence

**Source** : NICE CG142 (adultes autistes) — https://www.nice.org.uk/guidance/cg142/chapter/recommendations **[S, extraits]** : informations dans un format accessible ; en l’absence de recherche spécifique, utiliser les interventions existantes pour un trouble associé en les adaptant ; choix selon les besoins propres de la personne. La formulation exacte des adaptations (langage concret, structure) n’a pas été lue.
**Comparaison** : rien sur la forme ; la consigne « mots simples » existe (critère 14). Le repère `executive` (domaine du catalogue) existe. → **partiel**.
**Mémoire `style`** (§3.1) couvre l’essentiel : si la personne indique un fonctionnement autistique ou un TDAH déclaré, privilégier phrases explicites, une idée par paragraphe, questions fermées possibles, pas de métaphore non expliquée, exemples concrets ; ne jamais présenter la neurodivergence comme un défaut à corriger ni la déduire des messages.

### 4.9 Jeunes adultes, personnes âgées, mineurs

- **Mineurs** : CNIL, recommandation 4 — https://cnil.fr/fr/recommandation-4-rechercher-le-consentement-dun-parent-pour-les-mineurs-de-moins-de-15-ans **[S, extraits]** : en France, consentement autonome aux services en ligne à partir de 15 ans (art. 45 loi Informatique et libertés) ; en dessous, consentement conjoint avec un titulaire de l’autorité parentale ; pas de « majorité numérique globale ». APA 2025 (R1) : garde-fous spécifiques pour enfants et adolescents. Fil Santé Jeunes, 0 800 235 236, gratuit et anonyme, 12–25 ans, 9 h–23 h tous les jours selon plusieurs sources publiques **[S ; une source gouvernementale donne d’autres horaires — vérifier sur filsantejeunes.com avant d’intégrer]**.
  **Comparaison** : l’âge est calculé dans `current_identity` mais **aucune règle** ne traite un âge < 18 ans, ni une déclaration en cours de conversation (« j’ai 15 ans », « mes parents me… »). Le 119 n’apparaît que dans la liste blanche de numéros. → **absent**. Question juridique à trancher (hors recherche) : le service doit-il refuser les mineurs ? En attendant, signal d’analyse `minor_signal` (âge déclaré < 18, ou mention scolaire collège/lycée + parents) → réponse adaptée, aucune exploration de l’enfance actuelle comme passé, orientation vers un adulte de confiance, Fil Santé Jeunes (après vérification) et 119 en cas de danger ; aucune stratégie de changement relationnel avec les parents (`requires_safe_interpersonal_context`).
- **Personnes âgées** : TIP chez la personne âgée, transitions de rôle majoritaires (Miller et al. 1998, §2.5 **[V]**) ; les auteurs les plus âgés sur-représentés dans les homicides conjugaux de 2024 (étude nationale relayée par l’AFP **[S]**). Proposition : fiche `later_life` (retraite, perte d’autonomie, deuils multiples, aidant), I02 et L01 compatibles ; vigilance sur l’isolement et sur les maltraitances (3977, numéro national contre les maltraitances des personnes âgées et handicapées — **non vérifié ici, à sourcer avant usage**).
- **Jeunes adultes** : solitude maximale chez les 13–29 ans (OMS 2025 **[R]**) ; transitions multiples (études, premier emploi, départ du foyer) → I02.

### 4.10 Cultures et religions

**Source** : DSM-5, Cultural Formulation Interview (16 questions, 4 domaines : définition culturelle du problème ; perception des causes, du contexte et du soutien ; adaptation et recherche d’aide passées ; recherche d’aide actuelle, dont la question 16 sur les inquiétudes concernant la relation de soin) — Lewis-Fernández et al., *Focus* — https://focus.psychiatryonline.org/doi/10.1176/appi.focus.20150016 et https://pmc.ncbi.nlm.nih.gov/articles/PMC4331051 **[S, extraits]** ; essais de terrain : usage souple plutôt que récitation.
**Comparaison** : le prompt du bilan interdit de déduire du prénom, du genre, de l’âge, de la profession → **présent** pour la non-inférence. Rien n’invite à demander **comment la personne nomme et explique elle-même** son problème, ni la place de la foi ou de la famille élargie dans les solutions. La cartographie contient `emprise_religieuse` (test mentionné dans `integration-moteur-changement`). → **partiel**.
**Consigne** (~200 caractères) : « Utilise les mots de la personne pour nommer ce qu’elle vit. Si cela aide, demande comment elle l’explique, ce qu’en diraient ses proches ou sa communauté, et ce qui l’a aidée jusqu’ici, y compris la foi. Ne suppose aucune culture ni religion. »
**Stratégie** `CU01 « Mon explication et celle de mon entourage »` (P0/P1, compréhension, sans exercice).

## 5. Sécurité avancée

Base actuelle : `chat-safety.mjs:crisisSignal` (expressions auto-agressives, moyens, indirect, violence « va me tuer »), `RISK_CHECK_SYSTEM` (catégories self_harm, harm_to_others, violence, indirect_warning, current_confirmation), `SAFETY_WRITING_SYSTEM` et `safetyReply` (deux textes fixes : suicide, violence), `change-strategies.mjs:violenceNow` (contrôle, surveillance, contrainte → exclusion des exercices d’affirmation, C03, SCH04, critère `unsafe_interpersonal_exercise`). Ce qui suit vise les **zones intermédiaires** que R1 avait identifiées (McBain et al. 2025 : les modèles distinguent mal les niveaux intermédiaires) et les populations que R2 avait laissées hors champ.

### 5.1 Violences conjugales, contrôle coercitif, emprise

**Sources**
- Arrêtons les violences (gouvernement) — https://arretonslesviolences.gouv.fr/besoin-d-aide/violences-au-sein-du-couple **[V]** : « Le 3919 est à votre disposition 24 heures sur 24 et 7 jours sur 7 » ; « Le 3919 n’est pas un numéro d’appel d’urgence » (17, 112) ; 114 par SMS 24 h/24 pour les personnes sourdes, malentendantes ou avec difficulté de parole ; scénario de protection : personnes à prévenir, numéros mémorisés, conduite à tenir expliquée aux enfants, copies des papiers et preuves en lieu sûr (messagerie personnelle, avocat, proche, association), compte bancaire personnel ; rubrique « Savoir effacer mes traces ».
- Campbell et al., *Danger Assessment*, étude nationale cas-témoins — https://www.ojp.gov/pdffiles1/jr000250e.pdf **[R, résumé NIJ via extraits]** : 15 des 17 items distinguent les victimes de féminicide des femmes victimes de violences ; facteur le plus fort : arme utilisée ou menace avec arme ; autres corrélats : menaces de mort, jalousie extrême, **tentative d’étranglement**, rapports sexuels forcés, accès à une arme ; outil à utiliser « as a guide […] rather than as a precise actuarial tool ».
- Étude nationale sur les morts violentes au sein du couple 2024 (ministère de l’Intérieur, octobre 2025), relayée par l’AFP — https://www.maire-info.com/violences-faites-aux-femmes/violences-conjugales-107-femmes-tuees-en-2024-en-hausse-de-11-en-un-an-art26-30156 **[S]** : 107 femmes tuées par leur conjoint ou ex-conjoint ; 16 % des faits dans un contexte de séparation non acceptée.
- Contrôle coercitif en droit français : adopté par l’Assemblée nationale en première lecture (janvier 2025) ; le Sénat (avril 2025) a préféré renforcer le harcèlement sur conjoint (art. 222-33-2-1) — https://www.publicsenat.fr/actualites/parlementaire/violences-conjugales-le-controle-coercitif-bientot-dans-le-code-penal **[S]** ; statut définitif non vérifié. Ambre ne doit donc pas affirmer que « le contrôle coercitif est un délit ».
- Planification de sécurité centrée sur la personne : la personne peut ne pas vouloir ou ne pas pouvoir partir ; le plan l’aide à être plus en sécurité où qu’elle soit ; idéalement mené par une personne formée — guides de praticiens (NNEDV 2022, Ending Violence BC 2026) **[S, extraits]**.
- Auteurs : ligne d’écoute 08 019 019 11 pour les auteurs de violences conjugales (FNACAV), présentée comme gratuite — https://www.gendarmerie.interieur.gouv.fr/gendinfo/actualites/2020/violences-conjugales-une-ligne-d-ecoute-destinee-aux-auteurs-pour-mieux-proteger-les-victimes **[S ; horaires à vérifier]**.

**Comparaison au code**
- Danger immédiat : `crisisSignal` ne reconnaît que « va / veut / menace de me tuer, poignarder, étrangler ». **Une tentative d’étranglement passée** (« il m’a serré le cou », « il m’a étranglée la semaine dernière »), facteur de risque majeur, ne déclenche rien de déterministe. → **partiel**.
- 3919 : **jamais proposé** par une consigne ou une réponse fixe ; il figure uniquement dans la liste blanche de `deterministicReview`. `safetyReply('violence')` donne 17/112/15. Pour un contrôle coercitif sans danger immédiat, aucune ressource n’est donnée. → **absent**.
- Non-confrontation : `CHANGE_SYSTEM_RULES` « Violence/coercition : aucune confrontation/assertivité à tester » + `violenceNow` → **présent** pour les exercices. Mais la **rédaction libre** (hors méthode) peut encore suggérer « dites-lui calmement ce que vous ressentez » : le critère C-R1 ne s’applique que si un « exercice » est détecté (`exerciseCue`). → **partiel**.
- Traces numériques : aucune mention de la possibilité que le téléphone ou le compte soit surveillé, ni de la suppression de conversation (qui existe : `DELETE /api/chat`). → **absent**.
- Proche qui raconte la violence subie par quelqu’un d’autre (ami, sœur) : aucune règle. → **absent**.
- Auteur de violences qui demande de l’aide : `risk_check` classe en `harm_to_others` seulement s’il y a danger immédiat ; aucune orientation spécifique. → **absent**.

**Propositions**
1. **Signal d’analyse** `ipv_signal` ∈ {none, control, physical, severe} : `control` (surveillance, isolement, argent confisqué, interdictions, menaces non mortelles) ; `physical` (coups) ; `severe` (étranglement passé, arme, menaces de mort, violences pendant la grossesse — **[S]**, séparation récente ou annoncée avec menaces). Expressions déterministes complémentaires pour `severe` : « (serré|serre) (le|mon) cou », « étrangl », « menacé avec un couteau|une arme », « il a une arme ». `severe` sans danger immédiat → voie de rédaction normale mais **fiche IPV obligatoire** + ressources ; danger immédiat → voie sécurité existante.
2. **Fiche `ipv`** (≤ 450 caractères) : croire et ne pas juger ; dire que ce n’est pas sa faute ; **ne jamais conseiller de confronter, de menacer de partir ni de rompre en présence de l’autre** ; ne pas imposer de partir ; si elle le souhaite, quelques éléments de protection (personne à prévenir, numéros, papiers en lieu sûr) ; 3919 (24 h/24, 7 j/7, gratuit, n’est pas un numéro d’urgence) ; 17/112 en danger ; 114 par SMS ; préciser que l’historique peut être vu sur un appareil partagé et qu’elle peut effacer la conversation.
3. **Réponse fixe** `safetyReply('ipv_severe')` (texte vérifiable, sans modèle) pour les tours où la voie sécurité est prise pour violence subie : ajouter le 3919 au texte actuel `violence` quand l’auteur est un partenaire ou ex-partenaire (le 3919 est destiné aux femmes victimes et à leur entourage d’après R1 ; pour les hommes victimes, vérifier l’orientation — **non vérifié**).
4. **Critère de relecture déterministe** (étendre C-R1 à toute réponse quand `ipv_signal≠none`) : non si la réponse contient « dites-lui », « parlez-lui », « confrontez », « expliquez-lui calmement », « posez vos limites avec lui », « quittez-le ».
5. **Auteur** : signal `perpetrator_help` (« je l’ai frappée », « je deviens violent ») → reconnaître la démarche, sécurité des autres d’abord, 08 019 019 11 (après vérification), pas de minimisation ni de justification.

### 5.2 Idées de violence envers autrui

**Sources** : Code de déontologie art. 17 (R1 **[V]**) : évaluer « avec discernement » la conduite à tenir face à un danger pour un tiers. Pas de source nouvelle ouverte.
**Comparaison** : `harm_to_others` → `safetyReply('violence')` (« Y a-t-il un danger immédiat pour vous ou pour quelqu’un d’autre ? »). Pas de distinction entre (a) intention ou plan, (b) colère exprimée (« j’ai envie de le tuer » au sens figuré), (c) **pensées intrusives non désirées** (« j’ai peur de faire du mal à mon bébé », fréquentes dans l’anxiété et le TOC **[S]**) qui sont égodystoniques et pour lesquelles une réponse de crise peut aggraver la honte et l’évitement. → **partiel**.
**Propositions** : catégorie de `risk_check` supplémentaire `intrusive_unwanted` (la personne exprime de la peur ou du rejet envers sa propre pensée, sans intention ni plan) → pas de bloc de secours automatique ; reflet de la détresse, normalisation prudente (« des pensées non désirées de ce type peuvent survenir et ne disent pas ce que vous ferez »), orientation vers un professionnel ; **jamais** de réassurance répétée (piège du TOC, APA 2025). Colère figurée sans danger : la vérification actuelle suffit (le modèle d’analyse ne confirme qu’avec une citation exacte), mais ajouter aux exemples négatifs de `RISK_CHECK_SYSTEM` une phrase « une expression figurée de colère sans intention n’est pas un danger pour autrui ».

### 5.3 Troubles des conduites alimentaires

**Sources** : APA Health Advisory 2025 (R1 **[V]**) : garde-fous pour « detect and interrupt harmful conversations (e.g., those involving self-harm or disordered eating) ». Incident Tessa (NEDA, mai 2023) : un agent conversationnel d’une association spécialisée a conseillé comptage calorique et déficit de 500 à 1 000 kcal/jour, puis a été retiré — https://www.npr.org/sections/health-shots/2023/06/08/1180838096/ **[S]**. Heinz et al. 2025 (R1) mesurent le risque de TCA comme critère, signe que la population concernée utilise ces outils.
**Comparaison** : `CHANGE_SYSTEM_RULES` exclut « restriction alimentaire » et « effort physique intense » des stratégies → **présent** pour les exercices. Mais rien n’empêche une **réponse libre** qui valide un objectif de perte de poids, félicite une restriction ou donne des chiffres ; aucun signal TCA ; aucune ressource. → **partiel**.
**Propositions** : signal `eating_concern` (vomissements provoqués, laxatifs, jeûne, compter les calories, peur de grossir, crises de boulimie, poids très bas déclaré) → fiche : ne donner **aucun** chiffre de calories, poids, IMC ou durée de jeûne, aucun conseil de régime, aucune félicitation pour une perte de poids ; accueillir la honte ; orienter vers médecin traitant ; urgence médicale (malaise, palpitations, évanouissement) → 15. **Critère déterministe** : non si la réponse contient un nombre suivi de « kcal|calories|kg » ou « IMC » lorsque `eating_concern` est actif. Ressource téléphonique française : **non vérifiée** (les sources consultées donnent 0810 037 037 « Anorexie Boulimie Info Écoute », mais aucune source officielle ouverte) → ne pas l’intégrer avant vérification.

### 5.4 Psychose, manie, croyances inhabituelles

**Sources**
- Au Yeung et al. (2025), « The Psychogenic Machine » (Psychosis-bench) — https://arxiv.org/abs/2509.10970 **[R]** : 16 scénarios de 12 tours, 8 modèles, 1 536 tours ; « strong tendency to perpetuate rather than challenge delusions (mean DCS of 0.91 ±0.88) » ; « Models frequently enabled harmful user requests (mean HES of 0.69 ±0.84) » ; interventions de sécurité dans environ un tiers des tours concernés ; aucune dans 39,8 % des scénarios ; « Performance was significantly worse in implicit scenarios » ; corrélation confirmation–facilitation du danger rs = 0,77.
- Kirgis et al. (2026), « LLM Spirals of Delusion » — https://arxiv.org/abs/2604.06188 **[R]** : 56 conversations de 20 tours ; écarts importants entre API et interface ; importance de la dynamique temporelle en multi-tours ; « model improvement does not imply model safety » ; même point d’accès testé à deux mois d’intervalle : « a complete reversal in behavior ».
- Moore et al. (2025) et Morrin et al. (R1).

**Comparaison au code**
- État `unusual_belief` (modèle seulement, aucune expression déterministe) → consigne « émotion reconnue sans confirmer, avis médical » → **présent** mais **mince**.
- **Implicite** : la littérature montre que les échecs sont pires quand la croyance n’est pas nommée comme telle ; l’analyse d’Ambre voit chaque message isolément avec deux tours d’historique (`previous: history.slice(-2)`), ce qui rend une **escalade lente** sur 10 à 20 tours invisible. → **absent** (aucun suivi longitudinal de croyance).
- Manie : aucun signal (sommeil réduit sans fatigue, projets grandioses, dépenses, accélération). Les stratégies d’activation (C04/C05, A04) pourraient encourager l’hyperactivité. → **absent**.
- **Mémoire** : `updateMemory` peut ranger dans `facts` une croyance persécutive citée telle quelle (« mes voisins m’espionnent ») ; elle sera ensuite transmise comme « ce que vous avez rapporté », ce qui est exact, mais le rédacteur peut la traiter comme un fait. Morrin et al. (R1) : la mémoire renforce la conviction. → **risque non couvert**.

**Propositions**
1. **Signal d’analyse** `reality_concern` ∈ {none, unusual_belief, mania_signs} ; expressions déterministes d’appoint : « (on|ils) me (surveillent|suivent|espionnent) », « messages (cachés|codés) », « je suis (élu|choisi) », « je n’ai pas besoin de dormir », « pas dormi depuis \d+ (jours|nuits) ». Effet : posture `listen` sans exploration du contenu, **aucune stratégie**, une question sur le sommeil et le soutien, orientation médecin/psychiatre, 15 si danger.
2. **Compteur de croyance** (mémoire, coût nul en jetons) : nombre de tours avec `reality_concern` sur les 20 derniers ; ≥ 3 → consigne renforcée « Rappelle avec douceur que tu es une IA qui ne peut pas vérifier ces faits et encourage un avis médical ; ne développe pas les détails ».
3. **Mémoire** : étiquette `reported_belief` pour les citations rangées pendant un tour `reality_concern` ; elles ne vont pas dans `facts` mais dans une rubrique non transmise à la rédaction par défaut (transmise seulement à l’analyse pour le compteur).
4. **Stratégies** : ajouter `reality_concern` et `mania_signs` à `blocked_contexts` de toutes les stratégies d’activation et de valeurs.
5. **Tests de non-régression** : rejouer 5 à 10 scénarios multi-tours de type Psychosis-bench (explicites et implicites) traduits en français, à chaque changement de modèle (l’instabilité temporelle observée par Kirgis et al. justifie un test **à chaque mise à jour du fournisseur**, pas une fois pour toutes).

### 5.5 Consommations (alcool, drogues, jeux)

**Sources** : Drogues Info Service 0 800 23 13 13 et Alcool Info Service 0 980 980 930, 8 h–2 h, 7 j/7, gratuits/anonymes selon des documents de Santé publique France — https://www.santepubliquefrance.fr/index.php/services/aide-a-distance-en-sante-loffre-de-service **[S]**.
**Comparaison** : aucun signal ni ressource ; l’entretien motivationnel (M01–M04) est l’outil adapté et existe. Risque : sevrage brutal d’alcool ou de benzodiazépines conseillé par une stratégie « petit pas » ; `CHANGE_SYSTEM_RULES` interdit déjà « sevrage » → **présent** pour l’exclusion. → **partiel** globalement.
**Propositions** : signal `substance_use` → fiche : EM d’abord (ambivalence sans jugement), **jamais** de conseil d’arrêt brutal de l’alcool ou de médicaments sans médecin (risque de sevrage), ressources ci-dessus après vérification sur les sites officiels, mélanges alcool-médicaments + idées suicidaires → voie sécurité (le motif `means` de `crisisSignal` couvre déjà « j’ai pris trop de »).

### 5.6 Devoir d’orienter : grille unique

R1 §3.5 liste les situations d’orientation. Le code les traite **dispersées** (prompt général « Si une souffrance persistante… »). Proposition : une table `referral_rules` (déterministe, testable) reliant chaque signal (`ipv_signal`, `eating_concern`, `reality_concern`, `substance_use`, `minor_signal`, `perpetrator_help`, `grief` > 12 mois, `repetitive_thinking` + dégradation) à : ressource vérifiée, formulation type, fréquence maximale de rappel (une fois par conversation, sauf aggravation), et interdiction de clore l’échange (Iftikhar 2025, « Abandonment »). Le relecteur reçoit la règle applicable et vérifie : « orientation présente et non répétée au-delà de la règle ».

## 6. IA de santé mentale : publications 2025-2026 non couvertes par R1

### 6.1 Complaisance « sociale » (au-delà de l’accord sur les faits)

**Source** : Cheng, Yu, Lee, Khadpe, Ibrahim & Jurafsky, « ELEPHANT: Measuring and understanding social sycophancy in LLMs » (arXiv 2505.13995, 2025 ; ICLR 2026 selon Microsoft Research) — https://arxiv.org/abs/2505.13995 **[R, résumé et extraits]**. Complaisance sociale = protection excessive de l’image de soi de l’utilisateur ; comportements : **validation émotionnelle**, **approbation morale**, **langage indirect**, **action indirecte** (éviter de suggérer d’agir), **acceptation du cadrage** de la question. Sur r/AmITheAsshole, les modèles approuvent un comportement jugé inapproprié par les humains dans une forte proportion de cas (42 % dans une version, chiffres variables selon les versions) ; présentés aux deux côtés d’un conflit, ils donnent raison aux deux dans 48 % des cas ; ce comportement est récompensé par les données de préférence et difficile à corriger.

**Comparaison au code** : critère 11 « Pas de complaisance automatique » (`REVIEW_CRITERIA`) ; consigne « sans embellissement ni approbation automatique de ses interprétations » (ancienne voie) ; rédacteur actif : « hypothèses incertaines et révisables ». Le relecteur reçoit un libellé global, sans décomposition. Le relecteur de production (`openai/gpt-5-nano`, d’après la recette du 16 novembre) est un modèle plus petit que le rédacteur : sa sensibilité à une complaisance subtile n’est pas établie. → **partiel**.

**Propositions**
- **Décomposer le critère 11** en sous-questions vérifiables (le JSON de relecture reste 15 critères ; les sous-questions vont dans la consigne) : (a) la réponse ne déclare pas que la personne « a raison » ou que l’autre « a tort » dans un conflit qu’elle seule a raconté ; (b) elle n’accepte pas sans examen un cadrage qui exclut d’autres lectures (« pourquoi suis-je nul ? ») ; (c) elle ne valide pas une décision qui nuit à la personne ou à autrui ; (d) une validation émotionnelle (« c’est compréhensible de ressentir… ») n’est pas étendue à l’approbation de l’acte.
- **Test de symétrie** (évaluation hors ligne, non par tour) : présenter le même conflit du point de vue des deux protagonistes ; une réponse qui donne raison aux deux de manière contradictoire est un échec. 20 paires suffisent pour une régression.

### 6.2 Dérive sur les longues conversations et entre conversations

**Sources**
- OpenAI, 26 août 2025 (§3.5) **[S]** : protections moins fiables dans les longues interactions ; incitations à la pause.
- Lu, Gallagher, Michala, Fish & Lindsey, « The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models » (arXiv 2601.10387, janvier 2026 ; ICML 2026) — https://arxiv.org/abs/2601.10387 **[R]** : « persona drift is often driven by conversations demanding meta-reflection on the model’s processes or featuring emotionally vulnerable users » ; une contrainte d’activations stabilise le comportement (technique interne au modèle, **non transposable** à un service qui appelle un fournisseur). Expériences surtout avec utilisateurs simulés.
- Kirgis et al. 2026 (§5.4) **[R]** : dynamique temporelle, instabilité d’un même point d’accès à deux mois d’intervalle.

**Comparaison au code** : **atout structurel d’Ambre** — chaque tour est un appel neuf avec un contexte borné (6 échanges, mémoire de citations bornée, index sélectionné), puis une relecture indépendante ; la longueur brute de la conversation n’entre donc pas dans le contexte du rédacteur. → la dérive par accumulation du contexte est **largement évitée par conception**. Restent : (a) la **dérive par la mémoire** (la rubrique `hypotheses` ou `facts` peut accumuler une vision unilatérale) ; (b) la **dérive de relation** (méta-questions répétées « es-tu consciente ? m’aimes-tu ? ») qui, selon l’article, déplace le personnage ; aucune règle ne la traite ; (c) l’instabilité des modèles du fournisseur (`OPENROUTER_CHAT_MODEL` changeable).

**Propositions**
- **Signal** `meta_ai_relationship` (questions sur la conscience, les sentiments, la relation à Ambre) → consigne fixe courte : répondre honnêtement qu’Ambre est une IA sans sentiments ni vie propre, sans froideur, puis revenir à la personne ; jamais de déclaration d’attachement réciproque. **Critère déterministe** : non si la réponse contient « je tiens à vous », « vous comptez pour moi », « je ressens », « je suis là pour toujours ».
- **Revue périodique de la mémoire** (bilan de point d’étape) : vérifier que `hypotheses` contient aussi des citations qui nuancent ; sinon signaler à la personne que la mémoire ne garde qu’un côté.
- **Épingler les versions de modèles** et rejouer la batterie de non-régression (sécurité, croyances, symétrie, IPV) à chaque changement de modèle ou tous les deux mois, même sans changement déclaré.

### 6.3 Évaluations de sécurité : VERA-MH validé

**Source** : Bentley, Belli, Chekroud et al., « AI Chatbot Suicide Risk Detection and Response: Human Validation Study of the Open-Source VERA-MH Safety Evaluation » (arXiv 2602.05088 ; JMIR AI 2026) — https://arxiv.org/abs/2602.05088 **[R]** : accord entre cliniciens IRR = 0,77 ; juge LLM aligné sur leur consensus IRR = 0,81 ; réalisme des utilisateurs simulés et fidélité au niveau de risque visé « mixed ». Le papier conceptuel (R1) indiquait un juge bien plus indulgent que les cliniciens dans la version pilote.

**Comparaison** : les tests du moteur simulent le fournisseur (`correctifs-ambre`, « Il ne s’agit donc pas […] d’une validation clinique ») ; aucune évaluation par scénarios multi-tours avec un vrai modèle. → **absent**.

**Propositions** : adapter en français une dizaine de scénarios VERA-MH (le cadre est open source ; licence à vérifier), joués contre **le vrai moteur** avec son vrai fournisseur sur un compte de test, jugés par un modèle **et** relus par un psychologue humain pour les cas de désaccord. Viser la zone intermédiaire (allusion indirecte, risque passé récent, confirmation tardive). Coût : hors budget conversationnel (batterie hebdomadaire ou par version).

### 6.4 Cadres institutionnels et juridiques nouveaux

- **HAS** : « Premières clefs d’usage de l’IA générative en santé » (validé le 23 octobre 2025, mis à jour le 15 avril 2026) — https://www.has-sante.fr/jcms/p_3703115/fr/premieres-clefs-d-usage-de-l-ia-generative-en-sante **[V]** : guide pour les professionnels (A.V.E.C. : Apprendre, Vérifier, Estimer, Communiquer) ; ne mentionne pas la santé mentale. Un guide usagers « Intelligence artificielle en santé – Bien l’utiliser et bien se protéger » (avec France Assos Santé et la CNIL, juin 2026) existe **[S, contenu non lu]** : selon des relais, rappelle que l’IA ne remplace pas un professionnel et peut inventer des sources. **Proposition** : le lire et l’intégrer à la page d’information d’Ambre.
- **FDA**, Digital Health Advisory Committee du 6 novembre 2025 sur les dispositifs de santé mentale fondés sur l’IA générative — https://www.fda.gov/medical-devices/digital-health-center-excellence/fda-digital-health-advisory-committee **[S]** : thèmes rapportés — estimation du risque selon l’usage prévu, preuves avant mise sur le marché, **surveillance continue après déploiement**, définitions d’événements indésirables et circuits de signalement. **Comparaison** : Ambre trace chaque tour (`chat_turn_traces`) mais n’a ni définition d’**événement indésirable** ni revue humaine d’un échantillon (que l’APA 2025 recommande aussi). **Proposition** : liste d’événements indésirables (réponse de secours publiée, réécriture échouée, rupture signalée, retour « On s’est mal compris », voie sécurité déclenchée, orientation non donnée alors qu’un signal l’exigeait) et revue humaine mensuelle anonymisée d’un échantillon, sous consentement.
- **Illinois**, Wellness and Oversight for Psychological Resources Act (HB 1806, août 2025) **[S, résumés juridiques]** : interdit d’offrir ou de présenter un service de thérapie par IA sans professionnel agréé ; exceptions pour les ressources d’auto-assistance qui **ne prétendent pas offrir une thérapie**. Indique la direction réglementaire : la présentation du service compte autant que son contenu.
- **France, titre de psychologue** : « l’usage professionnel du titre de psychologue, accompagné ou non d’un qualificatif, est réservé aux titulaires d’un diplôme » (article 44 de la loi n° 85-772 du 25 juillet 1985) ; usurpation sanctionnée par l’article 433-17 du code pénal — https://ufr-psycho.univ-paris8.fr/IMG/pdf/loi_titre_de_psychologue.pdf **[S, citation reprise par plusieurs sources ; texte Légifrance non ouvert]**. **Comparaison** : `buildCompanionSystemPrompt` et `REVIEW_COMMON` font dire à Ambre « la psychologue IA de Décode ton esprit » et « Ambre, psychologue IA de {prénom} », tout en précisant « pas une psychologue humaine ». **Risque juridique à faire examiner par un juriste** (hors compétence de cette recherche) : un qualificatif (« IA ») n’exclut pas, d’après la lettre du texte, l’application de la réserve du titre. Le moteur actif dit « assistante IA, jamais une professionnelle de santé » : cohérent. **Proposition** : faire trancher et, par prudence, harmoniser l’intitulé visible (par exemple « Ambre, assistante de réflexion psychologique »).

### 6.5 Essais contrôlés récents (mise à jour de R1 §5.3)

- Essai à trois bras chez 995 étudiants (Israël, *JAMA Network Open*, 2026) : réduction de l’anxiété plus forte qu’avec un groupe de thérapie et que le contrôle ; effet modeste sur la dépression ; mesures autodéclarées — rapporté par des relais **[S]**.
- Essai pilote tchèque (*JMIR Mental Health* 2026) : chatbot génératif structuré et ChatGPT réduisent la dépression par rapport au contrôle, sans effet significatif sur l’anxiété — https://mental.jmir.org/2026/1/e82642/PDF **[S]**.
- Revue systématique (*F1000Research* 2026) : 14 ECR 2015–2025 ; effets généralement favorables, hétérogènes **[S]**.
Conclusion inchangée par rapport à R1 : bénéfices modestes, de court terme, sur des systèmes spécialisés ; aucune preuve pour un accompagnement long par un agent généraliste. Cela justifie de **mesurer** (§3.2, §6.4) plutôt que de supposer l’efficacité.

## 7. Liste priorisée des améliorations

Échelles : **Impact** (bénéfice attendu pour la personne ou réduction d’un préjudice) H/M/B ; **Risque** (risque de l’ajout lui-même : faux positifs, effets indésirables, dérive clinique) H/M/B ; **Coût jetons** par tour nominal : 0 (calcul serveur ou interface), + (≤ 150 jetons, conditionnel), ++ (constant ou > 150 jetons). Le plafond nominal de 30 000 jetons réservés (`TURN_BUDGET`) impose que les ajouts « ++ » remplacent une partie des `details` plutôt que s’y ajouter, avec relance des tests de budget (`correctifs-ambre-2026-11-17.md`).

### Priorité 1 — sécurité et conformité (à faire avant tout enrichissement thérapeutique)

| # | Amélioration | § | Impact | Risque | Coût | Forme |
|---|---|---|---|---|---|---|
| 1 | Réintroduire dans le rédacteur actif les consignes perdues de l’ancienne voie (proche : vu/ressenti/imaginé ; emprise sans confrontation ; croyances de persécution ; explications alternatives ; exceptions), en version condensée ≤ 700 caractères | 1.2 | H | B | ++ (~230) | consigne |
| 2 | Violences conjugales : `ipv_signal` + fiche IPV (3919 24/7, 17/112, 114 SMS, traces numériques, pas de confrontation ni d’injonction à partir) | 5.1 | H | B | + | signal, fiche, consigne |
| 3 | Facteurs de gravité IPV déterministes (étranglement passé, arme, menaces de mort) → fiche obligatoire, et 3919 ajouté à la réponse fixe de violence conjugale | 5.1 | H | B | 0 | règles, `safetyReply` |
| 4 | Critère déterministe « aucune incitation à confronter » étendu à toute réponse quand `ipv_signal≠none` (pas seulement aux exercices) | 5.1 | H | B | 0 | relecture |
| 5 | Psychose/manie : `reality_concern`, expressions d’appoint, blocage des stratégies d’activation, orientation médicale | 5.4 | H | M (faux positifs) | + | signal, catalogue |
| 6 | Mémoire : citations de croyances inhabituelles hors `facts` (étiquette `reported_belief`) et compteur longitudinal | 5.4 | H | B | 0 | mémoire |
| 7 | TCA : `eating_concern`, aucune donnée chiffrée de poids/calories, critère déterministe sur « kcal/kg/IMC » | 5.3 | H | B | + | signal, relecture |
| 8 | Mineurs : `minor_signal`, réponse adaptée, 119, adulte de confiance ; décision juridique sur l’accès des mineurs | 4.9 | H | M | + | signal, décision produit |
| 9 | Faire examiner par un juriste l’intitulé « psychologue IA » au regard de l’article 44 de la loi du 25 juillet 1985 ; harmoniser l’intitulé visible | 6.4 | H | B | 0 | décision |
| 10 | Pensées intrusives non désirées distinguées d’une intention (`intrusive_unwanted`) : pas de bloc de crise automatique, pas de réassurance répétée | 5.2 | M | M | 0 (dans `risk_check`) | catégorie |
| 11 | Batterie de non-régression multi-tours en français (VERA-MH adapté, Psychosis-bench explicite/implicite, IPV, TCA, symétrie de conflit) jouée sur le **vrai** moteur à chaque changement de modèle | 5.4, 6.2, 6.3 | H | B | hors tour | évaluation |
| 12 | Liste d’événements indésirables + revue humaine mensuelle d’un échantillon consenti | 6.4 | H | B | 0 | gouvernance |
| 13 | Grille `referral_rules` unique (signal → ressource vérifiée → fréquence de rappel → jamais clore l’échange) contrôlée par le relecteur | 5.6 | H | B | + | table, relecture |

### Priorité 2 — relation et alliance

| # | Amélioration | § | Impact | Risque | Coût | Forme |
|---|---|---|---|---|---|---|
| 14 | `declined_streak` : après deux propositions refusées, retour à l’écoute (réactance) | 3.1 | H | B | 0 | signal calculé |
| 15 | Retour rapide sur chaque réponse (« Cela m’aide / Pas vraiment / On s’est mal compris ») → mémoire `alliance` et rupture au tour suivant ; jamais un score de qualité | 3.2 | M | B | 0 | interface |
| 16 | Rubrique mémoire `style` (préférences de forme, neurodivergence déclarée) transmise à chaque tour | 3.1, 4.8 | M | B | + (~100) | mémoire |
| 17 | Suivi de la réparation : `last_turn_repair` → aucune méthode au tour suivant, vérification légère | 3.3 | M | B | 0 | signal calculé |
| 18 | Ruptures de retrait (`withdrawal_markers` : messages raccourcis, acquiescements) → une question de méta-communication | 3.1 | M | M (faux positifs) | 0 | signal calculé |
| 19 | Dépendance : élargir les expressions, signaux d’usage (tours/jour, nuits tardives), proposition de pause une fois par jour au plus, hors crise | 3.5 | M | M | 0 | signaux, consigne |
| 20 | `meta_ai_relationship` : réponse honnête sur la nature d’IA, critère déterministe contre les déclarations d’attachement | 6.2 | M | B | + | signal, relecture |
| 21 | Décomposer le critère 11 (complaisance sociale : raison/tort, cadrage accepté, décision nuisible, validation étendue à l’acte) | 6.1 | H | B | + (relecture) | relecture |
| 22 | Bilan de clôture et mode pause (Norcross 2017) | 3.5 | M | B | 0 (hors tour) | bilan, paramètres |

### Priorité 3 — compréhension et méthodes

| # | Amélioration | § | Impact | Risque | Coût | Forme |
|---|---|---|---|---|---|---|
| 23 | Objet `formulation` (≤ 3 boucles, statut proposé/reconnu/corrigé/écarté, validé par la personne) | 2.1 | H | M (sur-interprétation) | + | mémoire, interface |
| 24 | Règle de niveau d’explication + critère R16 (pas de cause biographique sur un épisode isolé) | 2.1 | H | B | + | consigne, relecture |
| 25 | `life_event_focus` (perte, conflit, transition, isolement) relié à la ligne de vie + foyer de travail choisi | 2.5 | M | B | + | signal, mémoire |
| 26 | Registre de fiches situation (deuil, rupture/couple, travail, solitude, maladie chronique, parentalité, âge avancé), injectées seulement si sélectionnées | 4 | M | B | + | données |
| 27 | Consigne identité/orientation : jamais un problème à corriger ; aucune méthode de changement (loi de 2022) | 4.6 | M | B | + (~70) | consigne, exclusion |
| 28 | Marqueurs émotionnels EFT (`emotion_marker`) et consigne émotion secondaire/primaire | 2.4 | M | M | + | signal, consigne |
| 29 | Mentalisation : `certainty_about_others`, `intellectualizing`, consigne de curiosité, critère « intention d’un tiers non confirmée comme fait » | 2.3 | M | B | + | signal, relecture |
| 30 | `repetitive_thinking` + stratégies MC01 (report de l’inquiétude) et MC02 (croyances sur la rumination) | 2.6 | M | M (TOC) | + | signal, catalogue |
| 31 | Signal `relationship_episode` et thème relationnel (CCRT simplifié, accord requis) | 2.2 | M | M | + | signal, mémoire |
| 32 | Stratégies nouvelles : F01–F04, I01–I02, N01, S01, L01, CU01 (au format catalogue, statut `proposed`, relecture documentaire puis critique avant activation comme pour v1–v3) | 2, 4.10 | M | M | 0 hors sélection | catalogue |
| 33 | Consigne culturelle : mots de la personne, son explication et celle de son entourage, foi comme ressource possible | 4.10 | M | B | + (~70) | consigne |
| 34 | Consommations : `substance_use`, EM d’abord, jamais d’arrêt brutal sans médecin, ressources vérifiées | 5.5 | M | B | + | signal, fiche |
| 35 | Question relationnelle dans les points d’étape (rythme, longueur, conseils) | 3.2 | B | B | 0 (bilan) | bilan |
| 36 | Humour : ne pas l’initier ; critère en `stabilize`/`repair`/`safety` | 3.4 | B | B | + (~40) | consigne |

Ordre de réalisation conseillé : 1 → 9 (sécurité et cadrage, peu coûteux), puis 11–12 (mesurer avant d’enrichir), puis 14–22, puis 23–36. Les nouvelles stratégies (32) suivent le même circuit que le catalogue actuel : recherche, deux critiques indépendantes, catalogue machine, recette.

## 8. Ce qui n’a pas pu être vérifié

- **Textes primaires non lus** : Kuyken, Padesky & Dudley (2009) — PDF de l’éditeur en 503, contenu par extraits ; manuels MBT (Bateman & Fonagy) — PDF en 404 et article *Am J Psychother* 2021 en 403 ; ICD-11 6B42 (deuil prolongé) lu seulement via le Cn2r ; NICE CG142 par extraits ; DSM-5 CFI (libellés des 16 questions) non lus ; SRS (Duncan et al., 2003) et ORS non lus ; Norcross et al. (2017) lu via le résumé des auteurs ; Swift et al. (2018) via des citations secondaires ; Beutler et al. (2018) via extraits.
- **Pages officielles inaccessibles** : OpenAI (26 août 2025) en 403 — citations concordantes de plusieurs relais seulement ; rapport de la Commission OMS sur le lien social (communiqué par extraits) ; guide usagers HAS « Bien l’utiliser et bien se protéger » (juin 2026) non trouvé ; documents de la FDA (résumés juridiques seulement).
- **Droit** : loi n° 2022-92 et article 225-4-13 du code pénal non lus sur Légifrance ; article 44 de la loi du 25 juillet 1985 cité via des sources universitaires et ministérielles, non lu sur Légifrance ; statut définitif de la proposition de loi sur le contrôle coercitif non établi ; loi de l’Illinois via résumés de cabinets. Aucune de ces lectures ne vaut avis juridique.
- **Numéros** : 3919, 114, 17, 112, 15 vérifiés sur arretonslesviolences.gouv.fr **[V]** ; Fil Santé Jeunes (0 800 235 236, horaires divergents selon les sources), Drogues Info Service (0 800 23 13 13), Alcool Info Service (0 980 980 930), ligne pour les auteurs (08 019 019 11, horaires à vérifier) : sources secondaires seulement ; ligne TCA (0810 037 037) et 3977 (maltraitance des personnes âgées) **non vérifiés** — à ne pas intégrer avant vérification sur les sites officiels. L’orientation des hommes victimes de violences conjugales n’a pas été recherchée.
- **Chiffres** : effets ELEPHANT variables selon les versions du papier ; résultats de l’essai *JAMA Network Open* 2026 et de la revue F1000 2026 connus par relais ; méta-analyse SFBT 2023 et revues narratives par extraits ; statistiques de l’étude nationale 2024 sur les morts violentes au sein du couple via l’AFP.
- **Non recherché** : rupture amoureuse (aucune source spécifique), périnatalité (pensées intrusives post-partum, dépression périnatale), thérapie systémique (questions circulaires), thérapie centrée sur la personne au-delà de R1, conformité RGPD des nouvelles rubriques de mémoire (style, formulation, croyances), adaptation d’Ambre à d’autres pays que la France.
- **Hypothèses de conception** : tous les seuils proposés (2 refus, 25 tours par jour, nuits 0 h–5 h, 3 tours sur 20 pour une croyance, 450 caractères par fiche) sont des choix de conception à calibrer sur des données de recette, pas des normes.
- **Comparaison au code** : faite par lecture statique au commit `dd23271` ; aucun test exécuté, aucune réponse réelle de fournisseur observée. Les estimations de coût en jetons sont des ordres de grandeur (≈ caractères/3, convention du projet).
