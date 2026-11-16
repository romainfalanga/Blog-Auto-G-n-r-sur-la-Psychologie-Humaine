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

Le prompt riche `psychology-framework.mjs:buildCompanionSystemPrompt` (explications alternatives, situation→interprétation→émotion→action, exceptions, réciprocité, emprise sans confrontation symétrique, croyances inhabituelles) et `JOURNEY_GUIDANCE` sont utilisés par l’**ancienne voie** (`chat-backend.mjs`, ligne ~396). Le moteur actif (`createAmbreEngine`, ligne ~797) reçoit un prompt de rédaction plus court écrit dans `work` et le contrat `engineContract`. Plusieurs consignes de fond présentes dans `buildCompanionSystemPrompt` ne sont donc **pas** transmises au rédacteur actif — à vérifier par un test de non-régression avant toute affirmation contraire. Cette observation conditionne plusieurs comparaisons ci-dessous (« présent dans l’ancienne voie seulement »).

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

