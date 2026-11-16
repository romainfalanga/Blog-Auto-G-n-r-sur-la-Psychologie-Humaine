# Revue critique d'exhaustivité de la Cartographie — 18 novembre 2026

Revue en lecture seule (aucun fichier existant modifié, aucun commit, aucun déploiement). Rédigée au fil de l'eau ; les sections marquées « en cours » sont complétées progressivement.

## 0. Périmètre et méthode

- Jeu examiné : `data/cartographie.json` (schéma 2) au commit `dd23271`, identique à `63e949f`/`e3428cc` pour les données, le moteur et le Worker (seul changement d'interface depuis : défilement des onglets mobiles, sans effet sur cette revue) — 59 questions, 6 thèmes, 109 cartes actives, 3 identifiants historiques, 493 liens, 260 sources, 11 ressources. Comptes recalculés par parseur JSON, identiques à `docs/integration-cartographie-2026-11-17.md`.
- Code lu : `static/js/cartographie.js` (moteur `compute` + ancien `mount` non utilisé par la page), `static/js/cartographie-ui.js` (interface réelle), `journal/src/cartographie.mjs` (calcul Worker, résumé transmis à Ambre), `layouts/_default/cartographie.html`.
- Documents lus : intégration et relecture finale du 17/11, plan d'exécution Ambre/Cartographie, lots de recherche (parcourus), registre `docs/audit-cartographie-correspondances-2026-11-17.json` (parcouru par parseur).
- Vérifications externes : recherches web ciblées et ouverture de sources ; chaque référence citée ci-dessous comme « ouverte » l'a été pendant cette revue. Aucune référence n'est proposée sans avoir été vue.
- Rappel : les décisions explicites du propriétaire (liste unique triée par nombre de vécus, retrait des introductions, retrait de l'encadré 3114 et du bouton de suppression) sont signalées comme telles ; la revue en mesure les effets sans les annuler d'autorité.

## 1. Constat d'ensemble (à lire d'abord)

Le graphe est sérieux, prudent et très documenté ; la plupart des défauts relevés ne sont pas des erreurs de fond mais des effets d'architecture. Les quatre problèmes les plus importants :

1. **Le tri public récompense la quantité de liens, pas la preuve.** L'interface (`cartographie-ui.js`, `renderResults`) trie par nombre de vécus cochés reliés à la carte, *tous types de liens confondus* (modèles cliniques, définitions et contextes compris). Le moteur calcule pourtant des niveaux (`convergence`, `piste`, `theorique`) que l'interface n'affiche plus (classe CSS seulement) ; Ambre, elle, reçoit l'ordre par familles de preuves. **La personne et Ambre ne voient donc pas le même classement.** Effets mesurés plus bas (§ 4).
2. **Le volume est illisible dès qu'une violence est cochée** : « Violences psychologiques » seule ouvre 49 cartes ; un profil de huit vécus de violence en ouvre 81 ; tout cocher en ouvre 109. Trois questions (violences physiques, sexuelles, psychologiques) portent 151 des 493 liens (31 %).
3. **Les 11 ressources protectrices existent dans les données mais ne sont plus proposées à l'écran** : aucune question sur les expériences positives (PCE/BCE). L'outil ne voit l'enfance que par ses manques.
4. **Plusieurs vécus fréquents et bien documentés sont presque muets** : précarité (3 liens, aucun vers dépression/anxiété), exil/guerre et migration (0 lien), secrets familiaux (0), témoin de violences entre adultes (7 liens dont 4 non empiriques).

## 2. Exhaustivité côté vécus (59 questions)

### 2.1 Couverture des instruments de référence

| Instrument | Domaines | Présents dans la Cartographie | Manques ou écarts |
| --- | --- | --- | --- |
| ACE CDC (Felitti 1998, 10 items) | violences physiques, psychologiques, sexuelles ; négligences ; violences conjugales ; addiction, trouble psychique, incarcération ; séparation | tous (`ace:true` sur 10 questions, séparation en paire exclusive) | aucun |
| ACE-IQ OMS | ACE + harcèlement, violence communautaire, violence collective | harcèlement, `violence_quartier`, guerre dans `exil` | la violence collective est fondue dans « exil » (voir 2.3) |
| CTQ-SF | 5 sous-échelles abus/négligences | les 5 | l'échelle de minimisation/déni n'a pas d'équivalent (normal) |
| MACE (Teicher & Parigger 2015, PLOS ONE, PMC4340880, ouvert) | 10 sous-échelles, **exposition année par année de 1 à 18 ans** | 9/10 | **témoin de violences sur la fratrie** (reporté par le lot 8, légitimement) ; MACE range aussi dans la maltraitance non verbale l'enfermement (placard, cave), les « secrets importants cachés » et les responsabilités d'adulte ; **aucune donnée d'âge, de durée ni d'auteur** n'est recueillie par la Cartographie |
| CECA-Q / CECA (Bifulco) | antipathie, négligence, violences, renversement des rôles, perte/séparation | présents (antipathie ≈ critique + violences psychologiques ; renversement ≈ parentification) | l'auteur (père, mère, autre adulte) n'est jamais demandé |
| PCE (Bethell 2019, JAMA Pediatr, PMID 31498386, ouvert) – 7 items ; BCE (Narayan 2018, Child Abuse Negl, PMID 28992958, ouvert) – 10 items | expériences positives | **aucune question affichée** ; 11 ressources dans les données, sans contrôle dans l'interface (`cartographie-ui.js` conserve seulement les valeurs déjà stockées) | manque majeur : Bethell trouve 72 % de cotes en moins de dépression/mauvaise santé mentale pour 6–7 PCE vs 0–2, à niveau d'ACE comparable (transversal) |
| Expanded/Philadelphia ACE | discrimination, quartier, harcèlement, placement, violence communautaire | présents | placement après la petite enfance (voir 2.3) |

### 2.2 Thèmes demandés par le propriétaire

| Thème | Statut | Commentaire |
| --- | --- | --- |
| Discrimination | présent | mais un exemple (« rejet familial lié à l'orientation ou à l'identité ») double l'exemple de `abus_emotionnel` : même vécu, deux cases |
| Harcèlement, cyberviolence | présent (exemple en ligne) | OK ; la cyberviolence n'a pas de case propre, ce qui est défendable (Kowalski 2014, Psychol Bull, PMID 24512111, ouvert : chevauchement fort avec le harcèlement classique) |
| Violence communautaire | présent (`violence_quartier`, 2 liens limités) | OK |
| Pauvreté, insécurité alimentaire, logement | fusionnés dans `precarite` | défendable pour la longueur, mais **3 liens seulement** et aucun vers dépression/anxiété (voir § 7) |
| Migration, placement, adoption | présents | `migration` 0 lien (cohérent, à dire) ; placement tardif mal couvert |
| Deuil | présent | le décès d'un parent par suicide figure en exemple, bien |
| Maladie d'un parent / de soi / de la fratrie | présents (3 cases) | le handicap de l'enfant lui-même n'apparaît que dans « discrimination » ; la maladie psychique d'un frère/sœur n'est nulle part |
| Handicap | partiel | voir ligne précédente |
| Fratrie | présent (violences, comparaisons, maladie) | manque « témoin de violences sur la fratrie » |
| Rejet lié à l'identité | présent en exemples (2 cases) | doublon |
| Guerre et catastrophes | guerre dans `exil`, catastrophe dans `traumatisme_ponctuel` | découpage ambigu (voir 2.3) |
| Emprise sectaire | présent (`emprise_religieuse`) | 3 liens, tous théoriques |
| Contrôle psychologique | en exemples de `controle` | **mal découpé** (voir 2.3) |
| Invalidation émotionnelle, surprotection, parentification | présents | parentification éclatée en 4 cases (voir 2.4) |
| Violences dans les premières relations amoureuses (adolescence) | **absent** | voir proposition § 9 |

### 2.3 Découpages à revoir

1. **`controle` mélange deux construits opposés.** Le libellé et la description (« décisions imposées, peu de place à ses avis, punitions pour le désaccord ») décrivent un contrôle comportemental autoritaire ; les exemples ajoutés (« culpabilisation », « retrait affectif ou intrusion dans les pensées ») décrivent le contrôle psychologique. Les résumés des liens portent presque tous sur le **contrôle psychologique** (Bleys 2018, PMID 29696640, ouvert : contrôle psychologique *centré sur la performance*). La littérature distingue ces dimensions ; un encadrement comportemental (suivi, règles) n'est pas un facteur de risque comparable. Recommandation : scinder en « Contrôle psychologique (culpabiliser, retirer son affection, s'immiscer dans les pensées) » et « Autorité stricte et punitions du désaccord », ne transférer aux deux que les liens dont la source mesure la dimension.
2. **`exil` réunit trois expositions différentes** (exil forcé, persécution, guerre vécue sur place). Le retrait des cinq liens fondés sur des prévalences est méthodologiquement juste ; mais la question n'a plus **aucun** lien, alors que la CIM-11 est déjà utilisée ailleurs comme contexte de classification (`critere`) pour `temoin_violences`, `traumatisme_ponctuel` et `deces_proche` → 6B40, et pour les violences répétées → 6B41. Par cohérence, rétablir pour `exil` les deux liens de **contexte de classification** (6B40 : événement extrêmement menaçant ; 6B41 : violence organisée, torture, génocide), déjà sourcés par `cim11_cddr_stress`, sans aucune valeur empirique. Une recherche ciblée de méta-analyse comparant enfants exposés/non exposés à la guerre n'a ramené que des prévalences (Attanayake 2009, PMID 19413154, repéré) : le refus d'un lien empirique reste fondé.
3. **`soins_instables` ne couvre que la petite enfance** (« dans les premières années ») ; un placement ASE à 10 ans en foyer n'a pas de case. Soit élargir l'énoncé (sans transférer les liens « jeune enfant »), soit créer une case « Placement hors de la famille » sans lien tant qu'aucune comparaison sérieuse n'est vérifiée.
4. **Séparation : trois cases pour un seul ensemble de sources.** `separation_conflit` et `divorce` reçoivent les mêmes résultats Auersperg 2019 (PMID 31622869, ouvert), qui ne distingue pas le conflit. Le découpage est cliniquement juste (Gilman 2003, Am J Psychiatry, PMID 12727699, ouvert : effet du divorce précoce sur la dépression adulte « plus prononcé avec un conflit parental élevé »), mais la Cartographie n'en tire aucune différence : c'est un découpage d'interface, pas de preuve. À documenter ou à exploiter (Gilman).
5. **`parent_absent`** : « ou un père qui n'a pas vécu avec vous » est genré et recouvre « séparation/perte de contact ». Écrire « un parent qui n'a pas vécu avec vous ».

### 2.4 Doublons et recouvrements

- Négligence affective ≈ « Sentiment de ne pas compter » ≈ « Se sentir seul » ≈ « Ne pas se sentir protégé ».
- Parentification ≈ « Se sentir responsable de l'humeur des adultes » ≈ « Devoir être sage et ne pas déranger » ≈ « Devoir se débrouiller seul trop tôt ».
- Critiques ≈ Affection liée à la réussite ≈ Pression scolaire ≈ « Devoir être parfait ».
- Harcèlement ≈ Moqueries sur le corps ≈ Peu d'amis ≈ « Se sentir différent ».
- « Honte de soi » ≈ « Honte de sa famille ».

Les `overlap_group` protègent le niveau de convergence côté Ambre, **mais pas le tri public**, qui compte les cases : cocher les quatre variantes de parentification fait monter « Rôle de sauveur » (100 % de liens théoriques) en tête (§ 4).

### 2.5 Le thème « Ressentis » : circularité

14 questions, 53 liens, dont **49 théoriques** ; 26 sont marqués `definitional` (la totalité des liens définitionnels du graphe) (la question paraphrase le schéma : « Honte de soi » → schéma d'imperfection et de honte ; « Peur d'être abandonné » → schéma d'abandon ; « Devoir être parfait » → exigences élevées). Ce n'est pas une association, c'est une reformulation. Deux conséquences : (a) le résultat semble « confirmer » ce que la personne vient de dire ; (b) le tri par nombre de vécus gonfle les schémas. Le cas `ressenti_pas_compter → sch_carence_affective` est même noté « limitée » alors que la source empirique (Pilkington 2021) mesure des expériences parentales, pas ce ressenti. Recommandation : afficher les liens définitionnels à part (« ce ressenti correspond à la description du schéma… ») et les exclure du compte servant au tri.

### 2.6 Formulations

- « Punitions corporelles « ordinaires » » : les guillemets peuvent se lire comme ironiques ou minimisants. Proposer « Fessées, tapes ou gifles éducatives ».
- `enfant_prefere_role` : identifiant et libellé divergent (« enfant préféré » / « rôle fixe ») ; sans conséquence pour l'utilisateur, mais source d'erreurs éditoriales.
- `traumatisme_ponctuel`, `maladie_enfant`, `adoption`, `violences_fratrie`, `emprise_religieuse`, et les 14 ressentis n'ont **aucun exemple**, contrairement aux autres ; pour les violences, la sobriété est un choix défendable, mais « Événement traumatique ponctuel » gagnerait un exemple neutre (accident de la route, incendie, catastrophe naturelle, attentat).
- `maladie_famille` est restreint à la fratrie, `parent_malade_physique` est sept lignes plus bas dans un autre ordre : rapprocher les deux.
- Aucune question ne précise « avant 18 ans » : la page n'a plus d'introduction (choix du propriétaire) et les descriptions de thèmes ne le disent pas. Ajouter « avant 18 ans » dans le titre du volet ou de chaque thème coûte une ligne.
- Échelle **Non / Oui / Souvent** : « Souvent » n'a pas de sens pour les événements uniques (adoption, décès, divorce, migration, incarcération). Il n'a d'ailleurs **aucun effet sur le calcul** (seulement transmis à Ambre). Remplacer, pour les événements, par Non / Oui, ou par « Oui, une fois / Oui, longtemps ». Ajouter « Je ne sais pas / je préfère ne pas répondre », distinct de « Non » (le moteur traite une absence de réponse comme une absence d'association, ce qui est correct, mais la personne n'a pas de moyen de dire qu'elle ne se souvient pas).

## 3. Exhaustivité côté sujets (109 cartes)

### 3.1 Ce qui est bien couvert

Les 18 schémas de Young ; attachement anxieux/évitant et désorganisation du jeune enfant ; dépression, anxiété, panique, agoraphobie, anxiété sociale et de séparation ; TSPT, TSPT complexe, adaptation, deuil prolongé ; dissociation, TDI, trouble neurologique fonctionnel, détresse corporelle ; psychoses, schizotypie, bipolarité ; TCA (agrégat + anorexie, boulimie, hyperphagie) ; substances, alcool, cannabis, tabac, jeux d'argent, écrans ; automutilations, idées et tentatives de suicide ; borderline ; névrosisme, narcissisme ; alexithymie, régulation émotionnelle, colère, rumination, honte, culpabilité, autocritique, perfectionnisme, imposture, estime de soi, hypervigilance ; codépendance, isolement, qualité relationnelle, violences conjugales, revictimisation ; santé physique, cardiovasculaire, diabète, surpoids, douleurs, migraine, fibromyalgie, sommeil, sexualité ; scolarité, emploi. L'exclusion explicite des troubles neurodéveloppementaux (note sourcée) est exemplaire. Le registre (`couverture-referentiels.json`, 1 567 décisions) trace déjà la plupart des absences.

### 3.2 Manques vérifiés et décision proposée

| Sujet manquant | Source sérieuse ouverte pendant cette revue | Proposition |
| --- | --- | --- |
| **Préoccupation excessive pour l'apparence (symptômes de dysmorphophobie, 6B21 en contexte)** | Longobardi 2022, *Body Image*, PMID 35085863 (résumé ouvert) : méta-analyse à trois niveaux, 27 articles, 9 167 participants ; moqueries r = 0,42, harcèlement r = 0,28, abus r = 0,22 avec les symptômes dysmorphiques | Carte symptomatique, **liens limités** depuis `moqueries_corps` et `harcelement` (et éventuellement abus), sans code diagnostique exact. Le registre la laisse « à examiner » faute de texte intégral : obtenir le texte avant intégration, mais c'est le manque le plus net pour une case déjà présente (`moqueries_corps` n'a aujourd'hui qu'un lien modéré). |
| **Parentalité : reproduction de la maltraitance** | Madigan 2019, *Dev Psychopathol*, PMID 30757994 (résumé ouvert) : 142 études, d = 0,45, effet plus faible quand la qualité méthodologique augmente pour les violences physiques ; Assink 2018, *Child Abuse Negl*, PMID 30086419 (résumé ouvert) : r = 0,29, biais de publication signalés | Sujet explicitement demandé par le propriétaire et absent. Très stigmatisant si mal formulé. Proposer une **note sourcée** (« la plupart des parents ayant subi des violences ne les reproduisent pas ; un risque moyen plus élevé est observé et des soutiens le réduisent ») plutôt qu'une carte, ou une carte `contextual_only` jamais suggérée par Ambre. |
| **Troubles de la personnalité autres que borderline** | Johnson 1999, *Arch Gen Psychiatry*, PMID 10401504 (résumé ouvert) : cohorte communautaire, maltraitance documentée par dossiers officiels, risque de trouble de la personnalité multiplié par plus de 4 ; 10 catégories DSM-IV sur 12 associées ; Solmi 2021, *Front Psychiatry*, PMID 34552513 (résumé ouvert) : seules les associations borderline atteignent la classe II ; antisocial et dépendant en classe IV | La CIM-11 a abandonné les catégories ; le retrait de `icd_6d10` est cohérent. Garder hors carte, mais **dire explicitement** dans la notice que la recherche associe la maltraitance à des difficultés de personnalité au sens large (Johnson), pas seulement borderline. |
| Modes de Young autres que « parent exigeant » ; styles soumission/évitement | Bach 2018, *Cogn Behav Ther*, PMID 29256336 (résumé ouvert) : expériences parentales frustrant les besoins → domaines de schémas → mode « enfant vulnérable », étude transversale | **Incohérence interne** : un seul mode et un seul style d'adaptation (surcompensation) sont présents, alors que le modèle en compte trois. Soit retirer `ten_mode_parent_exigeant` et `ten_surcompensation_schema` (1 lien théorique chacun), soit ajouter « enfant vulnérable », « parent punitif », « protecteur détaché », « soumission », « évitement » au même niveau théorique. La première option est plus sobre. |
| Big Five hors névrosisme ; AMPD (fonctionnement de la personnalité) ; HiTOP détachement | non revérifié (Credé 2023 inaccessible au lot 4) | laisser « à examiner » ; dire hors périmètre provisoire |
| Mécanismes de défense (DMRS) | aucune méta-analyse trouvée par les lots ; pas de recherche concluante ici | hors périmètre explicite (construit inféré par un clinicien, pas déclaratif) |
| « Dépendance affective » | pas de construit validé distinct ; codépendance déjà présente, attachement anxieux aussi | hors périmètre, avec renvoi vers attachement anxieux et codépendance |
| Mentalisation, confiance épistémique, auto-compassion | revues repérées par les lots, non lues en intégral | « à examiner » ; pas de lien sans lecture |
| Énurésie, encoprésie, mutisme sélectif, phobie spécifique, accumulation, trouble du jeu vidéo | registre : `a_examiner` ou `aucune_preuve_trouvee`, raisons données | conforme ; aucune source supplémentaire trouvée ici |
| Santé physique hors cardiovasculaire/métabolique/douleur | Hailes 2019, *Lancet Psychiatry*, PMID 31519507 (résumé ouvert) : violences sexuelles associées à toutes les conditions physiques étudiées (OR 1,4–1,9) | la carte agrégée existe ; pas de nouvelle carte nécessaire |

### 3.3 Redondances entre cartes (même sens, plusieurs cartes)

- **Suicidalité : 4 cartes** (agrégat + idées + tentatives + automutilations). Pour un profil avec négligence affective, la même source Angelakis 2020 alimente l'agrégat (« modérée ») et « tentatives » (« limitée ») : **même résultat, deux forces différentes**.
- **Santé : agrégat « santé physique » + cardiovasculaire** : Soares 2022 (PMID 34041541, ouvert) alimente les deux pour la négligence affective et physique, avec « modérée » d'un côté et « limitée » de l'autre.
- **TCA : 4 cartes** (agrégat historique + 3 spécifiques). Conservé pour ne pas réinterpréter des retours anciens : justifié techniquement, mais l'utilisateur voit deux fois « boulimie ».
- Isolement subi ≈ schéma d'isolement ; perfectionnisme ≈ exigences élevées ; honte ≈ schéma d'imperfection ; rôle de sauveur ≈ abnégation ≈ codépendance (4 facteurs communs chacun).

Recommandation : regrouper visuellement ces familles (une carte parente dépliable) ou, au minimum, harmoniser les forces quand la source et le résultat sont identiques.

## 4. Qualité du calcul et des résultats

### 4.1 Ce que fait réellement le code

- Moteur (`compute`, identique navigateur/Worker, parité testée) : pour chaque carte, liens des vécus cochés ; niveau `convergence` si au moins deux familles de recouvrement (`overlap_group`) ont un lien empirique modéré ou fort, `piste` si une seule, `theorique` sinon ou si la carte est contextuelle ; tri par niveau, puis familles, puis force maximale.
- Interface (`renderResults`) : **ignore ce tri** et classe par `r.factors.length`, nombre brut de vécus cochés reliés, **tous liens confondus** (théoriques, définitionnels, contextes CIM). Le moteur ne sert qu'à départager les égalités.
- `data.interpretation.ranking` affirme pourtant « Niveau maximal du lien ; jamais une somme » : **la notice des données contredit l'interface publiée.**
- L'intensité « Souvent » n'a aucun effet sur l'affichage ; elle est seulement transmise à Ambre.

### 4.2 Effets mesurés (simulation sur 2 000 profils aléatoires par taille, données publiées)

| Cases cochées | Cartes affichées (médiane / 90e centile) | Dépression ou anxiété dans les 3 premières | Cartes « théoriques » non contextuelles dans le top 5 (moyenne) | Écart top 5 interface / top 5 moteur (Ambre) |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 15 / 55 | 83 % | 1,5 | 0,9 carte |
| 6 | 29 / 65 | 93 % | 1,1 | 1,8 carte |
| 10 | 55 / 77 | 99 % | 0,8 | 1,8 carte |
| 15 | 74 / 85 | 100 % | 0,6 | 1,7 carte |

Profils concrets :
- « Violences psychologiques » seule : 49 cartes, toutes à « 1 vécu associé » ; l'ordre visible est alors celui du moteur, avec psychoses, bipolarité et TCA dans les 7 premières.
- Profil « parentification » (parentification, parent en souffrance, se sentir responsable, enfant sage, se débrouiller seul) : les 4 premières cartes (« Rôle de sauveur », « Abnégation », « Codépendance », « Besoins ») reposent sur des modèles cliniques ou des liens limités, devant dépression et anxiété qui ont des méta-analyses.
- Tout coché : « Idées suicidaires, tentatives ou automutilations » (carte contextuelle) arrive 4e, avec l'étiquette « Preuves solides » dans l'en-tête replié ; « Rôle de sauveur » (12 liens, tous théoriques) 7e.

Conclusion : **le tri par nombre de vécus produit un effet de généricité** (les sujets les plus étudiés, dépression/anxiété, sont presque toujours en tête, quelle que soit la personne) **et un effet de redondance des questions** (les cases qui se recouvrent poussent les schémas théoriques). Il ne mesure ni la spécificité ni la preuve.

### 4.3 Recommandations de calcul (respectant la liste unique voulue par le propriétaire)

1. Garder une liste unique et le compte de vécus visible, mais **ne compter que les liens empiriques et les familles distinctes** (`evidence_groups`), puis départager par force. Concrètement : trier comme le moteur, ou trier par `evidence_groups` puis `factors.length`. La personne et Ambre verraient enfin le même ordre.
2. Placer les cartes contextuelles (suicide, santé, jeune enfant, TOC, traits) **après** les autres, ou les replier dans un groupe « Repères de contexte », et ne pas afficher « Preuves solides » dans leur en-tête.
3. Exclure les liens `definitional` du compte de tri (ils reformulent la question).
4. Pondérer la spécificité : une carte reliée à presque tous les vécus (dépression : 31 vécus, anxiété : 26) informe moins qu'une carte reliée à 3 vécus. Un indicateur simple « association fréquente à beaucoup d'adversités » sur ces cartes suffit, sans calcul de risque.
5. Au-delà d'environ 15 cartes, afficher les 15 premières puis « Voir les N autres » (repli), et regrouper les familles de cartes redondantes (§ 3.3).
6. Harmoniser les forces quand une même source et un même résultat alimentent deux cartes (Angelakis 2020 ; Soares 2022).
7. Mettre à jour `interpretation.ranking` pour décrire l'ordre réellement affiché.

### 4.4 Ce que reçoit Ambre

`companionCartography` n'est transmis que si la cartographie est enregistrée avec consentement et verrouillée. Le résumé contient : vécus cochés avec intensité, réponses « Non », jusqu'à 15 convergences et 10 pistes (au plus 5 et 4 par catégorie), toutes les cartes contextuelles, jusqu'à 10 cartes théoriques **seulement si la personne s'y est reconnue**, retours « oui/non », notes (≤ 80 × 1 000 caractères), ressources, notes d'Ambre (≤ 60) ; et les preuves sourcées dans un budget de 6 000 caractères JSON, un lien par sujet avant le second. Les cartes contextuelles et suicidaires ne sont jamais proposées comme points de départ (`cartographyHighlights`).

Points d'attention :
- Ambre raisonne sur un ordre (familles de preuves) différent de celui que la personne voit (nombre de vécus) : si Ambre dit « le point le plus étayé est X », la personne peut le trouver en 12e position.
- Les **ressources** sont transmises à Ambre mais **ne peuvent plus être saisies** : pour tout nouveau compte, ce champ est toujours vide ; Ambre ne voit donc jamais les appuis de l'enfance.
- Les cartes contextuelles sont toutes transmises sans plafond (jusqu'à une quinzaine) ; leur présence dans le contexte est utile pour la sécurité mais occupe du budget.

## 5. Fiabilité des liens

### 5.1 Indicateurs globaux (calculés sur les données publiées)

- **316 liens sur 493 (64 %) reposent sur une seule source** ; 242 des 363 associations empiriques (67 %).
- Champ `read` des 260 sources : **185 lues au résumé seul**, 73 avec sections ou texte intégral, 2 pages web. **193 associations empiriques sur 363 (53 %) n'ont que des sources lues au résumé** ; 119 n'ont qu'une source, lue au résumé ; dont 54 classées « modérée » ou « forte ».
- 19 liens « modérés » reposent sur **une seule étude primaire** (cohorte), dont 7 lues au résumé seulement (Burrell 2022 ×2, Webb 2016, Copeland 2013 ×2, Takizawa 2014, Kosty 2020). Une cohorte unique, même grande, devrait plafonner à « limitée » si l'on applique la même exigence que pour les études transversales.
- Aucune source manquante, aucune URL absente ; toutes les références tirées ont été retrouvées sur Europe PMC avec le bon titre.
- Sources à statut faible : `schwartz_fawn` (page de praticienne, seule source de 4 liens théoriques vers « apaiser autrui ») ; `clance1978` (article fondateur, légitime comme modèle) ; `lane2016_gambling` (revue réelle, PDF hébergé par un site associatif : conserver le DOI de l'éditeur) ; `cunningham2016` (pointé vers IDEAS/RePEc plutôt que l'éditeur). Aucune source introuvable ni manifestement fictive.

### 5.2 Échantillon de 30 liens tirés au hasard (graine 20261118), vérifiés sur le résumé Europe PMC/PubMed de chaque source

| # | Lien (vécu → sujet, force) | Verdict | Détail vérifié |
| ---: | --- | --- | --- |
| 1 | surprotection → schéma dépendance (modèle) | non revérifié | extrait Guilford de Young inaccessible (contenu non PDF) ; cohérent avec le modèle |
| 2 | négligence physique → borderline (modérée) | conforme | Solmi 2021 : OR 5,73, classe II ; qualité des méta-analyses faible à modérée, bien signalée |
| 3 | ressenti « ne pas compter » → besoins (modèle) | partiel | Bach 2018 mesure des pratiques parentales et des domaines de schémas, ni ce ressenti ni l'expression des besoins |
| 4 | violences sexuelles → anxiété (**forte**) | sens conforme, **force discutable** | Chen 2010 OR 3,09 (cas-témoins + cohortes) ; Hailes 2019 OR 2,7 mais **qualité jugée faible** pour l'anxiété ; Li 2016 ne donne qu'un OR combiné dépression/anxiété → « modérée » |
| 5 | conflits parentaux → externalisation (modérée) | conforme | van Eldik 2020 : hostilité plus liée à l'externalisation, associations durables ; Rhoades 2008 porte sur les réactions des enfants (secondaire) |
| 6 | violences psychologiques → migraine (limitée) | conforme | Liu 2024 OR 1,71 ; Flor 2025 association « modérément faible » (+7 %) |
| 7 | violences psychologiques → automutilations (modérée) | conforme | Liu 2018 OR 3,03 |
| 8 | parent en souffrance → rôle de sauveur (modèle) | conforme | Gazzillo 2026, théorie contrôle-maîtrise : parents malades, en deuil, inversion des rôles |
| 9 | violences psychologiques → TOC (limitée, contexte) | conforme | Ou 2021 : r = 0,11 avec la sévérité chez des patients |
| 10 | violences sexuelles → TDI (CIM) | non revérifié | CDDR OMS non rouvert |
| 11 | critiques → mode parent exigeant (modèle) | non revérifié | Young inaccessible |
| 12 | affection conditionnelle → rôle de sauveur (modèle) | non revérifié, rapprochement indirect | le modèle relie plutôt l'acceptation conditionnelle à la recherche d'approbation |
| 13 | harcèlement → emploi (modérée) | conforme | Brimblecombe 2018 : moins d'emploi et de patrimoine à 50 ans ; revenus plus bas chez les femmes |
| 14 | parent avec addiction → dépendance à l'alcool 6C40.2 (modérée) | sens conforme, **force et code discutables** | Kosty 2020 : une cohorte, 730 descendants, critères DSM-IV, génétique non séparée → « limitée » ; correspondance DSM-IV/6C40.2 approximative |
| 15 | discrimination → estime de soi (modérée) | conforme | Schmitt 2014 : effet plus faible sur les issues positives, plus fort chez l'enfant ; Benner 2018 adolescence |
| 16 | parent avec addiction → trouble panique (modérée) | conforme | Zhang 2023 : alcoolisme parental OR 1,83 |
| 17 | violences physiques → hypervigilance (modèle) | conforme | McLaughlin 2020, modèle transdiagnostique |
| 18 | violences sexuelles → TOC (limitée) | conforme | Ou 2021 : obsessions r = 0,13, pas le score total |
| 19 | négligence physique → cardiovasculaire (modérée) | conforme, force à discuter | Soares 2022 : UK Biobank, HR 1,09–1,27 toutes formes, rétrospectif ; une seule cohorte |
| 20 | séparation conflictuelle → dépression (modérée) | conforme pour la séparation | Auersperg 2019 OR 1,29 ; Simbi 2020 séparation OR 3,14 (9 cas-témoins) ; le conflit n'est pas mesuré |
| 21 | absence de limites → droits excessifs (modèle) | non revérifié | Young inaccessible ; cohérent avec le domaine « limites déficientes » |
| 22 | violences sexuelles → boulimie (modérée) | conforme | Caslini 2016 OR 2,73 |
| 23 | peu d'amis → usage de substances (limitée) | **partiel (construit)** | Lacey 2014 étudie la CRP ; le tabagisme est un médiateur ; « fumer plus souvent » n'est pas un « usage problématique » |
| 24 | violences psychologiques → scolarité (modérée) | conforme | Zhang 2025 r global −0,16, effets plus nets pour violences psychologiques et négligence |
| 25 | critiques → perfectionnisme (modérée) | conforme | Curran 2022 : effets forts sur le perfectionnisme socialement prescrit, faibles à modérés sinon |
| 26 | violences psychologiques → anxiété (modérée) | conforme | Xiao 2023 |
| 27 | violences sexuelles → attachement évitant (modérée) | conforme | Li 2026 r = 0,20 ; Shahab 2025 : intrafamilial plus fort |
| 28 | violences sexuelles → narcissisme vulnérable (limitée) | plausible | Gao 2024 r = 0,20 global ; estimation par sous-type non visible dans le résumé |
| 29 | contrôle → autocritique (limitée) | conforme | Bleys 2018 : contrôle psychologique centré sur la performance (illustre le problème de découpage de `controle`) |
| 30 | parent en souffrance → dépression (modérée) | conforme | Rasic 2014 ; Lawrence 2019 RR dépression 1,31 chez enfants de parents anxieux |

Bilan : **19 conformes, 3 conformes avec force surestimée, 2 partiellement conformes (mesure ≠ construit de la carte), 1 plausible, 5 non revérifiables ici** (Young et CDDR). **Aucun lien de sens inversé ni inventé.** Les défauts sont de deux types : force « forte/modérée » attribuée sur une base plus faible que ce que la règle annonce, et glissement de construit entre ce que l'étude mesure et le libellé de la carte.

## 6. Expérience

- **Longueur** : 59 questions × 3 choix, plus des notes facultatives ; compter 10 à 20 minutes. Les six volets sont fermés au départ et indépendants, ce qui permet de s'arrêter ; mais rien n'indique qu'on peut ne remplir qu'une partie. Les volets dépassent souvent 10 questions (famille 13, climat 13, ressentis 14) ; le générateur autorise 14 par thème et 72 au total, il reste donc peu de place pour des ajouts : chaque nouvelle case doit être compensée par une fusion (voir § 2.4).
- **Ordre des thèmes** : la page ouvre sur « Violences et maltraitance », et la deuxième question est « Violences sexuelles ». Les questionnaires sensibles commencent en général par des domaines moins intrusifs. Proposer : Contexte → Situation familiale → Climat → Manques de soins → Violences → Ressentis, avec un repère « vous pouvez passer une question ».
- **Sobriété des violences** : les libellés sont factuels et sans détails, c'est bien. « Ordinaires » entre guillemets et l'absence de « avant 18 ans » sont les deux seuls écarts relevés.
- **Risque de suggestion** : le sens du raisonnement va toujours du vécu vers le sujet, jamais du symptôme vers un vécu supposé (aucune carte ne dit « si vous avez X, vous avez peut-être vécu Y ») : c'est l'essentiel et c'est respecté. Trois risques demeurent : (1) les ressentis paraphrasent les schémas, d'où un effet de confirmation (§ 2.5) ; (2) le bouton « Je me reconnais » sur des descriptions générales favorise l'effet Barnum, et les cartes non reconnues restent en place (seulement grisées) ; (3) le bouton « Liens étudiés (52) » sur chaque question montre les conséquences *avant* la réponse, ce qui peut orienter la réponse. Aucun énoncé n'invite à « retrouver » des souvenirs ; garder cette règle explicite dans la charte.
- **Sécurité** : sur demande du propriétaire, l'encadré 3114 a été retiré (documenté dans `plan-execution-ambre-cartographie-2026-11-17.md`). Les cartes suicidaires restent visibles, contextuelles, sans ressource. Le texte `notes.consulter` (3114, 119, CMP) existe dans les données mais n'est rendu nulle part. Option compatible avec la décision : une ligne « 3114, gratuit, 24 h/24 » **à l'intérieur du corps déplié** des quatre cartes suicidaires seulement, sans bandeau. À soumettre au propriétaire, pas à imposer.
- **Notices** : `notes.limites`, `notes.cumul`, `notes.multifinalite`, `notes.neurodeveloppement` et `notes.consulter` sont dans les données mais **plus affichés** (introductions retirées). L'avertissement « pas un diagnostic » n'apparaît que dans les cartes de troubles dépliées et dans la fenêtre « Liens étudiés ». La description de la page (`content/cartographie.md`) promet « les conséquences psychologiques que la recherche y associe » : « conséquences » est plus causal que ce que les données soutiennent ; « sujets associés » serait exact.
- **Accessibilité** : boutons `aria-pressed` groupés et étiquetés, `details/summary`, `dialog` modal, restauration du focus après rendu : bon niveau. Deux problèmes : (1) le conteneur des résultats est `aria-live="polite"` et entièrement reconstruit à chaque réponse, si bien qu'un lecteur d'écran peut relire des dizaines de cartes à chaque clic ; annoncer seulement le compteur (« 12 réponses · 31 associations ») dans une région live séparée ; (2) les étiquettes de force (« Preuves solides »…) ne sont distinguées que par le texte et une classe de couleur, ce qui est suffisant, mais la carte repliée n'indique pas si elle est contextuelle avant ouverture (seulement une phrase sur Ambre).
- **Suppression** : pour un compte connecté, la page n'offre plus de bouton de suppression (décision du propriétaire) ; seul le brouillon anonyme peut être effacé ici. La suppression du compte efface la cartographie (cascade, `journal/src/worker.mjs`), et la route `DELETE /api/cartographie` existe, mais aucun bouton ne l'appelle pour un membre connecté : effacer seulement cette donnée de santé, sans supprimer le compte, n'est plus possible depuis l'interface. À signaler au propriétaire (droit d'effacement ciblé).

## 7. Vécus muets ou sous-reliés

| Vécu | Liens | Observation | Source sérieuse ouverte |
| --- | ---: | --- | --- |
| `secret_famille` | 0 | aucune association retenue ; le bouton « Liens étudiés » disparaît sans explication | aucune trouvée ici ; MACE classe les « secrets importants cachés » comme maltraitance non verbale (instrument, pas preuve d'association) |
| `migration` | 0 | cohérent (pas de comparaison propre) ; à dire explicitement | — |
| `exil` | 0 | incohérence avec les autres expositions traumatiques (voir 2.3) | contexte CIM-11 déjà sourcé (`cim11_cddr_stress`) |
| `precarite` | 3 | aucun lien vers dépression ou anxiété, alors que c'est l'une des expositions les plus documentées | Gilman 2003, Am J Psychiatry, PMID 12727699 : cohorte prospective (mesures prénatales et à 7 ans, 1 104 personnes), faible statut socio-économique dans l'enfance → risque accru d'épisode dépressif majeur entre 18 et 39 ans, indépendamment du statut adulte ; Reiss 2013 (PMID 23746605, déjà source) : 52 études sur 55, risque ×2 à ×3 de troubles psychiques chez l'enfant ; Peverill 2021 (PMID 33278703, déjà source) : effets plus nets sur l'externalisé que l'internalisé |
| `temoin_violences` | 7 (3 empiriques) | **aucun lien vers l'externalisation** | Vu 2016, Clin Psychol Rev, PMID 27136293 : 74 études longitudinales, exposition aux violences conjugales → problèmes externalisés et internalisés ultérieurs, associations qui se renforcent avec le temps |
| `maladie_famille` | 2 | faible, cohérent avec la littérature disponible | — |
| `violence_quartier` | 2 | limité, cohérent | — |
| `emprise_religieuse` | 3 (théoriques) | cohérent après retrait Saldaña | — |

## 8. Corrections et ajouts priorisés

### Priorité 1 — sans nouvelle recherche, effet immédiat

1. **Aligner le tri public sur les familles de preuves** (ou au minimum exclure liens théoriques, définitionnels et contextes CIM du compte), placer les cartes contextuelles en fin de liste sans étiquette « Preuves solides », et corriger `interpretation.ranking` (§ 4.3). Respecte la liste unique voulue par le propriétaire.
2. **Rétablir la saisie des 11 ressources** (ou de questions PCE/BCE) dans « Mon enfance » : les données, le stockage, la validation et le résumé d'Ambre existent déjà. Sources : Bethell 2019 (déjà `bethell2019`), Narayan 2018 (PMID 28992958, ouvert, à ajouter).
3. **Scinder `controle`** en contrôle psychologique et autorité stricte, en ne réaffectant que les liens dont la source mesure la dimension (§ 2.3).
4. **Harmoniser les forces incohérentes** : Angelakis 2020 (suicidalité agrégée « modérée » vs tentatives « limitée ») ; Soares 2022 (cardiovasculaire « modérée » vs santé physique « limitée »).
5. **Abaisser** violences sexuelles → anxiété de « forte » à « modérée » (Hailes 2019 juge faible la qualité des méta-analyses sur l'anxiété) ; parent avec addiction → dépendance à l'alcool de « modérée » à « limitée » (une cohorte, génétique non séparée) ; et appliquer la même règle aux 19 liens « modérés » fondés sur une seule cohorte (§ 5.1).
6. **Corriger le construit** isolement social → « usage problématique de substances » (Lacey 2014 mesure le tabagisme comme médiateur) : retirer ou recibler vers le tabagisme avec « limitée ».
7. **Exil → 6B40 et 6B41 en contexte de classification** (`critere`, contextuel), par parité avec `traumatisme_ponctuel` et `temoin_violences`.
8. **Formulations** : « avant 18 ans » ; « Fessées, tapes ou gifles éducatives » ; « un parent qui n'a pas vécu avec vous » ; exemple neutre pour l'événement traumatique ; supprimer le doublon « rejet familial lié à l'orientation » de `discrimination` (le garder dans `abus_emotionnel`) ; échelle Non/Oui pour les événements uniques ; option « je ne sais pas ».
9. **Accessibilité** : région live limitée au compteur.
10. **Afficher pour les questions sans lien** une mention « aucune association suffisamment documentée n'est retenue pour l'instant » au lieu d'un bouton absent.

### Priorité 2 — ajouts de liens fondés sur une source ouverte pendant cette revue

| Ajout proposé | Force proposée | Source ouverte | Réserve |
| --- | --- | --- | --- |
| `temoin_violences` → `ten_externalisation` | modérée | Vu 2016, PMID 27136293 (résumé) | lire méthodes et tableau avant intégration |
| `precarite` → `dep_generale` | limitée (une cohorte) | Gilman 2003, PMID 12727699 (résumé) | dépression adulte ; ne pas étendre à l'anxiété sans autre source |
| `abus_sexuel` → `icd_6b60` : ajouter `hailes2019`, passer de « limitée » à « modérée » | modérée | Hailes 2019, PMID 31519507 (résumé) : trouble de conversion OR 3,3, qualité modérée | vérifier que la méta-analyse sous-jacente porte bien sur la conversion au sens 6B60 |
| Nouvelle carte symptomatique « Préoccupation excessive pour l'apparence » ← `moqueries_corps`, `harcelement` | limitée | Longobardi 2022, PMID 35085863 (résumé) | méta-analyse surtout transversale ; texte intégral à obtenir |
| Nouvelle question « Violences dans une relation amoureuse à l'adolescence » → dépression, idées suicidaires, substances, violences conjugales adultes | limitée (une cohorte) | Exner-Cortens 2013, Pediatrics, PMID 23230075 (résumé) : Add Health, 5 681 jeunes, effets différents selon le sexe, ajustés sur la maltraitance infantile | compenser par une fusion pour rester sous le plafond de 14 par thème |
| Note sourcée « parentalité » (et éventuelle carte contextuelle) | note | Madigan 2019, PMID 30757994 ; Assink 2018, PMID 30086419 (résumés) | formulation non stigmatisante obligatoire ; jamais suggérée par Ambre |

### Priorité 3 — structure

11. Regrouper les familles de cartes redondantes (suicidalité ×4, TCA ×4, santé ×2, sauveur/abnégation/codépendance).
12. Décider pour les modes et styles de Young : retirer les deux isolés ou compléter le modèle (§ 3.2).
13. Fusionner ou regrouper visuellement les ressentis qui paraphrasent un vécu (§ 2.4–2.5) et afficher les liens définitionnels à part ; libère de la place sous le plafond de 72 questions.
14. Réordonner les thèmes (§ 6) et les questions dans « Situation familiale » (maladies parent/fratrie côte à côte ; séparations côte à côte).
15. Remplacer à terme les 185 lectures « résumé seul » par des lectures de méthodes et résultats, en commençant par les 54 liens « modérés/forts » à source unique lue au résumé.
16. Suicidalité : proposer au propriétaire la ligne 3114 dans le corps déplié des cartes suicidaires (§ 6) ; bouton d'effacement ciblé de la cartographie pour les comptes.

## 9. Ce qui doit rester hors périmètre, et pourquoi

| Domaine | Raison |
| --- | --- |
| Autisme, TDAH, troubles du développement intellectuel, tics | origine surtout génétique et développementale ; la note `neurodeveloppement` (Thapar, Faraone, Tick) le dit déjà. La rendre visible. |
| Troubles neurocognitifs, catatonie, intoxications, sevrages, syndromes induits, troubles respiratoires du sommeil | aucune inférence possible depuis un vécu d'éducation |
| Catégories de personnalité DSM autres que borderline (cartes) | la CIM-11 est dimensionnelle ; seules les associations borderline sont de classe II (Solmi 2021). Mentionner Johnson 1999 dans une notice plutôt qu'en cartes |
| Mécanismes de défense | construits inférés par un clinicien (DMRS), pas d'association déclarative sérieuse trouvée |
| « Dépendance affective » | pas de construit validé distinct ; renvoyer vers attachement anxieux et codépendance |
| Cadres HiTOP/AMPD en tant que tels | structures de classement, pas des sujets ; leurs dimensions restent « à examiner » |
| Exil, guerre, migration comme causes empiriques | seules des prévalences sans groupe de comparaison ont été trouvées (y compris par cette revue) ; contexte CIM seulement |
| Secrets familiaux | aucune association sérieuse trouvée ; la question reste utile pour Ambre |
| ARFID, pica, mérycisme, énurésie, encoprésie, transe/possession, trouble délirant, schizoaffectif | registre : aucune preuve propre admissible, ou risque de pathologiser une pratique culturelle |
| Score ACE cumulé comme risque individuel | effet dose-réponse de population (Hughes 2017) ; jamais un risque personnel |
| Expositions prénatales et biologiques (alcool fœtal, prématurité) | pas des « vécus » rapportables ; hors questionnaire |

Ces exclusions doivent être **visibles** pour l'utilisateur, pas seulement dans le registre : une ligne « Ce que cette page ne relie pas à l'enfance, et pourquoi » suffit, ou la réintégration de la note `neurodeveloppement`.

## 10. Références ouvertes pendant cette revue

Toutes consultées via Europe PMC/PubMed (résumé) sauf mention.

- Teicher MH, Parigger A, 2015, PLOS ONE — MACE (texte PMC4340880, liste des sous-échelles).
- Bethell C et al., 2019, JAMA Pediatrics, PMID 31498386 — PCE.
- Narayan AJ et al., 2018, Child Abuse & Neglect, PMID 28992958 — BCE.
- Gilman SE et al., 2003, Am J Psychiatry, PMID 12727699 — perturbations familiales, statut socio-économique et dépression adulte.
- Reiss F, 2013, Soc Sci Med, PMID 23746605 ; Peverill M et al., 2021, Clin Psychol Rev, PMID 33278703.
- Vu NL et al., 2016, Clin Psychol Rev, PMID 27136293 — exposition aux violences conjugales.
- Exner-Cortens D et al., 2013, Pediatrics, PMID 23230075 — violences dans les relations amoureuses adolescentes.
- Longobardi C et al., 2022, Body Image, PMID 35085863 — adversités et symptômes dysmorphiques.
- Madigan S et al., 2019, Dev Psychopathol, PMID 30757994 ; Assink M et al., 2018, Child Abuse Negl, PMID 30086419 — transmission intergénérationnelle.
- Johnson JG et al., 1999, Arch Gen Psychiatry, PMID 10401504 ; Solmi M et al., 2021, Front Psychiatry, PMID 34552513 — personnalité.
- Hailes HP et al., 2019, Lancet Psychiatry, PMID 31519507 — violences sexuelles, revue parapluie.
- Kowalski RM et al., 2014, Psychol Bull, PMID 24512111 — cyberharcèlement.
- Bach B et al., 2018, Cogn Behav Ther, PMID 29256336 — schémas et mode enfant vulnérable.
- Attanayake V et al., 2009 (PMID 19413154) — repéré par recherche web seulement : prévalences chez enfants exposés à la guerre, sans groupe témoin ; non utilisé comme preuve.
- Sources de l'échantillon § 5.2 : Solmi 2021, Chen 2010, Li 2016, van Eldik 2020, Rhoades 2008, Liu 2024, Flor 2025, Liu 2018, Gazzillo 2026, Ou 2021, Brimblecombe 2018, Kosty 2020, Schmitt 2014, Benner 2018, Zhang 2023, Soares 2022, Auersperg 2019, Simbi 2020, Caslini 2016, Lacey 2014, Zhang 2025, Qu 2024, Curran 2022, Xiao 2023, Li 2026, Shahab 2025, Gao 2024, Bleys 2018, Rasic 2014, Lawrence 2019, McLaughlin 2020.
- Non accessibles ici : extrait Guilford de Young 2003 (réponse non PDF), CDDR OMS (non rouvert), page CIM-11 en ligne (contenu dynamique).
