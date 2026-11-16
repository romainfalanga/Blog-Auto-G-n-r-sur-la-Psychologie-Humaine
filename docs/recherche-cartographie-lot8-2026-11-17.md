# Lot 8 — vécus manquants, instruments et codes QE

Remise du 17 novembre 2026. Recherche effectuée dans la session du 16 novembre UTC ; version de travail fondée sur les 58 facteurs et 425 liens publiés. Lecture d’AGENTS.md, des documents de reprise, des règles éditoriales, de l’état utile de l’automatisation et de l’ensemble du plan de couverture. Seuls le présent rapport et `editorial/cartographie/liens-vecus-manquants-2026-11-17.json` ont été écrits par ce lot.

**Bilan : cinq nouvelles paires intégrables, deux facteurs proposés, cinq liens réservés aux propositions, quinze nouvelles sources, douze corrections proposées, vingt-huit décisions de couverture.** Aucun nouveau sujet. Aucun facteur stocké renommé ou supprimé ; aucune génération ni publication exécutée. Les cibles de 30–50 liens et 6–12 facteurs ne sont pas des quotas : la majorité des candidats se réduit à des exemples existants ou demande des preuves plus spécifiques.

## 1. Instruments réellement consultés

| Instrument | Documents ouverts et niveau de lecture | Résultat et limite |
| --- | --- | --- |
| ACE CDC-Kaiser | [Page officielle CDC](https://www.cdc.gov/violenceprevention/aces/about.html), sections Study Questionnaires et ACEs Definitions | Dix catégories confirmées, dont les deux négligences recueillies dans la seconde vague. Trois cases locales sont limitées au parent au lieu du membre du foyer. Le site possède onze cases `ace`, dont les deux séparations alternatives ; cela n’en fait pas un score ACE validé. |
| ACE-IQ OMS | [Questionnaire](https://cdn.who.int/media/docs/default-source/documents/child-maltreatment/ace-questionnaire.pdf?sfvrsn=baed215c_2), [guide question par question](https://cdn.who.int/media/docs/default-source/documents/child-maltreatment/ace-iq-question-by-question-document.pdf?sfvrsn=3018c40a_2), [guide d’analyse](https://cdn.who.int/media/docs/default-source/documents/child-maltreatment/ace-iq-guidance-for-analysing.pdf?sfvrsn=adfe12bb_2) | Treize catégories ; cotations binaire et fréquence. Le guide distingue violence communautaire vécue en personne, violence collective et harcèlement. La guerre ne nécessite pas d’exil. Les agressions sexuelles peuvent avoir n’importe quel auteur. |
| CTQ | [Fiche officielle Pearson](https://www.pearsonclinical.co.uk/en-gb/Store/Professional-Assessments/Personality-%26-Biopsychosocial/Brief/Childhood-Trauma-Questionnaire%3A-A-Retrospective-Self-Report/p/P100009055), Benefits/Features | Cinq sous-échelles confirmées et gradation de sévérité. **Le manuel commercial n’a pas été consulté.** Ni seuils, ni traduction française, ni équivalence de cotation vérifiés. |
| MACE | [Article primaire de 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0117423), méthodes/résultats ; [instrument S2](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0117423.s011&type=supplementary), items 1–41 ; [manuel MACEscore S8](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0117423.s017&type=supplementary), p. 1–5 | Dix types, 52 questions avec repérage annuel 1–18 ans ; séparation témoin/victime de la fratrie. S2 et S8 étaient en erreur dans le lecteur web, puis ont été ouverts en HTTP 200 et lus en mémoire par téléchargement direct. Le manuel S8 est celui du logiciel de cotation, pas un manuel clinique complet. |

Extraits courts de contrôle : CDC, « A household member went to prison » ; ACE-IQ analyse, « sorted into 13 categories » ; ACE-IQ guide, « not in your home or on TV, movies, or the radio » ; CTQ, « Physical Abuse, Sexual Abuse, Emotional Abuse, Physical Neglect and Emotional Neglect » ; MACE S2, « Please check all ages that apply » ; manuel MACEscore, « checking and correcting common entry errors ».

Les dix catégories ACE sont les trois violences, les deux négligences, les violences envers la mère, l’addiction du foyer, la souffrance psychique du foyer, l’incarcération du foyer et la séparation parentale. ACE-IQ élargit les violences aux membres du foyer, inclut la perte d’un parent dans sa catégorie parentale et ajoute pairs, communauté et collectif. Le MACE distingue les violences parentales verbales/non verbales, physiques, sexuelles, les deux négligences, les violences entre parents, sur la fratrie et les violences physiques/émotionnelles des pairs.

## 2. Correspondance des 58 facteurs

Cette table est une **correspondance éditoriale de contenu**, pas une validation du questionnaire ni une équivalence de score. `—` signifie absence de catégorie directe dans cet instrument. `~` signifie rapprochement partiel, sous-condition ou recouvrement : ne pas compter comme couverture exacte. Les colonnes QE ne posent aucun diagnostic automatique.

Abréviations : PA = violence physique ; SA = sexuelle ; EA = psychologique ; PN = négligence matérielle ; EN = affective ; DV = violence domestique observée ; SUB = substances au foyer ; MI = trouble psychique/suicidalité au foyer ; INC = incarcération au foyer ; SEP = séparation/perte parentale ; BULL = pairs ; COLL = violence collective. Pour MACE : PPA = physique parentale ; PVA = verbale parentale ; NVEA = non verbale parentale ; PEA = émotionnelle des pairs ; PPB = physique des pairs ; WIPV = violence entre parents ; WITS = violence observée sur la fratrie.

| Facteur stocké | ACE | ACE-IQ | CTQ | MACE | QE vérifié / réserve |
| --- | --- | --- | --- | --- | --- |
| `abus_physique` | PA~ | PA~ | PA | PPA~ | QE82.0 ; auteurs plus larges que parent/foyer |
| `abus_sexuel` | SA~ | SA~ | SA | SA~ | QE82.1 ; corriger restriction aux pairs plus âgés |
| `abus_emotionnel` | EA | EA | EA | PVA + NVEA | QE82.2 |
| `temoin_violences` | DV~ | DV~ | — | WIPV | QE83~ ; ACE centré mère, ACE-IQ tout foyer |
| `harcelement` | — | BULL | — | PEA + PPB | QE82.0/.2~ selon actes, pas code de harcèlement attribué |
| `negligence_physique` | PN | PN | PN | PN | QE82.3 ; QE90~ pour défaut de surveillance |
| `negligence_emotionnelle` | EN | EN | EN | EN | QE82.3~ ; gravité clinique non déduite du ressenti |
| `soins_instables` | — | — | — | — | QE93/QE94~ ; placement/institution ≠ instabilité nécessaire |
| `parent_trouble` | MI~ | MI~ | — | — | — ; parent plus étroit que foyer |
| `parent_addiction` | SUB~ | SUB~ | — | — | — ; jeu non inclus dans ACE SUB |
| `parent_incarcere` | INC~ | INC~ | — | — | QE60~ si absence ; parent plus étroit que foyer |
| `separation_conflit` | SEP~ | SEP~ | — | — | QE70.1 ; QE52.1~ si perte de relation |
| `deces_proche` | — | SEP~ pour parent | — | — | QE52.1, QE61~ selon proche |
| `maladie_famille` | — | — | — | — | QE92~ / QE70.2~ selon retentissement |
| `precarite` | PN~ seulement si besoins non couverts | PN~ seulement si besoins non couverts | PN~ | PN~ | — ; pauvreté ≠ négligence |
| `invalidation` | EA~ | EA~ | EA~ | PVA~ | QE82.2~ selon actes et retentissement |
| `critique` | EA~ | EA~ | EA~ | PVA/NVEA~ | QE95/QE96~ selon événements |
| `amour_conditionnel` | — | — | — | EN~ | QE52.0~ ; pas de correspondance spécifique |
| `surprotection` | — | — | — | — | QE91 |
| `controle` | — | — | — | — | QE95~ ; pas équivalent au contrôle psychologique |
| `imprevisibilite` | — | — | — | — | — |
| `parentification` | — | — | — | — | QE95~ pour responsabilités inadéquates à l’âge |
| `comparaison_fratrie` | — | — | — | — | QE52.0/QE96~ selon actes et retentissement |
| `secret_famille` | — | — | — | — | — ; QE96 vise l’événement dévalorisant, pas tout secret |
| `absence_limites` | — | PN~ surveillance | — | PN~ surveillance | QE90~ ; permissivité ≠ défaut grave de surveillance |
| `ressenti_pas_compter` | EN~ | EN~ | EN~ | EN~ | — ; impression distincte d’un déficit de soins établi |
| `ressenti_abandon` | — | EA~ menace explicite seulement | — | NVEA~ menace explicite seulement | — ; peur ≠ abandon observé |
| `ressenti_honte` | — | — | — | — | — ; QE96 nécessite un événement, pas la honte seule |
| `ressenti_responsable` | — | — | — | — | — |
| `ressenti_perfection` | — | — | — | — | — |
| `ressenti_peur` | EA~ si actes menaçants | EA~ si actes menaçants | — | NVEA/PEA~ si actes | — ; peur seule ≠ exposition |
| `ressenti_solitude` | — | — | — | — | — |
| `ressenti_colere_interdite` | — | — | — | — | — |
| `discrimination` | — | BULL~ par pairs | — | PEA~ par pairs | QE04 |
| `exil` | — | COLL | — | — | QE81 ; persécution QE04 selon contexte |
| `migration` | — | — | — | — | — ; ne pas coder automatiquement un problème |
| `traumatisme_ponctuel` | — | COLL~ selon événement | — | — | QE80/QE81/QE83~ selon événement |
| `maladie_enfant` | — | — | — | — | QE94~ hospitalisation prolongée sans parent seulement |
| `conflits_parents` | — | DV~ si menaces/violence | — | — | QE52.0~ si retentissement relationnel ; non équivalent WIPV |
| `triangulation` | — | — | — | — | QE52.0~ sans équivalence spécifique |
| `parent_absent` | — | SEP~ si perte/décès | — | — | QE60 ; QE52.1~ pour départ définitif |
| `divorce` | SEP | SEP | — | — | QE70.1 |
| `famille_recomposee` | — | — | — | — | QE92~ changement relationnel défavorable requis |
| `demenagements` | — | COLL~ si déplacement contraint | — | — | — ; fréquence seule non codée |
| `adoption` | — | — | — | — | — ; ne pas assimiler adoption à privation/placement |
| `violences_fratrie` | — | PA/EA~ membre du foyer | — | — ; différent WITS | QE82.0/.2~ selon violences ; victime, non témoin |
| `punitions_corporelles` | PA~ seuils | PA~ fréquence | PA~ seuils non vérifiés | PPA~ items 9–11 | QE82.0~ selon définition clinique |
| `pression_scolaire` | — | — | — | — | QE95~ pression inadéquate, pas toute exigence |
| `emprise_religieuse` | — | — | — | — | QE82.2/QE95~ selon actes ; croyance non codée |
| `honte_famille` | — | — | — | — | — ; événement stigmatisant éventuellement QE96 |
| `debrouiller_seul` | EN/PN~ selon actes | EN/PN~ selon actes | EN/PN~ | EN/PN~ | — ; impression ne suffit pas à QE82.3 |
| `pas_protege` | EN/PN~ selon actes | EN/PN~ selon actes | EN/PN~ | PN~ item 44 | — ; impression ne suffit pas à QE82.3 |
| `se_sentir_different` | — | — | — | — | — |
| `enfant_sage` | — | — | — | — | — |
| `enfant_prefere_role` | — | — | — | — | — |
| `moqueries_corps` | EA~ adulte foyer | EA/BULL~ auteur | EA~ | PVA/PEA~ | QE96/QE82.2~ selon actes et retentissement |
| `parent_malade_physique` | — | — | — | — | QE92/QE70.2~ selon retentissement et aide |
| `isolement_social_enfant` | — | BULL~ exclusion imposée | — | PEA~ exclusion imposée | — ; peu d’amis ≠ rejet imposé |

Le constat initial « ACE dix sur dix » est donc une couverture des grandes familles, pas une équivalence exacte. ACE-IQ : communauté absente ; foyer partiellement couvert ; collectif présent dans `exil`. CTQ : les cinq familles sont présentes, sans cotation validée. MACE : neuf familles présentes avec fusions, WITS manquant ; ni âge annuel ni durée ne sont recueillis. La distinction verbale/non verbale ou physique/émotionnelle peut rester dans les exemples. Le nombre de facteurs ne résout pas l’absence de fréquence, d’âge et de durée.

## 3. Codes QE : vérification officielle et limites

Le [navigateur OMS 2026-01](https://icd.who.int/browse/2026-01/mms/en) a été ouvert. Son contenu dynamique ne livrait pas les notices au lecteur web. Ses liens **Print Versions** et **Spreadsheet File** ont conduit aux archives officielles [PDF](https://icdcdn.who.int/static/releasefiles/2026-01/print-ICD-11-MMS-en.zip) et [SimpleTabulation](https://icdcdn.who.int/static/releasefiles/2026-01/SimpleTabulation-ICD-11-MMS-en.zip), téléchargées en HTTP 200 et extraites en mémoire. Aucun code QE n’a été copié depuis un site tiers.

Notices réellement lues : QE04 (p. imprimée 1755), QE52.0–.1 (1763–1764), QE60–62 et QE70.0–.2 (1764), QE80–83 (1765), QE90–95 (1766), QE96 (1767). Les intitulés et identifiants ont été recoupés dans la tabulation pour QE52.*, QE81–83 et QE90–96.

| Codes | Usage possible dans cette cartographie |
| --- | --- |
| QE82.0 / .1 / .2 / .3 | Antécédents physiques / sexuels / psychologiques / négligence. Contexte de la victime, sans déduction sur l’auteur ni la conséquence. |
| QE52.0 / .1 | Difficulté durable de relation soignant-enfant / perte d’une relation affective d’enfance ; pas synonyme de tout conflit ou sentiment d’abandon. |
| QE60 / QE61 / QE70.1 | Absence / disparition-décès familial / séparation-divorce. |
| QE92 / QE70.2 | Changement familial défavorable / proche dépendant ayant besoin de soins au domicile. |
| QE90 / QE91 / QE95 | Surveillance inadéquate / surprotection / pression éducative inadéquate. |
| QE93 / QE94 | Retrait du domicile / éducation institutionnelle. Ne prouve ni maltraitance ni dommage ; un placement peut protéger. |
| QE96 | Événement dévalorisant pendant l’enfance ; ne transforme pas la honte rapportée en exposition établie. |
| QE80 / QE81 / QE83 / QE04 | Crime-terrorisme / catastrophe-guerre / expérience effrayante d’enfance / discrimination-persécution. Choix conditionnel au vécu précis. |

Extrait de contrôle OMS : « applied to the victim of the maltreatment, not the perpetrator ». Le rapprochement facteur ↔ code est une proposition éditoriale ; aucune case n’attribue ces codes à une personne. Aucun code QE n’est ajouté comme sujet ou comme preuve empirique.

## 4. Décisions sur les candidats et les découpages

Le registre exhaustif de ce lot est `out_of_scope` : 28 entrées, avec statut et raison. Les termes classés `agrege_dans` gardent leur vécu dans les cases indiquées ; ils ne sont pas déclarés sans preuve ou inexistants.

- **Violence communautaire** : proposition `violence_quartier`, limitée au fait d’assister en personne à des violences réelles. Deux liens limités vers symptômes anxieux et dépressifs. Les études sur la communauté ne sont pas étendues à l’école, au harcèlement ou aux images médiatiques.
- **Témoin de violences sur la fratrie** : proposition `temoin_violences_fratrie`, distincte de `violences_fratrie` (être soi-même visé par sa fratrie). Trois liens limités, dont dissociation appuyée par deux échantillons. **Admission en attente** : le seuil du § 6.4 autorise des liens limités, mais les études disponibles ici sont transversales, alors que le § 0 cite seulement méta-analyse/revue systématique/cohorte pour les preuves empiriques. Ne pas activer sans arbitrage explicite de l’intégrateur sur cette incohérence du plan.
- **Violence collective sans exil** : déjà décrite dans `exil` ; exemple ajouté. Pas de scission qui ferait compter deux fois le même conflit.
- **Placement tardif** : à examiner. L’instabilité du placement a une littérature, mais ne prouve pas un effet propre du premier placement tardif. Élargir `soins_instables` ferait appliquer des liens de petite enfance à des placements adolescents ; conserver son périmètre actuel.
- **Cyberviolence** : exemple dans `harcelement`. **Violence sexuelle de pairs ou en ligne** : préciser `abus_sexuel`, en retirant la restriction aux pairs plus âgés. **Contenus sexuels** : distinguer exposition imposée, fortuite et consommation ; pas de case omnibus. **Contenus violents** : investigation spécifique restante, sans assimiler visionnage et violence communautaire.
- **Rejet orientation/identité, handicap ou apprentissages** : exemples dans `discrimination`, `harcelement` ou `abus_emotionnel` selon les actes. Aucune inférence vers une orientation, une identité ou un trouble neurodéveloppemental.
- **Adulte d’institution** : exemples dans les trois types de violences, sans nouvelle case pour le seul statut de l’auteur.
- **Internat précoce** : à examiner ; une méta-analyse de l’internat existe, mais son agrégat scolaire ne répond pas au scénario spécifique de séparation très précoce.
- **Suicide d’un proche** : exemple du décès ; distinguer suicide et tentative. Ne pas appliquer à tout deuil les effets d’un sous-groupe.
- **Contrôle** : expliciter la culpabilisation, le retrait affectif et l’intrusion ; réexaminer chaque source avant d’assimiler les règles comportementales au contrôle psychologique.
- **Précarité** : un lien externalisation ajouté ; ne pas confondre pauvreté et négligence. **Incarcération** : quatre paires ajoutées. **Secret familial, maladie de la fratrie, différence ressentie** : faiblesse ou investigation restante conservées, sans associations nouvelles fabriquées.
- **Baumrind, Rohner** : recouvrements conceptuels, pas équivalences de mesure ; le style démocratique relève des ressources du lot 7. **Parent narcissique** : pas de case diagnostique ; garder des descriptions de comportements.

La proposition communautaire recoupe des expositions physiques : `overlap_group: physical_violence` est volontairement conservateur et doit être validé. Pour le témoin de fratrie, `family_conflict` suit les violences domestiques observées. Aucun `ace: true` supplémentaire. Si les deux propositions étaient retenues : **60 facteurs**, dont violences 8, négligence 3, famille 13, climat 13, ressentis 14, contexte 9 ; plafonds 14/thème et 72 total respectés. Avec la seule proposition communautaire : 59.

## 5. Preuves empiriques et limites de lecture

| Source | Méthodes/résultats réellement lus ; extrait court de contrôle | Décision |
| --- | --- | --- |
| [Miliauskas et al., 2022](https://link.springer.com/article/10.1186/s12888-022-03873-8) | 42 études, 21 dans la synthèse quantitative ; méthodes, tableau 1, sous-groupes victime/témoin et résultats. Extrait : « Six works found an association for victims of community violence but not for witnesses ». | Deux liens proposés, limités. Associations anxieuses et dépressives hétérogènes ; des études nulles. Pas de transfert aux violences scolaires ou sexuelles, exclues par la revue. |
| [Teicher et Vitaliano, 2011](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0028852) | 1 412 jeunes adultes volontaires ; analyses ajustées, tableaux 1–2, limites. Extrait : « cross-sectional analysis of a convenience sample ». | Trois liens limités, conditionnels. Rétrospectif, sélection et confusion résiduelle ; pas de mécanisme causal repris des modèles de médiation. |
| [Teicher et Parigger, 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0117423) | Construction sur 1 051 participants, résultats WITS et dissociation ; sous-échelles et documents instrumentaux. Extrait : « witnessing violence toward siblings on the MACE showed a highly significant association with DES scores ». | Corroboration indépendante du lien dissociatif ; validation instrumentale transversale, pas preuve prospective. |
| [Gifford et al., 2019](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2748665) | Cohorte prospective GSMS, 1 420 enfants, 1 334 revus adultes ; méthodes/tableaux 1, 3–4. Extrait : « before or during the participant’s lifetime ». | Quatre liens limités. La mesure inclut figures parentales et incarcérations antérieures à la naissance : correspondance imparfaite avec la case. Associations ajustées pour anxiété, substances, diplôme et soutien social ; pas dépression, suicide ou emploi. |
| [Peverill et al., 2021 — source déjà existante](https://sdlab.fas.harvard.edu/sites/g/files/omnuum5291/files/sdlab/files/peverill_2021_ses_meta_analysis.pdf) | Méthodes et résultats 3.2–3.4, tableau 2 ; 13 cohortes, 26 715 participants. Extrait : « SES has stronger associations with childhood externalizing psychopathology than internalizing psychopathology ». | Un lien modéré vers externalisation, effets modestes. Jeunes aux États-Unis ; aucune inférence adulte. Mise à jour de `read` proposée sans redéclarer la source. |
| [Maguire et al., 2024](https://link.springer.com/article/10.1007/s40653-023-00606-1) | Méthodes, 14 études dont 9 longitudinales, résultats et synthèse narrative ; âges 0–18, définitions variables. Extrait : « Results of the relationship with internalising behaviours were less consistent ». | Justifie une réserve sur le placement tardif, pas un nouveau lien du placement en soi. Instabilité et difficultés peuvent être bidirectionnelles. |
| [Pathmendra et al., 2023](https://www.jmir.org/2023/1/e43116) | Méthodes, résultats Included Studies et Early Age of First Sex ; 19 études, dont 6 cohortes. Extrait : « Two prospective cohort studies reported conflicting findings ». | Pas deux liens spécifiques établis pour une exposition précoce imposée. Pas d’amalgame avec violences sexuelles, ni moralisation de la sexualité adolescente. |
| [Zhong et al., 2024](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1359626/full) | Codage, calculs, résultats et tableaux 2–4 ; 91 effets, hétérogénéité extrême, dimensions agrégées. Extrait : « boarding school was not a significant predictor of overall student development ». | Le résultat global est nul, certains domaines diffèrent. Ne répond pas aux deux liens propres à l’internat très précoce. |

Les « textes intégraux » ci-dessus désignent l’accès au texte et la lecture effective des sections indiquées, pas une lecture exhaustive de toutes les références ou annexes. Les résultats des moteurs de recherche ne sont pas des lectures d’article. Fowler 2009 a été repéré, mais ses essais de PDF ont échoué ; il n’est ni déclaré ni utilisé pour gonfler la preuve. Les liens modernes repérés sur cyberharcèlement, rejet familial et placements 2025–2026 non relus intégralement n’ont pas été transformés en preuves.

## 6. Les 17 liens de ressentis à revoir

Extraction du jeu publié, filtre `group == ressentis` et `relation_type == empirical_association`. Ce relevé n’abaisse pas automatiquement les liens ; il donne au lot 9 les paires exactes à examiner. Une mesure contemporaine de honte, solitude ou peur ne prouve pas qu’un ressenti rétrospectif d’enfance constitue une exposition indépendante.

| Facteur | Sujet | Sources existantes |
| --- | --- | --- |
| `pas_protege` | `pat_self_criticism` | `richter2009` |
| `ressenti_colere_interdite` | `pat_emotional_suppression` | `roth2009` |
| `ressenti_honte` | `anx_generale` | `candea2018` |
| `ressenti_honte` | `dep_generale` | `kim2011` |
| `ressenti_honte` | `icd_6b04` | `candea2018` |
| `ressenti_pas_compter` | `dep_generale` | `kidd2022` |
| `ressenti_pas_compter` | `icd_6a72` | `kidd2022` |
| `ressenti_pas_compter` | `ten_alexithymie` | `thorberg2011` |
| `ressenti_perfection` | `pat_conditional_worth` | `assor2004` |
| `ressenti_perfection` | `pat_perfectionism` | `lavrijsen2021`, `curran2022` |
| `ressenti_peur` | `anx_generale` | `mclaughlin2020` |
| `ressenti_peur` | `att_desorganise` | `madigan2006` |
| `ressenti_peur` | `ten_dysregulation_emotionnelle` | `mclaughlin2020` |
| `ressenti_peur` | `ten_hypervigilance` | `mclaughlin2020` |
| `ressenti_responsable` | `pat_guilt` | `kim2011`, `candea2018` |
| `ressenti_solitude` | `anx_generale` | `loades2020` |
| `ressenti_solitude` | `dep_generale` | `loades2020` |

## 7. Contrôle et remise

Le fragment n’ajoute aucune paire déjà publiée, aucun identifiant de source existant, aucun sujet existant redéclaré. Les liens de facteurs inconnus restent exclusivement dans `proposed_links`. Les propositions d’exemples ont `target: factor_editorial`, les réserves de périmètre `target: factor_scope`, et l’enrichissement de lecture Peverill `target: source_read` : ces objets sont ignorés par le générateur et doivent être interprétés par l’intégrateur, sans modification automatique de données enregistrées.

Vérifications finales à renseigner après contrôle local : syntaxe JSON, identifiants/paires, références et budget des résumés ; compilation en mémoire du générateur sans écriture, erreur d’empreinte d’audit attendue. Aucun test produit ni déploiement ne relève de ce lot.
