# Recherche Cartographie v5 — exhaustivité (fragment du 19 novembre 2026)

Rédigé le 17 novembre 2026 pour le lot C5 du plan v5. Livrable : `editorial/cartographie/liens-exhaustivite-v5-2026-11-19.json`. Aucun autre fichier modifié, rien de commité. La règle absolue du plan (§0) a été appliquée : aucun lien sans source réellement ouverte ; chaque source porte dans `read` ce qui a été lu (texte intégral ou sections, ou résumé seul).

## 0. Méthode et contrôle

- Lectures préalables : plan du 17/11 (§0, §1, §6.3, §6.6), revue d'exhaustivité du 18/11 (§3, §8, §9), `docs/v4-C-avancement.md`, `data/cartographie.json` (59 facteurs, 109 sujets, 496 liens, 261 sources), `corrections.json`, `couverture-referentiels.json` (statuts antérieurs des mêmes sujets), `scripts/build_cartographie.py`.
- Recherche : PubMed (E-utilities), Europe PMC (textes intégraux PMC), OpenAlex et Crossref (accès ouvert), WebSearch et WebFetch. Les textes fermés (Elsevier, Sage, BMJ en 403) sont signalés.
- Contrôle sans écriture du §6.6 sur l'arbre avec le fragment : `59 115 523 273` puis `OK hors empreinte audit` (+6 sujets, +27 liens, +12 sources). Contrôles manuels : aucune paire, aucun identifiant de sujet ni de source en collision avec l'existant ; aucun mot interdit ni pourcentage ; résumé le plus long : 287 caractères (plafond 350) ; chaque sujet nouveau a au moins un lien ; `hollander2016` et `lo2026_gaming`, déjà déclarés ailleurs, ne sont pas redéclarés.
- Le fragment se classe avant `liens-humeur…`, `liens-transparence…` et `liens-v2-*` : il ne redéclare donc rien d'existant (piège de fusion, plan §1). Toute modification de l'existant est dans `proposed_corrections`.

## 1. Difficultés de personnalité au sens large

| Source | Type | Lu | Apport |
| --- | --- | --- | --- |
| Johnson 1999, Arch Gen Psychiatry (`johnson1999_personnalite`, déjà publiée) | cohorte communautaire, maltraitance attestée par les services de protection | texte JAMA ouvert (méthodes, résultats) | 639 familles ; 10 catégories DSM-IV sur 12 associées ; violences physiques → antisociale, borderline, dépendante, dépressive… ; négligence → antisociale, évitante, borderline, narcissique, paranoïaque… ; groupes attestés très petits (15 violences physiques, 4 sexuelles, 23 négligences) |
| Johnson 2006, Arch Gen Psychiatry, PMID 16651515 | même cohorte (CIC), SCID-II à 22 et 33 ans | résumé | faible affection parentale → antisociale, évitante, borderline, dépressive, paranoïaque, schizoïde, schizotypique, après ajustement sur les troubles de l'enfant et des parents ; effet cumulatif |
| Miron 2026, Psychol Bull, PMID 42831821 | méta-analyse multiniveaux, 104 études | résumé | violences psychologiques plus associées que physiques et sexuelles aux syndromes de personnalité, ajusté sur leur cumul ; surtout transversal |
| de Ruiter 2022, PLoS One, PMC9365173 | méta-analyse, 47 études | sections (≈ 2/3 du texte) | traits psychopathiques : r ≈ 0,19 violences physiques, 0,21 négligence, 0,15 violences psychologiques, 0,10 violences sexuelles ; plus net pour le versant antisocial ; 34 études sur 47 en milieu judiciaire ou clinique |
| Solmi 2021 (déjà publiée) | revue parapluie | — | seules les associations borderline atteignent la classe II : justifie que les nouvelles cartes restent contextuelles |
| Credé 2023, Pers Individ Dif | méta-analyse ACE × traits non cliniques | **non lu** (texte fermé, aucun résumé dans PubMed, Crossref, OpenAlex) | laissé « à examiner » |

Propositions (dans `outcomes` et `links`) :

- **`ten_difficultes_personnalite`** « Difficultés de personnalité au sens large (repère de recherche) », `category: categorical_personality_diagnosis` (donc `contextual_only` automatique) : `abus_physique` et `negligence_physique` (Johnson 1999), `negligence_emotionnelle` (Johnson 2006), `abus_emotionnel` (Miron 2026), toutes **limitées**.
- **`ten_traits_psychopathiques`** « Traits psychopathiques (dimension de recherche) », contextuelle : `abus_physique`, `negligence_emotionnelle`, `negligence_physique`, `abus_emotionnel`, toutes **limitées**. `abus_sexuel` (effet très faible, risque de stigmatiser les victimes) reste dans `proposed_links`.
- Alternative plus sobre, si l'intégrateur juge la carte « psychopathie » trop proche de `ten_traits_personnalite_antisociale` : ajouter `deruiter2022_psychopathie` comme seconde source de `abus_physique → ten_traits_personnalite_antisociale` et ne pas créer la carte.
- Hors carte : PID-5/AMPD (aucune méta-analyse par domaine), Big Five hors névrosisme (Credé non lu), sévérité 6D10 (rapprochement documentaire seulement).
- Conséquence d'affichage : si ces cartes sont intégrées, reformuler l'item « Troubles de la personnalité autres que borderline » de « Ce que la carte ne couvre pas » (texte proposé dans `proposed_corrections`).

## 2. « Dépendance affective »

- Le générateur ne gère pas d'« autre nom » affiché. La clé `aliases` de `corrections.json` (lue dans `build()` : `aliases.get(link["outcome"])`, et remplacement de l'identifiant d'un sujet déclaré) **redirige un ancien identifiant vers un seul identifiant canonique** ; elle n'affiche rien et ne peut pas viser deux cartes. Elle ne convient donc pas.
- Solution immédiate, sans code : `outcome_overrides` sur les descriptions de `att_anxieux` (« On parle parfois de « dépendance affective », un terme courant qui n'est pas un diagnostic. ») et de `ten_codependance` (« … proche de ce qu'on appelle parfois « dépendance affective » »). Aucun lien ni force ne change.
- Solution durable (code, hors mission) : champ facultatif `also_known_as` recopié par `build_cartographie.py`, affiché sous le libellé et indexé par la recherche de la page.
- Sources : Cavalli 2025, J Behav Addict, PMID 40299575 (méta-analyse, 15 études, résumé) — la « love addiction » est liée surtout à l'attachement anxieux (corrélation modérée), très faiblement et en sens inverse à l'évitant ; Özal 2023, J Clin Med, PMID 37959234 (revue de portée, résumé) — pas de définition consensuelle. Aucun lien d'enfance vers ce concept : il reste `agrege_dans`.

## 3. Exil, guerre, migration, catastrophes

Résultat principal : **aucune méta-analyse comparant des enfants exposés à la guerre à des enfants non exposés** n'a été trouvée (PubMed : 30 méta-analyses guerre/réfugiés/catastrophes chez l'enfant examinées par titre ; Europe PMC pour les registres). Ce qui existe :

| Source | Plan | Lu | Décision |
| --- | --- | --- | --- |
| Eltanamly 2021, PMC7675766 | méta-analyse (38 études, 82 % transversales, 60 % Palestine/Israël) ; **dose d'exposition à l'intérieur de populations exposées** | sections (≈ 4/5 ; tableaux supplémentaires des effets par issue non lus) | `exil → ten_souffrance_enfance` en `proposed_links` seulement : pas de groupe non exposé |
| Brandt 2019, JAMA Psychiatry, PMID 31411649 | méta-analyse avec **groupes de comparaison** (réfugiés vs autres migrants et natifs ; 9 études nordiques et canadiennes) | résumé | `exil → icd_6a20` proposé (`proposed_links`, limitée) |
| Hollander 2016, BMJ, PMC4793153 (`hollander2016`, déjà déclaré dans `liens-contexte.json`) | cohorte de registre, 1,3 million ; réfugiés HR 2,9 vs nés en Suède, 1,7 vs autres migrants | sections (plan, résultats, forces et faiblesses) | même proposition ; **réserve** : arrivées après 1998 de personnes nées après 1984, donc jusqu'à environ 27 ans, pas seulement dans l'enfance ; sujet sensible |
| Rubens 2018, PMC6055700 | méta-analyse catastrophes naturelles (62 études) ; mesures surtout continues, 13 comparaisons géographiques, 3 avant/après | sections | `traumatisme_ponctuel → ten_souffrance_enfance` **limitée** (avec Furr) ; ajout comme source à `traumatisme_ponctuel → anx_generale` et `dep_generale` (forces inchangées) |
| Furr 2010, PMID 21114340 | méta-analyse 96 études, stress post-traumatique des jeunes après catastrophe, r groupé ≈ 0,19, plus fort avec proximité et menace perçue | résumé | idem |
| Attanayake 2009, Blackmore 2020, Zasiekina 2025, Adugna 2026, Olk 2025 | prévalences sans groupe témoin | titres et résumés déjà examinés par les lots | aucun lien (décision antérieure confirmée) |

Registres nordiques comparant enfants réfugiés et natifs : ils mesurent le recours aux soins, que l'accès biaise (une étude danoise trouve *moins* de contacts chez les enfants réfugiés) ; non utilisés.

## 4. Mécanismes de défense

- Aucune méta-analyse ni revue systématique reliant l'enfance aux défenses (Europe PMC : seules des revues sur la dépression, l'automutilation, le cancer et le DMRS-SR-30). Les études enfance → défenses sont transversales (Wang 2021, Fang 2020, Ferrajão 2024–2025), souvent en population clinique.
- Une donnée longitudinale : **Martin-Joy 2017** (étude Grant de Harvard, 72 hommes, défenses cotées par des juges indépendants vers 52 et 75 ans ; résumé) — une enfance chaleureuse va avec des défenses plus mûres au milieu de la vie, **écart disparu vers 75 ans**. Soldz et Vaillant 1998 (résumé) valident la hiérarchie du DSM-IV sans porter sur l'enfance ; Cramer et Block 1998 (résumé) relient la personnalité de l'enfant de 3–4 ans, pas un vécu, au déni à 23 ans ; Howell et Békés 2025 (revue narrative, résumé) relient l'attachement insécure aux défenses immatures.
- Proposition : carte **`ten_defenses_peu_adaptees`** « Mécanismes de défense peu adaptés (repère clinique) », contextuelle, catégorie nouvelle `defense_mechanism` (libellé d'interface à créer), **un seul lien théorique** (`negligence_emotionnelle`, `clinical_model`, Martin-Joy 2017). Les défenses une à une restent hors périmètre. Si l'intégrateur préfère ne pas ajouter de catégorie, garder le statu quo (hors périmètre) : c'est défendable.

## 5. Revue d'exhaustivité : sujets ou vécus encore absents

| Sujet | Décision | Source (lecture) |
| --- | --- | --- |
| **Solitude** | **ajout de liens** vers `pat_social_isolation` (sa description dit déjà « moins de liens que souhaité ») : violences psychologiques et négligence affective **modérées** ; violences physiques, sexuelles, négligence matérielle **limitées** ; description à préciser (« Se sentir seul… ») | de Heer 2024 (texte intégral via Europe PMC : g = 0,45 ; 0,54 pour violences psychologiques et négligence affective ; 0,35–0,39 pour les autres ; **mesures prospectives g = 0,15 non significatif**, dossiers 0,13 non significatif ; pas de biais de publication) ; Curtis 2025 (résumé, r = 0,30) |
| **Anxiété sociale distincte** | **ajout de 4 liens limités** vers `icd_6b04` (résumés à formuler en symptômes) ; correction du lien `harcelement → icd_6b04` pour dire que le sens est surtout inverse | Liu 2023 (résumé : maltraitance émotionnelle r = 0,25, physique 0,14, sexuelle 0,15) ; Deng 2024 (résumé : 19 études longitudinales, l'anxiété sociale prédit le harcèlement, pas l'inverse) |
| **Impulsivité** | **nouvelle carte** `ten_impulsivite` (trait déclaré, description qui l'écarte du TDAH), 5 liens limités | Liu 2019 (texte PMC, tableau 2 : OR 1,59 violences sexuelles à 3,10 violences psychologiques ; toutes études transversales ; tests de laboratoire non significatifs ; biais de publication signalé) |
| **Préoccupation pour l'apparence (symptômes dysmorphiques)** | **nouvelle carte** `ten_preoccupation_apparence`, `moqueries_corps` et `harcelement` limités | Longobardi 2022 (résumé ; texte fermé : Turin accès réservé, Valence illisible). La v4 l'avait différée pour ce motif : à trancher par l'intégrateur |
| **Violences dans les premières relations amoureuses (comme sujet)** | **nouvelle carte** `ten_violences_amoureuses_jeunes`, `harcelement` limité ; liens depuis la maltraitance en attente (Hébert 2019 ne détaille pas les formes dans le résumé) | Zych 2021 (résumé : victimisation scolaire → violences subies, OR ajusté ≈ 2) ; Hébert 2019 (résumé, 87 études) |
| Violences amoureuses à l'adolescence **comme vécu** | inchangé : décision du propriétaire (nouvelle question) | Exner-Cortens 2013 (déjà examiné en v4) |
| Catastrophes | lien vers `ten_souffrance_enfance` (voir §3) | Rubens 2018, Furr 2010 |
| Autocompassion | `proposed_links` vers `pat_self_criticism` (construit voisin, à décider) ou carte propre | Zhang 2023 (résumé : violences psychologiques r = −0,28, négligence affective −0,31) |
| Mentalisation | à examiner : r = −0,21 global, pas de résultat par forme dans le résumé | Yang 2024 (résumé) |
| Trouble du jeu vidéo | à examiner : types de victimisation non nommés dans le résumé ; couvert en partie par `ten_ecrans` | Lo 2026 (résumé relu) |
| Accumulation (6B24) | à examiner : revue systématique hétérogène, non spécifique | Chia 2021 (résumé) |
| Deuil prolongé (lien empirique avec l'enfance) | aucune preuve : les facteurs de risque sont la perte et l'attachement anxieux ; repères CIM existants suffisants | Buur 2024 (résumé) |
| Burn-out | aucune preuve : études transversales isolées (médecins, étudiants infirmiers) | — |
| Hypocondrie / anxiété de santé | aucun lien nouveau (cohorte danoise 2026 sans résumé accessible) ; lien existant conservé | — |
| Colère, agressivité adulte | déjà couvertes (`ten_colere`, `ten_externalisation`) | — |
| Trouble explosif intermittent, PID-5, Big Five hors névrosisme | à examiner (voir §1 et registre) | — |

Non examinés faute de temps (à reprendre) : trichotillomanie et excoriation, achats compulsifs, fatigue chronique, reconnaissance des émotions (deux méta-analyses de 2026 repérées par titre seulement), traits « insensibles et sans émotion » chez l'enfant.

## 6. Bilan des forces proposées

- **Modérées (2)** : violences psychologiques et négligence affective → solitude (`pat_social_isolation`).
- **Limitées (24)** : personnalité au sens large ×4, traits psychopathiques ×4, impulsivité ×5, solitude ×3, anxiété sociale ×4, apparence ×2, violences amoureuses ×1, catastrophes ×1.
- **Théorique (1)** : négligence affective → défenses peu adaptées.
- **En proposition seulement (5)** : exil → psychoses ; exil → souffrance dans l'enfance ; violences sexuelles → traits psychopathiques ; maltraitance → violences amoureuses ; violences psychologiques → autocritique (autocompassion).
- **Corrections (10)** : 2 descriptions pour « dépendance affective », 1 option de code (`also_known_as`), 1 description de solitude, reformulation de harcèlement → anxiété sociale, 2 ajouts de source (catastrophes → anxiété et dépression), 3 textes de « Ce que la carte ne couvre pas ».

## 7. Doutes et limites

- 8 des 12 sources du fragment ne sont lues qu’au résumé ; seules de Heer 2024, Liu 2019, de Ruiter 2022 et Rubens 2018 ont été lues par sections (Liu et de Ruiter : environ les deux tiers du texte). Johnson 1999 a été relu sur la page JAMA.
- **Effet de volume** (revue §4) : +6 cartes et +27 liens, surtout à partir des violences ; les nouvelles cartes de personnalité, de psychopathie et de défenses sont contextuelles pour ne pas remonter en tête. `ten_impulsivite`, `ten_preoccupation_apparence` et `ten_violences_amoureuses_jeunes` ne le sont pas.
- **Glissements de construit** : anxiété sociale mesurée par questionnaires reliée à la carte diagnostique 6B04 (déjà le cas pour les liens existants) ; « maltraitance émotionnelle » de Liu 2023 regroupe violences psychologiques et négligence affective ; « négligence » de de Ruiter et Johnson 1999 ne distingue pas matérielle et affective.
- **Sensibilité** : traits psychopathiques et psychoses chez les réfugiés : à soumettre à la relecture indépendante avant publication.
- La catégorie `defense_mechanism` n'a pas de libellé d'interface.
- Aucun audit d'empreinte relancé (hors mission) ; l'intégrateur doit le faire.
