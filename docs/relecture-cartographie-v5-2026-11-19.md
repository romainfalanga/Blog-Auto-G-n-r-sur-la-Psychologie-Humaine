# Relecture scientifique indépendante — Cartographie v5 (fragment exhaustivité du 19/11)

Rédigée le 17 novembre 2026 par un relecteur distinct de l'agent de recherche. Objet : `editorial/cartographie/propositions/liens-exhaustivite-v5-2026-11-19.json` (6 sujets, 27 liens, 12 sources, 16 sources proposées, 5 liens proposés, 10 corrections, 16 entrées hors périmètre), son rapport `docs/recherche-cartographie-v5-2026-11-19.md`, et cinq liens existants signalés par la relecture clinique de C5a (`docs/v5-C5a-avancement.md`). Règle appliquée : plan du 17/11, §0. **Aucun fichier de données ni de code modifié, rien n'est commité.**

## 0. Ce qui a été réellement ouvert

| Source | Lecture faite par le relecteur |
| --- | --- |
| de Heer 2024 (PMC11539460) | texte intégral Europe PMC : résumé, modérateurs (type, âge, mesure prospective, dossiers, plan), biais de publication |
| Liu 2019 (PMC6269232) | texte intégral PMC (efetch) : résultats par forme, mesures en laboratoire, biais de publication, discussion |
| de Ruiter 2022 (PMC9365173) | texte intégral : tableau 1 (par forme), facettes, type d'échantillon, biais de publication |
| Rubens 2018 (PMC6055700) | texte intégral : mesures d'exposition, modèle internalisé |
| Hollander 2016 (PMC4793153) | résumé PubMed et texte intégral (recherche d'une analyse par âge à l'arrivée : absente) |
| Palmier-Claus 2016 | texte intégral (Cambridge Core reader) : tableau 1 par forme, mesures, types I/II |
| Johnson 1999 | résumé PubMed et page JAMA (effectifs attestés : 15 violences physiques, 4 sexuelles, 23 négligences) |
| CIM-11, 6B40 | texte des lignes directrices (version d'essai sur le terrain, PMC6225017) : « experiencing the sudden, unexpected or violent death of a loved one » ; texte final CDDR non rouvert |
| Curtis 2025, Liu 2023, Johnson 2006, Miron 2026, Martin-Joy 2017, Longobardi 2022, Zych 2021, Furr 2010, Deng 2024, Hébert 2019, Brandt 2019, Cavalli 2025, Özal 2023, Zhang 2023, Buur 2024, Chia 2021, Yang 2024, Lo 2026, González-Díez 2017, Lawrence 2019, Burrell 2022, Varese 2012 | résumés PubMed (E-utilities) |
| Non rouverts | Eltanamly 2021, Howell 2025, Cramer 1998, Soldz 1998, Credé 2023 (fermé) ; texte intégral de Lawrence 2019 (403 sur JAACAP et sur le dépôt de Southampton), Longobardi 2022 et Liu 2023 (fermés) |

Constats transversaux :
- **Fausse affirmation du rapport (§4 et §7)** : la catégorie `defense_mechanism` a déjà un libellé d'interface (« Mécanisme de défense », `static/js/cartographie-ui.js`, l. 15). Sans incidence, puisque la carte est refusée.
- **Doubles comptes** : Liu 2023 ne sépare pas violences psychologiques et négligence affective (« maltraitance émotionnelle ») ; de Ruiter 2022 ne sépare pas négligence matérielle et affective (une seule ligne « Neglect », k = 20). Un même effet regroupé ne doit pas être présenté comme deux constats distincts sans le dire.
- **Affirmations absentes des sources** relevées dans les résumés : « études surtout transversales » (Liu 2023, Longobardi 2022 : rien de tel dans les résumés lus), « tests de laboratoire » cités par forme (Liu 2019 : seulement 4 études, sur la maltraitance globale), « beaucoup de personnes concernées ont des liens qui les soutiennent » et « peu d'études portent sur cette forme » (de Heer). Ces phrases sont corrigées plus bas.
- Les cartes contextuelles (`contextual_only`) ne sont plus affichées dans les résultats depuis C5a. Une carte contextuelle sert donc surtout à Ambre (qui ne reçoit plus les sujets masqués) et au registre, ce qui réduit son intérêt sans annuler les risques qu'elle porte.

## 1. Sources du fragment (`sources`)

| Id | Verdict | Remarque |
| --- | --- | --- |
| `deheer2024_solitude` | accepter | Méta-analyse multiniveau, de l'enfance à la vieillesse (pas seulement des adultes). g = 0,45 ; plus fort pour violences psychologiques et négligence affective ; **pas de différence dans les études prospectives ni dans celles qui utilisent des dossiers** ; écart plus faible avec l'âge ; pas de biais de publication |
| `curtis2025_solitude` | accepter | 20 études, adultes ; r = 0,30 (cumul) ; les cinq formes significatives ; hétérogénéité |
| `liu2023_anxiete_sociale` | accepter | 29 études ; émotionnelle r = 0,25, physique 0,14, sexuelle 0,15 ; forte hétérogénéité |
| `liu2019_impulsivite` | accepter | 55 études ; OR de 1,59 (sexuelle) à 3,10 (psychologique) ; analyses toutes transversales ; biais de publication (Egger significatif) ; tâches en laboratoire : 4 études, maltraitance globale, OR 1,07 non significatif |
| `deruiter2022_psychopathie` | **déplacer vers `proposed_sources`** | la carte qu'elle fonde est refusée (§2) ; garder pour le registre |
| `johnson2006_personnalite` | accepter | Children in the Community, 593 familles, SCID-II à 22 et 33 ans, ajusté sur les troubles de l'enfant et des parents |
| `miron2026_abus_emotionnel` | accepter | 104 études, 414 effets ; transversal et rétrospectif ; ajusté sur le cumul |
| `martinjoy2017_defenses` | **déplacer vers `proposed_sources`** | carte refusée (§2) |
| `longobardi2022_dysmorphie` | accepter (résumé seul) | 27 articles, 9 167 participants ; moqueries r = 0,42, harcèlement 0,28, violences 0,22 ; le champ `read` est exact |
| `zych2021_violences_amoureuses` | accepter | 23 projets ; harcèlement subi → violences amoureuses subies OR 2,51, ajusté 1,96, plus fort dans les études longitudinales et chez les filles |
| `rubens2018_catastrophes` | accepter | 62 études ; internalisé (hors stress post-traumatique) r = 0,18 ; exposition surtout mesurée par degré (13 comparaisons géographiques, 3 avant/après) |
| `furr2010_catastrophes` | accepter | 96 études ; r groupé 0,19 ; plus fort avec proximité, menace perçue, détresse ; effet surtout dans l'année |
| `deng2024_anxiete_sociale` (de `proposed_sources`) | **déplacer vers `sources`** | nécessaire à la correction 5 retenue |

## 2. Sujets proposés (`outcomes`)

| Sujet | Verdict | Décision et texte |
| --- | --- | --- |
| `ten_difficultes_personnalite` | **accepter** | Catégorie `categorical_personality_diagnosis` (contextuel automatique), libellé et description inchangés. L'entrée 6D10 `agrege_dans` du registre peut rester. |
| `ten_traits_psychopathiques` | **refuser** | Voir §2.1. |
| `ten_impulsivite` | **accepter avec correction** | Description : « Tendance déclarée à agir vite, sans prendre le temps de réfléchir, surtout sous le coup d'émotions fortes. Ce trait varie entre personnes et n'est pas un diagnostic ; il ne se confond pas avec le TDAH, que la carte ne relie pas aux vécus de l'enfance. » (la formule « qui ne vient pas de l'éducation » affirme une étiologie que les sources du fragment ne traitent pas ; la note `neurodeveloppement` suffit). Catégorie `personality_trait`, non contextuel. |
| `ten_preoccupation_apparence` | **accepter** (point sensible tranché §2.3) | Libellé, description et catégorie `symptom_group` inchangés. Ne pas lui donner le code 6B21. |
| `ten_violences_amoureuses_jeunes` | **accepter avec correction** | Description : « Subir des violences psychologiques, physiques ou sexuelles de la part d'un partenaire à l'adolescence ou au début de l'âge adulte. La responsabilité en revient à l'auteur des violences. » (même garde que `ten_revictimisation`). |
| `ten_defenses_peu_adaptees` | **refuser** | Voir §2.4. |

### 2.1 Carte « traits psychopathiques » — refusée
- de Ruiter 2022 est solide sur la forme (47 études, pas de biais de publication, effets r = 0,10 à 0,21), mais : aucune étude prospective (modérateur impossible) ; la négligence n'y est pas séparée ; l'association est plus forte pour les facettes « mode de vie » et « antisociale », c'est-à-dire ce que couvre déjà `ten_traits_personnalite_antisociale`.
- Le mot « psychopathique » appliqué à des personnes qui cochent des violences subies porte un risque réel de stigmatisation et d'auto-étiquetage, même avec une description prudente. Le gain d'information sur une carte de toute façon masquée est nul.
- L'alternative du rapport (ajouter de Ruiter comme seconde source de `abus_physique → ten_traits_personnalite_antisociale`) est **également refusée** : glissement de construit (psychopathie mesurée par PCL ou questionnaires ≠ symptômes de personnalité antisociale DSM) et la source existante suffit pour un lien limité.
- Liens 5 à 8 et lien proposé 3 refusés. Inscrire au registre : `{"referentiel": "Construit", "code": "traits_psychopathiques", "label": "Traits psychopathiques", "statut": "agrege_dans", "raison": "Méta-analyse de Ruiter 2022 : associations faibles à modestes, sans étude prospective ; versants mode de vie et antisocial couverts par ten_traits_personnalite_antisociale. Pas de carte propre, pour ne pas étiqueter des personnes ayant subi des violences.", "sources": ["deruiter2022_psychopathie"]}`.
- Le lien proposé 3 contient une inexactitude : l'association des violences sexuelles avec le score total (r = 0,10) est significative ; seule l'analyse par facette n'est significative que pour une facette.

### 2.2 « Difficultés de personnalité » — acceptée
Johnson 1999 et 2006 sont des cohortes communautaires prospectives avec entretiens structurés ; Miron 2026 confirme pour les violences psychologiques. Le libellé large et le statut contextuel évitent un diagnostic catégoriel. Les effectifs attestés de Johnson 1999 sont très petits (vérifié : 15, 4, 23) : force `limitee` obligatoire.

### 2.3 « Préoccupation pour l'apparence » (texte fermé) — acceptée, limitée
Le résumé donne le plan (méta-analyse à trois niveaux, 27 articles), les effets par forme et les modérateurs : c'est le niveau de lecture de la majorité des sources publiées (223 sur 244 au résumé). Deux réserves imposent `limitee` et des résumés sobres : texte fermé, et **chevauchement de construit** (les échelles de moqueries portent souvent sur l'apparence, comme les symptômes mesurés). Les mentions « études surtout transversales », non vérifiables, sont retirées.

### 2.4 Mécanismes de défense — refusée, statu quo
Une seule étude (72 hommes de l'étude Grant, cohorte d'étudiants de Harvard, donc ni femmes ni milieux variés) ; l'écart observé vers 52 ans **disparaît vers 75 ans** ; le lien était déclaré `clinical_model` alors que sa seule source est une étude empirique et non un modèle publié ; et le questionnaire ne peut pas repérer des défenses cotées en entretien. Garder l'item actuel « Mécanismes de défense » de « Ce que la carte ne couvre pas » ; correction 9 non appliquée. Registre : ajouter `{"referentiel": "Construit", "code": "defenses_maturite_globale", "label": "Maturité globale des défenses", "statut": "a_examiner", "raison": "Une seule cohorte de 72 hommes (Martin-Joy 2017) : enfance chaleureuse et défenses plus mûres vers 52 ans, écart disparu vers 75 ans. Insuffisant pour une carte.", "sources": ["martinjoy2017_defenses"]}`.

### 2.5 « Violences amoureuses des jeunes » — acceptée avec garde
Zych 2021 répond à la question posée (harcèlement subi → violences amoureuses subies, ajusté). La conclusion de Zych évoque des « dispositions antisociales » communes : elle vise l'auteur, pas la victime, et ne doit pas être reprise. D'où la phrase de responsabilité ajoutée au sujet et au résumé. Les liens depuis la maltraitance (Hébert 2019) restent en attente, comme proposé.

## 3. Liens du fragment (`links`)

Force retenue entre crochets.

| # | Paire | Verdict | Texte exact retenu |
| --- | --- | --- | --- |
| 1 | `abus_physique → ten_difficultes_personnalite` | accepter [limitee] | inchangé |
| 2 | `negligence_physique → ten_difficultes_personnalite` | accepter [limitee] | inchangé (la négligence attestée par les services porte surtout sur les besoins matériels et la surveillance : correspondance acceptable) |
| 3 | `negligence_emotionnelle → ten_difficultes_personnalite` | accepter avec correction [limitee] | « Dans une cohorte new-yorkaise suivie jusqu'à 33 ans en moyenne, un faible niveau d'affection parentale relevé lors d'entretiens pendant l'enfance et l'adolescence était associé à plus de troubles de la personnalité à l'âge adulte, en tenant compte des difficultés de l'enfant et des troubles des parents. Une seule cohorte. » |
| 4 | `abus_emotionnel → ten_difficultes_personnalite` | accepter avec correction [limitee] | « Une méta-analyse de 104 études, surtout transversales et fondées sur des souvenirs, trouve que les violences psychologiques sont associées à plus de symptômes de personnalité à l'âge adulte, souvent plus nettement que les violences physiques ou sexuelles, en tenant compte de leur cumul. » |
| 5–8 | `… → ten_traits_psychopathiques` | **refuser** | carte refusée (§2.1) |
| 9 | `abus_emotionnel → ten_impulsivite` | accepter avec correction [limitee] | « Une méta-analyse de 55 études trouve que les violences psychologiques sont, parmi les formes de maltraitance, les plus associées à une impulsivité déclarée par questionnaire. Les analyses sont toutes transversales, un biais de publication est signalé et les rares tests de laboratoire ne montrent pas de lien. » (« 55 études transversales » était inexact : 55 études, analyses transversales) |
| 10 | `abus_physique → ten_impulsivite` | accepter avec correction [limitee] | « Les violences physiques sont associées, en moyenne, à une impulsivité déclarée un peu plus élevée dans une méta-analyse d'analyses transversales. Le sens du lien et la part des facteurs familiaux partagés restent inconnus. » |
| 11 | `abus_sexuel → ten_impulsivite` | accepter [limitee] | inchangé |
| 12 | `negligence_physique → ten_impulsivite` | accepter [limitee] | inchangé |
| 13 | `negligence_emotionnelle → ten_impulsivite` | accepter avec correction [limitee] | « La négligence affective est associée, en moyenne, à une impulsivité déclarée un peu plus élevée dans une méta-analyse d'analyses transversales, où un biais de publication est signalé. » (les tests de laboratoire n'ont pas été analysés par forme) |
| 14 | `abus_emotionnel → pat_social_isolation` | accepter avec correction, **force abaissée** [limitee] | « Deux méta-analyses trouvent que les personnes ayant subi des violences psychologiques se sentent en moyenne plus seules, avec l'un des écarts les plus nets. Il repose surtout sur des souvenirs : les études fondées sur des dossiers ou sur des mesures prises pendant l'enfance ne trouvent pas d'écart net. » |
| 15 | `negligence_emotionnelle → pat_social_isolation` | accepter avec correction, **force abaissée** [limitee] | « La négligence affective est associée, dans deux méta-analyses, à un sentiment de solitude plus fréquent, avec l'un des écarts les plus nets. L'écart diminue avec l'âge et n'apparaît pas nettement dans les études fondées sur des dossiers ou sur des mesures prises pendant l'enfance. » |
| 16 | `abus_physique → pat_social_isolation` | accepter [limitee] | inchangé |
| 17 | `abus_sexuel → pat_social_isolation` | accepter avec correction [limitee] | « Les violences sexuelles sont associées à un sentiment de solitude un peu plus fréquent dans deux méta-analyses d'études surtout transversales, avec un écart plus faible que pour les violences psychologiques. » |
| 18 | `negligence_physique → pat_social_isolation` | accepter avec correction [limitee] | « La négligence matérielle est associée à un sentiment de solitude un peu plus fréquent dans deux méta-analyses d'études surtout transversales, avec un écart plus faible que pour la négligence affective. » |
| 19 | `abus_emotionnel → icd_6b04` | accepter avec correction [limitee] | « Une méta-analyse de 29 études trouve que la maltraitance émotionnelle (violences psychologiques et négligence affective regroupées) est la forme la plus associée à l'anxiété sociale. Le lien porte sur des symptômes, pas forcément sur un trouble diagnostiqué. » |
| 20 | `negligence_emotionnelle → icd_6b04` | accepter avec correction [limitee] | « Dans la même méta-analyse, la maltraitance émotionnelle, qui regroupe négligence affective et violences psychologiques, est associée à plus de symptômes d'anxiété sociale ; l'effet propre de la négligence n'y est pas isolé et les études sont très hétérogènes. » |
| 21 | `abus_physique → icd_6b04` | accepter avec correction [limitee] | « Les violences physiques sont associées à un peu plus de symptômes d'anxiété sociale dans une méta-analyse de 29 études, avec un effet faible. Le lien porte sur des symptômes, pas forcément sur un trouble diagnostiqué. » |
| 22 | `abus_sexuel → icd_6b04` | accepter [limitee] | inchangé |
| 23 | `moqueries_corps → ten_preoccupation_apparence` | accepter avec correction [limitee] | « Une méta-analyse de 27 articles trouve que les moqueries subies sont la forme d'adversité la plus associée aux symptômes de préoccupation excessive pour l'apparence. Elle mesure des symptômes, pas un diagnostic, et ne permet pas de dire dans quel sens va le lien. » |
| 24 | `harcelement → ten_preoccupation_apparence` | accepter [limitee] | inchangé (r = 0,28 : « faible à modéré » exact) |
| 25 | `harcelement → ten_violences_amoureuses_jeunes` | accepter avec correction [limitee] | « Une méta-analyse de 23 projets trouve que les adolescents harcelés à l'école subissent plus souvent des violences dans leurs relations amoureuses, y compris en tenant compte d'autres facteurs, avec un lien plus net dans les études qui suivent les jeunes dans le temps. La responsabilité des violences revient à leur auteur. » |
| 26 | `traumatisme_ponctuel → ten_souffrance_enfance` | accepter avec correction [limitee] | « Des méta-analyses sur les catastrophes naturelles trouvent que les jeunes plus exposés ont en moyenne un peu plus de symptômes de stress post-traumatique, d'anxiété ou de tristesse. Les effets sont modestes, mesurés surtout dans l'année qui suit et souvent selon le degré d'exposition plutôt que par comparaison avec des jeunes épargnés. » (les troubles du comportement sont retirés : la carte les compte à part) |
| 27 | `negligence_emotionnelle → ten_defenses_peu_adaptees` | **refuser** | carte refusée (§2.4) |

Points sensibles tranchés :
- **Anxiété sociale mesurée par questionnaire, reliée à la carte de trouble 6B04** : accepté, comme le lien existant `controle → icd_6b04`, à condition que chaque résumé dise qu'il s'agit de symptômes (fait ci-dessus). Créer une carte « symptômes d'anxiété sociale » doublerait 6B04 sans bénéfice.
- **Solitude** : la force `moderee` proposée n'est pas tenable. de Heer montre que l'écart n'apparaît pas dans les études prospectives ni dans celles qui utilisent des dossiers. Les deux constructs reposent alors sur la même perception rétrospective, d'où un biais de méthode commune. Les cinq liens passent à `limitee` ; le rang relatif (violences psychologiques et négligence affective plus nettes) reste dit dans le texte.

## 4. Liens proposés (`proposed_links`)

| # | Paire | Verdict |
| --- | --- | --- |
| 1 | `exil → icd_6a20` | **refuser** (rester hors graphe). Brandt 2019 porte sur des réfugiés de tous âges ; Hollander 2016 suit des personnes arrivées jusqu'à environ 27 ans sans analyse par âge à l'arrivée (vérifié dans le texte) ; un trouble grave ; et un risque réel de stigmatiser une population déjà exposée. Corriger le registre `exil_comparaison` (voir §6). |
| 2 | `exil → ten_souffrance_enfance` | laisser en proposition. Eltanamly mesure, comme Rubens, une dose à l'intérieur d'une population exposée, mais avec 60 % d'études (Palestine/Israël), surtout transversales, centrées sur la parentalité, et des effets par issue non lus. À rouvrir après lecture des tableaux S2–S3. |
| 3 | `abus_sexuel → ten_traits_psychopathiques` | **refuser** (carte refusée ; motif inexact, voir §2.1) |
| 4 | `abus_physique → ten_violences_amoureuses_jeunes` | laisser en attente, comme proposé : le résumé de Hébert ne détaille pas les formes |
| 5 | `abus_emotionnel → pat_self_criticism` | **refuser** : l'autocompassion faible n'est pas l'autocritique. Garder `autocompassion` « à examiner » au registre |

## 5. Corrections proposées (`proposed_corrections`)

| # | Cible | Verdict | Texte exact |
| --- | --- | --- | --- |
| 1 | `att_anxieux` (description) | accepter avec correction | « Craindre de perdre l'autre et chercher beaucoup de signes de son affection. Ce que l'on appelle couramment « dépendance affective », qui n'est pas un diagnostic, s'en rapproche. » (« On parle parfois de… » laissait croire à un synonyme ; Cavalli donne une corrélation r = 0,39, pas une identité) |
| 2 | `ten_codependance` (description) | **refuser** | Aucune source lue ne relie codépendance et « dépendance affective » : la première porte sur la prise en charge d'un proche en difficulté, la seconde sur le besoin du partenaire. Garder la description actuelle. **Une seule** mention de « dépendance affective » est donc retenue, sur `att_anxieux`. L'item existant « « Dépendance affective » » de « Ce que la carte ne couvre pas » peut rester tel quel. |
| 3 | champ `also_known_as` | accepter comme évolution future du code (hors fragment, non intégrable maintenant) | — |
| 4 | `pat_social_isolation` | accepter ; libellé recommandé « Solitude ou isolement subis » (la page liée traite déjà de la solitude choisie ou subie) | description : « Se sentir seul, avec moins de liens et de soutien que ce que l'on souhaiterait. » |
| 5 | `harcelement → icd_6b04` | accepter avec correction [limitee, sources `gonzalez2017` + `deng2024_anxiete_sociale`] | « Les jeunes harcelés ont en moyenne plus de symptômes d'anxiété sociale. Dans les études qui suivent les jeunes dans le temps, c'est surtout l'inverse qui s'observe : l'anxiété sociale annonce un harcèlement ultérieur. Cela ne rend jamais l'enfant responsable du harcèlement qu'il subit. » |
| 6 | `traumatisme_ponctuel → anx_generale` (source ajoutée) | accepter avec résumé mis à jour [limitee, `copeland2007` + `rubens2018_catastrophes`] | « Une cohorte d'enfants et une méta-analyse sur les catastrophes naturelles relient les événements traumatiques à un peu plus de difficultés anxieuses chez les jeunes, surtout quand les événements se cumulent ou que l'exposition est forte. La méta-analyse ne sépare pas anxiété et tristesse ; les effets sont modestes. » |
| 7 | `traumatisme_ponctuel → dep_generale` (source ajoutée) | accepter avec résumé mis à jour [limitee, idem] | « Une cohorte d'enfants et une méta-analyse sur les catastrophes naturelles relient les événements traumatiques à un peu plus de difficultés dépressives chez les jeunes, surtout quand les événements se cumulent ou que l'exposition est forte. La méta-analyse ne sépare pas tristesse et anxiété ; les effets sont modestes. » |
| 8 | item « Troubles de la personnalité autres que borderline » | accepter avec correction (psychopathie refusée) | « La CIM-11 décrit la personnalité par dimensions. Seules les associations avec le trouble borderline sont assez solides pour une carte de trouble ; les difficultés de personnalité au sens large figurent comme repère de recherche, jamais comme diagnostic. » |
| 9 | item « Mécanismes de défense » | **refuser** (carte refusée) | texte actuel conservé |
| 10 | item « Exil, guerre et migration comme causes mesurées » | accepter | texte proposé, inchangé |

Le harcèlement → anxiété sociale signalé comme « sous-évalué » par la relecture clinique : **la force `limitee` est maintenue**. L'association transversale est modérée (r = 0,27 sur 133 études), mais les 19 études longitudinales ne montrent pas que le harcèlement annonce l'anxiété sociale (β = 0,012, non significatif) ; c'est l'inverse qui est observé. Présenter ce lien comme plus fort dans le sens vécu → sujet serait trompeur. `gonzalez2017` reste pertinent (victimisation par les pairs liée à l'anxiété sociale en transversal).

## 6. Registre hors périmètre (`out_of_scope`)

Toutes les entrées sont acceptées, avec trois ajustements :
- `dependance_affective` : remplacer la fin de la raison par « Mention proposée dans la description d'Attachement anxieux seulement. »
- `exil_comparaison` : ajouter « Les comparaisons de réfugiés avec des natifs pour les troubles psychotiques (Brandt 2019, Hollander 2016) ne portent pas sur une exposition limitée à l'enfance : non retenues. » et ajouter `brandt2019_refugies` et `hollander2016` (déjà déclaré ailleurs : ne pas le redéclarer dans `proposed_sources` d'un fragment lu par le générateur) aux sources.
- ajouter les deux entrées `traits_psychopathiques` (§2.1) et `defenses_maturite_globale` (§2.4).

## 7. Cinq liens existants signalés par la relecture clinique (C5a)

| Lien | Vérification | Verdict | Texte exact |
| --- | --- | --- | --- |
| `negligence_physique → icd_6a60` (moderee, Palmier-Claus 2016) | Tableau 1 : négligence physique OR 2,26 (1,74–2,93), **7 études**, surtout cas-témoins avec le CTQ rétrospectif ; taux d'adversité **proches de ceux des témoins psychiatriques** (non spécifique) ; types I/II non distingués (4 études) | **corriger : force `limitee`** | « Une méta-analyse de 7 études cas-témoins, fondées surtout sur des questionnaires rétrospectifs, trouve plus de négligence matérielle rapportée par les personnes ayant un trouble bipolaire que par des témoins sans trouble, mais des taux proches de ceux d'autres patients en psychiatrie. Types I et II réunis. » |
| `parent_trouble → icd_6b05` (moderee, Lawrence 2019) | Résumé : 25 études familiales, diagnostics validés chez parents et enfants, groupe de comparaison ; risque accru d'anxiété chez l'enfant (RR 1,76), en particulier anxiété généralisée, **anxiété de séparation** et phobie spécifique ; pas de transmission du même trouble. Texte intégral non accessible (403) : RR propre à l'anxiété de séparation non vérifié. Le vécu coché est plus large (dépression, suicidalité) que la population étudiée (troubles anxieux) | accepter avec correction, **force `moderee` maintenue** (plan familial avec groupe témoin et diagnostics ; déjà plafonné par la règle de transmission familiale de C5a) | « Une méta-analyse d'études familiales trouve que les enfants et adolescents dont un parent a un trouble anxieux diagnostiqué ont plus souvent un trouble anxieux, dont l'anxiété de séparation, que ceux dont les parents n'en ont pas. Gènes et environnement familial s'y mêlent ; il n'y a pas de transmission propre à un trouble précis. » |
| `deces_proche → icd_6b40` (critere, CIM-11) | Lignes directrices CIM-11 : « experiencing the sudden, unexpected or violent death of a loved one » (version d'essai sur le terrain, PMC6225017). Le lien est exact, mais le vécu coché couvre tous les décès. Le trouble du deuil prolongé (6B42) serait plus juste pour un deuil ordinaire, mais aucune étude ne le relie à l'enfance (Buur 2024) et la carte n'existe pas | accepter avec correction [critere] | « Apprendre la mort soudaine, inattendue ou violente d'un proche fait partie des événements que la CIM-11 retient pour le stress post-traumatique. Un deuil attendu n'en fait pas partie ; la CIM-11 décrit à part le trouble du deuil prolongé, que la carte ne relie pas à l'enfance faute d'études. » |
| `deces_proche → icd_6a20` (limitee, Varese 2012 + Burrell 2022) | Varese : décès d'un parent non significatif ; Burrell : registre norvégien (655 477 personnes), décès d'un parent **par cause externe** avant 18 ans → risque accru de troubles psychotiques, surtout après suicide, chute ou intoxication | accepter avec correction [limitee] | « Le lien entre décès d'un parent et troubles psychotiques est incertain : non significatif dans une méta-analyse, présent dans un grand registre norvégien après un décès par suicide ou accident. Les deux sources ne portent pas sur les mêmes décès. » |
| `harcelement → icd_6b04` (limitee, González-Díez 2017) | voir §5, correction 5 | accepter la correction 5 [limitee] | texte de la correction 5 |

À signaler à l'intégrateur, hors des cinq liens : `negligence_emotionnelle → icd_6a60` (moderee) repose sur la même méta-analyse et le même plan (7 études, OR 2,62, non spécifique). Par cohérence, la faire passer aussi en `limitee`, avec un résumé calqué sur celui ci-dessus ; à valider par le propriétaire.

## 8. Liste prête à intégrer

Fragment à intégrer (renommer hors de `propositions/` selon l'usage de l'intégrateur ; il se classe avant `liens-humeur…`, `liens-transparence…` et `liens-v2-*`, donc ne redéclare rien d'existant) :

**Sources (`sources`, 11)** : `deheer2024_solitude`, `curtis2025_solitude`, `liu2023_anxiete_sociale`, `liu2019_impulsivite`, `johnson2006_personnalite`, `miron2026_abus_emotionnel`, `longobardi2022_dysmorphie`, `zych2021_violences_amoureuses`, `rubens2018_catastrophes`, `furr2010_catastrophes`, `deng2024_anxiete_sociale` (déplacée depuis `proposed_sources`). Vers `proposed_sources` : `deruiter2022_psychopathie`, `martinjoy2017_defenses`.

**Sujets (`outcomes`, 4)** : `ten_difficultes_personnalite` (inchangé), `ten_impulsivite` (description §2), `ten_preoccupation_apparence` (inchangé), `ten_violences_amoureuses_jeunes` (description §2).

**Liens (`links`, 21, tous `limitee`, `empirical_association`)** : 1, 2, 3*, 4*, 9*, 10*, 11, 12, 13*, 14* (force abaissée), 15* (force abaissée), 16, 17*, 18*, 19*, 20*, 21*, 22, 23*, 24, 25*, 26* (* = texte du §3). Retirés : 5, 6, 7, 8, 27.

**Corrections dans `corrections.json` ou l'existant** :
- `outcome_overrides` : `att_anxieux` (description §5-1) ; `pat_social_isolation` (description §5-4, libellé « Solitude ou isolement subis »).
- liens existants : `harcelement → icd_6b04` (§5-5, + `deng2024_anxiete_sociale`) ; `traumatisme_ponctuel → anx_generale` et `→ dep_generale` (§5-6 et 5-7, + `rubens2018_catastrophes`) ; `negligence_physique → icd_6a60` (`limitee`, §7) ; `parent_trouble → icd_6b05` (texte §7) ; `deces_proche → icd_6b40` (texte §7) ; `deces_proche → icd_6a20` (texte §7).
- `notes.hors_perimetre` : items « Troubles de la personnalité autres que borderline » (§5-8) et « Exil, guerre et migration… » (§5-10). Item « Mécanismes de défense » inchangé.
- registre : ajustements du §6.

**Non intégrés** : cartes `ten_traits_psychopathiques` et `ten_defenses_peu_adaptees` ; liens proposés 1, 3, 5 refusés ; 2 et 4 en attente ; corrections 2 et 9 refusées ; correction 3 (`also_known_as`) reportée à une évolution du code.

**Contrôles restant à l'intégrateur** : `python3 scripts/build_cartographie.py --check`, `node scripts/build_cartographie_module.mjs --check`, nouvel audit des correspondances (l'empreinte sera périmée), tests de C5a. Tous les résumés ci-dessus ont été vérifiés par script : 183 à 337 caractères, sans mot interdit ni pourcentage. Lacunes à conserver dans l'état de couverture : textes fermés de Longobardi, Liu 2023 et Lawrence 2019 ; tableaux S2–S3 d'Eltanamly ; Credé 2023 non lu ; décision sur `negligence_emotionnelle → icd_6a60`.
