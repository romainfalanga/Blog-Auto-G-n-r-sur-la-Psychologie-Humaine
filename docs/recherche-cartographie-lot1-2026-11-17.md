# Recherche du lot 1 — humeur, anxiété, TOC et périnatalité

Lot daté du 17 novembre 2026 ; recherches effectuées pendant la session du 16 novembre UTC. Périmètre : CIM-11 6A6, 6A7, 6B0, 6B2, 6E2 et correspondances DSM. Source de vérité : `data/cartographie.json` (58 facteurs, 82 sujets, 425 liens, 244 sources avant intégration).

Livrable : `editorial/cartographie/liens-humeur-anxiete-toc-2026-11-17.json`.

## Résultat et portée

- **8 nouveaux liens admis**, dont 6 modérés et 2 limités ; aucun lien fort.
- **3 nouveaux sujets** : TOC, symptômes dépressifs périnataux, symptômes prémenstruels importants. Les deux derniers sont des groupes symptomatiques contextuels sans code diagnostique.
- **9 sources nouvelles** : 4 empiriques lues dans leurs méthodes et résultats et 5 notices officielles de classification. Les sources existantes OMS et Palmier-Claus sont réutilisées par leur identifiant.
- **4 liens bipolaires conditionnels**, ignorés par le générateur, et **5 corrections proposées**. Aucun de ces liens ne doit être activé sans décision sur le regroupement des types I/II.
- **141 décisions de registre** : 99 CIM et 42 DSM, ces dernières comprenant aussi des spécifications et une correspondance avec le chapitre somatique. Ce ne sont pas 42 nouveaux diagnostics couverts.

Le catalogue officiel fournit **101 codes distincts** dans les cinq familles demandées. Deux racines reçoivent des liens nouveaux (6B01, 6B20) ; les 99 autres ont une décision individuelle dans `out_of_scope`, y compris sous-codes, rémissions, insight, catégories autres/non précisées. Répartition des décisions : 69 `agrege_dans`, 44 `hors_perimetre_justifie`, 21 `a_examiner`, 7 `aucune_preuve_trouvee`. `agrege_dans` n’attribue aucune couverture exacte supplémentaire.

Le nombre de liens est inférieur à la fourchette indicative du plan. Les principales restrictions sont documentées : articles disponibles au seul résumé, expositions cumulées impossibles à ramener à une case, scores symptomatiques confondus avec un diagnostic, et sous-types non stratifiés. Le registre est un inventaire complet du périmètre CIM demandé ; la recherche bibliographique est ciblée et reproductible, sans prétendre constituer une revue systématique exhaustive de toute la littérature.

## Vérification des classifications

### OMS : codes et contextes

[CDDR officiel, OMS, 2024](https://www.who.int/publications/i/item/9789240077263). **Lecture : sections**, liste des catégories p. 44–55 et 84 ; TOC p. 300–301 ; périnatalité p. 639–643, PDF officiel conservé dans l’espace d’audit. Notice existante `oms_cddr`, à enrichir plutôt qu’à redéclarer.

Passage vérifié, périnatalité : « commencing during pregnancy or within about 6 weeks after delivery » (11 mots). 6E20 est sans symptômes psychotiques, 6E21 avec symptômes psychotiques, 6E2Z non précisé. Une mesure dépressive jusqu’à un an postpartum ne couvre donc pas automatiquement ces codes. Quand un trouble spécifique répond aux exigences, le CDDR prévoit aussi sa classification propre.

Sous-codes contrôlés : épisodes et rémissions 6A60/6A61/6A70/6A71 ; insight 6B20–6B24 ; 6B25.0 trichotillomanie et 6B25.1 excoriation ; catégories Y/Z. Les qualificatifs ne sont pas transformés en cartes autonomes.

### APA : correspondances, sans inventer de codes DSM

[Répertoire officiel DSM-5-TR](https://www.psychiatry.org/psychiatrists/practice/dsm/educational-resources/dsm-5-tr-fact-sheets). **Lecture : index et fiches intégrales suivantes** :

- [Bipolar I and Bipolar II Disorders](https://www.psychiatry.org/getContentAsset/98fd2c17-93f0-42cd-9f41-755d77b862a5/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/APA-DSM5TR-BipolarIandBipolarIIDisorders?language=en), trois pages : distinction I/II et changements de critères/spécifications.
- [Other Specified Bipolar and Related Disorder](https://www.psychiatry.org/getContentAsset/ffc643b7-9382-4a29-a8c8-868dabcced10/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/APA-DSM5TR-OtherSpecifiedBipolarandRelatedDisorder?language=en), une page : catégorie autre spécifiée.
- [Other Specified Depressive Disorder](https://www.psychiatry.org/getContentAsset/9c456082-b1ed-4a7c-84f7-7b340fa10b01/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/APA-DSM5TR-OtherSpecifiedDepressiveDisorder?language=en), une page : exemple d’épisode superposé.
- [Unspecified Mood Disorder](https://www.psychiatry.org/getContentAsset/4d6aab2f-87cf-4b22-afc4-8b9035076051/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/APA-DSM5TR-UnspecifiedMoodDisorder?language=en), une page : catégorie rétablie dans les deux chapitres, comptée une seule fois.
- [Olfactory Reference Disorder](https://www.psychiatry.org/getContentAsset/06c72f8b-e85f-480d-937e-e90fd7d1ed3f/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/APA-DSM5TR-OlfactoryReferenceDisorder?language=en), une page. Passage : « added as an example of presentations that can be specified under an “other specified” designation » (18 mots). Contrairement au code CIM 6B22 autonome, le DSM-5-TR en fait un exemple dans la catégorie autre spécifiée.

Autres notices officielles lues :

| Source | Lecture et passage vérifié | Usage |
| --- | --- | --- |
| [APA — Depression, avril 2024](https://www.psychiatry.org/patients-families/depression/what-is-depression) | Sections des troubles apparentés. « PMS involves fewer and less severe symptoms than PMDD. » | Distinction PMS/PMDD ; périnatalité, persistance et bipolarité. |
| [APA — Anxiety Disorders](https://www.psychiatry.org/patients-families/anxiety-disorders/what-are-anxiety-disorders) | Sections des sept troubles ; passage du mutisme : « Children with selective mutism do not speak in some social situations where they are expected to speak » | Vérification des intitulés et contexte infantile ; aucun lien d’adversité déduit de la notice. |
| [APA — chapitre TOC, DSM-5, 2013](https://www.psychiatry.org/getContentAsset/8643622a-0412-4ca1-9de6-0624e882e935/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/APA-DSM5-Obsessive-Compulsive-Disorder.pdf?language=en) | Fiche intégrale, deux pages ; intitulés « hoarding disorder » et « excoriation (skin-picking) disorder ». | Les troubles apparentés sont séparés ; la personnalité obsessionnelle-compulsive n’est pas le TOC. Fiche ancienne complétée par les modifications TR. |
| [APA — DMDD, DSM-5, 2013](https://www.psychiatry.org/getContentAsset/27226543-3ae7-4d5b-be15-7089e810240e/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/APA-DSM5-Disruptive-Mood-Disregulation-Disorder.pdf?language=en) | Fiche intégrale, deux pages ; passage « The onset of symptoms must be before age 10 » | Tableau de l’enfant/adolescent, distinct du trouble bipolaire ; pas de carte adulte créée. |

**Limite de vérification DSM :** le manuel DSM-5-TR intégral sous licence n’a pas été lu. Le registre examine aussi les catégories induites par substance/médicament, dues à une affection médicale, autres spécifiées et non spécifiées. Pour celles que les notices ne détaillent pas, il signale que le libellé français exact et les critères complets restent à confirmer. Aucun nombre CIM-10-CM ou CIM-11 n’est présenté comme un code DSM autonome. La correspondance hypocondrie/anxiété de maladie est thématique et relève aussi du chapitre somatique, donc du lot 2.

## Preuves admises : passages, mesure et limites

Les résumés destinés au site sont volontairement plus courts que ces contrôles. Les chiffres ci-dessous servent à vérifier le sens et l’incertitude, pas à calculer un risque individuel.

### Ou et al., 2021 — deux liens TOC limités

[Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7854900/), DOI 10.3389/fpsyt.2020.612586. **Lecture : méthodes, résultats, tableau 2 et discussion**, texte intégral Europe PMC.

Dix études, 1 611 personnes avec TOC, majoritairement transversales. Violences émotionnelles : corrélation avec sévérité totale, r = 0,11 [0,03–0,19]. Abus sexuels : obsessions, r = 0,13 [0,03–0,23], pas score total. Passage : « The correlation was weak. »

Décision : deux liens `limitee`. Pas de nouveau lien pour violences physiques, négligences physiques/émotionnelles ou diagnostic incident. On ne déduit pas un risque d’apparition d’une association de sévérité chez des adultes déjà diagnostiqués.

### Zhang et al., 2023, publication en ligne 2021 — trois liens panique modérés

[Notice PubMed](https://pubmed.ncbi.nlm.nih.gov/34776038/), [manuscrit accepté UEA](https://ueaeprints.uea.ac.uk/id/eprint/81760/1/Accepted_Manuscript.pdf). **Lecture : méthodes/résultats p. 6–14**, tableaux des expositions.

34 études, 192 182 participants. Estimations après examen des valeurs extrêmes : abus sexuels OR 1,92 [1,37–2,46] ; violences physiques 1,71 [1,37–2,05] ; alcoolisme parental 1,83 [1,24–2,43]. Passage : « the majority of the studies relied on retrospective reporting ».

Rejets : émotionnel non concluant ; séparation/perte parentale non décomposable en divorce, décès ou absence. `parent_addiction` est limité explicitement à l’alcoolisme dans le résumé.

### Lombardi et al., 2023, publication en ligne 2021 — un lien périnatal modéré

[Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9660263/), DOI 10.1177/15248380211021611. **Lecture : méthodes, résultats et tableaux 1–3**, particulièrement le sous-groupe enfance.

36 études, 45 effets ; le sous-groupe enfance comprend 27 effets, OR 1,53 [1,33–1,73]. Passage : « Twenty-seven effect sizes from studies focused on childhood sexual victimization ».

Le résultat concerne des femmes de la grossesse à un an postpartum, avec symptômes et diagnostics mélangés. Décision : carte symptomatique contextuelle ; ni 6E20 ni 6E21 revendiqué. Les expositions sexuelles sur toute la vie ne sont pas attribuées à l’enfance ; le sous-groupe enfance a été contrôlé séparément.

### Bertone-Johnson et al., 2014 — deux liens prémenstruels modérés

[Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC4158950/). **Lecture : méthodes, résultats et tableaux 3–4**, texte intégral PMC.

Nurses’ Health Study 2 : 1 018 cas et 2 277 témoins ; violences émotionnelles élevées OR ajusté 2,5 [1,6–3,8], violences physiques sévères 1,8 [1,3–2,6]. Souvenirs d’abus recueillis en 2001, parfois après le début des symptômes : ce n’est pas une mesure prospective de l’exposition. Passage : « the CTQ corporal punishment items were not related to risk. »

Deux liens vers le syndrome prémenstruel important, **pas vers PMDD**. Aucun lien ajouté pour abus sexuels ni punitions corporelles, dont les associations retenues n’étaient pas concluantes après ajustement.

## Proposition bipolaire conditionnelle

[Palmier-Claus et al., 2016](https://doi.org/10.1192/bjp.bp.115.179655), source existante `palmierclaus2016`. **Lecture : méthodes, résultats, tableau 1, comparaison des types et limites**, intégral éditeur.

Le tableau vérifie les associations des violences physiques/sexuelles et négligences physiques/émotionnelles aux troubles bipolaires regroupés (OR 2,26–2,86). Passage : « No significant difference in childhood adversity was observed between type 1 and type 2 bipolar disorder ».

Cela ne valide pas un effet propre à chaque type. Proposition : `icd_6a60` devient un groupe clinique sans code exact, puis l’intégrateur peut examiner les quatre liens conditionnels. Ne pas ajouter 6A61 et ne pas dupliquer le lien émotionnel existant. La perte parentale n’est pas ajoutée.

## Sources examinées sans admission de liens

| Source | Lecture réelle et résultat utile | Décision |
| --- | --- | --- |
| [Racine et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7802164/) | Méthodes/résultats intégraux lus. 15 études ; ACE cumulatives rétrospectives. Passage : « In all studies, mothers provided a retrospective report of their ACEs. » | Les corrélations prénatales/postnatales ne désignent pas une case particulière ; aucun éventail artificiel de liens. |
| [Scott, Smith & Ellis, 2010](https://jamanetwork.com/journals/jamapsychiatry/fullarticle/210834) | Méthodes et résultats/tableaux lus ; 2 144 jeunes de 16–27 ans, suivi administratif médian 10 ans. Passage : « child protection agency history, used here as a proxy for maltreatment ». | Exposition globale aux services de protection, non à un vécu élémentaire. Même les résultats significatifs sur phobie spécifique, dysthymie ou TOC ne sont pas ventilés sur les 58 cases. |
| [Longobardi et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35085863/) | Résumé et notice, 27 articles, symptômes dysmorphiques. Éditeur et dépôt Torino : intégral non obtenu. | `a_examiner` pour 6B21 ; ni symptômes ni résumé convertis en diagnostic relié. |
| [Chia et al., 2021](https://pubmed.ncbi.nlm.nih.gov/34717158/) ; [notice ANU](https://researchportalplus.anu.edu.au/en/publications/interpersonal-attachment-early-family-environment-and-trauma-in-h/) | Résumé lu, 39 études. Dépôt ANU renvoie à une demande de copie ; aucun PDF intégral lu. | `a_examiner` pour accumulation ; séparer enfance/vie entière et vérifier les constructions familiales avant tout lien. |
| [Fu et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38516894/) ; [résumé éditeur](https://journals.sagepub.com/doi/abs/10.1177/15248380241235639) | Résumé seulement : méta-analyse postpartum, 24 études et 73 effets. | Pas d’admission. En particulier, incarcération d’un membre de famille ≠ incarcération d’un parent. |
| [Liu et al., 2025](https://pubmed.ncbi.nlm.nih.gov/39824317/) | Résumé seulement, 27 articles longitudinaux ; éditeur inaccessible. | Les listes de facteurs et diagnostics ne fournissent pas les cellules facteur × diagnostic. Aucune multiplication automatique pour TAG, phobie spécifique, agoraphobie ou panique. |
| [Cohorte anxiété de santé à 30 ans](https://pubmed.ncbi.nlm.nih.gov/35176517/) | Résumé seulement ; étude sur scores d’anxiété de santé, pas nécessairement diagnostic. | Pas de nouveau lien à 6B23. |
| [Revue anxiété sociale, 2023](https://www.sciencedirect.com/science/article/pii/S0165032723002434) | Notice/résumé repérés ; méthodes/résultats non lus. | Diagnostic et symptômes à séparer ; pas de lien admis. |
| [Revue sexe/genre et anxiété, IJERPH 2026](https://www.mdpi.com/1660-4601/23/11/1422) | Recherche indexée du 16 novembre 2026 ; ouvertures page, DOI et version HTML/XML en erreur. **Ni intégral ni résumé de page lu.** | Piste récente à reprendre, aucun résultat extrait admis. |

Ces pistes ne sont pas ajoutées à `sources` comme preuves de liens. Leurs URL sont conservées dans le rapport et, quand pertinent, dans `research_urls` du registre. Une notice indexée ou un résumé lu n’est jamais présenté comme une lecture intégrale.

## Requêtes ciblées et décisions sans nouveau lien

Les requêtes ont combiné les diagnostics avec `childhood maltreatment`, `childhood adversity`, `systematic review`, `meta-analysis` et/ou `cohort`. Recherches spécifiques :

| Diagnostic ou famille | Requête ou angle contrôlé | Motif de décision |
| --- | --- | --- |
| Cyclothymie | `"cyclothymic" "childhood maltreatment" systematic` | Tempéraments affectifs retrouvés, sans preuve diagnostique propre admissible. |
| Dépression persistante | `dysthymia childhood maltreatment meta-analysis cohort diagnosis` | Sujet déjà présent ; Scott cumule la maltraitance. Pas de nouveau lien élémentaire. |
| Dépression unique/récurrente | Contrôle des agrégats et études prospectives | Les alias restent réservés ; pas de séparation sur la base d’un score dépressif. |
| Trouble mixte anxiodépressif | `mixed depressive anxiety disorder childhood adversity` | [Travail de classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC9291022/) repéré ; pas d’association élémentaire admissible lue. |
| TAG, phobie spécifique | `"generalized anxiety disorder" "childhood" "meta-analysis" maltreatment` ; `"specific phobia" "childhood" "maltreatment" cohort meta-analysis` | Sources agrégées ou lecture limitée, voir Liu/Scott. |
| Anxiété sociale/séparation | `"social anxiety" "childhood maltreatment" meta-analysis emotional abuse` ; `"separation anxiety" "childhood maltreatment" "systematic review"` | Cartes déjà présentes ; pas de nouvelle association diagnostique spécifique intégralement contrôlée. |
| Mutisme sélectif | `selective mutism childhood trauma systematic review` | [Revue socioculturelle](https://doi.org/10.1177/13634615221146435) non lue intégralement ; ne pas supposer un traumatisme. |
| DMDD | `"disruptive mood dysregulation" "maltreatment" cohort review` | Travaux cliniques isolés ; pas de synthèse/cohorte développementale retenue. Âge enfant/adolescent, pas jeune enfant exclusivement. |
| Référence olfactive | `olfactory reference disorder childhood trauma` | [Cas clinique](https://pubmed.ncbi.nlm.nih.gov/41536747/) insuffisant pour le lien demandé. |
| Trichotillomanie | `trichotillomania childhood trauma systematic review` | Mémoire et recherches hétérogènes repérés ; aucun article admissible vérifié. |
| Excoriation | `excoriation childhood trauma` | [Étude cas-témoins](https://pmc.ncbi.nlm.nih.gov/articles/PMC5353104/) et cas cliniques, pas de synthèse/cohorte retenue. |
| Périnatalité psychotique | Contrôle des synthèses de dépression et critères 6E21 | Aucun effet propre à la catégorie psychotique dans les sources vérifiées. Absence de lien n’est pas preuve d’absence. |
| Résiduels et sous-codes | Contrôle individuel du catalogue CDDR | Pas d’effet propre démontré pour insight, sévérité, épisode actuel, rémission, autre/non précisé. |

Les facteurs non représentés dans les huit liens ne sont pas déclarés sans rapport avec ces sujets. Le lot n’ajoute rien lorsque la source ne sépare pas les expositions, ne mesure pas le diagnostic voulu ou ne donne pas accès aux méthodes/résultats. Les absences du registre sont donc des décisions éditoriales traçables, pas des négations cliniques.

## Contextes et intégration

- Aucun sujet enfant seulement n’est créé. DMDD et mutisme sélectif restent au registre ; leurs contextes sont explicités. Ne pas appliquer `young_child_only` au DMDD, qui comprend les adolescents.
- Les cartes périnatale et prémenstruelle portent `contextual_only: true`. Elles ne sont pas pertinentes pour toute personne et ne doivent pas apparaître comme des diagnostics.
- Les deux liens du TOC concernent la sévérité parmi les personnes déjà concernées. L’interface et son score de convergence doivent conserver cette limite explicite.
- La carte bipolaire publiée porte actuellement le code exact du type I alors que la source réanalysée est agrégée. La correction proposée doit être évaluée avec les liens existants, notamment la source de transmission familiale, avant activation des quatre propositions.
- Les anciens sujets, facteurs, alias et paires restent intacts. Aucun fichier généré, audit, code applicatif ou ressource des autres lots n’est modifié.

## Contrôles réalisés

Contrôle local contre le JSON publié : aucun identifiant de sujet/source dupliqué, aucune URL de source nouvelle déjà déclarée, aucune paire nouvelle ni conditionnelle déjà présente ; références de sources valides ; trois sujets nouveaux avec au moins un lien. Résumés des huit liens : 215 à 294 caractères, sous la limite de 350.

Appel du générateur en lecture seule, sans écrire de fichier : 58 facteurs, 91 sujets, 461 liens, 253 sources dans l’état partagé au moment du contrôle. Hors erreur attendue d’empreinte d’audit, le seul message était `lien triangulation → dep_generale : résumé de plus de 350 caractères`, provenant d’un autre fragment, signalé à l’intégrateur. Aucun problème du lot 1 identifié par cet appel. L’état partagé peut évoluer avec les autres agents ; les nombres globaux sont un instantané, pas le bilan final d’intégration.
