# Audit de la Cartographie : couverture, preuves et interprétation

Audit du 16 novembre 2026. Périmètre : données et calcul de la Cartographie, correspondances avec les catalogues internes, contexte transmis à Ambre. Les articles et parcours éditoriaux ne sont pas réécrits. Aucun déploiement ni appel à un modèle externe n’a été effectué pour cet audit.

## Conclusion et corrections prioritaires

La Cartographie est une sélection de rapprochements autour de vécus de l’enfance. Elle ne couvre pas tous les troubles du DSM-5-TR ou de la CIM-11 et ne constitue pas un instrument diagnostique. Avant correction, elle contenait **81 sujets, dont 17 schémas**, et un cumul de liens faibles ou théoriques pouvait être présenté comme une convergence. Après les changements décrits ci-dessous : **58 facteurs, 82 sujets, 425 liens, 244 références ; les 18 schémas du modèle de Young sont représentés**. Cela atteste la présence de ces concepts, pas leur validation chez une personne.

Les correctifs réalisés portent sur des écarts reproductibles : fin de l’addition des preuves et du multiplicateur d’intensité ; séparation entre associations empiriques, contextes de classification, modèles et définitions ; regroupement prudent des réponses qui se recouvrent ; protection des tableaux du jeune enfant ; conservation d’un « non » explicite ; conservation des références et de leur nature pour Ambre. Les 161 repères internes ont maintenant chacun une décision sémantique. Les 22 chapitres de la section II du DSM-5-TR ont un statut de présence thématique ; la comparaison de tous les diagnostics et la relecture clinique de chaque lien restent ouvertes.

**Formulation à conserver dans le produit :** des facteurs associés, même cumulés, ne suffisent pas à établir un trouble et ne sont pas nécessairement requis pour qu’il apparaisse. Les manifestations actuelles, leur durée, le retentissement, les autres explications et le contexte doivent être examinés. Certains diagnostics comportent des conditions particulières : cela n’autorise pas à transformer chaque vécu de la carte en condition nécessaire ou suffisante.

## Ce qui a été vérifié

Lecture de `AGENTS.md`, des documents de reprise, de l’état de couverture, du registre clinique, des catalogues psychologiques, des deux relectures de la Cartographie et de leurs corrections. Inspection des générateurs, des données publiques, du module Worker, du calcul navigateur, de la validation, du résumé, des preuves, des suggestions initiales et de la préparation des prompts.

Un instantané antérieur aux changements est conservé dans `/workspace/work/clinical-audit-20261116/baseline/`. Le script `/workspace/work/clinical-audit-20261116/audit.mjs` compare cet instantané avec le dépôt courant ; `audit-results.json` contient les empreintes, les 858 lignes cliniques, les 161 repères, l’inventaire des sujets et les cas de calcul. Il s’agit d’un audit structurel reproductible, **pas d’une certification de chaque affirmation médicale**.

## Couverture réelle et critères de comptage

| Objet | Résultat contrôlé | Sens exact |
| --- | ---: | --- |
| Facteurs proposés | 58 | Tous possèdent au moins un lien ; pas un inventaire des expériences humaines. |
| Sujets après correction | 82 | 22 entrées historiques de type `trouble`, 18 schémas, 3 repères d’attachement, 39 tendances. |
| Références exactes à un code CIM-11 | 19 sur 858 codes internes | Égalité du champ `code`, sans hériter automatiquement des sous-codes. |
| Codes regroupés | 3 | 6B80, 6B81 et 6B82 apparaissent dans une seule entrée agrégée ; pas trois tableaux individuels vérifiés. |
| Codes sans référence exacte | 839 | Comprend les 3 codes agrégés. Ce nombre ne mesure pas un taux de couverture clinique. |
| Exclusions documentées de l’inférence à partir de l’éducation | 38 codes | Catégories 6A00–6A05 et leurs précisions ; ne pas fabriquer des causes familiales des troubles neurodéveloppementaux. |
| Autres codes non explicitement référencés | 798 | Pas de correspondance formalisée ; ni preuve d’absence d’association dans la littérature, ni obligation de les relier à l’enfance. |
| DSM-5-TR | 22 chapitres de la section II examinés | Présence thématique : 12 partiels, 2 agrégés, 5 hors périmètre, 3 à examiner ; pas une comparaison de tous les diagnostics. |
| Schémas de Young | 18 sur les 18 concepts définis | Le schéma négativité/pessimisme manquait ; il est ajouté comme modèle théorique. |
| Catalogue psychologique | 161 décisions sémantiques sur 161 repères | 16 références exactes, 13 regroupements, 55 recouvrements partiels, 61 hors périmètre du questionnaire, 16 à examiner. |
| Pages de lecture | 62 sujets avec page, 20 sans page | Une source scientifique n’est pas une page pédagogique du site ; les deux compteurs restent distincts. |

Les 19 codes exacts sont : `6A20`, `6A60`, `6A72`, `6B01`, `6B02`, `6B04`, `6B05`, `6B23`, `6B40`, `6B41`, `6B42`, `6B43`, `6B44`, `6B45`, `6B60`, `6B64`, `6C20`, `6C40.2`, `6D11.5`. Le dernier est un qualificatif de configuration borderline ; ce n’est pas à lui seul un diagnostic autonome. L’entrée psychose dépasse son code 6A20, et les entrées dépression/anxiété regroupent symptômes et troubles sans code précis. Des catégories explicites évitent de compter ces entrées comme des diagnostics uniformes.

Le [relevé complet des correspondances](audit-cartographie-correspondances-2026-11-16.json) conserve pour chaque repère sa description, les alternatives examinées, les cibles exactes et la raison du statut. Il comprend aussi les 858 codes, les 82 sujets et les 58 facteurs. Le compteur de 14 identifiants `pat_` communs reste une mesure technique distincte des 16 correspondances sémantiques exactes. Une capacité actuelle peut n’être couverte que par une carte de difficulté, et une ressource n’est pas une conséquence clinique.

Les 16 mécanismes sans correspondance suffisante et à examiner sont : intellectualisation, besoin de certitude, lecture supposée des intentions, pensée tout ou rien, biais de confirmation, surinvestissement professionnel, surgénéralisation, raisonnement émotionnel, coûts irrécupérables, délai de récompense, minimisation, rationalisation, apprentissage par observation, jalousie, comparaison dans les médias sociaux et attention centrée sur soi. Leur présence dans ce suivi ne crée aucun lien causal avec l’enfance.

Les 61 entrées hors périmètre correspondent surtout à des compétences, ressources, préférences, fonctions cognitives et contextes actuels. Elles restent disponibles dans le catalogue du projet, le portrait et le dialogue ; leur exclusion du questionnaire n’est pas une prétention à expliquer toute la psychologie. Les 55 correspondances partielles explicitent les écarts de période, de fonction ou de niveau clinique. Les 13 regroupements ne sont pas comptés comme des cartes spécifiques.

Exemples de sujets présents au catalogue clinique mais sans carte spécifique : trouble bipolaire II, TOC, anxiété généralisée distincte de l’entrée générale anxiété, démences et syndromes secondaires. Leur absence appelle une décision de périmètre et une recherche adaptée, pas l’ajout automatique d’un lien causal. Les ressources, processus ordinaires, difficultés d’attention ou contraintes sociales du catalogue psychologique ne doivent pas davantage être transformés automatiquement en troubles.

L’état éditorial est un autre objet : `build_coverage_status.py --check` confirme toujours **1 467 articles, 123 personnages, 90 parcours cliniques, 64 cas en attente et 161 repères indexés**. `docs/cartographie-clinique-v17.md` contient des effectifs historiques plus anciens et renvoie déjà vers `docs/etat-couverture.md`. Ces effectifs historiques ne servent pas de preuve de couverture de la Cartographie interactive.

## Défauts établis et correction apportée

### Addition et fausse montée du niveau de preuve

L’ancien calcul classait une issue en `convergence` avec deux liens dont un modéré/solide/critère, ou avec trois liens quelconques non définitionnels. Sur le jeu réel, **18 sujets ne possédant que des liens limités ou théoriques** pouvaient ainsi converger. L’intensité 2 multipliait aussi les poids par 1,5, sans calibration documentée.

Le tri utilise désormais le **niveau empirique maximal**, sans somme et sans multiplicateur. `convergence` reste un nom technique : il exige plusieurs familles de vécus ayant chacune un lien empirique modéré ou solide. Il ne relève ni le niveau de preuve du lien ni la probabilité d’un trouble. Les familles de recouvrement sont un choix éditorial conservateur ; elles ne prouvent pas l’indépendance statistique des expositions. Par exemple, « violences psychologiques » et « critiques constantes » ne produisent plus deux familles distinctes.

Les données ajoutent `relation_type` : **303 associations empiriques, 83 rapprochements de modèles, 13 contextes de classification, 26 rapprochements définitionnels**. Les niveaux de preuve restent des appréciations éditoriales héritées des dossiers : ils ne sont pas une cotation GRADE validée.

Une étude longitudinale de Baldwin et collègues montre précisément pourquoi il faut distinguer différences moyennes entre groupes et prédiction individuelle à partir des ACE : les personnes ayant le même score peuvent avoir des résultats de santé différents, et la précision individuelle est faible. Cela soutient l’abandon d’un score de risque implicite, sans nier les associations observées. [Baldwin et al., JAMA Pediatrics, 2021](https://pubmed.ncbi.nlm.nih.gov/33492366/)

### Classification et conditions nécessaires

Cinq liens fondés seulement sur une classification étaient encore marqués « modérée », en plus des huit liens « critère ». Tous sont maintenant identifiés comme contextes de classification. Le libellé public devient « Contexte décrit par la CIM-11 ». En particulier, les violences physiques ou sexuelles infantiles citées dans le TSPT complexe sont des exemples de contexte ; chacune n’est pas une condition nécessaire pour toute personne ayant ce diagnostic.

Le CDDR demande une évaluation des caractéristiques essentielles et des frontières avec d’autres troubles et avec les réactions ordinaires. Pour les tableaux liés au stress, exposition et contexte ne remplacent pas les autres caractéristiques requises. La classification ne fournit pas un coefficient de risque personnel à additionner. [OMS, CDDR 2024](https://www.who.int/publications/i/item/9789240077263) ; lecture ciblée du [document OMS, p. 21–25, 344–360, copie accessible](https://www.drugsandalcohol.ie/41519/1/WHO_Clinical_descriptions_and_diagnostic_requirements_ICD-11.pdf).

### Âge et transmission à Ambre

Le cas `negligence_emotionnelle + negligence_physique + soins_instables` plaçait 6B44 et 6B45 en tête, avec un score 8,5. La note `jeune_enfant` était conservée dans le module, puis supprimée par la préparation du prompt historique. Le CDDR décrit ces tableaux chez l’enfant, avec manifestations précoces et soins gravement insuffisants ; il distingue aussi leurs suites possibles. On ne peut pas les attribuer à un adulte sur trois cases rétrospectives.

Les deux entrées portent maintenant `age_scope: young_child_only` et `contextual_only: true`, restent disponibles comme contexte et quittent les propositions automatiques. Les retours « je me reconnais » conservent ces restrictions et ne valent pas diagnostic. Les références, `source_ids`, la nature du lien, les rapprochements définitionnels et les restrictions d’âge sont transmis à Ambre ; son contexte reste borné et signale les troncatures. Les correctifs de prompts et d’index sont intégrés par l’agent backend sur le même chantier.

### Réponses, doublons et schéma manquant

Le backend accepte désormais `0` pour un non explicite, le conserve séparément dans `explicit_negative`, et ne le traite pas comme expérience vécue. Les réponses non renseignées restent absentes. Les deux formulations alternatives de séparation parentale ne se cumulent pas dans le décompte ACE ; une nouvelle soumission contradictoire est refusée. Ce questionnaire reste explicitement distinct d’un instrument ACE validé.

Le doublon de référence `fosco2016` / `fosco2016_profils`, identique par URL, est fusionné lors de la génération. Cela n’assure pas que toutes les publications ou cohortes partagées entre plusieurs articles ont été dédoublonnées.

Le schéma **négativité/pessimisme** est défini par Young et répertorié par l’ISST. Le nouveau lien avec les critiques/exigences renvoie au domaine théorique de survigilance/inhibition présenté par cette source. Son résumé dit explicitement que ce rapprochement ne démontre ni causalité ni présence du schéma. Aucune probabilité ou relation empirique nouvelle n’est inventée. [Young, liste des schémas](https://www.schematherapy.com/id73.htm) ; [ISST, concepts et domaines](https://www.schematherapysociety.com/Schema-Therapy).

## Références actuelles et travail clinique restant

L’APA décrit le DSM-5-TR comme un manuel d’évaluation et de diagnostic, et publie des mises à jour distinctes. La page officielle consultée mentionne notamment une révision des spécificateurs de sévérité des troubles alimentaires en mai 2026. Une prochaine comparaison devra figer la version applicable et distinguer modifications adoptées et simples propositions. La table des matières de l’édition APA consultée dans son aperçu éditeur, complétée par les notices officielles des chapitres, permet un relevé des 22 chapitres de la section II. La page officielle « DSM-5-TR Classification » a été identifiée, mais son ouverture renvoie HTTP 403 dans cet environnement. Aucun registre de tous les diagnostics DSM n’existe dans le dépôt. Les documents effectivement lus ne permettent donc pas de certifier une comparaison de tous leurs critères. [édition APA, aperçu et table des matières](https://books.google.com/books?vid=ISBN9780890425770) ; [classification officielle identifiée](https://psychiatryonline.org/doi/10.1176/appi.books.9780890425787.x00_Diagnostic_Classification) ; [APA, FAQ DSM](https://www.psychiatry.org/psychiatrists/practice/dsm/frequently-asked-questions) ; [mises à jour](https://www.psychiatry.org/psychiatrists/practice/dsm/updates-to-dsm/) ; [état des propositions et modifications](https://www.psychiatry.org/psychiatrists/practice/dsm/proposed-changes/).

Le CDC décrit les ACE comme des expériences de l’enfance susceptibles d’affecter durablement la santé et précise que ses exemples ne constituent pas une liste exhaustive. Il ne valide pas les 58 cases, leurs poids ou le classement de cette application. [CDC, présentation des ACE, 2 mars 2026](https://www.cdc.gov/aces/about/index.html).

La limite principale demeure la **validation clinique lien par lien** : les métadonnées actuelles indiquent une lecture du seul résumé pour 223 références. Les audits antérieurs vérifiaient des sous-ensembles de 68 et 85 liens, sans qu’on puisse les additionner en supposant leur indépendance. Le présent audit vérifie l’intégralité des identifiants et des comportements de calcul, mais pas les 425 affirmations contre chaque résultat complet d’étude.

Priorités suivantes, sans ajouter artificiellement des troubles :

1. Ajouter à chaque lien population/âge, type de mesure, temporalité, facteurs de confusion, limites, passage vérifié et date de revue ; revoir en premier les liens qualifiés de solides et les regroupements de diagnostics.
2. Faire une table DSM-5-TR diagnostic par diagnostic avec statuts « référencé », « agrégé », « hors périmètre justifié », « à examiner » ; aucune équivalence automatique avec un code CIM.
3. Faire relire les 161 décisions sémantiques et traiter les 16 mécanismes sans correspondance suffisante sous une forme adaptée. Préserver la distinction entre sujets actuels, ressources et expositions ; ne pas remplir artificiellement le questionnaire avec les 61 objets hors périmètre.
4. Distinguer plus finement attachement enfant/adulte, symptômes/troubles, idées suicidaires/automutilations et groupes hétérogènes. Un regroupement facilite une première lecture mais n’établit pas leur équivalence clinique.
5. Faire relire par un professionnel qualifié les allégations cliniques et les frontières d’âge. La réussite des tests de logiciel n’est pas une validation psychométrique.

## Vérifications exécutées

- `python3 scripts/build_cartographie.py --check` : données à jour.
- `node scripts/build_cartographie_module.mjs --check` : copie Worker à jour.
- `python3 scripts/build_coverage_status.py --check` : état éditorial à jour, inchangé par cet audit.
- `node scripts/build_psychology_catalog.mjs --check` : catalogue psychologique à jour.
- `node --test tests/cartographie.test.cjs journal/tests/cartographie.test.mjs` : 37 tests réussis lors de la première validation, incluant le calcul de toutes les paires de facteurs, les non explicites, l’âge, les références et la conservation des données.
- Audit reproductible : **1 711 combinaisons par version, aucune différence de classement navigateur/Worker**. L’accord logiciel existait déjà avant correction ; il ne suffisait pas à garantir la pertinence clinique de la règle.

Le générateur vérifie aussi les empreintes de l’audit sémantique et la présence de chacun des 161 identifiants : un changement des sources exige une mise à jour relue de cet audit. Les données n’ont pas été déployées. La couverture machine lisible est incluse dans `data/cartographie.json` (`coverage`, `interpretation`) et dans sa copie Worker. Les lacunes documentaires restent explicites.
