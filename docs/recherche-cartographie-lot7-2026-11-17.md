# Recherche lot 7 — santé, trajectoires et ressources

Lot `sante-trajectoires-ressources`, daté du 17 novembre 2026. Fichiers détenus : ce rapport et `editorial/cartographie/liens-sante-trajectoires-ressources-2026-11-17.json`. Aucun fichier généré, commit, push ni publication.

## Résultat proposé

Le fragment contient **4 nouveaux sujets contextuels, 14 liens, 12 nouvelles sources, 3 ressources et 100 décisions de registre**, ainsi que **6 corrections de liens et 1 correction bibliographique**. Aucun identifiant de sujet, de source ou de ressource publié n’est redéclaré ; aucune paire publiée n’est recopiée dans `links`.

Les quatre sujets — cardiovasculaire, diabète de type 2, migraine et fibromyalgie — portent `contextual_only: true`, `kind: tendance`, `domain: perceptions`, sans code CIM inventé ni nouvelle catégorie d’interface. Le type tendance est une contrainte du format, pas une qualification médicale : les libellés indiquent explicitement « repère de recherche ». Ils ne doivent pas alimenter une suggestion diagnostique, une probabilité de maladie ou une étiologie individuelle. L’intégrateur devra vérifier cette exclusion effective dans le calcul et dans le contexte d’Ambre.

Aucun lien `forte`. Les données cas-témoins ou essentiellement rétrospectives et les résultats peu précis restent limités. La cible de quantité du plan n’a pas été remplie artificiellement.

## Méthode et accès réel

AGENTS.md, la tête « État au 17 novembre » de la reprise et le plan de couverture ont été lus. Le catalogue publié, les fragments présents et les 161 décisions de l’audit ont été inspectés par parseur. Les sources des fragments sont consultées avant chaque déclaration nouvelle, pour éviter la règle du premier identifiant gagnant.

Recherches avec l’outil web `search_service_web_run`, puis ouvertures des pages PubMed, PMC, éditeurs et CDC. Plusieurs ouvertures PubMed ont alternativement retourné le résumé ou une simple ligne HHS ; PMC a parfois retourné un CAPTCHA. Ces écrans ne comptent pas comme des lectures. Quand la page était limitée, le résumé indexé a été lu via le service public Europe PMC, par `urllib` : `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:PMID%20AND%20SRC:MED&format=json&resultType=core`. Métadonnées, auteurs, titre, DOI et pagination ont été confrontés à la notice. Les documents XML complets ont été obtenus seulement lorsque l’accès fonctionnait.

L’accès au texte intégral n’est jamais attribué sur la seule disponibilité d’un lien. `read: section` précise les parties réellement parcourues ; aucune source n’est prétendue intégralement lue de bout en bout. Les études transversales isolées n’alimentent aucun lien empirique. Une revue systématique les réunissant reste admissible à niveau limité.

## Preuves retenues et limites

| Sujet / trajectoire | Lecture réelle | Décision |
| --- | --- | --- |
| Maladies cardiovasculaires | [Soares et al.](https://pubmed.ncbi.nlm.nih.gov/34041541/) : résumé, résultats, tableaux 1–2 et limites du XML [PMC9082818](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9082818/fullTextXML). Cohorte britannique : cinq expositions distinctes, écarts modestes, souvenirs rétrospectifs. | Cinq liens `moderee`, sans reprendre les modèles de médiation comme mécanismes causaux démontrés. Pas de multiplication en infarctus, AVC et toutes maladies vasculaires. |
| Diabète de type 2 | [Huang2015](https://pubmed.ncbi.nlm.nih.gov/26404480/) : résumé, 7 articles dont 4 cohortes et 3 enquêtes transversales ; associations modestes pour violences physiques et sexuelles. | Deux liens `moderee`, sous-types d’exposition effectivement distingués. |
| Actualisation diabète | [Zhu2022](https://jogh.org/2022/jogh-12-04082/) : résumé éditeur ; résultats, tableau 2 et discussion par type via [PMC9624439](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9624439/fullTextXML). La synthèse de 49 études mélange fréquemment les types de diabète ; elle retrouve les deux violences mais pas une association concluante pour la négligence. | Source de nuance dans les deux liens, sans prétendre confirmer spécifiquement tous les résultats de type 2. Pas de lien nouveau pour la négligence. Ne pas confondre violences verbales et émotionnelles, distinguées par cette revue. |
| Migraine | [Liu2024](https://pubmed.ncbi.nlm.nih.gov/39096662/) : résumé via Europe PMC, 12 études, trois formes de violence distinguées. [Flor2025](https://pubmed.ncbi.nlm.nih.gov/40210705/) : résumé via Europe PMC, méthode Burden of Proof et prudence sur violences psychologiques–migraine. | Trois liens `limitee`. Âges et dessins détaillés non disponibles dans le seul résumé Liu ; pas de mécanisme, pas de force supérieure par accumulation de références. |
| Fibromyalgie | [Häuser2011](https://pubmed.ncbi.nlm.nih.gov/20722042/) : résumé via Europe PMC ; 18 études cas-témoins. Résultats infantiles spécifiques pour violences physiques et sexuelles ; qualité souvent faible. | Deux liens `limitee`. Aucun lien psychologique infantile séparé tiré des résultats concernant les violences émotionnelles ou les expositions adultes. |
| Maladie infantile et école | [Pinquart et Teubert2012](https://pubmed.ncbi.nlm.nih.gov/22173882/) : notice et résumé via Europe PMC ; méta-analyse de 954 études, écart scolaire moyen modéré, grandes différences entre maladies. | Un lien `maladie_enfant → ten_difficultes_scolaires`, abaissé à `limitee` car le facteur inclut aussi une hospitalisation isolée non équivalente à la maladie chronique étudiée. Aucun diagnostic neurodéveloppemental ajouté ni inféré. |
| Harcèlement et emploi | [Brimblecombe2018](https://pubmed.ncbi.nlm.nih.gov/29803971/) : notice et résumé via Europe PMC ; cohorte britannique suivie jusqu’à 50 ans, emploi et patrimoine moins favorables, différence de revenus du travail surtout chez les femmes. [Takizawa2014](https://pubmed.ncbi.nlm.nih.gov/24743774/) : résumé relu via Europe PMC. | Un lien `moderee` vers la carte professionnelle existante. Les deux publications utilisent la même cohorte : elles ne valent pas deux réplications indépendantes. |
| Scolarité après maltraitance | [Zhang2025](https://pubmed.ncbi.nlm.nih.gov/39616953/) et [Qu2024](https://pubmed.ncbi.nlm.nih.gov/38651820/) : résumés relus via Europe PMC. | Maintien de l’agrégation `ten_difficultes_scolaires`, pas de nouveaux doublons par diplôme, note ou redoublement. Les sous-types de négligence ne doivent pas être inventés à partir de « neglect ». |
| Emploi après maltraitance | [Currie et Widom2010](https://pubmed.ncbi.nlm.nih.gov/20425881/) : résumé relu via Europe PMC ; cohorte judiciaire américaine, maltraitances documentées, adultes d’âge moyen. | Maintien de la carte existante, sans transformer les écarts socio-économiques en manque de capacités ou de volonté. La lecture du résumé global ne certifie pas chaque estimation par sous-type. |

Les articles complets de Liu, Häuser, Huang, Pinquart et Brimblecombe n’ont pas été lus. Leurs limites sont conservées dans les niveaux et dans ce rapport. La recherche n’est pas une nouvelle revue systématique exhaustive.

## Maladies et trajectoires non reliées

- **Asthme** : [Wang2026](https://doi.org/10.1186/s12889-026-26363-y), résumé, méthodes, résultats et discussion éditeur lus. Les 16 études sont transversales, avec forte hétérogénéité et un signal de biais de publication ; les résultats portent sur des nombres d’ACE. Aucun résultat cumulatif n’est réparti entre les cases du questionnaire.
- **Maladies auto-immunes** : [Jesuthasan2025](https://pubmed.ncbi.nlm.nih.gov/40320015/), résumé complet via Europe PMC, notice PubMed vérifiée ; éditeur en 403. Associations modestes ou variables selon les maladies, forte hétérogénéité et 21 études sur 27 à haut risque de biais. Polyarthrite rhumatoïde, psoriasis, sclérose en plaques et maladies inflammatoires intestinales restent `a_examiner` : l’exposition du résumé est agrégée. Lupus : résultat non concluant, inscrit séparément. Ne pas transférer le risque agrégé vers chaque facteur ou chaque maladie.
- **Démence** : [Abouelmagd2025, en ligne2024](https://doi.org/10.1007/s00127-024-02676-4), résumé, méthodes, résultats et discussion lus. Données faibles, exposition souvent cumulative ; sous-groupe perte parentale non concluant. La page contient des incohérences entre certaines valeurs, leurs intervalles et leurs descriptions : elles ne sont pas corrigées par supposition. [Logue2026](https://pubmed.ncbi.nlm.nih.gov/42026719/), résumé via Europe PMC, actualise avec 13 études et retrouve une association pour le décès d’un parent. Ce résultat ne justifie pas un lien applicable à tout `deces_proche`. Le registre demeure ouvert, aucun sujet diagnostique ajouté. Tout ajout futur resterait contextuel uniquement.
- **Trouble neurocognitif léger** : estimation très imprécise et non concluante dans la première revue ; pas de carte. « Aucune preuve trouvée » signifie ici absence de preuve exploitable dans la recherche effectuée, non preuve d’absence universelle.
- **Mortalité** : [Zheng2025](https://pubmed.ncbi.nlm.nih.gov/39608195/), résumé PubMed lu ; associations populationnelles, notamment avec précarité, maltraitance agrégée et placement. Pas de carte de mortalité : ce résultat de santé publique ne constitue ni tendance psychologique ni pronostic individuel.
- **Douleurs persistantes et surpoids** : cartes existantes conservées, aucune redéclaration. Les cartes contextuelles détaillées ne disent pas que les douleurs sont psychogènes.

Le registre JSON contient les décisions par maladie ou construit avec sources et raison. Il ne prétend pas couvrir toutes les maladies somatiques de la CIM ; les cancers, maladies respiratoires hors asthme et les diagnostics sans recherche propre documentée restent à instruire séparément, sans lien ajouté.

## Ressources, résilience et croissance

Les huit identifiants existants restent inchangés. Trois ajouts seulement :

1. `prot_changer_perspective` : réévaluation d’une situation, avec données limitées ; ne jamais demander de nier une violence.
2. `prot_traverser_inconfort` : appuis pour traverser une émotion ; ce n’est pas un devoir d’endurer seul ou de rester en danger.
3. `prot_appuis_materiels` : aides concrètes aux besoins essentiels, présentées comme recommandation de prévention et d’accès, pas comme effet tampon individuel prouvé.

Les deux premières s’appuient sur [Fritz2018](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2018.00230/full), dont ont été lus résumé, méthodes, tableaux 3–4, sections de résultats et discussion : revue de cohortes, résultats spécifiques et non uniformes, appuis interreliés. Les sections consultées ne permettent ni un score de résilience ni une hiérarchie individuelle des ressources. L’auto-efficacité n’a pas montré l’effet modérateur testé dans l’étude concernée : pas d’ajout automatique de cette ressource comme effet démontré sur toute adversité.

La troisième cite la [page de prévention du CDC](https://www.cdc.gov/aces/prevention/index.html), lue intégralement dans l’outil, datée du 8 octobre2024. La recommandation concerne les conditions de vie, l’alimentation, la garde des enfants et l’organisation du travail ; elle ne remplace pas des données d’efficacité individuelles.

[Han2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10528145/) a été lu au résumé et au début de l’introduction ; la suite a rencontré un CAPTCHA puis une erreur500 sur le XML. La revue distingue associations favorables directes et effets d’interaction avec l’adversité : les seconds sont moins constants. Pas de carte séparée pour chaque item BCE/PCE ni de calcul qui soustrait une ressource à un vécu. La revue plus récente Cunha a aussi été consultée au résumé via Europe PMC (PMID39614772) ; elle confirme l’inconstance des interactions mais n’est pas ajoutée comme référence redondante.

[La méta-analyse de Mangelsdorf2019](https://pubmed.ncbi.nlm.nih.gov/30589274/), résumé lu, distingue croissance perçue et changement mesuré dans le temps. La souffrance n’est pas une condition démontrée de croissance. Le concept reste au registre ; aucune carte ne promet un bénéfice, aucune ressource n’exige de « grandir grâce au trauma ». La récupération, l’absence de croissance ressentie et le choix de ne pas donner de sens à un événement restent légitimes.

Les 61 exclusions préexistantes du catalogue161 sont conservées individuellement, sans rouvrir artificiellement leurs étiologies. Dix-neuf décisions complémentaires concernent compétences, soutien, résilience et adaptations actuelles, soit **80 repères du catalogue**. Elles distinguent ressource, absence de difficulté et conséquence supposée de l’enfance. Les correspondances exactes sont dans `out_of_scope` ; `reframing` reste hors graphe même si une ressource voisine est proposée. Ces décisions ne prétendent pas faire d’une compétence générale une conséquence clinique.

## Corrections proposées, sans application directe

- Retirer `parent_addiction`, `parent_trouble`, `parent_incarcere` et `temoin_violences → ten_sante_physique` : leurs résumés reconnaissent eux-mêmes que la preuve porte surtout sur le cumul. La présente décision repose sur cette absence de preuve spécifique vérifiée ; les textes intégraux de Hughes et Felitti n’ont pas été relus dans ce lot. Une réserve verbale ne suffit pas à légitimer une contribution au graphe. Un intégrateur peut maintenir une paire seulement avec une autre preuve adaptée explicitement vérifiée.
- Remplacer la source et le résumé des deux liens `negligence_* → ten_sante_physique` : ne plus attribuer un résultat de diabète à chacune des deux formes de négligence. La nouvelle formulation concerne la santé cardiovasculaire, avec prudence.
- Corriger la source existante `fritz2018` de `type: etude` vers `type: revue`, et préciser la lecture. L’entrée de `proposed_corrections` comporte `target_type: source` et `source_id` ; elle doit être traitée séparément des corrections de paires, en conservant l’ID et l’URL. Aucun remplacement silencieux dans `sources`.

Le lot9 avait déjà proposé des corrections de scolarité et douleur. Aucun de ses changements n’a été appliqué ni recopié automatiquement. L’intégrateur doit réconcilier les décisions de maintien sur les sous-types de négligence avec la portée exacte des résumés, plutôt que prendre un statut « conserver » comme validation indépendante de tous les détails.

## Contrôles

Dernière passe effectuée :

- `b.build()` importé avec `sys.dont_write_bytecode=True`, sans appel écrivant : **59 facteurs,110 sujets,534 liens,276 sources** dans l’arbre partagé à cet instant ; **OK hors empreinte audit**, une seule erreur attendue d’audit périmé. Ces comptes incluent les travaux des autres lots.
- Aucune collision d’identifiant de sujet/source/ressource ni de paire avec les données publiées et les autres fragments présents ; aucun doublon interne.
- Toutes les sources citées existent, tous les facteurs appartiennent au jeu publié, chaque nouveau sujet possède un lien ; tous les résumés ont au plus **296 caractères** et passent le contrôle des mots interdits et pourcentages.
- `notes` ne contient que des chaînes ; les statuts du registre sont autorisés ; les quatre nouveaux sujets ont `domain` et `contextual_only: true`.
- Empreinte SHA-256 du fragment livré : `6b22d12ba6fdcf60d1499a95655bb0193798562a6db280f12520d5659a4538e0`.

Aucun artefact généré ni audit réempreinté. L’intégrateur doit mettre à jour les correspondances après relecture sémantique et traiter explicitement les sept propositions de correction, notamment l’objet bibliographique distinct. Le contrôle logiciel ne vaut pas relecture médicale indépendante.
