# Lot 5 — schémas, attachement et émotions

Recherche pour le chantier du 17 novembre ; consultations effectuées le **16 novembre 2026 UTC**. Fragment : `editorial/cartographie/liens-schemas-attachement-emotions-2026-11-17.json`. Aucune publication, aucun générateur exécuté en écriture, aucun commit.

## Livraison et limites

Quatre sujets, vingt paires nouvelles, cinq nouvelles références, quatre corrections de métadonnées de sources, vingt décisions hors périmètre et quatorze décisions de correspondance du catalogue. Les nouveaux sujets sont la honte, la rumination, la surcompensation et le mode parent exigeant. Les deux derniers sont exclusivement des **modèles cliniques** ; leurs catégories et libellés d’interface sont proposés dans `proposed_categories`.

Les dix-huit schémas existants sont conservés comme liste fermée. Il n’y a ni source, ni sujet, ni paire redéclarés depuis `data/cartographie.json`. Les corrections du lot 9 sur honte/imperfection, confiance/méfiance et désorganisation infantile demeurent à intégrer. Le nouveau sujet `ten_honte` donne une cible propre aux associations de Meng ; il ne valide aucun lien vers `sch_imperfection`.

Une estimation statistique et le niveau éditorial de preuve sont différents : ici `moderee` ne signifie pas « grand effet ». Aucune nouvelle association n’est classée `forte`. Les résultats transversaux ou rétrospectifs ne démontrent pas une causalité ; plusieurs synthèses partagent vraisemblablement des études primaires.

## Sources réellement consultées

Les liens ci-dessous pointent les documents consultés, y compris lorsque le DOI existant est conservé dans le graphe. « Sections » signifie une lecture ciblée, **pas tout le texte**.

| Source et accès | Lecture réelle et preuve utilisée |
| --- | --- |
| [Young, Klosko et Weishaar, 2003](https://www.guilford.com/excerpts/young.pdf?t=1), `young2003` existant | PDF éditeur, 62 pages. Définition/origines p. 7–10 ; figure 1.1 p. 14–17 ; styles p. 32–36 ; modes p. 37, 40, 43–44. P. 9, note 2 : besoins postulés à partir de théorie/observation clinique. Les deux nouveaux liens théoriques viennent des exemples de contrôle/surcompensation p. 34–35 et d’exigences intériorisées p. 44. Aucun remplacement automatique des anciens liens Carlucci. |
| [ISST, concepts centraux](https://www.schematherapysociety.com/Schema-Therapy), `isst_young_domains` existant | Sections des quatre concepts et des dix-huit schémas lues. Modèle, définition et structure ; aucune estimation d’association. |
| [May et al., 2022](https://onlinelibrary.wiley.com/doi/10.1002/cpp.2712), `may2022` existant | Méthodes, résultats, tableaux 1–5 et discussion lus. Le tableau 4 mesure les schémas individuels ; le tableau 5 mesure des domaines. Trois à cinq études par estimation ; autorapports surtout transversaux. |
| [Pilkington et al., 2021](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpp.2533), `pilkington2021` existant | Résumé éditeur réellement lu ; la demande de texte intégral a redirigé vers le résumé. Trente-trois études adultes, une longitudinale. Aucun résultat non visible attribué à un schéma particulier. |
| [Ditzer et al., 2023](https://www.careknowledge.com/media/56066/2023-78411-001.pdf), `ditzer2023` existant | PDF APA de 20 pages. Méthodes, tableau 1 p. PDF 8, résultats et limites p. PDF 11–12. Violences physiques : r = .13, IC [.06, .21] ; sexuelles : r = .14, IC [.09, .20]. Adultes, vécus rétrospectifs ; associations faibles. Aucune déduction vers l’autisme. |
| [Gruhn et Compas, 2020](https://cdn.vanderbilt.edu/t2-my/my-prd/wp-content/uploads/sites/2804/2018/04/Effects-of-maltreatment-on-coping-and-emotion-regulation-in-T-childhood-and-adolescence-A-meta-analytic-review.pdf), `gruhn2020` existant | PDF universitaire de 12 pages. Méthodes, tableaux 1–3, résultats, discussion et limites lus. Âges moyens admissibles : 5–18 ans. Dysrégulation : r = .28 ; évitement .25 ; suppression .24. Les sous-types d’exposition sont combinés : aucun nouveau lien propre à chaque sous-type n’en est déduit. |
| [Zhang et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36341581/), `zhang2023_estime` nouveau | Résumé PubMed seulement, 254 études. Estime de soi : violences émotionnelles −.23, physiques −.14, sexuelles −.14 ; négligence émotionnelle −.22, physique −.17. Effets par âge non extraits du résumé. |
| [Meng et al., 2026](https://pubmed.ncbi.nlm.nih.gov/42314249/), `meng2026` existant | Résumé PubMed seulement, 62 études. Honte : émotionnel .332, physique .196, sexuel .187 ; négligence émotionnelle .267, physique .203. Les mesures de honte varient ; résultat non transféré au questionnaire des schémas de Young. |
| [Hankin, Schweizer et Young, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12396740/), `hankin2025_rumination` nouveau | Manuscrit auteur PMC obtenu par urllib après blocage de l’outil web. Méthodes, tableau 3, résultats multivariables, discussion et limites lus. Cohorte de 607 jeunes, trois années de suivi, trajectoires 9–18 ans ; rumination mesurée par six items de brooding. Violences émotionnelles : β = .26, IC [.19, .33] dans le modèle par prédicteur. L’association ne demeure pas unique dans le modèle multivariable : niveau `limitee` et limite inscrite dans le résumé public. |
| [Mansueto et al., 2021](https://onlinelibrary.wiley.com/doi/10.1002/cpp.2590), `mansueto2021_rumination` nouveau | Résumé éditeur lu, 18 études. Revue systématique, sans méta-analyse ; inquiétude et rumination chez des adultes. Ne justifie pas à elle seule des liens pour tous les sous-types de maltraitance. |
| [Amini-Tehrani et al., 2026](https://research-repository.uwa.edu.au/en/publications/meta-analysis-of-associations-between-childhood-emotional-abuse-a/), `aminitehrani2026_regulation` nouveau | Résumé publié du dépôt UWA lu, 124 échantillons, 308 effets. Corrélations transversales ; tendances d’évitement r = .32, représentation émotionnelle aversive r = .24, forte hétérogénéité. Le texte ScienceDirect renvoie 403. Le préprint 2025 retrouvé a d’autres effectifs et n’est pas utilisé comme version publiée. Pas de nouvelle carte par stratégie à partir des résultats agrégés. |
| [van Eickels et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12023818/), `vaneickels2025_honte_culpabilite` nouveau | Sections introduction, méthodes, résultats/tableaux 3–4, discussion/limites lues via urllib et Wiley. Relations parent-enfant dysfonctionnelles et culpabilité excessive : r = .15 ; culpabilité adaptée : résultat non significatif. Les catégories parentales sont trop larges pour de nouveaux liens précis. Échantillons surtout non cliniques ; ne pas extrapoler aux troubles. |

Les échecs de PubMed/PMC dans l’outil web étaient des pages de cookies ou CAPTCHA. Seuls les articles effectivement récupérés par un autre accès portent une lecture plus détaillée. Le second article sur honte, Zhang 2025 (PMID 40505207), a pu être identifié mais son résumé n’a pas été lu de manière fiable : **non ajouté comme preuve**. L’umbrella review sur rumination, DOI 10.1016/j.neubiorev.2026.106818, n’a pas été accessible : pas de preuve reprise d’un extrait de recherche.

## Décisions sur les dix-huit schémas

« Existant » signifie que ce lot n’ajoute ni ne renforce la paire : ce n’est pas une validation globale des anciennes sources. Références de définition : figure 1.1 de Young et liste ISST. Référence empirique des quatre ajouts : tableau 4 de May.

| Identifiant | Décision du lot |
| --- | --- |
| `sch_abandon` | Ajout violences psychologiques : r .33 [.04, .57], niveau limité. |
| `sch_mefiance` | Existant ; corrections confiance→schéma laissées au lot 9. |
| `sch_carence_affective` | Existant. |
| `sch_imperfection` | Existant ; honte traitée séparément. |
| `sch_isolement` | Existant. |
| `sch_dependance` | Aucun ajout empirique. |
| `sch_vulnerabilite` | Ajout négligence affective : r .22 [.08, .36], limité. |
| `sch_fusion` | Aucun ajout empirique. |
| `sch_echec` | Association à négligence affective déjà présente ; pas de doublon. |
| `sch_droits_excessifs` | Ajout violences psychologiques : r .27 [.01, .49], limité. |
| `sch_controle_insuffisant` | Ajout négligence affective : r .32 [.13, .49], limité. |
| `sch_assujettissement` | Existant. |
| `sch_abnegation` | Aucun ajout empirique. |
| `sch_recherche_approbation` | Aucun ajout empirique. |
| `sch_negativite` | Aucun ajout empirique. |
| `sch_surcontrole` | Aucun ajout empirique. |
| `sch_exigences_elevees` | Aucun ajout empirique. |
| `sch_punition` | Aucun ajout empirique. |

Les limites des intervalles, les petites bases d’études et l’hétérogénéité motivent la retenue. L’existence d’une moyenne positive ne suffit pas si son intervalle traverse zéro. La liste reste à dix-huit ; les modes et styles ne reçoivent aucun préfixe `sch_`.

## Attachement et correspondances

[Li et al. 2026](https://journals.sagepub.com/doi/10.1177/15248380261455767) : résumé éditeur réellement lu, texte intégral restreint. Les synthèses adultes concernent les dimensions d’anxiété et d’évitement (r .23 et .20), sans valider ici une catégorie craintive distincte. Aucun nouvel effet par sous-type n’est inventé. Les données infantiles et les troubles précoces ne permettent aucune inférence de style amoureux adulte. Le lot 9 traite la carte infantile.

Les quatorze décisions détaillées et leurs sources sont dans `catalogue_decisions` : rumination, honte, identification, suppression, débordement, culpabilité, tolérance à la détresse, proximité/distance, sécurité, détachement, contrôle, confiance en soi, retrait et jalousie. La rumination reste une correspondance **partielle** avec le repère général de pensées en boucle ; la honte aussi, malgré la carte nouvelle. Les compétences ordinaires et les ressources ne deviennent pas des difficultés supposées. Les 61 repères déjà hors périmètre ne sont pas rouverts.

Les styles soumission et évitement ainsi que les modes non retenus figurent individuellement au registre. Un concept défini ne constitue pas une preuve pour toutes les cases. Les défenses DMRS relèvent du lot 4. La carte parent exigeant n’est ni un dix-neuvième schéma ni un diagnostic.

## Contrôles non écrivants

- Collisions avec les sujets, sources et paires publiés : **aucune**.
- Chaque sujet nouveau dispose d’un lien ; domaines renseignés ; catégories nouvelles accompagnées de libellés proposés.
- Résumé le plus long : **218 caractères** ; aucun pourcentage ni terme causal interdit.
- Appel `b.build()` réalisé sans écriture : 59 facteurs, 109 sujets, 516 liens et 268 sources au moment du contrôle, sur l’arbre partagé en évolution. **Seule erreur : empreinte d’audit périmée**, attendue.
- Ces comptes incluent les autres lots présents ; ils ne décrivent pas la production. Les métadonnées proposées sont à appliquer par le pilote, puis audit sémantique et relecture indépendante requis.
