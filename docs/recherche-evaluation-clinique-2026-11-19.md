# Recherche et contrat d’évaluation clinique — 19 novembre 2026

Statut : recherche et moteur descriptif implémenté après autorisation du contrôleur ; publication non attestée par ce document. Les 157 sujets sont inventoriés individuellement plus bas. Couvrir un sujet dans cet inventaire ne signifie pas disposer de sa procédure diagnostique automatisable.

## Conclusion et frontière

Le prochain bilan peut produire automatiquement un état des hypothèses et des informations disponibles. Il ne peut transformer les réponses annuelles, les corrélations de Mon enfance ou une rédaction d’IA en diagnostic établi. La zone « Mes diagnostics » doit distinguer les diagnostics rapportés comme posés par un clinicien des hypothèses du service. Un diagnostic rapporté reste une déclaration de l’utilisateur, sauf vérification indépendante explicite ; il n’est jamais « confirmé » par le modèle.

Le score `essential_share_desc` est un ordre de rapprochements, pas une probabilité. Le nombre de symptômes partagés ne vérifie ni leur simultanéité, ni les durées propres à chaque tableau, ni les exclusions, ni le retentissement. Les schémas, modes, styles d’attachement et dimensions de recherche n’ont pas à recevoir un diagnostic CIM inventé.

## Sources et degré de consultation

- OMS CDDR 2024, ISBN 9789240077263 : PDF local intégral et texte `work/reprise-nov19/sources/cddr.*`, empreinte `b66e7a5d6ed764a1d39ed6ed37d581d25b5b9b79a6ee74b4935ff6677e8e709b`. [Notice officielle](https://www.who.int/publications/i/item/9789240077263), [copie du document OMS chez HRB](https://www.drugsandalcohol.ie/41519/1/WHO_Clinical_descriptions_and_diagnostic_requirements_ICD-11.pdf). Passages déjà examinés détaillés dans `relecture-clinique-symptomes-v2-2026-11-19.md`; cette recherche réutilise ces lectures, sans prétendre avoir lu chaque assertion des 852 pages.
- HAS, [dépression adulte : diagnostic](https://www.has-sante.fr/jcms/c_2800512/fr/depression-adulte-diagnostic-fiche-de-synthese), [recommandations](https://www.has-sante.fr/jcms/c_2800526/fr/depression-adulte-recommandations), [présentation de la démarche](https://www.has-sante.fr/jcms/pprd_2974237/fr/depression-de-l-adulte-reperage-et-prise-en-charge-initiale). Texte de recommandations retrouvé dans l’index officiel ; page fiche ouverte mais ne rend pas le corps du PDF. Ne pas dire PDF intégral consulté. La démarche repose sur entretien, fonctionnement et recherche de causes associées ; examen et tests selon indications. La référence HAS 2017 est CIM-10 : ne pas fusionner ses seuils avec CIM-11.
- NICE [NG87 TDAH, §1.3](https://www.nice.org.uk/guidance/ng87/chapter/recommendations) : l’index officiel restitue l’exigence d’une évaluation clinique, psychosociale et développementale, et l’insuffisance d’une échelle seule. Pages/PDF directs renvoient 403 dans cet environnement ; consultation limitée aux passages indexés.
- NICE [CG185 bipolarité, §1.3](https://www.nice.org.uk/guidance/cg185/chapter/recommendations) : l’index officiel expose l’histoire détaillée des épisodes, fonctionnement intercritique et diagnostics différentiels. Accès direct 403 ; pas de prétention à lecture intégrale.
- NICE [CG178 psychoses, §1.3](https://www.nice.org.uk/guidance/cg178/chapter/recommendations) : passages officiels indexés, évaluation multidisciplinaire, antécédents, santé physique et substances/médicaments ; accès intégral direct bloqué.
- NICE [NG116 TSPT, §1.1–1.2](https://www.nice.org.uk/guidance/ng116/chapter/recommendations) : passages indexés, évaluation fonctionnelle et sécurité selon danger ; accès direct/PDF 403.

Les sources n’autorisent pas un algorithme à se substituer à ces procédures. Elles justifient les dimensions à recueillir et les inconnues à afficher. Ne pas recopier les inventaires sous licence ni présenter une synthèse originale comme questionnaire validé.

## Contrat proposé : buildClinicalAssessment(snapshot)

Fonction pure et déterministe, sans appel fournisseur, sans lecture de l’état mutable. Entrée versionnée :

```js
{
  schema_version: 1, complete: true,
  review_id, captured_at, // captured_at ISO dans le payload ; ligne D1 en secondes Unix
  sections: {
    symptoms: [{ answers: { [factorId]: 0 | 1 | 2 | 'unknown' | 'past' },
      notes: { factors: { [factorId]: 'texte complet, obligatoire pour past' } } }],
    childhood: [], conversation: [], professional_assessments: []
  }
}
```


Les anciens nombres de réponse doivent être convertis par un adaptateur versionné : aucune nouvelle valeur `past` déduite d’un texte ancien. La note historique est conservée mais n’est pas interprétée comme durée exacte. « Avant » n’implique ni guérison, ni rémission, ni diagnostic antérieur. « Oui ces 12 mois » n’implique pas présent aujourd’hui. « Non » concerne cette fenêtre, et ne suffit pas à écarter une histoire de manie ou une condition développementale.

Sortie par sujet :

```js
{
  outcome_id, kind, coverage_status, rule_refs,
  evidence: { current_window: [], historical: [], denied: [], unknown: [] },
  dimensions: {
    manifestations: 'supported'|'not_supported'|'unknown'|'conflicting',
    duration: 'unknown', contemporaneity: 'unknown', impairment: 'unknown',
    onset: 'unknown', exclusions: 'unknown', differential: 'unknown'
  },
  hypothesis_state: 'criterion_signals'|'association_only'|'historical_signals'|'historical_association_only'|'insufficient_information',
  missing_fact_ids: [], contradictory_fact_ids: [],
  clinician_report_refs: [],
  explanation_refs: []
}
```

Aucun état `diagnosed`, `ruled_out`, `confirmed`, aucun pourcentage. `criterion_signals` indique seulement une manifestation déclarée reliée à une caractéristique dans la banque, pas un critère diagnostique satisfait. `association_only` sépare les liens modérés/limités sans caractéristique essentielle déclarée. Les dimensions sans faits structurés sont toujours inconnues ; absence de ligne n’est jamais absence de symptôme. Les faits de conversation proposés par extraction IA doivent être confirmés par la personne avant de devenir faits structurés. Les citations de l’IA ou d’un bilan précédent ne sont jamais de nouvelles preuves.

## Règles interprétables permises dès maintenant

1. Vérifier les identifiants, versions et provenance. Rejeter `past` sans note ; conserver les réponses et notes originelles.
2. Séparer strictement les quatre réponses. Ne pas agréger des signes de périodes différentes en un même épisode.
3. Utiliser les liens de la banque pour lister les manifestations associées, sans déclarer les critères de diagnostic satisfaits. Familles communes comptées une fois uniquement pour l’ordre descriptif.
4. Les portes produit portent sur la même fenêtre que les symptômes. Usage passé ne déverrouille pas une dépendance actuelle ; usage seul ne compte pas comme manifestation de dépendance. Sans attribution du symptôme au produit, indiquer « produit concerné à préciser ».
5. Le constat explicite de danger reste un circuit séparé ; un signal historique isolé ne constitue pas un danger actuel, et la fenêtre de douze mois ne permet pas de trancher. Ne pas retarder une aide urgente au prochain bilan.
6. Un nouveau bilan crée un snapshot et une évaluation immuables. Reprise idempotente sur `(review_id, snapshot_id, rules_version)` ; publication de l’évaluation seulement avec le bilan validé. Un changement de banque ne réécrit pas les bilans anciens.
7. Afficher les différences avec le snapshot précédent comme « réponse modifiée » ou « information ajoutée », jamais comme amélioration clinique certaine. Suppression de compte/export incluent les snapshots et évaluations.

## Dimensions spécifiques à rechercher avant toute conclusion clinique

| Famille | Ce que les cases seules ne vérifient pas | Référence |
|---|---|---|
| Dépression | épisode commun, fréquence/intensité, durée, retentissement, histoire maniaque et causes médicales/substances | CDDR212–213,244–246; HAS2017 |
| Bipolarité | épisodes distincts, manie/hypomanie, histoire complète et exclusions, pas seulement humeur variable | CDDR217–240; NICECG185 |
| Psychose/schizoaffectif | persistance, niveau de conviction, simultanéité humeur/psychose, causes somatiques et produits | CDDR162–185; NICECG178 |
| TDAH/autisme/apprentissages | développement précoce, plusieurs contextes, fonctionnement, observations complémentaires et différentiel | CDDR124–149; NICENG87 |
| Anxiété/TOC | objet ou contexte exact, caractère persistant, évitement/rituels, retentissement et causes alternatives | CDDR266–334 |
| TSPT/TSPTc/deuil/adaptation | événement qualifiant, liens temporels, regroupement des domaines, durée et contexte culturel | CDDR339–352; NICENG116 |
| Dissociation | attribution au phénomène, mémoire, conscience/réalité, exclusion neurologique/substances | CDDR366–392 |
| Alimentation | séquence des comportements, compensation, état physique/nutritionnel et motivations | CDDR398–417 |
| Substances/jeux | produit précis, période commune, contrôle/priorité, conséquences, adaptation physique seule insuffisante | CDDR463–467,506–510 |
| Personnalité | fonctionnement durable et transversal, développement/culture, état hors épisode aigu | CDDR554–566 |
| Factice | tromperie établie, contexte/motifs; demander des soins ne prouve rien | CDDR593–594 |
| Neurocognition | déclin objectivé par évaluation, fonctionnement antérieur et causes alternatives | CDDR608–609 |
| Sommeil/sexualité/tics | référentiel spécifique, examens et exclusions selon cas ; aucune procédure universelle dans cette recherche | Couverture spécifique à approfondir |

## Tests nécessaires avant implémentation approuvée

- `past` sans note refusé ; note non convertie en durée ; Oui annuel non converti en symptôme actuel.
- Aucune preuve issue d’une sortie IA ; enfance jamais preuve d’un diagnostic.
- Cas tous Non, tous inconnus, passé seul, périodes incompatibles, contradictions explicites.
- Symptomatologie commune sans produit identifié ; traitement prescrit avec tolérance seule.
- Traits, schémas et groupes agrégés restent non diagnostiques même si toutes cases sont Oui.
- Source figée, reprise idempotente et absence de changement rétrospectif après édition des réponses.
- Isolation des comptes, partage limité, suppression/export, publication atomique et rollback.

## Couverture exhaustive des identifiants (inventaire, pas validation de 157 procédures)

Statuts : `clinical_reference_partial` = catégorie clinique, procédure à compléter et non automatisée ; `specifier_not_standalone` = qualificatif de personnalité, pas diagnostic autonome ; `aggregate_not_diagnosis` = groupe agrégé ou repère large ; `non_diagnostic_construct` = tendance/schéma/trait sans procédure diagnostique CIM ; `outside_cddr_pending` = autre chapitre nécessitant recherche spécifique. Tous peuvent recevoir un résumé de déclarations, aucun ne reçoit ici une conclusion diagnostique automatique.

| Identifiant | Sujet | Statut |
|---|---|---|
| `anx_generale` | Anxiété ou troubles anxieux | `aggregate_not_diagnosis` |
| `att_anxieux` | Attachement anxieux | `non_diagnostic_construct` |
| `att_evitant` | Attachement évitant | `non_diagnostic_construct` |
| `dep_generale` | Dépression ou symptômes dépressifs | `aggregate_not_diagnosis` |
| `icd_6a20` | Troubles psychotiques (dont la schizophrénie) | `aggregate_not_diagnosis` |
| `icd_6a60` | Troubles bipolaires (preuves agrégées) | `aggregate_not_diagnosis` |
| `icd_6a72` | Dysthymie | `clinical_reference_partial` |
| `icd_6b01` | Trouble panique | `clinical_reference_partial` |
| `icd_6b02` | Agoraphobie | `clinical_reference_partial` |
| `icd_6b04` | Trouble d'anxiété sociale | `clinical_reference_partial` |
| `icd_6b05` | Trouble d'anxiété de séparation | `clinical_reference_partial` |
| `icd_6b20` | Trouble obsessionnel-compulsif (TOC) | `clinical_reference_partial` |
| `icd_6b23` | Hypocondrie (anxiété liée à la santé) | `clinical_reference_partial` |
| `icd_6b40` | Trouble de stress post-traumatique | `clinical_reference_partial` |
| `icd_6b41` | Trouble de stress post-traumatique complexe | `clinical_reference_partial` |
| `icd_6b42` | Trouble du deuil prolongé | `clinical_reference_partial` |
| `icd_6b43` | Trouble de l'adaptation | `clinical_reference_partial` |
| `icd_6b60` | Trouble dissociatif à symptômes neurologiques | `clinical_reference_partial` |
| `icd_6b64` | Trouble dissociatif de l'identité | `clinical_reference_partial` |
| `icd_6b80` | Anorexie mentale | `clinical_reference_partial` |
| `icd_6b81` | Troubles des conduites alimentaires | `aggregate_not_diagnosis` |
| `icd_6b81_boulimie` | Boulimie nerveuse | `clinical_reference_partial` |
| `icd_6b82` | Hyperphagie boulimique | `clinical_reference_partial` |
| `icd_6c20` | Trouble de détresse physique | `clinical_reference_partial` |
| `icd_6c40_2` | Dépendance à l'alcool | `clinical_reference_partial` |
| `icd_6d11_5` | Trouble de la personnalité avec motif borderline | `specifier_not_standalone` |
| `pat_abandonment_fear` | Peur de perdre le lien | `non_diagnostic_construct` |
| `pat_appeasement` | Apaiser autrui pour préserver sa sécurité | `non_diagnostic_construct` |
| `pat_conditional_worth` | Valeur liée à la réussite | `non_diagnostic_construct` |
| `pat_conflict` | Difficulté à vivre les désaccords | `non_diagnostic_construct` |
| `pat_emotional_suppression` | Retenir l’expression émotionnelle | `non_diagnostic_construct` |
| `pat_guilt` | Culpabilité | `non_diagnostic_construct` |
| `pat_impostor` | Sentiment d’imposture | `non_diagnostic_construct` |
| `pat_needs_expression` | Difficulté à reconnaître et exprimer ses besoins | `non_diagnostic_construct` |
| `pat_overresponsibility` | Rôle de sauveur | `non_diagnostic_construct` |
| `pat_perfectionism` | Perfectionnisme coûteux | `non_diagnostic_construct` |
| `pat_self_criticism` | Autocritique | `non_diagnostic_construct` |
| `pat_social_isolation` | Solitude ou isolement subis | `non_diagnostic_construct` |
| `sch_abandon` | Schéma d'abandon et d'instabilité | `non_diagnostic_construct` |
| `sch_abnegation` | Schéma d'abnégation | `non_diagnostic_construct` |
| `sch_assujettissement` | Schéma d'assujettissement | `non_diagnostic_construct` |
| `sch_carence_affective` | Schéma de carence affective | `non_diagnostic_construct` |
| `sch_controle_insuffisant` | Schéma de contrôle de soi insuffisant | `non_diagnostic_construct` |
| `sch_dependance` | Schéma de dépendance et d'incompétence | `non_diagnostic_construct` |
| `sch_droits_excessifs` | Schéma de droits personnels excessifs | `non_diagnostic_construct` |
| `sch_echec` | Schéma d'échec | `non_diagnostic_construct` |
| `sch_exigences_elevees` | Schéma d'exigences élevées | `non_diagnostic_construct` |
| `sch_fusion` | Schéma de fusion et de personnalité atrophiée | `non_diagnostic_construct` |
| `sch_imperfection` | Schéma d'imperfection et de honte | `non_diagnostic_construct` |
| `sch_isolement` | Schéma d'isolement social | `non_diagnostic_construct` |
| `sch_mefiance` | Schéma de méfiance et d'abus | `non_diagnostic_construct` |
| `sch_negativite` | Schéma de négativité et de pessimisme | `non_diagnostic_construct` |
| `sch_punition` | Schéma de punition | `non_diagnostic_construct` |
| `sch_recherche_approbation` | Schéma de recherche d'approbation | `non_diagnostic_construct` |
| `sch_surcontrole` | Schéma de surcontrôle émotionnel | `non_diagnostic_construct` |
| `sch_vulnerabilite` | Schéma de vulnérabilité au danger | `non_diagnostic_construct` |
| `sym_6a02` | Troubles du spectre de l'autisme | `clinical_reference_partial` |
| `sym_6a03` | Trouble développemental de l'apprentissage | `clinical_reference_partial` |
| `sym_6a05` | Trouble du déficit de l'attention avec ou sans hyperactivité | `clinical_reference_partial` |
| `sym_6a21` | Trouble schizoaffectif | `clinical_reference_partial` |
| `sym_6a22` | Trouble schizotypique | `clinical_reference_partial` |
| `sym_6a23` | Trouble psychotique aigu et transitoire | `clinical_reference_partial` |
| `sym_6a24` | Trouble délirant | `clinical_reference_partial` |
| `sym_6a61` | Trouble bipolaire de type II | `clinical_reference_partial` |
| `sym_6a62` | Trouble cyclothymique | `clinical_reference_partial` |
| `sym_6a70` | Épisode dépressif unique | `clinical_reference_partial` |
| `sym_6a71` | Trouble dépressif récurrent | `clinical_reference_partial` |
| `sym_6a73` | Trouble anxieux et dépressif mixte | `clinical_reference_partial` |
| `sym_6b00` | Trouble d'anxiété généralisée | `clinical_reference_partial` |
| `sym_6b03` | Phobie spécifique | `clinical_reference_partial` |
| `sym_6b21` | Trouble de la perception du corps | `clinical_reference_partial` |
| `sym_6b22` | Trouble de perception olfactive | `clinical_reference_partial` |
| `sym_6b24` | Trouble d'accumulation compulsive | `clinical_reference_partial` |
| `sym_6b250` | Trichotillomanie | `clinical_reference_partial` |
| `sym_6b251` | Trouble d'excoriation compulsive | `clinical_reference_partial` |
| `sym_6b61` | Amnésie dissociative | `clinical_reference_partial` |
| `sym_6b65` | Trouble dissociatif de l'identité partiel | `clinical_reference_partial` |
| `sym_6b66` | Trouble dépersonnalisation-déréalisation | `clinical_reference_partial` |
| `sym_6b83` | Trouble de l’apport alimentaire évitant-restrictif | `clinical_reference_partial` |
| `sym_6b84` | Pica | `clinical_reference_partial` |
| `sym_6b85` | Trouble de rumination-régurgitation | `clinical_reference_partial` |
| `sym_6c21` | Dysphorie de l'intégrité corporelle | `clinical_reference_partial` |
| `sym_6c42` | Troubles dus à l'utilisation de cannabinoïdes de synthèse | `clinical_reference_partial` |
| `sym_6c43` | Troubles dus à la consommation d’opioïdes | `clinical_reference_partial` |
| `sym_6c44` | Troubles dus à la consommation de sédatifs, d’hypnotiques ou d’anxiolytiques | `clinical_reference_partial` |
| `sym_6c45` | Troubles dus à la consommation de cocaïne | `clinical_reference_partial` |
| `sym_6c46` | Troubles dus à l'utilisation de stimulants, comprenant les amphétamines, la méthamphétamine ou la méthcathinone | `clinical_reference_partial` |
| `sym_6c47` | Troubles dus à l'utilisation de cathinones de synthèse | `clinical_reference_partial` |
| `sym_6c49` | Troubles dus à la consommation d’hallucinogènes | `clinical_reference_partial` |
| `sym_6c4b` | Troubles dus à l'utilisation de produits volatils inhalés | `clinical_reference_partial` |
| `sym_6c4c` | Troubles dus à la consommation de methylene-dioxymethamphetamine [MDMA] ou de drogues apparentées, y compris le methylene-dioxyamphetamine [MDA] | `clinical_reference_partial` |
| `sym_6c4d` | Troubles dus à l'utilisation de médicaments dissociatifs, notamment la kétamine ou la phencyclidine [PCP] | `clinical_reference_partial` |
| `sym_6c51` | Trouble du jeu | `clinical_reference_partial` |
| `sym_6c70` | Pyromanie | `clinical_reference_partial` |
| `sym_6c71` | Kleptomanie | `clinical_reference_partial` |
| `sym_6c73` | Trouble explosif intermittent | `clinical_reference_partial` |
| `sym_6d10` | Trouble de la personnalité | `clinical_reference_partial` |
| `sym_6d110` | Affectivité négative dans un trouble de la personnalité ou une difficulté de la personnalité | `specifier_not_standalone` |
| `sym_6d111` | Détachement dans un trouble de la personnalité ou une difficulté de la personnalité | `specifier_not_standalone` |
| `sym_6d112` | Dissocialité dans un trouble de la personnalité ou une difficulté de la personnalité | `specifier_not_standalone` |
| `sym_6d113` | Désinhibition dans un trouble de la personnalité ou une difficulté de la personnalité | `specifier_not_standalone` |
| `sym_6d114` | Anankastie dans un trouble de la personnalité ou une difficulté de la personnalité | `specifier_not_standalone` |
| `sym_6d50` | Trouble factice imposé à soi-même | `clinical_reference_partial` |
| `sym_6d71` | Trouble neurocognitif léger | `clinical_reference_partial` |
| `sym_7a00` | Insomnie chronique | `outside_cddr_pending` |
| `sym_7a20` | Narcolepsie | `outside_cddr_pending` |
| `sym_7a21` | Hypersomnie idiopathique | `outside_cddr_pending` |
| `sym_7a26` | Syndrome d'insuffisance de sommeil | `outside_cddr_pending` |
| `sym_8a05` | Tics | `outside_cddr_pending` |
| `sym_ha00` | Trouble du désir sexuel hypoactif | `outside_cddr_pending` |
| `sym_ha01` | Trouble de l'excitation sexuelle | `outside_cddr_pending` |
| `sym_ha02` | Troubles de l'orgasme | `outside_cddr_pending` |
| `sym_ha03` | Troubles de l'éjaculation | `outside_cddr_pending` |
| `sym_ha20` | Trouble de la pénétration sexuelle douloureuse | `outside_cddr_pending` |
| `ten_alexithymie` | Difficulté à identifier et exprimer ses émotions (alexithymie) | `non_diagnostic_construct` |
| `ten_automutilations_non_suicidaires` | Automutilations sans intention suicidaire | `non_diagnostic_construct` |
| `ten_codependance` | Codépendance | `non_diagnostic_construct` |
| `ten_colere` | Colère retenue ou qui déborde | `non_diagnostic_construct` |
| `ten_conduites_suicidaires` | Idées suicidaires, tentatives ou automutilations | `non_diagnostic_construct` |
| `ten_difficultes_personnalite` | Difficultés de personnalité au sens large (repère de recherche) | `aggregate_not_diagnosis` |
| `ten_difficultes_professionnelles` | Difficultés d'emploi et de revenus à l'âge adulte | `non_diagnostic_construct` |
| `ten_difficultes_sexuelles` | Difficultés dans la vie sexuelle | `non_diagnostic_construct` |
| `ten_dissociation` | Symptômes dissociatifs | `non_diagnostic_construct` |
| `ten_douleurs_persistantes` | Douleurs persistantes et maux physiques à répétition | `non_diagnostic_construct` |
| `ten_dysregulation_emotionnelle` | Difficultés à réguler ses émotions | `non_diagnostic_construct` |
| `ten_ecrans` | Usage problématique des écrans, d'internet ou des jeux vidéo | `non_diagnostic_construct` |
| `ten_estime_soi` | Estime de soi fragilisée | `non_diagnostic_construct` |
| `ten_externalisation` | Comportements d’opposition, d’agressivité ou antisociaux | `non_diagnostic_construct` |
| `ten_honte` | Honte de soi | `non_diagnostic_construct` |
| `ten_hypervigilance` | Hypervigilance à la menace | `non_diagnostic_construct` |
| `ten_idees_suicidaires` | Idées suicidaires | `non_diagnostic_construct` |
| `ten_impulsivite` | Impulsivité (agir sans prendre le temps de réfléchir) | `non_diagnostic_construct` |
| `ten_inferences_negatives` | Interprétations négatives durables des difficultés | `non_diagnostic_construct` |
| `ten_jeu_argent_problematique` | Difficultés avec les jeux d’argent | `aggregate_not_diagnosis` |
| `ten_mode_parent_exigeant` | Mode du parent exigeant | `non_diagnostic_construct` |
| `ten_narcissisme_grandiose` | Grandiosité narcissique (dimension de recherche) | `non_diagnostic_construct` |
| `ten_narcissisme_vulnerable` | Vulnérabilité narcissique (dimension de recherche) | `non_diagnostic_construct` |
| `ten_nevrosisme` | Sensibilité aux émotions négatives (névrosisme) | `non_diagnostic_construct` |
| `ten_peur_intimite` | Peur de l'intimité affective | `non_diagnostic_construct` |
| `ten_preoccupation_apparence` | Préoccupation excessive pour l'apparence | `aggregate_not_diagnosis` |
| `ten_procrastination` | Procrastination | `non_diagnostic_construct` |
| `ten_qualite_relationnelle` | Relations proches moins satisfaisantes | `non_diagnostic_construct` |
| `ten_risques_sexuels` | Prises de risque dans la sexualité | `non_diagnostic_construct` |
| `ten_rumination` | Pensées en boucle (rumination) | `non_diagnostic_construct` |
| `ten_sexualite_compulsive` | Comportements sexuels difficiles à contrôler | `aggregate_not_diagnosis` |
| `ten_sommeil` | Difficultés de sommeil (insomnie, cauchemars) | `non_diagnostic_construct` |
| `ten_surcompensation_schema` | Surcompensation face aux schémas | `non_diagnostic_construct` |
| `ten_symptomes_depressifs_perinataux` | Symptômes dépressifs pendant la grossesse ou après la naissance | `aggregate_not_diagnosis` |
| `ten_symptomes_premenstruels` | Symptômes prémenstruels importants | `aggregate_not_diagnosis` |
| `ten_tabagisme_persistant` | Tabagisme quotidien persistant | `non_diagnostic_construct` |
| `ten_tentatives_suicide` | Tentatives de suicide | `non_diagnostic_construct` |
| `ten_traits_personnalite_antisociale` | Traits de personnalité antisociale (repère de recherche) | `non_diagnostic_construct` |
| `ten_traits_schizotypiques` | Traits schizotypiques mesurés par questionnaires | `non_diagnostic_construct` |
| `ten_usage_cannabis` | Usage problématique du cannabis | `aggregate_not_diagnosis` |
| `ten_usage_substances` | Usage problématique d’alcool, de tabac ou de drogues | `non_diagnostic_construct` |
| `ten_violences_amoureuses_jeunes` | Violences subies dans les premières relations amoureuses | `non_diagnostic_construct` |
| `ten_violences_couple` | Violences dans le couple à l'âge adulte | `non_diagnostic_construct` |

Inventaire : 157/157 identifiants, aucune omission silencieuse. Une future promotion de couverture exige une règle versionnée, ses sources réellement consultées et sa relecture indépendante.

## Implémentation autorisée après proposition

Le contrôleur a autorisé le moteur descriptif proposé. `journal/src/clinical-assessment.mjs` implémente désormais `buildClinicalAssessment(snapshot)` avec le contrat réel de capture : `schema_version:1`, `complete:true`, `review_id`, `captured_at`, et `sections.symptoms:[{answers,notes:{factors}}]`. L'adaptateur conserve les valeurs 0/1/2/unknown/past ; 1 et 2 indiquent la fenêtre annuelle, pas une intensité clinique. Toute note passée est requise. Les rubriques enfance, conversation, mémoire et diagnostics rapportés ne sont pas utilisées comme preuves des manifestations.

Le résultat porte `method:declared_manifestations_only`, `diagnostic_status:not_diagnostic`, version de règles, empreinte SHA-256 du JSON de référence, date et identifiant du bilan. Les 157 sujets reçoivent un statut de couverture et les signes déclarés, ni score clinique ni verdict. Les besoins d'attribution au produit restent explicites même avec une porte d'usage positive.

Neuf tests significatifs passent : inventaire exhaustif, passé avec note, distinction inconnu/non/absence, absence de preuve à partir des propos IA ou de l'enfance, prudence temporelle, portes produit, déterminisme et validation du snapshot. L'empreinte est contrôlée contre le fichier source par un test ; toute nouvelle version de banque doit mettre à jour l'empreinte et revoir les règles.

La recherche est suffisante pour ce contrat descriptif et son abstention explicite ; elle ne prétend pas fournir 157 procédures diagnostiques cliniquement validées. Les pages NICE inaccessibles et les catégories de couverture partielle restent signalées dans le document, sans transformer une impossibilité de consultation en vérification réussie.

## Révision de contrat après relecture indépendante

Version des règles `2026-11-19.2` : les contextes obligatoires sont typés (`product`, `perinatal`, `relationship`, `intent`, `behaviour`) dans `context_gates`. Les informations manquantes ne parlent de produit que pour un produit. Un lien modéré seul donne `association_only`, jamais `criterion_signals`. L'indication historique reste séparée.

La note complète n'est pas dupliquée dans chaque association : chaque preuve contient `note_ref`, `note_excerpt` (160 octets UTF-8 au maximum, sans couper un caractère), et `note_excerpted`. La note intégrale est conservée dans l'instantané et consultable dans son archive. Un test maximal vérifie que le résultat reste inférieur à 800 000 octets avec une note multioctet de plus de 5 000 caractères par question, sans perdre la fin du texte original dans le snapshot. Les neuf tests passent.
