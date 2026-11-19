# Relecture clinique indépendante — Mes symptômes v2 — 19 novembre 2026

## Statut

Relecture ciblée en cours, **pas de validation clinique exhaustive de la banque**. La simple résolution des références et des codes ne valide pas les poids. Ne pas présenter cette note comme un avis de psychologue humain ou comme une validation de toutes les associations.

## Sources effectivement ouvertes

- OMS, CDDR 2024, ISBN 9789240077263, PDF intégral de 852 pages via le [miroir de la bibliothèque nationale HRB](https://www.drugsandalcohol.ie/41519/1/WHO_Clinical_descriptions_and_diagnostic_requirements_ICD-11.pdf). Passages effectivement inspectés : schizotypie p.177–179, pica p.414–415, rumination-régurgitation p.416–417 ; index des catégories et mentions de licence. Le téléchargement initial était incomplet ; le fichier complet a ensuite été vérifié par empreinte (voir extension).
- [Schema Therapy : définitions des 18 schémas](https://www.schematherapy.com/id73.htm), ouverture et lecture initiale ; comparaison exhaustive des items restant à faire.
- [Kwapil et Barrantes-Vidal, Schizotypy: Looking Back and Moving Forward](https://pmc.ncbi.nlm.nih.gov/articles/PMC4373633/), ouverture et identification vérifiées ; ne vaut pas relecture de tous les liens associés.

## Corrections appliquées

1. Pica : les questions consommation de matières et ingrédients sont des variantes du même signe d'ingestion non nutritive. Famille commune `pica_ingestion_non_nutritive`, pour ne pas compter deux caractéristiques indépendantes.
2. Rumination : remonter, remâcher/recracher et absence de nausée décrivent le même phénomène ; famille commune `regurgitation`. Les formulations restent distinctes dans le questionnaire, sans tripler leur poids dans le classement.
3. La fugue correspond à une forme d'amnésie et n'est pas requise pour l'amnésie dissociative générale : lien `per_fugue` vers `sym_6b61` ramené à `moderee`, en accord avec la variante déjà présente dans les supplémentaires. Vérifier que la fusion n'annule pas cette correction.
4. Le discours fortement désorganisé de `per_pensee_desorganisee` ne doit pas constituer un critère essentiel de schizotypie : le CDDR distingue le discours inhabituel de l'incohérence majeure. Lien vers `sym_6a22` ramené à `moderee`; les liens vers les autres sujets ne sont pas modifiés.

## Conditions restantes

- Portes d'usage à vérifier pour **tous** les produits, y compris alcool, cannabis et nicotine. Une difficulté générale de dépendance ne doit pas désigner un produit non déclaré.
- Relire l'ensemble des sources non CIM, les catégories sommeil/sexualité hors CDDR et les associations des schémas/traits. Ces sources ne sont pas toutes consultées dans cette note.
- Contrôler après fusion la cohérence des familles, les poids et le plafond de critères sans créer artificiellement trois signes.
- Le classement de signes sur douze mois ne vérifie pas les durées, exclusions somatiques, substances, retentissement ou trajectoires nécessaires à un diagnostic. Ambre doit expliciter ce qui manque et ne jamais traiter ce classement comme une probabilité clinique calibrée.
- Les formulations originales ne doivent pas devenir une traduction suivie de listes sous licence sans dérivés ; aucune attestation juridique globale n'est fournie.

Conclusion actuelle : corrections locales nécessaires identifiées et appliquées ; **validation globale de diffusion non acquise par cette relecture ciblée**.

## Extension de la revue CIM-11

Le PDF et le texte local sont maintenant complets ; SHA-256 `b66e7a5d6ed764a1d39ed6ed37d581d25b5b9b79a6ee74b4935ff6677e8e709b`, identique au manifeste historique. Les pages ci-dessous désignent la pagination imprimée (PDF +17).

| Domaine | Passages lus | Vérification et réserves |
|---|---|---|
| Neurodéveloppement | 124–125,134,142–143,153–154 | Communication/routines, apprentissages, attention/impulsivité, tics : signes cohérents ; trajectoire développementale indispensable et non déduite des réponses annuelles. |
| Psychoses | 162–163,170–171,177–179,180,185 | Hallucinations, convictions, pensée, contrôle et symptômes négatifs distingués ; correction discours schizotypique effectuée. Schizoaffectif exige humeur et psychose contemporaines, non vérifiées par le score. |
| Humeur | 212–213,217–218,222–223,225,230,240,244,246,258–259 | Signes des épisodes cohérents. Les sous-types requièrent une histoire temporelle et des exclusions ; pas de déduction du type I/II ni d'épisode récurrent par simple addition. |
| Anxiété | 266,271,275–276,280,284,289 | Signes de peur, évitement, manifestations corporelles et contexte différenciés. Une crise isolée ne suffit pas au trouble panique. |
| Obsessionnel et apparentés | 300–301,308,314,317–318,322,327,329 | Obsessions/compulsions, apparence, odeurs, santé, accumulation, cheveux/peau : ancrages spécifiques lus ; le retentissement et les exclusions ne sont pas inférés. |
| Stress/deuil | 339,344–345,348–349,352 | Reviviscence distincte de souvenir/rumination ; cauchemars génériques PTSD abaissés, rumination générale adaptation abaissée. Deux signes précis suffisent à la banque adaptation, sans fabriquer un troisième. |
| Dissociation | 366,373,382,387,391–392 | Fugue non obligatoire pour l'amnésie ; réalité préservée pour dépersonnalisation ; différentiel somatique/substance non déterminable par questionnaire. |
| Alimentation | 398–399,404,407,410–411,414–417 | Restriction, crises, compensation, évitement et ingestion/régurgitation inspectés. Familles pica et rumination corrigées. |
| Symptômes corporels | 429–430,435 | Détresse corporelle ne signifie pas absence de maladie physique ; désir de handicap et conséquences distincts. |
| Consommations | 463–467 | Usage seul insuffisant ; portes retirées des critères classants. Tolérance/sevrage regroupés, comme priorité/poursuite malgré dommages. Prescription incluse dans les portes opioïdes/calmants. |
| Jeux et impulsions | 506,510,520,523,527 | Perte de contrôle, priorité, conséquences et motifs examinés ; sexualité moralement désapprouvée seule insuffisante. |
| Personnalité | 554–555,560–566 | Fonctionnement de soi/interpersonnel et domaines de traits examinés ; durée, généralisation et retentissement restent à explorer. |
| Factice/neurocognition | 593–594,608 | Demande de soins non assimilée à tromperie ; déclin cognitif doit être objectivé, non assimilé à un oubli ordinaire. |

Cette matrice atteste une revue des blocs essentiels et du rattachement des familles, pas la validation indépendante de chaque assertion secondaire des 1 124 liens. Les extraits de certaines lectures groupées ont été tronqués par l'outil : la liste n'atteste donc pas une lecture intégrale de tous ces chapitres.

### Corrections additionnelles appliquées

- `cauchemars_repetes` → `icd_6b40` : `moderee`, question non spécifique à l'événement.
- `pensees_en_boucle` → `icd_6b43` : `moderee`, préoccupation générique sans stress identifié. Exception documentée à deux critères, p.352.
- Les dix portes produits supplémentaires : `limitee`, avec explication que l'usage ne suffit pas. Opioïdes/calmants sous prescription également admissibles aux portes.
- `sup_chercher_a_etre_soigne` → `sym_6d50` : `limitee`, consultation seule non démonstrative. Exception deux critères, p.593–594.
- `sup_besoin_doses_plus_fortes`, `sup_mal_a_l_arret_du_produit`, `cmp_manque_a_l_arret`, `cmp_besoin_de_plus` : famille `adaptation_physiologique_produit`.
- `sup_conso_passe_avant` et `sup_continuer_malgre_problemes` : famille `priorite_consommation`.

Après ces corrections, régénération réussie : 513 questions, 157 sujets, 1 124 liens ; module 434 121 octets.

### Verdict de cette extension

La revue a corrigé des doubles comptages et des signes génériques surpondérés. **Pas de validation diagnostique ni de probabilité calibrée.** La validation globale reste conditionnée à la relecture non-CIM et à la contre-vérification des corrections dans le rendu, les calculs et les consignes d'Ambre. Les pondérations historiques explicitement abaissées pour tenir le plafond de 12 restent une limite éditoriale, pas une preuve scientifique de moindre valeur. Cette note ne certifie pas leur calibration.

## Contre-vérification finale du périmètre CIM

Les poids abaissés pour le seul plafond éditorial ont été réexaminés aux pages 212–213, 255 et 344–345. Réveils/indécision pour dépression, indécision pour dysthymie, et manifestations spécifiques du TSPT complexe ont retrouvé `critere`. Exceptions versionnées : dep_generale 14, icd_6a72 13, icd_6b41 17. Les cauchemars, la culpabilité et le détachement génériques restent `moderee` pour leur formulation sans contexte, avec motif clinique explicite remplaçant le plafond.

Les corrections de poids, les portes non classantes et les familles pica/régurgitation ont été contre-vérifiées dans le JSON généré. `build_symptomes_carte.py --check` réussit. Les premières portes alcool/cannabis/nicotine n'ont aucun lien classant, ce qui est correct.

Avis sur les corrections CIM examinées : favorable à leur intégration comme rapprochements descriptifs non diagnostiques. Le classement reste heuristique, sans probabilité clinique validée. Cet avis ne remplace pas la revue non-CIM ni les tests du moteur/UI, et ne constitue pas une attestation de chaque assertion secondaire non relue individuellement. Les documents de livraison doivent préserver cette portée exacte.
