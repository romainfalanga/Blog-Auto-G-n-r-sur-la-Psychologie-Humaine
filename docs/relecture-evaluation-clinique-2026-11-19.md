# Relecture indépendante — évaluation descriptive archivée — 19 novembre 2026

## Verdict

Le moteur examiné ne pose aucun diagnostic et n’affirme pas que tous les critères d’un trouble sont satisfaits. Il est utilisable comme **inventaire daté de déclarations et d’informations à préciser**, après correction des réserves de restitution ci-dessous. Il ne répond pas à une promesse de « diagnostics officiels » ou de 157 procédures cliniques complètes ; cette limite doit rester identique dans l’interface et dans les textes générés.

Une abstention sur la durée, la simultanéité, le retentissement, le début, les exclusions et le différentiel est justifiée par les entrées réellement consommées. Ce n’est pas un outil vide : il conserve les réponses, leurs notes, la fenêtre annuelle et les déclarations anciennes, avec provenance et version. En revanche, il ne doit pas être présenté comme une évaluation de ces six dimensions : pour l’instant, il constate seulement qu’elles n’ont pas été recueillies sous une forme exploitable.

## Périmètre et vérifications effectives

Lecture complète de `journal/src/clinical-assessment.mjs`, `journal/tests/clinical-assessment.test.mjs` et `docs/recherche-evaluation-clinique-2026-11-19.md`. Lecture ciblée de leur consommateur `journal/src/review-archive-interface.mjs`. Aucun code modifié par cette revue.

Exécution des sept tests : succès. Reproduction indépendante des catégories et comparaison automatique des 157 lignes de la recherche avec les sorties : aucune divergence. Répartition : 61 `clinical_reference_partial`, 12 `aggregate_not_diagnosis`, 68 `non_diagnostic_construct`, 6 `specifier_not_standalone`, 10 `outside_cddr_pending`.

Ces 157 statuts sont une couverture d’inventaire et de restitution, pas une validation clinique individuelle. La règle `coverage()` dépend des métadonnées de banque, pas d’un contrôle des exigences diagnostiques. En particulier, les catégories d’usage de substances peuvent contenir plusieurs sous-types ; leur statut partiel n’autorise pas à conclure à une dépendance précise.

Sources réellement réexaminées : PDF OMS CDDR déjà conservé sous `work/reprise-nov19/sources/cddr.pdf`, texte associé, introduction sur les professionnels qualifiés et le jugement clinique, section « Essential (required) features », page imprimée 22, distinction avec « Additional clinical features », puis exclusions/différentiel, pages 24–25. [Notice officielle OMS](https://www.who.int/publications/i/item/9789240077263). Ces passages distinguent les caractéristiques essentielles des caractéristiques associées et indiquent que la classification ne remplace pas l’évaluation de la personne. Ils confortent la frontière descriptive retenue. Les sources HAS/NICE de la recherche sont citées avec leurs limites d’accès ; je ne transforme pas leurs extraits indexés en lecture intégrale ni en validation nouvelle des seuils. Les sources non CDDR ont été examinées dans la revue indépendante antérieure ; aucune procédure diagnostique nouvelle n’est revendiquée ici.

## Corrections nécessaires avant restitution publiée

### 1. Les portes ne concernent pas toutes un produit

`clinical-assessment.mjs`, construction de `missing_fact_ids` : toute porte produit des besoins `symptom_product_attribution` et parfois `current_product_use`. Cela concerne à tort la grossesse, les relations amoureuses à l’adolescence/à l’âge adulte, la procrastination et l’absence d’intention suicidaire.

Correction concrète : typer les exigences (`substance`, `life_context`, `defining_declaration`) ou employer des identifiants génériques, puis n’ajouter l’attribution au produit que pour les consommations. Conserver pour chaque exigence l’identifiant, la réponse, sa période et son statut ; ne pas transformer une note libre en attribution prouvée.

Le rendu testé lit `outcome.product_gates.length`, alors que le moteur retourne un objet avec `required_factor_ids`. La précision de contexte n’est donc jamais montrée. Lire le tableau interne ou le nouveau contrat typé, et afficher le contexte correspondant, pas une phrase universelle sur les produits.

### 2. Une absence de réponse ne doit pas être attribuée à l’utilisateur

Le moteur conserve correctement `answer:'unanswered'` et `answer:'unknown'` comme deux valeurs distinctes dans `evidence.unknown`. Le rendu les regroupe sous « Réponses “Je ne sais pas” ». Une personne qui n’a jamais répondu se voit ainsi attribuer une réponse.

Correction concrète : deux groupes, « Non renseigné » et « Vous avez répondu “Je ne sais pas” », ou un intitulé englobant explicite et une valeur visible par ligne. Ne pas déduire de Non un diagnostic écarté : le moteur respecte déjà cette règle.

### 3. Ne pas confondre association isolée et appui caractéristique

La condition `some(e => e.strength !== 'limitee')` donne le même état `supported_signals` à une unique association modérée et à une caractéristique constitutive. Reproduction : `agir_sur_un_coup_de_tete:1` fait apparaître le sujet bipolaire agrégé avec un seul lien `moderee`, à côté d’autres sujets. Ce résultat n’est pas un diagnostic dans le code, mais une carte titrée par un trouble peut être comprise comme une hypothèse clinique personnellement étayée.

Correction faisable : distinguer « manifestation caractéristique rapportée » de « association non spécifique seulement ». Une association modérée seule peut rester dans les informations contextuelles, sans promouvoir le sujet à la liste des hypothèses soutenues. À défaut, nommer l’état « déclaration associée au sujet » et montrer explicitement le caractère non spécifique. Ne pas inventer de seuil numérique prétendument clinique pour résoudre ce point. Même plusieurs liens essentiels ne valident ni un syndrome complet ni un diagnostic.

### 4. Afficher les limites propres au sujet

Le moteur expose les statuts de couverture, mais le rendu consulté ne les montre pas sur les cartes. Il faut distinguer les construits descriptifs, les groupes agrégés, les qualificatifs et les catégories cliniques partielles. Sinon une carte de schéma peut avoir la même apparence qu’une maladie nommée, malgré des sémantiques différentes dans le JSON.

Proposition : une ligne sobre par carte (« style ou tendance descriptive », « groupe de manifestations », « repère clinique incomplet », « référentiel spécifique non évalué »). Pour les qualificatifs de personnalité, rappeler qu’ils ne sont pas des diagnostics autonomes. La référence OMS générale seule est une référence bibliographique, pas une règle clinique vérifiée pour chaque lien ; éviter le mot « critères vérifiés » à ce stade.

## Temporalité et limites de l’abstention

- Les valeurs 1 et 2 restent dans la fenêtre annuelle et ne deviennent pas un symptôme présent aujourd’hui. La note de `past` est obligatoire et non convertie en durée. Conforme.
- Les preuves annuelles et anciennes sont séparées. Les portes historiques exigent des réponses historiques : cela évite une promotion actuelle à partir d’un usage ancien. Cela ne démontre toutefois jamais que deux déclarations anciennes sont contemporaines. Le champ `contemporaneity:'unknown'` doit rester visible.
- Un sujet comportant des signes annuels et anciens reçoit un seul état principal annuel, mais les deux listes de preuves sont conservées. Cette priorité de présentation n’autorise pas à parler d’épisode continu ou de rémission.
- Toutes les dimensions autres que les manifestations restent inconnues, même si une note semble donner une durée. C’est une abstention correcte tant que les notes ne sont pas transformées silencieusement en faits. Une future collecte structurée devra dater, attribuer et confirmer les faits ; elle devra distinguer « déclaré » de « vérifié professionnellement ».
- « Tous Non » ne devient ni `ruled_out` ni absence de trouble à vie. Les états `not_supported`/`conflicting_information` du contrat documentaire ne sont pas encore implémentés. Ne pas annoncer une détection des contradictions qui n’existe pas.
- Les déclarations de clinicien sont ignorées comme preuve des symptômes, ce qui est correct. Cela ne remplace pas un registre séparé des diagnostics rapportés avec attribution, date et provenance ; aucun diagnostic « confirmé » ne peut être déduit de la mémoire ou d’une sortie d’IA.
- Le circuit de sécurité n’est pas évalué par cette fonction. Un signal historique ne démontre pas un danger actuel ; une réponse annuelle ne le tranche pas non plus. Le traitement des urgences doit rester distinct et ne pas attendre le bilan.

## Tests ciblés recommandés

1. Porte périnatale, relationnelle, NSSI et procrastination : aucun besoin de produit inventé ; rendu effectif de la précision du contexte.
2. Association modérée isolée : état contextuel/non spécifique, pas présentation comme hypothèse caractéristique soutenue.
3. Absence de réponse versus « Je ne sais pas » : libellés différents dans l’interface.
4. Toutes réponses Non : déclarations négatives visibles, aucune exclusion diagnostique.
5. Manifestations historiques à dates incompatibles : listes conservées, simultanéité inconnue, jamais fusion en épisode.
6. Construits/qualificatifs/hors CDDR : statuts visibles malgré toutes réponses Oui.
7. Le snapshot conservé et l’évaluation ne changent pas après modification des réponses ou de la banque : ceci relève de l’archivage et doit être testé à l’intégration, pas présumé par le déterminisme de cette fonction.

## Avis final du périmètre

Acceptable comme socle descriptif avec réserves de restitution à intégrer, **pas comme moteur de diagnostic ni comme validation de toutes les exigences des 157 sujets**. Les corrections 1 et 2 sont des erreurs concrètes de contrat/rendu ; les points 3 et 4 empêchent une surinterprétation prévisible des cartes. Le schéma « dimensions inconnues » est honnête et utile si les déclarations sont bien restituées, mais l’interface doit dire précisément quelles informations sont disponibles et lesquelles n’ont pas encore été évaluées.
