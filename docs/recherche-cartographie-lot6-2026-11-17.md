# Recherche cartographie — lot 6 — 17 novembre 2026

Fragment préparé, non publié : **2 sujets, 9 liens neufs, 9 sources nouvelles, 8 corrections proposées**. Les 16 repères initialement `a_examiner` et les 55 recouvrements `partiel` ont une décision explicite par `pattern_id`. Le registre contient aussi cinq décisions de famille. Aucun quota de liens n’a été forcé.

## Cadre et périmètre

AGENTS.md, reprise du 16 novembre (état du 17), plan de couverture complet, données publiées, catalogue de psychologie et audit du 16 novembre ont été lus. Les autres brouillons et le travail en cours sont préservés. Seuls le fragment de ce lot et ce rapport ont été écrits. Pas de commit, push, génération de fichiers publics ni publication.

Les associations ne deviennent ni diagnostics ni récits causaux individuels. Un comportement actuel, une compétence ordinaire, une mesure de laboratoire et un vécu infantile restent distincts. Le choix monétaire différé chez les jeunes est **contextuel**, sans hypothèse sur l’origine d’un trouble neurodéveloppemental.

## Ce qui est ajouté et pourquoi

| Sujet | Liens | Niveau | Limite décisive |
| --- | --- | --- | --- |
| `ten_inferences_negatives` | abus émotionnels ; violences sexuelles | limité | Style attributionnel précis ; études anciennes peu nombreuses, surtout rétrospectives. Ne représente pas toutes les distorsions. |
| `ten_recompense_immediate_jeunes` | précarité ; addiction parentale ; punitions corporelles | modéré pour revenus, limité pour les deux autres | Choix monétaires expérimentaux avant 18 ans ; carte contextuelle, pas prédiction des choix adultes. |
| `ten_codependance` existant | abus émotionnels ; négligence émotionnelle | limité | Quelques études transversales dans une revue systématique ; notion contestée, sans diagnostic consensuel. |
| `ten_qualite_relationnelle` existant | violences physiques ; négligence matérielle | modéré puis limité | Cohorte prospective américaine ; indicateur composite de fonctionnement amoureux. L’effet de négligence dépend des ajustements. |

## Sources effectivement consultées

Les chemins de lecture détaillés figurent dans `sources[].read`. Les pages PMC bloquées par le navigateur ont été obtenues avec `urllib`, leur HTML lu après retrait des balises ; les PDF accessibles ont été extraits en mémoire avec `pdftotext`. Un extrait de moteur de recherche n’est pas compté comme une lecture intégrale.

- **[Gibb 2002](https://pubmed.ncbi.nlm.nih.gov/11806020/)** : méthode, définition et résultats quantitatifs lus dans le [PDF auteur](https://www.binghamton.edu/psychology/labs/mood/pdfs/2002_gibb_childhood_mal_cog_styles_meta-analysis.pdf), pp. 225–233. Quatre études émotionnelles, dix sexuelles ; associations faibles, résultat sexuel surtout chez les adultes plus âgés. Pas de résultat physique significatif. Choix conservateur `limitee`, même si la synthèse est quantitative.
- **[Felton et al. 2026, en ligne 2025](https://doi.org/10.1017/S0954579425100886)** : méthodes, résultats par exposition, figures et discussion lus. Vingt-huit articles, vingt échantillons ; effet revenus très faible. Les différences familiales d’addiction et les résultats disciplinaires sont moins robustes. Aucune extrapolation aux adultes.
- **[Molina et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13067074/)** : méthode, tableau 2, synthèse développementale et limites lus. Les données pertinentes sont transversales, sans effet combiné. Divergence interne à signaler : répartition quantitative/qualitative/théorique différente entre résumé et §3.2. Aucun niveau supérieur à `limitee`.
- **[Labella et al. 2018](https://socialinteractionlab.psych.umn.edu/sites/socialinteractionlab.psych.umn.edu/files/files/2018/Labella%2520et%2520al%2520%2528PSPB%252C%25202018%2529.pdf)** : méthodes pp. 240–243, résultats/tableaux pp. 243–247 lus. Cohorte de 179 personnes, expositions documentées jusqu’à 17,5 ans, fonctionnement amoureux de 20 à 32 ans. Le lien physique reste après ajustement ; celui de négligence s’atténue. Le composite ne mesure pas toute relation amicale.
- **[Dariotis et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10341267/)** : méthodes et synthèses quantitatives/qualitatives lues. Quatre-vingt-quinze études, très majoritairement transversales ; résultats plus cohérents pour la dimension émotionnelle. Pas d’extension aux étiquettes familiales figées ni de devoir de « grandir grâce » à l’adversité.
- **[Curran et Hill 2022](https://pubmed.ncbi.nlm.nih.gov/35357848/)**, identifiant existant `curran2022` : [manuscrit accepté](https://ray.yorksj.ac.uk/id/eprint/5802/1/Curran%20%26%20Hill%20%282022%29%20PB.pdf), étude 1 méthodes/résultats et limites lues. Vingt et une études, dimensions distinctes ; association surtout socialement prescrite. La mesure des attentes et critiques recoupe des questionnaires de perfectionnisme, avec hétérogénéité et direction incertaine. Source non redéclarée.
- **[Kenyhercz et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11149883/)** : méthodes, synthèse famille d’origine et limites lues. Les cinq méta-analyses concernent surtout le fonctionnement social actuel ; les neuf études intergénérationnelles sont traitées narrativement. Aucun facteur existant n’équivaut au travail excessif du parent.
- **[Sharratt et Ridout 2025, en ligne 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12057305/)** : méthode, tableaux de résultats et limites lus. Échantillon transversal de 159 adultes, mesures agrégées d’adversité. Ne permet pas un lien propre surprotection/imprévisibilité → incertitude ; aucune carte nouvelle.
- **[Gisselbaek et al. 2026, en ligne 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC13129615/)** : méthodes et tableau des revues lus. La revue des facteurs parentaux de l’imposture est évaluée de qualité très faible. Complément prudent de l’existant, sans nouveaux liens par analogie entre faible soin et négligence.
- **[Jhang et al. 2023, en ligne 2022](https://doi.org/10.1002/mar.21750)** : plan, résultats de l’étude 3 et discussion/limites lus. Une piste empirique existe pour niveau socioéconomique rappelé et coûts irrécupérables. Ce n’est pas une cohorte ni une synthèse admissible selon le plan : registre `a_examiner`, pas affirmation d’absence de recherche.

## Corrections proposées, non appliquées

1. Étendre la trace de lecture de `curran2022` ; préciser les deux liens critiques/attentes scolaires et leurs dimensions.
2. Rendre `pat_parentification` contextuel : c’est un vécu, déjà recueilli comme facteur, et non une issue adulte.
3. Passer parentification → codépendance de modèle à association **limitée**, sur la revue récente, sans en faire des synonymes.
4. Compléter les sources parentification → symptômes anxieux/dépressifs en conservant le niveau limité et les distinctions de mesure.
5. Compléter le lien contrôle → imposture sans relever sa force.

Les résumés existants ne sont jamais redéclarés dans `links`. Les sources proposées par une correction de type `conserver` sont à fusionner avec les références existantes, sans les effacer. Les 61 exclusions anciennes ne sont pas rouvertes.

## Les 16 décisions

`nouvelle_carte` ne signifie pas couverture totale du repère. Aucun alias exact nouveau n’est proposé. Les exclusions motivées par un seuil documentaire restent révisables.

| pattern_id | Décision | Cible éventuelle | Raison |
| --- | --- | --- | --- |
| `intellectualization` | exclusion |  | Coordination lot 4 : exclusion proposée faute de lien enfance spécifique ; analyser utilement un problème ne suffit pas à conclure à une défense. Pas d’alias avec alexithymie. Sources du lot 4 non revérifiées ici. |
| `uncertainty` | exclusion |  | L’étude lue utilise un total d’adversités et un devis transversal. Elle ne permet pas d’isoler surprotection ou imprévisibilité ; seuil revue/cohorte non atteint pour un lien nouveau. Besoin de certitude et anxiété ne sont pas synonymes. |
| `mind_reading` | exclusion |  | L’inférence des jugements d’autrui n’est ni le style attributionnel mesuré ni une méfiance générale. Aucune revue ou cohorte spécifique suffisamment vérifiée ; pas de carte par voisinage lexical. |
| `all_or_nothing` | exclusion |  | La revue de style cognitif porte sur des inférences stables/globales et non sur une mesure distincte de pensée dichotomique. Le perfectionnisme ne suffit pas à en faire un alias. Aucun lien spécifique admissible trouvé. |
| `confirmation` | exclusion |  | Recherche ciblée sans source admissible reliant une exposition infantile définie au biais de confirmation. Processus de jugement ordinaire, sans alias diagnostique. |
| `overwork` | exclusion |  | La revue trouve peu de données sur la famille d’origine, avec résultats mixtes. Le travail excessif parental est distinct des 58 facteurs ; pas d’extrapolation depuis parentification ou pression scolaire. Le repère ordinaire n’est pas une addiction au travail. |
| `overgeneralization` | partiel | `ten_inferences_negatives` | Une nouvelle carte couvre les attributions globales après un événement négatif. Le repère inclut aussi des généralisations hors de ce contexte : correspondance partielle seulement, pas alias exact. |
| `emotional_reasoning` | exclusion |  | Prendre une émotion pour une preuve n’est pas mesuré par les études d’attributions négatives lues. Aucun lien infantile spécifique admissible trouvé ; aucune équivalence avec intensité émotionnelle. |
| `sunk_cost` | exclusion |  | Une publication rapporte des expériences de décision liées au niveau socioéconomique infantile rappelé. Ce n’est ni une cohorte développementale ni une synthèse : seuil d’admission du plan non atteint. Dire preuve préliminaire, non absence de recherche. |
| `reward_delay` | nouvelle_carte | `ten_recompense_immediate_jeunes` | Carte contextuelle de choix monétaire chez enfants/adolescents. Elle couvre le construit expérimental, seulement une partie du repère quotidien effort/bénéfice ; aucune origine infantile d’un trouble neurodéveloppemental ni identité adulte inférée. |
| `minimization` | exclusion |  | Coordination lot 4 : pas d’équivalence avec déni ou rationalisation. Respecter une discrétion choisie ; aucune nouvelle carte ni source du lot 4 revérifiée ici. |
| `rationalization` | exclusion |  | Coordination lot 4 : pas de lien infantile vérifié ni de motif inconscient inféré à partir d’une explication cohérente. Sources du lot 4 non revérifiées ici. |
| `observational_learning` | exclusion |  | L’apprentissage par observation est un mécanisme général, pas une issue clinique ou un vécu infantile à déduire d’un comportement. Il peut éclairer un récit explicite ; aucune nouvelle carte de conséquence. |
| `relational_jealousy` | exclusion |  | Coordination lot 5 : aucune preuve infantile spécifique suffisamment vérifiée. La jalousie n’est pas un alias d’attachement anxieux, de peur d’abandon ou de contrôle du partenaire ; sources du lot 5 non revérifiées ici. |
| `social_media_comparison` | exclusion |  | Le repère décrit l’environnement numérique actuel et les contenus sélectionnés. Il ne peut servir de conséquence de la comparaison fraternelle ni d’un vécu infantile ; aucun alias avec usage problématique des écrans. |
| `self_focused_attention` | partiel | `icd_6b04` | Le modèle de l’anxiété sociale peut comporter une surveillance de soi, mais le repère est transdiagnostique. Pas d’alias intégral ni de lien infantile distinct ajouté à partir de la seule cooccurrence clinique. |

## Relecture des 55 recouvrements partiels

Chaque ligne demeure **partielle** : aucune compétence ordinaire n’est transformée en trouble et aucune similitude de vocabulaire ne suffit à un alias. Les cibles anciennes et les alternatives ont été comparées aux descriptions du catalogue. Une mention de coordination rapporte une décision du lot concerné ; elle ne certifie pas sa bibliographie. Les nouveaux liens de ce lot seuls ont été vérifiés ici.

| pattern_id | Décision et cibles | Motif de non-équivalence |
| --- | --- | --- |
| `rumination` | partiel : `ten_rumination` | Coordination lot 5 : carte de rumination proposée. Le repère plus large inclut une réflexion répétée encore utile ; ne pas convertir toute répétition en rumination problématique. |
| `catastrophizing` | partiel : `sch_negativite`, `sch_vulnerabilite`, `ten_inferences_negatives` | Une anticipation catastrophique peut recouper les schémas et certaines inférences négatives, sans être un style durable ni une mesure spécifique dans Gibb. Pas de carte autonome faute de lien infantile propre vérifié. |
| `emotional_identification` | partiel : `ten_alexithymie` | La difficulté d’identification est représentée ; la capacité ordinaire et son apprentissage ne se confondent pas avec l’alexithymie. Coordination lot5 ; preuves de ce lot non revérifiées ici. |
| `anger_needs` | partiel : `ten_colere`, `ressenti_colere_interdite` | Expression ou rétention de colère présentes ; le repérage des besoins et la différence entre émotion et action sont plus larges. Coordination lot5 ; preuves de ce lot non revérifiées ici. |
| `shame` | partiel : `ten_honte` | Coordination lot 5 : carte distincte proposée, sans assimiler honte située et schéma durable d’imperfection. |
| `body_stress` | partiel : `ten_douleurs_persistantes`, `icd_6c20` | Coordination lot 7. Des sensations liées à une situation ne démontrent ni douleur chronique ni trouble de détresse corporelle. Les causes médicales et les émotions ordinaires restent ouvertes. |
| `comparison` | partiel : `comparaison_fratrie`, `moqueries_corps` | Distinguer expositions infantiles à des comparaisons et comparaison sociale actuelle. Le contexte numérique appartient encore à une autre description ; aucune flèche enfance → comparaison globale ajoutée. |
| `boundaries` | partiel : `sch_assujettissement`, `pat_needs_expression` | L’effacement et les besoins sont représentés ; exprimer librement une limite est une compétence plus large. |
| `reassurance` | partiel : `sch_recherche_approbation` | La recherche d’approbation couvre une partie du besoin de validation ; conseil ponctuel et réassurance de sécurité sont distincts. Coordination lot5 ; preuves de ce lot non revérifiées ici. |
| `reciprocity` | partiel : `ten_qualite_relationnelle`, `sch_abnegation` | Qualité des liens et sacrifice de soi touchent la réciprocité, sans évaluer le donner/recevoir dans chaque relation. |
| `trust` | partiel : `sch_mefiance` | Conserver le recouvrement : ajuster sa confiance selon des faits est une compétence, distincte d’un schéma de méfiance et de la confiance épistémique. Les pistes récentes retrouvées ne sont pas assez lues par facteur pour ajouter un lien. |
| `family_loyalty` | partiel : `sch_fusion`, `sch_assujettissement` | Fusion ou soumission peuvent recouper une tension de loyauté ; une loyauté choisie n’est ni l’une ni l’autre. Coordination lot5 ; preuves de ce lot non revérifiées ici. |
| `closeness_distance` | partiel : `att_anxieux`, `att_evitant`, `ten_peur_intimite` | Coordination lot 5 pour l’attachement. Choisir une distance ajustée n’est pas un style insécure ; la peur d’intimité ne décrit qu’une difficulté possible. Aucun alias total. |
| `relational_security` | partiel : `prot_adulte_chaleureux`, `prot_amities` | Des appuis sûrs sont recueillis ; ils ne mesurent pas toute la liberté de rester soi dans un lien actuel. Coordination lot7 ; preuves de ce lot non revérifiées ici. |
| `avoidance` | partiel : `ten_procrastination`, `att_evitant` | Le report d’action et certaines distances relationnelles ne couvrent pas toutes les formes d’évitement. |
| `overcontrol` | partiel : `sch_surcontrole`, `pat_perfectionism` | Coordination lot 5. Le surcontrôle des émotions, le perfectionnisme et la maîtrise générale des situations ne sont pas interchangeables ; les dimensions de Curran ne justifient pas cet alias. |
| `withdrawal` | partiel : `pat_social_isolation`, `att_evitant` | Un retrait temporaire pour récupérer peut être choisi et utile. Il ne mesure pas l’isolement subi, l’attachement évitant ou un trait de détachement ; aucune carte autonome ajoutée. |
| `impulsive_reaction` | partiel : `sch_controle_insuffisant`, `ten_externalisation` | Coordination lot 4 pour désinhibition. Une réaction sous tension n’est ni un trait stable, ni le choix monétaire différé étudié ici ; aucun alias avec la nouvelle carte de récompense. |
| `grief_change` | partiel : `deces_proche`, `icd_6b42` | Décès et deuil prolongé présents ; changements ordinaires, pertes non mortelles et deuil normal ne sont pas équivalents. |
| `autonomy` | partiel : `surprotection`, `controle`, `sch_dependance` | Les obstacles à l’autonomie sont approchés ; choix libre et dépendance matérielle réelle ne sont pas des schémas. Coordination lot5 ; preuves de ce lot non revérifiées ici. |
| `needs_expression` | partiel : `pat_needs_expression` | Même thème des besoins mais la carte porte une difficulté ; elle ne couvre pas toute la compétence de négociation. |
| `support_seeking` | partiel : `prot_soutien_apres`, `prot_soins` | L’existence d’aides est recueillie ; savoir demander et préférer un type de soutien restent distincts. Coordination lot7 ; preuves de ce lot non revérifiées ici. |
| `emotional_flexibility` | partiel : `ten_dysregulation_emotionnelle`, `prot_ressources_personnelles` | Les difficultés de régulation et des appuis sont présents ; leur absence ne démontre pas une souplesse émotionnelle. Coordination lot5 ; preuves de ce lot non revérifiées ici. |
| `personalization` | partiel : `pat_guilt`, `ressenti_responsable`, `pat_overresponsibility` | Responsabilité et culpabilité excessives recouvrent une partie de l’attribution personnelle ; pas un mécanisme cognitif identique. |
| `negative_filter` | partiel : `sch_negativite` | Le schéma négativité/pessimisme contient une attention au négatif ; un filtre ponctuel n’établit pas ce schéma. Coordination lot5 ; preuves de ce lot non revérifiées ici. |
| `distress_tolerance` | partiel : `ten_dysregulation_emotionnelle`, `sch_controle_insuffisant` | Coordination lot 5. Capacité à rester en contact avec l’inconfort et possibilités de soutien dépassent les difficultés de régulation ; pas d’équivalence avec un schéma. |
| `joy_permission` | partiel : `sch_surcontrole`, `sch_negativite`, `pat_guilt` | L’inhibition, l’anticipation négative et la culpabilité peuvent recouper ce thème ; aucune carte spécifique de permission de joie. Coordination lot5 ; preuves de ce lot non revérifiées ici. |
| `self_trust` | partiel : `sch_dependance`, `ten_estime_soi` | Coordination lot 5. Jugement situé, confiance dans ses préférences et estime globale ne sont pas identiques ; aucune preuve propre ne justifie de les fusionner. |
| `authenticity` | partiel : `pat_conditional_worth`, `sch_recherche_approbation` | La dépendance au regard extérieur est approchée ; la présentation choisie et protectrice de soi dépasse ces cartes. |
| `assertive_request` | partiel : `pat_needs_expression`, `sch_assujettissement` | Les difficultés à exprimer ses besoins sont présentes ; la demande négociable n’est pas évaluée comme compétence. |
| `receiving_help` | partiel : `sch_abnegation`, `pat_guilt` | Dette et sacrifice peuvent gêner la réception d’aide ; la capacité à recevoir reste un sujet distinct. |
| `power_imbalance` | partiel : `pat_appeasement`, `emprise_religieuse`, `controle` | Des contextes de pouvoir sont présents dans l’enfance ; dépendances professionnelles et économiques actuelles sont plus larges. |
| `family_mediation` | partiel : `triangulation`, `parentification` | Une médiation infantile imposée peut recouper ces facteurs ; une médiation adulte consentie n’est ni parentification ni rôle de sauveur. |
| `consistency_in_relationships` | partiel : `imprevisibilite`, `soins_instables`, `prot_adulte_chaleureux` | Prévisibilité des soins infantiles approchée ; cohérence des liens actuels n’est pas mesurée. |
| `separation_reunion` | partiel : `ressenti_abandon`, `parent_absent`, `pat_abandonment_fear` | Absence et crainte du lien sont présentes ; attentes de contact et retrouvailles ordinaires dépassent ce périmètre. Coordination lot5 ; preuves de ce lot non revérifiées ici. |
| `co_regulation` | partiel : `prot_adulte_chaleureux`, `prot_soutien_apres` | Présences soutenantes recueillies ; le processus concret d’apaisement partagé n’a pas de carte propre. Coordination lot7 ; preuves de ce lot non revérifiées ici. |
| `task_initiation` | partiel : `ten_procrastination` | Le report d’action est représenté ; manque de moyens, flou et fatigue ne sont pas couverts comme causes alternatives structurées. |
| `flexible_standards` | partiel : `pat_perfectionism`, `sch_exigences_elevees` | Adapter ses exigences est une compétence. Les dimensions mesurées du perfectionnisme n’autorisent pas à considérer toute exigence ou toute qualité de travail comme pathologique. |
| `financial_strain` | partiel : `precarite`, `ten_difficultes_professionnelles` | Précarité infantile et difficultés de revenus sont présentes ; dettes et possibilités actuelles restent à recueillir séparément. Coordination lot7 ; preuves de ce lot non revérifiées ici. |
| `housing_instability` | partiel : `precarite`, `demenagements`, `prot_installation` | Quelques contextes de logement instable sont présents ; déménagement choisi et instabilité actuelle ne sont pas identiques. Coordination lot7 ; preuves de ce lot non revérifiées ici. |
| `caregiver_load` | partiel : `parentification`, `parent_malade_physique`, `maladie_famille` | La parentification suppose un rôle infantile inadapté à l’âge. Une aidance adulte choisie, même lourde, ne doit pas devenir un alias de parentification ou de codépendance. |
| `illness_adjustment` | partiel : `maladie_enfant`, `icd_6b43` | Maladie vécue enfant et trouble d’adaptation sont présents ; une adaptation normale à une santé déclarée ne vaut pas diagnostic. Coordination lot7 ; preuves de ce lot non revérifiées ici. |
| `access_barriers` | partiel : `prot_soins`, `precarite` | Accès à des soins et précarité approchés ; délais, langue, transport ou accueil ne sont pas détaillés. Coordination lot7 ; preuves de ce lot non revérifiées ici. |
| `ambiguous_loss` | partiel : `parent_absent`, `separation_conflit` | Absence et perte de contact peuvent recouvrir certaines pertes ambiguës ; toute séparation n’en est pas une. |
| `community_participation` | partiel : `prot_ecole`, `prot_amities` | Appartenance scolaire et amitiés approchées ; voisinage et participation collective adulte ne sont pas évalués. Coordination lot7 ; preuves de ce lot non revérifiées ici. |
| `self_efficacy` | partiel : `prot_ressources_personnelles`, `sch_dependance` | Appuis et difficulté à se sentir compétent présents ; l’efficacité située ne se confond pas avec estime globale ou dépendance. Coordination lot7 ; preuves de ce lot non revérifiées ici. |
| `emotional_detachment` | partiel : `ten_dissociation`, `pat_emotional_suppression` | Coordination lots 4 et 5 : détachement comme trait, émoussement momentané, dissociation et suppression sont distincts. Pas d’alias automatique entre ces phénomènes. |
| `safety_behaviors` | partiel : `ten_hypervigilance`, `anx_generale` | Les comportements de sécurité se définissent par leur fonction dans la situation, pas par une vigilance élevée. Aucun lien infantile spécifique lu ; conserver le recouvrement sans carte ou alias. |
| `responsibility_repair` | partiel : `pat_guilt`, `pat_overresponsibility` | Responsabilité excessive représentée ; contribution réelle et réparation choisie restent des objets distincts. |
| `attentional_capture` | partiel : `ten_hypervigilance` | L’attention captée par une menace ne représente pas la captation par un signal neutre ou gratifiant. Une mesure cérébrale de menace ne prouve pas une hypervigilance vécue ; aucune nouvelle équivalence. |
| `anticipatory_arousal` | partiel : `ten_hypervigilance`, `anx_generale` | Activation corporelle avant un événement : conserver le recouvrement sans écarter enthousiasme, effort ou cause physique ; pas d’inférence infantile nouvelle. |
| `sleep_rhythm` | partiel : `ten_sommeil` | Les difficultés de sommeil sont présentes ; horaires, travail posté et possibilités de repos ne sont pas explicitement structurés. Coordination lot3 ; preuves de ce lot non revérifiées ici. |
| `rest_permission` | partiel : `pat_guilt`, `pat_perfectionism`, `sch_exigences_elevees` | La culpabilité ou les exigences peuvent gêner le repos, mais une absence de relais et des contraintes de travail sont des conditions actuelles distinctes. Pas d’alias avec addiction au travail. |
| `intimacy_communication` | partiel : `ten_peur_intimite`, `ten_difficultes_sexuelles` | Difficultés possibles présentes ; communication choisie, rythme et sécurité dépassent ces issues. |
| `role_expectations` | partiel : `enfant_prefere_role`, `enfant_sage`, `pression_scolaire`, `discrimination` | Les rôles familiaux assignés peuvent recouper certaines attentes sociales. Les normes actuelles liées aux identités ne sont pas une séquelle infantile générale ni une identité clinique. |

## Recherches non converties en liens

- Les mots clés ont couvert parentification, codépendance, imposture, perfectionnisme parental, style cognitif, incertitude, récompense différée, confiance, jalousie, travail excessif, biais de confirmation et coûts irrécupérables avec enfance/adversité/maltraitance et revue/cohorte. Recherche ciblée, non revue systématique exhaustive.
- Cao et al. sur maltraitance émotionnelle et couple : notice/résumé accessibles, texte complet non obtenu ; **aucun lien ajouté à partir de cette seule notice**.
- La méta-analyse de réponse cérébrale à la menace ne doit pas être convertie en carte d’hypervigilance vécue. Les liens `mclaughlin2020` restent à la charge du lot 9.
- Mentalisation et confiance : des publications récentes ont été repérées, mais leurs résultats par facteur n’ont pas été suffisamment vérifiés. Pas de nouvelle carte de mentalisation ni de confiance « déficiente ».
- Les typologies « héros/clown/enfant parfait » et la généralisation du « sauveur » ne reçoivent pas de nouvelles preuves empiriques par voisinage avec la parentification.
- Les trois défenses restent coordonnées avec le lot 4 ; honte/rumination et attachement avec le lot 5 ; ressources ordinaires avec le lot 7. Aucune carte ni paire concurrente n’a été créée pour ces lots.

## Contrôles

Contrôles finaux effectués par lecture et appel non écrivant à `b.build()`. L’audit daté du 16 doit être renouvelé par l’intégrateur ; il ne suffit pas de changer une empreinte. Une relecture clinique indépendante reste à réaliser avant intégration/publication.

- Validation JSON, identifiants, sources, facteurs originaux, domaines et relation explicite : aucune erreur.
- Aucune collision d’identifiant, URL de source redéclarée ou paire avec les données publiées et les fragments présents lors du contrôle. Chaque nouveau sujet porte au moins un lien.
- Neuf résumés entre 220 et 254 caractères ; aucun terme interdit ni pourcentage. Deux liens modérés, sept limités, aucun lien fort.
- `b.build()` sur l’arbre partagé : **59 facteurs, 110 sujets, 532 liens, 273 sources** au moment du contrôle. Une seule erreur, attendue : `audit des correspondances périmé : les facteurs, sujets, liens ou ressources ont changé`. Aucun autre défaut. Ces totaux incluent les autres lots et ne sont pas un état publié.
- Les 71 lignes du catalogue dans `out_of_scope` sont un registre de décisions pour l’intégrateur. Le champ `decision_semantique: partiel` conserve le statut d’audit partiel ; le `statut: a_examiner` imposé par le vocabulaire du registre ne signifie pas qu’il faudrait créer 55 cartes. Aucun alias complet n’est demandé.

La nouveauté est volontairement limitée. L’accès incomplet à des articles, les données transversales, les expositions composites et les différences entre compétences ordinaires et issues étudiées sont conservés comme limites, sans remplissage causal du graphe.
