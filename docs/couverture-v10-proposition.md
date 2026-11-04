# Couverture éditoriale V10 : audit et proposition

Base figée : `5eb09cf`, 4 novembre 2026. Le référentiel final est le [manifeste V10](../data/editorial_pack_v10.json), qui conserve le plan revu, les 13 fiches et les 156 récits. Les sujets et identités sont restés stables ; 28 angles secondaires ont été explicités après comparaison aux textes. Les constats initiaux ci-dessous décrivent la base de départ, et la grille finale référence les récits effectivement écrits.

Le catalogue est un ensemble de repères éditoriaux. Les récits sont fictifs. Ni l’un ni les autres ne prétendent couvrir toute la psychologie humaine, classer les troubles ou fournir un diagnostic.

## Inventaire vérifié

Les 231 fichiers Markdown publics au commit de référence ont été inventoriés avec leur frontmatter, leurs sujets et les recherches dans les corps. La concordance avec la matrice a été vérifiée. Les passages nécessaires au cas du sauveur et aux limites de provenance ont été relus dans le contenu réel. Cette revue de couverture n’est pas une validation scientifique ligne par ligne des 231 articles.

| Mesure | Résultat |
| --- | ---: |
| Articles publics, matrice et EXAMPLES | 231 chacun |
| Personnages existants | 20 |
| Nombre de récits par personnage | 11 à 13, médiane 12 |
| Sujets bruts différents | 191 |
| Articles pensées, émotions, schémas | 74, 77, 80 |
| Catalogue de repères | 161, 16 domaines, 44 sous-domaines |
| Slugs dupliqués ou différence matrice et contenu public | 0 |
| Corps entre 1800 et 2500 mots | 39 sur 231 |
| Corps sous 1800 mots | 192 sur 231 |
| Médiane et étendue des corps | 1708, de 612 à 2115 |
| Références des historiques absentes de la matrice courante | 60, dans les 20 personnages |

Le comptage exclut le frontmatter et inclut les titres du corps ; les mots conservent leurs apostrophes et traits d’union internes. Il peut différer légèrement du contrôleur du pack. La valeur déclarative `wordCount` ne remplace pas un comptage réel.

Le registre historique seul n’est pas une preuve de publication : Sophie y possède par exemple `genogramme-simplifie-seul-face-a-soi`, absent du corpus courant. Le contenu public et la matrice font foi pour la continuité. Root gère l’agrégation et la normalisation éventuelle des historiques.

Les répétitions brutes les plus fréquentes comptent trois occurrences : imposture, recherche d’approbation, exigences élevées, sacrifice de soi, isolement social, effet spotlight, trahison et mépris. Elles peuvent être utiles quand les situations diffèrent, mais elles ne remplacent pas des fonctions ou contextes absents. Sommeil, consentement, mémoire de travail et discrimination n’ont pas de sujet direct initial. Le sommeil apparaît dans 26 corps, la mémoire de travail dans un seul, consentement et discrimination dans aucun. Une mention incidente ne vaut pas un développement.

## Le cas du sauveur : présence conceptuelle et défaut de recherche

Le repère `overresponsibility`, « Rôle de sauveur », existe déjà dans le catalogue avec la nature `process`. Aucun récit public n’a ce rôle pour sujet principal. Il apparaît dans le sujet composite du triangle de Karpman et dans des corps voisins sur la codépendance ou le sacrifice. Ces thèmes se recouvrent sans être équivalents.

Exemples réels : [Karpman en couple](../content/posts/sortir-de-ses-schemas-repetitifs/jeux-psychologiques-karpman-couple.md), [codépendance en amitié](../content/posts/sortir-de-ses-schemas-repetitifs/causes-profondes-codependance-en-amitie.md), [sacrifice de soi en amitié](../content/posts/sortir-de-ses-schemas-repetitifs/abnegation-et-sacrifice-de-soi-en-amitie.md). Le premier traite plusieurs rôles, le deuxième le surinvestissement du lien, le troisième le coût du don de soi. Aider ne suffit pas à attribuer l’un de ces fonctionnements.

Au HEAD, `scripts/build_examples.py:160` sélectionne sujet et tags, puis trois mots-clés canoniques. La fonction `extractive_metadata`, ligne 109, conserve l’ouverture et la dernière scène avec leurs preuves. `journal/src/recommendations.mjs:40` indexe titre, description, ces extraits et les concepts. Il n’indexe pas le corps complet et n’utilise volontairement pas les traits du personnage comme sujet d’article. Une tendance dans la fiche de Léa ne rend donc pas tous ses récits pertinents pour le sauveur.

Repro local sur EXAMPLES, contexte vide et requête passée à `expandReadingQuery`, sans API :

| Requête au HEAD `5eb09cf` | Résultat observé |
| --- | --- |
| syndrome du sauveur | Imposture en amitié, Stockholm du quotidien, puis imposture après 60 ans ; Karpman cinquième. |
| complexe du sauveur | Stress post-traumatique complexe premier, Karpman deuxième. |
| besoin de sauver tout le monde | Effet bandwagon puis biais du monde juste. |

Le défaut combine une lacune de récit dédié et un rapprochement lexical sur « syndrome » ou « complexe ». Aline 1 fournit le sujet dédié. Alias pertinents : rôle de sauveur, complexe du sauveur, surresponsabilité, besoin de sauver les autres, aide sans s’oublier. Ils servent à rechercher des lectures, pas à établir une catégorie clinique. Root et v7_chat implémentent la vérification d’existence et les liens connexes ; ces changements en cours ne sont pas présentés comme présents au HEAD initial.

Au même HEAD, une demande sur le sommeil ou le consentement peut recevoir effet bandwagon à partir de termes génériques. Le contrôle d’existence doit porter sur le catalogue complet et distinguer article dédié, thème voisin et absence de correspondance. Une absence de lecture ne prouve rien sur le vécu du lecteur.

## Couverture des domaines

`D` : sujet principal direct ou synonyme éditorial proche. `C` : article connexe, sans équivalence complète. `—` : aucun sujet ou voisin suffisamment identifié par cette grille, sans prétendre qu’aucune phrase pertinente n’existe. Les rapprochements sont conservateurs : solitude n’est pas solitude choisie, somatisation n’est pas santé déclarée, illusion de contrôle n’est pas besoin de contrôle.

Grille des 161 repères : 44 avec sujet direct, 97 avec seulement un voisin identifié, 20 sans sujet ni voisin retenu. Un même récit peut apparaître pour plusieurs repères ; ces nombres ne sont pas des articles additionnables ni une mesure de validité scientifique.

Les 156 sujets root correspondent ici à 103 repères principaux distincts et 119 en ajoutant les angles secondaires proposés. Avec les sujets directs existants, l’union projetée est de 128 repères principaux sur 161. Il ne s’agit pas des 124 repères principaux de mon brouillon supersédé : le présent calcul porte exclusivement sur les sujets figés par root.

Un sujet prévu n’est pas encore un texte couvert. Un angle secondaire ne compte que si le corps lui donne une définition, une scène ou une distinction utile ; un mot dans un tag ne suffit pas. Les chiffres de projection devront être recalculés après lecture des nouveaux fichiers.

| Domaine | Repères | D initiaux | C initiaux | V10 principaux distincts | Avec secondaires proposés |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pensées et interprétations (`thinking`) | 18 | 12 | 6 | 7 | 9 |
| Émotions et corps (`emotions`) | 11 | 8 | 3 | 7 | 10 |
| Rapport à soi (`self`) | 9 | 3 | 6 | 8 | 9 |
| Relations et limites (`relationships`) | 12 | 2 | 10 | 10 | 11 |
| Parents et histoire familiale (`family`) | 10 | 4 | 5 | 1 | 1 |
| Sécurité et proximité (`attachment`) | 7 | 3 | 4 | 3 | 4 |
| Actions et habitudes (`action`) | 10 | 2 | 5 | 7 | 8 |
| Contexte et transitions (`context`) | 17 | 2 | 11 | 14 | 16 |
| Besoins et valeurs (`needs`) | 7 | 0 | 6 | 7 | 7 |
| Ressources et souplesse (`resources`) | 14 | 4 | 9 | 10 | 11 |
| Protections et adaptations (`adaptation`) | 10 | 3 | 6 | 5 | 5 |
| Attention et organisation (`executive`) | 10 | 0 | 7 | 5 | 7 |
| Corps, stress et sommeil (`body`) | 9 | 0 | 7 | 8 | 8 |
| Apprentissages et expérience (`learning`) | 5 | 0 | 4 | 3 | 3 |
| Intimité et vie affective (`intimacy`) | 6 | 1 | 4 | 4 | 5 |
| Identités et culture (`identity`) | 6 | 0 | 4 | 4 | 5 |

## Treize parcours figés

Les nouveaux prénoms sont distincts des 20 principaux, d’Ambre et de Romain, et avaient zéro occurrence dans les 231 corps au HEAD. Les auteurs gardent leurs secondaires et les faits de continuité dans leur manifeste de lot. L’ancien brouillon de plan n’écrase pas leurs choix. Les scènes doivent varier et la progression rester limitée, révisable, parfois décevante, sans guérison miracle.

### Aline, 39 ans

Responsable de quincaillerie ; en couple sans enfant, engagée dans une association. Surresponsabilité, limites et réciprocité. Une aide libre ou nécessaire ne prouve pas un rôle de sauveur. La fatigue de compassion exige une exposition à la détresse réellement écrite et ne doit pas devenir le nom de toute fatigue bénévole.

Sources de travail : `cci_assertiveness`, `nhs_relationships`, `nih_social`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Syndrome du sauveur : aider sans s’effacer | `overresponsibility` |  |
| 2 | Surresponsabilité et partage des tâches | `overresponsibility` | `reciprocity` |
| 3 | Culpabilité lorsque l’on se repose | `rest_permission` | `guilt` |
| 4 | Poser une limite sans retirer son affection | `boundaries` |  |
| 5 | Réciprocité dans les relations | `reciprocity` |  |
| 6 | Tolérer la déception de l’autre | `distress_tolerance` | `disappointment` |
| 7 | Demander si une aide est souhaitée | `active_listening` | `consent_boundaries` |
| 8 | Recevoir de l’aide sans se justifier | `receiving_help` |  |
| 9 | Garder un désaccord sans rompre le lien | `conflict` |  |
| 10 | Fatigue de compassion et relais concrets | `emotional_overload` | `support_seeking` |
| 11 | Quand l’aide ne résout pas tout | `overresponsibility` | `overcontrol` |
| 12 | Choisir une entraide fidèle à ses valeurs | `values_alignment` |  |

### Rémi, 32 ans

Agent logistique ; en colocation. Attention, temps, démarrage et appuis externes, sans déduction de TDAH. Séparer conditions du travail, fatigue et stratégies personnelles.

Sources de travail : `nimh_attention`, `nimh_working_memory`, `nhs_problem_solving`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Interruptions et surcharge attentionnelle | `attention_switching` | `attentional_capture` |
| 2 | Mémoire de travail et appuis extérieurs | `working_memory_load` | `external_supports` |
| 3 | Difficulté à démarrer une tâche | `task_initiation` |  |
| 4 | Découper une tâche sans perdre son but | `planning_steps` |  |
| 5 | Distractibilité et environnement | `distractibility` |  |
| 6 | Estimation du temps et marges réelles | `time_estimation` |  |
| 7 | Organisation du travail et charge réelle | `workload_context` |  |
| 8 | Procrastination et émotion de la tâche | `avoidance` |  |
| 9 | Perfectionnisme au moment de terminer | `perfectionism` |  |
| 10 | Erreur et manière de se parler | `self_criticism` |  |
| 11 | Déléguer et transmettre des consignes | `communication_clarification` | `external_supports` |
| 12 | Habitudes et indices du contexte | `habit_cues` |  |

### Maëlle, 27 ans

Opératrice agroalimentaire en horaires alternés ; en couple. Sommeil et récupération soumis à des horaires réels. Le sommeil ne s’obtient pas par volonté ; les changements d’organisation restent visibles et aucun effet de technique n’est garanti.

Sources de travail : `nhlbi_sleep`, `nhs_sleep`, `who_work`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Décalage du rythme en horaires alternés | `sleep_rhythm` |  |
| 2 | Inquiétude de ne pas réussir à dormir | `sleep_onset` | `uncertainty` |
| 3 | Repos et culpabilité de ne rien produire | `rest_permission` | `guilt` |
| 4 | Reconnaître la fatigue sans se juger | `fatigue_pacing` | `self_compassion` |
| 5 | Pensées répétées après un poste de nuit | `rumination` |  |
| 6 | Rythmes différents dans le couple | `work_home_conflict` |  |
| 7 | Stress lié aux conditions de travail | `workload_context` | `body_stress` |
| 8 | Récupération et appuis disponibles | `overwork` | `practical_support` |
| 9 | Percevoir ses besoins corporels | `body_awareness` | `needs_expression` |
| 10 | Fatigue et concentration au travail | `distractibility` |  |
| 11 | Retrouver un rythme après une période irrégulière | `sleep_rhythm` |  |
| 12 | Faire reconnaître des besoins de repos collectifs | `needs_expression` | `power_imbalance` |

### Samira, 41 ans

Caissière intérimaire ; mère seule d’un garçon de neuf ans. Finances, logement, parentalité seule et accès. Les contraintes ne sont pas des biais à corriger. Ne pas inventer un droit ni une issue administrative à partir d’un guide psychologique.

Sources de travail : `who_social`, `who_work`, `nih_social`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Précarité et place prise par les soucis | `financial_strain` |  |
| 2 | Incertitude du logement et sentiment de sécurité | `housing_instability` |  |
| 3 | Honte financière et regard des autres | `shame` |  |
| 4 | Demander les aides auxquelles on a droit | `access_barriers` | `support_seeking` |
| 5 | Charge mentale de la parentalité seule | `work_home_conflict` | `working_memory_load` |
| 6 | Colère face à une situation injuste | `anger_needs` |  |
| 7 | Pouvoir agir dans une marge étroite | `self_efficacy` | `power_imbalance` |
| 8 | Décider sous la pression du temps | `decision_overload` |  |
| 9 | Parler d’un budget sans réduire sa valeur | `conditional_worth` |  |
| 10 | Faire une place à un projet reporté | `purpose_direction` |  |
| 11 | Soutien de voisinage et réciprocité | `reciprocity` | `practical_support` |
| 12 | Dignité et besoins matériels fondamentaux | `stress_resources` | `autonomy` |

### Éloi, 19 ans

Apprenti cuisinier ; en foyer de jeunes travailleurs. Apprentissage, essais, retours et autonomie soutenue. Le progrès concerne des gestes et des situations, sans intelligence fixe ni style d’apprentissage attribué.

Sources de travail : `ies_learning`, `cci_procrastination`, `cci_perfectionism`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Recevoir un retour pendant l’apprentissage | `feedback_learning` |  |
| 2 | Sentiment d’efficacité et petites expériences | `self_efficacy` | `behavioral_experiment` |
| 3 | Expliquer un échec sans se condamner | `balanced_self_evaluation` |  |
| 4 | Comparaison avec des collègues expérimentés | `comparison` |  |
| 5 | Motivation et intérêt pour une activité | `motivation_ambivalence` |  |
| 6 | Apprendre plutôt que prouver sa valeur | `conditional_worth` |  |
| 7 | Demander de l’aide comme débutant | `support_seeking` |  |
| 8 | Progression et droit à l’essai | `behavioral_experiment` |  |
| 9 | Orientation et valeurs personnelles | `values_alignment` |  |
| 10 | Reprendre après une erreur concrète | `responsibility_repair` |  |
| 11 | Reconnaître ses compétences transférables | `skill_practice` |  |
| 12 | Autonomie et cadre qui soutient | `autonomy` |  |

### Inès, 30 ans

Bibliothécaire ; en couple avec une femme depuis trois ans. Consentement libre et révisable, intimité et vie privée dans un lien ordinaire. Ni dette sexuelle, ni fréquence de référence, ni trauma supposé.

Sources de travail : `who_sexual_health`, `nhs_relationships`, `cci_assertiveness`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Consentement et droit de changer d’avis | `consent_boundaries` |  |
| 2 | Différence de désir dans le couple | `desire_variation` |  |
| 3 | Parler d’une limite intime | `intimacy_communication` | `boundaries` |
| 4 | Vie privée au sein du couple | `privacy_boundaries` |  |
| 5 | Image du corps et intimité | `self_criticism` | `body_awareness` |
| 6 | Fatigue et disponibilité affective | `closeness_distance` | `affection_preferences` |
| 7 | Sécurité affective et demande claire | `relational_security` | `assertive_request` |
| 8 | Crainte du jugement dans l’intimité | `shame` | `safe_disclosure` |
| 9 | Réparer une conversation blessante | `repair_after_conflict` |  |
| 10 | Dire non sans fournir une preuve | `consent_boundaries` |  |
| 11 | Confiance et espace personnel | `trust` | `privacy_boundaries` |
| 12 | Construire des moments partagés sans obligation | `positive_connection` | `affection_preferences` |

### Vincent, 44 ans

Technicien ferroviaire ; vit avec son compagnon qui a une mobilité réduite acquise. Aidance réelle, relais et autonomie du compagnon. Distinguer aide nécessaire, surprotection et aide non demandée ; le proche conserve ses décisions.

Sources de travail : `nih_social`, `nhs_relationships`, `who_stress`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Charge d’aidance et tâches invisibles | `caregiver_load` |  |
| 2 | Autonomie du proche et décisions partagées | `autonomy` |  |
| 3 | Aide consentie et envie de protéger | `active_listening` | `overresponsibility` |
| 4 | Ambivalence des émotions dans l’aidance | `emotional_ambivalence` |  |
| 5 | Reconnaître sa fatigue sans culpabiliser | `fatigue_pacing` | `guilt` |
| 6 | Organiser un relais concret | `practical_support` |  |
| 7 | Préserver une relation de couple dans l’aidance | `role_transition` | `closeness_distance` |
| 8 | Rendre visible la charge d’organisation | `working_memory_load` | `caregiver_load` |
| 9 | Valeurs importantes qui entrent en tension | `values_conflict` |  |
| 10 | Colère et obstacles matériels | `anger_needs` | `stress_resources` |
| 11 | Demander du soutien en gardant ses limites | `support_seeking` | `boundaries` |
| 12 | Répit et entraide durable | `overwork` | `rest_permission` |

### Salomé, 29 ans

Luthière ; personne non binaire avec accords féminins choisis, célibataire. Reconnaissance, confidentialité et appartenance. Salomé est non binaire, choisit elle et les accords féminins ; F désigne la grammaire du futur registre. Aucune psychologie ne découle de son genre ; aucun dévoilement obligatoire.

Sources de travail : `who_social`, `who_work`, `who_sexual_health`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Reconnaissance du nom et manière d’être adressée | `identity_recognition` |  |
| 2 | Choisir ce que l’on révèle de soi | `disclosure_choice` |  |
| 3 | Pression des normes sociales | `role_expectations` |  |
| 4 | Sécurité et adaptation au contexte | `appeasement` | `power_imbalance` |
| 5 | Appartenance sans effacement de soi | `social_belonging` |  |
| 6 | Identité au-delà d’une seule étiquette | `authenticity` |  |
| 7 | Se parler avec bienveillance après une interaction | `self_compassion` |  |
| 8 | Vie privée et droit de ne pas expliquer | `privacy_boundaries` |  |
| 9 | Compétence professionnelle et regard des autres | `self_trust` | `impostor` |
| 10 | Poser une limite à la curiosité | `boundaries` |  |
| 11 | Continuité de soi au fil des changements | `identity_transition` |  |
| 12 | Alliances et soutien dans la vie quotidienne | `positive_connection` | `support_seeking` |

### Malik, 36 ans

Mécanicien de tramway ; arrivé du Sénégal par choix il y a quatre ans, en couple à distance. Migration choisie, appartenances et liens à distance. Aucun exil forcé ou trauma présumé, aucune culture homogène qui expliquerait sa personnalité.

Sources de travail : `who_migration`, `nih_social`, `who_social`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Appartenances multiples et sentiment de place | `bicultural_navigation` | `social_belonging` |
| 2 | Changer de codes selon le contexte | `bicultural_navigation` | `self_expression` |
| 3 | Langue et expression des émotions | `emotional_identification` |  |
| 4 | Entretenir un lien à distance | `consistency_in_relationships` | `affection_preferences` |
| 5 | Solitude choisie et solitude subie | `solitude_choice` | `social_isolation` |
| 6 | Reconnaître une discrimination sans se réduire à elle | `discrimination_context` |  |
| 7 | Stress d’adaptation et ressources concrètes | `migration_adjustment` | `stress_resources` |
| 8 | Démarches administratives et charge mentale | `working_memory_load` | `access_barriers` |
| 9 | Ce qui manque d’un lieu familier | `grief_change` |  |
| 10 | Négocier des attentes familiales différentes | `family_loyalty` |  |
| 11 | Soutiens dans plusieurs cercles | `community_participation` | `support_seeking` |
| 12 | Garder une identité souple et personnelle | `identity_transition` |  |

### Odile, 74 ans

Ancienne agente postale ; retraitée, vit seule par choix ; sa sœur est décédée il y a six mois. Vie après 70 ans, perte et transmission. Odile peut déjà aller bien ; les émotions varient sans calendrier ni étapes obligatoires. Vivre seule ne prouve pas l’isolement.

Sources de travail : `nhs_grief`, `nih_social`, `nimh_selfcare`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Liens qui continuent après un décès | `continuing_bonds` |  |
| 2 | Oscillation du deuil et retour au quotidien | `grief_change` | `emotional_ambivalence` |
| 3 | Solitude choisie après une perte | `solitude_choice` |  |
| 4 | Retraite et changement de rôle | `role_transition` |  |
| 5 | Objets et souvenirs sans obligation de garder | `continuing_bonds` |  |
| 6 | Dates anniversaires et émotions variables | `grief_change` |  |
| 7 | Demander une présence précise | `support_seeking` |  |
| 8 | Participation collective et liberté personnelle | `community_participation` | `autonomy` |
| 9 | Sens dans une action ordinaire | `meaning_search` |  |
| 10 | Anticiper sans vivre toutes les pertes à l’avance | `anticipatory_arousal` | `catastrophizing` |
| 11 | Joie et chagrin dans une même journée | `joy_permission` | `emotional_ambivalence` |
| 12 | Transmettre un souvenir à sa manière | `creative_expression` | `continuing_bonds` |

### Anaïs, 46 ans

Agente d’accueil municipale ; en couple, polyarthrite diagnostiquée et suivie médicalement. Polyarthrite déclarée et suivie, énergie et appuis. Les activités sont adaptées au contexte de soin ; une émotion n’est pas la cause présumée de la maladie ou de toute douleur.

Sources de travail : `nhs_pain`, `who_work`, `nimh_selfcare`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Énergie et adaptation du rythme avec une maladie | `fatigue_pacing` | `illness_adjustment` |
| 2 | Incertitude de santé et informations fiables | `uncertainty` | `health_declared` |
| 3 | Fatigue et droit au repos | `rest_permission` |  |
| 4 | Capacités fluctuantes et projets | `flexible_coping` |  |
| 5 | Décrire ses sensations sans tout interpréter | `body_awareness` |  |
| 6 | Communiquer un besoin d’aide concret | `assertive_request` | `practical_support` |
| 7 | Handicap invisible et reconnaissance | `health_declared` | `identity_recognition` |
| 8 | Plans flexibles pour une journée variable | `planning_steps` | `flexible_standards` |
| 9 | Identité au-delà de la maladie | `identity_transition` |  |
| 10 | Activités adaptées et valeurs personnelles | `movement_regulation` | `enjoyable_activity`, `values_alignment` |
| 11 | Demander un aménagement au travail | `workload_context` | `power_imbalance` |
| 12 | Soutien et espoir réaliste | `resilience` | `positive_connection` |

### Baptiste, 37 ans

Coordinateur de maintenance des transports ; vit avec sa compagne et leur fille de cinq ans. Valeurs, compromis et décisions révisables. Le doute n’est pas automatiquement un biais et aucune option unique n’est présentée comme psychologiquement saine.

Sources de travail : `nhs_problem_solving`, `nhs_thought_record`, `who_stress`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Valeurs personnelles et arbitrages du quotidien | `values_alignment` | `decision_tradeoffs` |
| 2 | Décider avec une part d’incertitude | `uncertainty` |  |
| 3 | Coût d’opportunité et choix renoncés | `decision_tradeoffs` |  |
| 4 | Réversibilité d’une décision | `behavioral_experiment` |  |
| 5 | Regret et informations disponibles à l’époque | `metacognitive_check` |  |
| 6 | Conflit entre deux valeurs importantes | `values_conflict` |  |
| 7 | Sens et actions ordinaires | `meaning_search` |  |
| 8 | Penser au soi de demain sans s’oublier | `reward_delay` |  |
| 9 | Distinguer contrôle et influence | `overcontrol` | `autonomy` |
| 10 | Engagement progressif dans un projet | `purpose_direction` |  |
| 11 | Objectifs réalistes et ressources réelles | `small_goals` | `stress_resources` |
| 12 | Réviser un choix sans nier le chemin parcouru | `sunk_cost` | `flexible_coping` |

### Élise, 54 ans

Gestionnaire d’une résidence municipale ; séparée depuis huit ans, amitiés stables, participe à un jardin partagé. Ressources et souplesse ordinaires. Humour, distraction et routines peuvent fonctionner ; observer leur utilité et leur coût sans intention inconsciente certaine.

Sources de travail : `who_stress`, `cci_distress`, `nimh_selfcare`.

| Épisode | Sujet root | Repère principal | Angle secondaire proposé |
| ---: | --- | --- | --- |
| 1 | Souplesse psychologique face à un imprévu | `flexible_coping` | `emotional_flexibility` |
| 2 | Humour et place laissée aux émotions | `humor_coping` | `emotional_identification` |
| 3 | Distraction choisie plutôt que fuite automatique | `distraction_coping` |  |
| 4 | Savourer une expérience ordinaire | `enjoyable_activity` |  |
| 5 | Gratitude sans nier ce qui va mal | `gratitude` |  |
| 6 | Routines qui soutiennent sans enfermer | `habit_cues` |  |
| 7 | Ressources collectives et entraide concrète | `community_participation` | `practical_support` |
| 8 | Bienveillance envers soi après un revers | `self_compassion` |  |
| 9 | Changer de perspective sans nier un fait | `reframing` |  |
| 10 | Optimisme réaliste et préparations utiles | `metacognitive_check` |  |
| 11 | Acceptation et action possible | `distress_tolerance` | `problem_solving` |
| 12 | Explorer une possibilité sans tout bouleverser | `behavioral_experiment` |  |

## Lacunes restantes et frontières

Le pack ouvre nettement le sommeil, les contraintes matérielles, l’aidance, la santé déclarée, le consentement et la reconnaissance identitaire. Il ne couvre pas tous les repères. Les sujets généraux ne doivent pas masquer les différences entre interruptions et mémoire de travail, entre aidance et sauvetage, entre sécurité et disponibilité permanente, entre autonomie et absence d’appuis.

Les repères suivants n’ont ni sujet direct initial ni angle explicite dans le rapprochement du plan root. Ils restent visibles, sans ajouter automatiquement des personnages ou des articles au pack :

- Parents et histoire familiale : Attentes parentales intériorisées (`parent_expectations`) ; Apprentissages émotionnels familiaux (`emotional_learning`) ; Frontières dans la famille (`family_boundaries`) ; Se retrouver médiateur familial (`family_mediation`) ; Comparaisons dans la fratrie (`sibling_comparison`) ; Sujets familiaux difficiles à aborder (`family_secrecy`).
- Sécurité et proximité : Vivre les séparations et retrouvailles (`separation_reunion`) ; S’apaiser dans une présence sûre (`co_regulation`).
- Actions et habitudes : Retrait relationnel (`withdrawal`).
- Contexte et transitions : Perte sans clôture claire (`ambiguous_loss`).
- Ressources et souplesse : Connaissance de soi (`self_awareness`) ; Revenir aux repères du présent (`grounded_attention`) ; Trouver un appui dans un environnement choisi (`nature_contact`).
- Protections et adaptations : Réduire l’importance de ce qui touche (`minimization`) ; Se rassurer par des précautions répétées (`safety_behaviors`) ; Créer une pause avant de répondre (`response_pause`).
- Attention et organisation : Distinguer les priorités (`priority_setting`) ; Se rappeler d’une intention au bon moment (`prospective_memory`) ; Se surveiller pendant un échange (`self_focused_attention`).
- Corps, stress et sommeil : Charge de l’environnement sensoriel (`sensory_load`).
- Apprentissages et expérience : Apprendre en observant les autres (`observational_learning`) ; Attendre la répétition d’une expérience (`learned_expectation`).
- Identités et culture : Se comparer dans un environnement numérique (`social_media_comparison`).

Une mention secondaire ne doit pas combler artificiellement ces lacunes. Une perte sans clôture, la mémoire prospective, les séparations et retrouvailles, la charge sensorielle ou les rôles dans la fratrie mériteraient chacun une expérience propre. Leur absence éditoriale ne les fait pas disparaître du catalogue du portrait.

La provenance des anciens textes reste un travail distinct. Dans l’article sur les « 5 blessures de l’âme », le corps associe enfance, comportements et morphologie et rapproche l’approche des théories de l’attachement sans référence primaire attachée à ces affirmations. Les nouveaux récits ne doivent pas reprendre ces associations comme un savoir établi. Les modèles historiques et populaires demandent un statut explicite ; un rappel final de consulter ne valide pas tout le corps.

## Bibliographie institutionnelle de travail

Consultation le 4 novembre 2026 sur les éditeurs directs. « Source primaire » signifie ici provenance institutionnelle directe : certaines ressources sont des guides ou synthèses, pas des études originales. Le rattachement documentaire par famille ne valide ni chaque fiche, ni un personnage, ni un lecteur. Un chiffre, une attribution précise ou une efficacité annoncée exige sa propre source vérifiée.

| ID | Source | Portée et limite |
| --- | --- | --- |
| `cci_assertiveness` | [Centre for Clinical Interventions — Assertiveness](https://www.cci.health.wa.gov.au/Resources/Looking-After-Yourself/Assertiveness) | Expression claire, refus, critique et respect mutuel. |
| `cci_compassion` | [Centre for Clinical Interventions — Self Compassion](https://www.cci.health.wa.gov.au/Resources/Looking-After-Yourself/Self-Compassion) | Autocritique, approche plus attentionnée de soi et pratiques d’apaisement. |
| `cci_distress` | [Centre for Clinical Interventions — Tolerating Distress](https://www.cci.health.wa.gov.au/Resources/Looking-After-Yourself/Tolerating-Distress) | Vécu de l’inconfort émotionnel, acceptation et réponses d’évitement ; pas un score de régulation. |
| `cci_perfectionism` | [Centre for Clinical Interventions — Perfectionism](https://www.cci.health.wa.gov.au/Resources/Looking-After-Yourself/Perfectionism) | Exigences coûteuses, règles rigides, valeur conditionnelle et expériences de changement. |
| `cci_procrastination` | [Centre for Clinical Interventions — Procrastination](https://www.cci.health.wa.gov.au/Resources/Looking-After-Yourself/Procrastination) | Report coûteux, clarification des tâches et passage à l’action ; ne confond pas contrainte et paresse. |
| `ies_learning` | [Institute of Education Sciences, What Works Clearinghouse — Organizing Instruction and Study to Improve Student Learning](https://ies.ed.gov/ncee/wwc/PracticeGuide/1) | Guide institutionnel de recherche sur espacement, exemples et récupération ; transfert éditorial prudent à l’apprentissage professionnel, sans style d’apprentissage fixe. |
| `nhlbi_sleep` | [NHLBI, NIH — How Sleep Works — Your Sleep/Wake Cycle](https://www.nhlbi.nih.gov/health/sleep/sleep-wake-cycle) | Rythmes veille-sommeil et rôle de l’environnement ; pas de protocole médicamenteux ni promesse de sommeil garanti. |
| `nhs_grief` | [NHS — Grief after bereavement or loss](https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/feelings-and-symptoms/grief-bereavement-loss/) | Diversité et non-linéarité des vécus de perte ; aucun calendrier universel ni inférence de deuil pathologique. |
| `nhs_pain` | [NHS — 10 ways to reduce pain](https://www.nhs.uk/live-well/pain/10-ways-to-ease-pain/) | Appuis quotidiens et ajustement d’activité dans le contexte du soin ; aucune douleur n’est attribuée à une émotion, aucune recommandation ne remplace le suivi. |
| `nhs_problem_solving` | [NHS — Problem solving](https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/self-help-cbt-techniques/problem-solving/) | Arbitrages concrets, priorités, solutions et revue de l’expérience. |
| `nhs_relationships` | [NHS — Maintaining healthy relationships and mental wellbeing](https://www.nhs.uk/every-mind-matters/lifes-challenges/maintaining-healthy-relationships-and-mental-wellbeing/) | Communication, respect, limites et soutiens ; pas une typologie de personnalité ou d’attachement. |
| `nhs_sleep` | [NHS — Fall asleep faster and sleep better](https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/how-to-fall-asleep-faster-and-sleep-better/) | Sommeil, routines et environnement ; pas un diagnostic d’insomnie. |
| `nhs_thought_record` | [NHS — Thought record](https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/self-help-cbt-techniques/thought-record/) | Distinguer situation, pensée, émotion, éléments favorables et alternatives. |
| `nih_social` | [National Institutes of Health — Social Wellness Toolkit](https://www.nih.gov/health-information/your-healthiest-self-wellness-toolkits/social-wellness-toolkit) | Liens, aidance, modèles familiaux et habitudes relationnelles ; ne valide pas les libellés familiaux individuellement. |
| `nimh_attention` | [NIMH — Attention — RDoC](https://www.nimh.nih.gov/research/research-funded-by-nimh/rdoc/constructs/attention) | Définition de recherche des fonctions attentionnelles ; aucune déduction de trouble ni méthode clinique validée par ce seul texte. |
| `nimh_selfcare` | [National Institute of Mental Health — Caring for Your Mental Health](https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health) | Repos, activités, priorités et soutien ; symptômes physiques ou persistants à discuter avec un professionnel. |
| `nimh_working_memory` | [NIMH — Working Memory — RDoC](https://www.nimh.nih.gov/research/research-funded-by-nimh/rdoc/constructs/working-memory) | Maintien temporaire, capacité limitée et interférences ; ne prouve pas un trouble ni l’efficacité de chaque aménagement proposé. |
| `who_migration` | [World Health Organization — Refugee and migrant mental health](https://www.who.int/news-room/fact-sheets/detail/refugee-and-migrant-mental-health) | Culture, migration, obstacles matériels et discrimination ; les expériences varient selon les personnes. |
| `who_sexual_health` | [World Health Organization — Defining sexual health](https://www.who.int/teams/sexual-and-reproductive-health-and-research/key-areas-of-work/sexual-health/defining-sexual-health) | Cadre de travail sur le respect, la sécurité et la diversité de l’intimité ; aucune norme de fréquence ou de désir. |
| `who_social` | [World Health Organization — Mental health: strengthening our response](https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response) | Déterminants individuels, sociaux et structurels, protection et accès aux soutiens. |
| `who_stress` | [World Health Organization — Doing What Matters in Times of Stress](https://www.who.int/publications/i/item/9789240003927) | Famille de la présence, de la distance aux pensées, des valeurs et de la bienveillance face au stress. |
| `who_work` | [World Health Organization — Mental health at work](https://www.who.int/news-room/fact-sheets/detail/mental-health-at-work) | Charge, marge de décision, insécurité, harcèlement et aménagements organisationnels. |

Pour le sauveur, CCI et NHS soutiennent les distinctions entre affirmation, écoute, limites et soutien ; ils ne valident pas un diagnostic « syndrome du sauveur ». Les ressources CCI peuvent traiter de troubles, mais leurs critères ne sont pas transformés en dépistage dans ce pack. Les pages NIMH RDoC décrivent des fonctions de recherche et ne valident ni une cause cérébrale du comportement ni l’efficacité de chaque aide pratique proposée. Le guide IES concerne l’enseignement ; son transfert au métier d’Éloi reste une adaptation éditoriale. La santé sexuelle OMS est un cadre de travail, pas une norme de désir.

Pour la douleur et les activités d’Anaïs, l’information générale ne remplace pas le contexte de soin. Les récits ne prescrivent pas un exercice standard ni ne reprennent un jugement sur l’effort. Pour la migration, les facteurs sociaux et communautaires sont considérés sans attribuer à Malik un trauma décrit dans une population plus large.

## Contrat des fichiers et intégration

Les sujets figés prévoient 156 récits : corpus cible de 387 articles et 33 personnages. Entre 1800 et 2500 mots par nouveau corps, soit au moins 280800 mots nouveaux ; cible 1900 à 2200, soit 296400 à 343200 mots. La rédaction est manuelle par ChatGPT, article par article, sans appel à `generate_articles.py`, à une API ou à un assemblage de paragraphes types.

La méthode existante, `generate_articles.py:1463`, demande un personnage récurrent au présent, une situation concrète, des explications simples, des techniques et un retour au récit. La longueur est fixée ligne 1492 et la structure après 1495. Ces instructions sont une charte, pas une garantie : les anciens comptages montrent que la rédaction et le contrôle doivent être dissociés. L’obligation d’un chercheur associé ne doit jamais pousser à inventer une attribution.

Chaque Markdown contient `editorial_pack: psychology-v10`, `episode: 1..12`, `personnage`, `sujet`, `contexte`, `title` de 65 caractères maximum, `description` de 155 caractères maximum, `slug` ASCII de six mots maximum, au moins cinq tags distincts, catégorie et répertoire actuels, `draft: false`. Un sujet de plan peut être reformulé en titre plus court sans changer le thème.

Le corps comporte au moins quatre H2 et trois H3, une scène avant le premier titre, deux ou trois techniques numérotées et une conclusion après `---`. Pas de gras, tiret long ou slash hors liens. Les exercices sont facultatifs et adaptables ; ils ne promettent pas une transformation garantie. Une source fiable doit soutenir les affirmations effectivement utilisées, sans citation scientifique décorative.

Contrôle en lecture seule : `python scripts/check_editorial_pack.py --allow-partial <fichiers>`, puis contrôle agrégé. Les auteurs écrivent leurs récits et un manifeste séparé de lot, avec relations et faits de continuité ; root agrège le registre, la matrice et les données publiques après validation. Les anciens historiques ne sont pas normalisés par ce rapport.

Après intégration, `build_examples.py` reconstruit les extraits et leurs preuves depuis les corps réels. Les 231 anciennes URL restent valides et les nouveaux fichiers, personnages, matrice, EXAMPLES et annuaires doivent concorder. La recette couvre les alias du sauveur, les demandes ordinaires, les négations et les lectures connexes. Elle ne conclut jamais à un vécu à partir d’un intérêt de lecture.

## Annexe : grille des 161 repères

Un exemple lié justifie le rapprochement de thème, sans certifier toutes les explications de l’ancien article. V10 liste les sujets principaux ; si seul un angle secondaire existe, il est signalé entre parenthèses.

| Repère | Initial | Exemple réel ou limite | V10 |
| --- | --- | --- | --- |
| Pensées en boucle (`rumination`) | D | [overthinking (surpensée)](../content/posts/reprendre-le-controle-de-ses-pensees/overthinking-surpensee-travail.md) | Maëlle 5 |
| Intellectualisation (`intellectualization`) | D | [intellectualisation](../content/posts/sortir-de-ses-schemas-repetitifs/intellectualisation-au-travail-hpi-solutions.md) | — |
| Besoin de certitude (`uncertainty`) | C | [inquiétude](../content/posts/comprendre-et-maitriser-ses-emotions/inquietude-decision-importante.md) | Aline 11, Maëlle 2, Vincent 8, Odile 6, Odile 10, Anaïs 2, Baptiste 2, Baptiste 9, Élise 6, Élise 10 |
| Anticipation du pire (`catastrophizing`) | D | [catastrophisation (dramatisation)](../content/posts/reprendre-le-controle-de-ses-pensees/catastrophisation-avec-enfants.md) | Odile 10 |
| Interpréter le regard des autres (`mind_reading`) | D | [lecture de pensée](../content/posts/reprendre-le-controle-de-ses-pensees/lecture-de-pensee-en-amitie-psychologie.md) | — |
| Pensée tout ou rien (`all_or_nothing`) | D | [pensée tout-ou-rien (pensée dichotomique)](../content/posts/reprendre-le-controle-de-ses-pensees/pensee-tout-ou-rien-conflit.md) | — |
| Attention aux indices qui confirment (`confirmation`) | D | [biais de confirmation](../content/posts/reprendre-le-controle-de-ses-pensees/biais-confirmation-conflit.md) | — |
| Capacité à changer de perspective (`reframing`) | C | [journaling de pensées](../content/posts/reprendre-le-controle-de-ses-pensees/journaling-pensees-intimite-sexualite-nadia.md) | Élise 9 |
| Identifier ses émotions (`emotional_identification`) | D | [alexithymie](../content/posts/comprendre-et-maitriser-ses-emotions/alexithymie-en-couple.md) | Vincent 4, Malik 3, Élise 2 |
| Retenir l’expression émotionnelle (`emotional_suppression`) | D | [suppression émotionnelle vs réévaluation cognitive](../content/posts/comprendre-et-maitriser-ses-emotions/suppression-emotionnelle-reevaluation-cognitive-couple.md) | — |
| Se sentir débordé (`emotional_overload`) | D | [accablement](../content/posts/comprendre-et-maitriser-ses-emotions/accablement-en-periode-de-reconversion-professionnelle.md) | Aline 10 |
| Colère et besoins contrariés (`anger_needs`) | D | [colère](../content/posts/comprendre-et-maitriser-ses-emotions/colere-en-situation-de-conflit-comprendre-s-affirmer.md) | Samira 6, Vincent 10 |
| Culpabilité (`guilt`) | D | [culpabilité](../content/posts/comprendre-et-maitriser-ses-emotions/culpabilite-avec-ses-enfants-comprendre-agir.md) | Aline 3, Maëlle 3, Vincent 5, Vincent 9, Odile 11, Anaïs 3 |
| Honte et exposition (`shame`) | D | [embarras](../content/posts/comprendre-et-maitriser-ses-emotions/embarras-en-societe-interactions.md) | Samira 3 |
| Signaux corporels du stress (`body_stress`) | C | [cohérence cardiaque](../content/posts/comprendre-et-maitriser-ses-emotions/coherence-cardiaque-au-travail-nadia.md) | Maëlle 7 |
| Autocritique (`self_criticism`) | C | [exigences élevées/critique excessive](../content/posts/sortir-de-ses-schemas-repetitifs/exigences-elevees-critique-excessive-au-travail.md) | Maëlle 4, Inès 5, Salomé 7, Élise 8 |
| Valeur liée à la réussite (`conditional_worth`) | C | [le syndrome de l'imposteur](../content/posts/sortir-de-ses-schemas-repetitifs/syndrome-imposteur-amitie-causes.md) | Samira 9 |
| Sentiment d’imposture (`impostor`) | D | [sentiment d'imposture](../content/posts/comprendre-et-maitriser-ses-emotions/sentiment-imposture-apres-un-succes.md) | — |
| Comparaison sociale (`comparison`) | D | [comparaison injuste](../content/posts/reprendre-le-controle-de-ses-pensees/comparaison-injuste-competition.md) | Éloi 4, Salomé 9 |
| Bienveillance envers soi (`self_compassion`) | D | [auto-compassion (Kristin Neff)](../content/posts/comprendre-et-maitriser-ses-emotions/auto-compassion-kristin-neff-en-famille.md) | Aline 3, Rémi 10, Éloi 3, Vincent 5, Salomé 7, Élise 8 |
| Identité en évolution (`identity_transition`) | C | [croyances limitantes](../content/posts/sortir-de-ses-schemas-repetitifs/croyances-limitantes-reconversion-professionnelle-methode.md) | Salomé 11, Malik 12, Anaïs 9 |
| Rôle de sauveur (`overresponsibility`) | C | [abnégation/sacrifice de soi](../content/posts/sortir-de-ses-schemas-repetitifs/abnegation-et-sacrifice-de-soi-en-amitie.md) | Aline 1, Aline 2 |
| Exprimer ses limites (`boundaries`) | C | [abnégation/sacrifice de soi](../content/posts/sortir-de-ses-schemas-repetitifs/abnegation-et-sacrifice-de-soi-en-amitie.md) | Aline 1, Aline 4, Aline 6, Aline 8, Maëlle 3, Maëlle 12, Inès 3, Inès 10, Vincent 11, Salomé 1, Salomé 4, Salomé 5, Salomé 8, Salomé 10, Odile 4, Anaïs 7 |
| Recherche de validation (`reassurance`) | D | [recherche d'approbation/reconnaissance](../content/posts/sortir-de-ses-schemas-repetitifs/recherche-approbation-parents.md) | — |
| Rapport au désaccord (`conflict`) | C | [les jeux psychologiques (triangle de Karpman : persécuteur, victime, sauveur)](../content/posts/sortir-de-ses-schemas-repetitifs/jeux-psychologiques-karpman-couple.md) | Aline 9, Inès 9, Odile 8 |
| Réciprocité (`reciprocity`) | C | [la codépendance](../content/posts/sortir-de-ses-schemas-repetitifs/causes-profondes-codependance-en-amitie.md) | Aline 1, Aline 5, Aline 8, Samira 11, Vincent 7, Salomé 5, Salomé 12, Malik 11, Élise 5 |
| Confiance et prudence (`trust`) | D | [trahison](../content/posts/comprendre-et-maitriser-ses-emotions/trahison-en-amitie-impacts-quotidiens.md) | Aline 2, Inès 4, Inès 11 |
| Attentes parentales intériorisées (`parent_expectations`) | C | [exigences élevées/critique excessive](../content/posts/sortir-de-ses-schemas-repetitifs/exigences-elevees-critique-excessive-au-travail.md) | Aline 9 |
| Place et rôles dans la famille (`family_roles`) | D | [l'adultification](../content/posts/sortir-de-ses-schemas-repetitifs/adultification-avec-ses-parents-sortir-schema.md) | — |
| Responsabilités précoces (`parentification`) | D | [la parentification](../content/posts/sortir-de-ses-schemas-repetitifs/parentification-avec-ses-parents-causes-solutions.md) | — |
| Loyauté et autonomie (`family_loyalty`) | D | [loyautés invisibles (Boszormenyi-Nagy)](../content/posts/sortir-de-ses-schemas-repetitifs/loyautes-invisibles-boszormenyi-nagy-fetes-famille.md) | Aline 9, Malik 10 |
| Apprentissages émotionnels familiaux (`emotional_learning`) | C | [refoulement](../content/posts/sortir-de-ses-schemas-repetitifs/refoulement-avec-ses-parents-causes.md) | Aline 11 |
| Frontières dans la famille (`family_boundaries`) | C | [la parentification](../content/posts/sortir-de-ses-schemas-repetitifs/parentification-avec-ses-parents-causes-solutions.md) | Aline 9, Aline 11, Malik 4, Malik 10 |
| Peur de perdre le lien (`abandonment_fear`) | D | [abandon émotionnel](../content/posts/comprendre-et-maitriser-ses-emotions/abandon-emotionnel-parents.md) | — |
| Proximité et distance (`closeness_distance`) | D | [attachement désorganisé/craintif (oscillation)](../content/posts/sortir-de-ses-schemas-repetitifs/attachement-desorganise-craintif-oscillation-rupture-amoureuse.md) | Malik 4 |
| Sécurité relationnelle (`relational_security`) | D | [attachement sécure (caractéristiques, comment le développer)](../content/posts/sortir-de-ses-schemas-repetitifs/attachement-secure-enfants-caracteristiques-developper.md) | Inès 7 |
| Perfectionnisme coûteux (`perfectionism`) | D | [exigences élevées/critique excessive](../content/posts/sortir-de-ses-schemas-repetitifs/exigences-elevees-critique-excessive-au-travail.md) | Rémi 9 |
| Évitement et report (`avoidance`) | C | [biais de l'autruche](../content/posts/reprendre-le-controle-de-ses-pensees/biais-autruche-deuil-comprendre-evitement.md) | Rémi 8, Élise 3 |
| Besoin de contrôle (`overcontrol`) | C | [illusion de contrôle](../content/posts/reprendre-le-controle-de-ses-pensees/illusion-controle-manager.md) | Aline 2, Vincent 3, Baptiste 9 |
| Retrait relationnel (`withdrawal`) | C | [isolement social/aliénation](../content/posts/sortir-de-ses-schemas-repetitifs/isolement-social-alienation-amitie.md) | Rémi 10 |
| Réagir dans l’urgence (`impulsive_reaction`) | D | [acting out (passage à l'acte)](../content/posts/sortir-de-ses-schemas-repetitifs/acting-out-passage-a-l-acte-famille.md) | — |
| Surinvestissement et récupération (`overwork`) | C | [fatigue de compassion](../content/posts/comprendre-et-maitriser-ses-emotions/fatigue-compassion-face-maladie.md) | Maëlle 8 |
| Perte et changement (`grief_change`) | D | [tristesse](../content/posts/comprendre-et-maitriser-ses-emotions/comprendre-tristesse-lors-d-un-deuil.md) | Odile 1, Odile 2, Odile 6 |
| Contraintes et ressources actuelles (`stress_resources`) | C | [charge mentale émotionnelle](../content/posts/comprendre-et-maitriser-ses-emotions/charge-mentale-emotionnelle-au-travail-solutions.md) | Aline 10, Maëlle 1, Maëlle 7, Samira 1, Vincent 12, Malik 7, Baptiste 11 |
| Appartenance et place (`social_belonging`) | C | [inadéquation](../content/posts/comprendre-et-maitriser-ses-emotions/sentiment-inadequation-ecole-etudes-hpi.md) | Malik 1, Malik 11 |
| Autonomie (`autonomy`) | C | [assujettissement](../content/posts/sortir-de-ses-schemas-repetitifs/assujettissement-face-a-l-autorite-se-liberer.md) | Aline 1, Aline 4, Aline 7, Aline 8, Éloi 5, Éloi 12, Inès 1, Inès 4, Inès 11, Inès 12, Vincent 2, Vincent 3, Vincent 6, Vincent 7, Vincent 9, Salomé 2, Salomé 5, Salomé 12, Malik 10, Odile 3, Odile 4, Odile 5, Odile 7, Odile 8, Anaïs 6, Baptiste 3, Élise 12 |
| Valeurs et cohérence (`values_alignment`) | C | [croyances limitantes](../content/posts/sortir-de-ses-schemas-repetitifs/croyances-limitantes-reconversion-professionnelle-methode.md) | Aline 12, Samira 10, Éloi 9, Inès 12, Vincent 9, Salomé 11, Malik 12, Odile 9, Anaïs 9, Anaïs 10, Baptiste 1, Baptiste 7, Baptiste 12, Élise 11 |
| Reconnaître ses besoins (`needs_expression`) | C | [abnégation/sacrifice de soi](../content/posts/sortir-de-ses-schemas-repetitifs/abnegation-et-sacrifice-de-soi-en-amitie.md) | Aline 5, Maëlle 3, Maëlle 9, Samira 6, Samira 9, Samira 12, Éloi 5, Inès 6, Vincent 1, Vincent 4, Vincent 5, Malik 2, Malik 3, Anaïs 6 |
| Demander du soutien (`support_seeking`) | C | [fatigue de compassion](../content/posts/comprendre-et-maitriser-ses-emotions/fatigue-compassion-face-maladie.md) | Aline 10, Maëlle 8, Samira 2, Samira 4, Éloi 7, Éloi 12, Vincent 1, Vincent 6, Vincent 11, Vincent 12, Malik 5, Malik 6, Malik 11, Odile 7, Anaïs 12, Élise 7 |
| Souplesse émotionnelle (`emotional_flexibility`) | D | [suppression émotionnelle vs réévaluation cognitive](../content/posts/comprendre-et-maitriser-ses-emotions/suppression-emotionnelle-reevaluation-cognitive-couple.md) | Élise 11 |
| Appuis dans les épreuves (`resilience`) | D | [résilience émotionnelle](../content/posts/comprendre-et-maitriser-ses-emotions/resilience-emotionnelle-lors-un-deuil-comprendre.md) | Anaïs 12 |
| Liens nourrissants (`positive_connection`) | C | [tendresse](../content/posts/comprendre-et-maitriser-ses-emotions/tendresse-avec-ses-enfants-bernard.md) | Maëlle 6, Anaïs 12 |
| Connaissance de soi (`self_awareness`) | C | [journaling de pensées](../content/posts/reprendre-le-controle-de-ses-pensees/journaling-pensees-intimite-sexualite-nadia.md) | Maëlle 9, Élise 9 |
| Tout ramener à sa responsabilité (`personalization`) | D | [personnalisation](../content/posts/reprendre-le-controle-de-ses-pensees/personnalisation-reseaux-sociaux.md) | — |
| Retenir surtout le négatif (`negative_filter`) | D | [biais de négativité](../content/posts/reprendre-le-controle-de-ses-pensees/biais-negativite-face-decision-importante-camille.md) | — |
| Étendre une expérience à toutes les situations (`overgeneralization`) | D | [surgénéralisation](../content/posts/reprendre-le-controle-de-ses-pensees/surgeneralisation-ecole-lycee.md) | Éloi 3 |
| Prendre un ressenti pour une preuve (`emotional_reasoning`) | D | [raisonnement émotionnel](../content/posts/reprendre-le-controle-de-ses-pensees/raisonnement-emotionnel-en-couple.md) | — |
| Règles intérieures rigides (`rigid_rules`) | D | [les "je dois/il faut" (obligations rigides)](../content/posts/reprendre-le-controle-de-ses-pensees/gerer-obligations-rigides-je-dois-il-faut-conflit.md) | Aline 3 |
| Résoudre un problème concret (`problem_solving`) | C | [accablement](../content/posts/comprendre-et-maitriser-ses-emotions/accablement-en-periode-de-reconversion-professionnelle.md) | Maëlle 5, Odile 10, Anaïs 2, Baptiste 2, Baptiste 9, Élise 1, Élise 10 |
| Arbitrer entre plusieurs priorités (`decision_tradeoffs`) | C | [biais de cadrage](../content/posts/reprendre-le-controle-de-ses-pensees/biais-de-cadrage-decision-importante-nadia.md) | Aline 12, Vincent 2, Vincent 9, Odile 5, Baptiste 1, Baptiste 3, Baptiste 6, Élise 12 |
| Saturation devant les choix (`decision_overload`) | C | [biais de cadrage](../content/posts/reprendre-le-controle-de-ses-pensees/biais-de-cadrage-decision-importante-nadia.md) | Samira 8 |
| Rester pour ce qui a déjà coûté (`sunk_cost`) | D | [biais du coût irrécupérable](../content/posts/reprendre-le-controle-de-ses-pensees/biais-cout-irrecoverable-rupture.md) | Baptiste 12 |
| Observer sa manière de penser (`metacognitive_check`) | C | [biais de l'angle mort](../content/posts/reprendre-le-controle-de-ses-pensees/biais-angle-mort-vieillissement-psychologie.md) | Maëlle 2, Maëlle 5, Anaïs 2, Anaïs 5, Baptiste 5 |
| Ressentir plusieurs émotions à la fois (`emotional_ambivalence`) | D | [ambivalence émotionnelle](../content/posts/comprendre-et-maitriser-ses-emotions/ambivalence-emotionnelle-reconversion-professionnelle-burn-out.md) | Vincent 4, Malik 9, Odile 2, Odile 6, Odile 11 |
| Traverser un inconfort émotionnel (`distress_tolerance`) | C | [technique RAIN (Reconnaître-Accepter-Investiguer-Ne pas s'identifier)](../content/posts/comprendre-et-maitriser-ses-emotions/technique-rain-seul-face-a-soi-meme.md) | Aline 6, Malik 9 |
| S’autoriser la joie (`joy_permission`) | C | [euphorie](../content/posts/comprendre-et-maitriser-ses-emotions/euphorie-apres-un-succes-psychologie.md) | Vincent 4, Odile 11 |
| Accueillir une déception (`disappointment`) | D | [déception](../content/posts/comprendre-et-maitriser-ses-emotions/deception-en-amitie-comprendre-attentes-relations.md) | Aline 6 |
| S’appuyer sur son propre jugement (`self_trust`) | C | [effet Dunning-Kruger](../content/posts/reprendre-le-controle-de-ses-pensees/effet-dunning-kruger-decision-importante.md) | Inès 10, Salomé 3 |
| Faire un bilan de soi nuancé (`balanced_self_evaluation`) | C | [méthode ABCDE d'Ellis](../content/posts/reprendre-le-controle-de-ses-pensees/methode-abcde-ellis-seule-face-a-soi-meme.md) | Aline 5, Maëlle 4, Salomé 7, Salomé 9, Baptiste 5 |
| Accorder son expression à ce que l’on vit (`authenticity`) | C | [le faux self (Winnicott)](../content/posts/sortir-de-ses-schemas-repetitifs/faux-self-winnicott-parents.md) | Salomé 6 |
| Vérifier le sens d’un échange (`communication_clarification`) | C | [lecture de pensée](../content/posts/reprendre-le-controle-de-ses-pensees/lecture-de-pensee-en-amitie-psychologie.md) | Aline 4, Aline 7, Maëlle 6, Maëlle 10, Inès 2, Inès 9, Vincent 2, Vincent 7, Salomé 9, Anaïs 5, Baptiste 6 |
| Écouter sans résoudre tout de suite (`active_listening`) | C | [empathie cognitive vs affective](../content/posts/comprendre-et-maitriser-ses-emotions/empathie-cognitive-vs-affective-avec-ses-parents.md) | Aline 7, Aline 11, Malik 3, Malik 4 |
| Faire une demande explicite (`assertive_request`) | C | [abnégation/sacrifice de soi](../content/posts/sortir-de-ses-schemas-repetitifs/abnegation-et-sacrifice-de-soi-en-amitie.md) | Aline 5, Maëlle 7, Maëlle 12, Inès 6, Inès 7, Vincent 10, Vincent 11, Odile 7, Anaïs 6, Anaïs 11 |
| Réparer après un désaccord (`repair_after_conflict`) | C | [remords](../content/posts/comprendre-et-maitriser-ses-emotions/remords-en-famille-comprendre-et-agir.md) | Aline 4, Inès 9, Vincent 3, Vincent 10 |
| Recevoir sans devoir rembourser immédiatement (`receiving_help`) | C | [fatigue de compassion](../content/posts/comprendre-et-maitriser-ses-emotions/fatigue-compassion-face-maladie.md) | Aline 8, Maëlle 4, Maëlle 8, Samira 11, Salomé 12, Anaïs 3, Élise 5 |
| Marge de choix dans un rapport de pouvoir (`power_imbalance`) | C | [assujettissement](../content/posts/sortir-de-ses-schemas-repetitifs/assujettissement-face-a-l-autorite-se-liberer.md) | Maëlle 7, Samira 6, Salomé 4, Malik 6 |
| Messages familiaux devenus des règles (`inherited_rules`) | D | [les drivers/messages contraignants (sois parfait, fais plaisir, sois fort, dépêche-toi, fais des efforts)](../content/posts/sortir-de-ses-schemas-repetitifs/drivers-contraignants-reconversion-professionnelle.md) | — |
| Se retrouver médiateur familial (`family_mediation`) | C | [la parentification](../content/posts/sortir-de-ses-schemas-repetitifs/parentification-avec-ses-parents-causes-solutions.md) | Aline 9 |
| Comparaisons dans la fratrie (`sibling_comparison`) | — | Aucun sujet ou voisin retenu | Samira 11 |
| Sujets familiaux difficiles à aborder (`family_secrecy`) | C | [transmissions transgénérationnelles](../content/posts/sortir-de-ses-schemas-repetitifs/transmissions-transgenerationnelles-avec-ses-enfants-comprendre.md) | Aline 11 |
| Prévisibilité de la disponibilité (`consistency_in_relationships`) | C | [attachement anxieux-préoccupé (hyperactivation du système)](../content/posts/sortir-de-ses-schemas-repetitifs/attachement-anxieux-preoccupe-couple.md) | Maëlle 6, Vincent 11 |
| Vivre les séparations et retrouvailles (`separation_reunion`) | C | [abandon émotionnel](../content/posts/comprendre-et-maitriser-ses-emotions/abandon-emotionnel-parents.md) | Inès 11 |
| S’apaiser dans une présence sûre (`co_regulation`) | C | [attachement sécure (caractéristiques, comment le développer)](../content/posts/sortir-de-ses-schemas-repetitifs/attachement-secure-enfants-caracteristiques-developper.md) | Inès 7 |
| Partager une vulnérabilité à son rythme (`safe_disclosure`) | C | [vulnérabilité](../content/posts/comprendre-et-maitriser-ses-emotions/vulnerabilite-en-couple-signes.md) | Inès 8, Vincent 4, Vincent 11 |
| Passer du projet au premier geste (`task_initiation`) | C | [accablement](../content/posts/comprendre-et-maitriser-ses-emotions/accablement-en-periode-de-reconversion-professionnelle.md) | Rémi 3, Rémi 8, Élise 3, Élise 12 |
| Repérer les déclencheurs d’une habitude (`habit_cues`) | — | Aucun sujet ou voisin retenu | Rémi 12, Maëlle 11, Élise 6 |
| Composer avec une récompense lointaine (`reward_delay`) | — | Aucun sujet ou voisin retenu | Baptiste 8 |
| Vouloir changer et vouloir conserver (`motivation_ambivalence`) | — | Aucun sujet ou voisin retenu | Baptiste 10, Élise 12 |
| Adapter ses exigences à la situation (`flexible_standards`) | C | [exigences élevées/critique excessive](../content/posts/sortir-de-ses-schemas-repetitifs/exigences-elevees-critique-excessive-au-travail.md) | Rémi 9, Maëlle 11, Anaïs 4, Élise 1 |
| Pression financière (`financial_strain`) | C | [culpabilité](../content/posts/comprendre-et-maitriser-ses-emotions/culpabilite-face-a-l-argent-et-finances.md) | Samira 1, Samira 3, Samira 9, Samira 12 |
| Instabilité du lieu de vie (`housing_instability`) | — | Aucun sujet ou voisin retenu | Samira 2 |
| Charge et organisation du travail (`workload_context`) | C | [fatigue de compassion](../content/posts/comprendre-et-maitriser-ses-emotions/fatigue-compassion-face-maladie.md) | Aline 10, Rémi 1, Rémi 7, Maëlle 1, Maëlle 7, Maëlle 10, Maëlle 12, Anaïs 11, Baptiste 7 |
| Expérience déclarée de discrimination (`discrimination_context`) | — | Aucun sujet ou voisin retenu | Malik 6 |
| Charge d’aidance (`caregiver_load`) | C | [fatigue de compassion](../content/posts/comprendre-et-maitriser-ses-emotions/fatigue-compassion-face-maladie.md) | Samira 5, Vincent 1, Vincent 2, Vincent 8 |
| S’adapter à un changement de santé déclaré (`illness_adjustment`) | C | [vulnérabilité au danger](../content/posts/sortir-de-ses-schemas-repetitifs/vulnerabilite-au-danger-face-a-la-maladie.md) | Anaïs 1, Anaïs 4, Anaïs 8, Anaïs 12 |
| Obstacles pour accéder à une aide (`access_barriers`) | — | Aucun sujet ou voisin retenu | Samira 4, Samira 7, Samira 12, Malik 8, Odile 3 |
| Changer de pays ou de milieu de vie (`migration_adjustment`) | C | [apathie](../content/posts/comprendre-et-maitriser-ses-emotions/apathie-en-voyage-ou-expatriation-comprendre.md) | Malik 7 |
| Passer d’un rôle de vie à un autre (`role_transition`) | C | [croyances limitantes](../content/posts/sortir-de-ses-schemas-repetitifs/croyances-limitantes-reconversion-professionnelle-methode.md) | Odile 4 |
| Perte sans clôture claire (`ambiguous_loss`) | — | Aucun sujet ou voisin retenu | Malik 9 |
| Garder une place au lien après une perte (`continuing_bonds`) | C | [mémoire émotionnelle](../content/posts/comprendre-et-maitriser-ses-emotions/memoire-emotionnelle-lors-dun-deuil.md) | Odile 1, Odile 5, Odile 12 |
| Isolement subi (`social_isolation`) | D | [isolement social/aliénation](../content/posts/sortir-de-ses-schemas-repetitifs/isolement-social-alienation-amitie.md) | Malik 5 |
| Demandes incompatibles entre plusieurs rôles (`work_home_conflict`) | C | [culpabilité](../content/posts/comprendre-et-maitriser-ses-emotions/culpabilite-avec-ses-enfants-comprendre-agir.md) | Maëlle 6, Baptiste 1 |
| Valeurs qui entrent en tension (`values_conflict`) | — | Aucun sujet ou voisin retenu | Aline 12, Samira 8, Éloi 9, Vincent 9, Malik 10, Baptiste 3, Baptiste 6 |
| Chercher du sens (`meaning_search`) | C | [vide intérieur](../content/posts/comprendre-et-maitriser-ses-emotions/vide-interieur-apres-rupture-amoureuse.md) | Odile 9, Baptiste 7 |
| Choisir une direction personnelle (`purpose_direction`) | C | [enthousiasme](../content/posts/comprendre-et-maitriser-ses-emotions/enthousiasme-reconversion-professionnelle-burn-out.md) | Aline 12, Éloi 9, Baptiste 10 |
| Découper un projet en étapes accessibles (`small_goals`) | C | [accablement](../content/posts/comprendre-et-maitriser-ses-emotions/accablement-en-periode-de-reconversion-professionnelle.md) | Aline 12, Rémi 4, Samira 10, Anaïs 4, Baptiste 10, Baptiste 11 |
| Recevoir un appui concret (`practical_support`) | C | [fatigue de compassion](../content/posts/comprendre-et-maitriser-ses-emotions/fatigue-compassion-face-maladie.md) | Aline 10, Maëlle 8, Samira 2, Samira 5, Samira 11, Vincent 1, Vincent 6, Vincent 12, Malik 11, Anaïs 1, Anaïs 6, Anaïs 11, Élise 7 |
| Trouver une place dans un collectif (`community_participation`) | — | Aucun sujet ou voisin retenu | Aline 12, Maëlle 12, Odile 8, Anaïs 9, Élise 7 |
| Revenir aux repères du présent (`grounded_attention`) | C | [technique RAIN (Reconnaître-Accepter-Investiguer-Ne pas s'identifier)](../content/posts/comprendre-et-maitriser-ses-emotions/technique-rain-seul-face-a-soi-meme.md) | Élise 4, Élise 9 |
| Préserver des activités agréables (`enjoyable_activity`) | C | [sublimation](../content/posts/sortir-de-ses-schemas-repetitifs/sublimation-reconversion-professionnelle.md) | Odile 9, Anaïs 10, Élise 4 |
| Reconnaître ce qui soutient (`gratitude`) | D | [gratitude](../content/posts/comprendre-et-maitriser-ses-emotions/gratitude-face-a-l-argent-et-aux-finances.md) | Élise 5 |
| Exprimer par la création (`creative_expression`) | D | [écriture expressive (Pennebaker)](../content/posts/comprendre-et-maitriser-ses-emotions/ecriture-expressive-pennebaker-en-famille-guide.md) | Odile 12 |
| Trouver un appui dans un environnement choisi (`nature_contact`) | C | [émerveillement](../content/posts/comprendre-et-maitriser-ses-emotions/emerveillement-en-voyage-ou-en-expatriation-guide.md) | Élise 4 |
| Choisir un temps seul qui ressource (`solitude_choice`) | C | [solitude](../content/posts/comprendre-et-maitriser-ses-emotions/solitude-face-a-la-solitude-comprendre-difference.md) | Malik 5, Odile 3 |
| Reconnaître sa capacité d’agir (`self_efficacy`) | C | [impuissance](../content/posts/comprendre-et-maitriser-ses-emotions/impuissance-deuil-reprendre-main.md) | Samira 7, Éloi 2, Éloi 6, Éloi 11, Élise 10 |
| Réduire l’importance de ce qui touche (`minimization`) | C | [déni](../content/posts/sortir-de-ses-schemas-repetitifs/deni-apres-un-echec-comprendre-et-agir.md) | Élise 2 |
| Construire une explication après coup (`rationalization`) | D | [rationalisation](../content/posts/sortir-de-ses-schemas-repetitifs/rationalisation-avec-ses-parents.md) | — |
| Se distraire pour souffler (`distraction_coping`) | — | Aucun sujet ou voisin retenu | Élise 3 |
| Utiliser l’humour pour traverser (`humor_coping`) | D | [humour comme défense](../content/posts/sortir-de-ses-schemas-repetitifs/humour-defense-conflit-hugo.md) | Élise 2 |
| Se sentir momentanément à distance (`emotional_detachment`) | D | [isolation de l'affect](../content/posts/sortir-de-ses-schemas-repetitifs/isolation-affect-conflit-comprendre-mecanisme.md) | — |
| Se rassurer par des précautions répétées (`safety_behaviors`) | C | [attachement anxieux-préoccupé (hyperactivation du système)](../content/posts/sortir-de-ses-schemas-repetitifs/attachement-anxieux-preoccupe-couple.md) | Élise 6 |
| Apaiser autrui pour préserver sa sécurité (`appeasement`) | C | [assujettissement](../content/posts/sortir-de-ses-schemas-repetitifs/assujettissement-face-a-l-autorite-se-liberer.md) | Salomé 4 |
| Choisir une réponse selon la situation (`flexible_coping`) | C | [résilience émotionnelle](../content/posts/comprendre-et-maitriser-ses-emotions/resilience-emotionnelle-lors-un-deuil-comprendre.md) | Samira 8, Vincent 10, Malik 2, Anaïs 8, Baptiste 4, Baptiste 12, Élise 1, Élise 11 |
| Créer une pause avant de répondre (`response_pause`) | C | [acting out (passage à l'acte)](../content/posts/sortir-de-ses-schemas-repetitifs/acting-out-passage-a-l-acte-famille.md) | Vincent 3, Vincent 10, Élise 8 |
| Reconnaître une responsabilité proportionnée (`responsibility_repair`) | C | [remords](../content/posts/comprendre-et-maitriser-ses-emotions/remords-en-famille-comprendre-et-agir.md) | Aline 6, Maëlle 5, Baptiste 5, Élise 8 |
| Attention facilement détournée (`distractibility`) | C | [effet Zeigarnik](../content/posts/reprendre-le-controle-de-ses-pensees/effet-zeigarnik-manager-perfectionniste.md) | Rémi 5, Maëlle 10 |
| Attention captée par une sollicitation (`attentional_capture`) | C | [effet spotlight](../content/posts/reprendre-le-controle-de-ses-pensees/effet-spotlight-ecole-etudes.md) | Rémi 5 |
| Coût des changements de tâche (`attention_switching`) | — | Aucun sujet ou voisin retenu | Rémi 1 |
| Rendre les étapes visibles (`planning_steps`) | C | [effet Zeigarnik](../content/posts/reprendre-le-controle-de-ses-pensees/effet-zeigarnik-manager-perfectionniste.md) | Aline 2, Rémi 3, Rémi 4, Rémi 6, Maëlle 11, Samira 1, Samira 4, Samira 5, Samira 7, Vincent 6, Vincent 8, Vincent 12, Malik 8, Anaïs 4, Anaïs 8, Baptiste 11 |
| Estimer le temps nécessaire (`time_estimation`) | — | Aucun sujet ou voisin retenu | Rémi 6 |
| Distinguer les priorités (`priority_setting`) | C | [charge mentale émotionnelle](../content/posts/comprendre-et-maitriser-ses-emotions/charge-mentale-emotionnelle-au-travail-solutions.md) | Rémi 7, Anaïs 8, Baptiste 8 |
| Trop d’informations à garder en tête (`working_memory_load`) | C | [effet Zeigarnik](../content/posts/reprendre-le-controle-de-ses-pensees/effet-zeigarnik-manager-perfectionniste.md) | Rémi 2, Rémi 11, Vincent 1, Vincent 8 |
| S’appuyer sur des repères extérieurs (`external_supports`) | C | [journaling de pensées](../content/posts/reprendre-le-controle-de-ses-pensees/journaling-pensees-intimite-sexualite-nadia.md) | Rémi 2, Rémi 11, Rémi 12, Maëlle 1, Maëlle 8, Vincent 8, Malik 7, Malik 8, Baptiste 8 |
| Se rappeler d’une intention au bon moment (`prospective_memory`) | — | Aucun sujet ou voisin retenu | Rémi 11, Vincent 8 |
| Reconnaître ses sensations corporelles (`body_awareness`) | C | [cohérence cardiaque](../content/posts/comprendre-et-maitriser-ses-emotions/coherence-cardiaque-au-travail-nadia.md) | Maëlle 9, Inès 5, Anaïs 5 |
| Charge de l’environnement sensoriel (`sensory_load`) | — | Aucun sujet ou voisin retenu | Maëlle 10 |
| Corps mobilisé avant un événement (`anticipatory_arousal`) | C | [anxiété](../content/posts/comprendre-et-maitriser-ses-emotions/anxiete-au-travail-signes-discrets.md) | Maëlle 2 |
| Difficulté à décrocher avant le sommeil (`sleep_onset`) | C | [rumination mentale](../content/posts/reprendre-le-controle-de-ses-pensees/rumination-mentale-seul-face-a-soi-meme-adrien.md) | Maëlle 2 |
| Rythmes et possibilités de sommeil (`sleep_rhythm`) | — | Aucun sujet ou voisin retenu | Maëlle 1, Maëlle 11 |
| Ajuster l’effort à l’énergie disponible (`fatigue_pacing`) | C | [fatigue de compassion](../content/posts/comprendre-et-maitriser-ses-emotions/fatigue-compassion-face-maladie.md) | Maëlle 4, Maëlle 9, Inès 6, Vincent 5, Vincent 12, Anaïs 1 |
| S’accorder du repos (`rest_permission`) | C | [abnégation/sacrifice de soi](../content/posts/sortir-de-ses-schemas-repetitifs/abnegation-et-sacrifice-de-soi-en-amitie.md) | Aline 3, Maëlle 3, Vincent 5, Vincent 12, Anaïs 3 |
| Santé ou fonctionnement explicitement déclaré (`health_declared`) | C | [somatisation](../content/posts/sortir-de-ses-schemas-repetitifs/somatisation-face-au-vieillissement-corps-parle.md) | Anaïs 1, Anaïs 2, Anaïs 3, Anaïs 5, Anaïs 7, Anaïs 10 |
| Trouver un mouvement qui convient (`movement_regulation`) | C | [sublimation](../content/posts/sortir-de-ses-schemas-repetitifs/sublimation-reconversion-professionnelle.md) | Anaïs 10 |
| Apprendre en observant les autres (`observational_learning`) | — | Aucun sujet ou voisin retenu | Éloi 2, Éloi 4 |
| Attendre la répétition d’une expérience (`learned_expectation`) | C | [prophéties autoréalisatrices](../content/posts/sortir-de-ses-schemas-repetitifs/propheties-autorealizatrices-face-decision-importante.md) | Éloi 10 |
| Tester une hypothèse par un petit essai (`behavioral_experiment`) | C | [méthode ABCDE d'Ellis](../content/posts/reprendre-le-controle-de-ses-pensees/methode-abcde-ellis-seule-face-a-soi-meme.md) | Samira 10, Éloi 8, Éloi 10, Baptiste 4 |
| Utiliser un retour sans s’y réduire (`feedback_learning`) | C | [effet Dunning-Kruger](../content/posts/reprendre-le-controle-de-ses-pensees/effet-dunning-kruger-decision-importante.md) | Éloi 1, Éloi 10, Baptiste 2, Baptiste 4 |
| Développer une compétence avec des appuis (`skill_practice`) | C | [effet Dunning-Kruger](../content/posts/reprendre-le-controle-de-ses-pensees/effet-dunning-kruger-decision-importante.md) | Éloi 1, Éloi 2, Éloi 6, Éloi 7, Éloi 8, Éloi 11, Éloi 12 |
| Exprimer et respecter un consentement libre (`consent_boundaries`) | — | Aucun sujet ou voisin retenu | Aline 7, Inès 1, Inès 3, Inès 10, Inès 12, Vincent 3, Vincent 6, Vincent 10 |
| Définir sa vie privée (`privacy_boundaries`) | C | [journaling de pensées](../content/posts/reprendre-le-controle-de-ses-pensees/journaling-pensees-intimite-sexualite-nadia.md) | Aline 11, Inès 4, Inès 11, Vincent 11, Salomé 2, Salomé 8, Salomé 10, Odile 12, Anaïs 7, Anaïs 11 |
| Accueillir des rythmes de désir différents (`desire_variation`) | C | [attachement évitant-dédaigneux (désactivation du système)](../content/posts/sortir-de-ses-schemas-repetitifs/attachement-evitant-dans-intimite.md) | Inès 2 |
| Comprendre un ressenti de jalousie (`relational_jealousy`) | D | [jalousie](../content/posts/comprendre-et-maitriser-ses-emotions/jalousie-en-situation-de-conflit-comprendre-agir.md) | — |
| Exprimer ses préférences de proximité (`affection_preferences`) | C | [tendresse](../content/posts/comprendre-et-maitriser-ses-emotions/tendresse-avec-ses-enfants-bernard.md) | Inès 2, Inès 8, Inès 12, Vincent 7 |
| Parler de son intimité à son rythme (`intimacy_communication`) | C | [journaling de pensées](../content/posts/reprendre-le-controle-de-ses-pensees/journaling-pensees-intimite-sexualite-nadia.md) | Inès 1, Inès 3, Inès 5, Inès 8, Vincent 7 |
| Composer avec plusieurs univers culturels (`bicultural_navigation`) | — | Aucun sujet ou voisin retenu | Malik 1, Malik 2, Malik 9, Malik 12 |
| Être reconnu dans la manière de se définir (`identity_recognition`) | — | Aucun sujet ou voisin retenu | Samira 3, Samira 12, Salomé 1, Salomé 6, Salomé 10, Salomé 12, Malik 1, Malik 6, Malik 12, Anaïs 7 |
| Choisir sa manière de se présenter (`self_expression`) | C | [le faux self (Winnicott)](../content/posts/sortir-de-ses-schemas-repetitifs/faux-self-winnicott-parents.md) | Salomé 1, Salomé 3, Salomé 6, Salomé 11, Anaïs 9 |
| Choisir à qui révéler une information personnelle (`disclosure_choice`) | C | [vulnérabilité](../content/posts/comprendre-et-maitriser-ses-emotions/vulnerabilite-en-couple-signes.md) | Salomé 2, Salomé 4, Salomé 8 |
| Se comparer dans un environnement numérique (`social_media_comparison`) | C | [effet de halo](../content/posts/reprendre-le-controle-de-ses-pensees/effet-de-halo-sur-les-reseaux-sociaux.md) | Salomé 9 |
| Attentes sociales autour des identités (`role_expectations`) | C | [surcontrôle émotionnel/inhibition](../content/posts/sortir-de-ses-schemas-repetitifs/surcontrole-emotionnel-famille.md) | Salomé 3 |
| Se surveiller pendant un échange (`self_focused_attention`) | C | [effet spotlight](../content/posts/reprendre-le-controle-de-ses-pensees/effet-spotlight-ecole-etudes.md) | Rémi 10 |

Statut : les références de la dernière colonne correspondent aux liens éditoriaux des récits rédigés et contrôlés, y compris leurs angles secondaires. « — » signifie aucun nouveau développement associé ; consulter la colonne de couverture initiale. Ces rapprochements ne sont pas des scores cliniques.


## Résultat de l’intégration

Les 13 nouveaux personnages ont chacun 12 récits, soit 156 textes originaux et 305,015 mots mesurés dans leurs corps. Le catalogue final compte 33 personnages et 387 articles. Les 231 articles de départ sont conservés. Aucun générateur d’articles ni fournisseur de génération de textes n’a été appelé pour ce pack.

Les récits nouveaux portent des liens éditoriaux revus vers 143 repères distincts. Avec les 44 sujets directs identifiés dans la base, leur union couvre 161 des 161 repères de cette grille. Un lien peut correspondre à un angle secondaire développé ; il ne signifie pas qu’un récit est dédié à chaque repère. Les 28 exigences secondaires explicites du plan ont toutes une correspondance dans le manifeste. Le référentiel reste ouvert et ne représente pas l’entièreté de la psychologie humaine.

Les historiques des 33 fiches ont été reconstruits depuis la matrice et les publications réelles : les 60 références anciennes sans publication ne servent plus de continuité. Les nouveaux liens sont exportés dans le catalogue de recherche, utilisés pour relier cartographie et lectures, et distingués des preuves personnelles. Les titres, citations et URLs proposés par Ambre proviennent du catalogue public canonique.
