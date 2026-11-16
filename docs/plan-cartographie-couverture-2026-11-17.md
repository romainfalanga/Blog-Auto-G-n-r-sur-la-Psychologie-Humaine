# Cartographie : plan du chantier de couverture — 17 novembre 2026

Cahier des charges pour les agents de recherche. Document de préparation : **aucune recherche web n'a été faite ici, aucun fichier existant n'a été modifié, rien n'a été commité ni déployé**. Les chiffres viennent d'une lecture par parseur de `data/cartographie.json`, des fragments `editorial/cartographie/`, de `scripts/build_cartographie.py` et des trois documents d'audit du 16 novembre. Les références bibliographiques citées comme « pistes » sont des points de départ **non vérifiés** : chaque agent doit les retrouver et les lire avant de s'en servir.

## 0. Règle absolue et état de départ

- Un lien vécu → sujet n'existe que s'il est documenté par une méta-analyse, une revue systématique, une étude de cohorte, une classification officielle ou un modèle clinique publié (ce dernier en `theorique` / `clinical_model`).
- Un sujet sans lien documenté avec l'enfance **n'est pas ajouté au graphe** : il est inscrit au registre « hors périmètre, pourquoi » (§ 6.5). « Aucune preuve trouvée » est un résultat valide et attendu.
- État vérifié : 58 facteurs (6 thèmes), 82 sujets, 425 liens, 244 sources, 8 ressources protectrices. `python3 scripts/build_cartographie.py --check` et `node scripts/build_cartographie_module.mjs --check` passent sur l'arbre actuel.
- Liens par niveau : 21 `forte`, 155 `moderee`, 128 `limitee`, 108 `theorique`, 13 `critere`. Par nature : 303 associations empiriques, 83 modèles cliniques, 26 définitionnels, 13 contextes de classification.
- Sources : 118 méta-analyses, 87 études, 15 revues, 15 théories, 7 recommandations, 2 classifications. **223 sur 244 lues au seul résumé** ; 233 liens sur 425 reposent sur une seule source.
- Grille des niveaux (plan du 15 novembre, à conserver) : `forte` = méta-analyses concordantes ; `moderee` = études longitudinales ou méta-analyse à effet modeste ; `limitee` = données transversales ou rares ; `theorique` = modèle clinique ; `critere` = contexte décrit par une classification (attribué automatiquement).

## 1. Ce que le générateur impose (lu dans `scripts/build_cartographie.py`)

| Contrôle | Effet |
| --- | --- |
| `kind` du sujet ∈ `trouble`, `schema`, `attachement`, `tendance` | sinon erreur |
| Identifiant de sujet commençant par `icd_6a00` … `icd_6a05` | **interdit** (neurodéveloppement) |
| `url` d'un sujet | doit pointer une page existante (`/personnages/<slug>/` ou `/posts/<section>/<slug>/`), sinon erreur → **omettre le champ en cas de doute** |
| `factor` d'un lien | doit exister dans `facteurs.json` |
| `strength` ∈ `forte`, `critere`, `moderee`, `limitee`, `theorique` | sinon erreur |
| `sources` d'un lien | liste non vide d'identifiants définis |
| `summary` | refuse `provoque`, `cause directement`, `entraîne forcément`, `garantit`, **`toujours`** (y compris « pas toujours ») et tout **chiffre suivi de `%`** |
| Source | `type` ∈ `classification`, `recommandation`, `meta-analyse`, `revue`, `etude`, `theorie` ; `url` commençant par `http` |
| `relation_type` | optionnel ; ∈ `empirical_association`, `classification_context`, `clinical_model`, `definitional`. Déduit sinon : sources toutes `classification` ou force `critere` → `classification_context` (force ramenée à `critere`) ; `theorique` → `clinical_model` ; reste → `empirical_association` |
| Facteur du thème `ressentis` (ou id `ressenti_*`) relié à un sujet `sch_*` | marqué `definitional` automatiquement, donc exclu de la convergence |
| Deux sujets reliés portant le même `label` | erreur |
| Sujet sans aucun lien | **non publié** (silencieusement) |
| Deux sources de même URL | fusionnées sous le premier identifiant |
| Empreinte de l'audit | toute modification des facteurs, sujets, liens ou ressources produit l'erreur « audit des correspondances périmé » (§ 6.6) |

**Pièges de fusion (vérifiés par essai sur une copie isolée) :** les fragments sont lus par ordre alphabétique de nom de fichier ; pour les sources et les sujets, **le premier identifiant lu gagne** ; pour un lien déjà existant, le plus fort gagne et, à force égale, le premier lu gagne, les sources étant fusionnées. Un fichier `liens-<lot>-2026-11-17.json` se classe avant `liens-transparence-…` et `liens-v2-*` : il **écraserait donc silencieusement** un résumé, une source ou un sujet définis dans ces fragments. Conséquence : un nouveau fragment ne redéclare **jamais** une paire (facteur, sujet), un identifiant de source ou un identifiant de sujet déjà publiés ; toute modification de l'existant passe par `proposed_corrections` (§ 6.3).

## 2. Carte de couverture par familles

Lecture : « sujets / liens » = cartes publiées et liens entrants ; preuve dominante entre parenthèses.

### 2.1 CIM-11, chapitre 06 (registre interne : 858 codes, 19 références exactes, 3 codes agrégés)

| Bloc | Couvert | Manques à instruire | Hors périmètre proposé |
| --- | --- | --- | --- |
| 6A0 neurodéveloppement | 0 ; note publique sourcée | — | 38 codes exclus par règle ; 6A06 et résiduels à inscrire au registre |
| 6A2 psychoses | `icd_6a20` groupe, 10 liens (1 forte, 5 modérées) | 6A22 schizotypique ; distinguer expériences psychotiques et trouble | 6A21, 6A23, 6A24 sans littérature propre attendue : à statuer |
| 6A4 catatonie | 0 | — | syndrome médical |
| 6A6 bipolaires | `icd_6a60`, 2 liens modérés | 6A61 type II, 6A62 cyclothymie | résiduels |
| 6A7 dépressifs | `dep_generale` 35 liens (6 fortes, 18 modérées) ; `icd_6a72` 2 limités | 6A73 mixte ; séparer symptômes et épisode (6A70/6A71 sont des alias) ; dépression périnatale (6E20/6E21) | spécificateurs, résiduels |
| 6B0 anxieux | `anx_generale` 27 liens ; 6B01 (2), 6B02 (1), 6B04 (3), 6B05 (1) | 6B00 anxiété généralisée distincte (alias aujourd'hui) ; 6B03 phobie spécifique ; 6B06 mutisme sélectif | résiduels |
| 6B2 TOC et apparentés | `icd_6b23` hypocondrie, 1 lien limité | 6B20 TOC ; 6B21 dysmorphie ; 6B24 accumulation ; 6B25 trichotillomanie/excoriation | 6B22 olfactif : preuve probablement absente |
| 6B4 stress | 6B40 (7), 6B41 (4), 6B42 (2), 6B43 (2), 6B44 et 6B45 (3 chacun, `contextual_only`) | néant au niveau des codes ; relire les 8 liens `critere` | résiduels |
| 6B6 dissociatifs | `ten_dissociation` 4 ; 6B60 (1) ; 6B64 (1 critère) | 6B61 amnésie ; 6B66 dépersonnalisation-déréalisation ; 6B65 | 6B62/6B63 transe et possession : dépendants de la culture, à statuer |
| 6B8 alimentation | `icd_6b81` agrégé 6B80–6B82, 5 liens | éclater anorexie, boulimie, hyperphagie si les preuves diffèrent | 6B83, 6B84, 6B85 : à statuer |
| 6C0 élimination | 0 | énurésie/encoprésie : seulement comme repère de l'enfant, `contextual_only` | si aucune méta-analyse |
| 6C2 détresse corporelle | `icd_6c20` 2 limités | — | 6C21 |
| 6C4 substances | `icd_6c40_2` 1 ; `ten_usage_substances` 14 (9 modérées) | preuves par produit (cannabis, opioïdes, nicotine) | intoxications, sevrages, troubles induits : environ 400 sous-codes |
| 6C5 comportements addictifs | `ten_ecrans` 3 limités | 6C50 jeu d'argent ; 6C51 jeu vidéo | résiduels |
| 6C7 impulsions | 0 | 6C72 comportement sexuel compulsif ; 6C73 explosif intermittent | 6C70, 6C71 si aucune preuve |
| 6C9 conduites | `ten_externalisation` 11 (6C90 en alias) | 6C91 distinct ; passage à l'âge adulte | — |
| 6D1 personnalité | `icd_6d11_5` 5 modérés | 6D10 sévérité ; cinq domaines de traits 6D11.0–.4 | — |
| 6D3 paraphiliques | 0 | — | **exclusion de principe** : ne jamais attribuer un trouble paraphilique à une victime |
| 6D5 factices | 0 | à instruire, preuve probablement insuffisante | probable |
| 6D7–6D8 neurocognitifs | 0 | association ACE–démence éventuelle : seulement en santé physique contextuelle (lot 7) | diagnostics eux-mêmes |
| 6E2 périnatal | 0 | voir 6A7 | — |
| 6E6, 6E40, résiduels | 0 | — | syndromes secondaires à une maladie |

### 2.2 Codes voisins de la CIM-11 (absents du registre interne ; **codes à vérifier dans le navigateur OMS avant citation**)

- Chapitre 07 sommeil-veille : `ten_sommeil` 7 liens (4 modérés), sans code. Manque : distinguer insomnie chronique et cauchemars.
- Chapitre 17 santé sexuelle : `ten_difficultes_sexuelles` 1, `ten_risques_sexuels` 1. Manque : dysfonctions et douleur sexuelle par type. Hors périmètre : incongruence de genre (aucune inférence à partir de l'enfance).
- Chapitre 21 symptômes : `ten_conduites_suicidaires` 15 liens (4 fortes) regroupe idées, tentatives et automutilations non suicidaires : à scinder.
- Chapitre 24 facteurs influant sur l'état de santé (QE…) : ce sont des **vécus**, pas des sujets ; aucune table de correspondance facteurs ↔ codes QE n'existe (antécédents de maltraitance, problèmes relationnels dans l'enfance, perte, éducation). À produire par le lot 8.
- Burn-out (phénomène professionnel) et douleur chronique : le premier est absent, la seconde est approchée par `ten_douleurs_persistantes` (6 liens).

### 2.3 DSM-5-TR

- Section II : 22 chapitres examinés au niveau thématique — 12 partiels, 2 agrégés, 5 hors périmètre (neurodéveloppement, dysphorie de genre, neurocognitifs, autres troubles, effets des médicaments), 3 à examiner (TOC et apparentés, élimination, paraphiliques). **Aucune table diagnostic par diagnostic** ; l'accès à la classification officielle renvoyait HTTP 403.
- Diagnostics sans équivalent direct à statuer : trouble disruptif avec dysrégulation émotionnelle, trouble dysphorique prémenstruel, stress aigu, les dix troubles de la personnalité catégoriels (seul le motif borderline CIM existe).
- Section III : modèle alternatif des troubles de la personnalité, conditions à l'étude, concepts culturels de détresse : **non examinés**.
- Obstacle d'interface : le libellé public du type `trouble` est « Trouble étudié (CIM-11) » ; un diagnostic propre au DSM serait mal étiqueté (§ 7).

### 2.4 Modèles cliniques et dimensions psychologiques

| Famille | Couvert | Manques | Hors périmètre / réserves |
| --- | --- | --- | --- |
| Schémas de Young | 18/18 sujets, 81 liens (majorité théoriques ou définitionnels) ; 6 schémas n'ont qu'un ou deux liens (`sch_controle_insuffisant`, `sch_fusion`, `sch_droits_excessifs`, `sch_echec`, `sch_negativite`, `sch_dependance`) | preuves empiriques par schéma ; source primaire lue (aujourd'hui `carlucci2018`, source secondaire, porte 38 liens et `young2003` est lu sur la présentation de l'éditeur) | — |
| Modes et styles d'adaptation (soumission, évitement, surcompensation) | 0 | à instruire comme modèle théorique | probablement peu de données développementales |
| Attachement | 3 sujets, 19 liens (1 forte, 13 modérées) | attachement sécure (plutôt ressource) ; `att_desorganise` repose sur des études du nourrisson alors que la carte se lit comme un style adulte ; style craintif adulte | — |
| Mécanismes de défense (échelle du DSM-IV, DMRS) | 0 sujet ; `pat_emotional_suppression` voisin (6 liens) | hiérarchie des défenses ; 3 des 16 repères à examiner en relèvent | défenses matures = ressources |
| Traits (Big Five, AMPD, HiTOP) | 0 | névrosisme/affectivité négative, détachement, antagonisme, désinhibition, psychoticisme | part héréditaire importante : n'ajouter que sur méta-analyse, résumé prudent |
| Régulation émotionnelle | `ten_dysregulation_emotionnelle` 4, `pat_emotional_suppression` 6, `ten_colere` 6 (limitée dominante) | rumination (aucune carte), stratégies distinctes | compétences = ressources |
| Dissociation | 3 sujets, 6 liens | voir 6B6 | — |
| Estime de soi | `ten_estime_soi` 3 | peu de facteurs reliés | — |
| Honte, culpabilité | `sch_imperfection` 9, `pat_guilt` 7 (5 théoriques), `pat_self_criticism` 3 | aucune carte « honte » propre : `meng2026` (honte) alimente le schéma d'imperfection | — |
| Alexithymie | `ten_alexithymie` 6 (3 modérés) | — | ne pas confondre avec l'autisme |
| Parentification, rôles familiaux | facteur `parentification` 9 liens ; `enfant_sage` et `enfant_prefere_role` 11 liens, tous théoriques ; sujet `pat_parentification` 3 limités | méta-analyses sur la parentification ; rôles familiaux sans base empirique solide | `pat_parentification` est un vécu présenté comme conséquence : à requalifier |
| Perfectionnisme, imposture | `pat_perfectionism` 4, `sch_exigences_elevees` 3, `pat_conditional_worth` 3, `pat_impostor` 4 | dimensions du perfectionnisme | — |
| Codépendance, « sauveur » | `ten_codependance` 5, `pat_overresponsibility` 12 (11 théoriques), `sch_abnegation` 5, `pat_appeasement` 7 (théoriques) | toute preuve empirique ; sinon maintien explicite en modèle | concept sans définition officielle : le dire |
| Hypervigilance | `ten_hypervigilance` 8 | biais attentionnel envers la menace (méta-analyses) | — |
| Compétences relationnelles | 8 sujets, 25 liens (limitée dominante) | jalousie, mentalisation, confiance | compétences ordinaires = ressources |
| Santé physique (étude ACE) | `ten_sante_physique` 9, `ten_douleurs_persistantes` 6, `ten_surpoids` 4, `ten_sommeil` 7, `icd_6c20` 2 | maladies détaillées (cardiovasculaire, diabète, respiratoire, auto-immunité), mortalité prématurée | maladies sans méta-analyse |
| Trajectoires sociales | `ten_difficultes_scolaires` 7, `ten_difficultes_professionnelles` 3, `ten_souffrance_enfance` 12 | — | — |
| Protection, résilience | 8 ressources, **hors graphe et hors calcul** | expériences positives de l'enfance (échelles BCE, PCE), effets tampons | — |
| Croissance post-traumatique | 0 | à instruire ; jamais comme attente ni contrepartie de l'adversité | — |

## 3. Côté vécus : les 58 facteurs face aux instruments

Thèmes actuels : violences 7, manques de soins 3, famille 13, climat 13, ressentis 14, contexte 8. La structure des instruments est rappelée de mémoire : le lot 8 la confirme sur les manuels avant toute décision.

| Instrument | Couverture | Écarts |
| --- | --- | --- |
| ACE (Felitti, CDC) | 10 catégories sur 10 (11 cases marquées `ace`, la séparation étant dédoublée et exclusive) | les cases disent « parent » là où l'instrument dit « membre du foyer » |
| ACE-IQ (OMS), 13 catégories | 11 nettes, 1 partielle | **violence communautaire absente** ; violence collective fondue dans `exil` |
| CTQ, 5 sous-échelles | 5 sur 5 | pas de gradation de fréquence |
| MACE, 10 types | 9, avec fusions | **témoin de violences sur la fratrie absent** ; violences verbale et non verbale fondues ; harcèlement physique et émotionnel fondus ; **âge et durée d'exposition non recueillis** |
| Questionnaires de polyvictimisation | partiel | vols et agressions hors foyer, victimisation sexuelle par des pairs ou en ligne non distinguées |

Vécus manquants à instruire (lot 8) : violence dans le quartier ou à l'école hors harcèlement ; violence collective sans exil ; témoin de violences sur un frère ou une sœur ; placement tardif en famille d'accueil ou en foyer (`soins_instables` ne vise que les premières années) ; cyberviolence et exposition précoce à des contenus sexuels ou violents ; rejet familial lié à l'orientation ou à l'identité ; violences par un adulte d'institution (école, sport, culte) ; internat précoce ; handicap ou trouble des apprentissages de l'enfant vécu comme stigmatisation (sans lien causal vers le neurodéveloppement) ; suicide d'un proche (fondu dans `deces_proche` et `parent_trouble`).

Découpages à revoir :
- `controle` mêle contrôle comportemental (style autoritaire) et **contrôle psychologique** ; ses sources portent sur les deux.
- `precarite` n'a que 2 liens malgré une littérature abondante.
- `harcelement` réunit école et en ligne ; `exil` réunit guerre et parcours d'exil ; `abus_sexuel` ne distingue pas l'auteur.
- `secret_famille` 1 lien, `maladie_famille` 2, `se_sentir_different` 2, `parent_incarcere` 3 : à étayer ou à déclarer faibles.
- Styles de Baumrind : autoritaire ≈ `controle`, permissif ≈ `absence_limites`, négligent ≈ `negligence_emotionnelle` ; le style démocratique manque comme ressource. Rejet parental (théorie de Rohner) réparti sur trois cases.
- « Parent narcissique » : aucune case, et c'est préférable ; décrire des comportements déjà présents (`amour_conditionnel`, `critique`, `parentification`, `triangulation`) et n'envisager une case que si des études portent sur ces comportements, jamais sur un diagnostic du parent.
- Les 14 ressentis sont des impressions rapportées, pas des expositions : 17 de leurs liens sont classés empiriques (ex. `ressenti_honte` → anxiété sociale) avec un risque de circularité.

## 4. Les 16 repères « à examiner » et les recouvrements à trancher

Question commune : existe-t-il une méta-analyse ou une cohorte reliant un vécu d'enfance à ce processus ? Sinon : hors périmètre, en le disant.

| Repère | Référentiel à consulter | Décision probable (à confirmer) |
| --- | --- | --- |
| `intellectualization`, `rationalization`, `minimization` | DMRS, échelle de fonctionnement défensif | lot 4 ; liens théoriques au mieux |
| `all_or_nothing`, `overgeneralization`, `emotional_reasoning`, `mind_reading` | distorsions cognitives (Beck, Burns), style cognitif négatif | lot 6 ; une seule carte « distorsions » si la preuve est globale |
| `confirmation`, `sunk_cost` | biais de jugement généraux | hors périmètre sauf preuve développementale |
| `uncertainty` | intolérance à l'incertitude | lot 6 ; lien possible avec surprotection ou imprévisibilité |
| `reward_delay` | dévalorisation temporelle | lot 6 ; lien possible avec précarité ou imprévisibilité |
| `observational_learning` | apprentissage social, transmission intergénérationnelle | mécanisme, pas sujet : note explicative plutôt que carte |
| `relational_jealousy` | attachement adulte | lot 5 ou 6 |
| `self_focused_attention` | modèle cognitif de l'anxiété sociale | probablement rattaché à `icd_6b04` |
| `overwork` | addiction au travail | lot 6 ; probablement limité ou hors périmètre |
| `social_media_comparison` | contexte actuel | hors périmètre du questionnaire |

Recouvrements partiels prioritaires parmi les 55 : `rumination` (seulement citée dans une ressource), `catastrophizing`, `shame`, `trust`, `closeness_distance`, `impulsive_reaction`, `distress_tolerance`, `emotional_detachment`, `safety_behaviors`, `comparison`, `overcontrol`, `withdrawal`, `self_trust`, `body_stress`. Pour chacun : décider carte propre, alias d'une carte existante ou maintien en recouvrement, avec la source qui le justifie. Les 61 repères hors périmètre (ressources, fonctions exécutives, préférences) ne sont pas rouverts.

## 5. Liens existants à revérifier en priorité (lot 9)

1. **Les 21 liens `forte`** : pour 18 d'entre eux, les sources empiriques ne sont lues qu'au résumé. À examiner d'abord : `migration` → `icd_6a20` (facteur large incluant un changement de région, sujet sensible) ; `conflits_parents` → `anx_generale` et `dep_generale` (source de 2008, effets à vérifier) ; `punitions_corporelles` → `ten_externalisation` ; les trois liens `harcelement` ; les quatre liens `abus_sexuel` appuyés sur `chen2010`.
2. **81 liens `forte`/`moderee` à source unique, dont 76 lus au résumé.** Sources portant seules le plus de liens : `li2026` (5), `auersperg2019` (4), `varese2012`, `ditzer2023`, `li2019_ipv`, `cyr2010`, `meng2026`, `vonderlin2018` (3 chacune).
3. **16 liens `moderee` sans méta-analyse ni revue** : `abus_physique` et `temoin_violences` → `ten_hypervigilance` ; `abus_sexuel` → `ten_revictimisation` ; `adoption` → `ten_conduites_suicidaires` ; `deces_proche` → `anx_generale`, `ten_conduites_suicidaires`, `ten_usage_substances` ; `demenagements` → `ten_conduites_suicidaires` ; `exil` → `icd_6b41` ; `harcelement` → `icd_6b01`, `icd_6b02`, `pat_social_isolation` ; `maladie_enfant` → `icd_6b40` ; `parent_addiction` → `icd_6c40_2` ; `traumatisme_ponctuel` → `anx_generale`, `dep_generale`.
4. **Glissements de construit** : `meng2026` (honte) → `sch_imperfection` pour trois facteurs ; attachement désorganisé du nourrisson (`cyr2010`, `madigan2006`, `martins2000`, `vanijzendoorn2020`) présenté comme style d'attachement ; `mclaughlin2020`, typé `etude` alors qu'il s'agit d'un article de modèle, porte 7 liens dont 2 modérés seuls ; `loades2020` (revue rapide sur l'isolement) pour `ressenti_solitude` → `dep_generale` modéré.
5. **Sources anciennes seules (avant 2012) pour 16 liens modérés** : `amato2001`, `sharpe2002`, `martins2000`, `juffer2005`, `vanijzendoorn2006`, `arriola2005`, `copeland2007`, `mcleod2007_anxiete`, `vanderbruggen2008`, `rhoades2008`, `cyr2010`, `sieh2010`, `goodman2011`. Chercher la méta-analyse qui les remplace.
6. **24 sources datées 2025–2026, toutes lues au résumé** : vérifier que chaque URL mène bien à l'article cité (identifiants PubMed et DOI), en priorité `li2026`, `meng2026`, `lynch2026`, `zeng2026`, `zarubin2026`.
7. **Formulations de mécanisme** que l'expression régulière ne détecte pas : `abus_emotionnel` → `sch_assujettissement` (« apprendre à taire ses besoins… ») ; `parent_malade_physique` → `icd_6c20` (« peut-être parce que… apprend à surveiller ») ; tout résumé reprenant un titre affirmatif.
8. **Sujets agrégés dont la force vaut pour l'ensemble** : `ten_conduites_suicidaires`, `dep_generale`, `anx_generale`, `icd_6b81`, `icd_6a20`, `ten_usage_substances`.
9. **Un lien empirique sans source empirique** : `imprevisibilite` → `att_anxieux` (source théorique seule).

## 6. Découpage en neuf lots de recherche

Règles communes : un lot = un agent = une session = un fichier `editorial/cartographie/liens-<lot>-2026-11-17.json`. Les lots 1 à 7 n'utilisent que les 58 facteurs existants et ne créent de sujets que dans leur périmètre ; un lien vers un sujet relevant d'un autre lot va dans `proposed_links`. Le lot 8 propose des facteurs : comme un facteur inconnu fait échouer le générateur, les liens de ces facteurs vont dans `proposed_links` (vers les sujets existants uniquement) et seuls les liens de facteurs existants vont dans `links`. Le lot 9 ne crée rien et ne produit que des corrections. Deux lots peuvent citer la même publication : l'intégrateur contrôle les collisions d'identifiants de sources (les URL identiques sont fusionnées par le générateur). Aucun lot ne modifie `facteurs.json`, `corrections.json`, un fragment existant ni le code. Ordre de consultation : classification officielle → recommandation → méta-analyse ou revue parapluie récente → cohorte prospective. Chaque source nouvelle est lue au moins dans ses sections méthode et résultats, et `read` le dit.

| Lot (nom du fragment) | Périmètre | Questions | Sources à ouvrir d'abord | Liens attendus |
| --- | --- | --- | --- | --- |
| 1 `humeur-anxiete-toc` | 6A6, 6A7, 6B0, 6B2, 6E2 ; chapitres DSM correspondants | Quels diagnostics distincts ont une association propre avec la maltraitance, le style parental, le harcèlement ? `anx_generale` et `dep_generale` doivent-elles être scindées ? | navigateur CIM-11 et CDDR de l'OMS, notices APA des chapitres, revues parapluies sur l'adversité et les troubles mentaux | 25–40 |
| 2 `psychose-dissociation-corps` | 6A2, 6B6, 6C2, 6B8, 6C0 ; scission de `ten_conduites_suicidaires` | Preuve par diagnostic alimentaire ? Dépersonnalisation, amnésie dissociative, schizotypie ? Énurésie comme repère de l'enfant ? | CDDR, recommandations NICE (alimentation, automutilation), méta-analyses par diagnostic | 20–35 |
| 3 `addictions-conduites-sommeil-sexualite` | 6C4, 6C5, 6C7, 6C9, 6D3, 6D5 ; chapitres 07 et 17 | Preuve par produit ? Jeu d'argent, jeu vidéo, comportement sexuel compulsif, trouble explosif intermittent ? Insomnie et cauchemars séparés ? Paraphilies et factices : rédiger l'exclusion | CIM-11 chapitres 06, 07, 17 ; classification internationale des troubles du sommeil ; rapports OMS sur l'alcool et les drogues | 20–35 |
| 4 `personnalite-traits-defenses` | 6D10–6D11, troubles de la personnalité du DSM, AMPD, Big Five, HiTOP, mécanismes de défense | Quels traits et troubles ont une méta-analyse avec l'adversité ? Les défenses immatures ? Comment écrire sans laisser croire que l'éducation fait le tempérament ? | CDDR (personnalité), section III du DSM-5-TR, publications du consortium HiTOP, manuel DMRS | 20–35 |
| 5 `schemas-attachement-emotions` | 18 schémas, modes, styles d'adaptation, attachement, régulation, honte, culpabilité, estime de soi, alexithymie | Preuve empirique par schéma ; attachement enfant ou adulte ; carte « honte » distincte ? ; rumination | ouvrage de Young, Klosko et Weishaar (texte, pas la présentation de l'éditeur), ISST, méta-analyses par construit | 35–50 |
| 6 `roles-cognitions-relations` | parentification, codépendance, sauveur, perfectionnisme, imposture, hypervigilance, compétences relationnelles, les 16 repères du § 4 et les recouvrements prioritaires | Quelles preuves au-delà du modèle théorique ? Quels repères deviennent une carte, un alias, une exclusion ? | méta-analyses par construit ; modèles cognitifs publiés | 25–40, plus 16 décisions |
| 7 `sante-trajectoires-ressources` | maladies physiques, douleur, mortalité, scolarité, emploi ; ressources, résilience, croissance post-traumatique | Quelles maladies ont une méta-analyse propre ? Faut-il détailler ou rester agrégé ? Quelles ressources ajouter, sous quelle forme ? | CDC (ACE), OMS, étude du fardeau mondial des maladies, revues parapluies | 25–40 liens, 5–10 ressources |
| 8 `vecus-manquants` | § 3 : vécus absents et découpages ; correspondance facteurs ↔ codes QE | Chaque vécu candidat figure-t-il dans un instrument officiel et dispose-t-il d'au moins deux liens empiriques ? Case nouvelle ou exemple ajouté à une case existante ? | manuel de l'ACE-IQ (OMS), CDC, CTQ, MACE, chapitre 24 de la CIM-11 | 30–50 liens, 6–12 facteurs proposés |
| 9 `reverification` | § 5 | Force, formulation et source de chaque lien fragile sont-elles tenables au vu du texte intégral et d'une méta-analyse plus récente ? | texte intégral des sources citées ; PubMed pour les remplaçantes | 0 lien nouveau ; 80–120 corrections proposées |

Ordre de grandeur total : 60 à 90 sujets et 200 à 320 liens nouveaux, 6 à 12 facteurs. Un lot qui en trouve nettement moins a bien travaillé s'il a rempli `out_of_scope`.

### 6.1 Identifiants des 58 facteurs (seuls admis dans `factor`)

- **violences** : `abus_physique`, `abus_sexuel`, `abus_emotionnel`, `temoin_violences`, `harcelement`, `violences_fratrie`, `punitions_corporelles`
- **negligence** : `negligence_physique`, `negligence_emotionnelle`, `soins_instables`
- **famille** : `parent_trouble`, `parent_addiction`, `parent_incarcere`, `separation_conflit`, `divorce`, `deces_proche`, `maladie_famille`, `precarite`, `conflits_parents`, `parent_absent`, `famille_recomposee`, `adoption`, `parent_malade_physique`
- **climat** : `invalidation`, `critique`, `amour_conditionnel`, `surprotection`, `controle`, `imprevisibilite`, `parentification`, `comparaison_fratrie`, `secret_famille`, `absence_limites`, `triangulation`, `pression_scolaire`, `emprise_religieuse`
- **ressentis** : `ressenti_pas_compter`, `ressenti_abandon`, `ressenti_honte`, `ressenti_responsable`, `ressenti_perfection`, `ressenti_peur`, `ressenti_solitude`, `ressenti_colere_interdite`, `honte_famille`, `debrouiller_seul`, `pas_protege`, `se_sentir_different`, `enfant_sage`, `enfant_prefere_role`
- **contexte** : `discrimination`, `exil`, `migration`, `traumatisme_ponctuel`, `maladie_enfant`, `demenagements`, `moqueries_corps`, `isolement_social_enfant`

### 6.2 Identifiants des 82 sujets existants (réutilisables dans `outcome`, à ne jamais redéclarer)

`anx_generale`, `dep_generale`, `icd_6a20`, `icd_6a60`, `icd_6a72`, `icd_6b01`, `icd_6b02`, `icd_6b04`, `icd_6b05`, `icd_6b23`, `icd_6b40`, `icd_6b41`, `icd_6b42`, `icd_6b43`, `icd_6b44`, `icd_6b45`, `icd_6b60`, `icd_6b64`, `icd_6b81`, `icd_6c20`, `icd_6c40_2`, `icd_6d11_5` ; `att_anxieux`, `att_evitant`, `att_desorganise` ; `sch_abandon`, `sch_mefiance`, `sch_carence_affective`, `sch_imperfection`, `sch_isolement`, `sch_dependance`, `sch_vulnerabilite`, `sch_fusion`, `sch_echec`, `sch_droits_excessifs`, `sch_controle_insuffisant`, `sch_assujettissement`, `sch_abnegation`, `sch_recherche_approbation`, `sch_negativite`, `sch_surcontrole`, `sch_exigences_elevees`, `sch_punition` ; `pat_abandonment_fear`, `pat_appeasement`, `pat_conditional_worth`, `pat_conflict`, `pat_emotional_suppression`, `pat_guilt`, `pat_impostor`, `pat_migration_adjustment`, `pat_needs_expression`, `pat_overresponsibility`, `pat_parentification`, `pat_perfectionism`, `pat_self_criticism`, `pat_social_isolation` ; `ten_alexithymie`, `ten_codependance`, `ten_colere`, `ten_conduites_suicidaires`, `ten_difficultes_professionnelles`, `ten_difficultes_scolaires`, `ten_difficultes_sexuelles`, `ten_dissociation`, `ten_douleurs_persistantes`, `ten_dysregulation_emotionnelle`, `ten_ecrans`, `ten_estime_soi`, `ten_externalisation`, `ten_hypervigilance`, `ten_peur_intimite`, `ten_procrastination`, `ten_qualite_relationnelle`, `ten_revictimisation`, `ten_risques_sexuels`, `ten_sante_physique`, `ten_sommeil`, `ten_souffrance_enfance`, `ten_surpoids`, `ten_usage_substances`, `ten_violences_couple`.

Identifiants déjà pris comme alias (ne pas réutiliser sans décision de l'intégrateur) : `icd_6a70`, `icd_6a71`, `icd_6b00`, `icd_6c90`, `icd_6d10`, `ten_symptomes_depressifs`, `ten_symptomes_anxieux`, `ten_suicide_automutilation`, `ten_substances`, `ten_alcool_adolescence`, `pat_emotional_identification`, `ten_regulation_emotionnelle`, `ten_comportements_antisociaux`, `ten_image_corps`. Sujets retirés : `pat_identity_transition`, `pat_disclosure_choice`.

Identifiants de sources existants : les extraire avant de commencer (`python3 -c "import json;print(*sorted(s['id'] for s in json.load(open('data/cartographie.json'))['sources']))"`) ; une source déjà présente se cite par son identifiant sans être redéclarée.

### 6.3 Format exact d'un fragment

Exemple complet et valide, repris tel quel de `editorial/cartographie/liens-v2-corps.json` (un sujet, un lien, ses deux sources) :

```json
{
 "part": "corps",
 "author": "recherche_corps",
 "outcomes": [
  {
   "id": "ten_sommeil",
   "kind": "tendance",
   "label": "Difficultés de sommeil (insomnie, cauchemars)",
   "description": "Du mal à s'endormir, des réveils fréquents, des nuits courtes ou des cauchemars qui reviennent.",
   "url": "/posts/reprendre-le-controle-de-ses-pensees/maelle-inquietude-endormissement/"
  }
 ],
 "links": [
  {
   "factor": "abus_physique",
   "outcome": "ten_sommeil",
   "strength": "moderee",
   "summary": "Avoir subi des violences physiques dans l'enfance est associé, en moyenne, à un peu plus de difficultés de sommeil à l'âge adulte (dans des études qui suivent les personnes dans le temps), et à plus d'insomnie et de cauchemars dès l'enfance. L'effet est modeste et beaucoup de personnes dorment bien.",
   "sources": ["yu2022_sommeil", "schonning2022"]
  }
 ],
 "sources": [
  {
   "id": "yu2022_sommeil",
   "citation": "Yu HJ, Liu X, Yang HG, Chen R, He QQ, 2022, The association of adverse childhood experiences and its subtypes with adulthood sleep problems: a systematic review and meta-analysis of cohort studies, Sleep Medicine 98:26-33",
   "url": "https://pubmed.ncbi.nlm.nih.gov/35777139/",
   "type": "meta-analyse",
   "read": "résumé"
  },
  {
   "id": "schonning2022",
   "citation": "Schønning V, Sivertsen B, Hysing M, Dovran A, Askeland KG, 2022, Childhood maltreatment and sleep in children and adolescents: a systematic review and meta-analysis, Sleep Medicine Reviews 63:101617",
   "url": "https://pubmed.ncbi.nlm.nih.gov/35313257/",
   "type": "meta-analyse",
   "read": "résumé"
  }
 ]
}
```

Lien théorique, repris de `liens-transparence-2026-11-16.json` : `{"factor": "critique", "outcome": "sch_negativite", "strength": "theorique", "relation_type": "clinical_model", "summary": "Le modèle de Young situe ce schéma dans un domaine décrit avec des milieux familiaux très exigeants ou punitifs. Ce rapprochement théorique avec les critiques et exigences ne démontre ni un effet causal ni la présence du schéma chez une personne.", "sources": ["isst_young_domains", "young_schemas"]}`.

Champs et valeurs autorisées :

| Objet | Champ | Règle |
| --- | --- | --- |
| fragment | `part`, `author` | texte libre ; `part` = nom du lot |
| fragment | `sources`, `outcomes`, `links` | listes (lues par le générateur) |
| fragment | `protections` | liste `{id, label, description, sources}` ; `id` unique en `prot_…` ; lot 7 seulement |
| fragment | `notes` | **liste de chaînes uniquement** ; un objet serait publié sur le site |
| fragment | `out_of_scope`, `proposed_factors`, `proposed_links`, `proposed_corrections` | listes ignorées par le générateur (vérifié), lues par l'intégrateur ; `proposed_links` suit le format de `links` |
| sujet | `id` | `^[a-z0-9_]{1,80}$`, nouveau ; `icd_<code en minuscules, point remplacé par _>` pour un code CIM-11, `ten_…` pour une tendance ou un symptôme, `att_…` pour l'attachement ; `sch_…` est fermé aux 18 schémas ; jamais `icd_6a00`–`icd_6a05` |
| sujet | `kind` | `trouble`, `schema`, `attachement`, `tendance` |
| sujet | `label` | unique parmi tous les sujets, langage courant |
| sujet | `description` | une ou deux phrases, sans promesse ni diagnostic |
| sujet | `code` | facultatif, code CIM-11 vérifié |
| sujet | `domain` | **à renseigner** : `perceptions`, `humeur`, `anxiete`, `relations`, `soi`, `emotions`, `comportements` (sinon déduction par expression régulière) |
| sujet | `category` | facultatif ; existants : `diagnosis`, `symptom_group`, `clinical_group`, `pattern_qualifier`, `schema`, `attachment_pattern`, `tendency`. Toute valeur nouvelle (ex. `personality_trait`, `defense_mechanism`, `schema_mode`, `health_outcome`, `cognitive_process`) est une **proposition** qui demande un libellé d'interface |
| sujet | `contextual_only`, `age_scope`, `note` | `true` ; `young_child_only` ; `jeune_enfant` ou `suicide`. À poser pour tout tableau propre à l'enfant et tout sujet suicidaire |
| sujet | `url` | à omettre, sauf page vérifiée |
| lien | `factor`, `outcome` | identifiants existants (§ 6.1) ou sujet déclaré dans le même fragment ; paire **inédite** |
| lien | `strength` | `forte`, `moderee`, `limitee`, `theorique` (ne pas écrire `critere` : il est attribué) |
| lien | `relation_type` | à écrire explicitement : `empirical_association`, `clinical_model`, `classification_context`, `definitional` |
| lien | `summary` | une à trois phrases en français courant ; association moyenne, jamais mécanisme ni causalité ; dire la population, le type d'étude et la taille de l'effet en mots ; ni pourcentage ni mots interdits (§ 1) |
| lien | `sources` | au moins un identifiant ; convention proposée pour `forte` : au moins deux méta-analyses concordantes, dont une lue au-delà du résumé |
| source | `id` | nouveau, forme `auteurAAAA` ou `auteurAAAA_theme` |
| source | `citation` | auteurs, année, titre, revue, volume et pages |
| source | `url` | page PubMed, DOI ou page officielle, ouverte et vérifiée |
| source | `type` | `classification`, `recommandation`, `meta-analyse`, `revue`, `etude`, `theorie` |
| source | `read` | `résumé`, `texte intégral`, `section`, `page`, suivi au besoin d'une précision |

Entrées non lues par le générateur :

```json
"out_of_scope": [{"referentiel": "CIM-11", "code": "6B22", "label": "…", "statut": "hors_perimetre_justifie", "raison": "…", "sources": ["…"]}],
"proposed_factors": [{"id": "violence_quartier", "group": "contexte", "label": "…", "description": "…", "examples": ["…"], "ace": false, "overlap_group": null, "instrument": "ACE-IQ, violence communautaire", "justification": "…"}],
"proposed_corrections": [{"factor": "…", "outcome": "…", "action": "abaisser", "strength": "limitee", "summary": "…", "sources": ["…"], "reason": "…"}]
```

`statut` ∈ `hors_perimetre_justifie`, `aucune_preuve_trouvee`, `agrege_dans`, `a_examiner`. `action` ∈ `conserver`, `abaisser`, `relever`, `reformuler`, `remplacer_source`, `retirer`.

### 6.4 Déclarer un nouveau facteur (intégrateur uniquement, sur proposition du lot 8)

- Fichier `editorial/cartographie/facteurs.json`, liste `factors` : `id`, `group` (un des six thèmes), `label`, `description`, `examples` (liste, éventuellement vide), `ace` (`true` réservé aux dix catégories d'origine : il alimente le compteur qui déclenche la note de cumul à partir de 4).
- `overlap_group` se pose dans `corrections.json` → `factor_overrides`. Valeurs existantes : `physical_violence`, `emotional_treatment`, `emotional_neglect`, `family_conflict`, `peer_rejection`, `family_care_burden`, `care_instability`, `reported_feelings`. Deux cases alternatives se déclarent dans `exclusive_factors`.
- **Un identifiant de facteur ne se renomme ni ne se supprime** : les réponses enregistrées sont indexées par identifiant et le Worker refuse tout identifiant inconnu. Une scission garde l'ancienne case et en ajoute une.
- Critères d'admission proposés : reconnu par un instrument officiel ; au moins deux liens empiriques de niveau `limitee` ou plus ; irréductible à un exemple d'une case existante.

### 6.5 Registre « hors périmètre, pourquoi »

Chaque lot remplit `out_of_scope` pour tout élément de son périmètre qu'il ne relie pas. L'intégrateur les réunit dans un fichier dont le nom ne commence pas par `liens-` (par exemple `editorial/cartographie/couverture-referentiels.json`), puis alimente `coverage.known_exclusions` et la table DSM-5-TR diagnostic par diagnostic réclamée par l'audit.

### 6.6 Contrôles

Pour un agent de recherche, sans rien écrire (l'erreur d'empreinte d'audit est attendue et filtrée) :

```bash
python3 -c "import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'scripts'); import build_cartographie as b; d,e=b.build(); print(len(d['factors']),len(d['outcomes']),len(d['links']),len(d['sources'])); print('\n'.join(x for x in e if 'audit des correspondances' not in x) or 'OK hors empreinte audit')"
```

L'agent vérifie en plus à la main : aucune collision d'identifiant ou de paire avec l'existant, chaque sujet nouveau a au moins un lien, chaque URL s'ouvre sur l'article cité.

Pour l'intégrateur, après fusion des neuf fragments et application des corrections retenues :

1. `python3 scripts/build_cartographie.py` (régénère aussi le module du Worker), puis `python3 scripts/build_cartographie.py --check` et `node scripts/build_cartographie_module.mjs --check`.
2. **Obstacle connu** : le générateur exige que `docs/audit-cartographie-correspondances-2026-11-16.json` porte l'empreinte des facteurs, sujets, liens et ressources. Le script qui l'a produit (`/workspace/work/clinical-audit-20261116/audit.mjs`) n'est pas dans le dépôt. Il faut soit le recréer dans `scripts/`, soit produire un audit daté du 17 novembre et y faire pointer le générateur ; les 161 décisions sémantiques doivent être relues, pas seulement réempreintées.
3. Mettre à jour les valeurs figées de `tests/cartographie.test.cjs` (82 sujets, 19 codes exacts, 839 codes non référencés), puis `node --test tests/cartographie.test.cjs journal/tests/cartographie.test.mjs` et les scénarios navigateur.
4. `python3 scripts/build_coverage_status.py --check` et `node scripts/build_psychology_catalog.mjs --check` (inchangés en principe).
5. Relecture indépendante d'un échantillon de liens par un agent distinct, comme pour les deux vagues précédentes, avant toute publication.

## 7. Risques d'interface et de calcul, et garde-fous

Mesures actuelles (simulation sur le jeu réel, 400 tirages par taille) :

| Vécus cochés | Sujets affichés en moyenne | dont convergences | dont pistes | dont modèles |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 19 | 1 | 7 | 11 |
| 10 | 44 | 6 | 13 | 25 |
| 25 | 65 | 17 | 13 | 35 |
| les 10 catégories ACE | 52 | 25 | 9 | 18 |

- **Résultats déjà longs** : dix cases cochées montrent plus de la moitié des 82 sujets. Doubler les sujets rendrait la page illisible et banaliserait chaque carte.
- **Convergence gonflée** : le palier exige deux familles de vécus à lien modéré ou solide. Chaque facteur nouveau sans `overlap_group` crée une famille, donc des convergences. Les dix catégories ACE en produisent déjà 25.
- **Contexte d'Ambre** : le résumé transmet au plus 15 convergences et 10 pistes, et les preuves tiennent dans 6 000 caractères avec six liens par sujet. Dix cases dépassent déjà le plafond des pistes en moyenne (13). La coupe suit palier, score, puis **ordre alphabétique du libellé** : avec plus de sujets, ce qui est transmis devient arbitraire.
- **Poids** : le jeu complet (fichier de 311 Ko) est inséré dans le HTML de la page ; le module du Worker pèse 245 Ko, dont 151 Ko de liens et 72 Ko de sources. Un doublement des liens mène vers 600 Ko de page et 480 Ko de module ; à mesurer sur mobile et à comparer à la limite de taille du Worker.
- **Questionnaire** : 58 questions, trois thèmes à 13 ou 14 ; au-delà de 70, l'abandon et les réponses mécaniques deviennent probables.
- **Libellés** : `trouble` s'affiche « Trouble étudié (CIM-11) » ; traits, défenses, modes et maladies physiques n'ont pas de libellé. `schema` est réservé aux 18 schémas.
- **Tests figés** : nombre de sujets, de codes, et comparaison navigateur/Worker sur toutes les paires de facteurs (1 711 cas pour 58 facteurs, 2 485 pour 70).

Garde-fous proposés :

1. Plafond de 14 questions par thème et de 72 au total ; préférer enrichir `examples` d'une case existante.
2. Tout facteur nouveau reçoit un `overlap_group` dès qu'il recoupe une famille existante.
3. `contextual_only: true` pour les tableaux propres à l'enfant, les maladies physiques détaillées, les diagnostics de personnalité catégoriels et tout sujet dont les seuls liens sont des contextes de classification : visibles comme repères, jamais suggérés ni mis en avant par Ambre.
4. Regrouper l'affichage par domaine avec un nombre limité de cartes ouvertes par domaine, le reste repliable ; aucun pourcentage.
5. Remplacer le départage alphabétique par un ordre explicite (nombre de familles de preuves, puis sujets reconnus par la personne) et ajouter un quota par catégorie dans le résumé transmis à Ambre ; conserver le signal `truncated`.
6. Fixer un budget : résumé de lien de 350 caractères au plus (maximum actuel 352), module du Worker sous 500 Ko, page sous 600 Ko ; au-delà, charger les sources à la demande.
7. Ne publier un sujet nouveau que s'il porte au moins un lien empirique ou un modèle clinique explicitement étiqueté ; les autres restent au registre hors périmètre.
8. Garder séparés les compteurs « sujets reliés », « sujets hors périmètre justifié » et « sujets à examiner » : c'est cette table, et non le nombre de cartes, qui répond à la demande de couverture « de l'ensemble de ce qui existe ».

## 8. Limites de ce plan

- Les familles du § 2 sont un regroupement éditorial : les effectifs par famille se recoupent et ne s'additionnent pas.
- Les codes CIM-11 hors chapitre 06 et la liste des diagnostics DSM-5-TR n'ont pas été vérifiés dans les référentiels : le registre interne ne couvre que le chapitre 06.
- Les décisions « probables » du § 4 et les manques du § 3 sont des hypothèses de travail, pas des conclusions.
- La contrainte clinique de l'audit demeure : aucun contrôle logiciel ne remplace la relecture des affirmations par un professionnel qualifié.
