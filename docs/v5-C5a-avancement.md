# Lot C5a — avancement (Cartographie : contextuels, classement L avec garde-fou, affichage)

## Contrat figé fourni par `journal/src/cartographie.mjs` (pour A5-P0)

Écrit en premier ; A5-P0 l'appelle sans le modifier.

### `computeCartography(answers, data = CARTOGRAPHIE, feedback = {})`
Renvoie `{results, sections, aceCount, selectedCount}`.
- `results` : sujets visibles (ni `retired`, ni `contextual_only`) dans l'**ordre affiché** (sections concaténées dans l'ordre ci-dessous). Chaque élément garde les champs historiques (`outcome`, `links`, `factors`, `associated_factors`, `associated_count`, `other_links`, `tier`, `score`, `evidence_groups`, `relation_types`) et ajoute `relevance` (entier, jamais affiché ni transmis), `section`, `frequent`, `max_points` (12/8/4/0 : meilleur lien mesuré), `has_empirical`, `evidence_families`, `familial` (toutes ses associations mesurées viennent d'un parent malade ou dépendant), `trauma_first`, `serious`.
- `sections` : `{recognized, top, frequent, other, dismissed}`, cinq tableaux d'éléments de `results`, partition exacte :
  - `recognized` : retour « oui » ;
  - `top` : 5 sujets sans retour (`SECTION_TOP = 5`) : d'abord les sujets `trauma_first`, puis l'ordre de pertinence, un trouble grave (`serious`) exigeant un lien solide ; affichés dans l'ordre de pertinence ;
  - `frequent` : sujets `frequent` ayant au moins un lien mesuré modéré ou solide pour ce profil, hors des autres sections (« Aussi associés à la plupart des vécus difficiles », en cartes) ;
  - `other` : les suivants ;
  - `dismissed` : retour « non ».
  Ordre d'affichage = `recognized, top, frequent, other, dismissed` (rang conservé dans chaque section).
- Valeurs de `section` : `recognized | top | frequent | other | dismissed` (même nom que la clé).

### `summarizeCartography(...)` (inchangé dans sa signature)
Ajoute `sections: {recognized:[], top:[], frequent:[], other:[], dismissed:[]}` (éléments `{id,label,kind,section,frequent,max_strength,familial_transmission?,...}`, `other` plafonné à 20, éléments courts) et `other_count`, `display_order` (10 premiers avec `section`, `frequent`, sans score), `display_count` (sans contextuels). `contextual: []` est conservé (toujours vide) pour la compatibilité.

### `cartographyPromptBlock(summary, budget = Infinity)`
Bloc compact transmis au modèle (remplace `cartographyPrompt` + `cartographyWords` de `chat-backend.mjs`, qui délègue désormais). Sans URL publique, sans score. Contient `display_order` (10, avec `section`), `sections: {top:[ids], frequent:[{id,label,max_strength,...}], other_count}`, `contextual: []`, et les listes historiques (`convergence`, `pistes`, `exploratory`, `recognized`, `dismissed`, `evidence`, notes). Les boucles de troncature existantes de `chat-backend.mjs` restent valides (mêmes champs). Si `budget` (longueur JSON maximale en caractères) est donné, réduit dans l'ordre : notes d'Ambre les plus anciennes, pistes, exploratoires, convergences non reconnues au-delà de 3, preuves, sources, notes de la personne au-delà de 4, puis (demande A5-P0) les entrées de `display_order` hors reconnus et top (`display_order_truncated`), `explicit_negative` (`explicit_negative_omitted`), `causes` au-delà de 12 (`causes_omitted`). Ne retire jamais `recognized`, `sections.top`, `sections.frequent`. Ne modifie jamais le résumé reçu. Profil tout coché : 22 790 caractères sans budget, 8 959 avec un budget de 9 000, plancher ≈ 6 200. Renvoie `null` sans carte.

### `cartographyIndexEntries(summary)`
Tableau d'entrées pour l'index d'Ambre : `{key, kind, label, content, updated_at, metadata}` (mêmes clés qu'aujourd'hui : `cause:<id>`, `cartography:negative:<id>`, `cartography:outcome:<id>`, `cartography:resource:<id>`, `note:<i>`). Les sujets suivent l'ordre des sections ; aucun sujet contextuel sauf retour ancien (marqué `contextual_only`). `metadata.section` indique la section. A5-P0 les passe à son `add(key, kind, label, content, updated_at, metadata)`.

## Fait

### Données (`scripts/build_cartographie.py`, `build_cartographie_module.mjs`)
- Nouvel objet `ranking` (hors `outcomes`, pour ne pas périmer l'empreinte de l'audit des correspondances) : `questions_with_measured_link` (N = 43), `specificity` entière par sujet = round(1000·ln(1+N/df)), `frequent` (4·df ≥ N : dépression, anxiété, externalisation, substances, souffrance dans l'enfance, + conduites suicidaires contextuelles), points, `section_top`, et les listes issues de la relecture clinique : `familial_factors` (parent_trouble, parent_addiction), `serious_outcomes` (icd_6a20, icd_6a60, icd_6b60), `trauma_outcomes` (icd_6b40, ten_dissociation), `trauma_factors` (groupe « violences » + exil, ressenti_peur, violence_quartier). Contrôles d'identifiants au build et dans le module.
- `interpretation.ranking` réécrit (spécificité, Souvent, garde-fou, transmission familiale, troubles graves, trauma, « ni une probabilité ni un risque personnel »).
- Régénéré : 109 sujets / 496 liens / 24 contextuels (fragment v5 non inclus, vérifié après le message du coordinateur). Tailles : `data/cartographie.json` ≈ 387 Ko (< 600 000), module ≈ 298 Ko (< 500 000).

### Classement (identique `static/js/cartographie.js` `compute(data, answers, feedback)` et `computeCartography(answers, data, feedback)`)
- Contextuels et retirés exclus. Points 12/8/4, autres liens 1 ; Oui ×2, Souvent ×3 ; meilleur lien par famille ; chaque lien × spécificité (plafonnée à 1000 pour les troubles graves et pour les liens issus d'un parent malade/dépendant).
- Tri : association mesurée d'abord ; **garde-fou** : meilleure preuve limitée — ou associations venant toutes d'un parent malade ou dépendant (contre-relecture) — jamais devant une preuve modérée ou solide (renforcé à la demande de la relecture ; la v5.1 demandait « devant solide ») ; puis pertinence, familles, force, identifiant binaire.
- Sections `recognized / top / frequent / other / dismissed` (voir contrat). Entiers uniquement : parité stricte.
- Code mort supprimé dans `cartographie.js` (`mount`, ancienne interface à filtres et 3114).

### Page (`cartographie-ui.js`, `style.css`)
- Une phrase de classement ; « Vous vous y reconnaissez (n) » ; « Le plus en lien avec vos réponses » (5 cartes) ; « Aussi associés à la plupart des vécus difficiles » (cartes complètes, détail dépliable) ; repli fermé « Autres sujets reliés à vos réponses (n) » (lignes qui s'ouvrent en carte sur place) ; repli « Écartés : « Pas moi » (n) ».
- Carte : titre `h4`, « Relié à : » (vécus mesurés d'abord, « (souvent) », « et N autres »), pastilles de preuve / « Associé à de nombreux vécus », catégorie et code ; « Je me reconnais », « Pas moi », « Voir le détail ». Mention de transmission familiale (carte quand toutes les associations mesurées en viennent ; détail à côté de chaque lien mesuré concerné). « Autres repères (modèles cliniques, définitions) — comptent peu dans l'ordre ».
- Retirés : légende « Repère de contexte », note « jeune enfant », compteurs d'en-tête, `carto-tier-*`/`is-dismissed` des cartes. Fenêtre « Liens étudiés » et son compteur sans sujets contextuels ; « non compté » → « poids faible ».
- Un retour déplace la carte ; un repli s'ouvre pour que le focus la suive ; état des replis conservé.

### Ambre (fonctions fournies ; branchement dans `ambre-engine.mjs` laissé à A5-P0)
- `summarizeCartography` : `sections`, `other_count`, `display_order` avec `section`/`frequent`, `contextual: []`, `selection_policy.ranking = relevance_specificity_v5_strength_guard`. Retour ancien « oui » sur un sujet contextuel conservé dans `recognized` (marqué) et ses preuves dans `cartographyEvidence`.
- `cartographyPromptBlock`, `cartographyIndexEntries` ajoutés ; `chat-backend.mjs` : `cartographyPrompt` délègue (seule modification, + import). `cartographyWords` de `chat-backend.mjs` est désormais inutilisée (à retirer par A5).
- Consigne `CHILDHOOD_CARTOGRAPHY_GUIDANCE` : sections, `familial_transmission`, jamais un rang comme un risque.
- Taille du bloc transmis (avant → après) : violences 17 885 → 13 825 caractères (−23 %), parentification 4 104 → 3 837 (−7 %), tout coché 31 260 → 22 819 (−27 %).

### Mesures (2 000 profils mixtes 1–20 cases, 30 % « Souvent », graine 20261119)
| | Dép./anx. 1er | top 3 | top 5 | sujets distincts top 5 | Jaccard entre profils | stabilité +1 case | stabilité « Souvent » | tête sur un seul lien limité | solide derrière limité |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Avant (affiché, contextuels inclus) | 87 % | 95 % | 96 % | 58 | 0,38 | 0,84 | 1,00 | 4 % | 41 % |
| L + garde-fou « solide » (avant relecture) | 26 % | 55 % | 71 % | 65 | 0,16 | 0,81 | 0,93 | 8 % | 0 % |
| **Final (après corrections de la relecture)** | **33 %** | **65 %** | **76 %** | **58** | **0,21** | **0,83** | **0,93** | **4 %** | **0 %** |

Par taille (final) : 3 cases 15/50/67 % ; 6 : 18/56/70 ; 10 : 26/65/78 ; 15 : 41/74/84. Profils typés : voir le rapport de relecture ci-dessous.

### Relecture clinique indépendante (session distincte, relecture seule)
Verdict initial : « non acceptable en l'état ». Corrections bloquantes appliquées : (1) transmission familiale (spécificité plafonnée + mention) ; (2) troubles graves (bipolaires, psychotiques, dissociatif à symptômes neurologiques) plafonnés et exclus du top sans lien solide ; (3) garde-fou étendu à « modéré » ; (4) section « Aussi associés » ouverte aux liens modérés ; (5) TSPT / symptômes dissociatifs remontés quand violence ou peur « Souvent » ; (6) phrase : « les sujets reliés à moins de vécus ».
À transmettre à la relecture des liens (C5b), non traité ici : négligence matérielle → troubles bipolaires (modérée) ; parent souffrance psychique → anxiété de séparation (modérée) ; décès → TSPT (contexte CIM) : préférer le trouble du deuil prolongé (absent du catalogue) ; décès → troubles psychotiques.
Contre-relecture (même relecteur) : corrections 1–6 confirmées ; un nouveau point bloquant — un sujet dont toutes les associations mesurées viennent d'un parent malade ou dépendant est traité comme « limité » par le garde-fou — appliqué ; parentification devient Anxiété, Dépression, Rôle de sauveur, Codépendance, Relations proches. Verdict final : « acceptable avec réserves » sous cette condition, remplie. 1re carte reposant seulement sur une transmission familiale : 0,0 % ; trouble grave sans lien solide dans le top : 0 %. À transmettre en plus : harcèlement → anxiété sociale (« limitée ») paraît sous-évalué.
Non bloquant, non fait : santé physique après les sujets psychologiques dans « Autres » ; plafond global de spécificité ; libellé « Névrosisme » ; mention adulte pour l'anxiété de séparation ; regroupement du repli long ; nombre de vécus reliés sur la carte.

### Tests
- Nouveaux : `journal/tests/v5-C5a-cartographie.test.mjs` (parité page/Worker sur 2 000 profils avec retours : sections, ordre, pertinence ; partition ; garde-fou ; bornes de simulation ; « Souvent » ; familles ; contrat résumé / bloc / index) ; `tests/v5-C-cartographie-browser.cjs` (390 et 1 440 px : sections, phrase, cartes fréquentes, replis, focus après « Je me reconnais » et « Pas moi », pas de défilement horizontal, ni 3114 ni filtres).
- Adaptés (devenus faux) : `journal/tests/cartographie.test.mjs` (contextual vide, parité avec retours et sections), `journal/tests/v4-C-cartographie.test.mjs` (plus de contextuels, ordre `tie/measured/model`, nouveau texte), `tests/cartographie.test.cjs`, `tests/cartographie-browser.cjs`, `tests/v4-C-cartographie-browser.cjs`.

## Reste / hors lot
- Test A5 `v5-A5-sources.test.mjs` « E: a point keeps … map reduced first » : échoue depuis les réductions demandées par A5 (plus de place → tous les moments gardés, y compris d'importance 1) ; il passe sans ces réductions. À adapter par A5 (par ex. vérifier que les moments omis ont une importance ≤ à celle des moments gardés).
- `cartographyIndexEntries` est branché par A5-P0 dans `ambre-engine.mjs` ; `cartographyWords` de `chat-backend.mjs` reste inutilisée (à retirer par A5).
- Décisions du propriétaire (conception § 5) toujours ouvertes : acceptation de la baisse de dépression/anxiété en tête ; notes anciennes sur un sujet devenu invisible (laissées comme les sujets retirés : lues par Ambre, non visibles sur la page).
- Points « à transmettre » ci-dessus pour C5b ; `content/cartographie.md` inchangé.

## Relecture finale (coordinateur) — appliquée
- Données sur des sujets masqués (contextuels ou retirés) : toujours stockées et validées, mais plus transmises à Ambre (exclues de `recognized`, `dismissed`, des notes de la personne et d'Ambre, des preuves, de l'index et du bloc ; `suicide_recognized` vaut toujours false) ; `chooseAmbreNotes` refuse ces sujets. Tests adaptés : `cartographie.test.mjs`, `ambre-engine.test.mjs` (contextuel avec ancien retour : plus indexé), `v5-C5a-cartographie.test.mjs`.
- Phrase : « Classés selon la solidité des études et ce qui est le plus propre à vos réponses ; ce n'est pas une mesure de votre risque. »
- Cibles tactiles ≥ 24 px (titres des replis, « Retirer », « Voir le détail ») ; pastilles et méta à 12 px.
- Réponse prudente non décomptée seulement si le fournisseur a échoué aux deux étapes (`analysis.failure_code` enregistré par le moteur, codes délai/erreur/HTTP) ; test `journal/tests/v5-C5a-quota.test.mjs`.
- `cartographyWords` retirée de `chat-backend.mjs`.
