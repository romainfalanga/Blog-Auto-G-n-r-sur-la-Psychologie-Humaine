# Consignes communes aux rédacteurs de domaine — « Mes symptômes » v2

Référence : `docs/plan-mes-symptomes-v2-2026-11-19.md`, section « Version 2.1 » (fait foi). Lot S0 : `themes.json`, `canoniques.json`, `attribution.json`.

## 1. Ce que vous produisez
Un fichier `editorial/symptomes/v2/liens-<domaine>.json`, pour les seuls sujets que `attribution.json` vous attribue :

```json
{"domain": "<domaine>",
 "symptoms": [{"id": "…", "canonical": false, "theme": "<id de themes.json>",
   "label": "…", "description": "…", "examples": ["…"],
   "safety": null, "overlap_group": "…",
   "links": [{"outcome": "<id cartographie>", "strength": "critere|moderee|limitee",
              "basis": "…", "summary": "…", "sources": ["<id source>"]}]}],
 "links_to_canonical": [{"symptom": "<id canonique>", "outcome": "<id cartographie>",
   "strength": "critere|moderee|limitee", "basis": "…", "summary": "…", "sources": ["…"]}],
 "sources": [{"id": "…", "type": "classification|etude|revue|manuel", "citation": "…", "url": "https://…", "licence": "…"}]}
```

- Sujets référencés par identifiant uniquement (jamais de libellé, description ou URL de sujet recopiés : le build les tire de `data/cartographie.json`).
- Aucun lien vers un sujet exclu (`attribution.json` → `excluded`) ni vers un sujet d'un autre domaine. Si un symptôme concerne un autre domaine, signalez-le dans votre rapport ; le dédoublonnage s'en charge.
- Les points de `attribution.json` → `to_decide` vous concernant sont tranchés dans votre rapport (choix et raison), pas en silence.

## 2. Canoniques : réutiliser, ne pas recréer
- Avant de créer un symptôme, cherchez son équivalent dans `canoniques.json`. S'il existe, ajoutez seulement un lien dans `links_to_canonical` (même structure qu'un lien, plus `symptom`).
- Ne modifiez jamais un canonique (libellé, description, thème, groupe) et ne le redéclarez pas dans `symptoms`. Une reformulation souhaitée se propose dans le rapport.
- Vos propres symptômes portent `canonical: false`. Identifiants en snake_case, stables, sans préfixe de domaine, uniques dans tout le jeu.

## 3. Poids des liens
- `critere` : caractéristique essentielle du sujet dans la CIM-11 (CDDR, « essential features ») ou, pour un sujet hors CIM-11 (tendance, schéma, attachement, trait), élément de sa définition dans la source qui le définit.
- `moderee` : caractéristique associée documentée par une source (CDDR « additional features », étude, revue, manuel du modèle).
- `limitee` : symptôme fréquent mais non spécifique (présent dans beaucoup de troubles ou en population générale).
- Chaque sujet attribué reçoit **3 à 12 liens `critere`** (canoniques compris), répartis sur des familles `overlap_group` distinctes autant que possible : le classement compte les familles, pas les questions.
- `basis` dit d'où vient le poids (« CDDR 6B01, caractéristique essentielle », « définition de Young, 2003 », « méta-analyse X ») ; `summary` explique en une phrase simple pourquoi ce signe se rattache au sujet, sans affirmer de diagnostic.
- Un poids douteux se descend d'un cran, jamais l'inverse.

## 4. Familles (`overlap_group`)
Un même signe clinique formulé de plusieurs façons partage une famille (ex. les trois questions de sommeil des canoniques forment `sommeil`). Rattachez votre symptôme à une famille canonique s'il en est une variante ; sinon donnez une famille nouvelle (par défaut son identifiant). Ne regroupez pas deux signes distincts d'un même sujet : cela réduirait artificiellement C.

## 5. Formulation
- Français simple, première personne implicite, forme infinitive ou nominale courte : « Avoir du mal à s'endormir », « Éviter des situations par peur ».
- `description` ≤ 120 caractères, concrète ; au plus 2 `examples` courts tirés de la vie quotidienne.
- Période unique : **« ces 12 derniers mois »**. Ne mettez pas d'autre période dans le libellé ; une durée propre à la définition (« presque tous les jours », « depuis au moins deux ans ») peut figurer dans la description.
- Ton non jugeant : pas de « trop », « anormal », « malade », « faute » ; décrire ce qui arrive, pas ce que la personne est. Pour les violences, la responsabilité revient à l'auteur ; les violences subies et exercées sont deux questions distinctes.
- Schémas, attachements, traits, tendances, modes : formuler des **manifestations observables** (pensées, réactions, gestes, situations), jamais le nom du concept ni une auto-étiquette (« Avoir un schéma d'abandon », « Être codépendant » sont refusés).
- Une question = un seul signe ; pas de double négation ; pas de question qui suppose la réponse.
- Réponses Non / Oui / Je ne sais pas : la question doit pouvoir y répondre.

## 6. Sécurité
- `safety` vaut `"suicide"` pour les idées de mort ou de suicide, `"automutilation"` pour se faire du mal volontairement, sinon `null`. Les deux canoniques `idees_de_mort` et `se_faire_du_mal` existent : ne créez pas de doublon, liez-les.
- Aucun texte d'alerte dans vos fichiers : la ligne avec le 3114 est gérée par l'interface.

## 7. Licences et sources
- **CDDR (OMS 2024) : CC BY-NC-ND 3.0 IGO**, vérifiée dans `editorial/symptomes/licences/cddr2024-licence-extrait.txt` : « This licence does not allow you to produce adaptations of the work (including translations) without permission ». Donc paraphrase originale courte, indépendante de l'ordre et des tournures du texte ; aucune traduction suivie, aucune liste de critères recopiée, citation de la source.
- **CIM-11 MMS : CC BY-ND 3.0 IGO** (`icd11-license.txt`) : codes et intitulés cités à l'identique seulement.
- **DSM : non réutilisé** (ni critères, ni seuils, ni formulations).
- Items de questionnaires (PHQ, GAD-7, ITQ…) : pas de recopie dans ce jeu ; vos libellés sont originaux.
- Chaque source porte une `url` vérifiable (page OMS, PubMed, DOI, PMC) et une `licence` ou mention de droits ; seules les sources effectivement lues sont citées. Identifiants de sources déjà employés : `oms_cddr`, `icd11_mms` (voir `canoniques.json`) ; réutilisez-les tels quels.

## 8. Avant de rendre
- JSON valide, identifiants uniques, thèmes existants, sujets attribués tous couverts avec 3 à 12 liens `critere`.
- Visez 15 à 25 questions par thème une fois tous les domaines réunis : signalez un thème que vous chargez fortement.
- Rapport court : nombre de symptômes créés, liens par sujet (critere/moderee/limitee), points tranchés, doutes pour la relecture indépendante.
