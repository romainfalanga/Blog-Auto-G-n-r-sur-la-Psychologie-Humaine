# Lot A5-P0 — avancement

## Fait
- Référence de budget figée avant tout changement (arbre 9bef77c) sur le profil rempli de l'audit v5 : `journal/tests/v5-A5-budget.test.mjs` (harnais `journal/tests/v5-A-helper.mjs`). Avant : analyse ≈ 2 400, rédaction 5 940–6 320, relecture ≈ 2 450 jetons ; bilan initial 12 423, point 12 525. Après : tour total +0 à +4,4 % (rédaction au plus +7,5 % en stratégie « idéale », car le budget des extraits est désormais rempli) ; bilan initial +0,2 %, point −1,3 %.
- A : `last_review` (dernier bilan prêt, daté, début + fin, 900 car.) dans l'entrée de rédaction ; payé par les extraits (budget commun 6 500, détails ≈ 5 600 → réellement 3 500–4 300 après règles de fond et changement). Exclu des détails. Consigne d'une phrase. Trace : `review_chars` dans le bloc « Sources pertinentes » ; Paramètres (`stored_elsewhere.reviews`, `usage_policy`) mis à jour.
- B : `selectDetails` suit l'ordre de l'analyse puis le classement lexical ; extraits début + fin de 700 car. ; un détail moins prioritaire est raccourci ou retiré, jamais un plus prioritaire ; budget calculé avant sélection. `analysisIndex` propose toujours les rubriques du portrait et les parties de la synthèse (≤ 10 lignes, plafond 2 600 inchangé).
- C : une entrée d'index par rubrique (`profile:self_description` … `profile:occupation`, libellés du formulaire) ; bilans : une source par rubrique (contexte actuel et émotions d'abord). Empreintes de sources inchangées.
- D : `portrait:axes`, `portrait:current_state`, `portrait:evolution` ; pistes avec `next_question` et contre-indices en tête ; bilans : axes, état actuel, évolution.
- E : bilans — Cartographie bornée par `cartographyPromptBlock(summary, 9000)` (≈ 9 500 au plancher du bloc, contre 16–20 Ko), puis bilan précédent 2 000, extraits par paliers 600 → 420 → 300, rubriques secondaires, moments > 8 et proches > 6, part de 30 % réservée aux récits ; moments classés par importance. Mesuré : initial 6/6 proches, 12/12 moments, 6 rubriques ; point 6 proches, 8 moments, contexte actuel + émotions, 6 échanges.
- G : âge au moment calculé (`private-context.mjs`, non stocké, « vers » si année seule), importance, prénoms des proches liés, rôle en français ; fin du récit et sens aujourd'hui préservés (extrait par champ).
- C5a branché : `cartographyIndexEntries` dans `knowledgeIndex` (sections, plus de contextuels), consigne de cartographie effective (sections), message d'accueil (reconnus puis `sections.top`).
- Réponse prudente en double panne (analyse + vérification) : non décomptée (`error_code='uncounted_cautious_reply'`, sans migration ; plafond 200/semaine conservé).
- README et `journey.test.mjs` (nouveau parcours sur le moteur de production ; l'ancien test marqué « legacy »).
- Tests : `v5-A5-budget` (10), `v5-A5-sources` (12) ; tests existants adaptés : `ambre-engine.test.mjs` (politique 5 600, `last_review`, contextuels hors index).

## Reste / hors lot
- H reporté à P2. P1 remplacera `last_review` par le fil des bilans.
- Dernier passage : `npm --prefix journal test` 722/722, `node --test tests/*.test.cjs` 39/39, `--check` de `verify.yml` réussis (les deux échecs transitoires du lot C5a ont disparu).
