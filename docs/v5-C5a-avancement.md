# Lot C5a — avancement (Cartographie : contextuels, classement L avec garde-fou, affichage)

## Contrat figé fourni par `journal/src/cartographie.mjs` (pour A5-P0)

Écrit en premier ; A5-P0 l'appelle sans le modifier.

### `computeCartography(answers, data = CARTOGRAPHIE, feedback = {})`
Renvoie `{results, sections, aceCount, selectedCount}`.
- `results` : sujets visibles (ni `retired`, ni `contextual_only`) dans l'**ordre affiché** (sections concaténées dans l'ordre ci-dessous). Chaque élément garde les champs historiques (`outcome`, `links`, `factors`, `associated_factors`, `associated_count`, `other_links`, `tier`, `score`, `evidence_groups`, `relation_types`) et ajoute `relevance` (entier, jamais affiché ni transmis), `section`, `frequent` (booléen), `max_points` (12/8/4/0 : meilleur lien mesuré), `has_empirical`.
- `sections` : `{recognized, top, frequent, other, dismissed}`, cinq tableaux d'éléments de `results`, partition exacte :
  - `recognized` : retour « oui » ;
  - `top` : les 5 premiers sujets sans retour (`SECTION_TOP = 5`) ;
  - `frequent` : sujets `frequent` ayant au moins un lien mesuré « forte » pour ce profil, hors des trois autres sections (« Aussi associés à la plupart des vécus difficiles ») ;
  - `other` : les suivants ;
  - `dismissed` : retour « non ».
  Ordre d'affichage = `recognized, top, frequent, other, dismissed` (rang conservé dans chaque section).
- Valeurs de `section` : `recognized | top | frequent | other | dismissed` (même nom que la clé).

### `summarizeCartography(...)` (inchangé dans sa signature)
Ajoute `sections: {recognized:[], top:[], frequent:[], other:[], dismissed:[]}` (éléments compacts `{id,label,kind,section,frequent,max_strength,...}`, `other` plafonné à 20), `display_order` (10 premiers avec `section`, `frequent`, sans score), `display_count` (sans contextuels). `contextual: []` est conservé (toujours vide) pour la compatibilité.

### `cartographyPromptBlock(summary, budget = Infinity)`
Bloc compact transmis au modèle (remplace `cartographyPrompt` + `cartographyWords` de `chat-backend.mjs`, qui délègue désormais). Sans URL publique. Si `budget` (longueur JSON maximale en caractères) est donné, réduit dans l'ordre : notes d'Ambre les plus anciennes, pistes, exploratoires, convergences non reconnues au-delà de 3, preuves, sources, notes de la personne au-delà de 4. Ne retire jamais `recognized`, `sections.top`, `sections.frequent`. Renvoie `null` sans carte.

### `cartographyIndexEntries(summary)`
Tableau d'entrées pour l'index d'Ambre : `{key, kind, label, content, updated_at, metadata}` (mêmes clés qu'aujourd'hui : `cause:<id>`, `cartography:negative:<id>`, `cartography:outcome:<id>`, `cartography:resource:<id>`, `note:<i>`). Les sujets suivent l'ordre des sections ; aucun sujet contextuel sauf retour ancien (marqué `contextual_only`). `metadata.section` indique la section. A5-P0 les passe à son `add(key, kind, label, content, updated_at, metadata)`.

## Fait
(voir ci-dessous, mis à jour au fil du lot)

## En cours / reste
