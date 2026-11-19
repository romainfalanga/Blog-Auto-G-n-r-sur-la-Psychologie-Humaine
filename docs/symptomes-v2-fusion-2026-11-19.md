# Banque Mes symptômes v2 — fusion et génération

## Contrat livré

`python scripts/build_symptomes_carte.py` produit `data/symptomes_carte.json` et `journal/src/symptomes-carte.generated.mjs` (export `SYMPTOMES_CARTE`). `--check` vérifie les fichiers sans écrire. Le schéma reprend groups/factors/outcomes/links de Mon enfance ; `ranking.version=essential_share_desc`, `requires` associe chaque sujet produit à la liste des facteurs d'usage qui doivent tous être Oui. Les facteurs portent `overlap_group` et `safety`. Les informations de sujet communes sont matérialisées depuis Mon enfance et ne sont jamais recopiées manuellement dans cette banque.

544 questions étaient présentes avant fusion. Les 34 fusions explicites et justifiées de `fusion.json` regroupent des descriptions concordantes ; les variantes de durée, de contexte, de modalité et les symptômes différents d'une même famille restent distincts. Trois questions neutres d'usage (alcool, cannabis, tabac/nicotine) complètent les dix portes produit existantes. Résultat initial : **513 questions, 157 sujets, 1 124 liens**, répartis dans **28 thèmes de 5 à 30 questions**. Ce volume remplace l'estimation préalable 220–260 : supprimer la moitié des manifestations n'aurait pas satisfait la demande de couverture. Les thèmes sont divisés par manifestations, notamment travail/études/argent, sommeil, apparence, sécurité des relations, mouvements/sens et produits utilisés. Les affectations sont explicites dans `presentation.json`, sans règle heuristique dans le build.

## Intégrité

Le générateur refuse identifiants dupliqués, sources inconnues ou contradictoires par URL, sujets inexistants ou exclus, poids inconnus, liens sans justification/source, aides dépassant 120 caractères, plus de deux exemples, signes de sécurité absents, exigences vers une question inexistante, fusion non justifiée ou sécurité différente, thèmes de plus de 30 questions et sujets hors 3–12 liens critere. Une exception précise et sourcée dans `criterion-exceptions.json` autorise deux critères pour le trouble de l’adaptation (CDDR p. 352) : préoccupation liée au stresseur et difficulté d’adaptation. Le nombre exact attendu est vérifié ; aucune dérogation globale. Ce plafond porte sur les liens, pas sur les familles indépendantes : certaines manifestations alternatives doivent partager leur famille.

Lorsqu'une fusion rapproche deux liens vers un même sujet, la politique explicitement consignée conserve le poids le plus fort et toutes les sources. Les désaccords cliniques doivent être résolus dans les sources éditoriales avant fusion, jamais par une promotion automatique supposée validée. La relecture a déjà corrigé la fugue (sous-forme, non caractéristique générale de l'amnésie) et la désorganisation forte du discours pour le trouble schizotypique. Elle a regroupé les alternatives du pica et les variantes de régurgitation dans leurs familles respectives.

Le module Worker conserve libellés, descriptions, exemples, familles, poids et résumés des liens ; il omet les notices de sources et le champ éditorial `basis`, disponibles intégralement dans le JSON public et les sources versionnées. Taille initiale après corrections : **434 358 octets**, sous le plafond de 500 000. Aucun fichier de Mon enfance n'est modifié par ce générateur.

## Vérification et limites

Tests dédiés : `python -m unittest discover -s journal/tests -p 'test_symptomes_carte_build.py'`. Ils contrôlent déterminisme, partition complète, non-modification du jeu enfance, portes produit, métadonnées communes et rejets de données invalides.

La validation du schéma ne vaut pas validation clinique. La relecture indépendante des sources est suivie séparément par l'agent clinique ; son rapport final et ses corrections font foi. Les décisions de périmètre de `DECISIONS.md` restent applicables. Aucun seuil de cette banque ne constitue un diagnostic ou une probabilité clinique calibrée.
