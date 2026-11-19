# « Mes symptômes » : recherche et conception — 19 novembre 2026

Document de recherche et de conception. **Aucun code modifié, rien commité ni déployé.** Rédigé au fil de l'eau : les sections se complètent à mesure que les sources sont ouvertes. Proposition de données : `editorial/symptomes/proposition-2026-11-19.json`.

Règles de lecture :
- « lu » = page officielle ou article ouvert et lu en entier ; « résumé » = seul le résumé, la notice ou un extrait a été lu ; « non ouvert » = information non vérifiée ici, à confirmer avant usage.
- Aucun item d'instrument n'est recopié dans ce document ou dans la proposition JSON, sauf mention explicite d'une licence lue qui le permet. Les items sont désignés par leur numéro et leur thème.
- Les seuils sont des **repères publiés**, jamais des diagnostics.
- Règle absolue héritée du plan du 17 novembre (§ 0) : aucun lien n'est créé sans source ; une absence de preuve est un résultat valide. Ici, aucun croisement « symptôme ↔ sujet » n'ajoute de lien vécu → sujet ; il ne fait que mettre en regard deux déclarations de la personne.

## 1. État de départ lu dans le dépôt

- `data/cartographie.json` : 113 sujets (`outcomes`), 7 domaines d'affichage (`perceptions`, `humeur`, `anxiete`, `relations`, `soi`, `emotions`, `comportements`), 4 natures (`trouble`, `schema`, `attachement`, `tendance`). Liste complète utilisée pour le tableau de correspondance (§ 5).
- Stockage existant : table `cartographies` (migration `0016`), une ligne par membre, `answers`, `feedback`, `consent` obligatoire (données de santé), `locked_at`, `version`.
- Ambre : `journal/src/ambre-engine.mjs` importe `cartographyIndexEntries` (`journal/src/cartographie.mjs`) ; résumé borné (`summarizeCartography`, `maxConvergence=15`, `maxPistes=10`) ; sécurité déterministe dans `chat-safety.mjs` (3114, 15, 112, 17) et vérification de risque sans index ni catalogue de troubles.
- Mon enfance : réponses Non / Oui / Je ne sais pas ; niveaux de preuve et codes CIM/DSM retirés du rendu (reprise du 17 novembre). La même discipline d'affichage est proposée ici.

