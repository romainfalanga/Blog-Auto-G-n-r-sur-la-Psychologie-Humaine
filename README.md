# Décode ton esprit

[Décode ton esprit](https://decodetonesprit.fr) réunit le catalogue public et l’espace personnel. Cet espace comprend `/mon-accompagnement/`, `/mon-portrait/`, `/mes-proches/`, `/ma-ligne-de-vie/` et `/mon-compte/`. Les anciennes adresses de récits personnalisés redirigent vers Ambre. Les données historiques restent dans l’export privé du compte.

Le catalogue réunit 531 articles et 45 personnages publics. Aux 387 récits des 33 personnages d’origine s’ajoutent 12 parcours cliniques de 12 épisodes. Chaque article comporte une rubrique « Du côté des proches » liée à sa scène. La recherche combine mots-clés et formulations proches, sans propositions à l’ouverture du champ. Les cartes affichent les repères psychologiques de chaque parcours.

## Personnalisation

Dans **Mon portrait**, chacun se présente avec ses propres mots : perception de soi, émotions, schémas récurrents et contexte actuel. Le questionnaire d’identité est facultatif. Les déclarations alimentent le portrait et les recommandations ; le prénom déclaré personnalise les échanges, tandis que le nom de famille et la date de naissance exacte restent hors des prompts IA. L’âge et le genre servent de contexte, jamais de preuve psychologique. **Mes proches** et **Ma ligne de vie** sont deux pages autonomes. Les dates des événements peuvent rester approximatives.

Le portrait distingue déclarations, hypothèses et intérêts de lecture. Ses éléments sont rattachés aux sources originales et à leur révision. Une correction ou suppression invalide les anciennes déductions ; la fiction ne devient jamais une biographie réelle. Les recommandations exploitent séparément résonance, différences, préférences de lecture et sujets à éviter.

Le mode de test `FREE_ACCESS_ENABLED=true` permet actuellement d’utiliser la plateforme sans paiement. Le tarif configuré pour une future activation est de 29 EUR/mois après sept jours gratuits. Le paiement dépend de la configuration Stripe.

**Ambre** propose une conversation IA, avec 100 messages utilisateur ayant reçu une réponse par semaine. La présentation reste le premier message des conversations, y compris les fils existants. Elle ne consomme aucun message. L’envoi apparaît immédiatement, avec un indicateur d’attente puis une révélation progressive de la réponse validée. La dictée française du navigateur affiche un texte modifiable, sans envoi automatique ni conservation d’audio par l’application ; sa disponibilité et son service vocal dépendent du navigateur.

Les propositions de modification du portrait, des proches et de la ligne de vie sont sourcées et modifiables avant acceptation. Les capacités sont calculées côté serveur puis revérifiées à la confirmation. Quand la mémoire des repères est activée dans Mes proches, Ambre peut aussi consigner automatiquement des observations relationnelles rapportées sur un proche déjà enregistré. Ces citations apparaissent dans sa cartographie et restent supprimables. Ambre peut proposer des articles déjà publiés et relier leur rubrique pour les proches à ce que la personne vient de partager. Chaque réponse contient au maximum deux questions.

La cartographie présente d’abord les pistes personnelles, puis leurs éléments d’appui, contre-indices, autres explications et questions. L’exploration des 161 repères reste accessible séparément. Ces repères ne constituent ni des diagnostics ni des probabilités cliniques validées.

## Architecture

Hugo/Netlify sert le catalogue public. Cloudflare Worker/D1 gère les comptes, les sources personnelles, la conversation et l’analyse du portrait. GitHub Actions relaie l’analyse du portrait pour les contextes longs. Les retards du planificateur sont possibles.

La clé OpenRouter d’Ambre reste un secret Cloudflare. Pour l’analyse des portraits, les Actions s’authentifient par OIDC auprès d’une passerelle qui vérifie versions, bail, droits et budgets avant chaque appel. Les textes privés ne sont écrits ni dans Git, ni dans les articles publics, ni dans les logs Actions. L’action éditoriale traite uniquement des personnages fictifs et lit son propre secret GitHub Actions lorsqu’elle est lancée en mode brouillon.

## Validation

```sh
node scripts/build_navigation.mjs --check
python scripts/check_editorial_pack.py
python scripts/integrate_editorial_pack.py --check
python scripts/clinical_editorial.py check
hugo --minify
node --test tests/*.test.cjs
npm --prefix journal test
python -m unittest discover -s journal/tests -p 'test_personal*.py'
node journal/tests/evaluate-recommendations.mjs
```

Voir [le plan et les décisions V4](docs/plan-accompagnement-v4.md), [le guide backend](journal/README.md), [les contrats V2](docs/contrats-implementation-v2.md), [les résultats algorithmiques et leurs limites](docs/implementation-algorithmique-v2.md) et [la recette de mise en ligne](docs/suivi-implementation-v2.md).

V10 : [plan, revue et validation](docs/plan-ambre-v10.md), [audit de couverture](docs/couverture-v10-proposition.md), [contrat de capacités](docs/plan-capacites-v10.md), [interface](docs/plan-interface-v10.md), [manifeste des 156 récits](data/editorial_pack_v10.json).

[Plan de refonte de l’espace personnel](docs/plan-espace-ambre-v14.md).

[Plan de navigation, proches et parcours cliniques](docs/plan-v17.md) · [Cartographie clinique](docs/cartographie-clinique-v17.md).
