# Décode ton esprit

[Décode ton esprit](https://decodetonesprit.fr) est un espace personnel. Le menu comprend Ambre (`/ambre/`), Ma ligne de vie, Mes proches, Mon enfance (`/cartographie/`), Mes symptômes (`/mes-symptomes/`), Mes sujets, Mes bilans, Mes diagnostics et Mon compte. L’adresse `/` redirige vers Ambre.

Le catalogue public (personnages et articles) a été retiré en novembre 2026 : ses anciennes adresses (`/personnages/…`, `/posts/…`) redirigent vers `/ambre/`. Le retrait des restes côté Worker (exemples, articles aimés, recommandations) fait l’objet d’un lot séparé.

Pour reprendre le travail : lire [les consignes du projet](AGENTS.md) et [la note de reprise](docs/reprise-2026-11-19-codex.md).

## Personnalisation

Dans **Mon portrait**, chacun se présente avec ses propres mots : perception de soi, émotions, schémas récurrents et contexte actuel. Le questionnaire d’identité est facultatif. Les déclarations alimentent le portrait et les recommandations ; le prénom déclaré personnalise les échanges, tandis que le nom de famille et la date de naissance exacte restent hors des prompts IA. L’âge et le genre servent de contexte, jamais de preuve psychologique. **Mes proches** et **Ma ligne de vie** sont deux pages autonomes. Les dates des événements peuvent rester approximatives.

Le portrait distingue déclarations, hypothèses et intérêts de lecture. Ses éléments sont rattachés aux sources originales et à leur révision. Une correction ou suppression invalide les anciennes déductions ; la fiction ne devient jamais une biographie réelle. Les recommandations exploitent séparément résonance, différences, préférences de lecture et sujets à éviter.

Le mode de test `FREE_ACCESS_ENABLED=true` permet actuellement d’utiliser la plateforme sans paiement. Le tarif configuré pour une future activation est de 29 EUR/mois après sept jours gratuits. Le paiement dépend de la configuration Stripe.

**Ambre** propose une conversation IA, avec 100 messages utilisateur ayant reçu une réponse par semaine. La présentation reste le premier message des conversations, y compris les fils existants. Elle ne consomme aucun message. L’envoi apparaît immédiatement, avec un indicateur d’attente puis une révélation progressive de la réponse validée. La dictée française du navigateur affiche un texte modifiable, sans envoi automatique ni conservation d’audio par l’application ; sa disponibilité et son service vocal dépendent du navigateur.

Les propositions de modification du portrait, des proches et de la ligne de vie sont sourcées et modifiables avant acceptation. Les capacités sont calculées côté serveur puis revérifiées à la confirmation. Avec l’accord de conversation Ambre, des observations relationnelles explicites sur un proche déjà enregistré peuvent être consignées automatiquement. Ces citations apparaissent dans sa cartographie et restent supprimables. Ambre peut proposer des articles déjà publiés et relier leur rubrique pour les proches à ce que la personne vient de partager. Chaque réponse contient au maximum deux questions.

La cartographie présente d’abord les pistes personnelles, puis leurs éléments d’appui, contre-indices, autres explications et questions. L’exploration des 161 repères reste accessible séparément. Ces repères ne constituent ni des diagnostics ni des probabilités cliniques validées.

## Architecture

Hugo/Netlify sert Mon enfance (`/cartographie/`), Mes symptômes (`/mes-symptomes/`) et la page 404, et relaie les autres pages vers le Worker. Cloudflare Worker/D1 (`journal/`) gère les comptes, les sources personnelles, la conversation et l’analyse du portrait. GitHub Actions relaie l’analyse du portrait pour les contextes longs. Les retards du planificateur sont possibles.

La clé OpenRouter d’Ambre reste un secret Cloudflare. Pour l’analyse des portraits, les Actions s’authentifient par OIDC auprès d’une passerelle qui vérifie versions, bail, droits et budgets avant chaque appel. Les textes privés ne sont écrits ni dans Git, ni dans les articles publics, ni dans les logs Actions.

## Validation

```sh
node scripts/build_navigation.mjs --check
node scripts/build_cartographie_module.mjs --check
python scripts/build_cartographie.py --check
python scripts/build_symptomes_carte.py --check
hugo --minify
HUGO_BIN=<chemin de hugo> node --test tests/*.test.cjs
npm --prefix journal test
python -m unittest discover -s journal/tests -p 'test_personal*.py'
python -m unittest discover -s journal/tests -p 'test_symptomes*.py'
```

Voir [le plan et les décisions V4](docs/plan-accompagnement-v4.md), [le guide backend](journal/README.md), [les contrats V2](docs/contrats-implementation-v2.md), [les résultats algorithmiques et leurs limites](docs/implementation-algorithmique-v2.md) et [la recette de mise en ligne](docs/suivi-implementation-v2.md).

V10 : [plan, revue et validation](docs/plan-ambre-v10.md), [contrat de capacités](docs/plan-capacites-v10.md), [interface](docs/plan-interface-v10.md).

[Plan de refonte de l’espace personnel](docs/plan-espace-ambre-v14.md).

[Plan de navigation, proches et parcours cliniques](docs/plan-v17.md).
