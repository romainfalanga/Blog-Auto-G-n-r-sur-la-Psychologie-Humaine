# Décode ton esprit

Site : https://decodetonesprit.fr. Pages personnelles : `/mon-portrait/`, `/mes-personnages/` et `/mon-compte/`. Les anciens liens `/mes-histoires/` restent valides.

L'accueil conserve les 231 articles et 20 personnages publics. Une barre de recherche combine mots-clés, formulations proches et suggestions ; les cartes restent stables à la fermeture des suggestions.

## Personnalisation

Dans **Mon portrait**, chacun peut aimer des lectures, préciser ce qui lui ressemble ou non, décrire ses proches et leurs relations, et ajouter des événements à une ligne de vie. Les dates peuvent être approximatives. Toutes ces informations sont facultatives ; un contexte personnel explicite suffit à démarrer la personnalisation sans like. Un nom ou un rôle seul ne suffit pas.

Le portrait distingue déclarations, hypothèses et intérêts de lecture. Ses éléments sont rattachés aux sources originales et à leur révision. Une correction ou suppression invalide les anciennes déductions ; la fiction ne devient jamais une biographie réelle. Les recommandations exploitent séparément résonance, différences, préférences de lecture et sujets à éviter.

**Mes personnages** propose trois identités fictives stables avec des histoires indépendantes. L'abonné choisit celle qu'il souhaite continuer. Sept histoires sont disponibles par semaine civile, du lundi à minuit au lundi suivant en Europe/Paris, partagées entre les trois personnages, sans report. La demande réserve un crédit ; seule la publication le consomme. Les anciennes histoires et leurs retours sont conservés.

Le tarif prévu reste 10 EUR/mois et le seuil de nouvelle souscription reste dix articles aimés. Le paiement dépend de la configuration Stripe ; ne pas confondre une interface publiée avec un encaissement activé.

## Architecture

Hugo/Netlify sert le catalogue public. Cloudflare Worker/D1 gère comptes, sources personnelles, personnages, crédits et histoires privées. GitHub Actions traite la file toutes les quinze minutes ; il ne crée aucun récit sans demande. Les retards du planificateur sont possibles.

La clé `OPENROUTER_API_KEY` reste un secret Cloudflare. Les Actions s'authentifient par OIDC auprès d'une passerelle qui vérifie versions, bail, droits et budgets avant chaque appel. Les textes privés ne sont écrits ni dans Git, ni dans les articles publics, ni dans les logs Actions.

## Validation

```sh
hugo --minify
node --test tests/*.test.cjs
npm --prefix journal test
python -m unittest discover -s journal/tests -p 'test_personal*.py'
node journal/tests/evaluate-recommendations.mjs
```

Voir [le guide backend](journal/README.md), [les contrats V2](docs/contrats-implementation-v2.md), [les résultats algorithmiques et leurs limites](docs/implementation-algorithmique-v2.md) et [la recette de mise en ligne](docs/suivi-implementation-v2.md).
