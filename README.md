# Décode ton esprit

Site : https://decodetonesprit.fr. Pages personnelles : `/mon-portrait/`, `/mes-proches/`, `/ma-ligne-de-vie/`, `/mon-personnage/` et `/mon-compte/`. Les anciens liens `/mes-histoires/` et `/mes-personnages/` restent valides.

L'accueil conserve les 231 articles et 20 personnages publics. Une barre de recherche combine mots-clés, formulations proches et suggestions ; les cartes restent stables à la fermeture des suggestions.

## Personnalisation

Dans **Mon portrait**, chacun se présente avec ses propres mots : perception de soi, émotions, schémas récurrents et contexte actuel. Le questionnaire d’identité est facultatif. Les déclarations alimentent le portrait et les recommandations ; nom, prénom et date de naissance exacte restent hors des prompts IA. L’âge et le genre servent de contexte, jamais de preuve psychologique. **Mes proches** et **Ma ligne de vie** sont deux pages autonomes. Les dates des événements peuvent rester approximatives.

Le portrait distingue déclarations, hypothèses et intérêts de lecture. Ses éléments sont rattachés aux sources originales et à leur révision. Une correction ou suppression invalide les anciennes déductions ; la fiction ne devient jamais une biographie réelle. Les recommandations exploitent séparément résonance, différences, préférences de lecture et sujets à éviter.

**Mon personnage** propose une seule identité fictive stable et une histoire suivie. Son démarrage nécessite un abonnement actif, dix articles aimés, au moins un texte de présentation personnelle et un portrait analysé à jour. Sept histoires sont disponibles par semaine civile, du lundi à minuit au lundi suivant en Europe/Paris, sans report. La demande réserve un crédit ; seule la publication le consomme. Les personnages secondaires de l’ancienne version restent en archives, avec leurs histoires et retours ; ils ne génèrent plus de nouveaux épisodes.

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
