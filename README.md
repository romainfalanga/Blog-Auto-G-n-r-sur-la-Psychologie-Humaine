# Décode ton esprit

Site : https://decodetonesprit.fr. Pages personnelles : `/mon-accompagnement/`, `/mon-portrait/`, `/mes-proches/`, `/ma-ligne-de-vie/`, `/mon-personnage/` et `/mon-compte/`. Les anciens liens `/mes-histoires/` et `/mes-personnages/` restent valides.

Le catalogue réunit 387 articles et 33 personnages publics. Les 231 récits antérieurs sont conservés ; les 13 nouveaux parcours comportent chacun 12 récits rédigés directement, sans appel au générateur d’articles. La recherche combine mots-clés et formulations proches, sans propositions à l’ouverture du champ. Les cartes affichent les repères psychologiques de chaque parcours.

## Personnalisation

Dans **Mon portrait**, chacun se présente avec ses propres mots : perception de soi, émotions, schémas récurrents et contexte actuel. Le questionnaire d’identité est facultatif. Les déclarations alimentent le portrait et les recommandations ; le prénom déclaré personnalise les échanges, tandis que le nom de famille et la date de naissance exacte restent hors des prompts IA. L’âge et le genre servent de contexte, jamais de preuve psychologique. **Mes proches** et **Ma ligne de vie** sont deux pages autonomes. Les dates des événements peuvent rester approximatives.

Le portrait distingue déclarations, hypothèses et intérêts de lecture. Ses éléments sont rattachés aux sources originales et à leur révision. Une correction ou suppression invalide les anciennes déductions ; la fiction ne devient jamais une biographie réelle. Les recommandations exploitent séparément résonance, différences, préférences de lecture et sujets à éviter.

**Mon personnage** propose une seule identité fictive stable et une histoire suivie. Son démarrage nécessite un abonnement actif ou un essai valide, dix articles aimés, au moins un texte de présentation personnelle et un portrait analysé à jour. Sept histoires sont disponibles par semaine civile, du lundi à minuit au lundi suivant en Europe/Paris, sans report. La demande réserve un crédit ; seule la publication le consomme. Les personnages secondaires de l’ancienne version restent en archives, avec leurs histoires et retours ; ils ne génèrent plus de nouveaux épisodes.

Le mode de test `FREE_ACCESS_ENABLED=true` permet actuellement d’utiliser la plateforme sans paiement, en conservant les quotas et les conditions de démarrage du personnage. Le tarif configuré pour une future activation est de 29 EUR/mois après sept jours gratuits. Le paiement dépend de la configuration Stripe ; une interface publiée ne signifie pas qu’un encaissement est activé.

**Ambre** propose une conversation IA, avec 100 messages utilisateur ayant reçu une réponse par semaine. Un fil vide affiche un accueil personnalisé sans appel au modèle ni consommation de message. L’envoi apparaît immédiatement, avec un indicateur d’attente puis une révélation progressive de la réponse validée. La dictée française du navigateur affiche un texte modifiable, sans envoi automatique ni conservation d’audio par l’application ; sa disponibilité et son service vocal dépendent du navigateur.

Les propositions de modification du portrait, des proches et de la ligne de vie sont sourcées, modifiables et appliquées uniquement après acceptation. Les capacités sont calculées côté serveur puis revérifiées à la confirmation. Les histoires demandées dans le chat utilisent le même pool de sept, avec un angle approuvé. L’historique reste distinct du profil ; son effacement depuis l’interface ne réinitialise pas les quotas.

La cartographie présente d’abord les pistes personnelles, puis leurs éléments d’appui, contre-indices, autres explications et questions. L’exploration des 161 repères reste accessible séparément. Ces repères ne constituent ni des diagnostics ni des probabilités cliniques validées.

## Architecture

Hugo/Netlify sert le catalogue public. Cloudflare Worker/D1 gère comptes, sources personnelles, personnages, crédits et histoires privées. GitHub Actions traite la file toutes les quinze minutes ; il ne crée aucun récit sans demande. Les retards du planificateur sont possibles.

La clé `OPENROUTER_API_KEY` reste un secret Cloudflare. Les Actions s'authentifient par OIDC auprès d'une passerelle qui vérifie versions, bail, droits et budgets avant chaque appel. Les textes privés ne sont écrits ni dans Git, ni dans les articles publics, ni dans les logs Actions.

## Validation

```sh
node scripts/build_navigation.mjs --check
python scripts/check_editorial_pack.py
python scripts/integrate_editorial_pack.py --check
hugo --minify
node --test tests/*.test.cjs
npm --prefix journal test
python -m unittest discover -s journal/tests -p 'test_personal*.py'
node journal/tests/evaluate-recommendations.mjs
```

Voir [le plan et les décisions V4](docs/plan-accompagnement-v4.md), [le guide backend](journal/README.md), [les contrats V2](docs/contrats-implementation-v2.md), [les résultats algorithmiques et leurs limites](docs/implementation-algorithmique-v2.md) et [la recette de mise en ligne](docs/suivi-implementation-v2.md).

V10 : [plan, revue et validation](docs/plan-ambre-v10.md), [audit de couverture](docs/couverture-v10-proposition.md), [contrat de capacités](docs/plan-capacites-v10.md), [interface](docs/plan-interface-v10.md), [manifeste des 156 récits](data/editorial_pack_v10.json).
