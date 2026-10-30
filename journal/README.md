# Décode ton esprit : journal personnel

Application Cloudflare distincte du blog Hugo existant. Le blog et sa publication de trois articles sont conservés. Le journal propose un personnage inspiré de chaque utilisateur, une fiction quotidienne privée et un retour structuré pour les récits suivants.

## Architecture

- Worker ES modules sans dépendance serveur, D1 dédiée en juridiction européenne.
- Comptes et sessions privées, profil modifiable, consentement au traitement IA, journal et historique, retours, mémoire déclarée modifiable, export et suppression.
- Stripe Checkout abonnement mensuel 10 EUR, portail client, webhook signé. Le serveur vérifie le tarif et l'état réel de l'abonnement.
- GitHub Actions exécute `scripts/generate_journals.py` toutes les heures. Une seule histoire par utilisateur et journée Europe/Paris, pour les comptes éligibles uniquement. Les horaires Actions ne garantissent pas une livraison à la minute.
- Le workflow s'authentifie auprès du Worker avec GitHub OIDC, borné au dépôt, à main et au fichier exact. Aucune clé d'accès permanente supplémentaire au journal dans GitHub.
- La clé IA reste dans GitHub Actions. OpenRouter est prioritaire avec `OPENROUTER_API_KEY`; en son absence le secret historique `MAMMOUTH_API_KEY` est utilisé. Le code ancien utilise réellement Mammouth.

## Adaptation de l'automatisation existante

Le script `generate_articles.py` reste le moteur du blog. Son approche est adaptée au journal : personnage d'abord, scène et angle, rédaction au présent, puis relecture indépendante de cohérence. Le contexte privé remplace les vingt personnages publics et la matrice SEO. Les retours et corrections sont injectés dans les prochains récits. Les scènes fictives ne deviennent jamais des faits biographiques ni une preuve de progrès réel. La mémoire explicite est modifiée uniquement par l'utilisateur. Une synthèse provisoire automatique conserve les déclarations et retours utiles au-delà des vingt retours récents, avec leur provenance ; elle reste visible, rectifiable et effaçable. Il ne s'agit pas d'un entraînement de modèle ni d'un diagnostic.

## Mise en service

URL prévue : https://decode-journal.romainfalanga83.workers.dev

1. Appliquer `migrations/0001.sql` à la D1 déclarée dans `wrangler.jsonc`.
2. Déployer `src/worker.mjs` avec son module `ui.mjs`.
3. Ajouter les secrets Worker `STRIPE_SECRET_KEY` et `STRIPE_WEBHOOK_SECRET`, puis la variable `STRIPE_PRICE_ID` correspondant à un prix Stripe actif à 1000 centimes EUR par mois. Utiliser d'abord un compte/test Stripe et valider tout le parcours avant les paiements réels.
4. Configurer le webhook vers `/api/billing/webhook` avec les événements indiqués dans le serveur. Activer le portail client Stripe pour la gestion et la résiliation.
5. Vérifier la présence de `OPENROUTER_API_KEY` dans les secrets Actions. Le nom des secrets référencés dans un workflow ne prouve pas leur présence. Une clé GitHub chiffrée n'est pas récupérable en lecture.
6. Le modèle par défaut est `openai/gpt-5-mini`. La variable Actions `JOURNAL_OPENROUTER_MODEL` permet de choisir un modèle compatible JSON. Vérifier disponibilité et coût du modèle sur le compte fournisseur.
7. Déclencher `Journal personnel quotidien` sur main après un abonnement de test, lire puis commenter le récit et contrôler que le prochain tient compte du retour.

Sans configuration Stripe, l'inscription et le profil fonctionnent mais aucun abonnement ni génération payante n'est prétendu actif. Le client indique l'indisponibilité du paiement. Aucun accès payant n'est activé à partir d'une simple URL de retour Checkout.

## Confidentialité et limites

Les profils et retours sont transmis au fournisseur IA choisi uniquement après consentement. La base est européenne; cela ne constitue pas une garantie de localisation européenne des traitements IA ou des runners GitHub. Les données privées ne sont jamais committées, ajoutées en artefact ou imprimées dans les logs du workflow. Les métadonnées de facturation restent chez Stripe et ne sont pas incluses dans les prompts.

La limite initiale est de dix travaux par exécution horaire, réservés un par un avec un bail de trente minutes ; trois tentatives au maximum par journée. L’exécution cesse de réserver de nouveaux travaux après trente minutes. C'est une limite d'exploitation à surveiller avant croissance du service, pas une promesse de capacité illimitée. Les frais IA, Stripe et Cloudflare ne sont pas inclus dans les 10 EUR comme coûts garantis : le prix demandé est le tarif client.

## Vérification

```sh
cd journal
npm run check
npm test
python -m unittest discover -s tests -p 'test_*.py'
```

Les tests locaux ne remplacent pas la recette Stripe de bout en bout et une exécution IA réelle avec les secrets du compte.

## État de livraison

Worker et D1 publiés le 30 octobre 2026. 7 tests backend et 6 tests de génération passent. Stripe n’est pas encore configuré ; aucun paiement réel ni génération réelle par fournisseur n’a été validé. Le modèle OpenRouter par défaut apparaît dans le catalogue public et accepte `response_format`. La récupération de mot de passe par e-mail n’est pas implémentée dans cette première version.
