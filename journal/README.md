# Espace personnel et Ambre

Pages de l’interface : `/mon-accompagnement/`, `/mon-portrait/`, `/mes-proches/`, `/ma-ligne-de-vie/`, `/mon-compte/`. Les données sont privées et authentifiées. Les anciens liens conservent les paramètres liés aux articles publics, à la navigation personnelle ou au paiement lorsque la destination les utilise.

## Base et migration

Appliquer les migrations `0001` à `0014` dans l'ordre. Sur une base existante, créer d'abord un point de restauration D1 et suspendre les générations. La migration `0005_personal_context_characters.sql` conserve IDs, corps, retours et ordre des histoires ; elle les rattache au premier personnage et inscrit leurs crédits dans leur semaine d'origine. Les portraits V1 sont invalidés pour être reconstruits à partir des sources originales. `0006_recommendation_exposures.sql` stocke uniquement des IDs d'articles publics récemment proposés.

Ne pas rejouer aveuglément une migration appliquée. Le déploiement de `worker.mjs` exige tous les modules importés de `src/`, notamment `personal-backend.mjs`, `private-context.mjs`, `recommendations.mjs`, sa baseline et le catalogue. `GENERATION_PAUSED=true` suspend la prise en charge et la publication des analyses ; lecture et édition personnelle restent disponibles.

Les proches, événements, associations, révisions, manifestes et demandes sont décrits dans [le contrat partagé](../docs/contrats-implementation-v2.md). Les créations utilisent une clé idempotente ; les modifications et suppressions vérifient la révision. Supprimer un proche conserve les événements liés. Export et suppression du compte couvrent les nouvelles tables.

## Portrait et recommandations

Un like, une relation décrite, un événement ou une mémoire personnelle peuvent démarrer le portrait. Un nom et rôle seuls ne servent pas de preuve. Les dérivés réutilisables sont attachés à une source, sa révision, son hash et la version de l'analyse. Toute assertion publiée exige une citation exacte d'une source autorisée. Un like sans commentaire reste un intérêt de faible confiance. Les très grands contextes sont traités par tranches avec couverture explicite, jamais silencieusement tronqués.

En mode `FREE_ACCESS_ENABLED=true`, la plateforme est accessible sans paiement et ces anciens plafonds gratuits ne bloquent plus les révisions de portrait. Le quota de conversation et les budgets techniques restent actifs. Les sources modifiées invalident l'analyse en cours ; les recommandations provisoires restent disponibles. Les préférences de style ou d'évitement ne deviennent pas des faits autobiographiques.

Le classement V2 est déterministe, sans appel IA par requête, et ne remplit pas la liste avec des sujets sans rapport. Les expositions sont bornées à cent IDs sur sept jours et ne modifient pas le portrait. Les résultats synthétiques ne démontrent pas une amélioration générale chez les lecteurs ayant déjà des likes : consulter [l'évaluation](../docs/implementation-algorithmique-v2.md).

## Accès et analyse du portrait

Le tarif configuré est de 29 EUR/mois après sept jours d’essai. La conversation permet 100 réponses réussies par semaine. Ambre sélectionne des lectures dans le catalogue public et peut proposer des mises à jour des données personnelles à vérifier avant enregistrement.

Le Worker analyse le portrait. Pour les contextes longs, le workflow `.github/workflows/generate-articles.yml` exécute `scripts/run_journals.py` toutes les quinze minutes, manuellement ou après modification du pipeline. OIDC vérifie le dépôt, la branche, le workflow et l’événement. La clé OpenRouter reste dans Cloudflare.

Les anciennes adresses du personnage et de ses histoires redirigent vers Ambre. La migration `0014_retire_personal_stories.sql` ferme les demandes de récit ouvertes et préserve les données historiques dans l’export du compte.

## Vérification

```sh
npm test
node --test ../tests/*.test.cjs
python -m unittest discover -s tests -p 'test_personal*.py'
node tests/evaluate-recommendations.mjs
```

La recette et les versions réellement publiées figurent dans [le suivi V2](../docs/suivi-implementation-v2.md). Les preuves historiques V1 sont archivées dans les documents précédents ; elles ne remplacent pas une vérification du pipeline V2.

## Accompagnement V4

`0007` ajoute le portrait déclaré et le personnage unique ; `0008` les métadonnées d’essai Stripe ; `0009` conversations, consentement et propositions ; `0010` l’angle approuvé d’une histoire. Les migrations sont additives.

`OPENROUTER_CHAT_MODEL` configure séparément le modèle de conversation (défaut `anthropic/claude-sonnet-5.5`). La clé reste `OPENROUTER_API_KEY`. Les messages utilisent un retour 202 et `waitUntil`, puis un polling authentifié ; le délai fournisseur est borné à 25 secondes. Les réponses échouées ne consomment pas un message, mais restent dans les compteurs de tentatives et de budget fournisseur.

Routes : `GET/DELETE /api/chat`, `PUT /api/chat/consent`, `POST /api/chat/messages`, `GET /api/chat/requests/:id`, `POST /api/chat/proposals/:id/accept|reject`. Les corrections des pages personnelles utilisent leurs versions et droits existants. `POST /api/billing/cancel` programme la fin de l’abonnement ou de l’essai sans dépendre du fournisseur IA.

Configurer Stripe avec un prix mensuel EUR 2900 centimes, un webhook signé et les trois bindings serveur requis. Le succès des tests simulés ne remplace pas une recette Stripe en mode test puis le contrôle de production. Ne pas déclencher un paiement pour une simple vérification technique.


## Cartographie et révisions V6

Catalogue partagé : `data/psychology_catalog.json`, génération par `node scripts/build_psychology_catalog.mjs` (contrôle `--check` en CI). Les 53 repères restent des processus/ressources éditoriaux, pas une classification diagnostique. La carte conserve observations, fonctions possibles, alternatives, citations personnelles et contre-indices. Les copies de la même preuve ne font pas converger une hypothèse.

Les enregistrements relancent l’analyse native avec debounce court et contrôle sémantique séparé. Un cron Worker chaque minute récupère les tâches interrompues. Pour les contextes dépassant 130 000 caractères, la tâche est laissée au pipeline Python par lots (workflow toutes les 15 minutes) ; la couverture reste explicite. Aucun secret GitHub supplémentaire n’est nécessaire.

Le statut Stripe n’est pas modifié par le mode de test. Pour rétablir le paiement requis, modifier explicitement `FREE_ACCESS_ENABLED`; ne pas fabriquer d’abonnement dans la base. La clé OpenRouter reste un secret du Worker. Les analyses coûtent des crédits au propriétaire même si l’interface est ouverte gratuitement.

`0011_native_portrait_candidate.sql` ajoute un candidat non publié, lié au job et invalidé à chaque changement de source. La reprise vérifie ce candidat sans refaire la synthèse ; publication uniquement après contrôle sémantique. `OPENROUTER_PORTRAIT_MODEL` vaut actuellement `anthropic/claude-sonnet-5.5` ; le pipeline Python conserve le modèle analytique général.
