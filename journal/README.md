# Journal personnel — V6

Pages de l’interface : `/mon-accompagnement/`, `/mon-personnage/`, `/mon-portrait/`, `/mes-proches/`, `/ma-ligne-de-vie/`, `/mon-compte/`. Les données sont privées et authentifiées. Les anciennes routes `/mes-histoires/`, `/mes-articles/` et `/journal/` conservent leurs paramètres lors des redirections.

## Base et migration

Appliquer les migrations `0001` à `0010` dans l'ordre. Sur une base existante, créer d'abord un point de restauration D1 et suspendre les générations. La migration `0005_personal_context_characters.sql` conserve IDs, corps, retours et ordre des histoires ; elle les rattache au premier personnage et inscrit leurs crédits dans leur semaine d'origine. Les portraits V1 sont invalidés pour être reconstruits à partir des sources originales. `0006_recommendation_exposures.sql` stocke uniquement des IDs d'articles publics récemment proposés.

Ne pas rejouer aveuglément une migration appliquée. Le déploiement de `worker.mjs` exige tous les modules importés de `src/`, notamment `personal-backend.mjs`, `private-context.mjs`, `recommendations.mjs`, sa baseline et le catalogue. `GENERATION_PAUSED=true` suspend claims, appels IA, préparations, continuations et publications ; lecture et édition personnelle restent disponibles.

Les proches, événements, associations, révisions, manifestes et demandes sont décrits dans [le contrat partagé](../docs/contrats-implementation-v2.md). Les créations utilisent une clé idempotente ; les modifications et suppressions vérifient la révision. Supprimer un proche conserve les événements liés. Export et suppression du compte couvrent les nouvelles tables.

## Portrait et recommandations

Un like, une relation décrite, un événement ou une mémoire personnelle peuvent démarrer le portrait. Un nom et rôle seuls ne servent pas de preuve. Les dérivés réutilisables sont attachés à une source, sa révision, son hash et la version de l'analyse. Toute assertion publiée exige une citation exacte d'une source autorisée. Un like sans commentaire reste un intérêt de faible confiance. Les très grands contextes sont traités par tranches avec couverture explicite, jamais silencieusement tronqués.

En mode `FREE_ACCESS_ENABLED=true`, la plateforme est accessible sans paiement et ces anciens plafonds gratuits ne bloquent plus les révisions de portrait. Les quotas de chat, d’histoires et les budgets techniques restent actifs. Les sources modifiées invalident l'analyse en cours ; les recommandations provisoires restent disponibles. Les préférences de style ou d'évitement ne deviennent pas des faits autobiographiques.

Le classement V2 est déterministe, sans appel IA par requête, et ne remplit pas la liste avec des sujets sans rapport. Les expositions sont bornées à cent IDs sur sept jours et ne modifient pas le portrait. Les résultats synthétiques ne démontrent pas une amélioration générale chez les lecteurs ayant déjà des likes : consulter [l'évaluation](../docs/implementation-algorithmique-v2.md).

## Un personnage et crédits

L'identité et le canon fictif sont stables ; les facettes personnelles sont neutralisées si leurs preuves ne sont plus actuelles. Les événements « pour me comprendre seulement » sont exclus des consignes fictives. Si aucune source n'est utilisable pour la fiction, les fiches restent explicitement exploratoires sans fausses preuves.

La préparation de la fiche ne consomme pas de crédit. Chaque demande continue le dernier épisode du personnage actif, avec une seule demande active par compte. Sept crédits par semaine civile Europe/Paris, sans report, sont réservés atomiquement. Rejouer la même demande, reprendre après correction ou republier une completion déjà reçue ne débite pas un second crédit. Un échec définitif, une annulation ou une expiration libère la réservation.

Le tarif est de 29 EUR/mois après sept jours d’essai, carte requise. Le chat (100 réponses réussies par semaine) est accessible pendant l’essai sans dix likes ; les histoires exigent dix likes, une présentation et un portrait analysé à jour. Les bindings Stripe sont requis pour encaisser ; aucun achat n'est nécessaire à la recette technique.

## Génération

Le workflow `.github/workflows/generate-articles.yml` exécute `scripts/run_journals.py` toutes les quinze minutes, manuellement ou après modification du pipeline. La préparation et les récits sont demandés explicitement. OIDC vérifie signature, dépôt, branche, workflow et événement. La clé OpenRouter reste dans Cloudflare.

La chaîne V2 extrait les sources, synthétise le portrait, prépare les personnages puis, pour chaque demande, cadre, rédige, contrôle, corrige si nécessaire et résume le texte final. Les anciennes résolutions sont recherchées dans tout l'historique du personnage. Les preuves du delta narratif citent le corps final. Les limites sont cumulées sur la demande : vingt appels maximum, trois tentatives, bail de trente minutes ; le runner cesse de réclamer de nouvelles tâches après vingt minutes et de lancer des appels après vingt-cinq minutes. Les erreurs publiques ne contiennent pas le contexte privé.

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

Le statut Stripe n’est pas modifié par le mode de test. Pour rétablir le paiement requis, modifier explicitement `FREE_ACCESS_ENABLED`; ne pas fabriquer d’abonnement dans la base. La clé OpenRouter reste un secret du Worker. Les générations coûtent des crédits au propriétaire même si l’interface est ouverte gratuitement.
