# Journal personnel — V2

Pages publiques de l'interface : `/mes-personnages/`, `/mon-portrait/`, `/mon-compte/`. Les données sont privées et authentifiées. Les anciennes routes `/mes-histoires/`, `/mes-articles/` et `/journal/` conservent leurs paramètres lors des redirections.

## Base et migration

Appliquer les migrations `0001` à `0006` dans l'ordre. Sur une base existante, créer d'abord un point de restauration D1 et suspendre les générations. La migration `0005_personal_context_characters.sql` conserve IDs, corps, retours et ordre des histoires ; elle les rattache au premier personnage et inscrit leurs crédits dans leur semaine d'origine. Les portraits V1 sont invalidés pour être reconstruits à partir des sources originales. `0006_recommendation_exposures.sql` stocke uniquement des IDs d'articles publics récemment proposés.

Ne pas rejouer aveuglément une migration appliquée. Le déploiement de `worker.mjs` exige tous les modules importés de `src/`, notamment `personal-backend.mjs`, `private-context.mjs`, `recommendations.mjs`, sa baseline et le catalogue. `GENERATION_PAUSED=true` suspend claims, appels IA, préparations, continuations et publications ; lecture et édition personnelle restent disponibles.

Les proches, événements, associations, révisions, manifestes et demandes sont décrits dans [le contrat partagé](../docs/contrats-implementation-v2.md). Les créations utilisent une clé idempotente ; les modifications et suppressions vérifient la révision. Supprimer un proche conserve les événements liés. Export et suppression du compte couvrent les nouvelles tables.

## Portrait et recommandations

Un like, une relation décrite, un événement ou une mémoire personnelle peuvent démarrer le portrait. Un nom et rôle seuls ne servent pas de preuve. Les dérivés réutilisables sont attachés à une source, sa révision, son hash et la version de l'analyse. Toute assertion publiée exige une citation exacte d'une source autorisée. Un like sans commentaire reste un intérêt de faible confiance. Les très grands contextes sont traités par tranches avec couverture explicite, jamais silencieusement tronqués.

Le portrait est gratuit, borné à trois démarrages par 24 h et dix minutes entre démarrages gratuits. Les sources modifiées invalident l'analyse en cours ; les recommandations provisoires restent disponibles. Les préférences de style ou d'évitement ne deviennent pas des faits autobiographiques.

Le classement V2 est déterministe, sans appel IA par requête, et ne remplit pas la liste avec des sujets sans rapport. Les expositions sont bornées à cent IDs sur sept jours et ne modifient pas le portrait. Les résultats synthétiques ne démontrent pas une amélioration générale chez les lecteurs ayant déjà des likes : consulter [l'évaluation](../docs/implementation-algorithmique-v2.md).

## Trois personnages et crédits

L'identité et le canon fictif sont stables ; les facettes personnelles sont neutralisées si leurs preuves ne sont plus actuelles. Les événements « pour me comprendre seulement » sont exclus des consignes fictives. Si aucune source n'est utilisable pour la fiction, les fiches restent explicitement exploratoires sans fausses preuves.

La préparation des trois fiches ne consomme pas de crédit. Chaque demande continue le dernier épisode du personnage choisi, avec une seule demande active par compte. Sept crédits par semaine civile Europe/Paris, sans report, sont réservés atomiquement. Rejouer la même demande, reprendre après correction ou republier une completion déjà reçue ne débite pas un second crédit. Un échec définitif, une annulation ou une expiration libère la réservation.

Le seuil de nouvelle souscription reste dix likes et le tarif prévu 10 EUR/mois. Un abonné existant peut générer avec zéro like s'il fournit une source personnelle exploitable. Les bindings Stripe sont requis pour encaisser ; aucun achat n'est nécessaire à la recette technique.

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
