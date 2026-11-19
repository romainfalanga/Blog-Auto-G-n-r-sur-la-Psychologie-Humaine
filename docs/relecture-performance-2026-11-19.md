# Relecture indépendante du lot P — 19 novembre 2026

Périmètre : code écrit avant la reprise Codex, référence `3ad3232`, comparée à la fusion PR 103 `329891e`. Relecture distincte de son auteur. Aucun changement de production requis pour les deux points encore marqués « à vérifier » : ils sont déjà implémentés.

- `ui.mjs` extrait script et style publics identiques pour tous les comptes dans `APP_ASSETS`, avec chemins dépendant du contenu. La page ne garde que le squelette et son amorçage privé.
- Le Worker sert uniquement les deux noms exacts, GET/HEAD, avec `public, max-age=31536000, immutable`. Un ancien nom ou une erreur garde `private, no-store`. Les réponses privées ne deviennent jamais publiques. Le proxy Netlify ne force aucun cache sur `/app/*`.
- `examples.mjs` est importé à la demande, avec promesse partagée, remise à zéro après échec et index de recommandation créé au premier usage. La portée est précise : les pages anonymes, les assets et les coquilles sans données privées évitent cet import ; `/api/me` connecté et les API privées le chargent toujours. Aucun gain absolu de démarrage Cloudflare n'est déduit de ce seul changement.
- `SpeechRecognition.available()` n'est appelé qu'au geste vers le micro ou à son premier démarrage, une seule fois. Exceptions, rejet asynchrone et destruction sont gérés. La dictée réelle dépend toujours des permissions et du navigateur de la personne.
- L'index Personnages est chargé au premier focus ou à la première saisie, avec reprise d'une recherche restaurée.
- Paramètres charge son résumé `/api/chat/insight` à l'ouverture de la page. Seule la trace détaillée `?turn_id=` attend l'onglet Réponses précédentes ; ne pas présenter ce dernier changement comme un report du résumé complet.
- Le placement smart est présent dans Wrangler ; il doit aussi être conservé dans les métadonnées du déploiement multipart.

## Mesure reproductible

`node scripts/perf/compare-worker.mjs 329891e 3ad3232` compare les mêmes modules aux deux révisions, sept processus isolés chacun. Mesures : octets HTML et assets, gzip standard, médiane d'import local. Aucun réseau, aucune donnée personnelle, aucun appel IA. Une visite répétée suppose les assets immuables déjà en cache. Les mesures d'import dépendent de la charge de la machine ; elles ne sont ni un LCP navigateur ni une mesure de latence D1/Cloudflare.

Les mesures navigateur de production après déploiement restent à rattacher à la recette globale ; ce rapport ne prétend pas les remplacer.

## Résultats locaux de la reprise

Les 22 tests cache, chargement et dictée passent après intégration du module symptômes. La comparaison PR 103 / PR 104 mesure 119 464 octets gzip de HTML auparavant, contre 1 734 octets de HTML et 118 464 octets de fichiers statiques ensuite. La première visite transfère donc une quantité comparable (120 198 octets), tandis qu’une visite avec les fichiers déjà en cache réduit fortement le HTML transféré. Les médianes d’import sous forte contention (2 226 / 2 295 ms, dispersion importante) ne montrent pas de gain exploitable de latence ; aucune conclusion de vitesse navigateur n’en est tirée. Preuves : `work/reprise-nov19/performance-comparison.json`, `performance-tests-final.log`.
