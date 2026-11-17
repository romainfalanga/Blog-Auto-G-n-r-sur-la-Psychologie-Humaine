# Relecture indépendante — accès, crédits et données

État relu le 17 novembre 2026, branche `codex/compte-timelines-ambre`. Relecture limitée aux modifications d’accès/readiness, crédits mensuels, migrations 0025/0026, timelines, notes et contexte. Aucun fichier de production ou test modifié par cette relecture. Aucun appel fournisseur ni intervention distante.

## P1 — appels fournisseur encore possibles après disparition des prérequis

Deux chemins reproduits avec SQLite réel et `fetch` simulé. Ces constats concernent l’état avant correction du propriétaire du lot ; ils lui ont été transmis immédiatement.

### Bilan : `journal/src/chat-backend.mjs:815`

`reviewCall` vérifie les prérequis au début, puis attend plusieurs opérations. La réservation `provider_budgets` ligne 821 ne vérifie que le bail du bilan. Le `fetch` ligne 824 ne relit pas les prérequis.

Scénario : un compte satisfait la checklist et démarre son bilan. Un second onglet supprime son troisième proche après la vérification initiale, avant la réservation technique. Le bilan appelle encore le fournisseur avec les données déjà préparées, bien que la checklist soit désormais incomplète. La garde de publication ultérieure ne supprime pas cet appel déjà parti.

Reproduction : harness `setup` de `journal/tests/v4-S-helper.mjs`, `seedCompanionPrerequisites(raw)`, wrapper de `env.DB.prepare` sur `INSERT INTO provider_budgets` contenant `ambre_reviews`. Juste avant son `run`, exécuter `DELETE FROM personal_people WHERE id='ready-person-a-1'`. Appeler `/api/chat/start` avec `{accepted:true,idempotency_key:'review-race-proof'}`. Le mock de `fetch` relit `readCompanionReadiness` et renvoie un JSON inoffensif. Résultat observé : `calls=1`, `complete=false`, `people.count=2` au moment de l’appel, réponse route 202.

Correction recommandée : inclure readiness, accès, consentement et réservation commerciale dans la garde atomique du budget, puis revalider avant l’envoi. Libérer les budgets techniques réservés si cette dernière vérification empêche un appel non encore émis.

### Portrait : `journal/src/personal-backend.mjs:255`

`proxyJobLLM` applique correctement la garde de readiness au batch de réservation, mais attend ensuite `reserveDaily` et lance `fetch` sans dernière vérification du job/contexte. Une modification intercalée entre ces deux étapes traverse donc la garde nouvellement ajoutée.

Reproduction : `setup({env:{DAILY_PROVIDER_TOKEN_CAP:'1000000'}})`, checklist complète, ajouter une description à `ready-event-a-1`, obtenir un job via `personal.claimJobs`. Wrapper du `run` de `INSERT INTO provider_global_budget` supprimant `ready-person-a-1`. Appeler `personal.proxyJobLLM` avec les versions et le bail du job, deux prompts factices et `max_tokens:128`. Le mock de `fetch` constate `complete=false`, `people.count=2`. Résultat observé : un appel fournisseur simulé, réponse proxy 200.

Correction recommandée : revalider le job et sa garde immédiatement avant le fetch, après la réservation quotidienne, avec restitution des réservations non utilisées lorsque l’envoi est empêché.

## Vérifications effectivement effectuées

Commande `node --test journal/tests/monthly-access.test.mjs journal/tests/timeline-data.test.mjs` : 16 tests réussis, zéro échec sur l’état relu.

Les cas exécutés couvrent notamment les deux réservations concurrentes au dernier crédit, le crédit commun initial/point, le remboursement d’un échec, la conservation de l’usage après suppression du contenu, les frontières mensuelles Paris, le backfill historique, les prérequis incomplets d’un compte déjà commencé, la non-publication après suppression d’un proche, la propriété des notes, les révisions CAS et replays, les événements partagés, les cascades, l’export, l’invalidation du contexte et l’absence de migration 0025.

Lecture supplémentaire des triggers 0025/0026 et des gardes de publication : le registre mensuel conserve les crédits consommés et reste rattaché au compte ; les notes incrémentent révision parent et version profil ; les notes acceptées et les observations Ambre gardent des provenances distinctes. Aucun autre défaut concret n’a été établi dans ces éléments pendant cette relecture.

La compatibilité avant 0026 n’est pas assurée par le nouveau Worker : `/api/me` lit directement `monthly_usage`. Il faut donc appliquer 0026 avant d’activer ce Worker. Ce point est une contrainte de déploiement, pas une preuve de panne sur la séquence migrations-puis-Worker documentée.

Cette relecture ne constitue pas une validation globale du produit, de l’interface, du déploiement distant ou de tous les chemins concurrentiels possibles.

## Contre-vérification après correction

Le propriétaire du lot a corrigé `reviewCall` pendant la relecture. Le même interleaving (suppression du troisième proche juste avant l’INSERT du budget fournisseur) a été rejoué indépendamment : zéro appel `fetch`, bilan `failed` / `prerequisites_required`, crédit bilan `released`, aucune ligne de budget fournisseur. Le défaut du chemin bilan est donc fermé pour cette reproduction.

La correction du portrait a également été rejouée indépendamment avec la suppression pendant la réservation quotidienne : zéro appel `fetch`, réponse 409 `context_changed`, `provider_calls=0` et `reserved_tokens=0` dans le job, et réservations quotidienne/hebdomadaire revenues à zéro. Le défaut du chemin portrait est fermé pour cette reproduction.

Dernière exécution des deux suites ciblées après ces corrections et ajout du test d’admission concurrente message/bilan : 19 tests réussis, zéro échec. Le nouveau test portrait en cours d’ajout par son auteur n’était pas encore inclus dans cette exécution ; la contre-vérification portrait ci-dessus a été exécutée séparément.

## Delta final — ressources statiques sur refus de prérequis

Relecture ciblée du retour 403 `prerequisites_required` dans `chat-backend.mjs:580` : le changement ajoute seulement `safety` / `safety_reply` et les ressources dans le texte d’erreur lorsque le signal lexical préexistant est présent. Ce retour reste avant toute insertion de tour, réservation et prise en charge moteur. Le détecteur lexical et les autres routes ne sont pas modifiés par ce delta.

Sondes indépendantes sur SQLite avec fournisseur interdit : compte incomplet + propos suicidaire ou menace de violence → 403 `prerequisites_required` et ressources appropriées ; message ordinaire → 403 sans `safety` ; mêmes mots dans le contenu de `/api/chat/start` et `/api/chat/review` → aucun champ `safety_reply` ; compte complet au plafond de 100 messages + message de crise → 429 `quota_exhausted` avec ressources. Sur ces scénarios : zéro appel fournisseur, zéro tour, zéro job moteur et zéro budget fournisseur créé.

Test dédié `incomplete accounts retain static crisis resources without any generation or quota consumption` exécuté : réussi. Trois tests sélectionnés de `v4-S-safety.test.mjs` et `chat-safety.test.mjs` (F17 et ressources) également réussis. Aucun contournement de quota/génération ni déclenchement supplémentaire dans les autres routes n’a été établi pour ce delta.
